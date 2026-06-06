//! Phase 6 fines-migration invariants (GPU-gated; skips without an adapter).
//!
//! U1 scope (this file's current tests): fines live in the `chem.w` lane, seeded uniformly onto
//! grains (`fines_fraction · grain_volume`) and zero on water, and with the feature gated off
//! (`fines_rate = 0`) nothing disturbs them. U2+ add the erosion/deposition conservation gates.
//!
//! Note on ordering: the water-loop reorder permutes slot order each rebuild, so a single post-step
//! `read_chem()`/`read_phases()` snapshot aligns slot-for-slot, but two snapshots do not. The
//! inertness gate therefore asserts a **set/total** invariant, not a per-index byte compare.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

/// A small mixed water+grain scene (water slab above a grain slab) so the coupling passes run.
fn mixed_scene() -> Scene {
    Scene {
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [2.0, 2.0, 2.0],
                max: [5.0, 3.0, 5.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [2.0, 4.0, 2.0],
                max: [5.0, 5.0, 5.0],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    }
}

/// Materials with a non-zero fines inventory so seeding is observable.
fn fines_materials() -> Materials {
    Materials {
        fines_fraction: 0.1,
        ..Materials::default()
    }
}

/// The seeded per-grain fines volume the solver computes: `fines_fraction · (π/6)·d³`.
fn expected_fines_seed(mats: &Materials) -> f32 {
    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    coffee_sim::models::fines::fines_seed(grain_volume, mats.fines_fraction)
}

/// Config with fines active and absorption OFF — a closed system (no water deactivation/drain), so
/// total fines (grain + water) is a clean conserved quantity.
fn fines_active_config() -> Config {
    Config {
        absorb_rate: 0.0,
        extract_rate: 0.0,
        fines_rate: 2.0,
        ..Config::default()
    }
}

/// Materials with a fines inventory and a low critical flux so the falling water reliably erodes.
fn fines_active_materials() -> Materials {
    Materials {
        fines_fraction: 0.1,
        fines_crit_flux: 0.1,
        ..Materials::default()
    }
}

fn total_fines(chem: &[[f32; 4]]) -> f32 {
    chem.iter().map(|c| c[3]).sum()
}

/// Total fines on grains vs water (single snapshot — chem[i]/phase[i] align within one readback).
fn fines_by_species(chem: &[[f32; 4]], phase: &[u32]) -> (f32, f32) {
    let mut grain = 0.0;
    let mut water = 0.0;
    for (c, &ph) in chem.iter().zip(phase) {
        if ph == 1 {
            grain += c[3];
        } else {
            water += c[3];
        }
    }
    (grain, water)
}

#[test]
fn fines_erosion_conserves_volume() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_active_materials();
    let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &fines_active_config(), &gpu);
    let n_grain = solver.read_phases().iter().filter(|&&p| p == 1).count();
    let t0 = n_grain as f32 * expected_fines_seed(&mats);
    assert!(t0 > 0.0);

    // Falling water drives high flux through the grains → erosion (grain → water).
    for _ in 0..25 {
        solver.step(DT, &EmissionInput::default());
    }
    let chem = solver.read_chem();
    let phase = solver.read_phases();
    let (_, water) = fines_by_species(&chem, &phase);
    assert!(
        (total_fines(&chem) - t0).abs() < 1e-3 * t0,
        "fines volume not conserved during erosion: {} vs {t0}",
        total_fines(&chem)
    );
    assert!(
        water > 1e-5,
        "no erosion: water gained no suspended fines under impact flux ({water})"
    );
}

#[test]
fn fines_deposit_from_stagnant_water() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_active_materials();
    // Isolated deposition: zero gravity + every mechanical pass off, so nothing moves and the local
    // flux is 0 everywhere → net_rate < 0 (deposition) for every water↔grain pair. A thin water
    // layer sits one unit above a grain layer (within h), pre-loaded with suspended fines.
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [2.0, 2.0, 2.0],
                max: [5.0, 2.0, 5.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [2.0, 3.0, 2.0],
                max: [5.0, 3.0, 5.0],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    };
    let cfg = Config {
        max_iters: 0,
        bed_max_iters: 0,
        drag_subiters: 0,
        buoyancy_scale: 0.0,
        xsph_viscosity_c: 0.0,
        grain_sleep_speed: 0.0,
        fines_rate: 2.0,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);

    // Pre-load suspended fines onto every water particle (seed order; pre-step, so aligned).
    let mut chem = solver.read_chem();
    let phase = solver.read_phases();
    for (c, &ph) in chem.iter_mut().zip(&phase) {
        if ph == 0 {
            c[3] = 0.05;
        }
    }
    solver.write_chem_for_test(&chem);

    let before = solver.read_chem();
    let (grain_before, water_before) = fines_by_species(&before, &phase);
    let total_before = total_fines(&before);
    assert!(water_before > 0.0, "test setup: water must carry fines");

    for _ in 0..30 {
        solver.step(DT, &EmissionInput::default());
    }
    let chem = solver.read_chem();
    let phase = solver.read_phases();
    let (grain_after, water_after) = fines_by_species(&chem, &phase);

    assert!(
        (total_fines(&chem) - total_before).abs() < 1e-3 * total_before,
        "fines volume not conserved during deposition: {} vs {total_before}",
        total_fines(&chem)
    );
    assert!(
        water_after < water_before,
        "no deposition: stagnant water kept its suspended fines ({water_after} vs {water_before})"
    );
    assert!(
        grain_after > grain_before,
        "deposited fines did not land on grains ({grain_after} vs {grain_before})"
    );
}

#[test]
fn fines_conserve_volume_in_saturated_tail() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_active_materials();
    // Lighter solve (conservation is independent of solve quality) to keep 2000 steps cheap.
    let cfg = Config {
        max_iters: 4,
        bed_max_iters: 4,
        ..fines_active_config()
    };
    let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &cfg, &gpu);
    let n_grain = solver.read_phases().iter().filter(|&&p| p == 1).count();
    let t0 = n_grain as f32 * expected_fines_seed(&mats);

    for _ in 0..1000 {
        solver.step(DT, &EmissionInput::default());
    }
    let mid = total_fines(&solver.read_chem());
    for _ in 0..1000 {
        solver.step(DT, &EmissionInput::default());
    }
    let late = total_fines(&solver.read_chem());

    assert!(
        (late - t0).abs() < 1e-3 * t0,
        "fines volume drifted over 2000 steps: {late} vs {t0}"
    );
    // The discriminating signal: the post-equilibrium tail slope must be ~0 (pins the one-signed
    // sub-ulp leak class — the wet_sat_cutoff lesson).
    assert!(
        (late - mid).abs() < 1e-5 * t0,
        "fines drift in the saturated tail (slope not flat): {mid} → {late}"
    );
}

#[test]
fn fines_do_not_perturb_momentum() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_active_materials();
    let momentum = |fines_rate: f32| -> [f32; 3] {
        let cfg = Config {
            fines_rate,
            ..fines_active_config()
        };
        let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &cfg, &gpu);
        for _ in 0..60 {
            solver.step(DT, &EmissionInput::default());
        }
        let vel = solver.read_velocities();
        let phase = solver.read_phases();
        vel.iter().zip(&phase).fold([0.0; 3], |mut acc, (v, &ph)| {
            let m = if ph == 1 {
                mats.grain_mass
            } else {
                mats.particle_mass
            };
            acc[0] += m * v[0];
            acc[1] += m * v[1];
            acc[2] += m * v[2];
            acc
        })
    };
    // Fines (U2) write only chem.w — no velocity writes — so the mechanical evolution is identical
    // up to the reorder's run-to-run nondeterminism. A momentum write-through would diverge O(1).
    let off = momentum(0.0);
    let on = momentum(2.0);
    let mag = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2])
        .sqrt()
        .max(1.0);
    let diff =
        ((on[0] - off[0]).powi(2) + (on[1] - off[1]).powi(2) + (on[2] - off[2]).powi(2)).sqrt();
    assert!(
        diff < 5e-2 * mag,
        "fines perturbed momentum: |Δp|={diff} vs |p|={mag} (fines must not write velocities)"
    );
}

#[test]
fn fines_seeded_on_grains_zero_on_water() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_materials();
    let solver = XpbdSolver::build(&mixed_scene(), &mats, &Config::default(), &gpu);
    // No step yet → chem is still in seed order, aligned with phases.
    let chem = solver.read_chem();
    let phases = solver.read_phases();
    let seed = expected_fines_seed(&mats);
    assert!(seed > 0.0, "test setup: fines seed must be non-zero");

    let mut n_grain = 0usize;
    for (i, (c, &ph)) in chem.iter().zip(&phases).enumerate() {
        if ph == 1 {
            n_grain += 1;
            assert!(
                (c[3] - seed).abs() < 1e-6,
                "grain {i}: chem.w = {} expected fines_seed {seed}",
                c[3]
            );
        } else {
            assert!(c[3].abs() < 1e-6, "water {i}: chem.w = {} expected 0", c[3]);
        }
    }
    assert!(n_grain > 0, "scene must contain grains");
}

#[test]
fn fines_inert_when_rate_zero() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_materials();
    // Default config: coupling/drag run, but fines_rate = 0 → no fines transfer pass.
    let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &Config::default(), &gpu);
    let seed = expected_fines_seed(&mats);

    // Total grain fines on the books before any step.
    let n_grain = solver.read_phases().iter().filter(|&&p| p == 1).count();
    let total_before = n_grain as f32 * seed;

    for _ in 0..30 {
        solver.step(DT, &EmissionInput::default());
    }

    // Single post-step snapshot: chem[i] and phase[i] align (set-invariant total).
    let chem = solver.read_chem();
    let phases = solver.read_phases();
    let mut total_grain = 0.0f32;
    let mut max_water = 0.0f32;
    for (c, &ph) in chem.iter().zip(&phases) {
        if ph == 1 {
            total_grain += c[3];
        } else {
            max_water = max_water.max(c[3].abs());
        }
    }
    assert!(
        (total_grain - total_before).abs() < 1e-4 * total_before.max(1.0),
        "grain fines total drifted with fines off: {total_grain} vs {total_before}"
    );
    assert!(
        max_water < 1e-6,
        "water gained suspended fines with fines off: max {max_water}"
    );
}

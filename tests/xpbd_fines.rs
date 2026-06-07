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

/// Mean y of the water particles (drawdown proxy: lower = drained farther).
fn water_mean_y(pos: &[[f32; 4]], phase: &[u32]) -> f32 {
    let (mut sum, mut n) = (0.0f32, 0.0f32);
    for (p, &ph) in pos.iter().zip(phase) {
        if ph == 0 {
            sum += p[1];
            n += 1.0;
        }
    }
    if n > 0.0 {
        sum / n
    } else {
        0.0
    }
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
fn fines_strain_onto_grains_during_flow() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    // Isolated straining (deep-bed filtration): grains start EMPTY (fines_fraction = 0, so no
    // release), water is pre-loaded with suspended fines and falls through the grain layer. As it
    // flows past the grains the suspended fines strain out onto them → water loses, grains gain.
    let mats = Materials {
        fines_fraction: 0.0,
        min_pore_fraction: 0.5, // water threads the porosity field so it flows past the grains
        ..Materials::default()
    };
    let scene = Scene {
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
    };
    let cfg = Config {
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
    assert!(grain_before < 1e-6, "test setup: grains start empty");

    for _ in 0..60 {
        solver.step(DT, &EmissionInput::default());
    }
    let chem = solver.read_chem();
    let phase = solver.read_phases();
    let (grain_after, water_after) = fines_by_species(&chem, &phase);

    assert!(
        (total_fines(&chem) - total_before).abs() < 1e-3 * total_before,
        "fines volume not conserved during straining: {} vs {total_before}",
        total_fines(&chem)
    );
    assert!(
        water_after < water_before,
        "no straining: flowing water kept all its suspended fines ({water_after} vs {water_before})"
    );
    assert!(
        grain_after > grain_before,
        "strained fines did not land on grains ({grain_after} vs {grain_before})"
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
fn fines_transfer_writes_no_velocity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_active_materials();
    // Mechanics fully off (incl. drag, so the harmonic-k combiner is never invoked) + zero gravity.
    // Water is given a constant downward velocity so it flows past the grains (flux > 0 ⇒ the fines
    // release/strain transfer is ACTIVE), but with no forces, predict+finalize preserve velocity
    // exactly. So every velocity must stay at its seeded value — proving the fines passes write only
    // chem.w, never vel.
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
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
    let phase = solver.read_phases();
    let vel0: Vec<[f32; 4]> = phase
        .iter()
        .map(|&ph| {
            if ph == 0 {
                [0.0, -1.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            }
        })
        .collect();
    solver.write_velocities_for_test(&vel0);
    for _ in 0..20 {
        solver.step(DT, &EmissionInput::default());
    }
    let vel = solver.read_velocities();
    let phase = solver.read_phases();
    let worst = vel
        .iter()
        .zip(&phase)
        .map(|(v, &ph)| {
            let expect_vy = if ph == 0 { -1.0 } else { 0.0 };
            (v[0]).abs().max((v[1] - expect_vy).abs()).max((v[2]).abs())
        })
        .fold(0.0f32, f32::max);
    // Tolerance well above the predict→finalize f32 round-trip drift (~1e-5 over 20 steps) but far
    // below any real velocity write (a stray impulse would be O(0.1)).
    assert!(
        worst < 1e-3,
        "fines transfer perturbed velocity (worst Δ {worst}); it must write only chem.w"
    );
}

#[test]
fn fines_clog_raises_drag_and_slows_drawdown() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    // crit_flux huge → the transfer is frozen (no erosion/deposition), so a written clog persists.
    // This isolates the fines→permeability response (U4) from the transfer (U2).
    let mats = Materials {
        fines_fraction: 0.1,
        fines_crit_flux: 1.0e6,
        ..Materials::default()
    };
    let cfg = Config {
        xsph_viscosity_c: 0.0,
        fines_rate: 2.0,
        ..Config::default()
    };
    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    let seed = coffee_sim::models::fines::fines_seed(grain_volume, mats.fines_fraction);
    let input = EmissionInput::default();

    let run = |clog: f32| {
        let mut solver = XpbdSolver::build(&Scene::pour_over(), &mats, &cfg, &gpu);
        if clog > 0.0 {
            let mut chem = solver.read_chem();
            let phase = solver.read_phases();
            for (c, &ph) in chem.iter_mut().zip(&phase) {
                if ph == 1 {
                    c[3] = seed + clog; // uniformly clog every grain above the baseline
                }
            }
            solver.write_chem_for_test(&chem);
        }
        for _ in 0..120 {
            solver.step(DT, &input);
        }
        let pos = solver.read_positions();
        let phase = solver.read_phases();
        water_mean_y(&pos, &phase)
    };

    let baseline = run(0.0);
    // A heavy clog (0.6·grain_volume of extra fines on every grain): the drawdown-slowdown effect
    // must clear GPU run-to-run scatter (~±0.02 on the gap). 0.3·gv left the gap right at the 0.1
    // threshold and flaked ~1-in-5; doubling the clog drives a robust, clearly-detectable slowdown.
    let clogged = run(0.6 * grain_volume);
    eprintln!("drawdown water mean y: baseline {baseline:.3}, clogged {clogged:.3}");
    assert!(
        clogged > baseline + 0.1,
        "fines clog did not slow drawdown: clogged {clogged:.3} vs baseline {baseline:.3}"
    );
}

#[test]
fn fines_extreme_clog_stays_stable() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials {
        fines_fraction: 0.1,
        fines_crit_flux: 1.0e6,
        ..Materials::default()
    };
    let cfg = Config {
        fines_rate: 2.0,
        ..Config::default()
    };
    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    let mut solver = XpbdSolver::build(&Scene::pour_over(), &mats, &cfg, &gpu);

    // Drive every grain's lodged fines far past the α_s clamp ceiling (the numerically stiffest
    // case — the φ_f / pore-fraction floors and the correction cap must hold).
    let mut chem = solver.read_chem();
    let phase = solver.read_phases();
    for (c, &ph) in chem.iter_mut().zip(&phase) {
        if ph == 1 {
            c[3] = 5.0 * grain_volume;
        }
    }
    solver.write_chem_for_test(&chem);

    let input = EmissionInput::default();
    for _ in 0..120 {
        solver.step(DT, &input);
    }
    let vel = solver.read_velocities();
    let vmax = vel
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        vmax.is_finite() && vmax <= cfg.max_speed + 1e-3,
        "extreme clog blew up: vmax {vmax}"
    );
    assert!(
        solver
            .read_positions()
            .iter()
            .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite()),
        "extreme clog produced non-finite positions"
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

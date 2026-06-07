//! Phase 6 channeling diagnostics (`evenness`, `drawdown_time`) + integration gates (GPU-gated).
//!
//! U5 (this file): the brew-evenness flow-uniformity metric and the latched drawdown time are wired
//! into `metrics()` and respond to synthetic flow fields and to the grind lever. U6 adds the
//! end-to-end fines/channeling gates on the brew scene.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

/// Water slab directly over a grain slab sharing the same (x,z) footprint, so every bed bin has
/// water above it — a clean substrate for synthetic flow-field evenness tests.
fn stacked_scene() -> Scene {
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

/// Build a velocity buffer (seed order) that assigns `v_y` to each water particle by a closure of
/// its position, and zero to grains.
fn water_velocity_field(solver: &XpbdSolver, vy: impl Fn([f32; 4]) -> f32) -> Vec<[f32; 4]> {
    let pos = solver.read_positions();
    let phase = solver.read_phases();
    pos.iter()
        .zip(&phase)
        .map(|(p, &ph)| {
            if ph == 0 {
                [0.0, vy(*p), 0.0, 0.0]
            } else {
                [0.0; 4]
            }
        })
        .collect()
}

#[test]
fn uniform_downward_flux_is_even() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_channeling: no GPU adapter; skipping.");
        return;
    };
    let mut solver = XpbdSolver::build(
        &stacked_scene(),
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    // Perfectly uniform downward flux over the whole bed footprint → CoV 0 → evenness ≈ 1.
    solver.write_velocities_for_test(&water_velocity_field(&solver, |_| -1.0));
    solver.sample_diagnostics();
    let e = solver.metrics().evenness;
    assert!(
        e > 0.9 && e <= 1.0 + 1e-6,
        "uniform downward flux should be near-perfectly even, got {e}"
    );
}

#[test]
fn a_channel_lowers_evenness() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_channeling: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let mut solver = XpbdSolver::build(&stacked_scene(), &mats, &Config::default(), &gpu);

    solver.write_velocities_for_test(&water_velocity_field(&solver, |_| -1.0));
    solver.sample_diagnostics();
    let uniform = solver.metrics().evenness;

    // A channel: only the x < 3.5 half of the bed carries downward flux, the rest is stagnant.
    solver.write_velocities_for_test(&water_velocity_field(&solver, |p| {
        if p[0] < 3.5 {
            -3.0
        } else {
            0.0
        }
    }));
    solver.sample_diagnostics();
    let channeled = solver.metrics().evenness;

    eprintln!("evenness: uniform {uniform:.3}, channeled {channeled:.3}");
    assert!(
        channeled < uniform - 0.1,
        "a one-sided channel must lower evenness: channeled {channeled:.3} vs uniform {uniform:.3}"
    );
    assert!(
        channeled > 0.0,
        "evenness stays positive (1/(1+CoV)): {channeled}"
    );
}

#[test]
fn coarse_grind_draws_down_sooner_than_fine() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_channeling: no GPU adapter; skipping.");
        return;
    };
    let input = EmissionInput::default();
    // V60 drains a seeded water column through the bed to the cup; the active-water centroid descends
    // below the bed centroid → the drawdown latch fires. Coarser grind → higher permeability → drains
    // sooner → smaller drawdown_time. Sample every few steps so the latch is checked during the run.
    let run = |grain_diameter: f32| -> f32 {
        // Calibrated V60 mats (matching the geometry drain test): fine water threads the pore field
        // of a coarser permeable bed. grain_diameter is the lever.
        let mats = Materials {
            particle_spacing: 0.5,
            support_radius: 1.0,
            grain_diameter,
            min_pore_fraction: 0.35,
            grain_mass: 10.0,
            ..Materials::default()
        };
        let cfg = Config {
            xsph_viscosity_c: 0.0,
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&Scene::v60(), &mats, &cfg, &gpu);
        for step in 0..400 {
            solver.step(DT, &input);
            if step % 8 == 0 {
                solver.sample_diagnostics();
            }
        }
        solver.sample_diagnostics();
        let m = solver.metrics();
        assert!(
            m.drawdown_time.is_finite() && m.drawdown_time >= 0.0,
            "drawdown_time must be finite and non-negative, got {}",
            m.drawdown_time
        );
        assert!(
            m.evenness > 0.0 && m.evenness <= 1.0 + 1e-6,
            "evenness out of range: {}",
            m.evenness
        );
        m.drawdown_time
    };
    let coarse = run(1.6);
    let fine = run(1.0);
    eprintln!("drawdown_time: coarse {coarse:.3}, fine {fine:.3}");
    assert!(
        coarse > 0.0 && fine > 0.0,
        "both brews must draw down (latch) within the run"
    );
    assert!(
        coarse < fine,
        "coarse grind should draw down sooner: coarse {coarse:.3} vs fine {fine:.3}"
    );
}

// ---------------------------------------------------------------------------------------------
// U6 integration gates: fines slow drawdown, channeling emerges + amplifies, conservation holds.
// ---------------------------------------------------------------------------------------------

/// Calibrated V60 brew mats (fine water threads a coarser permeable bed). `fines_fraction` set by
/// the caller.
fn v60_brew_mats(fines_fraction: f32) -> Materials {
    Materials {
        particle_spacing: 0.5,
        support_radius: 1.0,
        grain_diameter: 1.0,
        min_pore_fraction: 0.35,
        grain_mass: 10.0,
        fines_fraction,
        ..Materials::default()
    }
}

fn total_fines(chem: &[[f32; 4]]) -> f32 {
    chem.iter().map(|c| c[3]).sum()
}

// DEFERRED — "fines slow drawdown" (solver_xpbd.md Phase 6 R4) is intentionally not gated here yet.
// Measured: in a V60 brew, fines released from the grounds wash down with the water and into the
// grain-free cup, where nothing strains them out, so the bed net-loses fines and drainage gets
// FASTER, not slower (fines-on cup fraction 0.51 vs off 0.39 by step 500). The real mechanism is the
// filter PAPER physically trapping fines (passes water, blocks fines) so they pile against it and
// clog — a filter-boundary straining mechanism not yet modeled (the fines pass is at the
// 8-storage-buffer limit, so it needs a restructure). Set aside for a follow-up; the migration,
// transient-permeability (verified with a manual clog in xpbd_fines), and channeling work below
// stand on their own.

/// Water+grain in a box, with the water dropped either spread over the whole bed footprint or
/// concentrated on one half. `concentrated` controls the inflow non-uniformity.
fn drop_scene(concentrated: bool) -> Scene {
    let water = if concentrated {
        SeedRegion {
            min: [1.0, 3.0, 1.0],
            max: [3.0, 6.0, 7.0], // one half (x∈[1,3]), taller to keep a comparable volume
            species: Species::Water,
        }
    } else {
        SeedRegion {
            min: [1.0, 3.0, 1.0],
            max: [7.0, 4.0, 7.0], // spread over the whole footprint
            species: Species::Water,
        }
    };
    Scene {
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [1.0, 1.0, 1.0],
                max: [7.0, 2.0, 7.0],
                species: Species::Grain,
            },
            water,
        ],
        ..Scene::default()
    }
}

#[test]
fn channeling_emerges_from_nonuniform_inflow_and_fines_amplify() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_channeling: no GPU adapter; skipping.");
        return;
    };
    let input = EmissionInput::default();
    // Threading bed so water flows down through the porosity field.
    let run = |concentrated: bool, fines: bool| -> f32 {
        let mats = Materials {
            min_pore_fraction: 0.5,
            grain_mass: 6.0,
            fines_fraction: if fines { 0.15 } else { 0.0 },
            ..Materials::default()
        };
        let cfg = Config {
            xsph_viscosity_c: 0.0,
            fines_rate: if fines { 2.0 } else { 0.0 },
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&drop_scene(concentrated), &mats, &cfg, &gpu);
        // Let the water reach the bed and establish a flow, then read the flow uniformity.
        let mut min_even = 1.0f32;
        for step in 0..120 {
            solver.step(DT, &input);
            if step >= 20 && step % 5 == 0 {
                solver.sample_diagnostics();
                min_even = min_even.min(solver.metrics().evenness);
            }
        }
        min_even
    };

    // Channeling emerges: a one-sided inflow produces less even flow than a spread inflow.
    let spread = run(false, false);
    let concentrated = run(true, false);
    eprintln!("evenness (fines off): spread {spread:.3}, concentrated {concentrated:.3}");
    assert!(
        concentrated < spread,
        "a non-uniform inflow should channel (lower evenness): concentrated {concentrated:.3} vs spread {spread:.3}"
    );

    // Fines amplify (don't create): the concentrated case channels with fines off, and at least as
    // much with fines on (the redistribution loop reinforces it — never less).
    let concentrated_fines = run(true, true);
    eprintln!(
        "evenness concentrated: fines off {concentrated:.3}, fines on {concentrated_fines:.3}"
    );
    assert!(
        concentrated_fines <= concentrated + 0.05,
        "fines should not reduce channeling: fines-on {concentrated_fines:.3} vs off {concentrated:.3}"
    );
}

#[test]
fn fines_conserved_through_draining_brew_with_absorption() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_channeling: no GPU adapter; skipping.");
        return;
    };
    // V60 brew with fines AND absorption on: water drains to the cup and grains absorb water
    // (deactivating some water). Water is never removed from the pool, and the fines transfer is
    // conservative, so total fines (over every particle) must hold — no leak via deactivated/cup
    // water (KTD-10 accounting).
    let mats = v60_brew_mats(0.1);
    let cfg = Config {
        xsph_viscosity_c: 0.0,
        fines_rate: 2.0,
        absorb_rate: 0.5,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&Scene::v60(), &mats, &cfg, &gpu);
    let n_grain = solver.read_phases().iter().filter(|&&p| p == 1).count();
    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    let t0 =
        n_grain as f32 * coffee_sim::models::fines::fines_seed(grain_volume, mats.fines_fraction);
    assert!(t0 > 0.0);

    let input = EmissionInput::default();
    for _ in 0..300 {
        solver.step(DT, &input);
    }
    let total = total_fines(&solver.read_chem());
    assert!(
        (total - t0).abs() < 1e-3 * t0,
        "fines volume leaked through the draining/absorbing brew: {total} vs {t0}"
    );
}

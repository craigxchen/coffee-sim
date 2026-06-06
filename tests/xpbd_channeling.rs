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
        // Calibrated V60 mats (matching the geometry drain test): fine water threads a coarser
        // permeable bed (water_grain_distance < grain spacing). grain_diameter is the lever.
        let mats = Materials {
            particle_spacing: 0.5,
            support_radius: 1.0,
            grain_diameter,
            water_grain_distance: 0.35,
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

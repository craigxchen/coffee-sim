//! Phase 1.4 wetting/cohesion invariants (GPU-gated; skips without an adapter).
//!
//! U4 scope (this file's current tests): the moisture state on `pos.w` is seeded per species,
//! mirrored to `pred.w` each frame, and preserved by every `pos`/`pred` writer — including
//! `apply_drag_pred`, which runs in the drag/buoyancy subcycle and previously zeroed `.w`. The
//! volume/mass/momentum conservation tests (U5/U9) build on these once the absorption passes land.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

/// A small mixed water+grain scene: a water slab released just above a grain slab, so the coupling
/// passes (drag/buoyancy → `apply_drag_pred`) actually run.
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

/// Expected seeded moisture for a particle by phase: water `f_w = 1`, grain `V_abs = 0`.
fn seeded_for_phase(ph: u32) -> f32 {
    if ph == 0 {
        1.0
    } else {
        0.0
    }
}

#[test]
fn seeded_moisture_is_full_water_and_dry_grains() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let solver = XpbdSolver::build(
        &mixed_scene(),
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    let phase = solver.read_phases();
    let moisture = solver.read_moisture();
    assert_eq!(moisture.len(), phase.len());
    assert!(
        phase.contains(&0) && phase.contains(&1),
        "scene must be mixed"
    );
    for (i, (&ph, &m)) in phase.iter().zip(&moisture).enumerate() {
        assert_eq!(
            m,
            seeded_for_phase(ph),
            "particle {i} (phase {ph}) seeded moisture"
        );
    }
}

#[test]
fn moisture_lane_survives_steps_with_drag_and_buoyancy() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    // Default config keeps drag (subiters=4) and buoyancy (scale=1) on, so apply_drag_pred runs.
    let mut solver = XpbdSolver::build(
        &mixed_scene(),
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    let phase = solver.read_phases();
    let input = EmissionInput::default();

    for _ in 0..30 {
        solver.step(1.0 / 60.0, &input);
    }

    // No absorption pass exists yet (U5), so the moisture lane must be byte-for-byte unchanged.
    let moisture = solver.read_moisture();
    for (i, (&ph, &m)) in phase.iter().zip(&moisture).enumerate() {
        assert_eq!(
            m,
            seeded_for_phase(ph),
            "particle {i} moisture drifted (preservation bug)"
        );
    }

    // pred.w must still mirror pos.w at end of step — proving every pred writer (predict, apply_dp,
    // apply_drag_pred) preserved .w. A regression in apply_drag_pred (zeroing .w) trips this.
    let pred = solver.read_pred();
    for (i, (p, &m)) in pred.iter().zip(&moisture).enumerate() {
        assert!(
            (p[3] - m).abs() < 1.0e-6,
            "particle {i}: pred.w {} != pos.w {m}",
            p[3]
        );
    }
}

#[test]
fn reset_restores_seeded_moisture() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let scene = mixed_scene();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let phase = solver.read_phases();

    for _ in 0..10 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    solver.reset(&scene);

    let moisture = solver.read_moisture();
    for (i, (&ph, &m)) in phase.iter().zip(&moisture).enumerate() {
        assert_eq!(
            m,
            seeded_for_phase(ph),
            "particle {i} moisture not restored on reset"
        );
    }
}

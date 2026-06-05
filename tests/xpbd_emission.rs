//! Pour-emission invariants (GPU-gated; skips without an adapter).
//!
//! U1 scope (this file's current tests): the capacity-pool + active-count refactor. A scene with no
//! pour declares `capacity == active_count == seed`, so behavior is unchanged; a scene that declares
//! a pour gets extra dormant pool capacity that must be provably inert (no emission is wired until
//! U2, so `active_count` stays at the seed). Pass selection must treat a declared pour as "water
//! present" even when only grains are seeded.

use coffee_sim::emission::PourEvent;
use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

/// A straight-down pour from `kettle` at `flow` (sim-volume/s).
fn pour(kettle: [f32; 3], flow: f32) -> EmissionInput {
    EmissionInput {
        kettle_pos: kettle,
        flow_rate: flow,
        pour_angle: 0.0,
        event: PourEvent::None,
    }
}

/// An empty box that opts into a pour (no seeds) — for counting emission in isolation.
fn empty_pour_scene(pour_ml: f32) -> Scene {
    Scene {
        pour_water_ml: pour_ml,
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [12.0, 12.0, 12.0],
        regions: vec![],
        ..Scene::default()
    }
}

/// A mixed water+grain scene (water present, so `has_water` is true regardless of pour).
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

/// Non-pour scene: capacity is exactly the seed count, active_count equals it, and no dormant slots.
#[test]
fn non_pour_scene_capacity_equals_seed() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let solver = XpbdSolver::build(
        &mixed_scene(),
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    let seed = solver.read_phases().len() as u32;
    assert_eq!(solver.active_count(), seed);
    assert_eq!(
        solver.pool_capacity(),
        seed,
        "non-pour scene allocated headroom"
    );
}

/// Declaring a pour grows the pool capacity but NOT the live set (emission lands in U2); the dormant
/// slots must be provably inert — the live solve must match the no-pour build step-for-step.
#[test]
fn dormant_capacity_is_inert() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let cfg = Config::default();

    let mut base = XpbdSolver::build(&mixed_scene(), &mats, &cfg, &gpu);
    let pour = Scene {
        pour_water_ml: 500.0, // big headroom
        ..mixed_scene()
    };
    let mut withcap = XpbdSolver::build(&pour, &mats, &cfg, &gpu);

    let seed = base.read_phases().len() as u32;
    assert_eq!(base.pool_capacity(), seed);
    assert!(
        withcap.pool_capacity() > seed,
        "pour scene did not allocate headroom: {} vs seed {seed}",
        withcap.pool_capacity()
    );
    // Both start with the same live set (no emission yet).
    assert_eq!(base.active_count(), seed);
    assert_eq!(withcap.active_count(), seed);

    for _ in 0..40 {
        base.step(1.0 / 60.0, &EmissionInput::default());
        withcap.step(1.0 / 60.0, &EmissionInput::default());
    }
    // active_count unchanged (emission not wired), and the live solve is identical — the dormant
    // capacity does not perturb the simulated particles.
    assert_eq!(
        withcap.active_count(),
        seed,
        "active_count drifted without emission"
    );
    let pa = base.read_positions();
    let pb = withcap.read_positions();
    assert_eq!(pa.len(), pb.len());
    for (i, (a, b)) in pa.iter().zip(&pb).enumerate() {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() <= 1.0e-6,
                "dormant capacity perturbed particle {i} lane {k}: {} vs {}",
                a[k],
                b[k]
            );
        }
    }
}

/// Pass selection: a grain-only scene that declares a pour must still enable the water passes (else
/// poured water would never be solved). Regression for `has_water |= declares_pour`.
#[test]
fn grain_only_pour_scene_enables_water_passes() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let grain_only = Scene {
        pour_water_ml: 250.0,
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![SeedRegion {
            min: [2.0, 2.0, 2.0],
            max: [5.0, 3.0, 5.0],
            species: Species::Grain,
        }],
        ..Scene::default()
    };
    let solver = XpbdSolver::build(&grain_only, &Materials::default(), &Config::default(), &gpu);
    assert!(
        solver.has_water_passes(),
        "grain-only pour scene must enable water passes (has_water |= declares_pour)"
    );
    // And the same scene WITHOUT a pour keeps water passes off.
    let no_pour = Scene {
        pour_water_ml: 0.0,
        ..grain_only
    };
    let solver2 = XpbdSolver::build(&no_pour, &Materials::default(), &Config::default(), &gpu);
    assert!(
        !solver2.has_water_passes(),
        "grain-only scene without a pour should not enable water passes"
    );
}

/// reset() returns the live set to the seed (drops any growth) — checked here against the inert pool
/// (active_count stays at seed across reset).
#[test]
fn reset_restores_seed_active_count() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        pour_water_ml: 500.0,
        ..mixed_scene()
    };
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let seed = solver.read_phases().len() as u32;
    for _ in 0..10 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    solver.reset(&scene);
    assert_eq!(solver.active_count(), seed);
}

// --- U2: emission core ---------------------------------------------------------------------------

/// Rate fidelity + no starvation: over T seconds at constant flow, the emitted particle count tracks
/// flow/V_w·T (the arclength-credit emitter must NOT starve at realistic — sub-spacing-per-frame —
/// pour speeds, which the old per-frame floor did).
#[test]
fn emission_rate_matches_flow_no_starvation() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    // Solve off so active_count == emitted count (emission is independent of the solve).
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mut solver =
        XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
    let v_w = solver.water_particle_volume();
    let flow = 5.0;
    let steps = 120u32; // 2 s
    for _ in 0..steps {
        solver.step(DT, &pour([6.0, 11.0, 6.0], flow));
    }
    let emitted = solver.active_count();
    let expected = flow / v_w * (steps as f32 * DT);
    assert!(
        emitted > 0,
        "emitter starved (0 particles) — arclength credit broken"
    );
    assert!(
        (emitted as f32 - expected).abs() <= 0.10 * expected + 8.0,
        "emitted {emitted} far from expected {expected:.1}"
    );
    // Emitted-mass counter tracks the count.
    let m = solver.total_emitted_water_mass();
    assert!(
        (m - emitted as f32 * 1.0).abs() <= 1.0e-3,
        "emitted-mass {m} != count {emitted} × particle_mass"
    );
}

/// A gentle pour (sub-spacing travel per frame) still emits — direct guard against the starvation
/// failure mode the per-frame floor caused.
#[test]
fn gentle_pour_does_not_starve() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mut solver =
        XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
    // Very gentle flow → exit_speed·dt ≪ particle_spacing; the old floor(travel/spacing)=0 would emit 0.
    for _ in 0..240 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 0.4));
    }
    assert!(
        solver.active_count() > 0,
        "gentle pour starved to 0 particles"
    );
}

/// Emitted particles are seeded as full, cold-of-solute water at the pour temperature.
#[test]
fn emitted_particles_are_seeded_water() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mats = Materials::default();
    let mut solver = XpbdSolver::build(&empty_pour_scene(9999.0), &mats, &cfg, &gpu);
    for _ in 0..30 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 5.0));
    }
    let phase = solver.read_phases();
    let chem = solver.read_chem();
    let moisture = solver.read_moisture();
    assert!(!phase.is_empty(), "no emission");
    assert_eq!(phase.len(), solver.active_count() as usize);
    for (i, &ph) in phase.iter().enumerate() {
        assert_eq!(ph, 0, "emitted particle {i} not water");
        assert!((moisture[i] - 1.0).abs() <= 1.0e-6, "emitted f_w != 1");
        assert!(chem[i][0].abs() <= 1.0e-6, "emitted c != 0");
        assert!(
            (chem[i][1] - mats.pour_t).abs() <= 1.0e-6,
            "emitted T != pour_t"
        );
    }
}

/// Determinism: identical pour inputs produce identical emission (count + positions).
#[test]
fn emission_is_deterministic() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let run = || {
        let mut solver =
            XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
        for _ in 0..60 {
            solver.step(DT, &pour([6.0, 11.0, 6.0], 5.0));
        }
        (solver.active_count(), solver.read_positions())
    };
    let (n1, p1) = run();
    let (n2, p2) = run();
    assert_eq!(n1, n2, "emission count not deterministic");
    for (a, b) in p1.iter().zip(&p2) {
        assert_eq!(a, b, "emission positions not deterministic");
    }
}

/// Capacity clamp: emission never exceeds the allocated pool (active_count saturates at capacity).
#[test]
fn emission_clamps_at_capacity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    // Tiny dose → small capacity; pour far more than it can hold.
    let mut solver = XpbdSolver::build(&empty_pour_scene(50.0), &Materials::default(), &cfg, &gpu);
    let cap = solver.pool_capacity();
    assert!(cap > 0);
    for _ in 0..600 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 20.0));
        assert!(
            solver.active_count() <= cap,
            "active_count {} exceeded capacity {cap}",
            solver.active_count()
        );
    }
    assert_eq!(
        solver.active_count(),
        cap,
        "pour did not fill the pool to capacity"
    );
}

/// PBF-safe inlet: continuous emission into a closed box with the FULL incompressibility solve on
/// stays finite and bounded — no density-spike eruption (the failure the arclength/rest-density inlet
/// is designed to prevent).
#[test]
fn emission_into_full_solve_does_not_erupt() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let cfg = Config::default(); // full water solve on
    let scene = Scene {
        pour_water_ml: 4000.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [12.0, 16.0, 12.0],
        regions: vec![],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..300 {
        solver.step(DT, &pour([6.0, 14.0, 6.0], 20.0));
    }
    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    assert!(
        solver.active_count() > 50,
        "too little emitted ({}) to be a meaningful stability test",
        solver.active_count()
    );
    assert!(
        pos.iter()
            .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite()),
        "non-finite position (inlet eruption)"
    );
    let vmax = vel
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        vmax <= cfg.max_speed + 1.0,
        "velocity blew past the cap (eruption): vmax {vmax}"
    );
}

/// A `PourEvent::Reset` clears the emitter's backlog/credit so a stale accumulator can't drain after
/// a reset (the emitter contract; full sim restart is `reset()`).
#[test]
fn reset_event_clears_emitter_backlog() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mut solver =
        XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
    // Pour briefly to build some emitter state.
    for _ in 0..20 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 5.0));
    }
    let after_pour = solver.active_count();
    // A Reset event with zero flow clears the backlog and emits nothing further.
    let reset_input = EmissionInput {
        kettle_pos: [6.0, 11.0, 6.0],
        flow_rate: 0.0,
        pour_angle: 0.0,
        event: PourEvent::Reset,
    };
    solver.step(DT, &reset_input);
    assert_eq!(
        solver.active_count(),
        after_pour,
        "Reset event should not emit; backlog must be cleared, not drained"
    );
}

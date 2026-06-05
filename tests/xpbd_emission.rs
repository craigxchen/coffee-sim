//! Pour-emission invariants (GPU-gated; skips without an adapter).
//!
//! U1 scope (this file's current tests): the capacity-pool + active-count refactor. A scene with no
//! pour declares `capacity == active_count == seed`, so behavior is unchanged; a scene that declares
//! a pour gets extra dormant pool capacity that must be provably inert (no emission is wired until
//! U2, so `active_count` stays at the seed). Pass selection must treat a declared pour as "water
//! present" even when only grains are seeded.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

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

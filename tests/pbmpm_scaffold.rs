//! PB-MPM U1 scaffold gates: seam contract + canonical layout + GPU budget (structural only).
//!
//! These need a GPU adapter (to build a `GpuContext`), so they skip gracefully on hosts
//! without one. The catalog/registry row itself is covered without a GPU by the in-module test
//! in `engine::registry` (which auto-extends over `SolverId::all()`). Per the plan's visual-first
//! posture (KTD3), U1 carries NO behavioral/physics assertions — the bounce/conservation gates
//! land in the shared harness in Phase B (U6/U7).

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::registry::{build_solver, SolverId};
use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::{PbmpmSolver, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

/// A tiny water-only block (4³ lattice at the default spacing).
fn water_only_scene() -> Scene {
    Scene {
        regions: vec![SeedRegion {
            min: [1.0, 1.0, 1.0],
            max: [4.0, 4.0, 4.0],
            species: Species::Water,
        }],
        ..Scene::default()
    }
}

/// Build via the registry arm (the three-edit pattern's runtime path) on an EMPTY scene: one
/// step leaves finite (empty) state, the getters return cached results without a GPU sync, and
/// the transfer pipeline still dispatches (budget is real even at zero particles).
#[test]
fn builds_via_registry_and_steps_an_empty_scene() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        regions: vec![],
        ..Scene::default()
    };
    let mut solver = build_solver(
        SolverId::Pbmpm,
        &scene,
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    solver.step(1.0 / 60.0, &EmissionInput::default());

    let metrics = solver.metrics();
    let profile = solver.profile();
    let particles = solver.particles();
    assert_eq!(metrics.particle_count, 0, "empty scene seeds nothing");
    assert_eq!(particles.particle_count, 0);
    assert!(
        profile.dispatches_per_frame > 0,
        "the pipeline dispatches even on an empty scene"
    );
    // Storage-buffer grant gate (KTD5): every entry point stays within the device's 9 grant.
    const _: () = assert!(
        MAX_STORAGE_BUFFERS_PER_ENTRY_POINT <= 9,
        "an entry point exceeds the 9 storage-buffer grant"
    );
    println!(
        "pbmpm budgets (empty scene): dispatches/frame {} | max storage buffers per entry point {}",
        profile.dispatches_per_frame, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT
    );
}

/// Water-only scene: `particles()` exposes the canonical layout, the particle count round-trips,
/// state stays finite across steps, and stepping is bit-exact deterministic on a re-run (R6).
#[test]
fn water_only_layout_round_trips_and_steps_deterministically() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = water_only_scene();
    let mats = Materials::default();
    let cfg = Config::default();
    let mut solver = PbmpmSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..3 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }

    // Canonical ParticleBuffers layout (what ui::Renderer binds: pos/vel/phase + chem lanes).
    let particles = solver.particles();
    assert!(
        particles.particle_count > 0,
        "the water block seeds particles"
    );
    assert!(particles.position.is_some());
    assert!(particles.velocity.is_some());
    assert!(particles.phase_tag.is_some());
    assert!(particles.concentration.is_some());
    assert!(particles.temperature.is_some());

    // Particle count round-trips through the read-back surface, all water (phase 0).
    let phases = solver.read_phases();
    assert_eq!(phases.len(), particles.particle_count as usize);
    assert!(
        phases.iter().all(|&p| p == 0),
        "water-only scene tags all 0"
    );
    assert_eq!(solver.active_count(), particles.particle_count);

    // Finite state after stepping (U1: the block free-falls under scene gravity).
    let pos = solver.read_positions();
    assert!(
        pos.iter().all(|p| p.iter().all(|x| x.is_finite())),
        "non-finite positions"
    );

    // Deterministic stepping (R6): a second build stepped identically lands byte-identical.
    let mut twin = PbmpmSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..3 {
        twin.step(1.0 / 60.0, &EmissionInput::default());
    }
    assert_eq!(twin.read_positions(), solver.read_positions());

    let profile = solver.profile();
    println!(
        "pbmpm budgets (water-only, {} particles): dispatches/frame {} | max storage buffers per entry point {} | device request 9/stage",
        particles.particle_count, profile.dispatches_per_frame, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT
    );
}

/// U2 (light, per visual-first KTD3): a pour activates dormant water-pool slots — `water_count`
/// grows while the pour is on, never exceeds the pre-allocated capacity, and a quiet input
/// (`flow_rate = 0`) emits nothing. No per-particle behavioral assertion (the bounce/conservation
/// gates are Phase B). Uses the `high-velocity-jet-impact` scene, which declares a pour.
#[test]
fn pour_grows_water_count_within_capacity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::debug_high_velocity_jet_impact();
    let mats = Materials::default();
    let cfg = Config::default();
    let mut solver = PbmpmSolver::build(&scene, &mats, &cfg, &gpu);

    // The declared pour sizes a pool with dormant headroom above the seed.
    let seed = solver.active_count();
    let capacity = solver.capacity();
    assert!(
        capacity > seed,
        "a declared pour reserves headroom (capacity {capacity} should exceed seed {seed})"
    );

    // A quiet step emits nothing: the live count holds at the seed.
    solver.step(1.0 / 60.0, &EmissionInput::default());
    assert_eq!(
        solver.active_count(),
        seed,
        "no pour ⇒ no emission (live count holds at the seed)"
    );

    // An active pour above the cup grows the live count over several frames, never past capacity.
    let pour = EmissionInput {
        kettle_pos: [0.0, scene.box_max[1] - 1.0, 0.0],
        flow_rate: 60.0,
        pour_angle: 0.0,
        ..EmissionInput::default()
    };
    for _ in 0..30 {
        solver.step(1.0 / 60.0, &pour);
        assert!(
            solver.active_count() <= capacity,
            "live count {} exceeded capacity {capacity}",
            solver.active_count()
        );
    }
    let after_pour = solver.active_count();
    assert!(
        after_pour > seed,
        "an active pour activates dormant slots (live count {after_pour} should exceed seed {seed})"
    );

    // Reset returns the live count to the seed (pour-activated slots re-park).
    solver.reset(&scene);
    assert_eq!(
        solver.active_count(),
        seed,
        "reset returns the live count to the seed"
    );

    println!(
        "pbmpm pour activation: seed {seed} → after 30 pour frames {after_pour} (capacity {capacity})"
    );
}

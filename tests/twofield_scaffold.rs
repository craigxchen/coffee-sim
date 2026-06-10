//! Twofield U1 scaffold gates: seam contract + canonical layout + GPU budgets.
//!
//! These need a GPU adapter (to build a `GpuContext`), so they skip gracefully on hosts
//! without one. The catalog/registry row itself is covered without a GPU by the in-module
//! test in `engine::registry` (which auto-extends over `SolverId::all()`).

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::registry::{build_solver, SolverId};
use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{TwofieldSolver, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT};
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

/// Build via the registry arm (the three-edit pattern's runtime path) on an EMPTY scene:
/// one step leaves finite (empty) state, the getters return cached results without a GPU
/// sync, and the scaffold pass still dispatches (budget is real even at zero particles).
#[test]
fn builds_via_registry_and_steps_an_empty_scene() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        regions: vec![],
        ..Scene::default()
    };
    let mut solver = build_solver(
        SolverId::Twofield,
        &scene,
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    solver.step(1.0 / 60.0, &EmissionInput::default());

    // Getters are cached (no device.poll(Wait) inside them) — calling them right after a
    // step must be cheap and consistent.
    let metrics = solver.metrics();
    let profile = solver.profile();
    let particles = solver.particles();
    assert_eq!(metrics.particle_count, 0, "empty scene seeds nothing");
    assert_eq!(particles.particle_count, 0);
    assert!(
        profile.dispatches_per_frame > 0,
        "the scaffold pass dispatches even on an empty scene"
    );
    println!(
        "twofield U1 budgets (empty scene): dispatches/frame {} | max storage buffers per entry point {}",
        profile.dispatches_per_frame, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT
    );
}

/// Water-only scene: `particles()` exposes the canonical layout with a populated phase_tag
/// buffer, state stays finite across steps, seeding is deterministic, and the budget numbers
/// are printed for the ladder's record.
#[test]
fn water_only_scene_has_canonical_layout_and_finite_state() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = water_only_scene();
    let mats = Materials::default();
    let cfg = Config::default();
    let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
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

    // Populated phase tags: a water-only scene is all PHASE_WATER (0), range layout water-first.
    let phases = solver.read_phases();
    assert_eq!(phases.len(), particles.particle_count as usize);
    assert!(
        phases.iter().all(|&p| p == 0),
        "water-only scene tags all 0"
    );
    let (water, solid) = solver.phase_counts();
    assert_eq!(water, particles.particle_count);
    assert_eq!(solid, 0);

    // Finite state after stepping; the inert pass moves nothing (velocities seed at zero).
    let pos = solver.read_positions();
    assert!(
        pos.iter().all(|p| p.iter().all(|x| x.is_finite())),
        "non-finite positions"
    );

    // Deterministic build (R6): a second build seeds byte-identical positions.
    let twin = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
    assert_eq!(twin.read_positions(), solver.read_positions());

    let profile = solver.profile();
    assert!(profile.dispatches_per_frame > 0);
    println!(
        "twofield U1 budgets (water-only, {} particles): dispatches/frame {} | max storage buffers per entry point {} | device request 9/stage",
        particles.particle_count, profile.dispatches_per_frame, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT
    );
}

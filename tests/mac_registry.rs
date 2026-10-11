//! The reference backend participates in the exact same registry, runtime switching,
//! reset, and renderer-buffer contracts as GPU solvers.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::{
    registry::{build_solver, info_for, SolverId},
    scene::{SeedRegion, Species},
    Scene, Simulator,
};
use coffee_sim::models::Materials;
use coffee_sim::utils::{config::Config, gpu::GpuContext};

#[test]
fn mac_is_registered_and_builds_canonical_buffers_and_resets() {
    assert!(SolverId::all().contains(&SolverId::Mac));
    assert_eq!(SolverId::Mac.id(), "mac");
    assert!(info_for(SolverId::Mac).name.contains("CPU reference"));
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("mac_registry: no GPU adapter; physics tests run independently on CPU");
        return;
    };
    let scene = Scene {
        box_max: [4.0; 3],
        regions: vec![SeedRegion {
            min: [1.0; 3],
            max: [2.0; 3],
            species: Species::Water,
        }],
        ..Scene::default()
    };
    let mats = Materials::default();
    let cfg = Config {
        seed_jitter: 0.0,
        ..Config::default()
    };
    let mut solver = build_solver(SolverId::Mac, &scene, &mats, &cfg, &gpu);
    let view = solver.particles();
    assert_eq!(view.particle_count, 1);
    assert!(view.position.is_some() && view.velocity.is_some() && view.phase_tag.is_some());
    assert!(view.concentration.is_some() && view.temperature.is_some() && view.moisture.is_some());
    solver.step(1.0 / 60.0, &EmissionInput::default());
    assert_eq!(solver.metrics().particle_count, 1);
    assert_eq!(solver.profile().dispatches_per_frame, 0);
    let empty = Scene {
        regions: vec![],
        ..scene.clone()
    };
    solver.reset(&empty);
    assert_eq!(solver.particles().particle_count, 0);
    solver.reset(&scene);
    assert_eq!(solver.particles().particle_count, 1);
    drop(solver);
    let mut sim = Simulator::new(gpu, scene, mats, cfg, SolverId::NoopA);
    sim.switch_solver(SolverId::Mac);
    assert_eq!(sim.active_id(), SolverId::Mac);
    assert_eq!(
        sim.step(0.01, &EmissionInput::default())
            .particles
            .particle_count,
        1
    );
    sim.switch_solver(SolverId::NoopB);
    assert_eq!(
        sim.step(0.01, &EmissionInput::default())
            .particles
            .particle_count,
        0
    );
}

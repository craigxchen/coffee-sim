//! Phase 0 acceptance: the runtime solver-switch through the real `Simulator`.
//!
//! This needs a GPU adapter (to build a `GpuContext`), so it skips gracefully on hosts
//! without one — the data-level switch invariant is covered without a GPU by the unit
//! tests in `engine::registry`.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::registry::SolverId;
use coffee_sim::engine::{Scene, Simulator};
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Paradigm;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

#[test]
fn simulator_steps_and_switches_solver_at_runtime() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("phase0: no GPU adapter; skipping GPU-bound switch test.");
        return;
    };

    let mut sim = Simulator::new(
        gpu,
        Scene::v60(),
        Materials::default(),
        Config::default(),
        SolverId::NoopA,
    );
    let input = EmissionInput::default();

    // Steps and reports a (zeroed) profile.
    let state = sim.step(1.0 / 60.0, &input);
    assert_eq!(sim.frame(), 1);
    assert_eq!(state.profile.dispatches_per_frame, 0);
    assert!(state.profile.passes.is_empty());
    assert_eq!(state.particles.particle_count, 0);
    assert_eq!(sim.active_info().paradigm, Paradigm::PositionBased);

    // The runtime switch flips the active solver on the same scene.
    sim.switch_solver(SolverId::NoopB);
    assert_eq!(sim.active_id(), SolverId::NoopB);
    assert_eq!(sim.active_info().paradigm, Paradigm::ForceBased);

    let state = sim.step(1.0 / 60.0, &input);
    assert_eq!(sim.frame(), 2);
    assert_eq!(state.profile.dispatches_per_frame, 0);
}

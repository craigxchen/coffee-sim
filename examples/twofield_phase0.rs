//! Twofield phase-0 acceptance (headless).
//!
//! Builds the two-field solver on a tiny water-only scene, runs a handful of steps (U2: the
//! block free-falls through the APIC transfer pipeline), asserts finite state and
//! `dispatches_per_frame > 0`, and prints the GPU budget numbers the validation ladder tracks
//! (R8). Skips gracefully without a GPU adapter.
//!
//! Run with: `cargo run --example twofield_phase0`

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{TwofieldSolver, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_phase0: no GPU adapter available; skipping.");
        return;
    };
    println!(
        "adapter: {} | timestamp-query: {}",
        gpu.adapter.get_info().name,
        gpu.timestamps_supported
    );

    // Tiny water-only block (4³ lattice at the default spacing).
    let scene = Scene {
        regions: vec![SeedRegion {
            min: [1.0, 1.0, 1.0],
            max: [4.0, 4.0, 4.0],
            species: Species::Water,
        }],
        ..Scene::default()
    };
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();

    for _ in 0..5 {
        solver.step(1.0 / 60.0, &input);
    }
    solver.sample_diagnostics();

    let profile = solver.profile();
    let metrics = solver.metrics();
    assert!(
        profile.dispatches_per_frame > 0,
        "the transfer pipeline must dispatch every frame"
    );
    assert!(
        metrics.particle_count > 0,
        "the water block seeds particles"
    );

    // Finite-state gate: every position and velocity lane after 5 steps.
    for (p, v) in solver.read_positions().iter().zip(solver.read_velocities()) {
        assert!(
            p.iter().all(|x| x.is_finite()) && v.iter().all(|x| x.is_finite()),
            "non-finite particle state"
        );
    }

    let (water, solid) = solver.phase_counts();
    println!(
        "twofield budgets: dispatches/frame {} | max storage buffers per entry point {} | device request 9/stage (src/utils/gpu.rs, not raised)",
        profile.dispatches_per_frame, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT
    );
    println!(
        "particles {} (water {}, solid {}) | passes timed: {:?}",
        metrics.particle_count, water, solid, profile.passes
    );
    println!("twofield_phase0: OK");
}

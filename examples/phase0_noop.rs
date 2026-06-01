//! Phase 0 acceptance demo (headless).
//!
//! Runs the no-op solver through the `Simulator` at ~60 fps, prints the `Profile` each
//! reporting frame (zero passes is expected), and demonstrates the runtime solver-switch
//! (`NoopA → NoopB`). Skips gracefully when no GPU adapter is available.
//!
//! Run with: `cargo run --example phase0_noop`

use std::time::{Duration, Instant};

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::registry::SolverId;
use coffee_sim::engine::{Scene, Simulator};
use coffee_sim::models::Materials;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("phase0_noop: no GPU adapter available; skipping.");
        return;
    };
    println!(
        "adapter: {} | timestamp-query: {}",
        gpu.adapter.get_info().name,
        gpu.timestamps_supported
    );

    let mut sim = Simulator::new(
        gpu,
        Scene::v60(),
        Materials::default(),
        Config::default(),
        SolverId::NoopA,
    );
    let input = EmissionInput::default();

    let frame_budget = Duration::from_micros(16_667); // ~60 fps
    let total_frames = 120u64;

    for f in 0..total_frames {
        let start = Instant::now();

        // Demonstrate the runtime solver-switch halfway through.
        if f == total_frames / 2 {
            sim.switch_solver(SolverId::NoopB);
            println!("--- switched solver to \"{}\" ---", sim.active_info().name);
        }

        let state = sim.step(1.0 / 60.0, &input);

        if f % 30 == 0 || f == total_frames / 2 {
            println!(
                "frame {:>3} | solver {:<8} | dispatches/frame {} | passes {} | particles {}",
                f,
                sim.active_info().name,
                state.profile.dispatches_per_frame,
                state.profile.passes.len(),
                state.metrics.particle_count,
            );
        }

        // Pace to ~60 fps.
        if let Some(remaining) = frame_budget.checked_sub(start.elapsed()) {
            std::thread::sleep(remaining);
        }
    }

    let info = sim.active_info();
    println!(
        "done: {} frames; final solver \"{}\" ({:?}, owns_grid={}, {:?})",
        sim.frame(),
        info.name,
        info.paradigm,
        info.owns_grid,
        info.stability,
    );
}

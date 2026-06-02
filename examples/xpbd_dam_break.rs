//! PBF water-core dam-break (headless).
//!
//! Builds the XPBD water solver on a dam-break scene, runs it, and every K frames prints
//! the stability signals: bounding-box extents, max speed, max bucket occupancy, effective
//! iterations, dispatches/frame, and per-pass GPU µs. The interim "watch it settle"
//! artifact until the renderer lands. Skips when no GPU adapter is available.
//!
//! Run with: `cargo run --example xpbd_dam_break`

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

fn extents(pts: &[[f32; 4]]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in pts {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}

fn max_speed(vels: &[[f32; 4]]) -> f32 {
    vels.iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0, f32::max)
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_dam_break: no GPU adapter available; skipping.");
        return;
    };
    println!(
        "adapter: {} | timestamp-query: {}",
        gpu.adapter.get_info().name,
        gpu.timestamps_supported
    );

    let scene = Scene::dam_break();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    println!("particles: {}", solver.particles().particle_count);

    let total_frames = 600u32;
    let report_every = 60u32;

    for f in 0..total_frames {
        solver.step(1.0 / 60.0, &input);

        if f % report_every == 0 || f == total_frames - 1 {
            solver.sample_diagnostics();
            let pos = solver.read_positions();
            let vel = solver.read_velocities();
            let (lo, hi) = extents(&pos);
            let diag = solver.diagnostics();
            let prof = solver.profile();
            let ts_total: f32 = prof.passes.iter().map(|(_, us)| *us).sum();
            println!(
                "f{:>4} | extent x[{:.1},{:.1}] y[{:.1},{:.1}] z[{:.1},{:.1}] | vmax {:>6.2} | occ {:>3} | iters {} | overflow {} | dispatch {} | gpu {:.1}µs ({} passes)",
                f,
                lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
                max_speed(&vel),
                diag.max_occupancy,
                diag.effective_iters,
                diag.overflow,
                prof.dispatches_per_frame,
                ts_total,
                prof.passes.len(),
            );
        }
    }
}

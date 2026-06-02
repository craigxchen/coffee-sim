//! PBF water-core dam-break (headless): long-run stability check of the default config.
//! Reports the post-settle peak speed (eruption detector), settled speed, and max occupancy.

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

fn max_speed(v: &[[f32; 4]]) -> f32 {
    v.iter()
        .map(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt())
        .fold(0.0, f32::max)
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_dam_break: no GPU adapter available; skipping.");
        return;
    };
    let scene = Scene::dam_break();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();

    let frames = 1800u32;
    let n = solver.particles().particle_count as usize;
    let mut post_peak = 0.0f32;
    let mut max_fast = 0usize; // most particles > 15 in one post-settle frame (global-jump detector)
    let mut max_occ = 0u32;
    let mut overflow = false;
    let mut final_v = 0.0f32;
    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 10 == 0 || f == frames - 1 {
            solver.sample_diagnostics();
            let vel = solver.read_velocities();
            final_v = max_speed(&vel);
            let d = solver.diagnostics();
            max_occ = max_occ.max(d.max_occupancy);
            overflow |= d.overflow;
            if f > 250 {
                post_peak = post_peak.max(final_v);
                max_fast = max_fast.max(
                    vel.iter()
                        .filter(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt() > 15.0)
                        .count(),
                );
            }
        }
    }
    println!(
        "particles {n} | post-settle peak {post_peak:.2} | max global-fast {max_fast} | settled {final_v:.2} | occ {max_occ} | overflow {overflow}"
    );
}

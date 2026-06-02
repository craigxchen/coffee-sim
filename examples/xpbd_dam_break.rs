//! PBF water-core dam-break (headless): checks stability + liveliness + spread.
//! Beading shows up as a small settled footprint (a tight blob instead of a floor-wide pool).

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

fn span(pos: &[[f32; 4]], axis: usize) -> (f32, f32) {
    pos.iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p[axis]), hi.max(p[axis]))
        })
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_dam_break: no GPU adapter available; skipping.");
        return;
    };
    let scene = Scene::dam_break();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    let n = solver.particles().particle_count as usize;

    let frames = 1600u32;
    let mut collapse_peak = 0.0f32; // liveliness during the fall (f<120)
    let mut max_fast = 0usize; // global-jump detector (f>300)
    let mut max_occ = 0u32;
    let mut overflow = false;
    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 10 == 0 || f == frames - 1 {
            solver.sample_diagnostics();
            let vel = solver.read_velocities();
            let v = max_speed(&vel);
            let d = solver.diagnostics();
            max_occ = max_occ.max(d.max_occupancy);
            overflow |= d.overflow;
            if f < 120 {
                collapse_peak = collapse_peak.max(v);
            }
            if f > 300 {
                max_fast = max_fast.max(
                    vel.iter()
                        .filter(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt() > 15.0)
                        .count(),
                );
            }
        }
    }
    let pos = solver.read_positions();
    let (xl, xh) = span(&pos, 0);
    let (yl, yh) = span(&pos, 1);
    let (zl, zh) = span(&pos, 2);
    let settled = max_speed(&solver.read_velocities());
    println!(
        "n {n} | collapse peak {collapse_peak:.1} | settled {settled:.2} | global-fast {max_fast} | occ {max_occ} ovf {overflow}\n\
         settled footprint: x[{xl:.1},{xh:.1}] z[{zl:.1},{zh:.1}] height y[{yl:.1},{yh:.1}]  (floor is 32×32; wide+flat = spread, small = beading)"
    );
}

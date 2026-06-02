//! PBF water-core dam-break (headless): stability + spread + GPU cost at a given particle
//! resolution. Set `SPACING` (default 1.0) to change particle size/count — smaller spacing
//! packs more particles into the same 16³ block. `h` scales with spacing, so the physics is
//! resolution-consistent.

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
        eprintln!("no GPU adapter; skipping.");
        return;
    };
    let spacing: f32 = std::env::var("SPACING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mats = Materials {
        particle_spacing: spacing,
        support_radius: 2.0 * spacing,
        particle_mass: 1.0,
    };
    let scene = Scene::dam_break();
    let mut solver = XpbdSolver::build(&scene, &mats, &Config::default(), &gpu);
    let n = solver.particles().particle_count as usize;
    println!("spacing {spacing} -> {n} particles");
    let input = EmissionInput::default();

    let frames = 1500u32;
    let mut collapse_peak = 0.0f32;
    let mut settled = 0.0f32;
    let mut max_fast = 0usize;
    let mut max_occ = 0u32;
    let mut overflow = false;
    let mut gpu_us_samples: Vec<f32> = Vec::new();

    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 15 == 0 || f == frames - 1 {
            solver.sample_diagnostics();
            let vel = solver.read_velocities();
            let v = max_speed(&vel);
            let d = solver.diagnostics();
            max_occ = max_occ.max(d.max_occupancy);
            overflow |= d.overflow;
            let us: f32 = solver.profile().passes.iter().map(|(_, t)| *t).sum();
            if us > 0.0 {
                gpu_us_samples.push(us);
            }
            if f < 120 {
                collapse_peak = collapse_peak.max(v);
            } else {
                max_fast = max_fast.max(
                    vel.iter()
                        .filter(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt() > 15.0)
                        .count(),
                );
            }
            settled = v;
        }
    }

    gpu_us_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f32| {
        gpu_us_samples
            .get(
                ((gpu_us_samples.len() as f32 * p) as usize)
                    .min(gpu_us_samples.len().saturating_sub(1)),
            )
            .copied()
            .unwrap_or(0.0)
            / 1000.0
    };
    let pos = solver.read_positions();
    let (xl, xh) = span(&pos, 0);
    let (yl, yh) = span(&pos, 1);
    let (zl, zh) = span(&pos, 2);
    println!(
        "collapse {collapse_peak:.1} | settled {settled:.2} | global-fast {max_fast} | occ {max_occ} ovf {overflow}\n\
         GPU/frame: median {:.2} ms, p95 {:.2} ms, worst {:.2} ms (16.7 ms = 60 fps budget)\n\
         footprint x[{xl:.1},{xh:.1}] z[{zl:.1},{zh:.1}] y[{yl:.1},{yh:.1}]",
        pct(0.5), pct(0.95), pct(1.0)
    );
}

//! Water/bed coupling probe (headless): a water column poured onto a grain bed (mixed scene).
//! Reports whether water rests on / in the bed (vs tunneling through to the floor), the bed top,
//! stability, drawdown, momentum drift, and GPU cost.
//!
//! Run: `GRIND=1.4 cargo run --release --example coupling`

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

fn speed(v: [f32; 4]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn momentum(vel: &[[f32; 4]], phase: &[u32], mats: &Materials) -> [f32; 3] {
    vel.iter().zip(phase).fold([0.0; 3], |mut acc, (v, &ph)| {
        let m = if ph == 1 {
            mats.grain_mass
        } else {
            mats.particle_mass
        };
        acc[0] += m * v[0];
        acc[1] += m * v[1];
        acc[2] += m * v[2];
        acc
    })
}

/// (low 2% y, high 2% y) of the particles in `phase_sel`, robust to a few stragglers.
fn y_band(pos: &[[f32; 4]], phase: &[u32], sel: u32) -> (f32, f32) {
    let mut ys: Vec<f32> = pos
        .iter()
        .zip(phase)
        .filter(|(_, &ph)| ph == sel)
        .map(|(p, _)| p[1])
        .collect();
    if ys.is_empty() {
        return (0.0, 0.0);
    }
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = ys[ys.len() / 50];
    let hi = ys[ys.len() - 1 - ys.len() / 50];
    (lo, hi)
}

fn low_water_mean(pos: &[[f32; 4]], phase: &[u32]) -> f32 {
    let mut ys: Vec<f32> = pos
        .iter()
        .zip(phase)
        .filter(|(_, &ph)| ph == 0)
        .map(|(p, _)| p[1])
        .collect();
    if ys.is_empty() {
        return 0.0;
    }
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = (ys.len() / 20).max(1);
    ys.iter().take(n).sum::<f32>() / n as f32
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("no GPU adapter; skipping.");
        return;
    };
    let grind = std::env::var("GRIND")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.0)
        .max(0.1);
    let mut mats = Materials::default();
    mats.grain_diameter *= grind;
    let scene = Scene::pour_over();
    let floor_y = scene.box_min[1];
    let mut solver = XpbdSolver::build(&scene, &mats, &Config::default(), &gpu);
    let phase = solver.read_phases();
    let n = phase.len();
    let n_grain = phase.iter().filter(|&&p| p == 1).count();
    let n_water = n - n_grain;
    println!(
        "pour_over -> {n} particles ({n_water} water, {n_grain} grain), GRIND {grind:.2}, d {:.3}",
        mats.grain_diameter
    );
    let input = EmissionInput::default();
    let p0 = momentum(&solver.read_velocities(), &phase, &mats);

    let frames = 900u32;
    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 150 == 0 || f == frames - 1 {
            let pos = solver.read_positions();
            let vel = solver.read_velocities();
            let (wlo, whi) = y_band(&pos, &phase, 0);
            let (glo, ghi) = y_band(&pos, &phase, 1);
            let drawdown = low_water_mean(&pos, &phase);
            let vmax = vel.iter().map(|&v| speed(v)).fold(0.0, f32::max);
            let p = momentum(&vel, &phase, &mats);
            let drift =
                ((p[0] - p0[0]).powi(2) + (p[1] - p0[1]).powi(2) + (p[2] - p0[2]).powi(2)).sqrt();
            let in_box = pos.iter().all(|p| {
                (0..3).all(|a| p[a] >= scene.box_min[a] - 0.6 && p[a] <= scene.box_max[a] + 0.6)
            });
            eprintln!(
                "  f{f}: water y[{wlo:.1},{whi:.1}] low5 {drawdown:.2} grain y[{glo:.1},{ghi:.1}] vmax {vmax:.2} |p-p0| {drift:.3} in_box {in_box}"
            );
        }
    }

    let pos = solver.read_positions();
    let (wlo, whi) = y_band(&pos, &phase, 0);
    let (glo, ghi) = y_band(&pos, &phase, 1);
    let _ = floor_y;
    // The real step-1 invariant: water threads the PORES without passing through grain BODIES.
    // Measure the closest any water particle gets to any grain — should stay near the water–grain
    // contact distance d_wg (≈ grain_diameter), not collapse to ~0 (tunneling through a grain).
    let d_wg = mats.grain_diameter;
    let waters: Vec<[f32; 4]> = pos
        .iter()
        .zip(&phase)
        .filter(|(_, &p)| p == 0)
        .map(|(p, _)| *p)
        .collect();
    let grains: Vec<[f32; 4]> = pos
        .iter()
        .zip(&phase)
        .filter(|(_, &p)| p == 1)
        .map(|(p, _)| *p)
        .collect();
    let mut min_wg = f32::INFINITY;
    for w in &waters {
        for g in &grains {
            let d = ((w[0] - g[0]).powi(2) + (w[1] - g[1]).powi(2) + (w[2] - g[2]).powi(2)).sqrt();
            min_wg = min_wg.min(d);
        }
    }
    let drawdown = low_water_mean(&pos, &phase);
    println!(
        "final: water y[{wlo:.1},{whi:.1}], bed y[{glo:.1},{ghi:.1}]\n\
         low5 water y {drawdown:.2}; min water–grain distance {min_wg:.3} (d_wg {d_wg:.2}; tunneling if ≪ d_wg)"
    );
}

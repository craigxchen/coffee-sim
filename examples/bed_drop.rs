//! Dry granular-bed drop (headless): a grain block is dropped onto the flat floor and flows into
//! a heap. Reports the diagnostic angle of repose, settled speed, static %, max penetration, the
//! second-half height creep, and GPU cost. Set `SPACING` (default 1.0) to change grain size/count;
//! the grain contact diameter and support radius scale with it so the physics stays consistent.
//!
//! Run: `cargo run --example bed_drop`

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

/// `SPACING` env (grain pitch). Clamped to the floor below which the neighbor grid would exceed
/// the WebGPU 128 MB buffer-binding limit (same as the water example).
fn parse_spacing() -> f32 {
    const MIN_SPACING: f32 = 0.25;
    let requested: f32 = std::env::var("SPACING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let spacing = requested.max(MIN_SPACING);
    if spacing != requested {
        eprintln!(
            "SPACING {requested} is too small (grid buffer would exceed the GPU limit); using {spacing}"
        );
    }
    spacing
}

fn max_speed(v: &[[f32; 4]]) -> f32 {
    v.iter()
        .map(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt())
        .fold(0.0, f32::max)
}

/// Diagnostic angle of repose of the settled heap: peak height H over base radius R about the
/// horizontal centroid, θ = atan(H / R). Spheres roll, so this reads shallower than `atan(μ)`;
/// it's a sanity diagnostic, not a calibrated target.
fn repose_deg(pos: &[[f32; 4]], floor_y: f32) -> (f32, f32, f32) {
    let n = pos.len() as f32;
    let (cx, cz) = pos
        .iter()
        .fold((0.0f32, 0.0f32), |(ax, az), p| (ax + p[0], az + p[2]));
    let (cx, cz) = (cx / n, cz / n);
    let height = pos.iter().map(|p| p[1] - floor_y).fold(0.0, f32::max);
    let radius = pos
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[2] - cz).powi(2)).sqrt())
        .fold(0.0, f32::max);
    let deg = if radius > 1e-3 {
        (height / radius).atan().to_degrees()
    } else {
        90.0
    };
    (deg, height, radius)
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("no GPU adapter; skipping.");
        return;
    };
    let spacing = parse_spacing();
    let mats = Materials {
        particle_spacing: spacing,
        support_radius: 2.0 * spacing,
        grain_diameter: spacing,
        ..Materials::default()
    };
    let scene = Scene::bed_drop();
    let floor_y = scene.box_min[1];
    let cfg = Config::default();
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let n = solver.particles().particle_count as usize;
    println!("spacing {spacing} -> {n} grains");
    let input = EmissionInput::default();

    let frames = 1200u32;
    let mut collapse_peak = 0.0f32;
    let mut settled = 0.0f32;
    let mut max_occ = 0u32;
    let mut overflow = false;
    let mut max_pen = 0.0f32;
    let mut mid_height = 0.0f32; // peak height once settled — for the creep check vs the end
    let mut max_eff = 0u32; // worst-frame effective iterations (adaptive early-exit diagnostic)
    let mut last_eff = 0u32;
    let mut last_pen = 0.0f32; // settled (final-frame) penetration
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
            max_pen = max_pen.max(d.residual);
            max_eff = max_eff.max(d.effective_iters);
            last_eff = d.effective_iters;
            last_pen = d.residual;
            let us: f32 = solver.profile().passes.iter().map(|(_, t)| *t).sum();
            if us > 0.0 {
                gpu_us_samples.push(us);
            }
            if f < 180 {
                collapse_peak = collapse_peak.max(v);
            }
            if f == frames / 2 {
                let (_, h, _) = repose_deg(&solver.read_positions(), floor_y);
                mid_height = h;
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
    let vel = solver.read_velocities();
    let (deg, height, radius) = repose_deg(&pos, floor_y);
    // Static fraction from actual velocities (no freeze flag any more): grains below 0.1 units/s.
    let still = vel
        .iter()
        .filter(|v| max_speed(std::slice::from_ref(v)) < 0.1)
        .count();
    let still_pct = 100.0 * still as f32 / n.max(1) as f32;
    let creep = (mid_height - height).abs();
    println!(
        "collapse {collapse_peak:.1} | settled {settled:.3} | static {still}/{n} ({still_pct:.0}%) | \
         max-penetration {max_pen:.3}·d | occ {max_occ} ovf {overflow}\n\
         iters: worst {max_eff}, final {last_eff} (cap {}) | settled-pen {last_pen:.3}·d | \
         heap: height {height:.1}, base radius {radius:.1}, repose ≈ {deg:.0}° (diagnostic) | \
         creep (Δheight 2nd half) {creep:.2}\n\
         GPU/frame: median {:.2} ms, p95 {:.2} ms, worst {:.2} ms (16.7 ms = 60 fps budget)",
        cfg.bed_max_iters,
        pct(0.5),
        pct(0.95),
        pct(1.0)
    );
}

//! Scaling probe: how does per-frame GPU cost scale with particle count, and is that scaling
//! driven by memory locality? Sweeps the dam-break water scene across resolutions and reports, per
//! point: active particles, max per-cell occupancy, GPU µs/frame, µs per 1000 particles, and the
//! `compute_lambda + compute_dp` share of frame time. It fits the cost exponent `k` in
//! `total_µs ∝ N^k` (least squares on log-log) over the GPU-saturated range.
//!
//! This is the success gate for the particle-reorder optimization
//! (docs/plans/2026-06-06-001-feat-particle-reorder-locality-plan.md): success is the *curve*
//! — `k` bending from ~1.25 toward ~1.0 and the lambda+dp share dropping — while `max_occ`
//! stays unchanged (the attribution guard: positions/occupancy are untouched by a reorder).
//!
//! Run: `cargo run --release --example scaling_probe`.

use coffee_sim::emission::{EmissionInput, PourEvent};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;

#[derive(Clone, Copy)]
struct Row {
    spacing: f32,
    n: u32,
    max_occ: u32,
    gpu_us: f32,
    per_kpart: f32,
    share: f32, // compute_lambda + compute_dp fraction of frame µs
}

/// Resolution-consistent materials: support_radius = grain_diameter = 2·spacing (cell_size = h).
fn mats_for(s: f32) -> Materials {
    Materials {
        particle_spacing: s,
        support_radius: 2.0 * s,
        grain_diameter: 2.0 * s,
        water_grain_distance: 0.7 * s,
        grain_mass: 10.0,
        ..Materials::default()
    }
}

/// Run one resolution: warm up with `drive`, then time `window` frames. Returns the measured row.
fn measure(
    gpu: &GpuContext,
    scene: &Scene,
    s: f32,
    warmup: u32,
    window: u32,
    drive: &EmissionInput,
) -> Row {
    let mats = mats_for(s);
    let cfg = Config::default();
    let mut solver = XpbdSolver::build(scene, &mats, &cfg, gpu);

    for _ in 0..warmup {
        solver.step(DT, drive);
    }

    let mut gpu_us_sum = 0.0f32;
    let mut hot_sum = 0.0f32;
    let mut samples = 0u32;
    let mut max_occ = 0u32;
    for _ in 0..window {
        solver.step(DT, drive);
        solver.sample_diagnostics();
        let prof = solver.profile();
        let total = prof.total_micros();
        if total > 0.0 {
            let hot: f32 = prof
                .passes
                .iter()
                .filter(|(l, _)| l == "compute_lambda" || l == "compute_dp")
                .map(|(_, us)| us)
                .sum();
            gpu_us_sum += total;
            hot_sum += hot;
            samples += 1;
        }
        max_occ = max_occ.max(solver.diagnostics().max_occupancy);
    }

    let n = solver.active_count();
    let gpu_us = if samples > 0 {
        gpu_us_sum / samples as f32
    } else {
        0.0
    };
    let share = if gpu_us_sum > 0.0 {
        hot_sum / gpu_us_sum
    } else {
        0.0
    };
    let per_kpart = gpu_us / (n as f32 / 1000.0).max(1e-6);
    Row {
        spacing: s,
        n,
        max_occ,
        gpu_us,
        per_kpart,
        share,
    }
}

/// Least-squares slope of ln(gpu_us) vs ln(n) — the cost exponent k in total_µs ∝ N^k.
fn fit_exponent(rows: &[Row]) -> f32 {
    let pts: Vec<(f32, f32)> = rows
        .iter()
        .filter(|r| r.n > 0 && r.gpu_us > 0.0)
        .map(|r| ((r.n as f32).ln(), r.gpu_us.ln()))
        .collect();
    let m = pts.len() as f32;
    if m < 2.0 {
        return f32::NAN;
    }
    let sx: f32 = pts.iter().map(|p| p.0).sum();
    let sy: f32 = pts.iter().map(|p| p.1).sum();
    let sxx: f32 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: f32 = pts.iter().map(|p| p.0 * p.1).sum();
    (m * sxy - sx * sy) / (m * sxx - sx * sx)
}

fn sweep(
    gpu: &GpuContext,
    label: &str,
    scene: &Scene,
    spacings: &[f32],
    warmup: u32,
    drive: &EmissionInput,
) {
    println!("\n=== {label} ===");
    println!(
        "{:>8} {:>9} {:>8} {:>11} {:>10} {:>8}",
        "spacing", "active", "max_occ", "gpu_us", "us/Kpart", "lam+dp%"
    );
    let mut rows = Vec::new();
    for &s in spacings {
        let r = measure(gpu, scene, s, warmup, 40, drive);
        println!(
            "{:>8.3} {:>9} {:>8} {:>11.1} {:>10.2} {:>7.0}%",
            r.spacing,
            r.n,
            r.max_occ,
            r.gpu_us,
            r.per_kpart,
            r.share * 100.0
        );
        rows.push(r);
    }
    // Fit excludes the smallest point (GPU-underutilized: fixed launch overhead inflates its
    // µs/Kpart and flattens the apparent slope). The headline metric is the µs/Kpart climb over
    // the GPU-saturated range = max/min, which is what the reorder must flatten.
    let saturated: Vec<Row> = rows.iter().skip(1).map(|r| Row { ..*r }).collect();
    let k = fit_exponent(&saturated);
    // Climb over the SATURATED range only (the small-N point is launch-overhead-dominated and would
    // otherwise masquerade as the peak). Flat ⇒ ~1.0; super-linear ⇒ grows with N.
    let kpart_min = saturated
        .iter()
        .map(|r| r.per_kpart)
        .fold(f32::INFINITY, f32::min);
    let kpart_max = saturated.iter().map(|r| r.per_kpart).fold(0.0, f32::max);
    let climb = if kpart_min > 0.0 {
        kpart_max / kpart_min
    } else {
        0.0
    };
    println!(
        "  fit (N≥{}): total_us ∝ N^{k:.3}   |   µs/Kpart climb (max/min): {climb:.2}x   |   N range: {}..{}",
        saturated.first().map(|r| r.n).unwrap_or(0),
        rows.first().map(|r| r.n).unwrap_or(0),
        rows.last().map(|r| r.n).unwrap_or(0),
    );
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("scaling_probe: no GPU adapter; cannot run.");
        return;
    };

    // dam_break (water-only, isolates the density-solve hot path). 16³ block ⇒ N ≈ (16/s)³.
    let idle = EmissionInput {
        kettle_pos: [0.0, 0.0, 0.0],
        flow_rate: 0.0,
        pour_angle: 0.0,
        event: PourEvent::None,
    };
    sweep(
        &gpu,
        "dam_break (water-only)",
        &Scene::dam_break(),
        &[1.27, 0.64, 0.45, 0.34, 0.27],
        180,
        &idle,
    );
}

//! PB-MPM scaling probe (U6): how does the per-FRAME GPU cost of the single-phase water-impact
//! scene scale with particle count, from ~5k toward the 200k real-time target — and WHERE does
//! the frame go (the dominant pass, named)?
//!
//! Mirrors `examples/twofield_scaling.rs`. The composition is the PRE-REGISTERED gate scene of
//! `tests/solver_perf.rs` (water-only column, box-edge composed, free-fall impact + center
//! pour), measured at the FROZEN bounce-winning `iteration_count` from `Config::default()`.
//! PB-MPM runs ONE substep per frame, so per-frame cost = the per-pass timestamp sum; the
//! iteration loop's cost shows up as the repeated `p2g`/`g2p`/`particle_update` labels
//! aggregated below (each label sums its per-iteration instances).
//!
//! Run: `cargo run --release --example pbmpm_scaling`.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::scene::{Scene, SeedRegion, Species};
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::PbmpmSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;

// The pre-registered gate composition (KEEP IN SYNC with tests/solver_perf.rs).
const WATER_COL: f32 = 52.0;
const DROP_GAP: f32 = 6.0;
const JET_FLOW: f32 = 70.0;
const WARMUP_FRAMES: u32 = 50;
const MEASURE_FRAMES: u32 = 40;

// Swept box edges: N spans ~5k → ~200k at the fixed composition.
const EDGES: &[f32] = &[8.0, 14.0, 22.0, 32.0, 44.0, 56.0, 61.0];

fn scene(edge: f32) -> Scene {
    let water_lo = DROP_GAP;
    let water_hi = water_lo + WATER_COL;
    let y = (water_hi + 16.0).round();
    Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 6000.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [edge, y, edge],
        solids: Vec::new(),
        regions: vec![SeedRegion {
            min: [0.7, water_lo, 0.7],
            max: [edge - 0.7, water_hi, edge - 0.7],
            species: Species::Water,
        }],
    }
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_scaling: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config::default();
    println!(
        "pbmpm scaling (frozen iteration_count {}, dt {DT}):",
        cfg.pbmpm_iteration_count
    );
    println!(
        "{:>8} {:>10} {:>12} {:>14} {:>10}",
        "edge", "N", "median ms", "µs/Kpart", "disp/frame"
    );

    for &edge in EDGES {
        let sc = scene(edge);
        let mut solver = PbmpmSolver::build(&sc, &Materials::default(), &cfg, &gpu);
        let axis = edge / 2.0;
        let quiet = EmissionInput::default();
        let pour = EmissionInput {
            kettle_pos: [axis, DROP_GAP + WATER_COL + 6.0, axis],
            flow_rate: JET_FLOW,
            pour_angle: 0.0,
            ..EmissionInput::default()
        };
        for f in 0..WARMUP_FRAMES {
            let drive = if f < WARMUP_FRAMES / 3 { &quiet } else { &pour };
            solver.step(DT, drive);
        }

        let mut samples: Vec<f32> = Vec::new();
        let mut agg: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        let mut dispatches = 0u32;
        for _ in 0..MEASURE_FRAMES {
            solver.step(DT, &pour);
            solver.sample_diagnostics();
            let prof = solver.profile();
            dispatches = prof.dispatches_per_frame;
            let sub = prof.total_micros();
            if sub > 0.0 {
                samples.push(sub);
                for (label, us) in &prof.passes {
                    *agg.entry(label.clone()).or_insert(0.0) += us;
                }
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = samples.get(samples.len() / 2).copied().unwrap_or(0.0);
        let n = solver.active_count().max(1);
        println!(
            "{:>8.1} {:>10} {:>12.3} {:>14.1} {:>10}",
            edge,
            n,
            median_us / 1000.0,
            median_us / (n as f32 / 1000.0),
            dispatches
        );

        let denom = samples.len().max(1) as f32;
        let mut breakdown: Vec<(String, f32)> =
            agg.into_iter().map(|(k, v)| (k, v / denom)).collect();
        breakdown.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f32 = breakdown.iter().map(|(_, us)| us).sum();
        for (label, us) in breakdown.iter().take(8) {
            println!(
                "           {:<18} {:>9.1} µs  ({:>4.1}%)",
                label,
                us,
                100.0 * us / total.max(1.0e-6)
            );
        }
    }
}

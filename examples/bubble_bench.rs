//! TEMPORARY armed-pocket probe for the bubble_fine reduction work (not committed).
//! Full-cross-section water slab over trapped air (the guaranteed enclosed-pocket state from
//! tests/twofield_cavity.rs::u3_rerun_assembled_operator_mid_jet_and_post_stop), scaled to the
//! gate-scene grid, so bubble_fine runs ARMED every sweep. Prints median frame ms + per-pass µs.

use coffee_sim::engine::scene::{Scene, SeedRegion, Species};
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("bubble_bench: no GPU adapter");
        return;
    };
    let edge = 61.0f32;
    let scene = Scene {
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0; 3],
        box_max: [edge, 40.0, edge],
        pour_water_ml: 0.0,
        solids: Vec::new(),
        regions: vec![SeedRegion {
            // Full-cross-section slab sealed to the walls: the air below is enclosed.
            min: [0.5, 8.5, 0.5],
            max: [edge - 0.4, 20.6, edge - 0.4],
            species: Species::Water,
        }],
        ..Scene::default()
    };
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let quiet = EmissionInput::default();
    for _ in 0..5 {
        solver.step(DT, &quiet);
    }
    println!(
        "bubble_bench: N={}, pocket_present={}",
        solver.active_count(),
        solver.pocket_present()
    );
    let mut samples: Vec<f32> = Vec::new();
    let mut agg: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let mut frames = 0u32;
    for _ in 0..30 {
        solver.step(DT, &quiet);
        solver.sample_diagnostics();
        let prof = solver.profile();
        let sub = prof.total_micros();
        if sub > 0.0 {
            samples.push(sub);
            frames += 1;
            for (label, us) in &prof.passes {
                *agg.entry(label.clone()).or_insert(0.0) += us;
            }
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples.get(samples.len() / 2).copied().unwrap_or(0.0);
    let mut rows: Vec<(String, f32)> = agg
        .into_iter()
        .map(|(k, v)| (k, v / frames.max(1) as f32))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f32 = rows.iter().map(|(_, us)| us).sum();
    println!(
        "bubble_bench: pocket_present={} median {:.3} ms/frame over {} frames",
        solver.pocket_present(),
        median / 1000.0,
        frames
    );
    for (label, us) in rows.iter().take(12) {
        println!(
            "  {:<20} {:>9.1} µs ({:>4.1}%)",
            label,
            us,
            100.0 * us / total.max(1.0e-6)
        );
    }
}

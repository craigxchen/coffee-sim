//! Run the MAC WebGPU backend and print actual GPU conservation diagnostics.
//! Run with `cargo run --release --example mac_gpu_pourover` (GPU required).

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::{
    scene::{SeedRegion, Species},
    Scene,
};
use coffee_sim::models::Materials;
use coffee_sim::solvers::{base::Solver, mac_gpu::MacGpuSolver};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

fn main() {
    let scene = Scene {
        box_max: [6.0, 8.0, 4.0],
        gravity: [0.0, -4.0, 0.0],
        regions: vec![SeedRegion {
            min: [1.0, 0.0, 1.0],
            max: [5.0, 1.5, 3.0],
            species: Species::Grain,
        }],
        ..Scene::default()
    };
    let mats = Materials {
        particle_spacing: 0.25,
        particle_mass: 0.25f32.powi(3),
        grain_diameter: 0.5,
        grain_mass: 0.1,
        ..Materials::default()
    };
    let mut cfg = Config {
        absorb_rate: 0.5,
        extract_rate: 1.0,
        seed_jitter: 0.0,
        drag_scale: 0.005,
        ..Config::default()
    };
    cfg.mac.cell_size = 0.5;
    cfg.mac.open_bottom = true;
    cfg.mac.capillary_pressure = 0.02;
    cfg.nozzle_radius = 0.3;
    let gpu = GpuContext::new_headless().expect("GPU required");
    let mut sim = MacGpuSolver::build(&scene, &mats, &cfg, &gpu);
    let start = std::time::Instant::now();
    for frame in 0..360 {
        sim.step(
            1.0 / 60.0,
            &EmissionInput {
                kettle_pos: [3.0, 5.0, 2.0],
                flow_rate: if frame < 180 { 0.6 } else { 0.0 },
                ..EmissionInput::default()
            },
        );
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        if frame % 60 == 59 {
            let d = sim.diagnostics();
            println!("t={:.1}s water: free={:.6} absorbed={:.6} drained={:.6}; solute={:.6}; errors water={:.2e} solute={:.2e}; pressure residual={:.2e}",
                d.time, d.free_water, d.absorbed_water, d.drained_water,
                d.dissolved_solute + d.retained_solute + d.drained_solute,
                d.water_error, d.solute_error, d.pressure_residual);
        }
    }
    let d = sim.diagnostics();
    assert!(d.water_error.abs() < 1e-4 && d.solute_error.abs() < 1e-4);
    assert!(
        d.absorbed_water > 0.0 && d.dissolved_solute + d.retained_solute + d.drained_solute > 0.0
    );
    println!(
        "WebGPU: {:.2}s wall time for {:.1}s simulation; {} active particles.",
        start.elapsed().as_secs_f64(),
        d.time,
        sim.diagnostics().particle_count
    );
}

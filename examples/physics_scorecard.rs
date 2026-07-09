//! Quick headless physics scorecard across the active solver families.
//!
//! This is an observation harness, not a gate: it prints the same small metric table after a short
//! run so calibration changes are easy to compare. XPBD and two-field run the V60 pour scene; PB-MPM
//! is still water-only, so it runs the high-velocity cup-jet impact scene honestly instead.
//!
//! Run: `cargo run --release --example physics_scorecard`
//! Tunables: `FRAMES=`, `SPACING=`, `FLOW_ML_S=`.

use coffee_sim::emission::{EmissionInput, PourEvent};
use coffee_sim::engine::state::Metrics;
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::PbmpmSolver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;
const ML_PER_SIM_UNIT3: f32 = 5.20;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn v60_mats(spacing: f32) -> Materials {
    Materials {
        particle_spacing: spacing,
        support_radius: 2.0 * spacing,
        grain_diameter: 2.0 * spacing,
        grain_mass: 10.0,
        c_max: 0.3,
        fines_fraction: 0.05,
        ..Materials::default()
    }
}

fn xpbd_v60_cfg() -> Config {
    Config {
        absorb_rate: 0.5,
        extract_rate: 1.0,
        nozzle_radius: 0.25,
        max_speed: 25.0,
        substeps: 2,
        drag_beta_max: 0.92,
        drag_subiters: 6,
        impact_scale: 4.0,
        fines_rate: 0.5,
        ..Config::default()
    }
}

fn twofield_v60_cfg() -> Config {
    Config {
        solid_dynamics: true,
        tf_absorb_rate: 0.15,
        tf_wet_cohesion: 4.0,
        tf_filter_floor: true,
        nozzle_radius: 0.55,
        max_speed: 12.0,
        tf_flip_c_surface: 0.1,
        tf_flip_density_gate: 0.5,
        tf_flip_div_scale: 1.0,
        tf_flip_water_splash_cap: 20.0,
        ..Config::default()
    }
}

fn center_pour(flow_ml_s: f32) -> EmissionInput {
    EmissionInput {
        kettle_pos: [0.0, 2.5, 0.0],
        flow_rate: flow_ml_s / ML_PER_SIM_UNIT3,
        pour_angle: 0.0,
        event: PourEvent::None,
    }
}

fn jet_impact_pour() -> EmissionInput {
    EmissionInput {
        kettle_pos: [0.0, 2.5, 0.0],
        flow_rate: 30.0,
        pour_angle: 0.0,
        event: PourEvent::None,
    }
}

struct Row {
    solver: &'static str,
    scene: &'static str,
    metrics: Metrics,
    dispatches: u32,
    gpu_us: f32,
}

fn print_row(row: &Row) {
    println!(
        "{:<10} {:<13} {:>7} {:>8} {:>9.1} {:>8.3} {:>8.3} {:>8.3} {:>8.2}",
        row.solver,
        row.scene,
        row.metrics.particle_count,
        row.dispatches,
        row.gpu_us,
        100.0 * row.metrics.extraction_yield,
        100.0 * row.metrics.tds,
        row.metrics.evenness,
        row.metrics.drawdown_time,
    );
}

fn run_xpbd(gpu: &GpuContext, frames: u32, spacing: f32, flow_ml_s: f32) -> Row {
    let mats = v60_mats(spacing);
    let cfg = xpbd_v60_cfg();
    let mut solver = XpbdSolver::build(&Scene::v60_pour(), &mats, &cfg, gpu);
    let input = center_pour(flow_ml_s);
    for _ in 0..frames {
        solver.step(DT, &input);
    }
    solver.sample_diagnostics();
    let profile = solver.profile();
    Row {
        solver: "xpbd",
        scene: "v60-pour",
        metrics: solver.metrics(),
        dispatches: profile.dispatches_per_frame,
        gpu_us: profile.total_micros(),
    }
}

fn run_twofield(gpu: &GpuContext, frames: u32, spacing: f32, flow_ml_s: f32) -> Row {
    let mats = v60_mats(spacing);
    let cfg = twofield_v60_cfg();
    let mut solver = TwofieldSolver::build(&Scene::v60_pour(), &mats, &cfg, gpu);
    let input = center_pour(flow_ml_s);
    for _ in 0..frames {
        solver.step(DT, &input);
    }
    solver.sample_diagnostics();
    let profile = solver.profile();
    Row {
        solver: "twofield",
        scene: "v60-pour",
        metrics: solver.metrics(),
        dispatches: profile.dispatches_per_frame,
        gpu_us: profile.total_micros(),
    }
}

fn run_pbmpm(gpu: &GpuContext, frames: u32, spacing: f32) -> Row {
    let mats = Materials {
        particle_spacing: spacing,
        support_radius: 2.0 * spacing,
        ..Materials::default()
    };
    let cfg = Config {
        nozzle_radius: 0.25,
        max_speed: 25.0,
        ..Config::default()
    };
    let mut solver = PbmpmSolver::build(&Scene::debug_high_velocity_jet_impact(), &mats, &cfg, gpu);
    let input = jet_impact_pour();
    for _ in 0..frames {
        solver.step(DT, &input);
    }
    solver.sample_diagnostics();
    let profile = solver.profile();
    Row {
        solver: "pbmpm",
        scene: "cup-jet",
        metrics: solver.metrics(),
        dispatches: profile.dispatches_per_frame,
        gpu_us: profile.total_micros(),
    }
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("physics_scorecard: no GPU adapter; cannot run.");
        return;
    };
    let frames = env_u32("FRAMES", 180);
    let spacing = env_f32("SPACING", 0.32).max(0.08);
    let flow_ml_s = env_f32("FLOW_ML_S", 5.0);

    println!("physics scorecard after {frames} frames (spacing={spacing}, flow={flow_ml_s} mL/s)");
    println!(
        "{:<10} {:<13} {:>7} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "solver", "scene", "active", "dispatch", "gpu_us", "yield%", "tds%", "even", "drawdown"
    );
    let rows = [
        run_xpbd(&gpu, frames, spacing, flow_ml_s),
        run_twofield(&gpu, frames, spacing, flow_ml_s),
        run_pbmpm(&gpu, frames, spacing),
    ];
    for row in &rows {
        print_row(row);
    }
}

//! V60 brew (headless): the full pour-over with extraction + thermal on. Drives `Scene::v60()`
//! with the calibrated permeable-bed materials, prints yield / TDS / mean temperature over the
//! brew, and tracks the solute inventory `Σ(grain s_f+s_s) + Σ(active water c·f_w·V_w)` so the
//! conservation behavior (and the wetting↔concentration coupling) is visible.
//!
//! Run: `cargo run --release --example brew`
//! Tunables via env: `EXTRACT=<rate>` `ABSORB=<rate>` `STEPS=<n>`.

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Solute inventory = grain pools + active-water dissolved solute (the conserved quantity).
fn inventory(chem: &[[f32; 4]], moisture: &[f32], phase: &[u32], v_w: f32, roundoff: f32) -> f32 {
    chem.iter()
        .zip(moisture)
        .zip(phase)
        .map(|((c, &m), &ph)| {
            if ph == 1 {
                c[0] + c[1]
            } else if m > roundoff {
                c[0] * m * v_w
            } else {
                0.0
            }
        })
        .sum()
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("brew: no GPU adapter; cannot run.");
        return;
    };
    // Calibrated permeable V60 bed (SDF phase) + extraction/thermal on.
    let mats = Materials {
        particle_spacing: 0.5,
        support_radius: 1.0,
        grain_diameter: 1.0,
        min_pore_fraction: 0.35,
        grain_mass: 10.0,
        ..Materials::default()
    };
    let cfg = Config {
        absorb_rate: env_f32("ABSORB", 0.5),
        extract_rate: env_f32("EXTRACT", 1.0),
        ..Config::default()
    };
    let steps: u32 = std::env::var("STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(900);

    let scene = Scene::v60();
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();
    let n_water = phase.iter().filter(|&&p| p == 0).count();
    let n_grain = phase.iter().filter(|&&p| p == 1).count();
    let inv0 = inventory(
        &solver.read_chem(),
        &solver.read_moisture(),
        &phase,
        v_w,
        cfg.absorb_roundoff,
    );

    println!(
        "V60 brew: {n_water} water + {n_grain} grain | absorb={} extract={} | solute0={inv0:.4}",
        cfg.absorb_rate, cfg.extract_rate
    );
    println!(
        "  (yield = dissolved / total dry mass; absolute-band calibration deferred — see U7 notes)"
    );
    // `drift%` is the wetting absorption sink (solute carried into the grounds), not a numerical leak.
    println!("  step    t(s)   yield%    TDS%   meanT   inventory   drift%");
    let input = EmissionInput::default();
    let sample_every = (steps / 15).max(1);
    for step in 1..=steps {
        solver.step(1.0 / 60.0, &input);
        if step % sample_every == 0 || step == steps {
            solver.sample_diagnostics();
            let m = solver.metrics();
            let temp = solver.read_temperature();
            let mean_t: f32 = temp.iter().sum::<f32>() / temp.len() as f32;
            let inv = inventory(
                &solver.read_chem(),
                &solver.read_moisture(),
                &phase,
                v_w,
                cfg.absorb_roundoff,
            );
            let drift = 100.0 * (inv - inv0) / inv0.max(1.0e-9);
            println!(
                "  {step:>5}  {:>5.2}  {:>6.2}  {:>6.3}  {:>6.3}  {inv:>9.4}  {drift:>6.2}",
                step as f32 / 60.0,
                100.0 * m.extraction_yield,
                100.0 * m.tds,
                mean_t,
            );
        }
    }
}

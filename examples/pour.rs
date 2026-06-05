//! V60 continuous-pour brew (headless): water is injected over the brew from a pour recipe, threads
//! the grain bed, drains through the cone apex, and pools in the cup — the pour-through path the
//! fixed-column `brew` example cannot model. Prints yield / TDS / mean temperature, the emitted vs
//! in-domain+absorbed water balance (conservation under a source), and pool occupancy.
//!
//! Run: `cargo run --release --example pour`  (env: `STEPS=`, `FLOW=` mL/s, `EXTRACT=`, `ABSORB=`).

use coffee_sim::emission::pour::{PourCommand, PourPattern, PourScript};
use coffee_sim::emission::{EmissionInput, PourEvent};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

/// mL per sim-unit³ (KEEP.md §27) — converts a recipe's mL/s into the solver's volumetric flow_rate.
const ML_PER_SIM_UNIT3: f32 = 5.20;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Map a recipe sample at time `t` to the solver's per-frame pour input. Normalized `(x,z)` ∈ [-1,1]
/// scales by the bed radius; flow mL/s converts to sim-volume/s.
fn pour_input(script: &PourScript, t: f32, bed_radius: f32, kettle_y: f32) -> EmissionInput {
    let (nx, nz, flow_ml) = script.sample(t);
    EmissionInput {
        kettle_pos: [nx * bed_radius, kettle_y, nz * bed_radius],
        flow_rate: flow_ml / ML_PER_SIM_UNIT3,
        pour_angle: 0.0,
        event: PourEvent::None,
    }
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pour: no GPU adapter; cannot run.");
        return;
    };
    let mats = Materials {
        particle_spacing: 0.5,
        support_radius: 1.0,
        grain_diameter: 1.0,
        water_grain_distance: 0.35,
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
    let flow = env_f32("FLOW", 8.0);

    // A short pour recipe: bloom at center, then a spiral main pour, then drawdown.
    let script = PourScript {
        commands: vec![
            PourCommand {
                t_start: 0.0,
                t_end: 2.0,
                flow_rate: flow,
                pattern: PourPattern::Center,
            },
            PourCommand {
                t_start: 3.0,
                t_end: 10.0,
                flow_rate: flow,
                pattern: PourPattern::Spiral {
                    freq_hz: 0.6,
                    r_min: 0.1,
                    r_max: 0.7,
                },
            },
        ],
    };

    let scene = Scene::v60_pour();
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let v_w = solver.water_particle_volume();
    let rho_w = mats.particle_mass / v_w;

    println!(
        "V60 pour brew: pool capacity {} | absorb={} extract={} flow={flow} mL/s",
        solver.pool_capacity(),
        cfg.absorb_rate,
        cfg.extract_rate
    );
    println!("  step   t(s)  active  yield%   TDS%   meanT  | emitted  inDomain+absorbed  balErr%");

    let dt = 1.0 / 60.0;
    let sample_every = (steps / 15).max(1);
    for step in 1..=steps {
        let t = step as f32 * dt;
        solver.step(dt, &pour_input(&script, t, 2.0, 2.5));
        if step % sample_every == 0 || step == steps {
            solver.sample_diagnostics();
            let m = solver.metrics();
            let phase = solver.read_phases();
            let moisture = solver.read_moisture();
            let temp = solver.read_temperature();
            // Water-volume balance: emitted = in-domain water + absorbed-into-grains (conservation).
            let emitted_vol = solver.total_emitted_water_mass() / rho_w;
            let mut water_vol = 0.0f32;
            let mut absorbed_vol = 0.0f32;
            for (&ph, &mw) in phase.iter().zip(&moisture) {
                if ph == 0 {
                    water_vol += mw * v_w; // f_w · V_w
                } else {
                    absorbed_vol += mw; // V_abs
                }
            }
            let in_domain = water_vol + absorbed_vol;
            let bal_err = 100.0 * (in_domain - emitted_vol) / emitted_vol.max(1.0e-9);
            let mean_t = temp.iter().sum::<f32>() / temp.len().max(1) as f32;
            println!(
                "  {step:>5} {:>5.2} {:>6}  {:>5.2}  {:>5.3}  {:>5.3}  | {emitted_vol:>7.3}  {in_domain:>15.3}  {bal_err:>6.2}",
                t,
                solver.active_count(),
                100.0 * m.extraction_yield,
                100.0 * m.tds,
                mean_t,
            );
        }
    }
}

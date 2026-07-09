//! Seam-blend M0 observation probe (docs/plans/2026-07-09-002 U2 — an observation harness,
//! NOT a gate): step the `seam-static-column` scene and print the water column's vertical
//! extent, kinetic energy, and the bed invariants over time. The column stands iff min-y
//! holds at/above the bed surface (−8.0) and KE settles.
//!
//! Run: `cargo run --release --example seam_probe`
//! Tunables: `FRAMES=`, `SPACING=`, `SAT=` (bed pre-saturation fraction, default 1.0).

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::seam::SeamSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;
const BED_TOP: f32 = -8.0;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_probe: no GPU adapter; skipping.");
        return;
    };
    let s = env_f32("SPACING", 0.32);
    let frames = env_f32("FRAMES", 300.0) as u32;
    let sat = env_f32("SAT", 1.0);
    let mats = Materials {
        particle_spacing: s,
        support_radius: 2.0 * s,
        grain_diameter: 2.0 * s,
        grain_mass: 10.0,
        ..Materials::default()
    };
    let scene = Scene::debug_seam_static_column();
    let cfg = Config {
        pbmpm_flip_fraction: env_f32("FLIP", Config::default().pbmpm_flip_fraction),
        ..Config::default()
    };
    let mut seam = SeamSolver::build(&scene, &mats, &cfg, &gpu);
    seam.prewet_bed(sat);

    println!(
        "seam M0 static column: {} water + {} grains | spacing {s} | bed top {BED_TOP} | sat {sat}",
        seam.water_count(),
        seam.solid_count()
    );
    println!("frame   minY    meanY   maxY    KE/n      belowBed%  bedWater");
    let mut snapshots: Vec<(u32, Vec<[f32; 4]>)> = Vec::new();
    for frame in 0..frames {
        seam.step(DT, &EmissionInput::default());
        if frame % 20 == 0 || frame + 1 == frames {
            let live = seam.water_count() as usize;
            let pos = seam.water_solver().read_positions();
            let vel = seam.water_solver().read_velocities();
            snapshots.push((frame, pos.clone()));
            let (mut min_y, mut max_y, mut sum_y) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
            let mut ke = 0.0f64;
            let mut below = 0u32;
            for i in 0..live {
                let y = pos[i][1];
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                sum_y += y as f64;
                let v = vel[i];
                ke += 0.5 * ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64);
                if y < BED_TOP - mats.particle_spacing {
                    below += 1;
                }
            }
            // Vertical histogram over the bed band [-10, -7): 12 quarter-unit bins — shows
            // WHERE water sits (a plateau at the surface skin = blocked; uniform fill = leak).
            let mut bins = [0u32; 12];
            for i in 0..live {
                let y = pos[i][1];
                if (-10.0..-7.0).contains(&y) {
                    bins[(((y + 10.0) * 4.0) as usize).min(11)] += 1;
                }
            }
            println!(
                "{frame:>5} {min_y:7.3} {:8.3} {max_y:7.3} {:9.4} {:9.2} {:>8}  |{}",
                sum_y / live.max(1) as f64,
                ke / live.max(1) as f64,
                100.0 * below as f32 / live.max(1) as f32,
                seam.bed_water_count(),
                bins.map(|b| format!("{b:>5}")).join("")
            );
        }
    }

    // Leaker forensics: particles ending below the block band — where did they cross?
    // (pbmpm never reorders, so indices are stable across snapshots.)
    let last = &snapshots.last().unwrap().1;
    let live = seam.water_count() as usize;
    let leakers: Vec<usize> = (0..live).filter(|&i| last[i][1] < -8.6).collect();
    println!(
        "\nleakers below -8.6: {} of {live}. Trajectories of the first 4 (y | wall_dist):",
        leakers.len()
    );
    for &i in leakers.iter().take(4) {
        let mut line = format!("  p{i}:");
        for (frame, snap) in snapshots.iter().step_by(6) {
            let p = snap[i];
            let wall = (7.0 - p[0].abs()).min(7.0 - p[2].abs());
            line += &format!(" [{frame}] {:.2}|{:.2}", p[1], wall);
        }
        println!("{line}");
    }
}

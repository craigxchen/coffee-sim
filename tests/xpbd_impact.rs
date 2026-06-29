//! Water→grain dynamic-pressure impact coupling (the pour crater). GPU-gated; skips without an adapter.

use coffee_sim::emission::PourEvent;
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

fn pour(kettle: [f32; 3], flow: f32) -> EmissionInput {
    EmissionInput {
        kettle_pos: kettle,
        flow_rate: flow,
        pour_angle: 0.0,
        event: PourEvent::None,
    }
}

fn web_coffee_cfg(impact_scale: f32) -> Config {
    Config {
        absorb_rate: 0.5,
        extract_rate: 1.0,
        nozzle_radius: 0.25,
        max_speed: 25.0,
        substeps: 2,
        drag_beta_max: 0.92,
        drag_subiters: 6,
        impact_scale,
        ..Config::default()
    }
}

fn coffee_mats() -> Materials {
    let r = 0.16_f32;
    Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        grain_diameter: 2.0 * r,
        grain_mass: 10.0,
        ..Materials::default()
    }
}

// 85th-percentile grain-surface height in a radial ring [lo,hi).
fn grain_surface(pos: &[[f32; 4]], phase: &[u32], lo: f32, hi: f32) -> f32 {
    let mut v: Vec<f32> = pos
        .iter()
        .zip(phase)
        .filter(|(p, &ph)| {
            let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
            ph == 1 && rr >= lo && rr < hi
        })
        .map(|(p, _)| p[1])
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else {
        v[((v.len() as f32) * 0.85) as usize]
    }
}

#[test]
fn zzz_impact_calibration_probe() {
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    let mats = coffee_mats();
    eprintln!("(crater = center sinks below rim and below the impact=0 baseline)");
    for impact_scale in [0.0f32, 1.0, 4.0, 12.0, 30.0] {
        let cfg = web_coffee_cfg(impact_scale);
        let mut solver = XpbdSolver::build(&Scene::v60_pour(), &mats, &cfg, &gpu);
        for _ in 0..60 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], 0.0)); // settle dry bed
        }
        let c0 = grain_surface(&solver.read_positions(), &solver.read_phases(), 0.0, 0.6);
        let mut vy_min = 0.0f32; // most-negative central grain vy during contact
        for step in 0..260 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], 3.26));
            if (20..140).contains(&step) {
                let pos = solver.read_positions();
                let vel = solver.read_velocities();
                let phase = solver.read_phases();
                let (mut s, mut n) = (0.0f64, 0u32);
                for ((p, v), &ph) in pos.iter().zip(&vel).zip(&phase) {
                    let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
                    if ph == 1 && rr < 0.6 {
                        s += v[1] as f64;
                        n += 1;
                    }
                }
                vy_min = vy_min.min((s / n.max(1) as f64) as f32);
            }
        }
        let pos = solver.read_positions();
        let phase = solver.read_phases();
        let c = grain_surface(&pos, &phase, 0.0, 0.6);
        let rim = grain_surface(&pos, &phase, 1.2, 2.0);
        assert!(c.is_finite() && rim.is_finite(), "non-finite surface");
        eprintln!(
            "impact_scale={impact_scale:>5.1} | centerΔ={:+.2} rim={:+.2} center-rim={:+.2} | min grain vy@center={vy_min:.3}",
            c - c0,
            rim,
            c - rim
        );
    }
}

#[test]
fn zzz_impact_creep_probe() {
    // Static saturated bed, NO pour: does the impact term make grains creep?
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    let mats = coffee_mats();
    for impact_scale in [0.0f32, 12.0, 30.0] {
        let cfg = web_coffee_cfg(impact_scale);
        let mut solver = XpbdSolver::build(&Scene::v60_pour(), &mats, &cfg, &gpu);
        // wet the bed with a short pour, then stop and let it settle saturated
        for _ in 0..120 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], 3.26));
        }
        let p0 = solver.read_positions();
        let ph = solver.read_phases();
        for _ in 0..200 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], 0.0)); // no pour
        }
        let p1 = solver.read_positions();
        // mean grain displacement over the static phase + max grain speed at the end
        let (mut disp, mut n) = (0.0f64, 0u32);
        for ((a, b), &k) in p0.iter().zip(&p1).zip(&ph) {
            if k != 1 {
                continue;
            }
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            disp += d as f64;
            n += 1;
        }
        let vel = solver.read_velocities();
        let mut vmax = 0.0f32;
        for (v, &k) in vel.iter().zip(&ph) {
            if k == 1 {
                vmax = vmax.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
            }
        }
        eprintln!(
            "impact_scale={impact_scale:>5.1} | mean grain drift over 200 static steps={:.3} | max grain speed={vmax:.3}",
            disp / n.max(1) as f64
        );
    }
}

#[test]
fn zzz_water_dimple_probe() {
    // Does the pour dent the WATER pool surface (a crater/dimple at the impact point)?
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    let r = 0.16_f32;
    let mats = Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    };
    // free-surface height = 90th-pct water y in a radial ring
    let surf = |pos: &[[f32; 4]], lo: f32, hi: f32| -> f32 {
        let mut v: Vec<f32> = pos
            .iter()
            .filter(|p| {
                let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
                rr >= lo && rr < hi
            })
            .map(|p| p[1])
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            f32::NAN
        } else {
            v[((v.len() as f32) * 0.90) as usize]
        }
    };
    for flow in [3.26f32, 8.0, 14.0] {
        let cfg = Config {
            nozzle_radius: 0.25,
            max_speed: 25.0,
            xsph_viscosity_c: 0.02,
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&Scene::v60_pour_water_only(), &mats, &cfg, &gpu);
        // fill the cup first, then keep pouring and sample the steady surface profile
        for _ in 0..220 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], flow));
        }
        let (mut c, mut m, mut g) = (0.0f64, 0.0f64, 0.0f64);
        let mut k = 0u32;
        for _ in 0..40 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], flow));
            let pos = solver.read_positions();
            c += surf(&pos, 0.0, 0.4) as f64;
            m += surf(&pos, 0.4, 1.0) as f64;
            g += surf(&pos, 1.0, 1.8) as f64;
            k += 1;
        }
        let (c, m, g) = (c / k as f64, m / k as f64, g / k as f64);
        eprintln!("flow={flow:>5.1} | surface y: center(r<0.4)={c:.2} mid={m:.2} ring(1.0-1.8)={g:.2} | dimple(center-ring)={:+.3}", c-g);
    }
}

#[test]
fn zzz_water_cavity_probe() {
    // Pool cavity at the impact point, EXCLUDING the falling stream (fast particles).
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    let r = 0.16_f32;
    let mats = Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    };
    // pool surface (90th-pct y) using only SLOW water (|v|<3) so the stream column is excluded
    let pool_surf = |pos: &[[f32; 4]], vel: &[[f32; 4]], lo: f32, hi: f32| -> f32 {
        let mut v: Vec<f32> = pos
            .iter()
            .zip(vel)
            .filter(|(p, vv)| {
                let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
                let sp = (vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2]).sqrt();
                rr >= lo && rr < hi && sp < 3.0
            })
            .map(|(p, _)| p[1])
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            f32::NAN
        } else {
            v[((v.len() as f32) * 0.90) as usize]
        }
    };
    for flow in [3.26f32, 8.0, 14.0] {
        let cfg = Config {
            nozzle_radius: 0.25,
            max_speed: 25.0,
            xsph_viscosity_c: 0.02,
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&Scene::v60_pour_water_only(), &mats, &cfg, &gpu);
        for _ in 0..220 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], flow));
        }
        let (mut c, mut g) = (0.0f64, 0.0f64);
        let mut k = 0u32;
        for _ in 0..40 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], flow));
            let pos = solver.read_positions();
            let vel = solver.read_velocities();
            c += pool_surf(&pos, &vel, 0.0, 0.45) as f64;
            g += pool_surf(&pos, &vel, 0.9, 1.7) as f64;
            k += 1;
        }
        let (c, g) = (c / k as f64, g / k as f64);
        eprintln!("flow={flow:>5.1} | POOL surface: impact-center={c:.2} ring={g:.2} | cavity(center-ring)={:+.3}", c-g);
    }
}

#[test]
fn zzz_water_cavity_impact_probe() {
    // Does the new water↔water impact term punch a pool cavity? And is it stable?
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    let r = 0.16_f32;
    let mats = Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    };
    let pool_surf = |pos: &[[f32; 4]], vel: &[[f32; 4]], lo: f32, hi: f32| -> f32 {
        let mut v: Vec<f32> = pos
            .iter()
            .zip(vel)
            .filter(|(p, vv)| {
                let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
                let sp = (vv[0] * vv[0] + vv[1] * vv[1] + vv[2] * vv[2]).sqrt();
                rr >= lo && rr < hi && sp < 3.0
            })
            .map(|(p, _)| p[1])
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.is_empty() {
            f32::NAN
        } else {
            v[((v.len() as f32) * 0.90) as usize]
        }
    };
    for impact_scale in [0.0f32, 0.5, 1.0, 2.0, 4.0] {
        let cfg = Config {
            nozzle_radius: 0.25,
            max_speed: 25.0,
            xsph_viscosity_c: 0.02,
            impact_scale,
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&Scene::v60_pour_water_only(), &mats, &cfg, &gpu);
        for _ in 0..220 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], 3.26));
        }
        let (mut c, mut g) = (0.0f64, 0.0f64);
        let mut k = 0u32;
        let mut vmax = 0.0f32;
        for _ in 0..40 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], 3.26));
            let pos = solver.read_positions();
            let vel = solver.read_velocities();
            c += pool_surf(&pos, &vel, 0.0, 0.45) as f64;
            g += pool_surf(&pos, &vel, 0.9, 1.7) as f64;
            k += 1;
            for v in &vel {
                vmax = vmax.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
            }
        }
        let (c, g) = (c / k as f64, g / k as f64);
        eprintln!("impact_scale={impact_scale:>4.1} | POOL center={c:.2} ring={g:.2} cavity(center-ring)={:+.3} | max water speed={vmax:.1}", c-g);
    }
}

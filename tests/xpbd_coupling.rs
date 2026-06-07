//! Water/bed coupling invariants (GPU-gated; skips without an adapter).
//!
//! Step 1 (grain exclusion + pore-modulated water density; drag/buoyancy come later): a water
//! column poured onto a grain bed. The invariant is that water threads the **pores** without
//! passing through grain **bodies** (no tunneling), the bed isn't crushed, and nothing blows up.
//! Water percolating to the floor is expected — drag (step 2) is what resists it into a realistic
//! drawdown; here there's no resistance, so water simply drains through the pore network.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::permeability::darcy_drag_rate;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

fn speed(v: [f32; 4]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn finite(p: &[[f32; 4]]) -> bool {
    p.iter()
        .all(|q| q[0].is_finite() && q[1].is_finite() && q[2].is_finite())
}

fn dist(a: [f32; 4], b: [f32; 4]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn momentum(vel: &[[f32; 4]], phase: &[u32], mats: &Materials) -> [f32; 3] {
    vel.iter().zip(phase).fold([0.0; 3], |mut acc, (v, &ph)| {
        let m = if ph == 1 {
            mats.grain_mass
        } else {
            mats.particle_mass
        };
        acc[0] += m * v[0];
        acc[1] += m * v[1];
        acc[2] += m * v[2];
        acc
    })
}

fn kinetic_energy(vel: &[[f32; 4]], phase: &[u32], mats: &Materials) -> f32 {
    vel.iter()
        .zip(phase)
        .map(|(v, &ph)| {
            let m = if ph == 1 {
                mats.grain_mass
            } else {
                mats.particle_mass
            };
            0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        })
        .sum()
}

fn pnorm(p: [f32; 3]) -> f32 {
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

fn coupling_probe_config() -> Config {
    Config {
        max_iters: 1,
        min_iters: 1,
        bed_max_iters: 0,
        xsph_viscosity_c: 0.0,
        exclusion_relax: 0.0,
        drag_subiters: 1,
        buoyancy_scale: 0.0,
        seed_jitter: 0.0,
        ..Config::default()
    }
}

fn embedded_bed_scene() -> Scene {
    Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [2.0, 2.0, 2.0],
                max: [6.0, 6.0, 6.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [2.1, 2.1, 2.1],
                max: [5.9, 5.9, 5.9],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    }
}

fn variance(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len().max(1) as f32
}

fn water_values(values: &[f32], phase: &[u32]) -> Vec<f32> {
    values
        .iter()
        .zip(phase)
        .filter_map(|(&v, &ph)| (ph == 0).then_some(v))
        .collect()
}

fn point_scene(waters: &[[f32; 3]], grains: &[[f32; 3]], box_max: [f32; 3]) -> Scene {
    let regions = waters
        .iter()
        .map(|&p| SeedRegion {
            min: p,
            max: p,
            species: Species::Water,
        })
        .chain(grains.iter().map(|&p| SeedRegion {
            min: p,
            max: p,
            species: Species::Grain,
        }))
        .collect();
    Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max,
        regions,
        ..Scene::default()
    }
}

fn ring_points(n: usize, center: [f32; 3], radius: f32) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            [
                center[0] + radius * a.cos(),
                center[1],
                center[2] + radius * a.sin(),
            ]
        })
        .collect()
}

fn drag_only_config() -> Config {
    Config {
        max_iters: 0,
        bed_max_iters: 0,
        xsph_viscosity_c: 0.0,
        drag_scale: 0.02,
        drag_beta_max: 0.8,
        drag_subiters: 4,
        grain_sleep_speed: 0.0,
        ..Config::default()
    }
}

fn buoyancy_only_config(scale: f32) -> Config {
    Config {
        max_iters: 0,
        bed_max_iters: 0,
        xsph_viscosity_c: 0.0,
        drag_subiters: 0,
        buoyancy_scale: scale,
        grain_sleep_speed: 0.0,
        wake_threshold: 0.0,
        ..Config::default()
    }
}

#[test]
fn wide_coupling_radius_smooths_and_bounds_alpha_s_for_fine_water() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = embedded_bed_scene();
    let cfg = coupling_probe_config();
    let baseline_mats = Materials {
        particle_spacing: 0.45,
        support_radius: 0.9,
        grain_diameter: 1.0,
        coupling_radius: 0.9,
        ..Materials::default()
    };
    let wide_mats = Materials {
        coupling_radius: 2.5,
        ..baseline_mats.clone()
    };

    let mut baseline = XpbdSolver::build(&scene, &baseline_mats, &cfg, &gpu);
    baseline.step(0.0, &EmissionInput::default());
    let phase = baseline.read_phases();
    let baseline_alpha = water_values(&baseline.read_alpha_s(), &phase);

    let mut wide = XpbdSolver::build(&scene, &wide_mats, &cfg, &gpu);
    wide.step(0.0, &EmissionInput::default());
    let wide_alpha = water_values(&wide.read_alpha_s(), &wide.read_phases());

    assert_eq!(baseline_alpha.len(), wide_alpha.len());
    assert!(
        (wide.coupling_radius() - 2.5).abs() < 1.0e-6,
        "fine-water bed should use explicit h_c=2.5, got {}",
        wide.coupling_radius()
    );
    assert!(
        wide_alpha
            .iter()
            .all(|&a| (0.0..=cfg.packing_limit + 1.0e-6).contains(&a)),
        "wide h_c α_s exceeded [0, packing_limit]"
    );
    let baseline_var = variance(&baseline_alpha);
    let wide_var = variance(&wide_alpha);
    assert!(
        wide_var < baseline_var * 0.75,
        "wide h_c should smooth water-sampled α_s: baseline var {baseline_var:.6}, wide var {wide_var:.6}"
    );
}

#[test]
fn porosity_drag_factor_uses_dense_bed_epsilon_floor() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [4.0, 4.0, 4.0],
                max: [4.0, 4.0, 4.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [4.0, 4.0, 4.0],
                max: [4.0, 4.0, 4.0],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    };
    let cfg = Config {
        packing_limit: 0.95,
        fines_rate: 1.0,
        ..coupling_probe_config()
    };
    let mats = Materials {
        particle_spacing: 1.0,
        support_radius: 1.0,
        grain_diameter: 1.0,
        coupling_radius: 1.0,
        ..Materials::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    solver.step(0.0, &EmissionInput::default());

    let phase = solver.read_phases();
    let alpha = water_values(&solver.read_alpha_s(), &phase);
    assert!(
        alpha.iter().any(|&a| a > 0.65),
        "dense probe did not produce α_s above the ε floor trigger: max {:.3}",
        alpha.iter().copied().fold(0.0, f32::max)
    );

    let counts = solver.read_coupling_scale();
    let max_rate = counts.iter().map(|c| c[0]).fold(0.0, f32::max);
    let eps_floor = 0.35f32;
    let resolved = darcy_drag_rate(mats.grain_diameter, eps_floor, cfg.drag_scale);
    assert!(
        max_rate.is_finite() && max_rate <= resolved * 1.001,
        "ε floor should keep live porosity drag finite and capped: max_rate {max_rate}, floor cap {resolved}",
    );
}

#[test]
fn water_pbf_residual_is_unchanged_when_only_coupling_radius_differs() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::dam_break();
    let cfg = Config {
        seed_jitter: 0.0,
        xsph_viscosity_c: 0.0,
        ..Config::default()
    };
    let mats_h = Materials {
        coupling_radius: 0.0,
        ..Materials::default()
    };
    let mats_hc = Materials {
        coupling_radius: 6.0,
        ..Materials::default()
    };
    let mut baseline = XpbdSolver::build(&scene, &mats_h, &cfg, &gpu);
    let mut wide = XpbdSolver::build(&scene, &mats_hc, &cfg, &gpu);
    for _ in 0..3 {
        baseline.step(1.0 / 60.0, &EmissionInput::default());
        wide.step(1.0 / 60.0, &EmissionInput::default());
    }
    baseline.sample_diagnostics();
    wide.sample_diagnostics();
    let rb = baseline.diagnostics().residual;
    let rw = wide.diagnostics().residual;
    assert!(
        (rb - rw).abs() < 1.0e-5,
        "pure-water PBF residual should stay on h when h_c differs: h {rb:.8}, h_c {rw:.8}"
    );
}

#[test]
fn coupling_neighbor_count_diagnostics_are_readable() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = embedded_bed_scene();
    let cfg = coupling_probe_config();
    let mats = Materials {
        particle_spacing: 0.45,
        support_radius: 0.9,
        grain_diameter: 1.0,
        coupling_radius: 2.5,
        ..Materials::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    solver.step(0.0, &EmissionInput::default());
    solver.sample_diagnostics();
    let diag = solver.diagnostics();
    assert!(
        diag.coupling_neighbors_avg > 0.0
            && diag.coupling_neighbors_max >= diag.coupling_neighbors_avg,
        "expected readable avg/max opposite-phase counts, got avg {:.2}, max {:.2}",
        diag.coupling_neighbors_avg,
        diag.coupling_neighbors_max
    );
    assert!(
        diag.coupling_neighbors_max < 1024.0,
        "wide h_c neighbor count spike exceeded the U1 budget: max {:.0}",
        diag.coupling_neighbors_max
    );
}

#[test]
fn legacy_single_resolution_alpha_s_is_unchanged_when_hc_resolves_to_h() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [4.0, 4.0, 4.0],
                max: [4.0, 4.0, 4.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [5.0, 4.0, 4.0],
                max: [5.0, 4.0, 4.0],
                species: Species::Water,
            },
        ],
        ..Scene::dam_through_sand()
    };
    let cfg = coupling_probe_config();
    let default_mats = Materials::default();
    let explicit_mats = Materials {
        coupling_radius: default_mats.support_radius,
        ..default_mats.clone()
    };
    let mut derived = XpbdSolver::build(&scene, &default_mats, &cfg, &gpu);
    let mut explicit = XpbdSolver::build(&scene, &explicit_mats, &cfg, &gpu);
    assert_eq!(
        derived.coupling_radius().to_bits(),
        default_mats.support_radius.to_bits(),
        "legacy single-resolution scene should derive h_c=h"
    );
    derived.step(0.0, &EmissionInput::default());
    explicit.step(0.0, &EmissionInput::default());
    let a = derived.read_alpha_s();
    let b = explicit.read_alpha_s();
    assert_eq!(a.len(), b.len());
    let mut a_bits: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
    let mut b_bits: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
    a_bits.sort_unstable();
    b_bits.sort_unstable();
    for (idx, (&x, &y)) in a_bits.iter().zip(&b_bits).enumerate() {
        assert_eq!(
            x,
            y,
            "legacy h_c=h α_s byte-value multiset changed at sorted slot {idx}: {x:#010x} vs {y:#010x}"
        );
    }
}

fn low_water_mean(pos: &[[f32; 4]], phase: &[u32]) -> f32 {
    let mut ys: Vec<f32> = pos
        .iter()
        .zip(phase)
        .filter(|(_, &ph)| ph == 0)
        .map(|(p, _)| p[1])
        .collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = (ys.len() / 20).max(1);
    ys.iter().take(n).sum::<f32>() / n as f32
}

fn water_mean_y(pos: &[[f32; 4]], phase: &[u32]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (p, &ph) in pos.iter().zip(phase) {
        if ph == 0 {
            sum += p[1];
            n += 1;
        }
    }
    sum / n.max(1) as f32
}

fn grain_mean_y(pos: &[[f32; 4]], phase: &[u32]) -> f32 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for (p, &ph) in pos.iter().zip(phase) {
        if ph == 1 {
            sum += p[1];
            n += 1;
        }
    }
    sum / n.max(1) as f32
}

/// Closest approach between any sampled water particle and any sampled grain (strided for speed).
fn min_water_grain(pos: &[[f32; 4]], phase: &[u32]) -> f32 {
    let waters: Vec<[f32; 4]> = pos
        .iter()
        .zip(phase)
        .enumerate()
        .filter(|(i, (_, &p))| p == 0 && i % 8 == 0)
        .map(|(_, (p, _))| *p)
        .collect();
    let grains: Vec<[f32; 4]> = pos
        .iter()
        .zip(phase)
        .enumerate()
        .filter(|(i, (_, &p))| p == 1 && i % 4 == 0)
        .map(|(_, (p, _))| *p)
        .collect();
    let mut m = f32::INFINITY;
    for w in &waters {
        for g in &grains {
            m = m.min(dist(*w, *g));
        }
    }
    m
}

#[test]
fn buoyancy_conserves_momentum_and_lifts_grain() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [4.0, 5.0, 4.0],
                max: [4.0, 5.0, 4.0],
                species: Species::Water,
            },
            SeedRegion {
                min: [4.0, 4.0, 4.0],
                max: [4.0, 4.0, 4.0],
                species: Species::Grain,
            },
        ],
        ..Scene::default()
    };
    let mats = Materials {
        grain_mass: 2.0,
        grain_diameter: 0.5,
        ..Materials::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &buoyancy_only_config(1.0), &gpu);
    let phase = solver.read_phases();
    assert_eq!(phase, vec![0, 1]);
    solver.write_velocities_for_test(&[[0.0; 4], [0.0; 4]]);
    // PBF λ is negative under compression; pressure is -λ in the buoyancy kernels.
    solver.write_lambdas_for_test(&[-10.0, 0.0]);

    let p0 = momentum(&solver.read_velocities(), &phase, &mats);
    solver.step(1.0 / 60.0, &EmissionInput::default());
    let vel = solver.read_velocities();
    let p1 = momentum(&vel, &phase, &mats);
    let drift = pnorm([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]);
    assert!(drift < 1.0e-4, "buoyancy momentum drift: {drift}");
    assert!(vel[1][1] > 0.0, "grain was not lifted: vy={}", vel[1][1]);
    assert!(
        vel[0][1] < 0.0,
        "water did not receive the downward reaction: vy={}",
        vel[0][1]
    );
}

#[test]
fn buoyancy_pressure_proxy_lifts_fine_bed_without_blowup() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [12.0, 12.0, 12.0],
        regions: vec![
            SeedRegion {
                min: [4.0, 5.0, 4.0],
                max: [8.0, 7.0, 8.0],
                species: Species::Water,
            },
            SeedRegion {
                min: [4.0, 3.0, 4.0],
                max: [8.0, 4.0, 8.0],
                species: Species::Grain,
            },
        ],
        ..Scene::default()
    };
    let mats = Materials {
        grain_diameter: 0.5,
        grain_mass: 2.0,
        ..Materials::default()
    };
    let run = |scale: f32| {
        let mut solver = XpbdSolver::build(&scene, &mats, &buoyancy_only_config(scale), &gpu);
        let phase = solver.read_phases();
        let lambdas: Vec<f32> = phase
            .iter()
            .map(|&ph| if ph == 0 { -0.5 } else { 0.0 })
            .collect();
        solver.write_lambdas_for_test(&lambdas);
        let y0 = grain_mean_y(&solver.read_positions(), &phase);
        for _ in 0..4 {
            solver.step(1.0 / 60.0, &EmissionInput::default());
            solver.write_lambdas_for_test(&lambdas);
        }
        let pos = solver.read_positions();
        let vel = solver.read_velocities();
        (
            grain_mean_y(&pos, &phase) - y0,
            finite(&pos) && finite(&vel),
        )
    };

    let (off_rise, off_finite) = run(0.0);
    let (on_rise, on_finite) = run(1.0);
    assert!(
        off_finite && on_finite,
        "buoyancy pressure-proxy test blew up"
    );
    // Density-aware buoyancy (added in Phase 1.4) scales the lift by ρ_water/ρ_grain, so this very
    // dense grain (ρ≈30× water) lifts only a little — but it must still rise vs no buoyancy, finitely.
    assert!(
        on_rise > off_rise + 1.0e-4,
        "fine bed did not lift under seeded pressure proxy: on {on_rise:.6}, off {off_rise:.6}"
    );
}

#[test]
fn drag_conserves_momentum_and_damps_relative_velocity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [3.0, 4.0, 4.0],
                max: [3.0, 4.0, 4.0],
                species: Species::Water,
            },
            SeedRegion {
                min: [4.0, 4.0, 4.0],
                max: [4.0, 4.0, 4.0],
                species: Species::Grain,
            },
        ],
        ..Scene::default()
    };
    let mats = Materials {
        grain_mass: 2.0,
        grain_diameter: 0.5,
        ..Materials::default()
    };
    let cfg = Config {
        drag_scale: 5000.0,
        ..drag_only_config()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    assert_eq!(phase, vec![0, 1]);
    solver.write_velocities_for_test(&[[1.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.0]]);
    let input = EmissionInput::default();
    let p0 = momentum(&solver.read_velocities(), &phase, &mats);
    let mut prev_rel = 1.25f32;

    for step in 0..8 {
        solver.step(1.0 / 60.0, &input);
        let vel = solver.read_velocities();
        let p = momentum(&vel, &phase, &mats);
        let drift = pnorm([p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]]);
        assert!(drift < 2.0e-4, "momentum drift at step {step}: {drift}");

        let rel = vel[0][0] - vel[1][0];
        assert!(rel >= -1.0e-5, "relative velocity flipped sign: {rel}");
        assert!(
            rel <= prev_rel + 1.0e-5,
            "relative velocity did not decay: {rel} > {prev_rel}"
        );
        prev_rel = rel;
    }
    assert!(
        prev_rel < 0.2,
        "relative velocity did not damp enough: {prev_rel}"
    );
}

#[test]
fn darcy_drag_conserves_momentum_under_asymmetric_porosity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = point_scene(&[[3.0, 4.0, 4.0]], &[[4.0, 4.0, 4.0]], [8.0, 8.0, 8.0]);
    let mats = Materials {
        grain_mass: 2.0,
        grain_diameter: 1.0,
        coupling_radius: 2.0,
        ..Materials::default()
    };
    let cfg = Config {
        drag_scale: 2.0,
        drag_beta_max: 0.9,
        drag_subiters: 1,
        ..drag_only_config()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    assert_eq!(phase, vec![0, 1]);
    solver.write_velocities_for_test(&[[2.0, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]]);

    let p0 = momentum(&solver.read_velocities(), &phase, &mats);
    let ke0 = kinetic_energy(&solver.read_velocities(), &phase, &mats);
    solver.step(1.0 / 60.0, &EmissionInput::default());
    let vel = solver.read_velocities();
    let p1 = momentum(&vel, &phase, &mats);
    let ke1 = kinetic_energy(&vel, &phase, &mats);
    let drift = pnorm([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]);
    assert!(
        drift < 2.0e-4,
        "asymmetric-porosity drag momentum drift: {drift}"
    );
    assert!(ke1 <= ke0 + 1.0e-5, "drag increased KE: {ke0} -> {ke1}");

    let cs = solver.read_coupling_scale();
    assert!(
        (cs[0][0] - cs[1][0]).abs() > 1.0e-3,
        "test should exercise asymmetric local porosity, got beta_w {:.6}, beta_g {:.6}",
        cs[0][0],
        cs[1][0]
    );
}

#[test]
fn drag_neighbor_cap_prevents_neighbor_count_overdamping() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };

    let run = |water_count: usize| {
        let center = [5.0, 5.0, 5.0];
        let waters = ring_points(water_count, center, 0.9);
        let scene = point_scene(&waters, &[center], [10.0, 10.0, 10.0]);
        let mats = Materials {
            grain_mass: 3.0,
            grain_diameter: 1.0,
            coupling_radius: 2.5,
            ..Materials::default()
        };
        let cfg = Config {
            drag_scale: 50.0,
            drag_beta_max: 0.6,
            drag_subiters: 1,
            ..drag_only_config()
        };
        let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        let phase = solver.read_phases();
        let seeded: Vec<[f32; 4]> = phase
            .iter()
            .map(|&ph| {
                if ph == 0 {
                    [1.0, 0.0, 0.0, 0.0]
                } else {
                    [0.0; 4]
                }
            })
            .collect();
        solver.write_velocities_for_test(&seeded);
        solver.step(1.0 / 60.0, &EmissionInput::default());
        let grain = phase.iter().position(|&ph| ph == 1).unwrap();
        solver.read_velocities()[grain][0]
    };

    let few = run(4);
    let many = run(12);
    let rel = ((many - few) / few.max(1.0e-6)).abs();
    assert!(
        rel < 0.15,
        "aggregate drag should not grow with neighbor count: N=4 {few:.6}, N=12 {many:.6}"
    );
}

#[test]
fn drag_resolution_refinement_preserves_grain_impulse() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };

    let run = |spacing: f32, support_radius: f32, water_count: usize| {
        let center = [6.0, 6.0, 6.0];
        let waters = ring_points(water_count, center, 1.0);
        let scene = point_scene(&waters, &[center], [12.0, 12.0, 12.0]);
        let mats = Materials {
            particle_spacing: spacing,
            support_radius,
            grain_mass: 8.0,
            grain_diameter: 4.0,
            coupling_radius: 4.0,
            ..Materials::default()
        };
        let cfg = Config {
            drag_scale: 0.001,
            drag_beta_max: 0.95,
            drag_subiters: 1,
            ..drag_only_config()
        };
        let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        let phase = solver.read_phases();
        let seeded: Vec<[f32; 4]> = phase
            .iter()
            .map(|&ph| {
                if ph == 0 {
                    [1.0, 0.0, 0.0, 0.0]
                } else {
                    [0.0; 4]
                }
            })
            .collect();
        solver.write_velocities_for_test(&seeded);
        solver.step(1.0 / 60.0, &EmissionInput::default());
        let grain = phase.iter().position(|&ph| ph == 1).unwrap();
        solver.read_velocities()[grain][0]
    };

    let coarse = run(1.0, 2.0, 8);
    let fine = run(0.5, 1.0, 64);
    let ratio = fine / coarse.max(1.0e-8);
    assert!(
        (0.70..=1.30).contains(&ratio),
        "refining water particles should preserve grain drag impulse: coarse {coarse:.6}, fine {fine:.6}, ratio {ratio:.3}"
    );
}

#[test]
fn imposed_flow_mobilizes_grain_probe_and_weak_drag_does_not() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };

    let run = |drag_scale: f32| {
        let target = [5.0, 5.0, 5.0];
        let grains = vec![
            target,
            [4.2, 5.0, 5.0],
            [5.8, 5.0, 5.0],
            [5.0, 5.0, 4.2],
            [5.0, 5.0, 5.8],
            [4.2, 4.2, 5.0],
            [5.8, 4.2, 5.0],
            [5.0, 4.2, 4.2],
            [5.0, 4.2, 5.8],
        ];
        let mut waters = ring_points(8, [5.0, 5.8, 5.0], 0.7);
        waters.push([5.0, 6.0, 5.0]);
        let scene = point_scene(&waters, &grains, [10.0, 10.0, 10.0]);
        let mats = Materials {
            grain_mass: 1.5,
            grain_diameter: 1.0,
            coupling_radius: 2.5,
            ..Materials::default()
        };
        let cfg = Config {
            drag_scale,
            drag_beta_max: 0.85,
            drag_subiters: 4,
            wake_threshold: 0.01,
            grain_sleep_speed: 0.2,
            ..drag_only_config()
        };
        let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        let phase = solver.read_phases();
        let grain = phase.iter().position(|&ph| ph == 1).unwrap();
        let seeded: Vec<[f32; 4]> = phase
            .iter()
            .map(|&ph| {
                if ph == 0 {
                    [0.0, -4.0, 0.0, 0.0]
                } else {
                    [0.0; 4]
                }
            })
            .collect();
        solver.write_velocities_for_test(&seeded);
        let y0 = solver.read_positions()[grain][1];
        solver.step(1.0 / 60.0, &EmissionInput::default());
        let y1 = solver.read_positions()[grain][1];
        let fluid_impulse = solver.read_fluid_impulse()[grain];
        let normal_impulse = solver.read_normal_impulse()[grain];
        let resistance = mats.friction_mu * normal_impulse;
        ((y1 - y0).abs(), fluid_impulse, resistance)
    };

    let (moved, impulse, resistance) = run(Config::default().drag_scale);
    let (control_moved, control_impulse, _) = run(0.0);
    assert!(
        impulse > resistance + 1.0e-5,
        "drag impulse should exceed the measured yield budget: impulse {impulse:.6}, resistance {resistance:.6}"
    );
    assert!(
        moved > 1.0e-3,
        "strong imposed flow did not mobilize the grain probe: displacement {moved:.6}"
    );
    assert!(
        control_impulse <= 1.0e-7 && control_moved < moved * 0.1,
        "weak-drag control should stay nearly static: moved {control_moved:.6}, impulse {control_impulse:.6}, strong moved {moved:.6}"
    );
}

/// Fines active still runs through the same symmetric Darcy pair scale. This keeps the former
/// fines-gated path covered while ensuring it no longer diverges from the default coupling math.
#[test]
fn harmonic_drag_combiner_conserves_momentum() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [3.0, 4.0, 4.0],
                max: [3.0, 4.0, 4.0],
                species: Species::Water,
            },
            SeedRegion {
                min: [4.0, 4.0, 4.0],
                max: [4.0, 4.0, 4.0],
                species: Species::Grain,
            },
        ],
        ..Scene::default()
    };
    let mats = Materials {
        grain_mass: 2.0,
        grain_diameter: 0.5,
        ..Materials::default()
    };
    // fines_rate > 0 engages the harmonic combiner; fines_fraction = 0 keeps the transfer a no-op,
    // so this isolates the drag combiner.
    let cfg = Config {
        fines_rate: 2.0,
        ..drag_only_config()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    assert_eq!(phase, vec![0, 1]);
    solver.write_velocities_for_test(&[[1.0, 0.0, 0.0, 0.0], [-0.25, 0.0, 0.0, 0.0]]);
    let input = EmissionInput::default();
    let p0 = momentum(&solver.read_velocities(), &phase, &mats);
    for step in 0..8 {
        solver.step(1.0 / 60.0, &input);
        let p = momentum(&solver.read_velocities(), &phase, &mats);
        let drift = pnorm([p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]]);
        assert!(
            drift < 2.0e-4,
            "harmonic-drag momentum drift at step {step}: {drift}"
        );
    }
}

#[test]
fn drag_dense_blob_is_dissipative_at_fine_grind() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [12.0, 12.0, 12.0],
        regions: vec![
            SeedRegion {
                min: [5.5, 6.0, 5.5],
                max: [6.5, 6.0, 6.5],
                species: Species::Water,
            },
            SeedRegion {
                min: [4.5, 5.0, 4.5],
                max: [7.5, 7.0, 7.5],
                species: Species::Grain,
            },
        ],
        ..Scene::default()
    };
    let mats = Materials {
        grain_diameter: 0.5,
        grain_mass: 2.0,
        ..Materials::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &drag_only_config(), &gpu);
    let phase = solver.read_phases();
    let seeded: Vec<[f32; 4]> = phase
        .iter()
        .enumerate()
        .map(|(i, &ph)| {
            if ph == 0 {
                [1.5, 0.0, 0.0, 0.0]
            } else {
                let s = if i % 2 == 0 { -1.0 } else { 1.0 };
                [0.0, 0.25 * s, 0.0, 0.0]
            }
        })
        .collect();
    solver.write_velocities_for_test(&seeded);
    let input = EmissionInput::default();
    let ke0 = kinetic_energy(&solver.read_velocities(), &phase, &mats);

    for _ in 0..6 {
        solver.step(1.0 / 60.0, &input);
    }

    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    let ke1 = kinetic_energy(&vel, &phase, &mats);
    assert!(
        finite(&pos) && finite(&vel),
        "drag produced non-finite state"
    );
    assert!(ke1 <= ke0 + 1.0e-4, "drag increased KE: {ke0} -> {ke1}");
}

#[test]
fn coarse_grind_draws_down_faster_than_fine() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let input = EmissionInput::default();
    let run = |grain_diameter: f32| {
        let scene = Scene::pour_over();
        let mats = Materials {
            grain_diameter,
            ..Materials::default()
        };
        let cfg = Config {
            xsph_viscosity_c: 0.0,
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        let phase = solver.read_phases();
        for _ in 0..120 {
            solver.step(1.0 / 60.0, &input);
        }
        let pos = solver.read_positions();
        (low_water_mean(&pos, &phase), water_mean_y(&pos, &phase))
    };

    let (fine_low, fine_mean) = run(1.0);
    let (coarse_low, coarse_mean) = run(1.6);
    eprintln!(
        "drawdown water y: fine low {fine_low:.3} mean {fine_mean:.3}, coarse low {coarse_low:.3} mean {coarse_mean:.3}"
    );
    assert!(
        coarse_mean + 0.2 < fine_mean,
        "coarse grind should draw down farther: coarse mean {coarse_mean:.3}, fine mean {fine_mean:.3}"
    );
}

#[test]
fn water_threads_the_bed_without_tunneling() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::pour_over();
    let mats = Materials::default();
    let cfg = Config::default();
    let d_wg = mats.grain_diameter;
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    let n = phase.len();
    let n_water = phase.iter().filter(|&&p| p == 0).count();
    let n_grain = n - n_water;
    assert!(
        n_water > 500 && n_grain > 500,
        "expected a mixed scene ({n_water}w/{n_grain}g)"
    );
    let input = EmissionInput::default();

    let frames = 600u32;
    let mut min_wg = f32::INFINITY;
    let mut max_grain_top = 0.0f32;

    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 30 != 0 && f != frames - 1 {
            continue;
        }
        solver.sample_diagnostics();
        let pos = solver.read_positions();
        let vel = solver.read_velocities();

        assert!(
            finite(&pos) && finite(&vel),
            "non-finite state at frame {f}"
        );
        assert!(!solver.diagnostics().overflow, "grid overflow at frame {f}");
        let in_box = |p: &[f32; 4]| {
            (0..3).all(|a| p[a] >= scene.box_min[a] - 0.6 && p[a] <= scene.box_max[a] + 0.6)
        };
        assert!(pos.iter().all(in_box), "particle left the box at frame {f}");

        // Closest water–grain approach (after the impact transient settles).
        if f >= 120 {
            min_wg = min_wg.min(min_water_grain(&pos, &phase));
        }
        // Bed top (highest grain) — the bed must not be crushed flat by the water load.
        let gtop = pos
            .iter()
            .zip(&phase)
            .filter(|(_, &p)| p == 1)
            .map(|(p, _)| p[1])
            .fold(0.0, f32::max);
        max_grain_top = max_grain_top.max(gtop);
    }

    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    let vmax = vel.iter().map(|&v| speed(v)).fold(0.0, f32::max);
    let gtop = pos
        .iter()
        .zip(&phase)
        .filter(|(_, &p)| p == 1)
        .map(|(p, _)| p[1])
        .fold(0.0, f32::max);
    eprintln!(
        "min water–grain {min_wg:.3} (d_wg {d_wg:.2}), bed top {gtop:.1} (peak {max_grain_top:.1}), vmax {vmax:.2}"
    );

    // No tunneling: water never passes through a grain body (stays near the contact distance, not
    // collapsed to ~0). The exclusion is a soft constraint, so allow a modest overlap margin.
    assert!(
        min_wg > 0.55 * d_wg,
        "water tunneled through grains: min water–grain {min_wg:.3} < {:.3}",
        0.55 * d_wg
    );
    // Bed not crushed: the grain column keeps real height (not flattened to a monolayer).
    assert!(
        max_grain_top > 4.0,
        "bed crushed flat (peak grain top {max_grain_top:.1})"
    );
    // No blow-up: the coupled sim stays bounded.
    assert!(vmax < 25.0, "coupled sim unstable (vmax {vmax:.2})");
}

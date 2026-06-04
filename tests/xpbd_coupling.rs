//! Water/bed coupling invariants (GPU-gated; skips without an adapter).
//!
//! Step 1 (grain exclusion + pore-modulated water density; drag/buoyancy come later): a water
//! column poured onto a grain bed. The invariant is that water threads the **pores** without
//! passing through grain **bodies** (no tunneling), the bed isn't crushed, and nothing blows up.
//! Water percolating to the floor is expected — drag (step 2) is what resists it into a realistic
//! drawdown; here there's no resistance, so water simply drains through the pore network.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
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

fn drag_only_config() -> Config {
    Config {
        max_iters: 0,
        bed_max_iters: 0,
        xsph_viscosity_c: 0.0,
        drag_gamma: 0.02,
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
    let mut solver = XpbdSolver::build(&scene, &mats, &drag_only_config(), &gpu);
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

//! Phase 1.4 wetting/cohesion invariants (GPU-gated; skips without an adapter).
//!
//! U4 scope (this file's current tests): the moisture state on `pos.w` is seeded per species,
//! mirrored to `pred.w` each frame, and preserved by every `pos`/`pred` writer — including
//! `apply_drag_pred`, which runs in the drag/buoyancy subcycle and previously zeroed `.w`. The
//! volume/mass/momentum conservation tests (U5/U9) build on these once the absorption passes land.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

/// A small mixed water+grain scene: a water slab released just above a grain slab, so the coupling
/// passes (drag/buoyancy → `apply_drag_pred`) actually run.
fn mixed_scene() -> Scene {
    Scene {
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [2.0, 2.0, 2.0],
                max: [5.0, 3.0, 5.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [2.0, 4.0, 2.0],
                max: [5.0, 5.0, 5.0],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    }
}

/// Expected seeded moisture for a particle by phase: water `f_w = 1`, grain `V_abs = 0`.
fn seeded_for_phase(ph: u32) -> f32 {
    if ph == 0 {
        1.0
    } else {
        0.0
    }
}

/// Config that isolates absorption: density/bed/drag/buoyancy/viscosity off, wetting on. Lets the
/// conservation + competition tests observe the absorption passes alone.
fn absorb_only_config() -> Config {
    Config {
        max_iters: 0,
        bed_max_iters: 0,
        drag_subiters: 0,
        buoyancy_scale: 0.0,
        xsph_viscosity_c: 0.0,
        grain_sleep_speed: 0.0,
        absorb_rate: 0.5,
        ..Config::default()
    }
}

/// Per-grain dry volume (π/6·d³) and water-capacity V_cap = r_max·rho_ratio·V_dry, from materials.
fn capacity(mats: &Materials) -> f32 {
    let v_dry = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    mats.r_max * mats.rho_ratio * v_dry
}

/// A point seed region (one particle) of the given species.
fn point(p: [f32; 3], species: Species) -> SeedRegion {
    SeedRegion {
        min: p,
        max: p,
        species,
    }
}

/// Total absolute volume on the books: water `Σ f_w·V_w` + grain `Σ V_abs`. Conserved across
/// absorption (water volume lost == grain volume gained).
fn total_volume(moisture: &[f32], phase: &[u32], v_w: f32) -> f32 {
    moisture
        .iter()
        .zip(phase)
        .map(|(&m, &ph)| if ph == 0 { m * v_w } else { m })
        .sum()
}

/// Effective-mass momentum: water `particle_mass·f_w`, grain `grain_mass + ρ_w·V_abs`.
fn eff_momentum(
    vel: &[[f32; 4]],
    moisture: &[f32],
    phase: &[u32],
    mats: &Materials,
    rho_w: f32,
) -> [f32; 3] {
    vel.iter()
        .zip(moisture)
        .zip(phase)
        .fold([0.0; 3], |mut acc, ((v, &m), &ph)| {
            let mass = if ph == 0 {
                mats.particle_mass * m
            } else {
                mats.grain_mass + rho_w * m
            };
            acc[0] += mass * v[0];
            acc[1] += mass * v[1];
            acc[2] += mass * v[2];
            acc
        })
}

#[test]
fn seeded_moisture_is_full_water_and_dry_grains() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let solver = XpbdSolver::build(
        &mixed_scene(),
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    let phase = solver.read_phases();
    let moisture = solver.read_moisture();
    assert_eq!(moisture.len(), phase.len());
    assert!(
        phase.contains(&0) && phase.contains(&1),
        "scene must be mixed"
    );
    for (i, (&ph, &m)) in phase.iter().zip(&moisture).enumerate() {
        assert_eq!(
            m,
            seeded_for_phase(ph),
            "particle {i} (phase {ph}) seeded moisture"
        );
    }
}

#[test]
fn moisture_lane_survives_steps_with_drag_and_buoyancy() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    // Default config keeps drag (subiters=4) and buoyancy (scale=1) on, so apply_drag_pred runs.
    let mut solver = XpbdSolver::build(
        &mixed_scene(),
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    let input = EmissionInput::default();

    for _ in 0..30 {
        solver.step(1.0 / 60.0, &input);
    }

    // Read phase in the SAME post-step snapshot as moisture: the cell-order reorder permutes slots,
    // so phase[i] and moisture[i] only refer to the same particle when read from one snapshot.
    let phase = solver.read_phases();
    // No absorption pass exists yet (U5), so the moisture lane must be byte-for-byte unchanged.
    let moisture = solver.read_moisture();
    for (i, (&ph, &m)) in phase.iter().zip(&moisture).enumerate() {
        assert_eq!(
            m,
            seeded_for_phase(ph),
            "particle {i} moisture drifted (preservation bug)"
        );
    }

    // pred.w must still mirror pos.w at end of step — proving every pred writer (predict, apply_dp,
    // apply_drag_pred) preserved .w. A regression in apply_drag_pred (zeroing .w) trips this.
    let pred = solver.read_pred();
    for (i, (p, &m)) in pred.iter().zip(&moisture).enumerate() {
        assert!(
            (p[3] - m).abs() < 1.0e-6,
            "particle {i}: pred.w {} != pos.w {m}",
            p[3]
        );
    }
}

#[test]
fn reset_restores_seeded_moisture() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let scene = mixed_scene();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let phase = solver.read_phases();

    for _ in 0..10 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    solver.reset(&scene);

    let moisture = solver.read_moisture();
    for (i, (&ph, &m)) in phase.iter().zip(&moisture).enumerate() {
        assert_eq!(
            m,
            seeded_for_phase(ph),
            "particle {i} moisture not restored on reset"
        );
    }
}

#[test]
fn absorption_conserves_total_volume() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &absorb_only_config(), &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();

    let initial = total_volume(&solver.read_moisture(), &phase, v_w);
    for _ in 0..120 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    let moisture = solver.read_moisture();
    let final_vol = total_volume(&moisture, &phase, v_w);

    // Exact to float tolerance: volume only moves via the mirrored capped transfer (no discard).
    assert!(
        (final_vol - initial).abs() <= 1.0e-4 * initial.max(1.0),
        "volume drifted: {initial} -> {final_vol}"
    );
    // ...and absorption actually happened (some grain wetted, some water shrank), so it's non-trivial.
    let grain_wet = moisture
        .iter()
        .zip(&phase)
        .any(|(&m, &p)| p == 1 && m > 1.0e-5);
    let water_shrank = moisture
        .iter()
        .zip(&phase)
        .any(|(&m, &p)| p == 0 && m < 0.999);
    assert!(
        grain_wet && water_shrank,
        "no absorption occurred (test would be vacuous)"
    );
}

#[test]
fn absorption_conserves_momentum() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    // Gravity off + absorption-only so the only momentum exchange is the inelastic absorb merge.
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        ..mixed_scene()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &absorb_only_config(), &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();
    let rho_w = mats.particle_mass / v_w;

    // Give every water a sideways velocity so the transfer carries real momentum.
    let vel0: Vec<[f32; 4]> = phase
        .iter()
        .map(|&ph| {
            if ph == 0 {
                [1.5, 0.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            }
        })
        .collect();
    solver.write_velocities_for_test(&vel0);

    let p0 = eff_momentum(
        &solver.read_velocities(),
        &solver.read_moisture(),
        &phase,
        &mats,
        rho_w,
    );
    for _ in 0..30 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    let moisture = solver.read_moisture();
    let p1 = eff_momentum(&solver.read_velocities(), &moisture, &phase, &mats, rho_w);

    let drift =
        ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
    let scale = (p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2])
        .sqrt()
        .max(1.0);
    assert!(
        drift <= 1.0e-3 * scale,
        "momentum drift {drift} (scale {scale})"
    );
    // absorption happened
    assert!(moisture
        .iter()
        .zip(&phase)
        .any(|(&m, &p)| p == 1 && m > 1.0e-5));
}

#[test]
fn one_water_many_grains_no_oversubscription() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    // One water surrounded by three grains, all within the support radius (h=2).
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Water),
            point([3.3, 4.0, 4.0], Species::Grain),
            point([4.7, 4.0, 4.0], Species::Grain),
            point([4.0, 4.7, 4.0], Species::Grain),
        ],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &absorb_only_config(), &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();
    assert_eq!(phase, vec![0, 1, 1, 1]);

    solver.step(1.0 / 60.0, &EmissionInput::default());
    let m = solver.read_moisture();
    let grain_gain: f32 = m
        .iter()
        .zip(&phase)
        .filter(|(_, &p)| p == 1)
        .map(|(&v, _)| v)
        .sum();
    // The three grains together cannot absorb more than the single water particle's volume.
    assert!(
        grain_gain <= v_w + 1.0e-5,
        "grains over-subscribed water: {grain_gain} > {v_w}"
    );
    // And conservation holds for this micro-scene: water volume lost == grain volume gained.
    let water_lost = (1.0 - m[0]) * v_w;
    assert!(
        (water_lost - grain_gain).abs() <= 1.0e-5,
        "water {water_lost} != grain {grain_gain}"
    );
}

#[test]
fn one_grain_many_waters_never_overshoots_capacity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    // One grain surrounded by four waters, all within h.
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([3.3, 4.0, 4.0], Species::Water),
            point([4.7, 4.0, 4.0], Species::Water),
            point([4.0, 4.7, 4.0], Species::Water),
            point([4.0, 3.3, 4.0], Species::Water),
        ],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &absorb_only_config(), &gpu);
    let phase = solver.read_phases();

    for _ in 0..200 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
        let m = solver.read_moisture();
        let v_abs = m[0]; // the grain is region 0
        assert_eq!(phase[0], 1);
        assert!(
            v_abs <= v_cap + 1.0e-4,
            "grain overshot capacity: {v_abs} > {v_cap}"
        );
    }
}

#[test]
fn dry_grain_wets_monotonically_toward_capacity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([4.6, 4.0, 4.0], Species::Water),
        ],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &absorb_only_config(), &gpu);

    let mut prev = 0.0f32;
    for _ in 0..50 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
        let v_abs = solver.read_moisture()[0];
        assert!(
            v_abs >= prev - 1.0e-6,
            "grain moisture decreased: {v_abs} < {prev}"
        );
        assert!(v_abs <= v_cap + 1.0e-4, "grain overshot capacity");
        prev = v_abs;
    }
    assert!(prev > 0.0, "grain never wetted");
}

#[test]
fn wetting_with_full_solve_conserves_volume_and_stays_finite() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    // Full physics ON (density, bed, drag, buoyancy) PLUS absorption — the realistic mixed path,
    // exercising the f_w-weighted PBF density solve against the absorption bookkeeping.
    let cfg = Config {
        absorb_rate: 0.5,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();

    let initial = total_volume(&solver.read_moisture(), &phase, v_w);
    for _ in 0..150 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    let pos = solver.read_positions();
    let moisture = solver.read_moisture();

    // Only the wetting passes touch pos.w, so volume conservation holds even with the full solve.
    let final_vol = total_volume(&moisture, &phase, v_w);
    assert!(
        (final_vol - initial).abs() <= 1.0e-3 * initial.max(1.0),
        "volume drifted under full solve: {initial} -> {final_vol}"
    );
    // No blow-up at the wet front.
    assert!(
        pos.iter()
            .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite()),
        "non-finite position (wet-front blow-up)"
    );
    assert!(
        moisture
            .iter()
            .zip(&phase)
            .any(|(&m, &p)| p == 1 && m > 1.0e-5),
        "no absorption"
    );
}

#[test]
fn swelling_pushes_contacting_grains_apart() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    // Two grains exactly touching (centers grain_diameter apart) with water around them. Density off
    // (water just needs to sit near the grains to wet them); bed contact ON so swelling shows.
    let d = mats.grain_diameter;
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([4.0 + d, 4.0, 4.0], Species::Grain),
            point([4.0 + 0.5 * d, 5.0, 4.0], Species::Water),
            point([4.0 + 0.5 * d, 3.0, 4.0], Species::Water),
            point([4.0, 4.0, 5.0], Species::Water),
            point([4.0 + d, 4.0, 3.0], Species::Water),
        ],
        ..Scene::default()
    };
    let cfg = Config {
        max_iters: 0,
        drag_subiters: 0,
        buoyancy_scale: 0.0,
        xsph_viscosity_c: 0.0,
        grain_sleep_speed: 0.0,
        absorb_rate: 0.5,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    assert_eq!(&phase[0..2], &[1, 1]);

    let gap = |p: &[[f32; 4]]| {
        ((p[0][0] - p[1][0]).powi(2) + (p[0][1] - p[1][1]).powi(2) + (p[0][2] - p[1][2]).powi(2))
            .sqrt()
    };
    let gap0 = gap(&solver.read_positions());
    for _ in 0..150 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    let gap1 = gap(&solver.read_positions());
    let v_abs = solver.read_moisture();

    assert!(v_abs[0] > 1.0e-4 && v_abs[1] > 1.0e-4, "grains did not wet");
    // Swollen grains (d_eff > d) overlap at the original spacing, so contact pushes them apart.
    assert!(
        gap1 > gap0 + 0.02,
        "wetted grains did not swell apart: {gap0} -> {gap1}"
    );
}

#[test]
fn live_porosity_drag_path_conserves_momentum() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    // Drag ON with absorption ON (so live-porosity drag + effective masses are active), gravity off,
    // density/bed/buoyancy off. The only momentum exchanges are the symmetric drag impulse and the
    // inelastic absorption merge — both conservative — so Σ(m_eff·v) is invariant.
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        ..mixed_scene()
    };
    let cfg = Config {
        max_iters: 0,
        bed_max_iters: 0,
        buoyancy_scale: 0.0,
        xsph_viscosity_c: 0.0,
        grain_sleep_speed: 0.0,
        drag_subiters: 4,
        absorb_rate: 0.5,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();
    let rho_w = mats.particle_mass / v_w;

    let vel0: Vec<[f32; 4]> = phase
        .iter()
        .map(|&ph| {
            if ph == 0 {
                [1.5, 0.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            }
        })
        .collect();
    solver.write_velocities_for_test(&vel0);

    let p0 = eff_momentum(
        &solver.read_velocities(),
        &solver.read_moisture(),
        &phase,
        &mats,
        rho_w,
    );
    for _ in 0..30 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    let p1 = eff_momentum(
        &solver.read_velocities(),
        &solver.read_moisture(),
        &phase,
        &mats,
        rho_w,
    );
    let drift =
        ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
    let scale = (p0[0] * p0[0] + p0[1] * p0[1] + p0[2] * p0[2])
        .sqrt()
        .max(1.0);
    assert!(
        drift <= 2.0e-3 * scale,
        "drag+absorption momentum drift {drift} (scale {scale})"
    );
}

#[test]
fn wet_cohesion_holds_grains_tighter_than_no_cohesion() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    // Two grains just past dry contact (within the cohesion reach), water around them, gravity off.
    // s_peak high + modest r_max so cohesion rises monotonically as they wet and d_eff stays below
    // the gap (so the pair is in the cohesion regime, not the non-penetration regime).
    let d = 1.0;
    let gap0 = 1.15 * d;
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([4.0 + gap0, 4.0, 4.0], Species::Grain),
            point([4.0 + 0.5 * gap0, 4.8, 4.0], Species::Water),
            point([4.0 + 0.5 * gap0, 3.2, 4.0], Species::Water),
        ],
        ..Scene::default()
    };
    let run = |c_max: f32| -> f32 {
        let mats = Materials {
            grain_diameter: d,
            r_max: 0.3,  // modest swelling so d_eff stays below the gap
            s_peak: 0.9, // cohesion keeps rising as the grains wet over the run
            c_max,
            ..Materials::default()
        };
        let cfg = Config {
            max_iters: 0,
            drag_subiters: 0,
            buoyancy_scale: 0.0,
            xsph_viscosity_c: 0.0,
            grain_sleep_speed: 0.0,
            absorb_rate: 0.5,
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        for _ in 0..60 {
            solver.step(1.0 / 60.0, &EmissionInput::default());
        }
        let p = solver.read_positions();
        ((p[0][0] - p[1][0]).powi(2) + (p[0][1] - p[1][1]).powi(2) + (p[0][2] - p[1][2]).powi(2))
            .sqrt()
    };
    let gap_cohesive = run(3.0);
    let gap_none = run(0.0);
    // Same wetting/swelling in both runs; only cohesion differs, so it must hold the pair tighter.
    assert!(
        gap_cohesive < gap_none - 0.01,
        "wet cohesion did not tighten the pair: cohesive {gap_cohesive} vs none {gap_none}"
    );
}

/// A dense mixed scene (many particles per grid cell) with the full solve + absorption. This is the
/// regime where the old fixed 64-slot buckets overflowed and broke the gather symmetry → ~5% volume
/// gain; the counting-sort grid makes conservation exact regardless of density.
#[test]
fn dense_scene_absorption_conserves_volume() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let scene = Scene {
        box_max: [16.0, 24.0, 16.0],
        regions: vec![
            // A packed grain bed + a water column dropped onto it — dense cells under gravity.
            SeedRegion {
                min: [3.0, 2.0, 3.0],
                max: [13.0, 7.0, 13.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [4.0, 9.0, 4.0],
                max: [12.0, 16.0, 12.0],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    };
    let cfg = Config {
        absorb_rate: 0.5,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();
    assert!(
        phase.len() > 800,
        "scene not dense enough to stress the grid"
    );

    let initial = total_volume(&solver.read_moisture(), &phase, v_w);
    for _ in 0..150 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    let moisture = solver.read_moisture();
    let final_vol = total_volume(&moisture, &phase, v_w);
    // Counting-sort grid ⇒ no overflow ⇒ exact (the old buckets leaked several percent here).
    assert!(
        (final_vol - initial).abs() <= 1.0e-3 * initial,
        "dense-scene volume drifted: {initial} -> {final_vol}"
    );
    assert!(
        moisture
            .iter()
            .zip(&phase)
            .any(|(&m, &p)| p == 1 && m > 1.0e-4),
        "no absorption in dense scene"
    );
}

#[test]
fn fine_grind_high_rmax_absorption_is_stable() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_wetting: no GPU adapter; skipping.");
        return;
    };
    // Fine grind (small grain) + high r_max = the stiffest case (high drag, large swelling).
    let mats = Materials {
        r_max: 2.0,
        ..Materials::default()
    };
    let cfg = Config {
        absorb_rate: 0.8,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &cfg, &gpu);
    for _ in 0..200 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    let pos = solver.read_positions();
    let moisture = solver.read_moisture();
    assert!(
        pos.iter()
            .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite()),
        "non-finite position (blow-up at fine grind / high r_max)"
    );
    assert!(
        moisture.iter().all(|m| m.is_finite() && *m >= 0.0),
        "non-finite or negative moisture"
    );
}

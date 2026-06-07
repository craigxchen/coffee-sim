//! GPU tests for static-solid (SDF) boundary collision: particles rest on / inside analytic solids
//! without penetrating, the contact standoff is enforced, and solid-free scenes are unaffected.
//! GPU-gated like the other xpbd suites — skips cleanly when no adapter is present.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::sdf::{nearest, SdfPrimitive, SolidKind, MASK_ALL};
use coffee_sim::EmissionInput;
use glam::Vec3;

fn cup() -> SdfPrimitive {
    SdfPrimitive {
        kind: SolidKind::Cylinder {
            center: Vec3::ZERO,
            floor_y: 0.0,
            rim_y: 8.0,
            radius: 3.0,
        },
        species_mask: MASK_ALL,
        friction: 0.3,
    }
}

/// A closed-tip cone (constrains all the way to the apex), both species.
fn closed_cone() -> SdfPrimitive {
    SdfPrimitive {
        kind: SolidKind::Cone {
            center: Vec3::ZERO,
            apex_y: 0.0,
            top_y: 8.0,
            apex_r: 0.6,
            top_r: 4.0,
            thickness: 0.1,
            hole_radius: 0.6,
            apex_open: false,
        },
        species_mask: MASK_ALL,
        friction: 0.6,
    }
}

/// One water particle dropped into a cup settles on the floor surface without penetrating and at
/// near-rest — water projects to the wall surface (offset 0, like the box clamp; a positive offset
/// would re-inject an outward velocity on impact = a bounce).
#[test]
fn single_water_particle_rests_on_the_floor() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_geometry: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let cfg = Config::default();
    // A degenerate (min == max) region seeds exactly one particle at the point.
    let scene = Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-5.0, -2.0, -5.0],
        box_max: [5.0, 12.0, 5.0],
        regions: vec![SeedRegion {
            min: [0.0, 5.0, 0.0],
            max: [0.0, 5.0, 0.0],
            species: Species::Water,
        }],
        solids: vec![cup()],
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let input = EmissionInput::default();
    for _ in 0..400 {
        solver.step(1.0 / 60.0, &input);
    }
    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    assert!(!pos.is_empty(), "seeded at least one particle");
    let p = Vec3::new(pos[0][0], pos[0][1], pos[0][2]);
    let v = Vec3::new(vel[0][0], vel[0][1], vel[0][2]);
    assert!(p.is_finite() && v.is_finite(), "finite state");
    let s = nearest(&scene.solids, p, 0).signed;
    assert!(
        s >= -0.05,
        "water must not penetrate the cup floor: signed {s}"
    );
    assert!(
        s < 0.3,
        "water rests on the floor surface (offset 0): signed {s}"
    );
    assert!(v.length() < 1.0, "settled to near rest: |v| {}", v.length());
}

/// Regression for the contact-offset bounce: water poured into a closed cone settles to low speed.
/// A positive water contact offset re-injected an outward velocity on every wall impact, leaving a
/// handful of particles pinned at high speed indefinitely; projecting to the surface is dissipative.
#[test]
fn water_in_cone_settles_without_bouncing() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_geometry: no GPU adapter; skipping.");
        return;
    };
    let cone = SdfPrimitive {
        kind: SolidKind::Cone {
            center: Vec3::ZERO,
            apex_y: -3.0,
            top_y: 3.0,
            apex_r: 0.6,
            top_r: 4.6834,
            thickness: 0.05,
            hole_radius: 0.6,
            apex_open: false, // closed: water pools, isolating the wall interaction
        },
        species_mask: MASK_ALL,
        friction: 0.3,
    };
    let mats = Materials {
        particle_spacing: 0.5,
        support_radius: 1.0,
        grain_diameter: 0.5,
        ..Materials::default()
    };
    let scene = Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-7.0, -10.0, -7.0],
        box_max: [7.0, 10.0, 7.0],
        regions: vec![SeedRegion {
            min: [-2.0, 0.6, -2.0],
            max: [2.0, 2.6, 2.0],
            species: Species::Water,
        }],
        solids: vec![cone],
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &Config::default(), &gpu);
    let input = EmissionInput::default();
    for _ in 0..200 {
        solver.step(1.0 / 60.0, &input);
    }
    let vel = solver.read_velocities();
    assert!(!vel.is_empty());
    let mut sum = 0.0f32;
    let mut fast = 0u32;
    for v in &vel {
        let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(s.is_finite(), "finite velocity");
        sum += s;
        if s > 12.0 {
            fast += 1;
        }
    }
    let mean = sum / vel.len() as f32;
    // Settled pool: low mean speed and at most a rare transient spike (not a persistent bouncing set).
    assert!(mean < 3.0, "water settled (mean |v| {mean})");
    assert!(
        fast <= 2,
        "no persistent high-speed bouncing ({fast} particles |v|>12)"
    );
}

/// Grains poured into a closed-tip cone settle into the cavity and never penetrate the slanted wall
/// or leak past the closed tip; no grid overflow.
#[test]
fn grains_rest_in_cone_without_penetration() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_geometry: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let cfg = Config::default();
    let scene = Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-6.0, -2.0, -6.0],
        box_max: [6.0, 12.0, 6.0],
        regions: vec![SeedRegion {
            min: [-2.0, 1.0, -2.0],
            max: [2.0, 5.0, 2.0],
            species: Species::Grain,
        }],
        solids: vec![closed_cone()],
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let input = EmissionInput::default();
    for _ in 0..500 {
        solver.step(1.0 / 60.0, &input);
    }
    let pos = solver.read_positions();
    let phase = solver.read_phases();
    assert!(!pos.is_empty(), "seeded grains inside the cone");
    let mut min_signed = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    for (p, &ph) in pos.iter().zip(&phase) {
        let pt = Vec3::new(p[0], p[1], p[2]);
        assert!(pt.is_finite(), "finite grain position");
        min_signed = min_signed.min(nearest(&scene.solids, pt, ph).signed);
        min_y = min_y.min(p[1]);
    }
    assert!(
        min_signed >= -0.15,
        "grains penetrated the cone wall: min signed {min_signed}"
    );
    assert!(
        min_y >= -0.2,
        "grains leaked past the closed apex (y=0): min y {min_y}"
    );
    solver.sample_diagnostics();
    assert!(!solver.diagnostics().overflow, "no grid overflow");
}

/// A solid-free scene exercises the SDF code path as a strict no-op: the existing invariants
/// (finiteness, in-box within the soft margin, no overflow) still hold (R6).
#[test]
fn no_solids_path_is_inert() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_geometry: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::pour_over();
    assert!(scene.solids.is_empty(), "pour_over carries no solids");
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    for _ in 0..200 {
        solver.step(1.0 / 60.0, &input);
    }
    let pos = solver.read_positions();
    assert!(!pos.is_empty());
    let lo = scene.box_min;
    let hi = scene.box_max;
    for p in &pos {
        let pt = Vec3::new(p[0], p[1], p[2]);
        assert!(pt.is_finite(), "finite position");
        let in_box = p[0] >= lo[0] - 0.6
            && p[0] <= hi[0] + 0.6
            && p[1] >= lo[1] - 0.6
            && p[1] <= hi[1] + 0.6
            && p[2] >= lo[2] - 0.6
            && p[2] <= hi[2] + 0.6;
        assert!(in_box, "particle stays in the box: {pt:?}");
    }
    solver.sample_diagnostics();
    assert!(
        !solver.diagnostics().overflow,
        "no grid overflow without solids"
    );
}

/// End-to-end V60: with a permeable bed (fine water through a coarser bed), water passes the
/// grains-only filter and drains through the cone apex into the cup, while the grains stay trapped
/// above the filter tip. Conservation (no particle lost), finiteness, and no grid overflow hold.
#[test]
fn v60_water_drains_into_cup_grains_trapped() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_geometry: no GPU adapter; skipping.");
        return;
    };
    // Calibrated V60 mats (see examples/water_app.rs SCENE=v60): fine water, coarser permeable bed,
    // grains ~1.25x water density so the bed holds and water threads through the pore field.
    let mats = Materials {
        particle_spacing: 0.5,
        support_radius: 1.0,
        grain_diameter: 1.0,
        min_pore_fraction: 0.35,
        grain_mass: 10.0,
        ..Materials::default()
    };
    let scene = Scene::v60();
    let mut solver = XpbdSolver::build(&scene, &mats, &Config::default(), &gpu);
    let phase = solver.read_phases();
    let n0 = phase.len();
    let nw0 = phase.iter().filter(|&&p| p == 0).count();
    let ng0 = phase.iter().filter(|&&p| p == 1).count();
    assert!(nw0 > 0 && ng0 > 0, "v60 seeds water and grains");
    let input = EmissionInput::default();
    for _ in 0..400 {
        solver.step(1.0 / 60.0, &input);
    }
    let pos = solver.read_positions();
    let phase2 = solver.read_phases();
    // Conservation: no particle created or destroyed (absorption off by default).
    assert_eq!(pos.len(), n0, "particle count conserved");
    let mut in_cup = 0u32;
    let mut grain_min_y = f32::INFINITY;
    for (p, &ph) in pos.iter().zip(&phase2) {
        let pt = Vec3::new(p[0], p[1], p[2]);
        assert!(pt.is_finite(), "finite position");
        let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
        if ph == 0 {
            if r < 3.0 && p[1] > -8.0 && p[1] < -3.5 {
                in_cup += 1; // inside the cup volume
            }
        } else {
            grain_min_y = grain_min_y.min(p[1]);
        }
    }
    assert!(
        in_cup > 0,
        "water passed the filter + apex and reached the cup ({in_cup} particles)"
    );
    assert!(
        grain_min_y > -3.4,
        "grains stay trapped above the filter tip (min grain y {grain_min_y})"
    );
    solver.sample_diagnostics();
    assert!(!solver.diagnostics().overflow, "no grid overflow");
}

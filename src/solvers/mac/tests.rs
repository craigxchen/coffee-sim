use glam::DVec3;

use super::{
    grid::{Grid, Mobility},
    MacSimulation,
};
use crate::{
    emission::{EmissionInput, PourEvent},
    engine::{
        scene::{SeedRegion, Species},
        Scene,
    },
    models::Materials,
    utils::config::Config,
};

fn small() -> (Scene, Materials, Config) {
    let scene = Scene {
        box_max: [4.0; 3],
        gravity: [0.0; 3],
        regions: vec![],
        ..Scene::default()
    };
    let mats = Materials {
        particle_spacing: 0.5,
        grain_diameter: 0.5,
        particle_mass: 0.125,
        grain_mass: 0.15,
        ..Materials::default()
    };
    let mut cfg = Config {
        seed_jitter: 0.0,
        ..Config::default()
    };
    cfg.mac.cell_size = 1.0;
    cfg.mac.thermal_exchange = false;
    cfg.mac.swelling_ratio = 0.0;
    (scene, mats, cfg)
}

fn pair() -> MacSimulation {
    let (mut scene, mats, mut cfg) = small();
    scene.regions = vec![
        SeedRegion {
            min: [1.25; 3],
            max: [1.75; 3],
            species: Species::Water,
        },
        SeedRegion {
            min: [1.25; 3],
            max: [1.75; 3],
            species: Species::Grain,
        },
    ];
    cfg.absorb_rate = 1.0;
    cfg.extract_rate = 1.0;
    MacSimulation::new(&scene, &mats, &cfg)
}

#[test]
fn hydrostatic_mac_projection_converges_without_a_divergence_floor() {
    let scene = Scene {
        box_max: [3.0, 6.0, 3.0],
        regions: vec![],
        ..Scene::default()
    };
    let mut grid = Grid::new(&scene, 1.0, false);
    let dt = 0.01;
    for z in 0..3 {
        for y in 0..4 {
            for x in 0..3 {
                grid.cells[(z * 6 + y) * 3 + x].water = 1.0;
            }
        }
    }
    for f in &mut grid.faces {
        let wet = [f.left, f.right]
            .iter()
            .any(|i| i.is_some_and(|i| grid.cells[i].water > 0.0));
        if !wet {
            continue;
        }
        f.mass[0] = 1.0;
        f.volume[0] = 1.0;
        f.mobility = Mobility::new(1.0, 0.0, 0.0, f.blocked[0], true);
        f.velocity[0] = if f.axis == 1 && !f.blocked[0] {
            -20.0 * dt
        } else {
            0.0
        };
        f.coefficient = dt * f.mobility.ww;
    }
    let (iters, residual) = grid.project(dt, 4000, 1e-10);
    assert!(iters < 4000 && residual < 1e-10, "{iters} {residual}");
    let mut max_div: f64 = 0.0;
    for (i, cell) in grid.cells.iter().enumerate() {
        if cell.water == 0.0 {
            continue;
        }
        let div: f64 = grid.cell_faces[i]
            .iter()
            .map(|&(f, sign)| sign * grid.faces[f].velocity[0])
            .sum();
        max_div = max_div.max(div.abs());
    }
    assert!(max_div < 1e-8, "divergence {max_div}");
    // Pressure difference between adjacent submerged centers is rho*g*h.
    assert!((grid.cells[1].pressure - grid.cells[4].pressure - 20.0).abs() < 1e-5);
}

#[test]
fn unconstrained_two_way_drag_moves_the_grain_and_preserves_momentum() {
    let mut s = pair();
    s.cfg.absorb_rate = 0.0;
    s.cfg.extract_rate = 0.0;
    s.water[0].velocity = DVec3::X * 2.0;
    let rho = s.rho();
    let before = s.water_mass(&s.water[0]) * s.water[0].velocity;
    // Advancing a very short interval keeps the particles in the same support region.
    s.advance(0.001, &EmissionInput::default());
    let after = s.water_mass(&s.water[0]) * s.water[0].velocity
        + s.grains[0].mass(rho) * s.grains[0].velocity;
    assert!((after - before).length() < 1e-12, "{before:?} -> {after:?}");
    assert!(s.grains[0].velocity.x > 0.0);
    assert!(
        s.water[0].velocity.x > s.grains[0].velocity.x,
        "drag must allow slip"
    );
}

#[test]
fn absorption_preserves_water_solute_momentum_and_heat_even_when_water_runs_out() {
    let mut s = pair();
    s.cfg.extract_rate = 0.0;
    s.water[0].velocity = DVec3::new(2.0, -1.0, 3.0);
    let before = s.diagnostics();
    for _ in 0..20 {
        s.exchange(10.0);
    }
    let d = s.diagnostics();
    assert!(d.absorbed_water > 0.0 && d.free_water < before.free_water);
    assert!(
        d.water_error.abs() < 1e-12 && d.solute_error.abs() < 1e-12,
        "{d:?}"
    );
    assert!((d.momentum - before.momentum).length() < 1e-12);
    assert!((d.heat - before.heat).abs() < 1e-12);
    assert!(s.grains[0].absorbed <= s.grains[0].capacity + 1e-12);
    assert!(s.water.iter().all(|p| p.volume >= 0.0));
}

#[test]
fn stationary_immersion_extracts_with_absorption_disabled() {
    let mut s = pair();
    s.cfg.absorb_rate = 0.0;
    let before = s.diagnostics();
    for _ in 0..20 {
        s.exchange(0.05);
    }
    let d = s.diagnostics();
    assert!(d.dissolved_solute > 0.0);
    assert!(d.remaining_solute < before.remaining_solute);
    assert!(d.solute_error.abs() < 1e-12, "{d:?}");
    assert!((d.heat - before.heat).abs() < 1e-12);
    let c = s.water[0].solute / s.water_mass(&s.water[0]);
    assert!(c <= s.mats.c_sat as f64 + 1e-12);
}

#[test]
fn thermal_exchange_conserves_heat_and_relaxes_without_overshoot() {
    let mut s = pair();
    s.cfg.absorb_rate = 0.0;
    s.cfg.extract_rate = 0.0;
    s.cfg.mac.thermal_exchange = true;
    let initial = s.diagnostics().heat;
    let cw = s.water_heat_capacity(&s.water[0]);
    let cg = s.grain_heat_capacity(&s.grains[0]);
    let equilibrium = initial / (cw + cg);
    for _ in 0..20 {
        s.exchange(0.5);
    }
    let tw = s.water[0].heat / cw;
    let tg = s.grains[0].heat / cg;
    assert!((s.diagnostics().heat - initial).abs() < 1e-12);
    assert!((tw - equilibrium).abs() < 1e-6 && (tg - equilibrium).abs() < 1e-6);
    assert!(tw >= s.mats.t_amb as f64 && tw <= s.mats.pour_t as f64);
}

#[test]
fn absorption_capacity_matches_the_actual_dry_mass() {
    let s = pair();
    let g = &s.grains[0];
    let expected = s.mats.r_max as f64 * g.initial_mass / s.rho();
    assert!((g.capacity - expected).abs() < expected * 1e-6);
}

#[test]
fn absorbed_solute_does_not_diffuse_into_more_concentrated_liquid() {
    let mut s = pair();
    s.cfg.absorb_rate = 0.0;
    s.grains[0].absorbed = 0.1;
    s.grains[0].retained_solute = 0.001;
    s.grains[0].fast = 0.0;
    s.grains[0].slow = 0.0;
    s.water[0].solute = 0.004;
    let before = s.grains[0].retained_solute;
    s.exchange(1.0);
    assert_eq!(s.grains[0].retained_solute, before);
}

#[test]
fn extraction_agrees_under_timestep_refinement() {
    let run = |dt: f64, n| {
        let mut s = pair();
        s.cfg.absorb_rate = 0.0;
        for _ in 0..n {
            s.exchange(dt);
        }
        s.diagnostics().dissolved_solute
    };
    let coarse = run(0.02, 50);
    let fine = run(0.01, 100);
    let finer = run(0.005, 200);
    assert!((fine - finer).abs() < (coarse - fine).abs());
    assert!((fine - finer).abs() / finer < 0.01);
}

#[test]
fn dry_grains_do_not_extract_without_liquid_contact() {
    let mut s = pair();
    s.water[0].position = DVec3::splat(3.5);
    let before = s.diagnostics();
    s.exchange(10.0);
    assert_eq!(s.diagnostics().remaining_solute, before.remaining_solute);
    assert_eq!(s.diagnostics().absorbed_water, 0.0);
}

#[test]
fn overlapping_exchange_supports_cannot_overdraw_inventories() {
    let mut s = pair();
    // Unequal, partially overlapping weights, many grains competing for little liquid.
    let grain = s.grains[0].clone();
    for i in 0..15 {
        let mut g = grain.clone();
        g.position.x += 0.04 * i as f64;
        s.grains.push(g);
    }
    let before = s.diagnostics();
    for _ in 0..50 {
        s.exchange(0.1);
    }
    let d = s.diagnostics();
    assert!((d.water_error - before.water_error).abs() < 1e-12);
    assert!((d.solute_error - before.solute_error).abs() < 1e-12);
    assert!(s
        .grains
        .iter()
        .all(|g| g.fast >= -1e-14 && g.slow >= -1e-14 && g.absorbed <= g.capacity + 1e-12));
    assert!(s.water.iter().all(|p| p.volume >= 0.0 && p.solute >= 0.0));
}

#[test]
fn saturated_grain_moisture_does_not_remove_pore_water() {
    let mut s = pair();
    s.grains[0].absorbed = s.grains[0].capacity;
    s.cfg.extract_rate = 0.0;
    s.scatter();
    assert!(s.saturation_at(s.grains[0].position) < 0.5);
    assert_eq!(s.grains[0].moisture(), 1.0);
    let before = s.diagnostics().free_water;
    s.exchange(100.0);
    assert_eq!(s.diagnostics().free_water, before);
}

#[test]
fn a_saturated_bed_transmits_water_without_absorbing_it_again() {
    let (mut scene, mats, mut cfg) = small();
    scene.gravity = [0.0, -2.0, 0.0];
    scene.regions = vec![
        SeedRegion {
            min: [1.25, 0.5, 1.25],
            max: [1.75, 1.0, 1.75],
            species: Species::Grain,
        },
        SeedRegion {
            min: [1.25, 1.5, 1.25],
            max: [1.75, 2.0, 1.75],
            species: Species::Water,
        },
    ];
    cfg.mac.frozen_bed = true;
    cfg.mac.initial_moisture = 1.0;
    cfg.mac.open_bottom = true;
    cfg.absorb_rate = 100.0;
    cfg.drag_scale = 0.001;
    let mut s = MacSimulation::new(&scene, &mats, &cfg);
    let before = s.diagnostics();
    s.advance(2.0, &EmissionInput::default());
    let d = s.diagnostics();
    assert_eq!(d.absorbed_water, before.absorbed_water);
    assert!(d.drained_water > 0.5 * before.free_water, "{d:?}");
    assert!(d.water_error.abs() < 1e-12);
    assert!(d.momentum_error.length() < 1e-10, "{d:?}");
}

#[test]
fn full_frame_emission_and_reset_are_deterministic_and_conserved() {
    let (scene, mats, cfg) = small();
    let mut a = MacSimulation::new(&scene, &mats, &cfg);
    let mut b = MacSimulation::new(&scene, &mats, &cfg);
    let input = EmissionInput {
        kettle_pos: [2.0; 3],
        flow_rate: 0.1,
        ..EmissionInput::default()
    };
    a.advance(0.03, &input);
    b.advance(0.03, &input);
    let d = a.diagnostics();
    assert!((d.time - 0.03).abs() < 1e-12);
    assert!(
        (d.emitted_water - 0.03 * input.flow_rate as f64).abs() < 1e-12
            && d.water_error.abs() < 1e-12
    );
    for (x, y) in a.water.iter().zip(&b.water) {
        assert_eq!(x.position, y.position);
    }
    a.advance(
        1.0,
        &EmissionInput {
            event: PourEvent::Reset,
            ..input
        },
    );
    assert!(a.water.is_empty());
    assert_eq!(a.diagnostics().time, 0.0);
    a.advance(0.03, &input);
    for (x, y) in a.water.iter().zip(&b.water) {
        assert_eq!(x.position, y.position);
    }
}

#[test]
fn open_bottom_drains_water_and_retains_grains_with_exact_books() {
    let (mut scene, mats, mut cfg) = small();
    scene.gravity = [0.0, -2.0, 0.0];
    scene.regions = vec![
        SeedRegion {
            min: [1.25, 0.0, 1.25],
            max: [1.75, 0.5, 1.75],
            species: Species::Water,
        },
        SeedRegion {
            min: [2.25, 0.0, 2.25],
            max: [2.75, 0.5, 2.75],
            species: Species::Grain,
        },
    ];
    cfg.mac.open_bottom = true;
    let mut s = MacSimulation::new(&scene, &mats, &cfg);
    s.advance(1.0, &EmissionInput::default());
    let d = s.diagnostics();
    assert!(d.drained_water > 0.0);
    assert_eq!(s.grains.len(), 1);
    assert!(s.grains[0].position.y >= 0.0);
    assert!(d.water_error.abs() < 1e-12 && d.solute_error.abs() < 1e-12);
    assert!(d.momentum_error.length() < 1e-10, "{d:?}");
}

#[test]
fn empty_and_zero_timestep_are_finite_and_inert() {
    let (scene, mats, cfg) = small();
    let mut s = MacSimulation::new(&scene, &mats, &cfg);
    s.advance(0.0, &EmissionInput::default());
    s.advance(f64::NAN, &EmissionInput::default());
    assert_eq!(s.diagnostics().time, 0.0);
    s.advance(0.01, &EmissionInput::default());
    assert_eq!(s.diagnostics().water_error, 0.0);
}

#[test]
fn dense_pool_keeps_volume_and_settles_without_pressure_momentum_creation() {
    let (mut scene, mats, cfg) = small();
    scene.box_max = [4.0, 6.0, 4.0];
    scene.gravity = [0.0, -4.0, 0.0];
    scene.regions = vec![SeedRegion {
        min: [0.0; 3],
        max: [4.0, 2.0, 4.0],
        species: Species::Water,
    }];
    let mut s = MacSimulation::new(&scene, &mats, &cfg);
    for _ in 0..60 {
        s.advance(1.0 / 60.0, &EmissionInput::default());
    }
    let d = s.diagnostics();
    let max_fill = s.grid.cells.iter().map(|c| c.water).fold(0.0, f64::max);
    let max_speed = s
        .water
        .iter()
        .map(|p| p.velocity.length())
        .fold(0.0, f64::max);
    eprintln!("pool fill={max_fill}, speed={max_speed}, diag={d:?}");
    assert!(d.water_error.abs() < 1e-10);
    assert!(d.pressure_residual <= cfg.mac.pressure_tolerance);
    assert!(max_fill <= 1.1, "pool over-compressed: {max_fill}");
    assert!(
        max_speed < 3.0,
        "resting pool spontaneously accelerates: {max_speed}"
    );
    assert!(d.momentum_error.length() < 1e-8, "{d:?}");
}

#[test]
fn coupled_pressure_recovers_hydrostatics_even_with_extremely_stiff_drag() {
    for beta in [0.0, 1.0, 1e6] {
        let scene = Scene {
            box_max: [1.0, 5.0, 1.0],
            regions: vec![],
            ..Scene::default()
        };
        let mut grid = Grid::new(&scene, 1.0, false);
        let dt = 0.01;
        for cell in &mut grid.cells[..4] {
            cell.water = 0.4;
            cell.solid = 0.6;
        }
        for f in &mut grid.faces {
            f.mass = [0.4, 0.9];
            f.volume = [0.4, 0.6];
            f.mobility = Mobility::new(0.4, 0.9, dt * 0.4 * beta, f.blocked[0], true);
            f.velocity = f
                .mobility
                .apply(if f.axis == 1 { -0.4 * 20.0 * dt } else { 0.0 }, 0.0);
            f.coefficient = dt * 0.4 * 0.4 * f.mobility.ww;
        }
        let (_, residual) = grid.project(dt, 4000, 1e-13);
        assert!(residual < 1e-13);
        assert!(
            (grid.cells[0].pressure - grid.cells[1].pressure - 20.0).abs() < 1e-4,
            "beta={beta}"
        );
        assert!(grid
            .faces
            .iter()
            .filter(|f| f.axis == 1 && f.left.is_some_and(|i| i < 4))
            .all(|f| f.velocity[0].abs() < 1e-9));
    }
}

#[test]
fn sparse_water_preserves_ballistic_velocity_differences() {
    let (mut scene, mats, cfg) = small();
    scene.gravity = [0.0, -2.0, 0.0];
    scene.regions = vec![SeedRegion {
        min: [1.5; 3],
        max: [2.0, 2.5, 2.0],
        species: Species::Water,
    }];
    let mut sim = MacSimulation::new(&scene, &mats, &cfg);
    assert_eq!(sim.water.len(), 2);
    sim.water[0].velocity.y = -2.0;
    sim.water[1].velocity.y = -0.5;
    let imposed_momentum = sim.diagnostics().momentum_error;
    sim.advance(0.1, &EmissionInput::default());
    for (p, expected) in sim.water.iter().zip([-2.2, -0.7]) {
        assert!(
            (p.velocity.y - expected).abs() < 1e-10,
            "sparse falling water was numerically braked: {} vs {expected}",
            p.velocity.y
        );
    }
    assert!((sim.diagnostics().momentum_error - imposed_momentum).length() < 1e-10);
}

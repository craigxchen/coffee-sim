//! Runs the real WGSL passes, not a CPU approximation. Set REQUIRE_GPU=1 in GPU CI.
use coffee_sim::{
    engine::{
        scene::{SeedRegion, Species},
        Scene,
    },
    models::Materials,
    solvers::{
        base::Solver,
        mac::MacSimulation,
        mac_gpu::{GpuParticle, MacGpuSolver},
    },
    utils::{config::Config, gpu::GpuContext},
    EmissionInput,
};
fn gpu() -> Option<GpuContext> {
    let g = GpuContext::new_headless();
    assert!(
        g.is_some() || std::env::var_os("REQUIRE_GPU").is_none(),
        "GPU required"
    );
    g
}
fn setup() -> (Scene, Materials, Config) {
    let scene = Scene {
        box_max: [3.0; 3],
        gravity: [0.0; 3],
        regions: vec![],
        ..Default::default()
    };
    let mats = Materials {
        particle_spacing: 0.25,
        particle_mass: 0.25f32.powi(3),
        grain_diameter: 0.5,
        grain_mass: 0.1,
        ..Default::default()
    };
    let mut cfg = Config {
        seed_jitter: 0.0,
        absorb_rate: 0.5,
        extract_rate: 1.0,
        drag_scale: 0.005,
        ..Default::default()
    };
    cfg.mac.cell_size = 0.5;
    cfg.mac.gpu_particle_capacity = 256;
    cfg.mac.gpu_pressure_iterations = 120;
    (scene, mats, cfg)
}
fn wait(g: &GpuContext) {
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
}
fn read(g: &GpuContext, s: &MacGpuSolver) -> Vec<GpuParticle> {
    let size = s.capacity() as u64 * 80;
    let b = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = g.device.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(s.state_buffer(), 0, &b, 0, size);
    g.queue.submit(Some(e.finish()));
    b.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    wait(g);
    let data = b.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    b.unmap();
    result
}
#[test]
fn gpu_matches_reference_free_fall() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.gravity = [0.0, -2.0, 0.0];
    scene.regions = vec![SeedRegion {
        min: [1.0; 3],
        max: [1.25; 3],
        species: Species::Water,
    }];
    cfg.mac.thermal_exchange = false;
    let mut cpu = MacSimulation::new(&scene, &mats, &cfg);
    let mut s = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    for _ in 0..20 {
        cpu.advance(1.0 / 120.0, &EmissionInput::default());
        s.step(1.0 / 120.0, &EmissionInput::default());
        wait(&g);
    }
    let p = read(&g, &s);
    let actual = p.iter().find(|p| p.position[3] == 0.0).unwrap();
    let expected = &cpu.water()[0];
    assert!((actual.position[1] as f64 - expected.position.y).abs() < 2e-5);
    assert!((actual.velocity[1] as f64 - expected.velocity.y).abs() < 2e-5);
    assert!(s.profile().dispatches_per_frame > 0);
}
#[test]
fn gpu_exchange_conserves_water_solute_heat_and_matches_reference() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.regions = vec![
        SeedRegion {
            min: [1.0; 3],
            max: [1.5; 3],
            species: Species::Grain,
        },
        SeedRegion {
            min: [1.0; 3],
            max: [1.5; 3],
            species: Species::Water,
        },
    ];
    cfg.mac.frozen_bed = true;
    cfg.mac.swelling_ratio = 0.0;
    let mut cpu = MacSimulation::new(&scene, &mats, &cfg);
    let heat = cpu.diagnostics().heat;
    let mut s = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    for _ in 0..60 {
        cpu.advance(1.0 / 120.0, &EmissionInput::default());
        s.step(1.0 / 120.0, &EmissionInput::default());
        wait(&g);
    }
    let d = s.diagnostics();
    let c = cpu.diagnostics();
    assert!(d.absorbed_water > 0.0 && d.dissolved_solute > 0.0, "{d:?}");
    assert!(d.water_error.abs() < 2e-6, "{d:?}");
    assert!(d.solute_error.abs() < 2e-6, "{d:?}");
    assert!((d.heat as f64 - heat).abs() < 2e-5, "{d:?}");
    assert!(
        (d.absorbed_water as f64 - c.absorbed_water).abs() < 2e-5,
        "gpu {d:?} cpu {c:?}"
    );
    assert!((d.dissolved_solute as f64 - c.dissolved_solute).abs() < 2e-5);
    for p in read(&g, &s) {
        assert!(p
            .position
            .iter()
            .chain(p.inventory.iter())
            .all(|v| v.is_finite()));
        assert!(p.inventory[0] >= -1e-7 && p.inventory[1] >= -1e-7);
    }
}
#[test]
fn gpu_pool_pressure_reset_and_reused_emission_are_conservative() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.gravity = [0.0, -2.0, 0.0];
    scene.regions = vec![SeedRegion {
        min: [0.0; 3],
        max: [3.0, 0.5, 3.0],
        species: Species::Water,
    }];
    cfg.mac.gpu_particle_capacity = 600;
    let mut s = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    for _ in 0..30 {
        s.step(1.0 / 120.0, &EmissionInput::default());
        wait(&g);
    }
    let d = s.diagnostics();
    assert!(d.water_error.abs() < 1e-5, "{d:?}");
    assert!(d.pressure_residual < 0.005, "{d:?}");
    for p in read(&g, &s) {
        if p.position[3] == 0.0 {
            assert!(
                p.position[1] >= 0.0 && p.position[1] < 0.7,
                "{:?}",
                p.position
            );
        }
    }
    s.reset(&scene);
    s.step(1.0 / 120.0, &EmissionInput::default());
    wait(&g);
    assert!(s.diagnostics().time < 0.01);
}

#[test]
fn gpu_recycles_only_drained_slots_and_accounts_for_full_pool() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.gravity = [0.0, -4.0, 0.0];
    cfg.mac.open_bottom = true;
    cfg.mac.gpu_particle_capacity = 16;
    cfg.nozzle_radius = 0.1;
    let mut s = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    let inlet = EmissionInput {
        kettle_pos: [1.5, 0.2, 1.5],
        flow_rate: 1.0,
        ..Default::default()
    };
    for _ in 0..120 {
        s.step(1.0 / 120.0, &inlet);
        wait(&g);
    }
    let d = s.diagnostics();
    assert!(d.drained_water > 0.5, "{d:?}");
    assert!(
        (d.emitted_water + d.pending_water - 1.0).abs() < 2e-5,
        "{d:?}"
    );
    assert!(d.water_error.abs() < 2e-5, "{d:?}");
    // Force a genuinely full pool: one live slot in a closed box cannot be recycled.
    let mut full_cfg = cfg.clone();
    full_cfg.mac.gpu_particle_capacity = 1;
    full_cfg.mac.open_bottom = false;
    let mut full = MacGpuSolver::build(&scene, &mats, &full_cfg, &g);
    for _ in 0..20 {
        full.step(1.0 / 120.0, &inlet);
        wait(&g);
    }
    let f = full.diagnostics();
    assert!(f.pending_water > 0.1, "{f:?}");
    assert!(
        (f.emitted_water + f.pending_water - 20.0 / 120.0).abs() < 2e-6,
        "{f:?}"
    );
    assert!(f.water_error.abs() < 2e-6, "{f:?}");

    // Reset immediately with readback still pending and change the region on the same grid.
    s.step(1.0 / 120.0, &inlet);
    scene.regions.push(SeedRegion {
        min: [1.0; 3],
        max: [1.25; 3],
        species: Species::Water,
    });
    s.reset(&scene);
    s.step(1.0 / 120.0, &EmissionInput::default());
    wait(&g);
    let reset = s.diagnostics();
    assert!(reset.emitted_water == 0.0 && reset.time < 0.01, "{reset:?}");
    assert!((reset.free_water - 0.25f32.powi(3)).abs() < 1e-6);
}

#[test]
fn gpu_mutual_drag_and_contacts_preserve_total_momentum() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.box_max = [6.0; 3];
    cfg.mac.gpu_particle_capacity = 32;
    cfg.mac.swelling_ratio = 0.0;
    cfg.absorb_rate = 0.0;
    cfg.extract_rate = 0.0;
    cfg.mac.thermal_exchange = false;
    scene.regions = vec![
        SeedRegion {
            min: [2.0; 3],
            max: [3.0, 2.5, 2.5],
            species: Species::Grain,
        },
        SeedRegion {
            min: [2.25; 3],
            max: [2.5; 3],
            species: Species::Water,
        },
    ];
    let mut s = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    let mut particles = read(&g, &s);
    let mut expected = 0.0;
    for p in &mut particles {
        if p.position[3] <= 1.0 {
            p.velocity[0] = if p.position[3] == 1.0 { -0.5 } else { 2.0 };
            let mass = if p.position[3] == 1.0 {
                p.material[3] + p.inventory[1] + p.inventory[3]
            } else {
                p.inventory[0]
            };
            expected += mass * p.velocity[0];
        }
    }
    g.queue
        .write_buffer(s.state_buffer(), 0, bytemuck::cast_slice(&particles));
    for _ in 0..20 {
        s.step(1.0 / 120.0, &EmissionInput::default());
        wait(&g);
    }
    assert!(
        (s.diagnostics().momentum[0] - expected).abs() < 3e-6,
        "{:?}, expected {expected}",
        s.diagnostics()
    );
}

#[test]
fn sparse_jet_keeps_ballistic_acceleration_between_nozzle_and_floor() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.box_max = [4.0, 8.0, 4.0];
    scene.gravity = [0.0, -4.0, 0.0];
    scene.regions = vec![SeedRegion {
        min: [1.0, 4.0, 1.0],
        max: [1.25, 4.5, 1.25],
        species: Species::Water,
    }];
    cfg.mac.gpu_particle_capacity = 2;
    cfg.mac.thermal_exchange = false;
    let mut s = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    let mut initial = read(&g, &s);
    // A falling jet has a velocity gradient: the leading parcel has fallen farther.
    // Both parcels are far from walls and underfill the grid, but share grid faces.
    initial[0].velocity[1] = -6.0;
    initial[1].velocity[1] = -2.0;
    g.queue
        .write_buffer(s.state_buffer(), 0, bytemuck::cast_slice(&initial));
    for _ in 0..12 {
        s.step(1.0 / 120.0, &EmissionInput::default());
        wait(&g);
    }
    let actual = read(&g, &s);
    for i in 0..2 {
        let expected = initial[i].velocity[1] - 4.0 * 0.1;
        assert!(
            (actual[i].velocity[1] - expected).abs() < 2e-4,
            "parcel {i}: air must not brake a sparse jet: got {}, ballistic {expected}",
            actual[i].velocity[1]
        );
    }
}

#[test]
fn v60_outlet_jet_accelerates_through_the_empty_cup() {
    let Some(g) = gpu() else { return };
    let (_, mats, mut cfg) = setup();
    let mut scene = Scene::v60_pour();
    scene.regions = vec![SeedRegion {
        min: [0.0, -4.5, 0.0],
        max: [0.25, -4.0, 0.25],
        species: Species::Water,
    }];
    cfg.mac.cell_size = 0.8; // actual web grid, including the filter and cup SDFs
    cfg.mac.gpu_particle_capacity = 2;
    let mut s = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    let mut initial = read(&g, &s);
    initial[0].velocity[1] = -6.0;
    initial[1].velocity[1] = -2.0;
    g.queue
        .write_buffer(s.state_buffer(), 0, bytemuck::cast_slice(&initial));
    for _ in 0..36 {
        s.step(1.0 / 120.0, &EmissionInput::default());
        wait(&g);
    }
    let actual = read(&g, &s);
    for i in 0..2 {
        let expected = initial[i].velocity[1] + scene.gravity[1] * 0.3;
        assert!(
            (actual[i].velocity[1] - expected).abs() < 3e-4,
            "premature braking in cup: {:?}, expected velocity {expected}",
            actual[i]
        );
        assert!(
            actual[i].position[1] > -7.6,
            "test must stay clear of floor stencil"
        );
    }
}

#[test]
fn inlet_is_distributed_over_physics_time_instead_of_render_frames() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.box_max = [4.0, 8.0, 4.0];
    cfg.nozzle_radius = 0.2;
    cfg.mac.thermal_exchange = false;
    let inlet = EmissionInput {
        kettle_pos: [2.0, 6.0, 2.0],
        flow_rate: 0.2,
        ..Default::default()
    };
    let mut coarse = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    let mut fine = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    coarse.step(1.0 / 30.0, &inlet);
    for _ in 0..4 {
        fine.step(1.0 / 120.0, &inlet);
        wait(&g);
    }
    let centroid = |s: &MacGpuSolver| {
        let particles = read(&g, s);
        let mut volume = 0.0;
        let mut moment = 0.0;
        for p in particles.iter().filter(|p| p.position[3] == 0.0) {
            volume += p.inventory[0];
            moment += p.inventory[0] * p.position[1];
        }
        (volume, moment / volume)
    };
    let a = centroid(&coarse);
    let b = centroid(&fine);
    assert!((a.0 - b.0).abs() < 1e-7, "{a:?} vs {b:?}");
    assert!(
        (a.1 - b.1).abs() < 2e-5,
        "frame-dependent jet: {a:?} vs {b:?}"
    );
}

#[test]
fn cfl_backlog_does_not_inject_future_water_into_the_present() {
    let Some(g) = gpu() else { return };
    let (mut scene, mats, mut cfg) = setup();
    scene.box_max = [4.0, 8.0, 4.0];
    cfg.nozzle_radius = 0.04; // Forces the inlet CFL below the frame dispatch budget.
    cfg.mac.thermal_exchange = false;
    let inlet = EmissionInput {
        kettle_pos: [2.0, 6.0, 2.0],
        flow_rate: 0.2,
        ..Default::default()
    };
    let mut sim = MacGpuSolver::build(&scene, &mats, &cfg, &g);
    sim.step(1.0 / 30.0, &inlet);
    wait(&g);
    let d = sim.diagnostics();
    assert!(d.pending_time > 0.01, "test must exercise overload: {d:?}");
    assert!(
        (d.emitted_water - inlet.flow_rate * d.time).abs() < 1e-7,
        "{d:?}"
    );
    assert!(
        (d.pending_water - inlet.flow_rate * d.pending_time).abs() < 1e-7,
        "{d:?}"
    );
    assert!(d.water_error.abs() < 1e-7, "{d:?}");
}

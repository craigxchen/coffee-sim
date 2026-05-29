use super::*;
use std::sync::mpsc;

use bytemuck::cast_slice;

const DEFAULT_LONG_SETTLE_FRAMES: u32 = 600;
const DEFAULT_LONG_SETTLE_LOG_EVERY_FRAMES: u32 = 120;
const LONG_SETTLE_FRAMES_ENV: &str = "COFFEE_SIM_LONG_SETTLE_FRAMES";
const LONG_SETTLE_LOG_FRAMES_ENV: &str = "COFFEE_SIM_LONG_SETTLE_LOG_FRAMES";
const DEFAULT_SHAPE_SETTLE_FRAMES: u32 = 240;
const SHAPE_SETTLE_FRAMES_ENV: &str = "COFFEE_SIM_SHAPE_SETTLE_FRAMES";

fn env_u32_or(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

// ── Device setup ──

fn request_adapter() -> Option<wgpu::Adapter> {
    if std::env::var_os("COFFEE_SIM_SKIP_GPU_TESTS").is_some() {
        return None;
    }

    let instance = wgpu::Instance::default();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()
}

fn create_device_with_limits(
    adapter: &wgpu::Adapter,
    limits: wgpu::Limits,
    label: &'static str,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .ok()
}

fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let adapter = request_adapter()?;
    create_device_with_limits(&adapter, required_limits(), "coffee-sim test device")
}

// ── Readback helpers ──

#[derive(Debug)]
struct MassSnapshot {
    active_particle_mass: f32,
    active_water_particle_mass: f32,
    bed_held_mass: f32,
    water_slots: u32,
}

fn readback_mass_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> MassSnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;
    let bed_size = (sim.num_bed as usize * 32).max(4) as u64;

    let particle_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle mass staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bed_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bed mass staging"),
        size: bed_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mass readback"),
    });
    encoder.copy_buffer_to_buffer(
        &sim.buffers.particles,
        0,
        &particle_staging,
        0,
        particle_size,
    );
    encoder.copy_buffer_to_buffer(&sim.buffers.bed_extract, 0, &bed_staging, 0, bed_size);
    queue.submit(Some(encoder.finish()));

    let particle_slice = particle_staging.slice(..);
    let bed_slice = bed_staging.slice(..);
    let (tx, rx) = mpsc::channel();
    let tx_particles = tx.clone();
    particle_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx_particles.send(result).expect("particle map callback");
    });
    bed_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("bed map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("particle map recv").expect("particle map");
    rx.recv().expect("bed map recv").expect("bed map");

    let particle_view = particle_slice.get_mapped_range();
    let particle_f32 = cast_slice::<u8, f32>(&particle_view);
    let mut active_particle_mass = 0.0;
    let mut active_water_particle_mass = 0.0;
    for i in 0..particle_count {
        let mass = particle_f32[i * 8 + 7];
        active_particle_mass += mass;
        if i >= sim.num_bed as usize {
            active_water_particle_mass += mass;
        }
    }
    drop(particle_view);
    particle_staging.unmap();

    let bed_view = bed_slice.get_mapped_range();
    let bed_f32 = cast_slice::<u8, f32>(&bed_view);
    let mut bed_held_mass = 0.0;
    for i in 0..sim.num_bed as usize {
        bed_held_mass += bed_f32[i * 8];
    }
    drop(bed_view);
    bed_staging.unmap();

    MassSnapshot {
        active_particle_mass,
        active_water_particle_mass,
        bed_held_mass,
        water_slots: sim.num_water,
    }
}

#[derive(Debug)]
struct WaterParticleVolumeSnapshot {
    all_finite: bool,
    active_count: u32,
    active_mass: f32,
    rest_volume: f32,
    current_volume: f32,
    mean_j: f32,
    min_j: f32,
    max_j: f32,
}

#[derive(Debug)]
struct WaterGridPackingSnapshot {
    all_finite: bool,
    active_count: u32,
    deposited_rest_volume: f32,
    deposited_current_volume: f32,
    max_rest_fraction: f32,
    max_current_fraction: f32,
    max_packed_fraction: f32,
    fractional_cell_count: u32,
    max_fractional_fraction: f32,
    overpacked_cell_count: u32,
    overpacked_volume_fraction: f32,
}

#[derive(Debug)]
struct WaterVelocitySnapshot {
    all_finite: bool,
    active_count: u32,
    active_mass: f32,
    kinetic_energy: f32,
    rms_speed: f32,
    mean_speed: f32,
    lateral_rms_speed: f32,
    max_speed: f32,
    momentum: [f32; 3],
}

#[derive(Debug)]
struct FilterContactSnapshot {
    all_finite: bool,
    active_count: u32,
    near_wall_count: u32,
    outside_paper_count: u32,
    outward_jet_count: u32,
    tangential_sheet_count: u32,
    stuck_wall_count: u32,
    near_wall_fraction: f32,
    outside_paper_fraction: f32,
    outward_jet_fraction: f32,
    tangential_sheet_fraction: f32,
    stuck_wall_fraction: f32,
    max_wall_penetration_m: f32,
    max_outward_speed_m_s: f32,
    max_tangential_speed_m_s: f32,
}

fn readback_particle_data(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
) -> Vec<f32> {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particle data readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &staging, 0, particle_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("particle data map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("particle data map recv")
        .expect("particle data map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.unmap();
    data
}

fn readback_affine_data(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
) -> Vec<f32> {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let affine_size = (particle_count * 48).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: affine_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("affine data readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.affine, 0, &staging, 0, affine_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("affine data map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("affine data map recv")
        .expect("affine data map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.unmap();
    data
}

fn readback_metrics_data(sim: &MpmSim3D, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
    let metrics_size = (state::METRICS_SLOT_COUNT * std::mem::size_of::<u32>()) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("metrics data readback"),
        size: metrics_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("metrics data readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.metrics, 0, &staging, 0, metrics_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("metrics data map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("metrics data map recv")
        .expect("metrics data map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    staging.unmap();
    data
}

fn readback_bed_extract_data(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
) -> Vec<f32> {
    let bed_size = (sim.num_bed as usize * 32).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bed_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bed extract data readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.bed_extract, 0, &staging, 0, bed_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("bed extract data map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("bed extract data map recv")
        .expect("bed extract data map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.unmap();
    data
}

fn readback_water_diagnostics_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> WaterDiagnostics {
    let data = readback_particle_data(sim, device, queue, "water diagnostics staging");
    let affine = readback_affine_data(sim, device, queue, "water affine diagnostics staging");
    sim.water_diagnostics_from_particle_data(&data, &affine)
}

fn readback_filter_contact_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> FilterContactSnapshot {
    let data = readback_particle_data(sim, device, queue, "filter contact staging");
    let Some(filter) = sim.settings.filter.as_ref() else {
        return FilterContactSnapshot {
            all_finite: true,
            active_count: 0,
            near_wall_count: 0,
            outside_paper_count: 0,
            outward_jet_count: 0,
            tangential_sheet_count: 0,
            stuck_wall_count: 0,
            near_wall_fraction: 0.0,
            outside_paper_fraction: 0.0,
            outward_jet_fraction: 0.0,
            tangential_sheet_fraction: 0.0,
            stuck_wall_fraction: 0.0,
            max_wall_penetration_m: 0.0,
            max_outward_speed_m_s: 0.0,
            max_tangential_speed_m_s: 0.0,
        };
    };

    let dx = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32;
    let inactive_thresh = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML * 0.1;
    let filter_bot_abs = filter.center.y + filter.bot_y;
    let filter_top_abs = filter.center.y + filter.top_y;
    let near_wall_margin = dx * 2.5;
    let outside_tolerance = dx * 1.25;
    let outward_jet_speed = units::sim_speed_from_meters_per_second(0.12);
    let tangential_sheet_speed = units::sim_speed_from_meters_per_second(0.12);
    let stuck_speed = units::sim_speed_from_meters_per_second(0.015);

    let start = sim.num_bed as usize;
    let end = start + sim.num_water as usize;
    let mut snapshot = FilterContactSnapshot {
        all_finite: true,
        active_count: 0,
        near_wall_count: 0,
        outside_paper_count: 0,
        outward_jet_count: 0,
        tangential_sheet_count: 0,
        stuck_wall_count: 0,
        near_wall_fraction: 0.0,
        outside_paper_fraction: 0.0,
        outward_jet_fraction: 0.0,
        tangential_sheet_fraction: 0.0,
        stuck_wall_fraction: 0.0,
        max_wall_penetration_m: 0.0,
        max_outward_speed_m_s: 0.0,
        max_tangential_speed_m_s: 0.0,
    };

    for i in start..end {
        let base = i * 8;
        let x = data[base];
        let y = data[base + 1];
        let z = data[base + 2];
        let vx = data[base + 4];
        let vy = data[base + 5];
        let vz = data[base + 6];
        let mass = data[base + 7];

        snapshot.all_finite &= x.is_finite()
            && y.is_finite()
            && z.is_finite()
            && vx.is_finite()
            && vy.is_finite()
            && vz.is_finite()
            && mass.is_finite();

        if mass <= inactive_thresh || y < filter_bot_abs || y > filter_top_abs {
            continue;
        }

        snapshot.active_count += 1;
        let dx_from_center = x - filter.center.x;
        let dz_from_center = z - filter.center.z;
        let radial = (dx_from_center * dx_from_center + dz_from_center * dz_from_center).sqrt();
        let local_y = y - filter.center.y;
        let inner_radius = filter.inner_radius_at_y(local_y);
        let outer_radius = filter.radius_at_y(local_y);
        let gap_to_inner_wall = inner_radius - radial;
        let wall_penetration = (radial - outer_radius).max(0.0);
        snapshot.max_wall_penetration_m = snapshot
            .max_wall_penetration_m
            .max(wall_penetration * units::METERS_PER_SIM_UNIT);

        if radial > outer_radius + outside_tolerance {
            snapshot.outside_paper_count += 1;
        }

        let near_wall =
            gap_to_inner_wall.abs() <= near_wall_margin || radial > inner_radius - near_wall_margin;
        if !near_wall {
            continue;
        }

        snapshot.near_wall_count += 1;
        let inv_radial = if radial > 1e-6 { 1.0 / radial } else { 0.0 };
        let radial_dir_x = dx_from_center * inv_radial;
        let radial_dir_z = dz_from_center * inv_radial;
        let outward_speed = vx * radial_dir_x + vz * radial_dir_z;
        let lateral_speed_sq = vx * vx + vz * vz;
        let tangential_speed = (lateral_speed_sq - outward_speed * outward_speed)
            .max(0.0)
            .sqrt();
        let speed = (lateral_speed_sq + vy * vy).sqrt();
        snapshot.max_outward_speed_m_s = snapshot
            .max_outward_speed_m_s
            .max(outward_speed.max(0.0) * units::METERS_PER_SIM_UNIT);
        snapshot.max_tangential_speed_m_s = snapshot
            .max_tangential_speed_m_s
            .max(tangential_speed * units::METERS_PER_SIM_UNIT);

        if outward_speed > outward_jet_speed {
            snapshot.outward_jet_count += 1;
        }
        if tangential_speed > tangential_sheet_speed {
            snapshot.tangential_sheet_count += 1;
        }
        if speed < stuck_speed {
            snapshot.stuck_wall_count += 1;
        }
    }

    let active = snapshot.active_count.max(1) as f32;
    snapshot.near_wall_fraction = snapshot.near_wall_count as f32 / active;
    snapshot.outside_paper_fraction = snapshot.outside_paper_count as f32 / active;
    snapshot.outward_jet_fraction = snapshot.outward_jet_count as f32 / active;
    snapshot.tangential_sheet_fraction = snapshot.tangential_sheet_count as f32 / active;
    snapshot.stuck_wall_fraction = snapshot.stuck_wall_count as f32 / active;
    snapshot
}

fn readback_water_particle_volume_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> WaterParticleVolumeSnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("water particle volume staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("water particle volume readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &staging, 0, particle_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("water particle volume map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("water particle volume map recv")
        .expect("water particle volume map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view);

    let dx = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32;
    let particle_vol = dx * dx * dx * 0.25;
    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
    let inactive_thresh = nominal_mass * 0.1;

    let start = sim.num_bed as usize;
    let end = start + sim.num_water as usize;
    let mut all_finite = true;
    let mut active_count = 0u32;
    let mut active_mass = 0.0_f32;
    let mut rest_volume = 0.0_f32;
    let mut current_volume = 0.0_f32;
    let mut j_sum = 0.0_f32;
    let mut min_j = f32::MAX;
    let mut max_j = f32::MIN;

    for i in start..end {
        let x = data[i * 8];
        let y = data[i * 8 + 1];
        let z = data[i * 8 + 2];
        let j = data[i * 8 + 3];
        let mass = data[i * 8 + 7];

        all_finite &=
            x.is_finite() && y.is_finite() && z.is_finite() && j.is_finite() && mass.is_finite();

        if mass <= inactive_thresh {
            continue;
        }

        let mass_scale = mass / nominal_mass;
        active_count += 1;
        active_mass += mass;
        rest_volume += mass_scale * particle_vol;
        current_volume += mass_scale * particle_vol * j;
        j_sum += j;
        if j < min_j {
            min_j = j;
        }
        if j > max_j {
            max_j = j;
        }
    }
    drop(view);
    staging.unmap();

    let n = active_count.max(1) as f32;
    WaterParticleVolumeSnapshot {
        all_finite,
        active_count,
        active_mass,
        rest_volume,
        current_volume,
        mean_j: j_sum / n,
        min_j: if active_count > 0 { min_j } else { 0.0 },
        max_j: if active_count > 0 { max_j } else { 0.0 },
    }
}

fn readback_water_grid_packing_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> WaterGridPackingSnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("water grid packing staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("water grid packing readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &staging, 0, particle_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("water grid packing map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("water grid packing map recv")
        .expect("water grid packing map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view);

    let [gx, gy, gz] = sim.settings.grid_dims;
    let dx = sim.settings.bounds_size.x / gx as f32;
    let cell_volume = dx * dx * dx;
    let particle_vol = cell_volume * 0.25;
    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
    let inactive_thresh = nominal_mass * 0.1;
    let origin = Vec3::new(
        -sim.settings.bounds_size.x * 0.5,
        -sim.settings.bounds_size.y * 0.5,
        -sim.settings.bounds_size.z * 0.5,
    );
    let total_cells = gx as usize * gy as usize * gz as usize;
    let mut rest_grid = vec![0.0_f32; total_cells];
    let mut current_grid = vec![0.0_f32; total_cells];

    let mut all_finite = true;
    let mut active_count = 0u32;
    let start = sim.num_bed as usize;
    let end = start + sim.num_water as usize;

    for p in start..end {
        let x = data[p * 8];
        let y = data[p * 8 + 1];
        let z = data[p * 8 + 2];
        let j = data[p * 8 + 3];
        let mass = data[p * 8 + 7];

        all_finite &=
            x.is_finite() && y.is_finite() && z.is_finite() && j.is_finite() && mass.is_finite();

        if mass <= inactive_thresh {
            continue;
        }

        active_count += 1;
        let pos = Vec3::new(x, y, z);
        let grid_pos = (pos - origin) / dx;
        let base = Vec3::new(
            (grid_pos.x - 0.5).floor(),
            (grid_pos.y - 0.5).floor(),
            (grid_pos.z - 0.5).floor(),
        );
        let fx = grid_pos - base;
        let wx = [
            0.5 * (1.5 - fx.x).powi(2),
            0.75 - (fx.x - 1.0).powi(2),
            0.5 * (fx.x - 0.5).powi(2),
        ];
        let wy = [
            0.5 * (1.5 - fx.y).powi(2),
            0.75 - (fx.y - 1.0).powi(2),
            0.5 * (fx.y - 0.5).powi(2),
        ];
        let wz = [
            0.5 * (1.5 - fx.z).powi(2),
            0.75 - (fx.z - 1.0).powi(2),
            0.5 * (fx.z - 0.5).powi(2),
        ];
        let base_x = base.x as i32;
        let base_y = base.y as i32;
        let base_z = base.z as i32;
        let rest_particle_volume = particle_vol * mass / nominal_mass.max(1e-6);
        let current_particle_volume = rest_particle_volume * j.clamp(0.40, 2.00);

        for (i, wx_i) in wx.iter().enumerate() {
            for (j_idx, wy_j) in wy.iter().enumerate() {
                for (k, wz_k) in wz.iter().enumerate() {
                    let cell_x = base_x + i as i32;
                    let cell_y = base_y + j_idx as i32;
                    let cell_z = base_z + k as i32;
                    if cell_x < 0 || cell_y < 0 || cell_z < 0 {
                        continue;
                    }
                    if cell_x >= gx as i32 || cell_y >= gy as i32 || cell_z >= gz as i32 {
                        continue;
                    }

                    let weight = wx_i * wy_j * wz_k;
                    let cell = cell_z as usize * gx as usize * gy as usize
                        + cell_y as usize * gx as usize
                        + cell_x as usize;
                    rest_grid[cell] += weight * rest_particle_volume;
                    current_grid[cell] += weight * current_particle_volume;
                }
            }
        }
    }

    drop(view);
    staging.unmap();

    let mut deposited_rest_volume = 0.0_f32;
    let mut deposited_current_volume = 0.0_f32;
    let mut max_rest_fraction = 0.0_f32;
    let mut max_current_fraction = 0.0_f32;
    let mut max_packed_fraction = 0.0_f32;
    let mut fractional_cell_count = 0u32;
    let mut max_fractional_fraction = 0.0_f32;
    let mut overpacked_cell_count = 0u32;
    let mut overpacked_volume_fraction = 0.0_f32;
    let max_rest_volume_fraction = 1.20_f32;

    for (&rest_volume, &current_volume) in rest_grid.iter().zip(current_grid.iter()) {
        deposited_rest_volume += rest_volume;
        deposited_current_volume += current_volume;
        let rest_fraction = rest_volume / cell_volume.max(1e-8);
        let current_fraction = current_volume / cell_volume.max(1e-8);
        let packed_fraction = rest_fraction.max(current_fraction);
        max_rest_fraction = max_rest_fraction.max(rest_fraction);
        max_current_fraction = max_current_fraction.max(current_fraction);
        max_packed_fraction = max_packed_fraction.max(packed_fraction);
        if packed_fraction > 0.0 && packed_fraction < 1.0 {
            fractional_cell_count += 1;
            max_fractional_fraction = max_fractional_fraction.max(packed_fraction);
        }
        if packed_fraction > max_rest_volume_fraction {
            overpacked_cell_count += 1;
            overpacked_volume_fraction += packed_fraction - max_rest_volume_fraction;
        }
    }

    WaterGridPackingSnapshot {
        all_finite,
        active_count,
        deposited_rest_volume,
        deposited_current_volume,
        max_rest_fraction,
        max_current_fraction,
        max_packed_fraction,
        fractional_cell_count,
        max_fractional_fraction,
        overpacked_cell_count,
        overpacked_volume_fraction,
    }
}

fn readback_water_velocity_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> WaterVelocitySnapshot {
    readback_water_velocity_snapshot_in_y_range(
        sim,
        device,
        queue,
        f32::NEG_INFINITY,
        f32::INFINITY,
    )
}

fn readback_water_velocity_snapshot_in_y_range(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    y_min: f32,
    y_max: f32,
) -> WaterVelocitySnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("water velocity staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("water velocity readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &staging, 0, particle_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("water velocity map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("water velocity map recv")
        .expect("water velocity map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view);

    let inactive_thresh = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML * 0.1;
    let start = sim.num_bed as usize;
    let end = start + sim.num_water as usize;
    let mut all_finite = true;
    let mut active_count = 0u32;
    let mut active_mass = 0.0_f32;
    let mut kinetic_energy = 0.0_f32;
    let mut mass_weighted_speed_sq = 0.0_f32;
    let mut mass_weighted_lateral_speed_sq = 0.0_f32;
    let mut speed_sum = 0.0_f32;
    let mut max_speed = 0.0_f32;
    let mut momentum = [0.0_f32; 3];

    for i in start..end {
        let y = data[i * 8 + 1];
        let vx = data[i * 8 + 4];
        let vy = data[i * 8 + 5];
        let vz = data[i * 8 + 6];
        let mass = data[i * 8 + 7];

        all_finite &=
            y.is_finite() && vx.is_finite() && vy.is_finite() && vz.is_finite() && mass.is_finite();

        if mass <= inactive_thresh || y < y_min || y > y_max {
            continue;
        }

        let speed_sq = vx * vx + vy * vy + vz * vz;
        let speed = speed_sq.sqrt();
        active_count += 1;
        active_mass += mass;
        kinetic_energy += 0.5 * mass * speed_sq;
        mass_weighted_speed_sq += mass * speed_sq;
        mass_weighted_lateral_speed_sq += mass * (vx * vx + vz * vz);
        speed_sum += speed;
        max_speed = max_speed.max(speed);
        momentum[0] += mass * vx;
        momentum[1] += mass * vy;
        momentum[2] += mass * vz;
    }
    drop(view);
    staging.unmap();

    let n = active_count.max(1) as f32;
    WaterVelocitySnapshot {
        all_finite,
        active_count,
        active_mass,
        kinetic_energy,
        rms_speed: (mass_weighted_speed_sq / active_mass.max(1e-6)).sqrt(),
        mean_speed: speed_sum / n,
        lateral_rms_speed: (mass_weighted_lateral_speed_sq / active_mass.max(1e-6)).sqrt(),
        max_speed,
        momentum,
    }
}

#[derive(Debug)]
struct DiagSnapshot {
    all_finite: bool,
    total_mass: f32,
    active_count: u32,
    min_mass: f32,
    max_mass: f32,
    x_min: f32,
    x_max: f32,
    x_mean: f32,
    x_extent: f32,
    y_min: f32,
    y_max: f32,
    y_mean: f32,
    y_extent: f32,
    z_min: f32,
    z_max: f32,
    z_mean: f32,
    z_extent: f32,
    x_neg_y_max: f32,
    x_pos_y_max: f32,
    z_neg_y_max: f32,
    z_pos_y_max: f32,
    mean_j: f32,
    min_j: f32,
    max_j: f32,
}

#[derive(Debug)]
struct BedFilterContainmentSnapshot {
    all_finite: bool,
    active_count: u32,
    max_radial_excess: f32,
    min_floor_clearance: f32,
}

#[derive(Debug)]
struct SaturatedBedMotionSnapshot {
    all_finite: bool,
    active_count: u32,
    saturated_count: u32,
    mean_compression: f32,
    saturated_mean_compression: f32,
}

#[derive(Clone, Copy, Debug)]
struct BedParticleState {
    pos: [f32; 3],
    mass: f32,
    saturation: f32,
}

#[derive(Debug)]
struct BedParticleMotionDeltaSnapshot {
    all_finite: bool,
    active_count: u32,
    saturated_count: u32,
    tracked_count: u32,
    mean_displacement: f32,
    max_displacement: f32,
    mean_downward_displacement: f32,
}

fn readback_bed_filter_containment_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> BedFilterContainmentSnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bed filter containment staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bed filter containment readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &staging, 0, particle_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result)
            .expect("bed filter containment map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("bed filter containment map recv")
        .expect("bed filter containment map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view);

    let filter = FilterConfig::default();
    let filter_bot_abs = filter.center.y + filter.bot_y;
    let filter_top_abs = filter.center.y + filter.top_y;
    let dx = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32;
    let bed_radius = dx * 0.62;
    let inactive_thresh = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML * 0.1;
    let mut all_finite = true;
    let mut active_count = 0u32;
    let mut max_radial_excess = f32::MIN;
    let mut min_floor_clearance = f32::MAX;

    for i in 0..sim.num_bed as usize {
        let x = data[i * 8];
        let y = data[i * 8 + 1];
        let z = data[i * 8 + 2];
        let mass = data[i * 8 + 7];

        all_finite &= x.is_finite() && y.is_finite() && z.is_finite() && mass.is_finite();
        if mass <= inactive_thresh {
            continue;
        }
        active_count += 1;

        let local_y = (y - filter.center.y).clamp(filter.bot_y, filter.top_y);
        let inner_radius = filter.inner_radius_at_y(local_y);
        let radial = (x * x + z * z).sqrt();
        max_radial_excess = max_radial_excess.max(radial + bed_radius - inner_radius);
        min_floor_clearance = min_floor_clearance.min(y - (filter_bot_abs + bed_radius));

        if y < filter_bot_abs || y > filter_top_abs {
            all_finite = all_finite && y.is_finite();
        }
    }
    drop(view);
    staging.unmap();

    BedFilterContainmentSnapshot {
        all_finite,
        active_count,
        max_radial_excess: if active_count > 0 {
            max_radial_excess
        } else {
            0.0
        },
        min_floor_clearance: if active_count > 0 {
            min_floor_clearance
        } else {
            0.0
        },
    }
}

fn readback_bed_particle_states(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<BedParticleState> {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;
    let bed_size = (sim.num_bed as usize * 32).max(4) as u64;

    let particle_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bed particle state particle staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bed_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bed particle state extract staging"),
        size: bed_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bed particle state readback"),
    });
    encoder.copy_buffer_to_buffer(
        &sim.buffers.particles,
        0,
        &particle_staging,
        0,
        particle_size,
    );
    encoder.copy_buffer_to_buffer(&sim.buffers.bed_extract, 0, &bed_staging, 0, bed_size);
    queue.submit(Some(encoder.finish()));

    let particle_slice = particle_staging.slice(..);
    let bed_slice = bed_staging.slice(..);
    let (tx, rx) = mpsc::channel();
    let tx_particles = tx.clone();
    particle_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx_particles
            .send(result)
            .expect("bed particle state particle map callback");
    });
    bed_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result)
            .expect("bed particle state extract map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("bed particle state particle map recv")
        .expect("bed particle state particle map");
    rx.recv()
        .expect("bed particle state extract map recv")
        .expect("bed particle state extract map");

    let particle_view = particle_slice.get_mapped_range();
    let particle_f32 = cast_slice::<u8, f32>(&particle_view);
    let bed_view = bed_slice.get_mapped_range();
    let bed_f32 = cast_slice::<u8, f32>(&bed_view);

    let mut states = Vec::with_capacity(sim.num_bed as usize);
    for i in 0..sim.num_bed as usize {
        states.push(BedParticleState {
            pos: [
                particle_f32[i * 8],
                particle_f32[i * 8 + 1],
                particle_f32[i * 8 + 2],
            ],
            mass: particle_f32[i * 8 + 7],
            saturation: bed_f32[i * 8 + 7],
        });
    }

    drop(bed_view);
    bed_staging.unmap();
    drop(particle_view);
    particle_staging.unmap();

    states
}

fn bed_particle_motion_delta_snapshot(
    before: &[BedParticleState],
    after: &[BedParticleState],
    active_mass_threshold: f32,
    saturated_threshold: f32,
    center_radius: f32,
) -> BedParticleMotionDeltaSnapshot {
    let count = before.len().min(after.len());
    let mut all_finite = true;
    let mut active_count = 0u32;
    let mut saturated_count = 0u32;
    let mut tracked_count = 0u32;
    let mut displacement_sum = 0.0_f32;
    let mut max_displacement = 0.0_f32;
    let mut downward_sum = 0.0_f32;

    for i in 0..count {
        let b = before[i];
        let a = after[i];
        all_finite &= b.pos.iter().all(|v| v.is_finite())
            && a.pos.iter().all(|v| v.is_finite())
            && b.mass.is_finite()
            && a.mass.is_finite()
            && a.saturation.is_finite();

        if a.mass <= active_mass_threshold {
            continue;
        }
        active_count += 1;

        let saturated = a.saturation >= saturated_threshold;
        if saturated {
            saturated_count += 1;
        }

        let radial = a.pos[0].hypot(a.pos[2]);
        if !saturated || radial > center_radius {
            continue;
        }

        let dx = a.pos[0] - b.pos[0];
        let dy = a.pos[1] - b.pos[1];
        let dz = a.pos[2] - b.pos[2];
        let displacement = dx.hypot(dy).hypot(dz);
        tracked_count += 1;
        displacement_sum += displacement;
        max_displacement = max_displacement.max(displacement);
        downward_sum += (b.pos[1] - a.pos[1]).max(0.0);
    }

    BedParticleMotionDeltaSnapshot {
        all_finite,
        active_count,
        saturated_count,
        tracked_count,
        mean_displacement: displacement_sum / tracked_count.max(1) as f32,
        max_displacement,
        mean_downward_displacement: downward_sum / tracked_count.max(1) as f32,
    }
}

fn readback_saturated_bed_motion_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> SaturatedBedMotionSnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;
    let bed_size = (sim.num_bed as usize * 32).max(4) as u64;

    let particle_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("saturated bed particle staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bed_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("saturated bed extract staging"),
        size: bed_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("saturated bed motion readback"),
    });
    encoder.copy_buffer_to_buffer(
        &sim.buffers.particles,
        0,
        &particle_staging,
        0,
        particle_size,
    );
    encoder.copy_buffer_to_buffer(&sim.buffers.bed_extract, 0, &bed_staging, 0, bed_size);
    queue.submit(Some(encoder.finish()));

    let particle_slice = particle_staging.slice(..);
    let bed_slice = bed_staging.slice(..);
    let (tx, rx) = mpsc::channel();
    let tx_particles = tx.clone();
    particle_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx_particles
            .send(result)
            .expect("saturated bed particle map callback");
    });
    bed_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("saturated bed extract map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("saturated bed particle map recv")
        .expect("saturated bed particle map");
    rx.recv()
        .expect("saturated bed extract map recv")
        .expect("saturated bed extract map");

    let particle_view = particle_slice.get_mapped_range();
    let particle_f32 = cast_slice::<u8, f32>(&particle_view);
    let bed_view = bed_slice.get_mapped_range();
    let bed_f32 = cast_slice::<u8, f32>(&bed_view);

    let inactive_thresh = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML * 0.1;
    let mut all_finite = true;
    let mut active_count = 0u32;
    let mut saturated_count = 0u32;
    let mut compression_sum = 0.0_f32;
    let mut saturated_compression_sum = 0.0_f32;

    for i in 0..sim.num_bed as usize {
        let mass = particle_f32[i * 8 + 7];
        let compression = bed_f32[i * 8 + 3];
        let saturation = bed_f32[i * 8 + 7];

        all_finite &= mass.is_finite() && compression.is_finite() && saturation.is_finite();
        if mass <= inactive_thresh {
            continue;
        }

        active_count += 1;
        compression_sum += compression;
        if saturation >= 0.65 {
            saturated_count += 1;
            saturated_compression_sum += compression;
        }
    }

    drop(bed_view);
    bed_staging.unmap();
    drop(particle_view);
    particle_staging.unmap();

    SaturatedBedMotionSnapshot {
        all_finite,
        active_count,
        saturated_count,
        mean_compression: compression_sum / active_count.max(1) as f32,
        saturated_mean_compression: saturated_compression_sum / saturated_count.max(1) as f32,
    }
}

fn readback_diag_snapshot_range(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    start: usize,
    count: usize,
) -> DiagSnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("diag staging"),
        size: particle_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("diag readback"),
    });
    encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &staging, 0, particle_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("diag map callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("diag map recv").expect("diag map");

    let view = slice.get_mapped_range();
    let data = cast_slice::<u8, f32>(&view);

    let inactive_thresh = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML * 0.1;
    let mut total_mass = 0.0_f32;
    let mut active_count = 0u32;
    let mut min_mass = f32::MAX;
    let mut max_mass = f32::MIN;
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    let mut z_min = f32::MAX;
    let mut z_max = f32::MIN;
    let mut x_sum = 0.0_f32;
    let mut y_sum = 0.0_f32;
    let mut z_sum = 0.0_f32;
    let mut x_neg_y_max = f32::MIN;
    let mut x_pos_y_max = f32::MIN;
    let mut z_neg_y_max = f32::MIN;
    let mut z_pos_y_max = f32::MIN;
    let mut j_sum = 0.0_f32;
    let mut j_min = f32::MAX;
    let mut j_max = f32::MIN;
    let mut all_finite = true;

    let end = (start + count).min(particle_count);
    for i in start..end {
        let x = data[i * 8];
        let z = data[i * 8 + 2];
        let mass = data[i * 8 + 7];
        let j = data[i * 8 + 3];
        let y = data[i * 8 + 1];

        all_finite &=
            x.is_finite() && y.is_finite() && z.is_finite() && j.is_finite() && mass.is_finite();

        if mass <= inactive_thresh {
            continue;
        }
        active_count += 1;
        total_mass += mass;
        if mass < min_mass {
            min_mass = mass;
        }
        if mass > max_mass {
            max_mass = mass;
        }
        if x < x_min {
            x_min = x;
        }
        if x > x_max {
            x_max = x;
        }
        if y < y_min {
            y_min = y;
        }
        if y > y_max {
            y_max = y;
        }
        if z < z_min {
            z_min = z;
        }
        if z > z_max {
            z_max = z;
        }
        x_sum += x;
        y_sum += y;
        z_sum += z;
        if x < 0.0 {
            x_neg_y_max = x_neg_y_max.max(y);
        } else {
            x_pos_y_max = x_pos_y_max.max(y);
        }
        if z < 0.0 {
            z_neg_y_max = z_neg_y_max.max(y);
        } else {
            z_pos_y_max = z_pos_y_max.max(y);
        }
        j_sum += j;
        if j < j_min {
            j_min = j;
        }
        if j > j_max {
            j_max = j;
        }
    }
    drop(view);
    staging.unmap();

    let n = active_count.max(1) as f32;
    DiagSnapshot {
        all_finite,
        total_mass,
        active_count,
        min_mass: if active_count > 0 { min_mass } else { 0.0 },
        max_mass: if active_count > 0 { max_mass } else { 0.0 },
        x_min: if active_count > 0 { x_min } else { 0.0 },
        x_max: if active_count > 0 { x_max } else { 0.0 },
        x_mean: x_sum / n,
        x_extent: if active_count > 0 { x_max - x_min } else { 0.0 },
        y_min: if active_count > 0 { y_min } else { 0.0 },
        y_max: if active_count > 0 { y_max } else { 0.0 },
        y_mean: y_sum / n,
        y_extent: if active_count > 0 { y_max - y_min } else { 0.0 },
        z_min: if active_count > 0 { z_min } else { 0.0 },
        z_max: if active_count > 0 { z_max } else { 0.0 },
        z_mean: z_sum / n,
        z_extent: if active_count > 0 { z_max - z_min } else { 0.0 },
        x_neg_y_max: if active_count > 0 { x_neg_y_max } else { 0.0 },
        x_pos_y_max: if active_count > 0 { x_pos_y_max } else { 0.0 },
        z_neg_y_max: if active_count > 0 { z_neg_y_max } else { 0.0 },
        z_pos_y_max: if active_count > 0 { z_pos_y_max } else { 0.0 },
        mean_j: j_sum / n,
        min_j: if active_count > 0 { j_min } else { 0.0 },
        max_j: if active_count > 0 { j_max } else { 0.0 },
    }
}

fn readback_diag_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> DiagSnapshot {
    readback_diag_snapshot_range(
        sim,
        device,
        queue,
        0,
        (sim.num_water + sim.num_bed) as usize,
    )
}

fn readback_bed_diag_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> DiagSnapshot {
    readback_diag_snapshot_range(sim, device, queue, 0, sim.num_bed as usize)
}

fn benchmark_bed_bounds_y() -> (f32, f32) {
    let settings = MpmSettings::benchmark_center_pour();
    let bed = settings.bed.as_ref().expect("benchmark scene has a bed");
    (bed.center.y + bed.bot_y, bed.center.y + bed.top_y)
}

// ── Debug scene smoke tests ──

#[test]
fn debug_scene_seed_dispatch_matches_scene_type() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    for scene in DebugScene::ALL {
        let mut sim = MpmSim3D::new(&device, &queue, scene.settings());
        scene.seed(&mut sim, &queue);

        match scene {
            DebugScene::FilterWaterBlock
            | DebugScene::SeededPaperWallSheet
            | DebugScene::FilterApexDrain
            | DebugScene::CupWallFloorCornerContact
            | DebugScene::AsymmetricCupMoundSettle
            | DebugScene::HydrostaticColumn
            | DebugScene::DamBreakSlosh
            | DebugScene::HighVelocityJetImpact
            | DebugScene::UniformBedSaturation => {
                assert!(
                    sim.water_slots_used() > 0,
                    "{} should seed initial water particles",
                    scene.id(),
                );
            }
            DebugScene::OffCenterFilterWallPour
            | DebugScene::SparseFreeJet
            | DebugScene::PermeabilityComparison
            | DebugScene::ParticleCapacityStress => {
                assert_eq!(
                    sim.water_slots_used(),
                    0,
                    "{} should rely on live inflow instead of seeded water",
                    scene.id(),
                );
                assert!(
                    sim.exit_speed_m_s() > 0.0,
                    "{} live-inflow scene should start with a non-zero exit speed",
                    scene.id(),
                );
            }
        }
    }
}

// ── Pipeline validation ──

#[test]
fn pipelines_fit_within_required_limits() {
    let Some(adapter) = request_adapter() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some((device, queue)) =
        create_device_with_limits(&adapter, required_limits(), "required-limits device")
    else {
        eprintln!("skipping: adapter does not support required limits");
        return;
    };

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _sim = MpmSim3D::new(&device, &queue, MpmSettings::default_v60());
    let error = pollster::block_on(error_scope.pop());
    assert!(
        error.is_none(),
        "MpmSim3D::new produced a validation error under required_limits(): {error:?}",
    );
}

#[test]
fn pipelines_exceed_spec_default_limits() {
    let Some(adapter) = request_adapter() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Some((device, queue)) =
        create_device_with_limits(&adapter, wgpu::Limits::default(), "spec-default device")
    else {
        eprintln!("skipping: adapter does not support spec default limits");
        return;
    };

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _sim = MpmSim3D::new(&device, &queue, MpmSettings::default_v60());
    let error = pollster::block_on(error_scope.pop());
    assert!(
        error.is_some(),
        "expected a validation error when constructing MpmSim3D at spec-default limits, but \
         pipeline creation succeeded — `required_limits()` may no longer be necessary",
    );
}

// ── Mass balance ──

#[test]
fn mass_readback_harness() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let settings = MpmSettings::benchmark_center_pour();
    let mut sim = MpmSim3D::new(&device, &queue, settings);
    for _ in 0..10 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let snapshot = readback_mass_snapshot(&sim, &device, &queue);
    assert!(snapshot.active_particle_mass.is_finite());
    assert!(snapshot.active_water_particle_mass.is_finite());
    assert!(snapshot.bed_held_mass.is_finite());
    assert!(snapshot.active_particle_mass >= 0.0);
    assert!(snapshot.active_water_particle_mass >= 0.0);
    assert!(snapshot.bed_held_mass >= 0.0);
}

#[test]
fn active_pour_particle_loss_matches_bed_gain() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..45 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let before = readback_mass_snapshot(&sim, &device, &queue);

    for _ in 0..10 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let after = readback_mass_snapshot(&sim, &device, &queue);

    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
    let emitted_mass = after.water_slots.saturating_sub(before.water_slots) as f32 * nominal_mass;
    let water_gain = after.active_water_particle_mass - before.active_water_particle_mass;
    let bed_gain = after.bed_held_mass - before.bed_held_mass;
    let particle_loss_to_bed = emitted_mass - water_gain;
    let err = (particle_loss_to_bed - bed_gain).abs();
    let tolerance = (bed_gain.abs() * 0.10).max(nominal_mass * 40.0);

    assert!(
        bed_gain > nominal_mass,
        "active pour did not transfer measurable water into bed: before={before:?} after={after:?}",
    );
    assert!(
        err <= tolerance,
        "water particle loss should match bed-held gain during active pour: \
         emitted={emitted_mass} water_gain={water_gain} particle_loss_to_bed={particle_loss_to_bed} \
         bed_gain={bed_gain} err={err} tolerance={tolerance} before={before:?} after={after:?}",
    );
}

#[test]
fn active_pour_rest_volume_loss_matches_bed_gain() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..45 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let before_mass = readback_mass_snapshot(&sim, &device, &queue);
    let before_volume = readback_water_particle_volume_snapshot(&sim, &device, &queue);

    for _ in 0..10 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let after_mass = readback_mass_snapshot(&sim, &device, &queue);
    let after_volume = readback_water_particle_volume_snapshot(&sim, &device, &queue);

    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
    let dx = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32;
    let particle_vol = dx * dx * dx * 0.25;
    let emitted_volume = after_mass
        .water_slots
        .saturating_sub(before_mass.water_slots) as f32
        * particle_vol;
    let water_rest_volume_gain = after_volume.rest_volume - before_volume.rest_volume;
    let bed_gain_volume =
        (after_mass.bed_held_mass - before_mass.bed_held_mass) / nominal_mass * particle_vol;
    let particle_volume_loss_to_bed = emitted_volume - water_rest_volume_gain;
    let err = (particle_volume_loss_to_bed - bed_gain_volume).abs();
    let tolerance = (bed_gain_volume.abs() * 0.25).max(particle_vol * 24.0);

    assert!(
        before_volume.all_finite && after_volume.all_finite,
        "active pour produced non-finite water volume state: before={before_volume:?} after={after_volume:?}",
    );
    assert!(
        bed_gain_volume > particle_vol,
        "active pour did not transfer measurable volume into bed: before_mass={before_mass:?} after_mass={after_mass:?}",
    );
    assert!(
        err <= tolerance,
        "water particle rest-volume loss should match bed-held gain during active pour: \
         emitted={emitted_volume} water_gain={water_rest_volume_gain} \
         particle_loss_to_bed={particle_volume_loss_to_bed} bed_gain={bed_gain_volume} \
         err={err} tolerance={tolerance} before={before_volume:?} after={after_volume:?}",
    );
}

#[test]
fn saturated_center_pour_particle_loss_matches_bed_gain() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..150 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let before = readback_mass_snapshot(&sim, &device, &queue);
    let before_bed = readback_saturated_bed_motion_snapshot(&sim, &device, &queue);

    for _ in 0..20 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let after = readback_mass_snapshot(&sim, &device, &queue);
    let after_bed = readback_saturated_bed_motion_snapshot(&sim, &device, &queue);

    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
    let emitted_mass = after.water_slots.saturating_sub(before.water_slots) as f32 * nominal_mass;
    let water_gain = after.active_water_particle_mass - before.active_water_particle_mass;
    let bed_gain = after.bed_held_mass - before.bed_held_mass;
    let particle_loss_to_bed = emitted_mass - water_gain;
    let err = (particle_loss_to_bed - bed_gain).abs();
    let tolerance = (bed_gain.abs() * 0.10).max(nominal_mass * 40.0);

    assert!(
        before_bed.all_finite && after_bed.all_finite,
        "saturated center pour produced non-finite bed state: before_bed={before_bed:?} after_bed={after_bed:?}",
    );
    assert!(
        before_bed.saturated_count > 8 && after_bed.saturated_count >= before_bed.saturated_count,
        "center pour did not create a sustained saturated bed population: before_bed={before_bed:?} after_bed={after_bed:?}",
    );
    assert!(
        bed_gain > nominal_mass,
        "saturated center pour did not transfer measurable water into bed: before={before:?} after={after:?}",
    );
    assert!(
        err <= tolerance,
        "saturated center-pour water particle loss should match bed-held gain: \
         emitted={emitted_mass} water_gain={water_gain} particle_loss_to_bed={particle_loss_to_bed} \
         bed_gain={bed_gain} err={err} tolerance={tolerance} before={before:?} after={after:?}",
    );
}

#[test]
fn bed_settling_stability() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let snapshot = readback_bed_diag_snapshot(&sim, &device, &queue);
    assert!(snapshot.all_finite, "dry bed produced non-finite state");
    assert!(
        snapshot.active_count > 0,
        "dry bed lost all active particles"
    );
    assert!(
        snapshot.y_extent > 0.8,
        "dry bed collapsed to a near-point: {:?}",
        snapshot
    );
    assert!(
        snapshot.min_j > 0.55,
        "dry bed over-compressed during settle: {:?}",
        snapshot
    );
    assert!(
        snapshot.max_j < 1.25,
        "dry bed over-expanded during settle: {:?}",
        snapshot
    );
}

#[test]
fn bed_long_run_creep_is_bounded_without_water() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_diag_snapshot(&sim, &device, &queue);

    for _ in 0..240 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let late = readback_bed_diag_snapshot(&sim, &device, &queue);

    assert!(
        late.all_finite,
        "long-run dry bed produced non-finite state"
    );
    let mean_drop = (settled.y_mean - late.y_mean).abs();
    assert!(
        mean_drop < 0.35,
        "dry bed continued creeping after settle: settled={:?} late={:?}",
        settled,
        late
    );
    assert!(
        late.min_j > 0.5,
        "dry bed hit the compaction clamp during long-run settle: {:?}",
        late
    );
}

#[test]
#[ignore = "target scenario: requires free granular coffee emission before replacing the pre-seated bed scaffold"]
fn dry_grounds_pour_into_empty_filter_forms_stable_tapered_bed() {
    // This is intentionally a target, not a fake passing regression. The
    // production scene currently initializes a pre-seated coffee-bed scaffold.
    // A realistic dry-grounds pour needs bed-phase particle emission, granular
    // contact/friction, filter collision, and a settle criterion that does not
    // collapse the whole dose into the apex.
    panic!(
        "implement dry coffee-ground emission into an empty filter before enabling this scenario"
    );
}

#[test]
fn bed_first_water_impact_is_bounded() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_diag_snapshot(&sim, &device, &queue);

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..45 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let impacted = readback_bed_diag_snapshot(&sim, &device, &queue);

    assert!(
        impacted.all_finite,
        "bed produced non-finite state under first water impact"
    );
    assert!(
        impacted.y_extent > settled.y_extent * 0.65,
        "first water impact collapsed bed shape too quickly: settled={settled:?} impacted={impacted:?}",
    );
    assert!(
        impacted.min_j > 0.45,
        "first water impact over-compressed bed: settled={settled:?} impacted={impacted:?}",
    );
    assert!(
        (impacted.y_mean - settled.y_mean).abs() < 0.55,
        "first water impact displaced bed centroid too abruptly: settled={settled:?} impacted={impacted:?}",
    );
}

#[test]
fn bed_short_pour_retains_shape() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_diag_snapshot(&sim, &device, &queue);

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let wet = readback_bed_diag_snapshot(&sim, &device, &queue);
    let dx = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32;

    assert!(wet.all_finite, "short pour produced non-finite bed state");
    assert!(
        wet.y_extent > settled.y_extent * 0.5,
        "short pour collapsed bed shape too aggressively: settled={settled:?} wet={wet:?}",
    );
    assert!(
        wet.min_j > 0.4,
        "short pour over-compressed bed: settled={settled:?} wet={wet:?}",
    );
    assert!(
        wet.max_j <= 1.4001,
        "short pour over-expanded bed: settled={settled:?} wet={wet:?}",
    );
    assert!(
        wet.x_extent <= settled.x_extent * 1.04 + dx
            && wet.z_extent <= settled.z_extent * 1.04 + dx,
        "short pour pushed the coffee bed laterally into a wall-piled shape: \
         settled={settled:?} wet={wet:?}",
    );
}

#[test]
fn bed_does_not_rebound_after_pour_off() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_diag_snapshot(&sim, &device, &queue);

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let wet = readback_bed_diag_snapshot(&sim, &device, &queue);

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..90 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let recovered = readback_bed_diag_snapshot(&sim, &device, &queue);

    assert!(
        settled.all_finite && wet.all_finite && recovered.all_finite,
        "post-pour bed recovery produced non-finite state: settled={settled:?} wet={wet:?} recovered={recovered:?}",
    );
    assert_eq!(
        wet.active_count, recovered.active_count,
        "bed active particle count changed during post-pour recovery: wet={wet:?} recovered={recovered:?}",
    );
    assert!(
        wet.y_mean <= settled.y_mean + 0.05,
        "short pour did not leave bed measurably compressed before recovery check: settled={settled:?} wet={wet:?}",
    );
    assert!(
        recovered.y_mean <= wet.y_mean + 0.12,
        "bed centroid rebounded upward after pour-off: settled={settled:?} wet={wet:?} recovered={recovered:?}",
    );
    assert!(
        recovered.y_extent <= wet.y_extent * 1.12,
        "bed expanded vertically after pour-off: settled={settled:?} wet={wet:?} recovered={recovered:?}",
    );
    assert!(
        recovered.max_j <= 1.4001,
        "bed elastic volume recovery exceeded wet-bed bound after pour-off: wet={wet:?} recovered={recovered:?}",
    );
}

#[test]
fn wet_bed_stays_inside_filter_paper() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let containment = readback_bed_filter_containment_snapshot(&sim, &device, &queue);
    let tolerance = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32 * 0.25;
    assert!(
        containment.all_finite && containment.active_count > 0,
        "wet bed filter containment readback was invalid: {containment:?}",
    );
    assert!(
        containment.max_radial_excess <= tolerance,
        "wet bed escaped radially through filter paper: tolerance={tolerance} containment={containment:?}",
    );
    assert!(
        containment.min_floor_clearance >= -tolerance,
        "wet bed escaped below filter apex: tolerance={tolerance} containment={containment:?}",
    );
}

#[test]
fn saturated_bed_particles_remain_mechanically_coupled() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let snapshot = readback_saturated_bed_motion_snapshot(&sim, &device, &queue);
    assert!(
        snapshot.all_finite && snapshot.active_count > 0,
        "saturated bed motion readback was invalid: {snapshot:?}",
    );
    assert!(
        snapshot.saturated_count > 8,
        "center pour did not create a saturated bed population: {snapshot:?}",
    );
    assert!(
        snapshot.saturated_mean_compression >= snapshot.mean_compression * 0.35,
        "saturated bed particles are lagging the deforming bed instead of moving with it: {snapshot:?}",
    );
}

#[test]
fn saturated_center_bed_particles_receive_bounded_motion() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_particle_states(&sim, &device, &queue);

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let impacted = readback_bed_particle_states(&sim, &device, &queue);

    let dx = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32;
    let active_thresh = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML * 0.1;
    let motion =
        bed_particle_motion_delta_snapshot(&settled, &impacted, active_thresh, 0.65, dx * 4.0);

    assert!(
        motion.all_finite && motion.active_count > 0,
        "saturated bed motion readback was invalid: {motion:?}",
    );
    assert!(
        motion.saturated_count > 8 && motion.tracked_count > 3,
        "center pour did not wet enough central bed particles for a coupling check: {motion:?}",
    );
    assert!(
        motion.mean_displacement > dx * 0.015,
        "saturated central bed particles did not receive measurable motion from the pour: {motion:?}",
    );
    assert!(
        motion.mean_downward_displacement > dx * 0.005,
        "saturated central bed particles did not move downward under center impact: {motion:?}",
    );
    assert!(
        motion.mean_displacement < sim.settings.bounds_size.y * 0.05
            && motion.max_displacement < sim.settings.bounds_size.y * 0.25,
        "saturated central bed particles moved implausibly far under center impact: {motion:?}",
    );
}

#[test]
fn water_mass_stable_after_pour_off() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..30 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let m0 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let m1 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

    let drift = (m1 - m0).abs() / m0.max(1e-6);
    assert!(
        drift < 0.02,
        "water mass drifted {:.2}% after pour-off (m0={m0}, m1={m1})",
        drift * 100.0
    );
}

#[test]
fn water_pool_stable_against_cup_floor() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..300 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let m0 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let m1 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

    let drift = (m1 - m0).abs() / m0.max(1e-6);
    assert!(
        drift < 0.02,
        "pooled water drifted {:.2}% after settle (m0={m0}, m1={m1})",
        drift * 100.0
    );
}

#[test]
fn water_j_stays_near_rest_after_cup_settle() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let volume = readback_water_particle_volume_snapshot(&sim, &device, &queue);

    assert!(
        volume.all_finite,
        "water J readback was non-finite: {volume:?}"
    );
    assert!(
        volume.active_count > 0,
        "free-stream cup settle produced no active water particles: {volume:?}",
    );
    assert!(
        volume.active_mass > 0.0 && volume.rest_volume > 0.0 && volume.current_volume > 0.0,
        "settled water volume readback was empty or negative: {volume:?}",
    );
    assert!(
        volume.min_j > 0.35,
        "settled water over-compressed relative to rest volume: {volume:?}",
    );
    assert!(
        volume.max_j < 2.5,
        "settled water over-expanded relative to rest volume: {volume:?}",
    );
    assert!(
        (volume.mean_j - 1.0).abs() < 0.35,
        "settled water mean J drifted too far from rest volume: {volume:?}",
    );
}

#[test]
fn water_only_settle_satisfies_realism_properties() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..30 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let pour_off = readback_water_diagnostics_snapshot(&sim, &device, &queue);

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..90 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_water_diagnostics_snapshot(&sim, &device, &queue);

    let mass_drift =
        (settled.active_mass - pour_off.active_mass).abs() / pour_off.active_mass.max(1e-6);
    let kinetic_ratio = settled.kinetic_energy / pour_off.kinetic_energy.max(1e-6);
    let vertical_rms_m_s = units::sim_speed_to_meters_per_second(settled.vertical_rms_speed);
    let surface_rms_m = settled.surface_rms_y * units::METERS_PER_SIM_UNIT;
    let surface_peak_to_peak_m = settled.surface_peak_to_peak_y * units::METERS_PER_SIM_UNIT;
    let surface_tilt_height_m = settled.surface_tilt_height_y * units::METERS_PER_SIM_UNIT;
    let residual_rms_m = settled.surface_residual_rms_y * units::METERS_PER_SIM_UNIT;

    assert!(
        pour_off.all_finite && settled.all_finite,
        "water-only diagnostics found non-finite particles: pour_off={pour_off:?} settled={settled:?}",
    );
    assert!(
        pour_off.active_count > 0 && settled.active_count > 0,
        "water-only realism check had no active water: pour_off={pour_off:?} settled={settled:?}",
    );
    assert!(
        mass_drift < 0.02,
        "water-only active mass drifted {:.2}% during settle: pour_off={pour_off:?} settled={settled:?}",
        mass_drift * 100.0,
    );
    assert!(
        kinetic_ratio < 0.05,
        "water-only kinetic energy did not decay after pour-off: ratio={kinetic_ratio:.3} \
         pour_off={pour_off:?} settled={settled:?}",
    );
    assert!(
        vertical_rms_m_s < 0.020,
        "settled water retained too much vertical motion: vertical_rms={vertical_rms_m_s:.4} m/s \
         settled={settled:?}",
    );
    assert!(
        settled.surface_bin_count > settled.surface_possible_bins / 2,
        "settled cup surface was too sparse for a levelness check: settled={settled:?}",
    );
    assert!(
        surface_rms_m < 0.006 && surface_peak_to_peak_m < 0.025,
        "settled water surface remained too rough: rms={:.2}mm peak_to_peak={:.2}mm \
         settled={settled:?}",
        surface_rms_m * 1000.0,
        surface_peak_to_peak_m * 1000.0,
    );
    assert!(
        surface_tilt_height_m < 0.012 && residual_rms_m < 0.006,
        "settled water surface stayed non-level: tilt_height={:.2}mm residual_rms={:.2}mm \
         settled={settled:?}",
        surface_tilt_height_m * 1000.0,
        residual_rms_m * 1000.0,
    );
    if settled.hydrostatic_sample_count > 0 && settled.hydrostatic_depth_m > 0.004 {
        let expected_gradient = 1_000.0 * units::STANDARD_GRAVITY_M_S2;
        let gradient_error = (settled.hydrostatic_gradient_pa_per_m - expected_gradient).abs()
            / expected_gradient.max(1e-6);
        assert!(
            settled.hydrostatic_bottom_higher && gradient_error < 0.08,
            "hydrostatic pressure ordering was implausible: expected_gradient={expected_gradient:.1} \
             gradient_error={gradient_error:.3} settled={settled:?}",
        );
    }
}

#[test]
fn pooled_water_particle_volume_stable_after_pour_off() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_water_particle_volume_snapshot(&sim, &device, &queue);

    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let late = readback_water_particle_volume_snapshot(&sim, &device, &queue);

    let current_volume_drift =
        (late.current_volume - settled.current_volume).abs() / settled.current_volume.max(1e-6);
    let rest_volume_drift =
        (late.rest_volume - settled.rest_volume).abs() / settled.rest_volume.max(1e-6);
    let mean_j_drift = (late.mean_j - settled.mean_j).abs();

    assert!(
        settled.all_finite && late.all_finite,
        "pooled water produced non-finite particle volume state: settled={settled:?} late={late:?}",
    );
    assert!(
        settled.active_count > 0 && late.active_count > 0,
        "pooled water volume readback had no active particles: settled={settled:?} late={late:?}",
    );
    assert!(
        current_volume_drift < 0.12,
        "pooled water current volume drifted {:.2}% after pour-off: settled={settled:?} late={late:?}",
        current_volume_drift * 100.0,
    );
    assert!(
        rest_volume_drift < 0.02,
        "pooled water rest volume drifted {:.2}% after pour-off: settled={settled:?} late={late:?}",
        rest_volume_drift * 100.0,
    );
    assert!(
        mean_j_drift < 0.2,
        "pooled water mean J drifted after pour-off: settled={settled:?} late={late:?}",
    );
}

#[test]
fn first_stage_grid_volume_packing_stays_bounded_after_pour_off() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_water_grid_packing_snapshot(&sim, &device, &queue);

    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let late = readback_water_grid_packing_snapshot(&sim, &device, &queue);

    assert!(
        settled.all_finite && late.all_finite,
        "pooled water produced non-finite grid packing state: settled={settled:?} late={late:?}",
    );
    assert!(
        settled.active_count > 0 && late.active_count > 0,
        "grid packing readback had no active water: settled={settled:?} late={late:?}",
    );
    assert_eq!(
        settled.active_count, late.active_count,
        "pooled water changed active particle count while checking grid packing: settled={settled:?} late={late:?}",
    );
    assert!(
        settled.deposited_rest_volume > 0.0
            && settled.deposited_current_volume > 0.0
            && late.deposited_rest_volume > 0.0
            && late.deposited_current_volume > 0.0,
        "grid packing deposited no volume: settled={settled:?} late={late:?}",
    );
    assert!(
        settled.max_rest_fraction > 0.0
            && settled.max_current_fraction > 0.0
            && late.max_rest_fraction > 0.0
            && late.max_current_fraction > 0.0,
        "grid packing fractions were empty: settled={settled:?} late={late:?}",
    );
    assert!(
        settled.overpacked_cell_count > 0 && settled.overpacked_volume_fraction > 0.0,
        "grid packing check did not exercise overpacked cells: settled={settled:?} late={late:?}",
    );
    assert!(
        late.max_packed_fraction < 1.75
            && late.max_packed_fraction <= settled.max_packed_fraction * 1.10,
        "first-stage grid packing grew beyond bounded headroom after pour-off: settled={settled:?} late={late:?}",
    );
    assert!(
        late.max_current_fraction <= settled.max_current_fraction * 1.10
            && late.max_rest_fraction <= settled.max_rest_fraction * 1.10,
        "first-stage grid peak volume fractions grew beyond bounded headroom after pour-off: settled={settled:?} late={late:?}",
    );
}

#[test]
fn pressure_projection_residual_metrics_track_post_projection_cells() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut settings = MpmSettings::benchmark_free_stream();
    settings.pressure_rbgs_pairs = 12;
    let mut sim = MpmSim3D::new(&device, &queue, settings);
    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..12 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let metrics = readback_metrics_data(&sim, &device, &queue);
    let fluid_cells = metrics[state::METRIC_FLUID_CELLS_IDX];
    let residual_cells = metrics[state::METRIC_PROJECTION_RESIDUAL_CELLS_IDX];
    let residual_max =
        metrics[state::METRIC_PROJECTION_RESIDUAL_MAX_IDX] as f32 / state::METRICS_DIV_FP_SCALE;
    let residual_mean = metrics[state::METRIC_PROJECTION_RESIDUAL_SUM_IDX] as f32
        / state::METRICS_RESIDUAL_SUM_FP_SCALE
        / (residual_cells as f32).max(1.0);

    assert!(
        fluid_cells > 0 && residual_cells > 0,
        "projection residual metrics did not see active fluid cells: metrics={metrics:?}",
    );
    assert!(
        residual_cells <= fluid_cells,
        "post-projection residual cells should be bounded by classified fluid cells: \
         fluid_cells={fluid_cells} residual_cells={residual_cells} metrics={metrics:?}",
    );
    assert!(
        residual_max.is_finite() && residual_mean.is_finite() && residual_mean <= residual_max,
        "projection residual metrics were inconsistent: max={residual_max:.4} \
         mean={residual_mean:.4} metrics={metrics:?}",
    );
}

#[test]
fn fractional_free_surface_pressure_preserves_sparse_stream_velocity() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pressure_rbgs_pairs: u32,
    ) -> (WaterVelocitySnapshot, WaterGridPackingSnapshot) {
        let mut settings = MpmSettings::benchmark_free_stream();
        settings.viscosity = 0.0;
        settings.pressure_rbgs_pairs = pressure_rbgs_pairs;
        settings.spout.nozzle_radius = 0.45;
        settings.spout.max_flow_rate_ml_s = 1.2;
        settings.spout.origin = Vec3::new(0.0, 4.2, 0.0);
        let mut sim = MpmSim3D::new(device, queue, settings);

        sim.set_exit_speed_m_s(DEFAULT_BREW.gentle_pour_exit_speed_m_s);
        for _ in 0..20 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        (
            readback_water_velocity_snapshot(&sim, device, queue),
            readback_water_grid_packing_snapshot(&sim, device, queue),
        )
    }

    let (unprojected_velocity, unprojected_packing) = run_case(&device, &queue, 0);
    let (projected_velocity, projected_packing) = run_case(&device, &queue, 40);
    let rms_ratio = projected_velocity.rms_speed / unprojected_velocity.rms_speed.max(1e-6);
    let mean_ratio = projected_velocity.mean_speed / unprojected_velocity.mean_speed.max(1e-6);
    let lateral_growth =
        projected_velocity.lateral_rms_speed / unprojected_velocity.lateral_rms_speed.max(1e-6);
    let mass_drift = (projected_velocity.active_mass - unprojected_velocity.active_mass).abs()
        / unprojected_velocity.active_mass.max(1e-6);

    assert!(
        unprojected_velocity.all_finite
            && projected_velocity.all_finite
            && unprojected_packing.all_finite
            && projected_packing.all_finite,
        "free-surface pressure comparison produced non-finite state: \
         unprojected_velocity={unprojected_velocity:?} projected_velocity={projected_velocity:?} \
         unprojected_packing={unprojected_packing:?} projected_packing={projected_packing:?}",
    );
    assert!(
        projected_velocity.active_count > 0 && unprojected_velocity.active_count > 0,
        "free-surface pressure comparison produced no active water: \
         unprojected_velocity={unprojected_velocity:?} projected_velocity={projected_velocity:?}",
    );
    assert!(
        mass_drift < 0.02,
        "pressure projection changed sparse-stream active mass: drift={:.2}% \
         unprojected_velocity={unprojected_velocity:?} projected_velocity={projected_velocity:?}",
        mass_drift * 100.0,
    );
    assert!(
        projected_packing.fractional_cell_count > projected_velocity.active_count,
        "test did not exercise a fractional free surface: \
         unprojected_packing={unprojected_packing:?} projected_packing={projected_packing:?}",
    );
    assert!(
        projected_packing.max_fractional_fraction > 0.20,
        "fractional free-surface occupancy was too weak for a pressure-weight regression: \
         projected_packing={projected_packing:?}",
    );
    assert!(
        (0.94..=1.08).contains(&rms_ratio) && (0.90..=1.20).contains(&mean_ratio),
        "fractional free-surface pressure should not materially change sparse-stream speed: \
         rms_ratio={rms_ratio:.3} mean_ratio={mean_ratio:.3} \
         unprojected_velocity={unprojected_velocity:?} projected_velocity={projected_velocity:?} \
         unprojected_packing={unprojected_packing:?} projected_packing={projected_packing:?}",
    );
    assert!(
        lateral_growth < 1.15,
        "fractional free-surface pressure injected lateral sparse-stream motion: \
         lateral_growth={lateral_growth:.3} unprojected_velocity={unprojected_velocity:?} \
         projected_velocity={projected_velocity:?} projected_packing={projected_packing:?}",
    );
}

#[test]
fn pooled_water_keeps_multilayer_depth_after_long_settle() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());
    let dx = sim.settings.bounds_size.x / sim.settings.grid_dims[0] as f32;

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_diag_snapshot(&sim, &device, &queue);

    for _ in 0..600 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let late = readback_diag_snapshot(&sim, &device, &queue);

    let mass_drift = (late.total_mass - settled.total_mass).abs() / settled.total_mass.max(1e-6);
    assert!(
        settled.all_finite && late.all_finite,
        "pooled water produced non-finite state after long settle: settled={settled:?} late={late:?}",
    );
    assert_eq!(
        settled.active_count, late.active_count,
        "pooled water changed active particle count after long settle: settled={settled:?} late={late:?}",
    );
    assert!(
        mass_drift < 0.01,
        "pooled water mass drifted after long settle: drift={:.2}% settled={settled:?} late={late:?}",
        mass_drift * 100.0,
    );
    assert!(
        late.y_extent >= dx * 4.0,
        "pooled water collapsed below a four-cell occupied depth after long settle: dx={dx} settled={settled:?} late={late:?}",
    );
    assert!(
        late.mean_j < 1.45,
        "pooled water represented long-settle support mostly as particle expansion: settled={settled:?} late={late:?}",
    );
}

#[test]
fn pooled_water_kinetic_energy_decays_after_pour_off() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    // Let the cone outlet transient detach before checking cup-level decay;
    // otherwise the test measures the last falling sheet, not pooled water.
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_water_velocity_snapshot(&sim, &device, &queue);

    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let late = readback_water_velocity_snapshot(&sim, &device, &queue);

    let kinetic_ratio = late.kinetic_energy / settled.kinetic_energy.max(1e-6);
    let rms_ratio = late.rms_speed / settled.rms_speed.max(1e-6);
    let momentum0 = settled.momentum[0]
        .hypot(settled.momentum[1])
        .hypot(settled.momentum[2]);
    let momentum1 = late.momentum[0]
        .hypot(late.momentum[1])
        .hypot(late.momentum[2]);
    let momentum_ratio = momentum1 / momentum0.max(1e-6);

    assert!(
        settled.all_finite && late.all_finite,
        "pooled water produced non-finite velocity state: settled={settled:?} late={late:?}",
    );
    assert!(
        settled.active_count > 0 && late.active_count > 0,
        "pooled water velocity readback had no active particles: settled={settled:?} late={late:?}",
    );
    assert_eq!(
        settled.active_count, late.active_count,
        "pooled water changed active particle count while checking velocity decay: settled={settled:?} late={late:?}",
    );
    assert!(
        (late.active_mass - settled.active_mass).abs() / settled.active_mass.max(1e-6) < 0.02,
        "pooled water mass drifted while checking velocity decay: settled={settled:?} late={late:?}",
    );
    assert!(
        kinetic_ratio < 1.05,
        "pooled water gained kinetic energy after pour-off: ratio={kinetic_ratio:.3} settled={settled:?} late={late:?}",
    );
    assert!(
        rms_ratio < 1.02,
        "pooled water RMS speed increased after pour-off: ratio={rms_ratio:.3} settled={settled:?} late={late:?}",
    );
    assert!(
        momentum_ratio < 1.20,
        "pooled water net momentum grew after pour-off: ratio={momentum_ratio:.3} settled={settled:?} late={late:?}",
    );
}

#[test]
fn higher_viscosity_damps_pooled_water_kinetic_energy() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viscosity: f32,
    ) -> WaterVelocitySnapshot {
        let mut settings = MpmSettings::benchmark_free_stream();
        settings.viscosity = viscosity;
        let mut sim = MpmSim3D::new(device, queue, settings);

        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for _ in 0..180 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        sim.set_exit_speed_m_s(0.0);
        for _ in 0..300 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        readback_water_velocity_snapshot(&sim, device, queue)
    }

    let inviscid = run_case(&device, &queue, 0.0);
    let viscous = run_case(&device, &queue, 1.2);

    assert!(
        inviscid.all_finite && viscous.all_finite,
        "viscosity comparison produced non-finite velocity state: inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        inviscid.active_count > 0 && viscous.active_count > 0,
        "viscosity comparison had no active water: inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        (viscous.active_mass - inviscid.active_mass).abs() / inviscid.active_mass.max(1e-6) < 0.02,
        "viscosity changed active water mass unexpectedly: inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        viscous.kinetic_energy < inviscid.kinetic_energy * 0.90,
        "higher viscosity should lower pooled-water kinetic energy: inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        viscous.rms_speed < inviscid.rms_speed * 0.95,
        "higher viscosity should lower pooled-water RMS speed: inviscid={inviscid:?} viscous={viscous:?}",
    );
}

#[test]
fn viscosity_preserves_falling_stream_velocity() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        viscosity: f32,
    ) -> WaterVelocitySnapshot {
        let mut settings = MpmSettings::benchmark_free_stream();
        settings.viscosity = viscosity;
        settings.spout.origin = Vec3::new(0.0, 4.2, 0.0);
        let mut sim = MpmSim3D::new(device, queue, settings);

        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for _ in 0..45 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        readback_water_velocity_snapshot(&sim, device, queue)
    }

    let inviscid = run_case(&device, &queue, 0.0);
    let viscous = run_case(&device, &queue, 1.2);
    let rms_ratio = viscous.rms_speed / inviscid.rms_speed.max(1e-6);
    let mean_ratio = viscous.mean_speed / inviscid.mean_speed.max(1e-6);

    assert!(
        inviscid.all_finite && viscous.all_finite,
        "falling stream viscosity comparison produced non-finite velocity state: inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        inviscid.active_count > 0 && viscous.active_count > 0,
        "falling stream comparison had no active water: inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        (viscous.active_mass - inviscid.active_mass).abs() / inviscid.active_mass.max(1e-6) < 0.05,
        "viscosity changed falling stream active water mass unexpectedly: inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        rms_ratio > 0.94 && rms_ratio < 1.06,
        "viscosity should not damp sparse falling stream RMS speed: ratio={rms_ratio:.3} inviscid={inviscid:?} viscous={viscous:?}",
    );
    assert!(
        mean_ratio > 0.94 && mean_ratio < 1.06,
        "viscosity should not damp sparse falling stream mean speed: ratio={mean_ratio:.3} inviscid={inviscid:?} viscous={viscous:?}",
    );
}

#[test]
fn slow_spout_translation_does_not_whip_free_stream() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        translate_spout: bool,
    ) -> WaterVelocitySnapshot {
        let mut settings = MpmSettings::benchmark_free_stream();
        settings.spout.origin = Vec3::new(0.0, 4.2, 0.0);
        let mut sim = MpmSim3D::new(device, queue, settings);

        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for frame in 0..90 {
            if translate_spout {
                let t = (frame as f32 + 1.0) / 90.0;
                sim.set_spout_position(-0.3 * t, 4.2, 0.0);
            }
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        readback_water_velocity_snapshot(&sim, device, queue)
    }

    let stationary = run_case(&device, &queue, false);
    let translated = run_case(&device, &queue, true);
    let stationary_lateral_ratio = stationary.lateral_rms_speed / stationary.rms_speed.max(1e-6);
    let translated_lateral_ratio = translated.lateral_rms_speed / translated.rms_speed.max(1e-6);
    let mean_vx = translated.momentum[0] / translated.active_mass.max(1e-6);

    assert!(
        stationary.all_finite && translated.all_finite && translated.active_count > 0,
        "translated free stream produced invalid velocity state: stationary={stationary:?} translated={translated:?}",
    );
    assert!(
        translated_lateral_ratio <= stationary_lateral_ratio * 1.15 + 0.02,
        "slow spout translation amplified lateral stream motion: stationary_ratio={stationary_lateral_ratio:.3} translated_ratio={translated_lateral_ratio:.3} stationary={stationary:?} translated={translated:?}",
    );
    assert!(
        mean_vx.abs() < translated.rms_speed * 0.12,
        "slow spout translation injected excessive net x momentum relative to stream speed: \
         mean_vx={mean_vx:.3} translated={translated:?}",
    );
}

#[test]
fn slow_spout_translation_does_not_whip_post_bed_stream() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        translate_spout: bool,
    ) -> WaterVelocitySnapshot {
        let mut sim = MpmSim3D::new(device, queue, MpmSettings::benchmark_center_pour());
        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for frame in 0..150 {
            if translate_spout {
                let t = (frame as f32 + 1.0) / 150.0;
                sim.set_spout_position(-0.3 * t, 7.1, 0.0);
            }
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        // Region between the filter exit and the main cup pool. This catches
        // stream behavior downstream of the bed without letting the whole cup
        // dominate the velocity statistic.
        readback_water_velocity_snapshot_in_y_range(&sim, device, queue, -6.2, -3.35)
    }

    let stationary = run_case(&device, &queue, false);
    let translated = run_case(&device, &queue, true);
    let stationary_lateral_ratio = stationary.lateral_rms_speed / stationary.rms_speed.max(1e-6);
    let translated_lateral_ratio = translated.lateral_rms_speed / translated.rms_speed.max(1e-6);

    assert!(
        stationary.all_finite && translated.all_finite,
        "post-bed stream produced invalid velocity state: stationary={stationary:?} translated={translated:?}",
    );
    if stationary.active_count <= 20 || translated.active_count <= 20 {
        assert!(
            stationary.active_count <= 20 && translated.active_count <= 20,
            "slow spout translation changed whether water exited the bed window: stationary={stationary:?} translated={translated:?}",
        );
        return;
    }
    assert!(
        stationary.active_count > 20 && translated.active_count > 20,
        "post-bed stream readback did not capture enough water particles: stationary={stationary:?} translated={translated:?}",
    );
    assert!(
        translated_lateral_ratio <= (stationary_lateral_ratio * 1.30 + 0.06).max(0.16),
        "slow spout translation amplified post-bed lateral stream motion: stationary_ratio={stationary_lateral_ratio:.3} translated_ratio={translated_lateral_ratio:.3} stationary={stationary:?} translated={translated:?}",
    );
}

#[test]
fn coffee_bed_slows_post_bed_downward_flow() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        with_bed: bool,
    ) -> WaterVelocitySnapshot {
        let settings = if with_bed {
            MpmSettings::benchmark_center_pour()
        } else {
            let mut settings = MpmSettings::benchmark_center_pour();
            settings.bed = None;
            settings
        };
        let mut sim = MpmSim3D::new(device, queue, settings);
        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for _ in 0..150 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        readback_water_velocity_snapshot_in_y_range(&sim, device, queue, -6.2, -3.35)
    }

    let open_filter = run_case(&device, &queue, false);
    let coffee_bed = run_case(&device, &queue, true);
    let open_downward_speed =
        (-open_filter.momentum[1] / open_filter.active_mass.max(1e-6)).max(0.0);
    let bed_downward_speed = (-coffee_bed.momentum[1] / coffee_bed.active_mass.max(1e-6)).max(0.0);

    assert!(
        open_filter.all_finite && coffee_bed.all_finite,
        "post-bed velocity readback was invalid: open_filter={open_filter:?} coffee_bed={coffee_bed:?}",
    );
    assert!(
        open_filter.active_count > 20,
        "open-filter post-bed velocity readback did not capture enough water: open_filter={open_filter:?}",
    );
    let throughput_ratio = coffee_bed.active_mass / open_filter.active_mass.max(1e-6);
    assert!(
        throughput_ratio < 0.35 || bed_downward_speed < open_downward_speed * 0.85,
        "coffee bed should either throttle downstream water or slow what exits: \
         throughput_ratio={throughput_ratio:.3} open_downward_speed={open_downward_speed:.3} \
         bed_downward_speed={bed_downward_speed:.3} open_filter={open_filter:?} coffee_bed={coffee_bed:?}",
    );
}

#[test]
fn coffee_bed_builds_visible_water_above_surface() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        with_bed: bool,
    ) -> WaterVelocitySnapshot {
        let (_, bed_top_y) = benchmark_bed_bounds_y();
        let mut settings = MpmSettings::benchmark_center_pour();
        if !with_bed {
            settings.bed = None;
        }
        let mut sim = MpmSim3D::new(device, queue, settings);

        sim.set_exit_speed_m_s(0.0);
        for _ in 0..60 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for _ in 0..180 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        readback_water_velocity_snapshot_in_y_range(
            &sim,
            device,
            queue,
            bed_top_y - 0.05,
            bed_top_y + 0.60,
        )
    }

    let open_filter = run_case(&device, &queue, false);
    let coffee_bed = run_case(&device, &queue, true);
    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;

    assert!(
        open_filter.all_finite && coffee_bed.all_finite,
        "surface-band water readback was invalid: open_filter={open_filter:?} coffee_bed={coffee_bed:?}",
    );
    assert!(
        coffee_bed.active_count >= open_filter.active_count + 24
            && coffee_bed.active_mass >= open_filter.active_mass * 1.5 + nominal_mass * 12.0,
        "coffee bed did not build visibly more active water just above the surface: \
         open_filter={open_filter:?} coffee_bed={coffee_bed:?}",
    );
}

#[test]
fn faster_pour_builds_more_water_above_coffee_bed() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        water_speed_m_s: f32,
    ) -> (WaterVelocitySnapshot, f32, f32) {
        let (_, bed_top_y) = benchmark_bed_bounds_y();
        let mut sim = MpmSim3D::new(device, queue, MpmSettings::benchmark_center_pour());

        sim.set_exit_speed_m_s(0.0);
        for _ in 0..60 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        sim.set_exit_speed_m_s(water_speed_m_s);
        for _ in 0..180 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        let flow_rate = sim.flow_rate_ml_s();
        let emitted_mass = sim.total_emitted_mass();
        let above_surface = readback_water_velocity_snapshot_in_y_range(
            &sim,
            device,
            queue,
            bed_top_y - 0.10,
            bed_top_y + 1.10,
        );
        (above_surface, flow_rate, emitted_mass)
    }

    let (slow_above, slow_flow, slow_emitted) =
        run_case(&device, &queue, DEFAULT_BREW.gentle_pour_exit_speed_m_s);
    let (fast_above, fast_flow, fast_emitted) =
        run_case(&device, &queue, DEFAULT_BREW.high_pour_exit_speed_m_s);
    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
    let slow_surface_fraction = slow_above.active_mass / slow_emitted.max(1e-6);
    let fast_surface_fraction = fast_above.active_mass / fast_emitted.max(1e-6);

    assert!(
        slow_above.all_finite && fast_above.all_finite,
        "rate comparison produced invalid water readback: slow={slow_above:?} fast={fast_above:?}",
    );
    assert!(
        fast_flow >= slow_flow * 2.5 && fast_emitted >= slow_emitted * 2.5,
        "test did not create distinct slow and fast pours: \
         slow_flow={slow_flow:.3} fast_flow={fast_flow:.3} \
         slow_emitted={slow_emitted:.3} fast_emitted={fast_emitted:.3}",
    );
    assert!(
        fast_above.active_count >= slow_above.active_count + 48
            && fast_above.active_mass >= slow_above.active_mass * 2.0 + nominal_mass * 24.0,
        "faster pour should build a visibly larger water pool above the coffee bed: \
         slow_flow={slow_flow:.3} fast_flow={fast_flow:.3} \
         slow_fraction={slow_surface_fraction:.3} fast_fraction={fast_surface_fraction:.3} \
         slow={slow_above:?} fast={fast_above:?}",
    );
    assert!(
        fast_surface_fraction >= slow_surface_fraction * 0.55,
        "faster pour should not only increase total emitted mass; it should retain a comparable \
         share of water above the bed: slow_fraction={slow_surface_fraction:.3} \
         fast_fraction={fast_surface_fraction:.3} slow={slow_above:?} fast={fast_above:?}",
    );
}

#[test]
fn fine_grind_pools_more_than_coarse_grind() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn settings_with_grind(grind_diameter_um: f32) -> MpmSettings {
        let mut settings = MpmSettings::benchmark_center_pour();
        let bed = settings.bed.as_mut().expect("benchmark scene has a bed");
        bed.initial_permeability = super::brew_config::kozeny_carman_permeability_m2(
            grind_diameter_um,
            bed.initial_porosity,
        );
        settings
    }

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grind_diameter_um: f32,
    ) -> (WaterVelocitySnapshot, WaterVelocitySnapshot) {
        let (bed_bot_y, bed_top_y) = benchmark_bed_bounds_y();
        let mut sim = MpmSim3D::new(device, queue, settings_with_grind(grind_diameter_um));

        sim.set_exit_speed_m_s(0.0);
        for _ in 0..60 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for _ in 0..210 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        let above_surface = readback_water_velocity_snapshot_in_y_range(
            &sim,
            device,
            queue,
            bed_top_y - 0.10,
            bed_top_y + 0.75,
        );
        let below_bed = readback_water_velocity_snapshot_in_y_range(
            &sim,
            device,
            queue,
            bed_bot_y - 3.60,
            bed_bot_y - 0.25,
        );
        (above_surface, below_bed)
    }

    let (fine_above, fine_below) = run_case(&device, &queue, 350.0);
    let (coarse_above, coarse_below) = run_case(&device, &queue, 1_100.0);
    let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;

    assert!(
        fine_above.all_finite
            && fine_below.all_finite
            && coarse_above.all_finite
            && coarse_below.all_finite,
        "grind comparison produced invalid water readback: \
         fine_above={fine_above:?} fine_below={fine_below:?} \
         coarse_above={coarse_above:?} coarse_below={coarse_below:?}",
    );
    assert!(
        fine_above.active_mass >= coarse_above.active_mass * 1.15 + nominal_mass * 8.0
            && fine_above.active_count >= coarse_above.active_count + 12,
        "fine grind should retain more active water above the bed surface: \
         fine_above={fine_above:?} coarse_above={coarse_above:?}",
    );
    let fine_downward_flux = (-fine_below.momentum[1]).max(0.0);
    let coarse_downward_flux = (-coarse_below.momentum[1]).max(0.0);
    assert!(
        fine_downward_flux <= coarse_downward_flux * 0.85
            && (fine_below.rms_speed <= coarse_below.rms_speed * 0.88
                || fine_below.active_mass <= coarse_below.active_mass * 0.85),
        "fine grind should reduce below-bed downward flux, not merely slow water so it lingers \
         in the readback band: fine_flux={fine_downward_flux:.3} \
         coarse_flux={coarse_downward_flux:.3} fine_below={fine_below:?} \
         coarse_below={coarse_below:?}",
    );
}

#[test]
fn coffee_bed_retains_water_above_bed_surface() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    fn run_case(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        with_bed: bool,
    ) -> WaterVelocitySnapshot {
        let settings = if with_bed {
            MpmSettings::benchmark_center_pour()
        } else {
            let mut settings = MpmSettings::benchmark_center_pour();
            settings.bed = None;
            settings
        };
        let mut sim = MpmSim3D::new(device, queue, settings);
        sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
        for _ in 0..240 {
            sim.step_frame(device, queue, 1.0 / 60.0);
        }

        readback_water_velocity_snapshot_in_y_range(&sim, device, queue, -0.9, 0.2)
    }

    let open_filter = run_case(&device, &queue, false);
    let coffee_bed = run_case(&device, &queue, true);

    assert!(
        open_filter.all_finite && coffee_bed.all_finite,
        "above-bed water readback was invalid: open_filter={open_filter:?} coffee_bed={coffee_bed:?}",
    );
    let open_near_surface_speed = open_filter.rms_speed;
    let bed_near_surface_speed = coffee_bed.rms_speed;

    assert!(
        coffee_bed.active_count > open_filter.active_count + 20
            && coffee_bed.active_mass > open_filter.active_mass * 1.5,
        "coffee bed did not retain a visible top-bed water population: open_filter={open_filter:?} coffee_bed={coffee_bed:?}",
    );
    assert!(
        bed_near_surface_speed < open_near_surface_speed * 0.65,
        "coffee bed should turn the fast falling stream into slower near-surface water: \
         open_near_surface_speed={open_near_surface_speed:.3} bed_near_surface_speed={bed_near_surface_speed:.3} \
         open_filter={open_filter:?} coffee_bed={coffee_bed:?}",
    );
}

#[test]
#[ignore = "property target: detects filter-paper sticking and side jets without blocking the default suite"]
fn center_pour_filter_contact_has_no_side_jets() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..210 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let contact = readback_filter_contact_snapshot(&sim, &device, &queue);
    assert!(
        contact.all_finite,
        "filter contact readback found non-finite water particles: {contact:?}",
    );
    assert!(
        contact.active_count > 0,
        "filter contact scenario had no active water: {contact:?}",
    );
    assert!(
        contact.outside_paper_fraction < 0.01 && contact.max_wall_penetration_m < 0.004,
        "water escaped outside the paper filter band: {contact:?}",
    );
    assert!(
        contact.outward_jet_fraction < 0.015 && contact.max_outward_speed_m_s < 0.18,
        "water formed a high-speed outward side jet near the filter paper: {contact:?}",
    );
    assert!(
        contact.tangential_sheet_fraction < 0.04 && contact.max_tangential_speed_m_s < 0.20,
        "water formed a fast tangential sheet along the filter paper: {contact:?}",
    );
}

#[test]
fn saturated_bed_extracts_solute_into_mobile_water() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping GPU test: no suitable adapter/device");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::debug_uniform_bed_saturation());
    sim.seed_uniform_bed_saturation(&queue);

    let initial_bed = readback_bed_extract_data(&sim, &device, &queue, "initial extraction bed");
    let initial_extractable: f32 = (0..sim.num_bed as usize)
        .map(|i| initial_bed[i * 8 + 4] + initial_bed[i * 8 + 6])
        .sum();

    for _ in 0..36 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let diagnostics = readback_water_diagnostics_snapshot(&sim, &device, &queue);
    let final_bed = readback_bed_extract_data(&sim, &device, &queue, "final extraction bed");
    let mut final_fast = 0.0_f32;
    let mut final_dissolved = 0.0_f32;
    let mut final_slow = 0.0_f32;
    let mut all_finite = diagnostics.all_finite;
    for i in 0..sim.num_bed as usize {
        let fast = final_bed[i * 8 + 4];
        let dissolved = final_bed[i * 8 + 5];
        let slow = final_bed[i * 8 + 6];
        all_finite &= fast.is_finite() && dissolved.is_finite() && slow.is_finite();
        final_fast += fast;
        final_dissolved += dissolved;
        final_slow += slow;
    }

    let final_total_soluble =
        final_fast + final_dissolved + final_slow + diagnostics.dissolved_solute_mass;

    assert!(all_finite, "extraction state contained non-finite values");
    assert!(
        diagnostics.dissolved_solute_mass > 0.01,
        "wet mobile water should carry extracted solute: {diagnostics:?}"
    );
    assert!(
        final_fast + final_slow < initial_extractable,
        "bed extractable reservoirs should deplete once saturated"
    );
    assert!(
        final_total_soluble <= initial_extractable * 1.02,
        "solute should not grow materially: initial={initial_extractable:.4} \
         final={final_total_soluble:.4}"
    );
    assert!(
        final_total_soluble >= initial_extractable * 0.90,
        "solute should remain approximately conserved: initial={initial_extractable:.4} \
         final={final_total_soluble:.4}"
    );
}

#[test]
#[ignore = "property target: checks a seeded water block against filter-wall sticking and non-hydrostatic settling"]
fn filter_water_block_settles_without_wall_sheeting() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_filter_water_block());
    sim.seed_filter_water_block(&queue);

    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let initial = readback_water_diagnostics_snapshot(&sim, &device, &queue);

    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_water_diagnostics_snapshot(&sim, &device, &queue);
    let contact = readback_filter_contact_snapshot(&sim, &device, &queue);

    let mass_drift =
        (settled.active_mass - initial.active_mass).abs() / initial.active_mass.max(1e-6);
    let kinetic_ratio = settled.kinetic_energy / initial.kinetic_energy.max(1e-6);
    let vertical_rms_m_s = units::sim_speed_to_meters_per_second(settled.vertical_rms_speed);

    assert!(
        initial.all_finite && settled.all_finite && contact.all_finite,
        "filter water block produced non-finite state: initial={initial:?} \
         settled={settled:?} contact={contact:?}",
    );
    assert!(
        initial.active_count > 0 && settled.active_count > 0 && contact.active_count > 0,
        "filter water block had no active water: initial={initial:?} \
         settled={settled:?} contact={contact:?}",
    );
    assert!(
        mass_drift < 0.03,
        "filter water block mass drifted {:.2}% while settling: initial={initial:?} \
         settled={settled:?}",
        mass_drift * 100.0,
    );
    assert!(
        kinetic_ratio < 0.50 && vertical_rms_m_s < 0.035,
        "filter water block retained too much motion: kinetic_ratio={kinetic_ratio:.3} \
         vertical_rms={vertical_rms_m_s:.4}m/s initial={initial:?} settled={settled:?}",
    );
    assert!(
        contact.outside_paper_fraction < 0.01 && contact.max_wall_penetration_m < 0.004,
        "filter water block escaped outside the paper band: {contact:?}",
    );
    assert!(
        contact.outward_jet_fraction < 0.01 && contact.max_outward_speed_m_s < 0.12,
        "filter water block generated outward jets near the paper: {contact:?}",
    );
    assert!(
        contact.stuck_wall_fraction < 0.25,
        "filter water block left too much nearly stationary water stuck to the paper wall: {contact:?}",
    );
}

#[test]
#[ignore = "long-horizon target: tracks pooled-water shape collapse without blocking the default suite"]
fn pooled_water_shape_stays_bounded_after_initial_settle() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_exit_speed_m_s(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_diag_snapshot(&sim, &device, &queue);

    let shape_settle_frames = env_u32_or(SHAPE_SETTLE_FRAMES_ENV, DEFAULT_SHAPE_SETTLE_FRAMES);
    for _ in 0..shape_settle_frames {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let late = readback_diag_snapshot(&sim, &device, &queue);

    let settled_volume_proxy = settled.x_extent * settled.y_extent * settled.z_extent;
    let late_volume_proxy = late.x_extent * late.y_extent * late.z_extent;
    let volume_proxy_ratio = late_volume_proxy / settled_volume_proxy.max(1e-6);
    let height_ratio = late.y_extent / settled.y_extent.max(1e-6);

    assert!(
        settled.all_finite && late.all_finite,
        "pooled water produced non-finite state"
    );
    assert!(
        settled.active_count == late.active_count,
        "pooled water changed active particle count after settling: settled={settled:?}, late={late:?}",
    );
    assert!(
        height_ratio > 0.8,
        "pooled water height kept shrinking after {shape_settle_frames} extra settle frames: \
         settled={settled:?}, late={late:?}",
    );
    assert!(
        volume_proxy_ratio > 0.75,
        "pooled water occupied volume proxy kept shrinking after {shape_settle_frames} extra settle frames: \
         settled={settled:?}, late={late:?}",
    );
}

// ── Extended diagnostics ──

#[test]
#[ignore = "long-horizon diagnostic: run explicitly or via the scheduled workflow"]
fn volume_conservation_long_settle() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let settle_frames = env_u32_or(LONG_SETTLE_FRAMES_ENV, DEFAULT_LONG_SETTLE_FRAMES);
    let log_every_frames = env_u32_or(
        LONG_SETTLE_LOG_FRAMES_ENV,
        DEFAULT_LONG_SETTLE_LOG_EVERY_FRAMES,
    )
    .min(settle_frames)
    .max(1);
    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_exit_speed_m_s(DEFAULT_BREW.high_pour_exit_speed_m_s);
    for f in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
        if (f + 1) % 60 == 0 {
            let d = readback_diag_snapshot(&sim, &device, &queue);
            eprintln!(
                "[POUR  t={:5.1}s] particles={:5} mass={:8.3} y=[{:6.2},{:6.2}] ext={:5.2} J=[{:.3},{:.3}] mass_range=[{:.4},{:.4}]",
                sim.total_time, d.active_count, d.total_mass,
                d.y_min, d.y_max, d.y_extent,
                d.min_j, d.max_j, d.min_mass, d.max_mass,
            );
        }
    }

    sim.set_exit_speed_m_s(0.0);
    let d0 = readback_diag_snapshot(&sim, &device, &queue);
    eprintln!("\n=== POUR OFF at t={:.1}s ===", sim.total_time);
    eprintln!(
        "  baseline: particles={} mass={:.3} y_extent={:.2} mean_J={:.4}",
        d0.active_count, d0.total_mass, d0.y_extent, d0.mean_j,
    );

    for f in 0..settle_frames {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
        if (f + 1) % log_every_frames == 0 || f + 1 == settle_frames {
            let d = readback_diag_snapshot(&sim, &device, &queue);
            let mass_drift = (d.total_mass - d0.total_mass) / d0.total_mass.max(1e-6) * 100.0;
            let ext_drift = (d.y_extent - d0.y_extent) / d0.y_extent.max(1e-6) * 100.0;
            eprintln!(
                "[SETTLE t={:6.1}s] particles={:5} mass={:8.3} ({:+.2}%) y_ext={:5.2} ({:+.2}%) y_mean={:6.2} J=[{:.3},{:.3}]",
                sim.total_time, d.active_count, d.total_mass, mass_drift,
                d.y_extent, ext_drift, d.y_mean, d.min_j, d.max_j,
            );
        }
    }

    let d_final = readback_diag_snapshot(&sim, &device, &queue);
    let final_mass_drift = (d_final.total_mass - d0.total_mass).abs() / d0.total_mass.max(1e-6);
    assert!(
        final_mass_drift < 0.01,
        "mass drifted {:.2}% over {} settle frames ({:.1}s simulated, expected <1%)",
        final_mass_drift * 100.0,
        settle_frames,
        settle_frames as f32 / 60.0,
    );
}

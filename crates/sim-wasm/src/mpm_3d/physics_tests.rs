use super::*;
use std::sync::mpsc;

use bytemuck::cast_slice;

// ── Device setup ──

fn request_adapter() -> Option<wgpu::Adapter> {
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
    bed_held_mass: f32,
}

fn readback_mass_snapshot(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> MassSnapshot {
    let particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (particle_count * 32).max(4) as u64;
    let bed_size = (sim.num_bed as usize * 80).max(4) as u64;

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
    for i in 0..particle_count {
        active_particle_mass += particle_f32[i * 8 + 7];
    }
    drop(particle_view);
    particle_staging.unmap();

    let bed_view = bed_slice.get_mapped_range();
    let bed_f32 = cast_slice::<u8, f32>(&bed_view);
    let mut bed_held_mass = 0.0;
    for i in 0..sim.num_bed as usize {
        bed_held_mass += bed_f32[i * 20];
    }
    drop(bed_view);
    bed_staging.unmap();

    MassSnapshot {
        active_particle_mass,
        bed_held_mass,
    }
}

#[derive(Debug)]
struct DiagSnapshot {
    all_finite: bool,
    total_mass: f32,
    active_count: u32,
    min_mass: f32,
    max_mass: f32,
    y_min: f32,
    y_max: f32,
    y_mean: f32,
    y_extent: f32,
    mean_j: f32,
    min_j: f32,
    max_j: f32,
}

fn readback_diag_snapshot_range(
    sim: &MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    particle_offset: usize,
    particle_count: usize,
) -> DiagSnapshot {
    let total_particle_count = (sim.num_water + sim.num_bed) as usize;
    let particle_size = (total_particle_count * 32).max(4) as u64;

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
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    let mut y_sum = 0.0_f32;
    let mut j_sum = 0.0_f32;
    let mut j_min = f32::MAX;
    let mut j_max = f32::MIN;

    let mut all_finite = true;
    let particle_end = particle_offset + particle_count;
    for i in particle_offset..particle_end {
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
        active_count += 1;
        total_mass += mass;
        if mass < min_mass {
            min_mass = mass;
        }
        if mass > max_mass {
            max_mass = mass;
        }
        if y < y_min {
            y_min = y;
        }
        if y > y_max {
            y_max = y;
        }
        y_sum += y;
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
        y_min: if active_count > 0 { y_min } else { 0.0 },
        y_max: if active_count > 0 { y_max } else { 0.0 },
        y_mean: y_sum / n,
        y_extent: if active_count > 0 { y_max - y_min } else { 0.0 },
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
    assert!(snapshot.bed_held_mass.is_finite());
    assert!(snapshot.active_particle_mass >= 0.0);
    assert!(snapshot.bed_held_mass >= 0.0);
}

#[test]
fn water_mass_stable_after_pour_off() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_kettle_angle(36.0);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_kettle_angle(0.0);
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

    sim.set_kettle_angle(36.0);
    for _ in 0..300 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_kettle_angle(0.0);
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

// ── Extended diagnostics ──

#[test]
fn volume_conservation_long_settle() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

    sim.set_kettle_angle(36.0);
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

    sim.set_kettle_angle(0.0);
    let d0 = readback_diag_snapshot(&sim, &device, &queue);
    eprintln!("\n=== POUR OFF at t={:.1}s ===", sim.total_time);
    eprintln!(
        "  baseline: particles={} mass={:.3} y_extent={:.2} mean_J={:.4}",
        d0.active_count, d0.total_mass, d0.y_extent, d0.mean_j,
    );

    for f in 0..7200 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
        if (f + 1) % 600 == 0 {
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
        "mass drifted {:.2}% over 120s settle (expected <1%)",
        final_mass_drift * 100.0,
    );
}

#[test]
fn bed_settling_stability() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_kettle_angle(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let snapshot = readback_bed_diag_snapshot(&sim, &device, &queue);
    assert!(
        snapshot.all_finite,
        "bed particles produced non-finite state: {snapshot:?}"
    );
    assert!(
        snapshot.active_count > 0,
        "expected active bed particles after settle"
    );
    assert!(
        snapshot.mean_j >= 0.9 && snapshot.mean_j <= 1.1,
        "bed mean J drifted outside settle band: {snapshot:?}",
    );
    assert!(
        snapshot.min_j > 0.6 && snapshot.max_j < 1.48,
        "bed J approached safety clamps during settle: {snapshot:?}",
    );
    assert!(
        snapshot.y_extent > 1.0,
        "bed y_extent collapsed unexpectedly: {snapshot:?}",
    );
}

#[test]
fn bed_long_run_creep_is_bounded_without_water() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_kettle_angle(0.0);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_diag_snapshot(&sim, &device, &queue);

    for _ in 0..480 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let later = readback_bed_diag_snapshot(&sim, &device, &queue);

    let mean_y_drift = (later.y_mean - settled.y_mean).abs();
    let extent_drift = (later.y_extent - settled.y_extent).abs();
    assert!(
        mean_y_drift < 0.22,
        "dry bed kept creeping in mean height after settling (settled={settled:?}, later={later:?})",
    );
    assert!(
        extent_drift < 0.28,
        "dry bed shape kept drifting after settling (settled={settled:?}, later={later:?})",
    );
}

#[test]
fn water_bed_mass_conservation() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_kettle_angle(36.0);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    sim.set_kettle_angle(0.0);
    let before = readback_mass_snapshot(&sim, &device, &queue);
    let total_before = before.active_particle_mass + before.bed_held_mass;

    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let after = readback_mass_snapshot(&sim, &device, &queue);
    let total_after = after.active_particle_mass + after.bed_held_mass;
    let drift = (total_after - total_before).abs() / total_before.max(1e-6);
    assert!(
        drift < 0.01,
        "combined water + bed-held mass drifted {:.2}% after pour-off (before={total_before}, after={total_after})",
        drift * 100.0,
    );
}

#[test]
fn bed_j_off_clamps() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_kettle_angle(36.0);
    for _ in 0..180 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let snapshot = readback_bed_diag_snapshot(&sim, &device, &queue);
    assert!(
        snapshot.all_finite,
        "bed particles produced non-finite state: {snapshot:?}"
    );
    assert!(
        snapshot.active_count > 0,
        "expected active bed particles during pour"
    );
    assert!(
        snapshot.min_j > 0.55 && snapshot.max_j < 1.45,
        "bed J hit safety rails under normal pour: {snapshot:?}",
    );
}

#[test]
fn finer_grind_keeps_more_free_water_than_coarser_grind() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut fine_settings = MpmSettings::benchmark_center_pour();
    if let Some(bed) = fine_settings.bed.as_mut() {
        bed.mean_grind_size = 0.78;
        bed.grind_size_spread = 0.38;
        bed.fines_fraction = 0.26;
        bed.initial_permeability = 0.0015;
    }

    let mut coarse_settings = MpmSettings::benchmark_center_pour();
    if let Some(bed) = coarse_settings.bed.as_mut() {
        bed.mean_grind_size = 1.35;
        bed.grind_size_spread = 0.22;
        bed.fines_fraction = 0.06;
        bed.initial_permeability = 0.0038;
    }

    let mut fine_sim = MpmSim3D::new(&device, &queue, fine_settings);
    let mut coarse_sim = MpmSim3D::new(&device, &queue, coarse_settings);
    fine_sim.set_kettle_angle(36.0);
    coarse_sim.set_kettle_angle(36.0);
    for _ in 0..120 {
        fine_sim.step_frame(&device, &queue, 1.0 / 60.0);
        coarse_sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    fine_sim.set_kettle_angle(0.0);
    coarse_sim.set_kettle_angle(0.0);
    for _ in 0..30 {
        fine_sim.step_frame(&device, &queue, 1.0 / 60.0);
        coarse_sim.step_frame(&device, &queue, 1.0 / 60.0);
    }

    let fine = readback_mass_snapshot(&fine_sim, &device, &queue);
    let coarse = readback_mass_snapshot(&coarse_sim, &device, &queue);
    assert!(
        fine.active_particle_mass > coarse.active_particle_mass * 1.005,
        "expected finer grind to slow uptake/drawdown (fine={fine:?}, coarse={coarse:?})",
    );
}

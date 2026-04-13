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
    encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &particle_staging, 0, particle_size);
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
        bed_held_mass += bed_f32[i * 8];
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
    x_min: f32,
    x_max: f32,
    x_extent: f32,
    y_min: f32,
    y_max: f32,
    y_mean: f32,
    y_extent: f32,
    z_min: f32,
    z_max: f32,
    z_extent: f32,
    mean_j: f32,
    min_j: f32,
    max_j: f32,
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
    let mut y_sum = 0.0_f32;
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

        all_finite &= x.is_finite()
            && y.is_finite()
            && z.is_finite()
            && j.is_finite()
            && mass.is_finite();

        if mass <= inactive_thresh {
            continue;
        }
        active_count += 1;
        total_mass += mass;
        if mass < min_mass { min_mass = mass; }
        if mass > max_mass { max_mass = mass; }
        if x < x_min { x_min = x; }
        if x > x_max { x_max = x; }
        if y < y_min { y_min = y; }
        if y > y_max { y_max = y; }
        if z < z_min { z_min = z; }
        if z > z_max { z_max = z; }
        y_sum += y;
        j_sum += j;
        if j < j_min { j_min = j; }
        if j > j_max { j_max = j; }
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
        x_extent: if active_count > 0 { x_max - x_min } else { 0.0 },
        y_min: if active_count > 0 { y_min } else { 0.0 },
        y_max: if active_count > 0 { y_max } else { 0.0 },
        y_mean: y_sum / n,
        y_extent: if active_count > 0 { y_max - y_min } else { 0.0 },
        z_min: if active_count > 0 { z_min } else { 0.0 },
        z_max: if active_count > 0 { z_max } else { 0.0 },
        z_extent: if active_count > 0 { z_max - z_min } else { 0.0 },
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
    readback_diag_snapshot_range(sim, device, queue, 0, (sim.num_water + sim.num_bed) as usize)
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
    sim.set_kettle_angle(0.0);
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
fn bed_first_water_impact_is_bounded() {
    let Some((device, queue)) = create_test_device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_center_pour());
    sim.set_kettle_angle(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_diag_snapshot(&sim, &device, &queue);

    sim.set_kettle_angle(36.0);
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
    sim.set_kettle_angle(0.0);
    for _ in 0..60 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let settled = readback_bed_diag_snapshot(&sim, &device, &queue);

    sim.set_kettle_angle(36.0);
    for _ in 0..120 {
        sim.step_frame(&device, &queue, 1.0 / 60.0);
    }
    let wet = readback_bed_diag_snapshot(&sim, &device, &queue);

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

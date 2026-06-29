//! Solver-agnostic physical-realism evals.
//!
//! These tests deliberately run through the modular `FrameSolver` seam. They
//! sample only shared metrics and render-facing particle buffers, so adding a
//! solver should mean registering it and letting these evals exercise it without
//! adding a solver-specific test path here.

use std::sync::mpsc;

use bytemuck::cast_slice;
use coffee_sim_core::Vec3;

use crate::solvers::base::{CommonMetrics, FrameContext, FrameSolver, SceneSpec, SolverId};
use crate::solvers::registry::build_solver;
use crate::ui::ParticleRenderSource;

const DT: f32 = 1.0 / 60.0;
const ACTIVE_PARK_Y: f32 = -1.0e5;
const MIN_FALL_DELTA: f32 = 0.01;

#[derive(Clone, Copy, Debug)]
struct ParticleSample {
    pos: Vec3,
    vel: Vec3,
    phase: u32,
}

#[derive(Clone, Debug)]
struct EvalCase {
    solver: SolverId,
    scene: SceneSpec,
}

impl EvalCase {
    fn label(&self) -> String {
        format!("{}::{:?}", self.solver.id(), self.scene)
    }
}

fn request_adapter() -> Option<wgpu::Adapter> {
    if std::env::var_os("COFFEE_SIM_SKIP_GPU_TESTS").is_some() {
        return None;
    }

    let instance = wgpu::Instance::default();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()
}

fn create_eval_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let adapter = request_adapter()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("coffee-sim solver-agnostic eval device"),
        required_features: wgpu::Features::empty(),
        required_limits: crate::solvers::mpm::required_limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .ok()
}

fn eval_cases() -> Vec<EvalCase> {
    SolverId::all()
        .iter()
        .copied()
        .flat_map(|solver| {
            [
                EvalCase {
                    solver,
                    scene: SceneSpec::CenterPour,
                },
                EvalCase {
                    solver,
                    scene: SceneSpec::FreeStream,
                },
            ]
        })
        .collect()
}

fn build_case(device: &wgpu::Device, queue: &wgpu::Queue, case: &EvalCase) -> Box<dyn FrameSolver> {
    build_solver(case.solver, device, queue, &case.scene)
        .unwrap_or_else(|err| panic!("{} failed to build: {err}", case.label()))
}

fn step_frames(
    solver: &mut dyn FrameSolver,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    frames: u32,
) -> CommonMetrics {
    let mut metrics = solver.snapshot().metrics;
    for _ in 0..frames {
        metrics = solver.step_frame(FrameContext {
            device,
            queue,
            dt: DT,
        });
    }
    metrics
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    bytes: u64,
) -> Vec<u8> {
    let bytes = bytes.max(4);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("solver eval readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("solver eval readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("eval readback callback");
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("eval readback recv")
        .expect("eval readback map");

    let data = slice.get_mapped_range().to_vec();
    staging.unmap();
    data
}

fn sample_particles(
    solver: &dyn FrameSolver,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<ParticleSample> {
    let snapshot = solver.snapshot();
    let count = snapshot.render.particle_count();
    if count == 0 {
        return Vec::new();
    }

    match snapshot.render.particle_source() {
        ParticleRenderSource::Packed { render_buffer } => {
            let bytes = (count * 8 * std::mem::size_of::<f32>()) as u64;
            let data = read_buffer(device, queue, render_buffer, bytes);
            let floats = cast_slice::<u8, f32>(&data);
            floats
                .chunks_exact(8)
                .filter_map(|p| {
                    let pos = Vec3::new(p[0], p[1], p[2]);
                    let color_t = p[3];
                    let radius = p[4];
                    if pos.y <= ACTIVE_PARK_Y || radius <= 0.0 || color_t <= -900.0 {
                        return None;
                    }
                    Some(ParticleSample {
                        pos,
                        vel: Vec3::ZERO,
                        phase: if color_t >= 0.0 { 0 } else { 1 },
                    })
                })
                .collect()
        }
        ParticleRenderSource::Canonical {
            positions,
            velocities,
            phases,
        } => {
            let vec4_bytes = (count * 4 * std::mem::size_of::<f32>()) as u64;
            let phase_bytes = (count * std::mem::size_of::<u32>()) as u64;
            let pos_data = read_buffer(device, queue, positions, vec4_bytes);
            let vel_data = read_buffer(device, queue, velocities, vec4_bytes);
            let phase_data = read_buffer(device, queue, phases, phase_bytes);
            let pos = cast_slice::<u8, f32>(&pos_data);
            let vel = cast_slice::<u8, f32>(&vel_data);
            let phase = cast_slice::<u8, u32>(&phase_data);

            (0..count)
                .filter_map(|i| {
                    let p = Vec3::new(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]);
                    if p.y <= ACTIVE_PARK_Y {
                        return None;
                    }
                    Some(ParticleSample {
                        pos: p,
                        vel: Vec3::new(vel[i * 4], vel[i * 4 + 1], vel[i * 4 + 2]),
                        phase: phase[i],
                    })
                })
                .collect()
        }
    }
}

fn assert_metrics_are_plausible(label: &str, metrics: CommonMetrics) {
    assert!(
        metrics.particle_count > 0,
        "{label}: solver reported no particles: {metrics:?}",
    );
    assert!(
        metrics.max_particles == 0 || metrics.particle_count <= metrics.max_particles as usize,
        "{label}: particle count exceeded capacity: {metrics:?}",
    );
    for (name, value) in [
        ("sim_time_s", metrics.sim_time_s),
        ("total_emitted_mass", metrics.total_emitted_mass),
        ("total_emitted_ml", metrics.total_emitted_ml),
        ("flow_rate_ml_s", metrics.flow_rate_ml_s),
        ("exit_speed", metrics.exit_speed),
        ("max_abs_divergence", metrics.max_abs_divergence),
        (
            "projection_residual_max_abs_divergence",
            metrics.projection_residual_max_abs_divergence,
        ),
        ("mean_tds", metrics.mean_tds),
        ("extraction_yield", metrics.extraction_yield),
    ] {
        assert!(
            value.is_finite(),
            "{label}: metric {name} was not finite: {metrics:?}",
        );
    }
    assert!(
        metrics.total_emitted_mass >= -1.0e-5 && metrics.total_emitted_ml >= -1.0e-5,
        "{label}: emitted water went negative: {metrics:?}",
    );
}

fn assert_samples_are_finite_and_bounded(
    label: &str,
    samples: &[ParticleSample],
    bounds_size: Vec3,
) {
    assert!(
        !samples.is_empty(),
        "{label}: no active particle samples were available",
    );
    let half = bounds_size * 0.5;
    let margin = Vec3::new(2.5, 3.5, 2.5);
    for (i, sample) in samples.iter().enumerate() {
        assert!(
            vec3_is_finite(sample.pos) && vec3_is_finite(sample.vel),
            "{label}: sample {i} had non-finite state: {sample:?}",
        );
        assert!(
            sample.pos.x >= -half.x - margin.x
                && sample.pos.x <= half.x + margin.x
                && sample.pos.y >= -half.y - margin.y
                && sample.pos.y <= half.y + margin.y
                && sample.pos.z >= -half.z - margin.z
                && sample.pos.z <= half.z + margin.z,
            "{label}: sample {i} escaped scene bounds {:?} with margin {:?}: {sample:?}",
            bounds_size,
            margin,
        );
    }
}

fn vec3_is_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn water_centroid_y(samples: &[ParticleSample]) -> Option<f32> {
    let mut sum = 0.0;
    let mut count = 0u32;
    for sample in samples.iter().filter(|sample| sample.phase == 0) {
        sum += sample.pos.y;
        count += 1;
    }
    (count > 0).then_some(sum / count as f32)
}

#[test]
fn eval_physical_state_is_finite_bounded_and_within_capacity() {
    let Some((device, queue)) = create_eval_device() else {
        eprintln!("no GPU adapter available; skipping solver-agnostic physical state eval");
        return;
    };

    for case in eval_cases() {
        let label = case.label();
        let mut solver = build_case(&device, &queue, &case);
        solver.set_water_velocity_m_s(0.12);
        let metrics = step_frames(solver.as_mut(), &device, &queue, 4);
        let snapshot = solver.snapshot();
        assert_metrics_are_plausible(&label, metrics);
        assert!(
            snapshot.render.render_radius().is_finite() && snapshot.render.render_radius() > 0.0,
            "{label}: invalid render radius {}",
            snapshot.render.render_radius(),
        );
        let samples = sample_particles(solver.as_ref(), &device, &queue);
        assert_samples_are_finite_and_bounded(&label, &samples, snapshot.render.bounds_size());
    }
}

#[test]
fn eval_center_pour_emits_water_monotonically_without_overflowing_capacity() {
    let Some((device, queue)) = create_eval_device() else {
        eprintln!("no GPU adapter available; skipping solver-agnostic emission eval");
        return;
    };

    for &solver_id in SolverId::all() {
        let case = EvalCase {
            solver: solver_id,
            scene: SceneSpec::CenterPour,
        };
        let label = case.label();
        let mut solver = build_case(&device, &queue, &case);
        solver.set_water_velocity_m_s(0.16);

        let before = solver.snapshot().metrics;
        let after = step_frames(solver.as_mut(), &device, &queue, 8);

        assert_metrics_are_plausible(&label, after);
        assert!(
            after.total_emitted_mass + 1.0e-5 >= before.total_emitted_mass,
            "{label}: total emitted mass regressed: before={before:?} after={after:?}",
        );
        assert!(
            after.total_emitted_ml + 1.0e-4 >= before.total_emitted_ml,
            "{label}: total emitted mL regressed: before={before:?} after={after:?}",
        );
        assert!(
            after.flow_rate_ml_s >= 0.0 && after.exit_speed_m_s >= 0.0,
            "{label}: negative flow/speed after pour: {after:?}",
        );
        assert!(
            after.max_particles == 0 || after.particle_count <= after.max_particles as usize,
            "{label}: center pour overflowed capacity: {after:?}",
        );
    }
}

#[test]
fn eval_free_stream_falls_after_inflow_stops() {
    let Some((device, queue)) = create_eval_device() else {
        eprintln!("no GPU adapter available; skipping solver-agnostic gravity eval");
        return;
    };

    for &solver_id in SolverId::all() {
        let case = EvalCase {
            solver: solver_id,
            scene: SceneSpec::FreeStream,
        };
        let label = case.label();
        let mut solver = build_case(&device, &queue, &case);

        solver.set_water_velocity_m_s(0.18);
        step_frames(solver.as_mut(), &device, &queue, 8);
        solver.set_water_velocity_m_s(0.0);

        let early = sample_particles(solver.as_ref(), &device, &queue);
        let Some(early_y) = water_centroid_y(&early) else {
            panic!("{label}: no water samples after warmup");
        };

        step_frames(solver.as_mut(), &device, &queue, 10);
        let late = sample_particles(solver.as_ref(), &device, &queue);
        let Some(late_y) = water_centroid_y(&late) else {
            panic!("{label}: no water samples after falling interval");
        };

        assert!(
            late_y < early_y - MIN_FALL_DELTA,
            "{label}: free-stream water did not fall after inflow stopped: early_y={early_y:.4} late_y={late_y:.4}",
        );
    }
}

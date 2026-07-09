//! Solver-agnostic physical-realism evals.
//!
//! The contract here is intentionally the same one the browser uses: build a
//! registered solver, step frames through `FrameSolver`, then inspect only
//! shared metrics and render-facing particle sources. No concrete solver type,
//! downcast, shader-private buffer, or solver-specific scene setup is allowed in
//! this module. A new solver should enter these evals by registering itself.

use std::sync::mpsc;

use bytemuck::cast_slice;
use coffee_sim_core::Vec3;

use crate::solvers::base::{CommonMetrics, FrameContext, FrameSolver, SceneSpec, SolverId};
use crate::solvers::registry::{build_solver, required_limits};
use crate::ui::{ParticleRenderSource, RenderView};

const FRAME_DT: f32 = 1.0 / 60.0;
const PARKED_PARTICLE_Y: f32 = -1.0e5;
const MIN_FREE_STREAM_FALL: f32 = 0.01;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvalScenario {
    CenterPour,
    FreeStream,
}

impl EvalScenario {
    const ALL: [Self; 2] = [Self::CenterPour, Self::FreeStream];

    fn scene_spec(self) -> SceneSpec {
        match self {
            Self::CenterPour => SceneSpec::CenterPour,
            Self::FreeStream => SceneSpec::FreeStream,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::CenterPour => "center-pour",
            Self::FreeStream => "free-stream",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SolverCase {
    solver: SolverId,
    scenario: EvalScenario,
}

impl SolverCase {
    fn all() -> Vec<Self> {
        SolverId::all()
            .iter()
            .copied()
            .flat_map(|solver| {
                EvalScenario::ALL
                    .iter()
                    .copied()
                    .map(move |scenario| Self { solver, scenario })
            })
            .collect()
    }

    fn label(self) -> String {
        format!("{}::{}", self.solver.id(), self.scenario.label())
    }
}

struct EvalDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl EvalDevice {
    fn new() -> Option<Self> {
        if std::env::var_os("COFFEE_SIM_SKIP_GPU_TESTS").is_some() {
            return None;
        }

        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("coffee-sim solver-agnostic eval device"),
            required_features: wgpu::Features::empty(),
            required_limits: required_limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()?;

        Some(Self { device, queue })
    }

    fn read_buffer(&self, source: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        let bytes = bytes.max(4);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solver eval readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("solver eval readback"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("eval readback callback");
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("eval readback recv")
            .expect("eval readback map");

        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    fn read_vec4_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<[f32; 4]> {
        let bytes = (count * 4 * std::mem::size_of::<f32>()) as u64;
        let data = self.read_buffer(buffer, bytes);
        cast_slice::<u8, f32>(&data)
            .chunks_exact(4)
            .map(|lane| [lane[0], lane[1], lane[2], lane[3]])
            .collect()
    }

    fn read_u32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
        let bytes = (count * std::mem::size_of::<u32>()) as u64;
        let data = self.read_buffer(buffer, bytes);
        cast_slice::<u8, u32>(&data).to_vec()
    }
}

struct SolverRun<'a> {
    case: SolverCase,
    solver: Box<dyn FrameSolver + 'a>,
}

impl<'a> SolverRun<'a> {
    fn build(eval: &EvalDevice, case: SolverCase) -> Self {
        let solver = build_solver(
            case.solver,
            &eval.device,
            &eval.queue,
            &case.scenario.scene_spec(),
        )
        .unwrap_or_else(|err| panic!("{} failed to build: {err}", case.label()));
        Self { case, solver }
    }

    fn label(&self) -> String {
        self.case.label()
    }

    fn set_flow_speed_m_s(&mut self, speed: f32) {
        self.solver.set_water_velocity_m_s(speed);
    }

    fn step(&mut self, eval: &EvalDevice, frames: u32) -> CommonMetrics {
        let mut metrics = self.solver.snapshot().metrics;
        for _ in 0..frames {
            metrics = self.solver.step_frame(FrameContext {
                device: &eval.device,
                queue: &eval.queue,
                dt: FRAME_DT,
            });
        }
        metrics
    }

    fn metrics(&self) -> CommonMetrics {
        self.solver.snapshot().metrics
    }

    fn particle_cloud(&self, eval: &EvalDevice) -> ParticleCloud {
        let snapshot = self.solver.snapshot();
        ParticleCloud::from_render_view(eval, &snapshot.render)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParticlePhase {
    Water,
    Solid,
}

#[derive(Clone, Copy, Debug)]
struct ParticleSample {
    position: Vec3,
    velocity: Vec3,
    phase: ParticlePhase,
}

impl ParticleSample {
    fn is_finite(self) -> bool {
        vec3_is_finite(self.position) && vec3_is_finite(self.velocity)
    }
}

#[derive(Debug)]
struct ParticleCloud {
    bounds_size: Vec3,
    render_radius: f32,
    samples: Vec<ParticleSample>,
}

impl ParticleCloud {
    fn from_render_view(eval: &EvalDevice, view: &RenderView<'_>) -> Self {
        let samples = match view.particle_source() {
            ParticleRenderSource::Packed { render_buffer } => {
                Self::read_packed_render_particles(eval, render_buffer, view.particle_count())
            }
            ParticleRenderSource::Canonical {
                positions,
                velocities,
                phases,
            } => Self::read_canonical_particles(
                eval,
                positions,
                velocities,
                phases,
                view.particle_count(),
            ),
        };

        Self {
            bounds_size: view.bounds_size(),
            render_radius: view.render_radius(),
            samples,
        }
    }

    fn read_packed_render_particles(
        eval: &EvalDevice,
        render_buffer: &wgpu::Buffer,
        count: usize,
    ) -> Vec<ParticleSample> {
        let bytes = (count * 8 * std::mem::size_of::<f32>()) as u64;
        let data = eval.read_buffer(render_buffer, bytes);
        cast_slice::<u8, f32>(&data)
            .chunks_exact(8)
            .filter_map(|p| {
                let position = Vec3::new(p[0], p[1], p[2]);
                let color_t = p[3];
                let radius = p[4];
                if position.y <= PARKED_PARTICLE_Y || radius <= 0.0 || color_t <= -900.0 {
                    return None;
                }
                Some(ParticleSample {
                    position,
                    velocity: Vec3::ZERO,
                    phase: if color_t >= 0.0 {
                        ParticlePhase::Water
                    } else {
                        ParticlePhase::Solid
                    },
                })
            })
            .collect()
    }

    fn read_canonical_particles(
        eval: &EvalDevice,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        phases: &wgpu::Buffer,
        count: usize,
    ) -> Vec<ParticleSample> {
        let positions = eval.read_vec4_f32(positions, count);
        let velocities = eval.read_vec4_f32(velocities, count);
        let phases = eval.read_u32(phases, count);

        positions
            .iter()
            .zip(&velocities)
            .zip(&phases)
            .filter_map(|((pos, vel), phase)| {
                let position = Vec3::new(pos[0], pos[1], pos[2]);
                if position.y <= PARKED_PARTICLE_Y {
                    return None;
                }
                Some(ParticleSample {
                    position,
                    velocity: Vec3::new(vel[0], vel[1], vel[2]),
                    phase: if *phase == 0 {
                        ParticlePhase::Water
                    } else {
                        ParticlePhase::Solid
                    },
                })
            })
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn water_centroid_y(&self) -> Option<f32> {
        let mut sum = 0.0;
        let mut count = 0u32;
        for sample in self
            .samples
            .iter()
            .filter(|sample| sample.phase == ParticlePhase::Water)
        {
            sum += sample.position.y;
            count += 1;
        }
        (count > 0).then_some(sum / count as f32)
    }
}

fn with_eval_device(run: impl FnOnce(&EvalDevice)) {
    let Some(eval) = EvalDevice::new() else {
        eprintln!("no GPU adapter available; skipping solver-agnostic physical realism eval");
        return;
    };
    run(&eval);
}

fn all_registered_cases(eval: &EvalDevice) -> impl Iterator<Item = SolverRun<'_>> {
    SolverCase::all()
        .into_iter()
        .map(|case| SolverRun::build(eval, case))
}

fn all_registered_solver_runs(
    eval: &EvalDevice,
    scenario: EvalScenario,
) -> impl Iterator<Item = SolverRun<'_>> {
    SolverId::all()
        .iter()
        .copied()
        .map(move |solver| SolverRun::build(eval, SolverCase { solver, scenario }))
}

fn assert_metrics_are_physically_plausible(label: &str, metrics: CommonMetrics) {
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

fn assert_cloud_is_finite_and_inside_scene(label: &str, cloud: &ParticleCloud) {
    assert!(
        cloud.render_radius.is_finite() && cloud.render_radius > 0.0,
        "{label}: render radius was invalid: {}",
        cloud.render_radius,
    );
    assert!(
        !cloud.is_empty(),
        "{label}: no active particle samples were available",
    );

    let half = cloud.bounds_size * 0.5;
    let margin = Vec3::new(2.5, 3.5, 2.5);
    for (i, sample) in cloud.samples.iter().enumerate() {
        assert!(
            sample.is_finite(),
            "{label}: sample {i} had non-finite state: {sample:?}",
        );
        assert!(
            sample.position.x >= -half.x - margin.x
                && sample.position.x <= half.x + margin.x
                && sample.position.y >= -half.y - margin.y
                && sample.position.y <= half.y + margin.y
                && sample.position.z >= -half.z - margin.z
                && sample.position.z <= half.z + margin.z,
            "{label}: sample {i} escaped scene bounds {:?} with margin {:?}: {sample:?}",
            cloud.bounds_size,
            margin,
        );
    }
}

fn vec3_is_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[test]
fn eval_all_solvers_keep_physical_state_finite_bounded_and_within_capacity() {
    with_eval_device(|eval| {
        for mut run in all_registered_cases(eval) {
            run.set_flow_speed_m_s(0.12);
            let metrics = run.step(eval, 4);
            let cloud = run.particle_cloud(eval);
            let label = run.label();

            assert_metrics_are_physically_plausible(&label, metrics);
            assert_cloud_is_finite_and_inside_scene(&label, &cloud);
        }
    });
}

#[test]
fn eval_center_pour_water_accounting_is_monotone_and_capacity_bounded() {
    with_eval_device(|eval| {
        for mut run in all_registered_solver_runs(eval, EvalScenario::CenterPour) {
            run.set_flow_speed_m_s(0.16);
            let before = run.metrics();
            let after = run.step(eval, 8);
            let label = run.label();

            assert_metrics_are_physically_plausible(&label, after);
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
    });
}

#[test]
fn eval_free_stream_water_falls_after_inflow_stops() {
    with_eval_device(|eval| {
        for mut run in all_registered_solver_runs(eval, EvalScenario::FreeStream) {
            run.set_flow_speed_m_s(0.18);
            run.step(eval, 8);
            run.set_flow_speed_m_s(0.0);

            let label = run.label();
            let early = run.particle_cloud(eval);
            let Some(early_y) = early.water_centroid_y() else {
                panic!("{label}: no water samples after warmup");
            };

            run.step(eval, 10);
            let late = run.particle_cloud(eval);
            let Some(late_y) = late.water_centroid_y() else {
                panic!("{label}: no water samples after falling interval");
            };

            assert!(
                late_y < early_y - MIN_FREE_STREAM_FALL,
                "{label}: free-stream water did not fall after inflow stopped: early_y={early_y:.4} late_y={late_y:.4}",
            );
        }
    });
}

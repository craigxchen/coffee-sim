//! Headless GPU + CPU profiler for the MPM step pipeline.
//!
//! This is a measurement harness, not part of the shipped simulation. It runs
//! the solver on the native (non-wasm) GPU backend exactly like
//! [`physics_tests`](super::physics_tests), but re-encodes each solver pass into
//! its own timestamped compute pass so per-pass GPU cost can be read back via
//! `wgpu` timestamp queries. Aggregated results are written as JSON for
//! offline bottleneck analysis.
//!
//! Run it (release is important — debug numbers are not representative):
//!
//! ```text
//! cargo test -p coffee-sim-wasm --lib --release profile_mpm_pipeline -- --ignored --nocapture
//! ```
//!
//! Tunable via environment variables:
//! - `COFFEE_SIM_PROFILE_SCENE`   scene preset: `center_pour` (default) | `free_stream` | `water_block`
//! - `COFFEE_SIM_PROFILE_SOLVER`  solver spec: `rbgs` (default) | `mpm:jacobi-cg` | `mpm:sparse-cg` | `dfsph` | `xpbd`
//! - `COFFEE_SIM_PROFILE_SOLVERS` comma-separated runnable solver specs, or `all`
//! - `COFFEE_SIM_PROFILE_CG_ITERATIONS` CG iterations per substep (defaults to scene RBGS pairs)
//! - `COFFEE_SIM_PROFILE_WARMUP`  frames to run before measuring (default 60)
//! - `COFFEE_SIM_PROFILE_FRAMES`  instrumented frames to measure (default 120)
//! - `COFFEE_SIM_PROFILE_CAL`     production `step_frame` calibration frames (default 30)
//! - `COFFEE_SIM_PROFILE_OUT`     output JSON path (default `<repo>/target/coffee-sim-profile.json`)
//!
//! ## Method and caveat
//!
//! Production [`MpmSim3D::step_frame`](super::MpmSim3D::step_frame) batches every
//! solver pass into a single compute pass. To time individual passes, the
//! instrumented path here opens a separate compute pass per pass type and writes
//! GPU timestamps at the pass boundaries (`wgpu::Features::TIMESTAMP_QUERY`,
//! which is broadly supported including on Metal stage-boundary sampling).
//! Splitting passes can slightly inflate *absolute* GPU time versus production,
//! but the MPM passes are almost entirely serialized by data dependencies, so
//! the *relative* per-pass ranking — which is what identifies bottlenecks — is
//! preserved. The Gauss-Seidel pressure solve keeps its red/black dispatches
//! interleaved in one pass, matching production exactly. A separate calibration
//! loop runs the real `step_frame` so the report also carries the honest
//! production frame cost.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc;
use std::time::Instant;

use bytemuck::cast_slice;
use serde::Serialize;

use super::inflow::{EmissionResult, MASS_UNITS_PER_ML, PARTICLES_PER_ML};
use super::pressure::{PressureContext, PressureSolverKind};
use super::state::{METRICS_SLOT_COUNT, NUM_THREADS};
use super::xpbd::XpbdPipelines;
use super::{
    dispatch_size, encode_mpm_substep_schedule, required_limits, MpmDispatch, MpmDispatchSizes,
    MpmPassLabel, MpmScheduleOp, MpmSettings, MpmSim3D, MIN_TIMESTAMP_QUERY_CAPACITY,
};

const DEFAULT_WARMUP_FRAMES: u32 = 60;
const DEFAULT_MEASURED_FRAMES: u32 = 120;
const DEFAULT_CALIBRATION_FRAMES: u32 = 30;
const FRAME_DT: f32 = 1.0 / 60.0;
const XPBD_CONSTRAINT_ITERATIONS: u32 = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SimulationBackendKind {
    Mpm,
    Dfsph,
    Xpbd,
}

impl SimulationBackendKind {
    fn id(self) -> &'static str {
        match self {
            Self::Mpm => "mpm",
            Self::Dfsph => "dfsph",
            Self::Xpbd => "xpbd",
        }
    }
}

impl fmt::Display for SimulationBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for SimulationBackendKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mpm" | "mpm-3d" => Ok(Self::Mpm),
            "dfsph" | "dfsph-water" => Ok(Self::Dfsph),
            "xpbd" => Ok(Self::Xpbd),
            other => Err(format!(
                "unknown simulation backend '{other}'; available backends: mpm, dfsph, xpbd"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum XpbdSolverKind {
    Gpu,
}

impl XpbdSolverKind {
    fn id(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
        }
    }
}

impl fmt::Display for XpbdSolverKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for XpbdSolverKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "gpu" => Ok(Self::Gpu),
            other => Err(format!(
                "unknown xpbd solver '{other}'; available solvers: gpu"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SolverSpec {
    Mpm { pressure: PressureSolverKind },
    Dfsph,
    Xpbd { solver: XpbdSolverKind },
}

impl SolverSpec {
    fn mpm(pressure: PressureSolverKind) -> Self {
        Self::Mpm { pressure }
    }

    fn backend(self) -> SimulationBackendKind {
        match self {
            Self::Mpm { .. } => SimulationBackendKind::Mpm,
            Self::Dfsph => SimulationBackendKind::Dfsph,
            Self::Xpbd { .. } => SimulationBackendKind::Xpbd,
        }
    }

    fn id(self) -> String {
        match self {
            Self::Mpm { pressure } => {
                format!("{}-{}", SimulationBackendKind::Mpm.id(), pressure.id())
            }
            Self::Dfsph => SimulationBackendKind::Dfsph.id().to_string(),
            Self::Xpbd { solver } => {
                format!("{}-{}", SimulationBackendKind::Xpbd.id(), solver.id())
            }
        }
    }
}

impl fmt::Display for SolverSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Mpm { pressure } => write!(f, "{}:{pressure}", SimulationBackendKind::Mpm),
            Self::Dfsph => f.write_str(SimulationBackendKind::Dfsph.id()),
            Self::Xpbd { solver } => write!(f, "{}:{solver}", SimulationBackendKind::Xpbd),
        }
    }
}

impl FromStr for SolverSpec {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        if let Some((backend, solver)) = trimmed.split_once(':') {
            let backend = backend.parse::<SimulationBackendKind>()?;
            return match backend {
                SimulationBackendKind::Mpm => Ok(Self::mpm(solver.parse::<PressureSolverKind>()?)),
                SimulationBackendKind::Dfsph => match solver.trim().to_ascii_lowercase().as_str() {
                    "" | "water" => Ok(Self::Dfsph),
                    other => Err(format!(
                        "unknown dfsph solver '{other}'; available solvers: water"
                    )),
                },
                SimulationBackendKind::Xpbd => Ok(Self::Xpbd {
                    solver: solver.parse::<XpbdSolverKind>()?,
                }),
            };
        }
        if trimmed.eq_ignore_ascii_case("dfsph") || trimmed.eq_ignore_ascii_case("dfsph-water") {
            return Ok(Self::Dfsph);
        }
        if trimmed.eq_ignore_ascii_case("xpbd") {
            return Ok(Self::Xpbd {
                solver: XpbdSolverKind::Gpu,
            });
        }
        Ok(Self::mpm(trimmed.parse::<PressureSolverKind>()?))
    }
}

fn env_u32_or(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

// ── GPU timestamp ring ──

/// Owns the timestamp query set plus the resolve/readback buffers reused across
/// every substep.
struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    period_ns: f32,
    capacity: u32,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, period_ns: f32, capacity: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("mpm profiler timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });
        let bytes = (capacity as u64) * 8;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm profiler resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm profiler readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buf,
            read_buf,
            period_ns,
            capacity,
        }
    }

    /// Map the readback buffer, return per-slot tick values for `0..used`.
    fn read_ticks(&self, device: &wgpu::Device, used: u32) -> Vec<u64> {
        let bytes = (used as u64) * 8;
        let slice = self.read_buf.slice(0..bytes);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("timestamp map recv")
            .expect("timestamp map");
        let ticks = {
            let data = slice.get_mapped_range();
            // `slice` spans exactly `used * 8` bytes, so the cast yields exactly
            // `used` ticks — nothing to trim.
            cast_slice::<u8, u64>(&data).to_vec()
        };
        self.read_buf.unmap();
        ticks
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct ProfilePassLabel {
    id: String,
    display: String,
    category: String,
}

impl ProfilePassLabel {
    fn new(id: &'static str, display: &'static str, category: &'static str) -> Self {
        Self {
            id: id.to_string(),
            display: display.to_string(),
            category: category.to_string(),
        }
    }

    fn mpm(label: MpmPassLabel) -> Self {
        Self {
            id: label.id().to_string(),
            display: label.display().to_string(),
            category: label.category().to_string(),
        }
    }
}

#[derive(Clone, Debug)]
struct ProfilePassMeta {
    display: String,
    category: String,
}

/// Records pass labels and the query-set slot indices used within one substep.
struct FrameRecorder<'a> {
    qs: Option<&'a wgpu::QuerySet>,
    capacity: u32,
    next: u32,
    scopes: Vec<(ProfilePassLabel, u32, u32)>,
}

impl<'a> FrameRecorder<'a> {
    fn new(timer: Option<&'a GpuTimer>) -> Self {
        let capacity = timer.map_or(0, |t| t.capacity);
        Self {
            qs: timer.map(|t| &t.query_set),
            capacity,
            next: 0,
            scopes: Vec::with_capacity(capacity as usize / 2),
        }
    }

    /// Reserve a begin/end timestamp pair for a compute pass labeled `label`.
    fn writes_mpm(&mut self, label: MpmPassLabel) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        self.writes(ProfilePassLabel::mpm(label))
    }

    fn writes(&mut self, label: ProfilePassLabel) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        let qs = self.qs?;
        let (begin, end) = self.reserve(label);
        Some(wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    fn reserve(&mut self, label: ProfilePassLabel) -> (u32, u32) {
        let begin = self.next;
        let end = self.next + 1;
        self.next += 2;
        // `assert!` (not `debug_assert!`): the harness mandates `--release`, and
        // overrunning the query set would otherwise surface as an opaque wgpu
        // validation panic instead of this actionable message.
        assert!(
            self.next <= self.capacity,
            "query set capacity {} exceeded ({} slots needed); increase the profiler timestamp capacity estimate",
            self.capacity,
            self.next
        );
        self.scopes.push((label, begin, end));
        (begin, end)
    }
}

/// Encode one timed single-pipeline compute pass.
fn timed_pass(
    encoder: &mut wgpu::CommandEncoder,
    rec: &mut FrameRecorder<'_>,
    bind_group: &wgpu::BindGroup,
    label: MpmPassLabel,
    pipeline: &wgpu::ComputePipeline,
    dispatch: MpmDispatch<'_>,
) {
    let timestamp_writes = rec.writes_mpm(label);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label.display()),
        timestamp_writes,
    });
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_pipeline(pipeline);
    match dispatch {
        MpmDispatch::Direct(workgroups) => pass.dispatch_workgroups(workgroups, 1, 1),
        MpmDispatch::Indirect { buffer, offset } => {
            pass.dispatch_workgroups_indirect(buffer, offset);
        }
    }
}

/// Encode one timed single-pipeline compute pass with a solver-neutral label.
fn timed_profile_pass(
    encoder: &mut wgpu::CommandEncoder,
    rec: &mut FrameRecorder<'_>,
    bind_group: &wgpu::BindGroup,
    label: ProfilePassLabel,
    pipeline: &wgpu::ComputePipeline,
    workgroups: u32,
) {
    let pass_label = label.display.clone();
    let timestamp_writes = rec.writes(label);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(pass_label.as_str()),
        timestamp_writes,
    });
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

// ── Per-frame accumulators ──

#[derive(Default)]
struct FrameSums {
    cpu_emit_ms: f64,
    cpu_uniforms_ms: f64,
    cpu_encode_ms: f64,
    cpu_submit_ms: f64,
    gpu_wait_ms: f64,
    gpu_passes_ms: f64,
    per_label_ms: BTreeMap<String, f64>,
    /// How many timed compute passes of each label ran this frame (e.g.
    /// `boundary_project` runs once per occurrence × substeps).
    per_label_count: BTreeMap<String, u32>,
    per_label_meta: BTreeMap<String, ProfilePassMeta>,
}

/// Run one instrumented frame, advancing `sim` and folding per-pass GPU times
/// (when `timer` is present) plus CPU sub-timings into `FrameSums`.
///
/// Mirrors [`MpmSim3D::step_frame`](super::MpmSim3D::step_frame); keep the pass
/// order here in sync with it.
fn step_frame_instrumented(
    sim: &mut MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dt: f32,
    timer: Option<&GpuTimer>,
) -> FrameSums {
    let dt = dt.min(1.0 / 30.0);
    let substeps = sim.settings.substeps.max(1);
    let sub_dt = dt / substeps as f32;
    let mass_per_particle = MASS_UNITS_PER_ML / PARTICLES_PER_ML;

    let mut sums = FrameSums::default();

    for _ in 0..substeps {
        // 1. Emit inflow particles (CPU-driven buffer write).
        let t = Instant::now();
        let EmissionResult { emitted, .. } = sim.inflow.emit_particles(
            queue,
            &sim.buffers,
            &sim.settings.spout,
            sub_dt,
            mass_per_particle,
            sim.num_water,
            sim.num_bed,
            sim.settings.max_particles,
        );
        sim.num_water += emitted;
        sums.cpu_emit_ms += ms(t);

        // 2. Update uniforms.
        let t = Instant::now();
        sim.write_uniforms(queue, sub_dt);
        let pressure_pairs = sim.pressure_rbgs_pairs_for_substep();
        sim.last_pressure_rbgs_pairs = pressure_pairs;
        sums.cpu_uniforms_ms += ms(t);

        // Dispatch sizing (identical to step_frame).
        let total_cells =
            sim.settings.grid_dims[0] * sim.settings.grid_dims[1] * sim.settings.grid_dims[2];
        let num_particles = sim.num_water + sim.num_bed;
        let dispatch = MpmDispatchSizes {
            cell_wg: dispatch_size(total_cells, NUM_THREADS),
            particle_wg: dispatch_size(num_particles, NUM_THREADS),
            bed_wg: dispatch_size(sim.num_bed, NUM_THREADS),
            metrics_wg: dispatch_size(METRICS_SLOT_COUNT as u32, 8),
        };
        let pressure_ctx = PressureContext {
            kind: sim.settings.pressure_solver,
            cell_wg: dispatch.cell_wg,
            rbgs_pairs: pressure_pairs,
            cg_iterations: sim.settings.pressure_cg_iterations,
        };

        // 3. Encode all passes, each in its own timestamped compute pass.
        let t_encode = Instant::now();
        let mut rec = FrameRecorder::new(timer);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm profiled step"),
        });

        // Grid/grid_vel clears are transfer ops (cheap memsets). They are not
        // separately timestamped — `CommandEncoder::write_timestamp` needs the
        // less-portable TIMESTAMP_QUERY_INSIDE_ENCODERS feature — but their cost
        // is still captured in the per-substep `gpu_wait` total.
        encoder.clear_buffer(&sim.buffers.grid, 0, None);
        encoder.clear_buffer(&sim.buffers.grid_vel, 0, None);

        let bg = &sim.pipelines.bind_group;
        encode_mpm_substep_schedule(&sim.pipelines, &sim.buffers, dispatch, pressure_ctx, |op| {
            match op {
                MpmScheduleOp::Pipeline {
                    label,
                    pipeline,
                    dispatch,
                } => {
                    timed_pass(&mut encoder, &mut rec, bg, label, pipeline, dispatch);
                }
                MpmScheduleOp::PressureSolve {
                    label,
                    pressure,
                    ctx,
                } => {
                    let timestamp_writes = rec.writes_mpm(label);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(label.display()),
                        timestamp_writes,
                    });
                    pass.set_bind_group(0, bg, &[]);
                    pressure.encode_solve(&mut pass, ctx);
                }
                MpmScheduleOp::CopyBufferToBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
                }
            }
        });

        let used = rec.next;
        if let Some(t) = timer {
            encoder.resolve_query_set(&t.query_set, 0..used, &t.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&t.resolve_buf, 0, &t.read_buf, 0, (used as u64) * 8);
        }
        sums.cpu_encode_ms += ms(t_encode);

        // 4. Submit.
        let t_submit = Instant::now();
        queue.submit(Some(encoder.finish()));
        sums.cpu_submit_ms += ms(t_submit);

        // 5. Wait for GPU completion (submit -> GPU done). Measured the same way
        //    whether or not timestamps are read back, so the value is comparable
        //    across adapters.
        let t_wait = Instant::now();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        sums.gpu_wait_ms += ms(t_wait);

        // 6. Read back per-pass timestamps. The GPU is already idle here, so the
        //    map round-trip is pure readback latency and is deliberately NOT
        //    counted in gpu_wait — it only shows up in the whole-frame wall time.
        if let Some(t) = timer {
            let ticks = t.read_ticks(device, used);
            for (label, b, e) in &rec.scopes {
                let delta = ticks[*e as usize].saturating_sub(ticks[*b as usize]);
                let pass_ms = delta as f64 * t.period_ns as f64 / 1.0e6;
                sums.per_label_meta
                    .entry(label.id.clone())
                    .or_insert_with(|| ProfilePassMeta {
                        display: label.display.clone(),
                        category: label.category.clone(),
                    });
                *sums.per_label_ms.entry(label.id.clone()).or_insert(0.0) += pass_ms;
                *sums.per_label_count.entry(label.id.clone()).or_insert(0) += 1;
                sums.gpu_passes_ms += pass_ms;
            }
        }

        sim.total_time += sub_dt;
    }

    sums
}

fn fold_timestamp_scopes(
    sums: &mut FrameSums,
    timer: &GpuTimer,
    device: &wgpu::Device,
    rec: &FrameRecorder<'_>,
) {
    let ticks = timer.read_ticks(device, rec.next);
    for (label, b, e) in &rec.scopes {
        let delta = ticks[*e as usize].saturating_sub(ticks[*b as usize]);
        let pass_ms = delta as f64 * timer.period_ns as f64 / 1.0e6;
        sums.per_label_meta
            .entry(label.id.clone())
            .or_insert_with(|| ProfilePassMeta {
                display: label.display.clone(),
                category: label.category.clone(),
            });
        *sums.per_label_ms.entry(label.id.clone()).or_insert(0.0) += pass_ms;
        *sums.per_label_count.entry(label.id.clone()).or_insert(0) += 1;
        sums.gpu_passes_ms += pass_ms;
    }
}

fn step_frame_dfsph_instrumented(
    sim: &mut MpmSim3D,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dt: f32,
    timer: Option<&GpuTimer>,
) -> FrameSums {
    let dt = dt.min(1.0 / 30.0);
    let substeps = sim.settings.substeps.max(1);
    let sub_dt = dt / substeps as f32;
    let mass_per_particle = MASS_UNITS_PER_ML / PARTICLES_PER_ML;
    let mut sums = FrameSums::default();

    for substep_idx in 0..substeps {
        let t = Instant::now();
        let EmissionResult { emitted, .. } = sim.inflow.emit_particles(
            queue,
            &sim.buffers,
            &sim.settings.spout,
            sub_dt,
            mass_per_particle,
            sim.num_water,
            sim.num_bed,
            sim.settings.max_particles,
        );
        sim.num_water += emitted;
        sums.cpu_emit_ms += ms(t);

        let t = Instant::now();
        sim.write_uniforms(queue, sub_dt);
        let pressure_pairs = sim.pressure_rbgs_pairs_for_substep();
        sim.last_pressure_rbgs_pairs = pressure_pairs;
        sums.cpu_uniforms_ms += ms(t);

        let total_cells =
            sim.settings.grid_dims[0] * sim.settings.grid_dims[1] * sim.settings.grid_dims[2];
        let dispatch = MpmDispatchSizes {
            cell_wg: dispatch_size(total_cells, NUM_THREADS),
            particle_wg: dispatch_size(sim.num_water + sim.num_bed, NUM_THREADS),
            bed_wg: dispatch_size(sim.num_bed, NUM_THREADS),
            metrics_wg: dispatch_size(METRICS_SLOT_COUNT as u32, 8),
        };
        let water_hash_wg = dispatch_size(total_cells + sim.settings.max_particles, NUM_THREADS);
        let water_wg = dispatch_size(sim.num_water, NUM_THREADS);
        let pressure_ctx = PressureContext {
            kind: sim.settings.pressure_solver,
            cell_wg: dispatch.cell_wg,
            rbgs_pairs: pressure_pairs,
            cg_iterations: sim.settings.pressure_cg_iterations,
        };

        let t_encode = Instant::now();
        let mut rec = FrameRecorder::new(timer);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dfsph profiled step"),
        });
        encoder.clear_buffer(&sim.buffers.grid, 0, None);
        encoder.clear_buffer(&sim.buffers.grid_vel, 0, None);
        let common = &sim.pipelines.common;
        let mpm_bg = &sim.pipelines.bind_group;
        if dispatch.metrics_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::MetricsClear,
                &common.metrics_clear,
                MpmDispatch::Direct(dispatch.metrics_wg),
            );
        }
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::BedLookupClear,
            &common.bed_lookup_clear,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        if dispatch.bed_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::BedLookupScatter,
                &common.bed_lookup_scatter,
                MpmDispatch::Direct(dispatch.bed_wg),
            );
        }

        let dfsph = &sim.pipelines.dfsph;
        timed_profile_pass(
            &mut encoder,
            &mut rec,
            &dfsph.bind_group,
            ProfilePassLabel::new("dfsph.hash_clear", "dfsph_hash_clear", "dfsph"),
            &dfsph.water_hash_clear,
            water_hash_wg,
        );
        if water_wg > 0 {
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &dfsph.bind_group,
                ProfilePassLabel::new("dfsph.hash_scatter", "dfsph_hash_scatter", "dfsph"),
                &dfsph.water_hash_scatter,
                water_wg,
            );
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &dfsph.bind_group,
                ProfilePassLabel::new("dfsph.density_factor", "dfsph_density_factor", "dfsph"),
                &dfsph.density_factor,
                water_wg,
            );
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &dfsph.bind_group,
                ProfilePassLabel::new(
                    "dfsph.divergence_estimate",
                    "dfsph_divergence_estimate",
                    "dfsph",
                ),
                &dfsph.divergence_estimate,
                water_wg,
            );
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &dfsph.bind_group,
                ProfilePassLabel::new("dfsph.divergence_solve", "dfsph_divergence_solve", "dfsph"),
                &dfsph.divergence_solve,
                water_wg,
            );
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &dfsph.bind_group,
                ProfilePassLabel::new(
                    "dfsph.predict_nonpressure",
                    "dfsph_predict_nonpressure",
                    "dfsph",
                ),
                &dfsph.predict_nonpressure,
                water_wg,
            );
            for _ in 0..2 {
                timed_profile_pass(
                    &mut encoder,
                    &mut rec,
                    &dfsph.bind_group,
                    ProfilePassLabel::new("dfsph.density_star", "dfsph_density_star", "dfsph"),
                    &dfsph.density_star,
                    water_wg,
                );
                timed_profile_pass(
                    &mut encoder,
                    &mut rec,
                    &dfsph.bind_group,
                    ProfilePassLabel::new("dfsph.density_solve", "dfsph_density_solve", "dfsph"),
                    &dfsph.density_solve,
                    water_wg,
                );
            }
        }

        if dispatch.particle_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::P2G,
                &common.p2g,
                MpmDispatch::Direct(dispatch.particle_wg),
            );
        }
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::GridUpdate,
            &common.grid_update,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::BoundaryProject,
            &common.boundary_project,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::PressureClassify,
            sim.pipelines
                .pressure
                .classify_pipeline_for(pressure_ctx.kind),
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        {
            let timestamp_writes = rec.writes_mpm(MpmPassLabel::PressureSolve);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(MpmPassLabel::PressureSolve.display()),
                timestamp_writes,
            });
            pass.set_bind_group(0, mpm_bg, &[]);
            sim.pipelines.pressure.encode_solve(&mut pass, pressure_ctx);
        }
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::PressureProject,
            sim.pipelines
                .pressure
                .project_pipeline_for(pressure_ctx.kind),
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::BoundaryProject,
            &common.boundary_project,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::PressureResidual,
            sim.pipelines
                .pressure
                .residual_pipeline_for(pressure_ctx.kind),
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::PackingPrepare,
            &common.packing_prepare,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::PackingApply,
            &common.packing_apply,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::BoundaryProject,
            &common.boundary_project,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::ViscosityPrepare,
            &common.viscosity_prepare,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::ViscosityApply,
            &common.viscosity_apply,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::BoundaryProject,
            &common.boundary_project,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        if dispatch.particle_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::G2P,
                &common.g2p,
                MpmDispatch::Direct(dispatch.particle_wg),
            );
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::BedCoupling,
                &common.bed_coupling,
                MpmDispatch::Direct(dispatch.particle_wg),
            );
        }
        if dispatch.bed_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::ExtractionAdvect,
                &common.extraction_advect,
                MpmDispatch::Direct(dispatch.bed_wg),
            );
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::BedDynamics,
                &common.bed_dynamics,
                MpmDispatch::Direct(dispatch.bed_wg),
            );
        }
        if substep_idx + 1 == substeps && dispatch.particle_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::PrepareRender,
                &common.prepare_render,
                MpmDispatch::Direct(dispatch.particle_wg),
            );
        }
        let used = rec.next;
        if let Some(t) = timer {
            encoder.resolve_query_set(&t.query_set, 0..used, &t.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&t.resolve_buf, 0, &t.read_buf, 0, (used as u64) * 8);
        }
        sums.cpu_encode_ms += ms(t_encode);

        let t_submit = Instant::now();
        queue.submit(Some(encoder.finish()));
        sums.cpu_submit_ms += ms(t_submit);

        let t_wait = Instant::now();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        sums.gpu_wait_ms += ms(t_wait);

        if let Some(t) = timer {
            fold_timestamp_scopes(&mut sums, t, device, &rec);
        }

        sim.total_time += sub_dt;
    }

    sums
}

fn step_frame_xpbd_instrumented(
    sim: &mut MpmSim3D,
    xpbd: &XpbdPipelines,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dt: f32,
    timer: Option<&GpuTimer>,
) -> FrameSums {
    let dt = dt.min(1.0 / 30.0);
    let substeps = sim.settings.substeps.max(1);
    let sub_dt = dt / substeps as f32;
    let mass_per_particle = MASS_UNITS_PER_ML / PARTICLES_PER_ML;
    let mut sums = FrameSums::default();

    for substep_idx in 0..substeps {
        let t = Instant::now();
        let EmissionResult { emitted, .. } = sim.inflow.emit_particles(
            queue,
            &sim.buffers,
            &sim.settings.spout,
            sub_dt,
            mass_per_particle,
            sim.num_water,
            sim.num_bed,
            sim.settings.max_particles,
        );
        sim.num_water += emitted;
        sums.cpu_emit_ms += ms(t);

        let t = Instant::now();
        sim.write_uniforms(queue, sub_dt);
        sums.cpu_uniforms_ms += ms(t);

        let total_cells =
            sim.settings.grid_dims[0] * sim.settings.grid_dims[1] * sim.settings.grid_dims[2];
        let dispatch = MpmDispatchSizes {
            cell_wg: dispatch_size(total_cells, NUM_THREADS),
            particle_wg: dispatch_size(sim.num_water + sim.num_bed, NUM_THREADS),
            bed_wg: dispatch_size(sim.num_bed, NUM_THREADS),
            metrics_wg: dispatch_size(METRICS_SLOT_COUNT as u32, 8),
        };
        let hash_wg = dispatch_size(total_cells + sim.settings.max_particles, NUM_THREADS);
        let water_wg = dispatch_size(sim.num_water, NUM_THREADS);

        let t_encode = Instant::now();
        let mut rec = FrameRecorder::new(timer);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("xpbd profiled step"),
        });
        let common = &sim.pipelines.common;
        let mpm_bg = &sim.pipelines.bind_group;
        if dispatch.metrics_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::MetricsClear,
                &common.metrics_clear,
                MpmDispatch::Direct(dispatch.metrics_wg),
            );
        }
        timed_pass(
            &mut encoder,
            &mut rec,
            mpm_bg,
            MpmPassLabel::BedLookupClear,
            &common.bed_lookup_clear,
            MpmDispatch::Direct(dispatch.cell_wg),
        );
        if dispatch.bed_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::BedLookupScatter,
                &common.bed_lookup_scatter,
                MpmDispatch::Direct(dispatch.bed_wg),
            );
        }

        if water_wg > 0 {
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &xpbd.bind_group,
                ProfilePassLabel::new("xpbd.predict", "xpbd_predict", "xpbd"),
                &xpbd.predict,
                water_wg,
            );
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &xpbd.bind_group,
                ProfilePassLabel::new("xpbd.hash_clear", "xpbd_hash_clear", "xpbd"),
                &xpbd.hash_clear,
                hash_wg,
            );
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &xpbd.bind_group,
                ProfilePassLabel::new("xpbd.hash_scatter", "xpbd_hash_scatter", "xpbd"),
                &xpbd.hash_scatter,
                water_wg,
            );
            for _ in 0..XPBD_CONSTRAINT_ITERATIONS {
                timed_profile_pass(
                    &mut encoder,
                    &mut rec,
                    &xpbd.bind_group,
                    ProfilePassLabel::new("xpbd.solve_density", "xpbd_solve_density", "xpbd"),
                    &xpbd.solve_density,
                    water_wg,
                );
                timed_profile_pass(
                    &mut encoder,
                    &mut rec,
                    &xpbd.bind_group,
                    ProfilePassLabel::new("xpbd.solve_bounds", "xpbd_solve_bounds", "xpbd"),
                    &xpbd.solve_bounds,
                    water_wg,
                );
            }
            timed_profile_pass(
                &mut encoder,
                &mut rec,
                &xpbd.bind_group,
                ProfilePassLabel::new("xpbd.velocity_update", "xpbd_velocity_update", "xpbd"),
                &xpbd.velocity_update,
                water_wg,
            );
        }

        if dispatch.particle_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::BedCoupling,
                &common.bed_coupling,
                MpmDispatch::Direct(dispatch.particle_wg),
            );
        }
        if dispatch.bed_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::ExtractionAdvect,
                &common.extraction_advect,
                MpmDispatch::Direct(dispatch.bed_wg),
            );
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::BedDynamics,
                &common.bed_dynamics,
                MpmDispatch::Direct(dispatch.bed_wg),
            );
        }
        if substep_idx + 1 == substeps && dispatch.particle_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                mpm_bg,
                MpmPassLabel::PrepareRender,
                &common.prepare_render,
                MpmDispatch::Direct(dispatch.particle_wg),
            );
        }

        let used = rec.next;
        if let Some(t) = timer {
            encoder.resolve_query_set(&t.query_set, 0..used, &t.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&t.resolve_buf, 0, &t.read_buf, 0, (used as u64) * 8);
        }
        sums.cpu_encode_ms += ms(t_encode);

        let t_submit = Instant::now();
        queue.submit(Some(encoder.finish()));
        sums.cpu_submit_ms += ms(t_submit);

        let t_wait = Instant::now();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        sums.gpu_wait_ms += ms(t_wait);

        if let Some(t) = timer {
            fold_timestamp_scopes(&mut sums, t, device, &rec);
        }

        sim.total_time += sub_dt;
    }

    sums
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

// ── Statistics ──

#[derive(Serialize, Clone, Default)]
struct Stat {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    total_ms: f64,
    frames: usize,
}

impl Stat {
    fn from_samples(mut v: Vec<f64>) -> Self {
        let frames = v.len();
        if frames == 0 {
            return Self::default();
        }
        let total: f64 = v.iter().sum();
        v.sort_by(|a, b| a.total_cmp(b));
        let pct = |q: f64| {
            let idx = ((q * (frames as f64 - 1.0)).round() as usize).min(frames - 1);
            v[idx]
        };
        Self {
            mean_ms: total / frames as f64,
            min_ms: v[0],
            max_ms: v[frames - 1],
            p50_ms: pct(0.5),
            p95_ms: pct(0.95),
            total_ms: total,
            frames,
        }
    }
}

#[derive(Serialize)]
struct PassStat {
    label: String,
    display_label: String,
    category: String,
    passes_per_frame: f64,
    share_of_gpu_pct: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    total_ms: f64,
}

#[derive(Serialize)]
struct SolverMetadata {
    backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pressure: Option<PressureSolverMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dfsph: Option<DfsphSolverMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    xpbd: Option<XpbdSolverMetadata>,
}

#[derive(Serialize)]
struct PressureSolverMetadata {
    kind: String,
    iterations_per_substep: u32,
}

#[derive(Serialize)]
struct DfsphSolverMetadata {
    kind: String,
    divergence_iterations_per_substep: u32,
    density_iterations_per_substep: u32,
    grid_pressure_kind: String,
    grid_pressure_iterations_per_substep: u32,
}

#[derive(Serialize)]
struct XpbdSolverMetadata {
    kind: String,
    constraint_iterations_per_substep: u32,
}

#[derive(Serialize)]
struct Metadata {
    scene: String,
    adapter: String,
    backend: String,
    device_type: String,
    timestamps_supported: bool,
    timestamp_period_ns: f32,
    warmup_frames: u32,
    measured_frames: u32,
    calibration_frames: u32,
    substeps_per_frame: u32,
    frame_dt_s: f32,
    grid_dims: [u32; 3],
    total_cells: u32,
    max_particles: u32,
    solver: SolverMetadata,
    pressure_rbgs_pairs: u32,
    water_particles_start: u32,
    water_particles_end: u32,
    bed_particles: u32,
    total_particles_end: u32,
}

#[derive(Serialize)]
struct FrameTimings {
    /// Real `step_frame` (single batched compute pass), waited to GPU completion.
    production_frame: Stat,
    /// Full instrumented frame including per-substep readback (split passes).
    instrumented_wall: Stat,
    /// Sum of per-pass GPU timestamp deltas.
    gpu_passes_sum: Stat,
    /// Submit -> GPU-done wall time (per frame, summed over substeps).
    gpu_wait: Stat,
    /// GPU time in `gpu_wait` not attributed to a timed pass: grid clears,
    /// inter-pass gaps, query resolve/copy, and scheduling slack. Makes the
    /// breakdown self-checking: `gpu_passes_sum + gpu_unattributed` ~= `gpu_wait`.
    gpu_unattributed: Stat,
    cpu_emit: Stat,
    cpu_uniforms: Stat,
    cpu_encode: Stat,
    cpu_submit: Stat,
}

#[derive(Serialize)]
struct ProfileReport {
    schema_version: u32,
    metadata: Metadata,
    frame_timings: FrameTimings,
    gpu_passes: Vec<PassStat>,
    bottlenecks: Vec<String>,
}

struct ProfileMeasurements {
    wall_ms: Vec<f64>,
    gpu_passes_sum: Vec<f64>,
    gpu_wait: Vec<f64>,
    gpu_unattributed: Vec<f64>,
    cpu_emit: Vec<f64>,
    cpu_uniforms: Vec<f64>,
    cpu_encode: Vec<f64>,
    cpu_submit: Vec<f64>,
    per_label: BTreeMap<String, Vec<f64>>,
    label_dispatch_count: BTreeMap<String, u64>,
    label_meta: BTreeMap<String, ProfilePassMeta>,
}

impl ProfileMeasurements {
    fn with_capacity(frames: u32) -> Self {
        let capacity = frames as usize;
        Self {
            wall_ms: Vec::with_capacity(capacity),
            gpu_passes_sum: Vec::with_capacity(capacity),
            gpu_wait: Vec::with_capacity(capacity),
            gpu_unattributed: Vec::with_capacity(capacity),
            cpu_emit: Vec::with_capacity(capacity),
            cpu_uniforms: Vec::with_capacity(capacity),
            cpu_encode: Vec::with_capacity(capacity),
            cpu_submit: Vec::with_capacity(capacity),
            per_label: BTreeMap::new(),
            label_dispatch_count: BTreeMap::new(),
            label_meta: BTreeMap::new(),
        }
    }

    fn record(&mut self, wall_ms: f64, sums: FrameSums) {
        self.wall_ms.push(wall_ms);
        self.gpu_passes_sum.push(sums.gpu_passes_ms);
        self.gpu_wait.push(sums.gpu_wait_ms);
        self.gpu_unattributed
            .push((sums.gpu_wait_ms - sums.gpu_passes_ms).max(0.0));
        self.cpu_emit.push(sums.cpu_emit_ms);
        self.cpu_uniforms.push(sums.cpu_uniforms_ms);
        self.cpu_encode.push(sums.cpu_encode_ms);
        self.cpu_submit.push(sums.cpu_submit_ms);
        for (label, value) in sums.per_label_ms {
            self.per_label.entry(label).or_default().push(value);
        }
        for (label, count) in sums.per_label_count {
            *self.label_dispatch_count.entry(label).or_insert(0) += count as u64;
        }
        for (label, meta) in sums.per_label_meta {
            self.label_meta.entry(label).or_insert(meta);
        }
    }

    fn finish(
        self,
        production_ms: Vec<f64>,
        measured_frames: u32,
    ) -> (FrameTimings, Vec<PassStat>) {
        let total_pass_mean: f64 = self
            .per_label
            .values()
            .map(|v| v.iter().sum::<f64>() / v.len().max(1) as f64)
            .sum();
        let mut gpu_passes: Vec<PassStat> = self
            .per_label
            .into_iter()
            .map(|(label, samples)| {
                let stat = Stat::from_samples(samples);
                let passes = *self.label_dispatch_count.get(&label).unwrap_or(&0) as f64
                    / measured_frames.max(1) as f64;
                let meta = self
                    .label_meta
                    .get(&label)
                    .expect("profile label metadata recorded with samples");
                PassStat {
                    label,
                    display_label: meta.display.clone(),
                    category: meta.category.clone(),
                    passes_per_frame: passes,
                    share_of_gpu_pct: if total_pass_mean > 0.0 {
                        stat.mean_ms / total_pass_mean * 100.0
                    } else {
                        0.0
                    },
                    mean_ms: stat.mean_ms,
                    min_ms: stat.min_ms,
                    max_ms: stat.max_ms,
                    p50_ms: stat.p50_ms,
                    p95_ms: stat.p95_ms,
                    total_ms: stat.total_ms,
                }
            })
            .collect();
        gpu_passes.sort_by(|a, b| b.mean_ms.total_cmp(&a.mean_ms));

        let frame_timings = FrameTimings {
            production_frame: Stat::from_samples(production_ms),
            instrumented_wall: Stat::from_samples(self.wall_ms),
            gpu_passes_sum: Stat::from_samples(self.gpu_passes_sum),
            gpu_wait: Stat::from_samples(self.gpu_wait),
            gpu_unattributed: Stat::from_samples(self.gpu_unattributed),
            cpu_emit: Stat::from_samples(self.cpu_emit),
            cpu_uniforms: Stat::from_samples(self.cpu_uniforms),
            cpu_encode: Stat::from_samples(self.cpu_encode),
            cpu_submit: Stat::from_samples(self.cpu_submit),
        };

        (frame_timings, gpu_passes)
    }
}

fn top_pass_bottlenecks(gpu_passes: &[PassStat]) -> Vec<String> {
    gpu_passes
        .iter()
        .take(5)
        .map(|pass| {
            format!(
                "{}: {:.3} ms/frame ({:.1}% of GPU pass time, {:.0} passes/frame)",
                pass.label, pass.mean_ms, pass.share_of_gpu_pct, pass.passes_per_frame
            )
        })
        .collect()
}

// ── Device setup ──

fn request_adapter() -> Option<wgpu::Adapter> {
    if std::env::var_os("COFFEE_SIM_SKIP_GPU_TESTS").is_some() {
        return None;
    }
    let instance = wgpu::Instance::default();
    // Match production's adapter selection (see renderer.rs) so the profiled
    // device is the one the app actually runs on — matters on multi-GPU hosts
    // where the default (PowerPreference::None) may pick the integrated GPU.
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..wgpu::RequestAdapterOptions::default()
    }))
    .ok()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProfileScene {
    CenterPour,
    FreeStream,
    WaterBlock,
}

impl ProfileScene {
    fn id(self) -> &'static str {
        match self {
            Self::CenterPour => "center_pour",
            Self::FreeStream => "free_stream",
            Self::WaterBlock => "water_block",
        }
    }

    fn mpm_settings(self) -> MpmSettings {
        match self {
            Self::CenterPour => MpmSettings::benchmark_center_pour(),
            Self::FreeStream => MpmSettings::benchmark_free_stream(),
            Self::WaterBlock => MpmSettings::benchmark_filter_water_block(),
        }
    }
}

impl fmt::Display for ProfileScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for ProfileScene {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "center_pour" | "center-pour" => Ok(Self::CenterPour),
            "free_stream" | "free-stream" => Ok(Self::FreeStream),
            "water_block" | "water-block" => Ok(Self::WaterBlock),
            other => Err(format!(
                "unknown profile scene '{other}'; available scenes: center_pour, free_stream, water_block"
            )),
        }
    }
}

fn profile_solver_specs() -> Vec<SolverSpec> {
    if let Ok(value) = std::env::var("COFFEE_SIM_PROFILE_SOLVERS") {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("all") {
            return runnable_solver_specs();
        }
        return trimmed
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| {
                part.parse::<SolverSpec>()
                    .unwrap_or_else(|err| panic!("{err}"))
            })
            .collect();
    }
    let solver = std::env::var("COFFEE_SIM_PROFILE_SOLVER").unwrap_or_else(|_| "rbgs".into());
    vec![solver
        .parse::<SolverSpec>()
        .unwrap_or_else(|err| panic!("{err}"))]
}

fn runnable_solver_specs() -> Vec<SolverSpec> {
    let mut specs: Vec<_> = PressureSolverKind::ALL
        .iter()
        .copied()
        .map(SolverSpec::mpm)
        .collect();
    specs.push(SolverSpec::Dfsph);
    specs.push(SolverSpec::Xpbd {
        solver: XpbdSolverKind::Gpu,
    });
    specs
}

fn xpbd_profiler_timestamp_query_capacity(substeps: u32) -> u32 {
    let xpbd_passes = 4 + XPBD_CONSTRAINT_ITERATIONS * 2;
    let shared_tail_passes = 7;
    let passes_per_substep = xpbd_passes + shared_tail_passes;
    (substeps.max(1) * passes_per_substep * 2).max(MIN_TIMESTAMP_QUERY_CAPACITY)
}

fn output_path() -> PathBuf {
    if let Ok(p) = std::env::var("COFFEE_SIM_PROFILE_OUT") {
        return PathBuf::from(p);
    }
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/coffee-sim-profile.json"
    ))
}

fn output_path_for_solver(base: &PathBuf, solver: SolverSpec, multiple: bool) -> PathBuf {
    if !multiple {
        return base.clone();
    }
    let stem = base
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("coffee-sim-profile");
    let extension = base.extension().and_then(|value| value.to_str());
    let solver_id = solver.id();
    let file_name = match extension {
        Some(ext) => format!("{stem}-{solver_id}.{ext}"),
        None => format!("{stem}-{solver_id}"),
    };
    base.with_file_name(file_name)
}

#[test]
fn profiler_solver_specs_parse_backend_qualified_values() {
    assert_eq!(
        "rbgs".parse::<SolverSpec>(),
        Ok(SolverSpec::mpm(PressureSolverKind::Rbgs))
    );
    assert_eq!(
        "mpm:sparse-cg".parse::<SolverSpec>(),
        Ok(SolverSpec::mpm(PressureSolverKind::SparseCg))
    );
    assert_eq!("dfsph".parse::<SolverSpec>(), Ok(SolverSpec::Dfsph));
    assert_eq!("dfsph:water".parse::<SolverSpec>(), Ok(SolverSpec::Dfsph));
    assert_eq!(
        "xpbd".parse::<SolverSpec>(),
        Ok(SolverSpec::Xpbd {
            solver: XpbdSolverKind::Gpu
        })
    );
    assert_eq!(
        "xpbd:gpu".parse::<SolverSpec>(),
        Ok(SolverSpec::Xpbd {
            solver: XpbdSolverKind::Gpu
        })
    );
    assert!("xpbd:cpu".parse::<SolverSpec>().is_err());
}

#[test]
fn profiler_all_expands_to_runnable_gpu_solver_specs() {
    assert_eq!(
        runnable_solver_specs(),
        vec![
            SolverSpec::mpm(PressureSolverKind::Rbgs),
            SolverSpec::mpm(PressureSolverKind::JacobiCg),
            SolverSpec::mpm(PressureSolverKind::SparseCg),
            SolverSpec::Dfsph,
            SolverSpec::Xpbd {
                solver: XpbdSolverKind::Gpu
            },
        ]
    );
    assert!(
        runnable_solver_specs()
            .iter()
            .any(|solver| matches!(solver, SolverSpec::Xpbd { .. })),
        "XPBD GPU path should be included in `all` so same-scene solver comparisons do not require code changes"
    );
}

#[test]
fn profiler_scenes_parse_shared_profile_scene_ids() {
    assert_eq!(
        "center_pour".parse::<ProfileScene>(),
        Ok(ProfileScene::CenterPour)
    );
    assert_eq!(
        "free-stream".parse::<ProfileScene>(),
        Ok(ProfileScene::FreeStream)
    );
    assert_eq!(
        "water_block".parse::<ProfileScene>(),
        Ok(ProfileScene::WaterBlock)
    );
    assert!("low_complexity_shortcut".parse::<ProfileScene>().is_err());
}

struct ProfilerRunConfig<'a> {
    scene: ProfileScene,
    warmup: u32,
    measured: u32,
    calibration: u32,
    base_output_path: &'a PathBuf,
    multiple_outputs: bool,
}

struct ProfilerDeviceContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    info: &'a wgpu::AdapterInfo,
    timestamps_supported: bool,
    period_ns: f32,
}

fn write_report(path: &PathBuf, report: &ProfileReport) {
    let json = serde_json::to_string_pretty(report).expect("serialize report");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    std::fs::write(path, json).expect("write profile json");
}

fn print_report_summary(path: &PathBuf, report: &ProfileReport) {
    let solver_kind = report
        .metadata
        .solver
        .pressure
        .as_ref()
        .map(|pressure| pressure.kind.as_str())
        .or_else(|| {
            report
                .metadata
                .solver
                .dfsph
                .as_ref()
                .map(|dfsph| dfsph.kind.as_str())
        })
        .or_else(|| {
            report
                .metadata
                .solver
                .xpbd
                .as_ref()
                .map(|xpbd| xpbd.kind.as_str())
        })
        .unwrap_or("n/a");
    println!(
        "\n=== {} profile ({}, solver: {}) ===",
        report.metadata.solver.backend, report.metadata.scene, solver_kind
    );
    println!(
        "adapter: {} [{}] | cells: {} | particles: {} water + {} bed",
        report.metadata.adapter,
        report.metadata.backend,
        report.metadata.total_cells,
        report.metadata.water_particles_end,
        report.metadata.bed_particles,
    );
    println!(
        "production step_frame: {:.3} ms/frame (p95 {:.3})",
        report.frame_timings.production_frame.mean_ms, report.frame_timings.production_frame.p95_ms,
    );
    if report.metadata.timestamps_supported {
        println!(
            "instrumented GPU passes sum: {:.3} ms/frame  (+ {:.3} unattributed = {:.3} gpu_wait)\n",
            report.frame_timings.gpu_passes_sum.mean_ms,
            report.frame_timings.gpu_unattributed.mean_ms,
            report.frame_timings.gpu_wait.mean_ms,
        );
        println!(
            "{:<20} {:>10} {:>8} {:>9}",
            "pass", "ms/frame", "%gpu", "passes/f"
        );
        for pass in &report.gpu_passes {
            println!(
                "{:<20} {:>10.3} {:>7.1}% {:>9.0}",
                pass.label, pass.mean_ms, pass.share_of_gpu_pct, pass.passes_per_frame
            );
        }
    } else {
        println!(
            "GPU timestamps unavailable on this adapter - no per-pass breakdown. \
             GPU time (submit -> done): {:.3} ms/frame\n",
            report.frame_timings.gpu_wait.mean_ms,
        );
    }
    println!("\nJSON written to: {}", path.display());
}

fn profile_mpm_solver(
    ctx: &ProfilerDeviceContext<'_>,
    run: &ProfilerRunConfig<'_>,
    solver: SolverSpec,
    pressure: PressureSolverKind,
) {
    let mut settings = run.scene.mpm_settings();
    settings.pressure_solver = pressure;
    settings.pressure_cg_iterations = std::env::var("COFFEE_SIM_PROFILE_CG_ITERATIONS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(settings.pressure_rbgs_pairs);
    let grid_dims = settings.grid_dims;
    let total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
    let max_particles = settings.max_particles;
    let substeps = settings.substeps.max(1);

    let mut sim = MpmSim3D::new(ctx.device, ctx.queue, settings);

    for _ in 0..run.warmup {
        sim.step_frame(ctx.device, ctx.queue, FRAME_DT);
    }
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let bed_particles = sim.num_bed;
    let pressure_rbgs_pairs = sim
        .last_pressure_rbgs_pairs
        .max(sim.settings.pressure_rbgs_pairs);

    let mut production_ms = Vec::with_capacity(run.calibration as usize);
    for _ in 0..run.calibration {
        let t = Instant::now();
        sim.step_frame(ctx.device, ctx.queue, FRAME_DT);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        production_ms.push(ms(t));
    }

    let query_capacity = sim.profiler_timestamp_query_capacity();
    let timer = ctx
        .timestamps_supported
        .then(|| GpuTimer::new(ctx.device, ctx.period_ns, query_capacity));
    let mut measurements = ProfileMeasurements::with_capacity(run.measured);

    let water_particles_start = sim.num_water;
    for _ in 0..run.measured {
        let t = Instant::now();
        let sums =
            step_frame_instrumented(&mut sim, ctx.device, ctx.queue, FRAME_DT, timer.as_ref());
        measurements.record(ms(t), sums);
    }
    let water_particles_end = sim.num_water;
    let (frame_timings, gpu_passes) = measurements.finish(production_ms, run.measured);

    let mut bottlenecks = top_pass_bottlenecks(&gpu_passes);
    let cpu_total = frame_timings.cpu_emit.mean_ms
        + frame_timings.cpu_uniforms.mean_ms
        + frame_timings.cpu_encode.mean_ms
        + frame_timings.cpu_submit.mean_ms;
    let cpu_gpu_verdict = if !ctx.timestamps_supported {
        "n/a - no GPU timestamps on this adapter; compare CPU against gpu_wait instead"
    } else if cpu_total > frame_timings.gpu_passes_sum.mean_ms * 0.25 {
        "CPU-side is non-trivial"
    } else {
        "GPU-bound (CPU orchestration negligible)"
    };
    bottlenecks.push(format!(
        "CPU orchestration: {:.3} ms/frame vs GPU passes {:.3} ms/frame -> {}",
        cpu_total, frame_timings.gpu_passes_sum.mean_ms, cpu_gpu_verdict
    ));

    let report = ProfileReport {
        schema_version: 2,
        metadata: Metadata {
            scene: run.scene.to_string(),
            adapter: ctx.info.name.clone(),
            backend: format!("{:?}", ctx.info.backend),
            device_type: format!("{:?}", ctx.info.device_type),
            timestamps_supported: ctx.timestamps_supported,
            timestamp_period_ns: ctx.period_ns,
            warmup_frames: run.warmup,
            measured_frames: run.measured,
            calibration_frames: run.calibration,
            substeps_per_frame: substeps,
            frame_dt_s: FRAME_DT,
            grid_dims,
            total_cells,
            max_particles,
            solver: SolverMetadata {
                backend: solver.backend().to_string(),
                pressure: Some(PressureSolverMetadata {
                    kind: sim.pressure_solver_kind().to_string(),
                    iterations_per_substep: sim.pressure_solver_iterations_per_substep(),
                }),
                dfsph: None,
                xpbd: None,
            },
            pressure_rbgs_pairs,
            water_particles_start,
            water_particles_end,
            bed_particles,
            total_particles_end: water_particles_end + bed_particles,
        },
        frame_timings,
        gpu_passes,
        bottlenecks,
    };

    let path = output_path_for_solver(run.base_output_path, solver, run.multiple_outputs);
    write_report(&path, &report);
    print_report_summary(&path, &report);
}

fn profile_dfsph_solver(
    ctx: &ProfilerDeviceContext<'_>,
    run: &ProfilerRunConfig<'_>,
    solver: SolverSpec,
) {
    let settings = run.scene.mpm_settings();
    let grid_dims = settings.grid_dims;
    let total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
    let max_particles = settings.max_particles;
    let substeps = settings.substeps.max(1);
    let mut sim = MpmSim3D::new(ctx.device, ctx.queue, settings);

    for _ in 0..run.warmup {
        step_frame_dfsph_instrumented(&mut sim, ctx.device, ctx.queue, FRAME_DT, None);
    }
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let bed_particles = sim.num_bed;
    let mut production_ms = Vec::with_capacity(run.calibration as usize);
    for _ in 0..run.calibration {
        let t = Instant::now();
        step_frame_dfsph_instrumented(&mut sim, ctx.device, ctx.queue, FRAME_DT, None);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        production_ms.push(ms(t));
    }

    let timer = ctx
        .timestamps_supported
        .then(|| GpuTimer::new(ctx.device, ctx.period_ns, MIN_TIMESTAMP_QUERY_CAPACITY));
    let mut measurements = ProfileMeasurements::with_capacity(run.measured);

    let water_particles_start = sim.num_water;
    for _ in 0..run.measured {
        let t = Instant::now();
        let sums = step_frame_dfsph_instrumented(
            &mut sim,
            ctx.device,
            ctx.queue,
            FRAME_DT,
            timer.as_ref(),
        );
        measurements.record(ms(t), sums);
    }
    let water_particles_end = sim.num_water;
    let (frame_timings, gpu_passes) = measurements.finish(production_ms, run.measured);

    let mut bottlenecks = top_pass_bottlenecks(&gpu_passes);
    bottlenecks.push(
        "DFSPH backend profiles GPU water pressure correction plus the shared MPM grid/bed/render tail; \
         tiled pressure projection from codex/dfsph-water remains a future sparse optimization."
            .to_string(),
    );

    let report = ProfileReport {
        schema_version: 2,
        metadata: Metadata {
            scene: run.scene.to_string(),
            adapter: ctx.info.name.clone(),
            backend: format!("{:?}", ctx.info.backend),
            device_type: format!("{:?}", ctx.info.device_type),
            timestamps_supported: ctx.timestamps_supported,
            timestamp_period_ns: ctx.period_ns,
            warmup_frames: run.warmup,
            measured_frames: run.measured,
            calibration_frames: run.calibration,
            substeps_per_frame: substeps,
            frame_dt_s: FRAME_DT,
            grid_dims,
            total_cells,
            max_particles,
            solver: SolverMetadata {
                backend: solver.backend().to_string(),
                pressure: None,
                dfsph: Some(DfsphSolverMetadata {
                    kind: "water".to_string(),
                    divergence_iterations_per_substep: 1,
                    density_iterations_per_substep: 2,
                    grid_pressure_kind: sim.pressure_solver_kind().to_string(),
                    grid_pressure_iterations_per_substep: sim
                        .pressure_solver_iterations_per_substep(),
                }),
                xpbd: None,
            },
            pressure_rbgs_pairs: sim
                .last_pressure_rbgs_pairs
                .max(sim.settings.pressure_rbgs_pairs),
            water_particles_start,
            water_particles_end,
            bed_particles,
            total_particles_end: water_particles_end + bed_particles,
        },
        frame_timings,
        gpu_passes,
        bottlenecks,
    };

    let path = output_path_for_solver(run.base_output_path, solver, run.multiple_outputs);
    write_report(&path, &report);
    print_report_summary(&path, &report);
}

fn profile_xpbd_solver(
    ctx: &ProfilerDeviceContext<'_>,
    run: &ProfilerRunConfig<'_>,
    solver: SolverSpec,
    kind: XpbdSolverKind,
) {
    let settings = run.scene.mpm_settings();
    let grid_dims = settings.grid_dims;
    let total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
    let max_particles = settings.max_particles;
    let substeps = settings.substeps.max(1);
    let mut sim = MpmSim3D::new(ctx.device, ctx.queue, settings);
    let xpbd = XpbdPipelines::new(ctx.device, &sim.buffers);

    for _ in 0..run.warmup {
        step_frame_xpbd_instrumented(&mut sim, &xpbd, ctx.device, ctx.queue, FRAME_DT, None);
    }
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let bed_particles = sim.num_bed;
    let mut production_ms = Vec::with_capacity(run.calibration as usize);
    for _ in 0..run.calibration {
        let t = Instant::now();
        step_frame_xpbd_instrumented(&mut sim, &xpbd, ctx.device, ctx.queue, FRAME_DT, None);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        production_ms.push(ms(t));
    }

    let query_capacity = xpbd_profiler_timestamp_query_capacity(substeps);
    let timer = ctx
        .timestamps_supported
        .then(|| GpuTimer::new(ctx.device, ctx.period_ns, query_capacity));
    let mut measurements = ProfileMeasurements::with_capacity(run.measured);

    let water_particles_start = sim.num_water;
    for _ in 0..run.measured {
        let t = Instant::now();
        let sums = step_frame_xpbd_instrumented(
            &mut sim,
            &xpbd,
            ctx.device,
            ctx.queue,
            FRAME_DT,
            timer.as_ref(),
        );
        measurements.record(ms(t), sums);
    }
    let water_particles_end = sim.num_water;
    let (frame_timings, gpu_passes) = measurements.finish(production_ms, run.measured);

    let mut bottlenecks = top_pass_bottlenecks(&gpu_passes);
    bottlenecks.push(
        "XPBD backend uses GPU prediction, spatial hash, density constraints, bounds constraints, \
         and velocity update over the shared particle buffers; browser parity and physical validation \
         remain the next gates before treating it as production-equivalent."
            .to_string(),
    );

    let report = ProfileReport {
        schema_version: 2,
        metadata: Metadata {
            scene: run.scene.to_string(),
            adapter: ctx.info.name.clone(),
            backend: format!("{:?}", ctx.info.backend),
            device_type: format!("{:?}", ctx.info.device_type),
            timestamps_supported: ctx.timestamps_supported,
            timestamp_period_ns: ctx.period_ns,
            warmup_frames: run.warmup,
            measured_frames: run.measured,
            calibration_frames: run.calibration,
            substeps_per_frame: substeps,
            frame_dt_s: FRAME_DT,
            grid_dims,
            total_cells,
            max_particles,
            solver: SolverMetadata {
                backend: solver.backend().to_string(),
                pressure: None,
                dfsph: None,
                xpbd: Some(XpbdSolverMetadata {
                    kind: kind.to_string(),
                    constraint_iterations_per_substep: XPBD_CONSTRAINT_ITERATIONS,
                }),
            },
            pressure_rbgs_pairs: 0,
            water_particles_start,
            water_particles_end,
            bed_particles,
            total_particles_end: water_particles_end + bed_particles,
        },
        frame_timings,
        gpu_passes,
        bottlenecks,
    };

    let path = output_path_for_solver(run.base_output_path, solver, run.multiple_outputs);
    write_report(&path, &report);
    print_report_summary(&path, &report);
}

#[test]
#[ignore = "profiling harness; run explicitly with --ignored --release"]
fn profile_mpm_pipeline() {
    let Some(adapter) = request_adapter() else {
        eprintln!("profile_mpm_pipeline: no GPU adapter available; skipping.");
        return;
    };

    let features = adapter.features();
    let timestamps_supported = features.contains(wgpu::Features::TIMESTAMP_QUERY);
    let required_features = if timestamps_supported {
        wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    };

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("coffee-sim profiler device"),
        required_features,
        required_limits: required_limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .expect("request profiler device");

    let info = adapter.get_info();
    let scene = std::env::var("COFFEE_SIM_PROFILE_SCENE")
        .unwrap_or_else(|_| "center_pour".into())
        .parse::<ProfileScene>()
        .unwrap_or_else(|err| panic!("{err}"));
    let solvers = profile_solver_specs();
    let warmup = env_u32_or("COFFEE_SIM_PROFILE_WARMUP", DEFAULT_WARMUP_FRAMES);
    let measured = env_u32_or("COFFEE_SIM_PROFILE_FRAMES", DEFAULT_MEASURED_FRAMES);
    let calibration = env_u32_or("COFFEE_SIM_PROFILE_CAL", DEFAULT_CALIBRATION_FRAMES);
    let base_output_path = output_path();
    let multiple_outputs = solvers.len() > 1;
    let period_ns = queue.get_timestamp_period();

    if !timestamps_supported {
        eprintln!(
            "profile_mpm_pipeline: adapter lacks TIMESTAMP_QUERY; \
             reporting frame-level timings only (no per-pass breakdown)."
        );
    }

    let device_ctx = ProfilerDeviceContext {
        device: &device,
        queue: &queue,
        info: &info,
        timestamps_supported,
        period_ns,
    };
    let run = ProfilerRunConfig {
        scene,
        warmup,
        measured,
        calibration,
        base_output_path: &base_output_path,
        multiple_outputs,
    };

    for &solver in &solvers {
        match solver {
            SolverSpec::Mpm { pressure } => {
                profile_mpm_solver(&device_ctx, &run, solver, pressure);
            }
            SolverSpec::Dfsph => {
                profile_dfsph_solver(&device_ctx, &run, solver);
            }
            SolverSpec::Xpbd { solver: kind } => {
                profile_xpbd_solver(&device_ctx, &run, solver, kind);
            }
        }
    }
}

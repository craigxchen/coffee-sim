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
//! - `COFFEE_SIM_PROFILE_SPARSE_PRESSURE`  `1`/`true` routes the pressure solve through the sparse tile RBGS path (default dense)
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
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use bytemuck::cast_slice;
use serde::Serialize;

use crate::profiling::{
    env_flag, parse_solver_specs, solver_list_summary, solver_run_metadata, SolverRunMetadata,
    SolverSpec,
};

use super::inflow::{EmissionResult, MASS_UNITS_PER_ML, PARTICLES_PER_ML};
use super::state::{sparse_tile_slot_count, tile_count, METRICS_SLOT_COUNT, NUM_THREADS};
use super::{dispatch_size, required_limits, MpmSettings, MpmSim3D};

/// Query-set slots to allocate per substep. One substep records ~23 passes ×
/// 2 timestamps; 64 leaves comfortable headroom.
const QUERY_CAPACITY: u32 = 64;

const DEFAULT_WARMUP_FRAMES: u32 = 60;
const DEFAULT_MEASURED_FRAMES: u32 = 120;
const DEFAULT_CALIBRATION_FRAMES: u32 = 30;
const FRAME_DT: f32 = 1.0 / 60.0;

#[derive(Clone, Debug)]
struct ProfilerRunOptions {
    scene: String,
    warmup: u32,
    measured: u32,
    calibration: u32,
    require_gpu: bool,
}

impl ProfilerRunOptions {
    fn from_legacy_env() -> Self {
        Self {
            scene: std::env::var("COFFEE_SIM_PROFILE_SCENE")
                .unwrap_or_else(|_| "center_pour".into()),
            warmup: env_u32_or("COFFEE_SIM_PROFILE_WARMUP", DEFAULT_WARMUP_FRAMES),
            measured: env_u32_or("COFFEE_SIM_PROFILE_FRAMES", DEFAULT_MEASURED_FRAMES),
            calibration: env_u32_or("COFFEE_SIM_PROFILE_CAL", DEFAULT_CALIBRATION_FRAMES),
            require_gpu: env_flag("COFFEE_SIM_PROFILE_REQUIRE_GPU"),
        }
    }

    fn from_profile_args() -> Self {
        Self {
            scene: profile_arg_value("scene")
                .or_else(|| std::env::var("COFFEE_SIM_PROFILE_SCENE").ok())
                .unwrap_or_else(|| "center_pour".into()),
            warmup: profile_arg_value("warmup")
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| env_u32_or("COFFEE_SIM_PROFILE_WARMUP", DEFAULT_WARMUP_FRAMES)),
            measured: profile_arg_value("frames")
                .or_else(|| profile_arg_value("measured"))
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| {
                    env_u32_or("COFFEE_SIM_PROFILE_FRAMES", DEFAULT_MEASURED_FRAMES)
                }),
            calibration: profile_arg_value("cal")
                .or_else(|| profile_arg_value("calibration"))
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| {
                    env_u32_or("COFFEE_SIM_PROFILE_CAL", DEFAULT_CALIBRATION_FRAMES)
                }),
            require_gpu: profile_arg_flag("require-gpu")
                || profile_arg_flag("require_gpu")
                || env_flag("COFFEE_SIM_PROFILE_REQUIRE_GPU"),
        }
    }
}

fn env_u32_or(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn profile_args() -> Vec<String> {
    std::env::var("COFFEE_SIM_PROFILE_ARGS")
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn profile_arg_value(name: &str) -> Option<String> {
    let long = format!("--{name}");
    let key = name.replace('-', "_");
    let args = profile_args();
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if arg == &long {
            return iter.peek().map(|value| (*value).clone().to_string());
        }
        if let Some(value) = arg.strip_prefix(&(long.clone() + "=")) {
            return Some(value.to_string());
        }
        if let Some(value) = arg.strip_prefix(&(key.clone() + "=")) {
            return Some(value.to_string());
        }
    }
    None
}

fn profile_arg_flag(name: &str) -> bool {
    let long = format!("--{name}");
    let key = name.replace('-', "_");
    profile_args().iter().any(|arg| {
        arg == &long
            || arg == &(key.clone() + "=true")
            || arg == &(key.clone() + "=1")
            || arg == &(long.clone() + "=true")
            || arg == &(long.clone() + "=1")
    })
}

// ── GPU timestamp ring ──

/// Owns the timestamp query set plus the resolve/readback buffers reused across
/// every substep.
struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    period_ns: f32,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, period_ns: f32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("mpm profiler timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_CAPACITY,
        });
        let bytes = (QUERY_CAPACITY as u64) * 8;
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

/// Records pass labels and the query-set slot indices used within one substep.
struct FrameRecorder<'a> {
    qs: Option<&'a wgpu::QuerySet>,
    next: u32,
    scopes: Vec<(&'static str, u32, u32)>,
}

impl<'a> FrameRecorder<'a> {
    fn new(qs: Option<&'a wgpu::QuerySet>) -> Self {
        Self {
            qs,
            next: 0,
            scopes: Vec::with_capacity(QUERY_CAPACITY as usize / 2),
        }
    }

    /// Reserve a begin/end timestamp pair for a compute pass labeled `label`.
    fn writes(&mut self, label: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        let qs = self.qs?;
        let (begin, end) = self.reserve(label);
        Some(wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    fn reserve(&mut self, label: &'static str) -> (u32, u32) {
        let begin = self.next;
        let end = self.next + 1;
        self.next += 2;
        // `assert!` (not `debug_assert!`): the harness mandates `--release`, and
        // overrunning the query set would otherwise surface as an opaque wgpu
        // validation panic instead of this actionable message.
        assert!(
            self.next <= QUERY_CAPACITY,
            "query set capacity {} exceeded ({} slots needed); raise QUERY_CAPACITY",
            QUERY_CAPACITY,
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
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    workgroups: u32,
) {
    let timestamp_writes = rec.writes(label);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
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
    per_label_ms: BTreeMap<&'static str, f64>,
    /// How many timed compute passes of each label ran this frame (e.g.
    /// `boundary_project` runs once per occurrence × substeps).
    per_label_count: BTreeMap<&'static str, u32>,
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
        let cell_wg = dispatch_size(total_cells, NUM_THREADS);
        let particle_wg = dispatch_size(num_particles, NUM_THREADS);
        let bed_wg = dispatch_size(sim.num_bed, NUM_THREADS);
        let metrics_wg = dispatch_size(METRICS_SLOT_COUNT as u32, 8);
        let sparse_pressure = sim.settings.sparse_pressure;
        let sparse_clear_wg =
            dispatch_size(sparse_tile_slot_count(sim.settings.grid_dims), NUM_THREADS);
        let tile_wg = tile_count(sim.settings.grid_dims);

        // 3. Encode all passes, each in its own timestamped compute pass.
        let t_encode = Instant::now();
        let mut rec = FrameRecorder::new(timer.map(|t| &t.query_set));
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
        let p = &sim.pipelines;

        if metrics_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                bg,
                "metrics_clear",
                &p.metrics_clear,
                metrics_wg,
            );
        }
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "bed_lookup_clear",
            &p.bed_lookup_clear,
            cell_wg,
        );
        if bed_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                bg,
                "bed_lookup_scatter",
                &p.bed_lookup_scatter,
                bed_wg,
            );
        }
        if particle_wg > 0 {
            timed_pass(&mut encoder, &mut rec, bg, "p2g", &p.p2g, particle_wg);
        }
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "grid_update",
            &p.grid_update,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "boundary_project",
            &p.boundary_project,
            cell_wg,
        );
        if sparse_pressure {
            timed_pass(
                &mut encoder,
                &mut rec,
                bg,
                "sparse_tiles_clear",
                &p.sparse_tiles_clear,
                sparse_clear_wg,
            );
        }
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "classify_cells",
            &p.classify_cells,
            cell_wg,
        );

        // Pressure: interleaved red/black GS in a single pass (matches production).
        // Sparse over-dispatches one workgroup per tile; dense one per 64 cells.
        {
            let timestamp_writes = rec.writes("pressure_solve");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pressure_solve"),
                timestamp_writes,
            });
            pass.set_bind_group(0, bg, &[]);
            for _ in 0..pressure_pairs {
                if sparse_pressure {
                    pass.set_pipeline(&p.pressure_rbgs_red_sparse);
                    pass.dispatch_workgroups(tile_wg, 1, 1);
                    pass.set_pipeline(&p.pressure_rbgs_black_sparse);
                    pass.dispatch_workgroups(tile_wg, 1, 1);
                } else {
                    pass.set_pipeline(&p.pressure_rbgs_red);
                    pass.dispatch_workgroups(cell_wg, 1, 1);
                    pass.set_pipeline(&p.pressure_rbgs_black);
                    pass.dispatch_workgroups(cell_wg, 1, 1);
                }
            }
        }

        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "project_pressure",
            &p.project_pressure,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "boundary_project",
            &p.boundary_project,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "pressure_residual",
            &p.pressure_residual,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "packing_prepare",
            &p.packing_prepare,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "packing_apply",
            &p.packing_apply,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "boundary_project",
            &p.boundary_project,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "viscosity_prepare",
            &p.viscosity_prepare,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "viscosity_apply",
            &p.viscosity_apply,
            cell_wg,
        );
        timed_pass(
            &mut encoder,
            &mut rec,
            bg,
            "boundary_project",
            &p.boundary_project,
            cell_wg,
        );
        if particle_wg > 0 {
            timed_pass(&mut encoder, &mut rec, bg, "g2p", &p.g2p, particle_wg);
            timed_pass(
                &mut encoder,
                &mut rec,
                bg,
                "bed_coupling",
                &p.bed_coupling,
                particle_wg,
            );
        }
        if bed_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                bg,
                "extraction_advect",
                &p.extraction_advect,
                bed_wg,
            );
            timed_pass(
                &mut encoder,
                &mut rec,
                bg,
                "bed_dynamics",
                &p.bed_dynamics,
                bed_wg,
            );
        }
        if particle_wg > 0 {
            timed_pass(
                &mut encoder,
                &mut rec,
                bg,
                "prepare_render",
                &p.prepare_render,
                particle_wg,
            );
        }

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
            for &(label, b, e) in &rec.scopes {
                let delta = ticks[e as usize].saturating_sub(ticks[b as usize]);
                let pass_ms = delta as f64 * t.period_ns as f64 / 1.0e6;
                *sums.per_label_ms.entry(label).or_insert(0.0) += pass_ms;
                *sums.per_label_count.entry(label).or_insert(0) += 1;
                sums.gpu_passes_ms += pass_ms;
            }
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
struct Metadata {
    solver_run: SolverRunMetadata,
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
    metadata: Metadata,
    frame_timings: FrameTimings,
    gpu_passes: Vec<PassStat>,
    bottlenecks: Vec<String>,
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

fn scene_settings(name: &str) -> MpmSettings {
    match name {
        "free_stream" => MpmSettings::benchmark_free_stream(),
        "water_block" => MpmSettings::benchmark_filter_water_block(),
        _ => MpmSettings::benchmark_center_pour(),
    }
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

#[test]
#[ignore = "profiling harness; run explicitly with --ignored --release"]
fn profile_mpm_pipeline() {
    run_profile_mpm_pipeline(
        "profile_mpm_pipeline",
        ProfilerRunOptions::from_legacy_env(),
        solver_run_metadata(&[SolverSpec::Mpm], SolverSpec::Mpm),
    );
}

#[test]
#[ignore = "solver-neutral profiling harness; run explicitly with --ignored --release"]
fn profile_solvers() {
    run_profile_solvers_from_env();
}

fn run_profile_solvers_from_env() {
    if profile_arg_flag("list-solvers") || env_flag("COFFEE_SIM_PROFILE_LIST_SOLVERS") {
        print!("{}", solver_list_summary());
        return;
    }

    let solvers = selected_solver_specs_from_env();
    let options = ProfilerRunOptions::from_profile_args();
    if profile_arg_flag("dry-run-json") || env_flag("COFFEE_SIM_PROFILE_DRY_RUN_JSON") {
        print_dry_run_plan(&solvers, &options, true);
        return;
    }
    if profile_arg_flag("dry-run") || env_flag("COFFEE_SIM_PROFILE_DRY_RUN") {
        print_dry_run_plan(&solvers, &options, false);
        return;
    }

    for &solver in &solvers {
        match solver {
            SolverSpec::Mpm => run_profile_mpm_pipeline(
                "profile_solvers",
                options.clone(),
                solver_run_metadata(&solvers, solver),
            ),
        }
    }
}

fn selected_solver_specs_from_env() -> Vec<SolverSpec> {
    let value = profile_arg_value("solvers")
        .or_else(|| profile_arg_value("solver"))
        .or_else(|| std::env::var("COFFEE_SIM_PROFILE_SOLVERS").ok())
        .or_else(|| std::env::var("COFFEE_SIM_PROFILE_SOLVER").ok())
        .unwrap_or_else(|| "mpm".into());
    parse_solver_specs(&value).unwrap_or_else(|err| panic!("{err}"))
}

#[derive(Serialize)]
struct DryRunPlan {
    version: u32,
    solvers: Vec<String>,
    scene: String,
    warmup_frames: u32,
    measured_frames: u32,
    calibration_frames: u32,
    require_gpu: bool,
    solver_runs: Vec<SolverRunMetadata>,
}

fn dry_run_plan(solvers: &[SolverSpec], options: &ProfilerRunOptions) -> DryRunPlan {
    DryRunPlan {
        version: 1,
        solvers: solvers
            .iter()
            .map(|solver| solver.id().to_string())
            .collect(),
        scene: options.scene.clone(),
        warmup_frames: options.warmup,
        measured_frames: options.measured,
        calibration_frames: options.calibration,
        require_gpu: options.require_gpu,
        solver_runs: solvers
            .iter()
            .map(|&solver| solver_run_metadata(solvers, solver))
            .collect(),
    }
}

fn print_dry_run_plan(solvers: &[SolverSpec], options: &ProfilerRunOptions, json: bool) {
    let plan = dry_run_plan(solvers, options);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).expect("dry-run json")
        );
        return;
    }
    println!(
        "profile_solvers dry run: scene={} solvers={} warmup={} frames={} cal={} require_gpu={}",
        plan.scene,
        plan.solvers.join(","),
        plan.warmup_frames,
        plan.measured_frames,
        plan.calibration_frames,
        plan.require_gpu
    );
}

#[test]
fn profile_solvers_dry_run_plan_records_mpm_metadata() {
    let options = ProfilerRunOptions {
        scene: "center_pour".into(),
        warmup: 1,
        measured: 2,
        calibration: 3,
        require_gpu: false,
    };
    let solvers = parse_solver_specs("all").expect("all resolves");
    let plan = dry_run_plan(&solvers, &options);
    assert_eq!(plan.version, 1);
    assert_eq!(plan.solvers, vec!["mpm"]);
    assert_eq!(plan.scene, "center_pour");
    assert_eq!(plan.solver_runs.len(), 1);
    assert_eq!(plan.solver_runs[0].current, "mpm");
}

fn run_profile_mpm_pipeline(
    entrypoint: &str,
    options: ProfilerRunOptions,
    solver_run: SolverRunMetadata,
) {
    let Some(adapter) = request_adapter() else {
        if options.require_gpu {
            panic!("{entrypoint}: no GPU adapter available and require_gpu=true");
        }
        eprintln!("{entrypoint}: no GPU adapter available; skipping.");
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
    let scene = options.scene;
    let warmup = options.warmup;
    let measured = options.measured;
    let calibration = options.calibration;

    let mut settings = scene_settings(&scene);
    settings.sparse_pressure = std::env::var("COFFEE_SIM_PROFILE_SPARSE_PRESSURE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let grid_dims = settings.grid_dims;
    let total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
    let max_particles = settings.max_particles;
    let substeps = settings.substeps.max(1);
    let period_ns = queue.get_timestamp_period();

    if !timestamps_supported {
        eprintln!(
            "profile_mpm_pipeline: adapter lacks TIMESTAMP_QUERY; \
             reporting frame-level timings only (no per-pass breakdown)."
        );
    }

    let mut sim = MpmSim3D::new(&device, &queue, settings);

    // Warm up: fill the pour, settle the bed, warm the driver.
    for _ in 0..warmup {
        sim.step_frame(&device, &queue, FRAME_DT);
    }
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let bed_particles = sim.num_bed;
    let pressure_rbgs_pairs = sim
        .last_pressure_rbgs_pairs
        .max(sim.settings.pressure_rbgs_pairs);

    // Calibration: honest production-path frame cost (single batched pass).
    let mut production_ms = Vec::with_capacity(calibration as usize);
    for _ in 0..calibration {
        let t = Instant::now();
        sim.step_frame(&device, &queue, FRAME_DT);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        production_ms.push(ms(t));
    }

    // Instrumented measurement.
    let timer = timestamps_supported.then(|| GpuTimer::new(&device, period_ns));
    let mut wall_ms = Vec::with_capacity(measured as usize);
    let mut gpu_passes_sum = Vec::with_capacity(measured as usize);
    let mut gpu_wait = Vec::with_capacity(measured as usize);
    let mut gpu_unattributed = Vec::with_capacity(measured as usize);
    let mut cpu_emit = Vec::with_capacity(measured as usize);
    let mut cpu_uniforms = Vec::with_capacity(measured as usize);
    let mut cpu_encode = Vec::with_capacity(measured as usize);
    let mut cpu_submit = Vec::with_capacity(measured as usize);
    let mut per_label: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut label_dispatch_count: BTreeMap<&'static str, u64> = BTreeMap::new();

    // Capture the particle count at the true start of the measured window —
    // after warmup AND calibration, both of which keep emitting inflow.
    let water_particles_start = sim.num_water;
    for _ in 0..measured {
        let t = Instant::now();
        let sums = step_frame_instrumented(&mut sim, &device, &queue, FRAME_DT, timer.as_ref());
        wall_ms.push(ms(t));
        gpu_passes_sum.push(sums.gpu_passes_ms);
        gpu_wait.push(sums.gpu_wait_ms);
        gpu_unattributed.push((sums.gpu_wait_ms - sums.gpu_passes_ms).max(0.0));
        cpu_emit.push(sums.cpu_emit_ms);
        cpu_uniforms.push(sums.cpu_uniforms_ms);
        cpu_encode.push(sums.cpu_encode_ms);
        cpu_submit.push(sums.cpu_submit_ms);
        for (label, value) in sums.per_label_ms {
            per_label.entry(label).or_default().push(value);
        }
        for (label, count) in sums.per_label_count {
            *label_dispatch_count.entry(label).or_insert(0) += count as u64;
        }
    }
    let water_particles_end = sim.num_water;

    // Build per-pass stats, sorted by mean descending.
    let total_pass_mean: f64 = per_label
        .values()
        .map(|v| v.iter().sum::<f64>() / v.len().max(1) as f64)
        .sum();
    let mut gpu_passes: Vec<PassStat> = per_label
        .into_iter()
        .map(|(label, samples)| {
            let stat = Stat::from_samples(samples);
            let passes =
                *label_dispatch_count.get(label).unwrap_or(&0) as f64 / measured.max(1) as f64;
            PassStat {
                label: label.to_string(),
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
        instrumented_wall: Stat::from_samples(wall_ms),
        gpu_passes_sum: Stat::from_samples(gpu_passes_sum),
        gpu_wait: Stat::from_samples(gpu_wait),
        gpu_unattributed: Stat::from_samples(gpu_unattributed),
        cpu_emit: Stat::from_samples(cpu_emit),
        cpu_uniforms: Stat::from_samples(cpu_uniforms),
        cpu_encode: Stat::from_samples(cpu_encode),
        cpu_submit: Stat::from_samples(cpu_submit),
    };

    let mut bottlenecks = Vec::new();
    for pass in gpu_passes.iter().take(5) {
        bottlenecks.push(format!(
            "{}: {:.3} ms/frame ({:.1}% of GPU pass time, {:.0} passes/frame)",
            pass.label, pass.mean_ms, pass.share_of_gpu_pct, pass.passes_per_frame
        ));
    }
    let cpu_total = frame_timings.cpu_emit.mean_ms
        + frame_timings.cpu_uniforms.mean_ms
        + frame_timings.cpu_encode.mean_ms
        + frame_timings.cpu_submit.mean_ms;
    let cpu_gpu_verdict = if !timestamps_supported {
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
        metadata: Metadata {
            solver_run,
            scene: scene.clone(),
            adapter: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            timestamps_supported,
            timestamp_period_ns: period_ns,
            warmup_frames: warmup,
            measured_frames: measured,
            calibration_frames: calibration,
            substeps_per_frame: substeps,
            frame_dt_s: FRAME_DT,
            grid_dims,
            total_cells,
            max_particles,
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

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    let path = output_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    std::fs::write(&path, &json).expect("write profile json");

    // Console summary (visible with --nocapture).
    println!("\n=== MPM profile ({}) ===", report.metadata.scene);
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

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
//! cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
//! ```
//!
//! Tunable via environment variables:
//! - `COFFEE_SIM_PROFILE_SCENE`   scene preset: `center_pour` (default) | `free_stream` | `water_block`
//! - `COFFEE_SIM_PROFILE_SOLVER`  solver spec: `rbgs` (default) | `mpm:jacobi-cg` | `mpm:sparse-cg` | `dfsph` | `xpbd`
//! - `COFFEE_SIM_PROFILE_SOLVERS` comma-separated runnable solver specs, or `all`
//! - `COFFEE_SIM_PROFILE_PRESSURE_OPERATOR` pressure operator: `collocated` (default)
//! - `COFFEE_SIM_PROFILE_CG_ITERATIONS` CG iterations per substep (defaults to scene RBGS pairs)
//! - `COFFEE_SIM_PROFILE_WARMUP`  frames to run before measuring (default 60)
//! - `COFFEE_SIM_PROFILE_FRAMES`  instrumented frames to measure (default 120)
//! - `COFFEE_SIM_PROFILE_CAL`     production `step_frame` calibration frames (default 30)
//! - `COFFEE_SIM_PROFILE_OUT`     output JSON path (default `<repo>/target/coffee-sim-profile.json`)
//! - `COFFEE_SIM_PROFILE_ARGS`    optional CLI-style overrides, e.g.
//!   `--solvers all --scene center_pour --warmup 10 --frames 30 --cal 5 --out target/profile.json`
//!   or kwargs-style `solvers=all scene=center_pour frames=30 out=target/profile.json`
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
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::mpsc;
use std::time::Instant;

use bytemuck::cast_slice;
use serde::Serialize;

use super::inflow::{EmissionResult, MASS_UNITS_PER_ML, PARTICLES_PER_ML};
use super::pressure::{PressureContext, PressureOperatorKind, PressureSolverKind};
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

    fn runnable_specs() -> Vec<Self> {
        let mut specs: Vec<_> = PressureSolverKind::ALL
            .iter()
            .copied()
            .map(Self::mpm)
            .collect();
        specs.push(Self::Dfsph);
        specs.push(Self::Xpbd {
            solver: XpbdSolverKind::Gpu,
        });
        specs
    }

    fn supports_pressure_operator(self, operator: PressureOperatorKind) -> bool {
        match self {
            Self::Mpm {
                pressure: PressureSolverKind::Rbgs | PressureSolverKind::SparseCg,
            } => operator == PressureOperatorKind::Collocated,
            Self::Mpm {
                pressure: PressureSolverKind::JacobiCg,
            } => true,
            Self::Dfsph => operator == PressureOperatorKind::Collocated,
            Self::Xpbd { .. } => true,
        }
    }

    fn profile(self, ctx: &ProfilerDeviceContext<'_>, run: &ProfilerRunConfig<'_>) {
        assert!(
            self.supports_pressure_operator(run.pressure_operator),
            "solver '{self}' does not support pressure_operator='{}'; use mpm:jacobi-cg for the staggered operator",
            run.pressure_operator
        );
        profile_solver_backend(ctx, run, self);
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

#[derive(Default)]
struct ProfilerCliArgs {
    scene: Option<String>,
    solver: Option<String>,
    solvers: Option<String>,
    pressure_operator: Option<String>,
    warmup: Option<u32>,
    measured: Option<u32>,
    calibration: Option<u32>,
    output: Option<PathBuf>,
    cg_iterations: Option<u32>,
    dry_run: Option<bool>,
    dry_run_json: Option<bool>,
    list_solvers: Option<bool>,
    help: Option<bool>,
}

impl ProfilerCliArgs {
    fn from_env_args() -> Self {
        let profile_args = std::env::var("COFFEE_SIM_PROFILE_ARGS").ok();
        let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
        Self::from_env_and_args(profile_args.as_deref(), raw_args, !cfg!(test))
    }

    fn from_env_and_args<I, S>(
        profile_args: Option<&str>,
        process_args: I,
        accept_raw_process_args: bool,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut cli = Self::default();
        if let Some(value) = profile_args {
            cli.merge(Self::parse(value.split_whitespace()).unwrap_or_else(|err| panic!("{err}")));
        }

        let raw_args = process_args
            .into_iter()
            .map(|arg| arg.as_ref().to_string())
            .collect::<Vec<_>>();
        let mut args = raw_args.iter();
        while let Some(arg) = args.next() {
            if arg == "--profile" || arg == "--profile-args" {
                cli.merge(Self::parse(args).unwrap_or_else(|err| panic!("{err}")));
                return cli;
            }
        }

        if accept_raw_process_args && !raw_args.is_empty() {
            cli.merge(Self::parse(raw_args).unwrap_or_else(|err| panic!("{err}")));
        }

        cli
    }

    fn parse<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut parsed = Self::default();
        let mut args = args
            .into_iter()
            .map(|arg| arg.as_ref().to_string())
            .peekable();
        while let Some(arg) = args.next() {
            let (flag, inline_value) = match arg.split_once('=') {
                Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
                None => (arg, None),
            };
            let flag = match flag.as_str() {
                "scene" => "--scene".to_string(),
                "solver" => "--solver".to_string(),
                "solvers" => "--solvers".to_string(),
                "pressure_operator" | "pressure-operator" | "operator" => {
                    "--pressure-operator".to_string()
                }
                "warmup" => "--warmup".to_string(),
                "frames" | "measured" => "--frames".to_string(),
                "cal" | "calibration" => "--cal".to_string(),
                "out" | "output" => "--out".to_string(),
                "cg_iterations" | "cg-iterations" => "--cg-iterations".to_string(),
                "dry_run" | "dry-run" => "--dry-run".to_string(),
                "dry_run_json" | "dry-run-json" => "--dry-run-json".to_string(),
                "list_solvers" | "list-solvers" => "--list-solvers".to_string(),
                "help" => "--help".to_string(),
                _ => flag,
            };
            let mut take_value = |name: &str| -> Result<String, String> {
                if let Some(value) = inline_value.clone() {
                    return Ok(value);
                }
                args.next()
                    .ok_or_else(|| format!("missing value for profiler option {name}"))
            };
            match flag.as_str() {
                "--scene" => parsed.scene = Some(take_value("--scene")?),
                "--solver" => parsed.solver = Some(take_value("--solver")?),
                "--solvers" => parsed.solvers = Some(take_value("--solvers")?),
                "--pressure-operator" | "--operator" => {
                    parsed.pressure_operator = Some(take_value("--pressure-operator")?);
                }
                "--warmup" => {
                    parsed.warmup = Some(parse_positive_u32("--warmup", &take_value("--warmup")?)?)
                }
                "--frames" | "--measured" => {
                    parsed.measured =
                        Some(parse_positive_u32("--frames", &take_value("--frames")?)?);
                }
                "--cal" | "--calibration" => {
                    parsed.calibration = Some(parse_positive_u32("--cal", &take_value("--cal")?)?);
                }
                "--out" | "--output" => parsed.output = Some(PathBuf::from(take_value("--out")?)),
                "--cg-iterations" => {
                    parsed.cg_iterations = Some(parse_positive_u32(
                        "--cg-iterations",
                        &take_value("--cg-iterations")?,
                    )?);
                }
                "--dry-run" => {
                    parsed.dry_run = Some(
                        inline_value
                            .as_deref()
                            .map(parse_bool)
                            .transpose()?
                            .unwrap_or(true),
                    );
                }
                "--dry-run-json" => {
                    parsed.dry_run_json = Some(parse_optional_bool(inline_value.as_deref())?);
                }
                "--list-solvers" => {
                    parsed.list_solvers = Some(parse_optional_bool(inline_value.as_deref())?);
                }
                "--help" | "-h" => {
                    parsed.help = Some(parse_optional_bool(inline_value.as_deref())?);
                }
                other => {
                    return Err(format!(
                        "unknown profiler option '{other}'; supported options: \
                         --scene, --solver, --solvers, --pressure-operator, --warmup, --frames, --cal, --out, --cg-iterations, --dry-run, --dry-run-json, --list-solvers, --help, \
                         or kwargs scene=, solver=, solvers=, pressure_operator=, warmup=, frames=, cal=, out=, cg_iterations=, dry_run=, dry_run_json=, list_solvers=, help="
                    ));
                }
            }
        }
        Ok(parsed)
    }

    fn merge(&mut self, other: Self) {
        self.scene = other.scene.or(self.scene.take());
        self.solver = other.solver.or(self.solver.take());
        self.solvers = other.solvers.or(self.solvers.take());
        self.pressure_operator = other.pressure_operator.or(self.pressure_operator.take());
        self.warmup = other.warmup.or(self.warmup);
        self.measured = other.measured.or(self.measured);
        self.calibration = other.calibration.or(self.calibration);
        self.output = other.output.or(self.output.take());
        self.cg_iterations = other.cg_iterations.or(self.cg_iterations);
        self.dry_run = other.dry_run.or(self.dry_run);
        self.dry_run_json = other.dry_run_json.or(self.dry_run_json);
        self.list_solvers = other.list_solvers.or(self.list_solvers);
        self.help = other.help.or(self.help);
    }
}

fn parse_positive_u32(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} must be a positive integer, got '{value}'"))
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!(
            "--dry-run must be a boolean when a value is provided, got '{other}'"
        )),
    }
}

fn parse_optional_bool(value: Option<&str>) -> Result<bool, String> {
    value
        .map(parse_bool)
        .transpose()
        .map(|value| value.unwrap_or(true))
}

fn env_u32(name: &str) -> Option<u32> {
    env_u32_value(std::env::var(name).ok())
}

fn env_u32_value(value: Option<String>) -> Option<u32> {
    value.and_then(|v| v.parse::<u32>().ok()).filter(|v| *v > 0)
}

fn parse_solver_specs(value: &str) -> Vec<SolverSpec> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return runnable_solver_specs();
    }
    let specs = trimmed
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.parse::<SolverSpec>()
                .unwrap_or_else(|err| panic!("{err}"))
        })
        .collect::<Vec<_>>();
    assert!(
        !specs.is_empty(),
        "no profiler solvers selected; use a solver spec like mpm:rbgs, dfsph, xpbd:gpu, or all"
    );
    specs
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
            operator: sim.settings.pressure_operator,
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
            operator: sim.settings.pressure_operator,
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
        let dfsph = &sim.pipelines.dfsph;
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
            sim.pipelines.pressure.classify_pipeline_for(pressure_ctx),
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
            sim.pipelines.pressure.project_pipeline_for(pressure_ctx),
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
            sim.pipelines.pressure.residual_pipeline_for(pressure_ctx),
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
            for iteration in 0..XPBD_CONSTRAINT_ITERATIONS {
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
                    ProfilePassLabel::new("xpbd.apply_density", "xpbd_apply_density", "xpbd"),
                    &xpbd.apply_density,
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
                if iteration + 1 < XPBD_CONSTRAINT_ITERATIONS {
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
                }
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
    operator: String,
    iterations_per_substep: u32,
}

#[derive(Serialize)]
struct DfsphSolverMetadata {
    kind: String,
    divergence_iterations_per_substep: u32,
    density_iterations_per_substep: u32,
    grid_pressure_kind: String,
    grid_pressure_operator: String,
    grid_pressure_iterations_per_substep: u32,
}

#[derive(Serialize)]
struct XpbdSolverMetadata {
    kind: String,
    constraint_iterations_per_substep: u32,
}

#[derive(Clone, Debug, Serialize)]
struct SolverRunMetadata {
    requested: Vec<String>,
    current: String,
    ordinal: usize,
    count: usize,
    multiple_outputs: bool,
}

#[derive(Serialize)]
struct DryRunPlan {
    schema_version: u32,
    mode: &'static str,
    scene: String,
    pressure_operator: String,
    warmup_frames: u32,
    measured_frames: u32,
    calibration_frames: u32,
    base_output_path: String,
    multiple_outputs: bool,
    solvers: Vec<DryRunSolverPlan>,
}

#[derive(Serialize)]
struct DryRunSolverPlan {
    solver: String,
    metadata: SolverMetadata,
    output_path: String,
    grid_dims: [u32; 3],
    substeps: u32,
    max_particles: u32,
    pressure_rbgs_pairs: u32,
    pressure_cg_iterations: u32,
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
    solver_run: SolverRunMetadata,
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

fn profile_solver_specs(cli: &ProfilerCliArgs) -> Vec<SolverSpec> {
    profile_solver_specs_with_env(
        cli,
        std::env::var("COFFEE_SIM_PROFILE_SOLVERS").ok(),
        std::env::var("COFFEE_SIM_PROFILE_SOLVER").ok(),
    )
}

fn profile_solver_specs_with_env(
    cli: &ProfilerCliArgs,
    solvers_env: Option<String>,
    solver_env: Option<String>,
) -> Vec<SolverSpec> {
    if let Some(value) = cli.solvers.as_deref() {
        return parse_solver_specs(value);
    }
    if let Some(value) = cli.solver.as_deref() {
        return parse_solver_specs(value);
    }
    if let Some(value) = solvers_env {
        return parse_solver_specs(&value);
    }
    let solver = solver_env.unwrap_or_else(|| "rbgs".into());
    parse_solver_specs(&solver)
}

fn runnable_solver_specs() -> Vec<SolverSpec> {
    SolverSpec::runnable_specs()
}

fn solver_list_summary() -> String {
    let mut summary = String::from("available profiler solvers:\n");
    for solver in runnable_solver_specs() {
        summary.push_str("  ");
        summary.push_str(&solver.to_string());
        summary.push('\n');
    }
    summary.push_str("  all\n");
    summary
}

fn profiler_usage() -> &'static str {
    "usage: profile_solvers [OPTIONS]\n\
\n\
Options:\n\
  --scene <center_pour|free_stream|water_block>\n\
  --solver <SOLVER>          Run one solver, or all\n\
  --solvers <A,B|all>        Run multiple solvers\n\
  --pressure-operator <collocated|staggered>\n\
  --warmup <N> --frames <N> --cal <N>\n\
  --out <PATH>\n\
  --cg-iterations <N>\n\
  --dry-run[=true|false]     Print the resolved run plan without GPU work\n\
  --dry-run-json             Print the resolved run plan as JSON without GPU work\n\
  --list-solvers             Print available solver specs\n\
  -h, --help                 Print this help\n\
\n\
Kwargs are also accepted, e.g. scene=center_pour solvers=all frames=30.\n"
}

fn xpbd_profiler_timestamp_query_capacity(substeps: u32) -> u32 {
    let xpbd_passes = xpbd_profiled_gpu_passes_per_substep();
    let shared_tail_passes = 7;
    let passes_per_substep = xpbd_passes + shared_tail_passes;
    (substeps.max(1) * passes_per_substep * 2).max(MIN_TIMESTAMP_QUERY_CAPACITY)
}

fn xpbd_profiled_gpu_passes_per_substep() -> u32 {
    4 + XPBD_CONSTRAINT_ITERATIONS * 3 + (XPBD_CONSTRAINT_ITERATIONS - 1) * 2
}

fn dfsph_profiler_timestamp_query_capacity(substeps: u32) -> u32 {
    // Worst-case instrumented DFSPH substep: DFSPH neighbor/constraint passes
    // plus the shared dense MPM grid/pressure/bed/render tail.
    let passes_per_substep = 40;
    (substeps.max(1) * passes_per_substep * 2).max(MIN_TIMESTAMP_QUERY_CAPACITY)
}

fn output_path(cli: &ProfilerCliArgs) -> PathBuf {
    output_path_with_env(cli, std::env::var("COFFEE_SIM_PROFILE_OUT").ok())
}

fn output_path_with_env(cli: &ProfilerCliArgs, output_env: Option<String>) -> PathBuf {
    if let Some(path) = cli.output.as_ref() {
        return path.clone();
    }
    if let Some(p) = output_env {
        return PathBuf::from(p);
    }
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/coffee-sim-profile.json"
    ))
}

fn output_path_for_solver(base: &Path, solver: SolverSpec, multiple: bool) -> PathBuf {
    if !multiple {
        return base.to_path_buf();
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
    let specs = runnable_solver_specs();
    assert_eq!(
        specs,
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
        specs.iter()
            .any(|solver| matches!(solver, SolverSpec::Xpbd { .. })),
        "XPBD GPU path should be included in `all` so same-scene solver comparisons do not require code changes"
    );
}

#[test]
fn profiler_pressure_operator_support_is_explicit() {
    assert!(SolverSpec::mpm(PressureSolverKind::JacobiCg)
        .supports_pressure_operator(PressureOperatorKind::Staggered));
    assert!(!SolverSpec::mpm(PressureSolverKind::Rbgs)
        .supports_pressure_operator(PressureOperatorKind::Staggered));
    assert!(!SolverSpec::mpm(PressureSolverKind::SparseCg)
        .supports_pressure_operator(PressureOperatorKind::Staggered));
    assert!(!SolverSpec::Dfsph.supports_pressure_operator(PressureOperatorKind::Staggered));
    assert!(SolverSpec::Xpbd {
        solver: XpbdSolverKind::Gpu
    }
    .supports_pressure_operator(PressureOperatorKind::Staggered));
}

#[test]
fn profiler_xpbd_schedule_refreshes_hash_and_splits_density_apply() {
    assert_eq!(
        xpbd_profiled_gpu_passes_per_substep(),
        4 + XPBD_CONSTRAINT_ITERATIONS * 3 + (XPBD_CONSTRAINT_ITERATIONS - 1) * 2
    );
    assert!(
        xpbd_profiler_timestamp_query_capacity(1)
            >= (xpbd_profiled_gpu_passes_per_substep() + 7) * 2
    );
}

#[test]
fn profiler_dfsph_timestamp_capacity_covers_shared_pressure_schedule() {
    assert!(dfsph_profiler_timestamp_query_capacity(1) >= 80);
    assert!(dfsph_profiler_timestamp_query_capacity(2) >= 160);
}

#[test]
fn profiler_runnable_solver_ids_are_unique() {
    let mut ids = runnable_solver_specs()
        .into_iter()
        .map(SolverSpec::id)
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), SolverSpec::runnable_specs().len());
}

#[test]
fn profiler_cli_args_parse_key_value_and_separate_forms() {
    let args = ProfilerCliArgs::parse([
        "--solvers=all",
        "--scene",
        "water_block",
        "--pressure-operator=collocated",
        "--warmup=2",
        "--frames",
        "3",
        "--cal",
        "4",
        "--out",
        "target/profile.json",
        "--cg-iterations",
        "5",
    ])
    .expect("profiler args parse");

    assert_eq!(args.solvers.as_deref(), Some("all"));
    assert_eq!(args.scene.as_deref(), Some("water_block"));
    assert_eq!(args.pressure_operator.as_deref(), Some("collocated"));
    assert_eq!(args.warmup, Some(2));
    assert_eq!(args.measured, Some(3));
    assert_eq!(args.calibration, Some(4));
    assert_eq!(args.output, Some(PathBuf::from("target/profile.json")));
    assert_eq!(args.cg_iterations, Some(5));
}

#[test]
fn profiler_cli_discovery_flags_do_not_need_gpu() {
    let args = ProfilerCliArgs::parse(["--list-solvers", "--help"]).expect("profiler args parse");
    assert_eq!(args.list_solvers, Some(true));
    assert_eq!(args.help, Some(true));

    let solvers = solver_list_summary();
    assert!(solvers.contains("mpm:rbgs"));
    assert!(solvers.contains("mpm:jacobi-cg"));
    assert!(solvers.contains("mpm:sparse-cg"));
    assert!(solvers.contains("dfsph"));
    assert!(solvers.contains("xpbd:gpu"));
    assert!(solvers.contains("all"));
    assert!(profiler_usage().contains("--solvers <A,B|all>"));
}

#[test]
fn profiler_cli_args_parse_kwargs_forms() {
    let args = ProfilerCliArgs::parse([
        "solver=xpbd",
        "scene=center_pour",
        "pressure_operator=collocated",
        "warmup=6",
        "measured=7",
        "calibration=8",
        "output=target/profile.json",
        "cg_iterations=9",
        "dry_run=true",
        "dry_run_json=true",
    ])
    .expect("profiler kwargs parse");

    assert_eq!(args.solver.as_deref(), Some("xpbd"));
    assert_eq!(args.scene.as_deref(), Some("center_pour"));
    assert_eq!(args.pressure_operator.as_deref(), Some("collocated"));
    assert_eq!(args.warmup, Some(6));
    assert_eq!(args.measured, Some(7));
    assert_eq!(args.calibration, Some(8));
    assert_eq!(args.output, Some(PathBuf::from("target/profile.json")));
    assert_eq!(args.cg_iterations, Some(9));
    assert_eq!(args.dry_run, Some(true));
    assert_eq!(args.dry_run_json, Some(true));
}

#[test]
fn profiler_native_raw_cli_args_select_all_solvers_on_one_scene() {
    let args = ProfilerCliArgs::from_env_and_args(
        None,
        [
            "--scene",
            "water_block",
            "--solvers",
            "all",
            "--frames",
            "3",
            "--warmup",
            "2",
            "--cal",
            "1",
            "--out",
            "target/native-profile.json",
        ],
        true,
    );
    let selection = ProfileSelection::from_cli_with_env(&args, |_| None);

    assert_eq!(selection.scene, ProfileScene::WaterBlock);
    assert_eq!(selection.solvers, runnable_solver_specs());
    assert_eq!(
        selection.solver_ids,
        runnable_solver_specs()
            .into_iter()
            .map(SolverSpec::id)
            .collect::<Vec<_>>()
    );
    assert_eq!(selection.warmup, 2);
    assert_eq!(selection.measured, 3);
    assert_eq!(selection.calibration, 1);
    assert_eq!(
        selection.base_output_path,
        PathBuf::from("target/native-profile.json")
    );
    assert!(selection.multiple_outputs);
}

#[test]
fn profiler_env_solver_selection_uses_same_scene_path() {
    let args = ProfilerCliArgs::from_env_and_args(
        Some("scene=water_block frames=5 warmup=4 cal=3"),
        std::iter::empty::<&str>(),
        false,
    );
    let selection = ProfileSelection::from_cli_with_env(&args, |name| match name {
        "COFFEE_SIM_PROFILE_SOLVERS" => Some("all".to_string()),
        "COFFEE_SIM_PROFILE_OUT" => Some("target/env-profile.json".to_string()),
        _ => None,
    });

    assert_eq!(selection.scene, ProfileScene::WaterBlock);
    assert_eq!(selection.solvers, runnable_solver_specs());
    assert_eq!(selection.warmup, 4);
    assert_eq!(selection.measured, 5);
    assert_eq!(selection.calibration, 3);
    assert_eq!(
        selection.base_output_path,
        PathBuf::from("target/env-profile.json")
    );
    assert!(selection.multiple_outputs);
}

#[test]
fn profiler_dry_run_summary_lists_all_solver_outputs_and_scene_settings() {
    let args = ProfilerCliArgs::from_env_and_args(
        None,
        [
            "--scene",
            "center_pour",
            "--solvers",
            "all",
            "--frames",
            "1",
            "--warmup",
            "1",
            "--cal",
            "1",
            "--out",
            "target/dry-profile.json",
            "--dry-run",
        ],
        true,
    );
    assert_eq!(args.dry_run, Some(true));

    let selection = ProfileSelection::from_cli_with_env(&args, |_| None);
    let summary = selection.dry_run_summary();

    assert!(summary.contains("coffee-sim profiler dry run"));
    assert!(summary.contains("scene: center_pour"));
    assert!(summary.contains("solvers: mpm-rbgs, mpm-jacobi-cg, mpm-sparse-cg, dfsph, xpbd-gpu"));
    assert!(summary.contains("target/dry-profile-mpm-rbgs.json"));
    assert!(summary.contains("target/dry-profile-mpm-jacobi-cg.json"));
    assert!(summary.contains("target/dry-profile-mpm-sparse-cg.json"));
    assert!(summary.contains("target/dry-profile-dfsph.json"));
    assert!(summary.contains("target/dry-profile-xpbd-gpu.json"));
    assert!(summary.contains("grid=80x115x80"));
    assert!(summary.contains("rbgs_pairs=40"));
    assert!(summary.contains("cg_iterations=40"));
}

#[test]
fn profiler_dry_run_json_is_machine_readable_same_scene_plan() {
    let args = ProfilerCliArgs::from_env_and_args(
        None,
        [
            "--scene",
            "center_pour",
            "--solvers",
            "all",
            "--frames",
            "1",
            "--warmup",
            "1",
            "--cal",
            "1",
            "--out",
            "target/dry-profile.json",
            "--dry-run-json",
        ],
        true,
    );
    assert_eq!(args.dry_run_json, Some(true));

    let selection = ProfileSelection::from_cli_with_env(&args, |_| None);
    let value: serde_json::Value =
        serde_json::from_str(&selection.dry_run_json()).expect("dry run json parses");
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["mode"], "dry_run");
    assert_eq!(value["scene"], "center_pour");
    assert_eq!(value["solvers"].as_array().expect("solver array").len(), 5);
    assert_eq!(value["solvers"][0]["solver"], "mpm:rbgs");
    assert_eq!(value["solvers"][4]["solver"], "xpbd:gpu");
    assert_eq!(value["solvers"][0]["metadata"]["backend"], "mpm");
    assert_eq!(value["solvers"][0]["metadata"]["pressure"]["kind"], "rbgs");
    assert_eq!(
        value["solvers"][0]["metadata"]["pressure"]["iterations_per_substep"],
        40
    );
    assert_eq!(value["solvers"][3]["metadata"]["dfsph"]["kind"], "water");
    assert_eq!(
        value["solvers"][4]["metadata"]["xpbd"]["constraint_iterations_per_substep"],
        XPBD_CONSTRAINT_ITERATIONS
    );
    assert_eq!(
        value["solvers"][0]["grid_dims"],
        serde_json::json!([80, 115, 80])
    );
    assert_eq!(value["solvers"][0]["pressure_rbgs_pairs"], 40);
    assert_eq!(value["solvers"][0]["pressure_cg_iterations"], 40);
}

#[test]
fn profiler_cli_solver_overrides_solver_env() {
    let args = ProfilerCliArgs::parse(["--solver", "xpbd:gpu"]).expect("profiler args parse");
    let selection = ProfileSelection::from_cli_with_env(&args, |name| match name {
        "COFFEE_SIM_PROFILE_SOLVERS" => Some("all".to_string()),
        "COFFEE_SIM_PROFILE_SOLVER" => Some("dfsph".to_string()),
        _ => None,
    });

    assert_eq!(
        selection.solvers,
        vec![SolverSpec::Xpbd {
            solver: XpbdSolverKind::Gpu
        }]
    );
    assert!(!selection.multiple_outputs);
}

#[test]
fn profiler_all_solvers_share_selected_scene_settings() {
    let base = PathBuf::from("target/profile.json");
    let scene = ProfileScene::FreeStream;
    let scene_settings = scene.mpm_settings();
    let run = ProfilerRunConfig {
        scene,
        warmup: 1,
        measured: 1,
        calibration: 1,
        cg_iterations: None,
        pressure_operator: PressureOperatorKind::Collocated,
        solver_ids: runnable_solver_specs()
            .into_iter()
            .map(SolverSpec::id)
            .collect(),
        base_output_path: &base,
        multiple_outputs: true,
    };

    for solver in runnable_solver_specs() {
        let settings = settings_for_solver(solver, &run);
        assert_eq!(settings.grid_dims, scene_settings.grid_dims, "{solver}");
        assert_eq!(settings.substeps, scene_settings.substeps, "{solver}");
        assert_eq!(
            settings.max_particles, scene_settings.max_particles,
            "{solver}"
        );
        assert_eq!(
            settings.pressure_rbgs_pairs, scene_settings.pressure_rbgs_pairs,
            "{solver}"
        );
        assert_eq!(
            settings.pressure_cg_iterations, scene_settings.pressure_cg_iterations,
            "{solver}"
        );
    }
}

#[test]
fn profiler_mpm_cg_defaults_to_scene_cg_budget() {
    let base = PathBuf::from("target/profile.json");
    let mut scene_settings = ProfileScene::CenterPour.mpm_settings();
    scene_settings.pressure_rbgs_pairs = 12;
    scene_settings.pressure_cg_iterations = 34;
    let run = ProfilerRunConfig {
        scene: ProfileScene::CenterPour,
        warmup: 1,
        measured: 1,
        calibration: 1,
        cg_iterations: None,
        pressure_operator: PressureOperatorKind::Collocated,
        solver_ids: vec!["mpm-sparse-cg".to_string()],
        base_output_path: &base,
        multiple_outputs: false,
    };

    let settings = settings_for_solver_with_base(
        SolverSpec::mpm(PressureSolverKind::SparseCg),
        &run,
        scene_settings,
    );
    assert_eq!(settings.pressure_rbgs_pairs, 12);
    assert_eq!(settings.pressure_cg_iterations, 34);
}

#[test]
fn profiler_harness_args_require_profile_sentinel() {
    let ignored_harness_args =
        ProfilerCliArgs::from_env_and_args(None, ["--solver", "xpbd:gpu"], false);
    assert!(ignored_harness_args.solver.is_none());

    let forwarded_args = ProfilerCliArgs::from_env_and_args(
        None,
        ["profile_solvers", "--profile", "--solver", "xpbd:gpu"],
        false,
    );
    assert_eq!(forwarded_args.solver.as_deref(), Some("xpbd:gpu"));
}

#[test]
fn profiler_later_args_can_disable_dry_run() {
    let args = ProfilerCliArgs::from_env_and_args(
        Some("dry_run=true solver=dfsph"),
        ["--solver", "xpbd:gpu", "--dry-run=false"],
        true,
    );

    assert_eq!(args.solver.as_deref(), Some("xpbd:gpu"));
    assert_eq!(args.dry_run, Some(false));
}

#[test]
fn profiler_later_args_can_disable_discovery_flags() {
    let args = ProfilerCliArgs::from_env_and_args(
        Some("dry_run_json=true list_solvers=true help=true"),
        [
            "--dry-run-json=false",
            "--list-solvers=false",
            "--help=false",
        ],
        true,
    );

    assert_eq!(args.dry_run_json, Some(false));
    assert_eq!(args.list_solvers, Some(false));
    assert_eq!(args.help, Some(false));
}

#[test]
fn profiler_cli_solver_overrides_expand_to_specs() {
    let args = ProfilerCliArgs::parse(["--solvers", "mpm:jacobi-cg,dfsph,xpbd"])
        .expect("profiler args parse");
    assert_eq!(
        profile_solver_specs(&args),
        vec![
            SolverSpec::mpm(PressureSolverKind::JacobiCg),
            SolverSpec::Dfsph,
            SolverSpec::Xpbd {
                solver: XpbdSolverKind::Gpu
            },
        ]
    );
}

#[test]
fn profiler_cli_singular_solver_all_expands_to_runnable_specs() {
    let args = ProfilerCliArgs::parse(["--solver", "all"]).expect("profiler args parse");
    assert_eq!(profile_solver_specs(&args), runnable_solver_specs());
}

#[test]
fn profiler_output_path_for_solver_keeps_multi_solver_outputs_distinct() {
    let base = PathBuf::from("target/coffee-sim-profile.json");
    assert_eq!(
        output_path_for_solver(&base, SolverSpec::Dfsph, false),
        base
    );
    assert_eq!(
        output_path_for_solver(&base, SolverSpec::Dfsph, true),
        PathBuf::from("target/coffee-sim-profile-dfsph.json")
    );
    assert_eq!(
        output_path_for_solver(
            &base,
            SolverSpec::Xpbd {
                solver: XpbdSolverKind::Gpu
            },
            true,
        ),
        PathBuf::from("target/coffee-sim-profile-xpbd-gpu.json")
    );
}

#[test]
fn profiler_solver_run_metadata_records_requested_same_scene_set() {
    let base = PathBuf::from("target/profile.json");
    let solvers = vec![
        SolverSpec::mpm(PressureSolverKind::Rbgs),
        SolverSpec::Dfsph,
        SolverSpec::Xpbd {
            solver: XpbdSolverKind::Gpu,
        },
    ];
    let run = ProfilerRunConfig {
        scene: ProfileScene::CenterPour,
        warmup: 1,
        measured: 1,
        calibration: 1,
        cg_iterations: None,
        pressure_operator: PressureOperatorKind::Collocated,
        solver_ids: solvers.iter().copied().map(SolverSpec::id).collect(),
        base_output_path: &base,
        multiple_outputs: true,
    };

    let metadata = solver_run_metadata(&run, SolverSpec::Dfsph);
    assert_eq!(metadata.requested, vec!["mpm-rbgs", "dfsph", "xpbd-gpu"]);
    assert_eq!(metadata.current, "dfsph");
    assert_eq!(metadata.ordinal, 2);
    assert_eq!(metadata.count, 3);
    assert!(metadata.multiple_outputs);
}

#[test]
fn profiler_solver_settings_keep_scene_and_selected_solver_separate() {
    let base = PathBuf::from("target/profile.json");
    let run = ProfilerRunConfig {
        scene: ProfileScene::WaterBlock,
        warmup: 1,
        measured: 1,
        calibration: 1,
        cg_iterations: Some(17),
        pressure_operator: PressureOperatorKind::Collocated,
        solver_ids: vec!["mpm-sparse-cg".to_string()],
        base_output_path: &base,
        multiple_outputs: false,
    };

    let mpm = settings_for_solver(SolverSpec::mpm(PressureSolverKind::SparseCg), &run);
    assert_eq!(mpm.pressure_solver, PressureSolverKind::SparseCg);
    assert_eq!(mpm.pressure_operator, PressureOperatorKind::Collocated);
    assert_eq!(mpm.pressure_cg_iterations, 17);
    assert_eq!(
        mpm.grid_dims,
        ProfileScene::WaterBlock.mpm_settings().grid_dims
    );

    let dfsph = settings_for_solver(SolverSpec::Dfsph, &run);
    assert_eq!(
        dfsph.grid_dims, mpm.grid_dims,
        "backend comparisons should keep the selected scene geometry"
    );
    assert_eq!(
        dfsph.pressure_cg_iterations,
        ProfileScene::WaterBlock
            .mpm_settings()
            .pressure_cg_iterations,
        "non-MPM backend grid tail should inherit the scene pressure budget"
    );

    let xpbd = settings_for_solver(
        SolverSpec::Xpbd {
            solver: XpbdSolverKind::Gpu,
        },
        &run,
    );
    assert_eq!(
        xpbd.grid_dims, mpm.grid_dims,
        "XPBD backend should run on the same selected scene"
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
    cg_iterations: Option<u32>,
    pressure_operator: PressureOperatorKind,
    solver_ids: Vec<String>,
    base_output_path: &'a PathBuf,
    multiple_outputs: bool,
}

struct ProfileSelection {
    scene: ProfileScene,
    solvers: Vec<SolverSpec>,
    pressure_operator: PressureOperatorKind,
    solver_ids: Vec<String>,
    warmup: u32,
    measured: u32,
    calibration: u32,
    cg_iterations: Option<u32>,
    base_output_path: PathBuf,
    multiple_outputs: bool,
}

impl ProfileSelection {
    fn from_cli(cli: &ProfilerCliArgs) -> Self {
        Self::from_cli_with_env(cli, |name| std::env::var(name).ok())
    }

    fn from_cli_with_env<F>(cli: &ProfilerCliArgs, mut get_env: F) -> Self
    where
        F: FnMut(&str) -> Option<String>,
    {
        let scene_env = get_env("COFFEE_SIM_PROFILE_SCENE");
        let pressure_operator_env = get_env("COFFEE_SIM_PROFILE_PRESSURE_OPERATOR");
        let solvers_env = get_env("COFFEE_SIM_PROFILE_SOLVERS");
        let solver_env = get_env("COFFEE_SIM_PROFILE_SOLVER");
        let warmup_env = get_env("COFFEE_SIM_PROFILE_WARMUP");
        let frames_env = get_env("COFFEE_SIM_PROFILE_FRAMES");
        let calibration_env = get_env("COFFEE_SIM_PROFILE_CAL");
        let cg_iterations_env = get_env("COFFEE_SIM_PROFILE_CG_ITERATIONS");
        let output_env = get_env("COFFEE_SIM_PROFILE_OUT");

        let scene = cli
            .scene
            .clone()
            .or(scene_env)
            .unwrap_or_else(|| "center_pour".into())
            .parse::<ProfileScene>()
            .unwrap_or_else(|err| panic!("{err}"));
        let solvers = profile_solver_specs_with_env(cli, solvers_env, solver_env);
        let pressure_operator = cli
            .pressure_operator
            .clone()
            .or(pressure_operator_env)
            .unwrap_or_else(|| PressureOperatorKind::Collocated.to_string())
            .parse::<PressureOperatorKind>()
            .unwrap_or_else(|err| panic!("{err}"));
        for solver in &solvers {
            assert!(
                solver.supports_pressure_operator(pressure_operator),
                "solver '{solver}' does not support pressure_operator='{pressure_operator}'; use mpm:jacobi-cg for the staggered operator"
            );
        }
        let solver_ids = solvers
            .iter()
            .copied()
            .map(SolverSpec::id)
            .collect::<Vec<_>>();
        let warmup = cli
            .warmup
            .or_else(|| env_u32_value(warmup_env))
            .unwrap_or(DEFAULT_WARMUP_FRAMES);
        let measured = cli
            .measured
            .or_else(|| env_u32_value(frames_env))
            .unwrap_or(DEFAULT_MEASURED_FRAMES);
        let calibration = cli
            .calibration
            .or_else(|| env_u32_value(calibration_env))
            .unwrap_or(DEFAULT_CALIBRATION_FRAMES);
        let base_output_path = output_path_with_env(cli, output_env);
        let multiple_outputs = solvers.len() > 1;
        Self {
            scene,
            solvers,
            pressure_operator,
            solver_ids,
            warmup,
            measured,
            calibration,
            cg_iterations: cli
                .cg_iterations
                .or_else(|| env_u32_value(cg_iterations_env)),
            base_output_path,
            multiple_outputs,
        }
    }

    fn dry_run_summary(&self) -> String {
        let mut summary = String::new();
        use std::fmt::Write as _;

        let solver_ids = self
            .solvers
            .iter()
            .copied()
            .map(SolverSpec::id)
            .collect::<Vec<_>>();
        let _ = writeln!(summary, "coffee-sim profiler dry run");
        let _ = writeln!(summary, "scene: {}", self.scene);
        let _ = writeln!(summary, "solvers: {}", solver_ids.join(", "));
        let _ = writeln!(summary, "pressure_operator: {}", self.pressure_operator);
        let _ = writeln!(
            summary,
            "frames: warmup={} measured={} calibration={}",
            self.warmup, self.measured, self.calibration
        );
        let _ = writeln!(summary, "base_output: {}", self.base_output_path.display());
        let run = ProfilerRunConfig {
            scene: self.scene,
            warmup: self.warmup,
            measured: self.measured,
            calibration: self.calibration,
            cg_iterations: self.cg_iterations,
            pressure_operator: self.pressure_operator,
            solver_ids,
            base_output_path: &self.base_output_path,
            multiple_outputs: self.multiple_outputs,
        };
        for &solver in &self.solvers {
            let settings = settings_for_solver(solver, &run);
            let output =
                output_path_for_solver(&self.base_output_path, solver, self.multiple_outputs);
            let _ = writeln!(
                summary,
                "- {} output={} grid={}x{}x{} substeps={} max_particles={} rbgs_pairs={} cg_iterations={}",
                solver,
                output.display(),
                settings.grid_dims[0],
                settings.grid_dims[1],
                settings.grid_dims[2],
                settings.substeps,
                settings.max_particles,
                settings.pressure_rbgs_pairs,
                settings.pressure_cg_iterations
            );
        }
        summary
    }

    fn dry_run_plan(&self) -> DryRunPlan {
        let solver_ids = self
            .solvers
            .iter()
            .copied()
            .map(SolverSpec::id)
            .collect::<Vec<_>>();
        let run = ProfilerRunConfig {
            scene: self.scene,
            warmup: self.warmup,
            measured: self.measured,
            calibration: self.calibration,
            cg_iterations: self.cg_iterations,
            pressure_operator: self.pressure_operator,
            solver_ids,
            base_output_path: &self.base_output_path,
            multiple_outputs: self.multiple_outputs,
        };
        let solvers = self
            .solvers
            .iter()
            .copied()
            .map(|solver| {
                let settings = settings_for_solver(solver, &run);
                DryRunSolverPlan {
                    solver: solver.to_string(),
                    metadata: dry_run_solver_metadata(solver, &settings),
                    output_path: output_path_for_solver(
                        &self.base_output_path,
                        solver,
                        self.multiple_outputs,
                    )
                    .display()
                    .to_string(),
                    grid_dims: settings.grid_dims,
                    substeps: settings.substeps,
                    max_particles: settings.max_particles,
                    pressure_rbgs_pairs: settings.pressure_rbgs_pairs,
                    pressure_cg_iterations: settings.pressure_cg_iterations,
                }
            })
            .collect();

        DryRunPlan {
            schema_version: 1,
            mode: "dry_run",
            scene: self.scene.to_string(),
            pressure_operator: self.pressure_operator.to_string(),
            warmup_frames: self.warmup,
            measured_frames: self.measured,
            calibration_frames: self.calibration,
            base_output_path: self.base_output_path.display().to_string(),
            multiple_outputs: self.multiple_outputs,
            solvers,
        }
    }

    fn dry_run_json(&self) -> String {
        serde_json::to_string_pretty(&self.dry_run_plan()).expect("serialize dry run plan")
    }
}

struct ProfilerDeviceContext<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    info: &'a wgpu::AdapterInfo,
    timestamps_supported: bool,
    period_ns: f32,
}

fn write_report(path: &Path, report: &ProfileReport) {
    let json = serde_json::to_string_pretty(report).expect("serialize report");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    std::fs::write(path, json).expect("write profile json");
}

fn print_report_summary(path: &Path, report: &ProfileReport) {
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

fn solver_run_metadata(run: &ProfilerRunConfig<'_>, solver: SolverSpec) -> SolverRunMetadata {
    let current = solver.id();
    let ordinal = run
        .solver_ids
        .iter()
        .position(|id| id == &current)
        .map(|index| index + 1)
        .unwrap_or(1);
    SolverRunMetadata {
        requested: run.solver_ids.clone(),
        current,
        ordinal,
        count: run.solver_ids.len(),
        multiple_outputs: run.multiple_outputs,
    }
}

trait ProfileBackend {
    fn calibration_step(&mut self, sim: &mut MpmSim3D, device: &wgpu::Device, queue: &wgpu::Queue);

    fn measured_step(
        &mut self,
        sim: &mut MpmSim3D,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timer: Option<&GpuTimer>,
    ) -> FrameSums;

    fn timestamp_query_capacity(&self, sim: &MpmSim3D) -> u32;

    fn solver_metadata(&self, solver: SolverSpec, sim: &MpmSim3D) -> SolverMetadata;

    fn pressure_rbgs_pairs(&self, sim: &MpmSim3D) -> u32 {
        sim.last_pressure_rbgs_pairs
            .max(sim.settings.pressure_rbgs_pairs)
    }

    fn append_bottlenecks(&self, bottlenecks: &mut Vec<String>, frame_timings: &FrameTimings);
}

struct MpmProfileBackend;

impl ProfileBackend for MpmProfileBackend {
    fn calibration_step(&mut self, sim: &mut MpmSim3D, device: &wgpu::Device, queue: &wgpu::Queue) {
        sim.step_frame(device, queue, FRAME_DT);
    }

    fn measured_step(
        &mut self,
        sim: &mut MpmSim3D,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timer: Option<&GpuTimer>,
    ) -> FrameSums {
        step_frame_instrumented(sim, device, queue, FRAME_DT, timer)
    }

    fn timestamp_query_capacity(&self, sim: &MpmSim3D) -> u32 {
        sim.profiler_timestamp_query_capacity()
    }

    fn solver_metadata(&self, solver: SolverSpec, sim: &MpmSim3D) -> SolverMetadata {
        SolverMetadata {
            backend: solver.backend().to_string(),
            pressure: Some(PressureSolverMetadata {
                kind: sim.pressure_solver_kind().to_string(),
                operator: sim.pressure_operator_kind().to_string(),
                iterations_per_substep: sim.pressure_solver_iterations_per_substep(),
            }),
            dfsph: None,
            xpbd: None,
        }
    }

    fn append_bottlenecks(&self, bottlenecks: &mut Vec<String>, frame_timings: &FrameTimings) {
        let cpu_total = frame_timings.cpu_emit.mean_ms
            + frame_timings.cpu_uniforms.mean_ms
            + frame_timings.cpu_encode.mean_ms
            + frame_timings.cpu_submit.mean_ms;
        let cpu_gpu_verdict = if frame_timings.gpu_passes_sum.mean_ms <= 0.0 {
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
    }
}

struct DfsphProfileBackend;

impl ProfileBackend for DfsphProfileBackend {
    fn calibration_step(&mut self, sim: &mut MpmSim3D, device: &wgpu::Device, queue: &wgpu::Queue) {
        step_frame_dfsph_instrumented(sim, device, queue, FRAME_DT, None);
    }

    fn measured_step(
        &mut self,
        sim: &mut MpmSim3D,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timer: Option<&GpuTimer>,
    ) -> FrameSums {
        step_frame_dfsph_instrumented(sim, device, queue, FRAME_DT, timer)
    }

    fn timestamp_query_capacity(&self, sim: &MpmSim3D) -> u32 {
        dfsph_profiler_timestamp_query_capacity(sim.settings.substeps)
    }

    fn solver_metadata(&self, solver: SolverSpec, sim: &MpmSim3D) -> SolverMetadata {
        SolverMetadata {
            backend: solver.backend().to_string(),
            pressure: None,
            dfsph: Some(DfsphSolverMetadata {
                kind: "water".to_string(),
                divergence_iterations_per_substep: 1,
                density_iterations_per_substep: 2,
                grid_pressure_kind: sim.pressure_solver_kind().to_string(),
                grid_pressure_operator: sim.pressure_operator_kind().to_string(),
                grid_pressure_iterations_per_substep: sim.pressure_solver_iterations_per_substep(),
            }),
            xpbd: None,
        }
    }

    fn append_bottlenecks(&self, bottlenecks: &mut Vec<String>, _frame_timings: &FrameTimings) {
        bottlenecks.push(
            "DFSPH backend profiles GPU water pressure correction plus the shared MPM grid/bed/render tail; \
             DFSPH active pressure-tile compaction is staged separately and is not wired into the shared MPM shader."
                .to_string(),
        );
    }
}

struct XpbdProfileBackend {
    kind: XpbdSolverKind,
    pipelines: XpbdPipelines,
}

impl ProfileBackend for XpbdProfileBackend {
    fn calibration_step(&mut self, sim: &mut MpmSim3D, device: &wgpu::Device, queue: &wgpu::Queue) {
        step_frame_xpbd_instrumented(sim, &self.pipelines, device, queue, FRAME_DT, None);
    }

    fn measured_step(
        &mut self,
        sim: &mut MpmSim3D,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timer: Option<&GpuTimer>,
    ) -> FrameSums {
        step_frame_xpbd_instrumented(sim, &self.pipelines, device, queue, FRAME_DT, timer)
    }

    fn timestamp_query_capacity(&self, sim: &MpmSim3D) -> u32 {
        xpbd_profiler_timestamp_query_capacity(sim.settings.substeps)
    }

    fn solver_metadata(&self, solver: SolverSpec, _sim: &MpmSim3D) -> SolverMetadata {
        SolverMetadata {
            backend: solver.backend().to_string(),
            pressure: None,
            dfsph: None,
            xpbd: Some(XpbdSolverMetadata {
                kind: self.kind.to_string(),
                constraint_iterations_per_substep: XPBD_CONSTRAINT_ITERATIONS,
            }),
        }
    }

    fn pressure_rbgs_pairs(&self, _sim: &MpmSim3D) -> u32 {
        0
    }

    fn append_bottlenecks(&self, bottlenecks: &mut Vec<String>, _frame_timings: &FrameTimings) {
        bottlenecks.push(
            "XPBD backend uses GPU prediction, spatial hash, density constraints, bounds constraints, \
             and velocity update over the shared particle buffers; browser parity and physical validation \
             remain the next gates before treating it as production-equivalent."
                .to_string(),
        );
    }
}

fn settings_for_solver(solver: SolverSpec, run: &ProfilerRunConfig<'_>) -> MpmSettings {
    settings_for_solver_with_base(solver, run, run.scene.mpm_settings())
}

fn settings_for_solver_with_base(
    solver: SolverSpec,
    run: &ProfilerRunConfig<'_>,
    mut settings: MpmSettings,
) -> MpmSettings {
    match solver {
        SolverSpec::Mpm { pressure } => {
            settings.pressure_solver = pressure;
            settings.pressure_operator = run.pressure_operator;
            settings.pressure_cg_iterations =
                run.cg_iterations.unwrap_or(settings.pressure_cg_iterations);
        }
        SolverSpec::Dfsph => {
            settings.pressure_operator = run.pressure_operator;
        }
        SolverSpec::Xpbd { .. } => {}
    }
    settings
}

fn dry_run_solver_metadata(solver: SolverSpec, settings: &MpmSettings) -> SolverMetadata {
    match solver {
        SolverSpec::Mpm { pressure } => SolverMetadata {
            backend: solver.backend().to_string(),
            pressure: Some(PressureSolverMetadata {
                kind: pressure.to_string(),
                operator: settings.pressure_operator.to_string(),
                iterations_per_substep: settings.pressure_cg_iterations,
            }),
            dfsph: None,
            xpbd: None,
        },
        SolverSpec::Dfsph => SolverMetadata {
            backend: solver.backend().to_string(),
            pressure: None,
            dfsph: Some(DfsphSolverMetadata {
                kind: "water".to_string(),
                divergence_iterations_per_substep: 1,
                density_iterations_per_substep: 2,
                grid_pressure_kind: settings.pressure_solver.to_string(),
                grid_pressure_operator: settings.pressure_operator.to_string(),
                grid_pressure_iterations_per_substep: settings.pressure_cg_iterations,
            }),
            xpbd: None,
        },
        SolverSpec::Xpbd { solver: kind } => SolverMetadata {
            backend: solver.backend().to_string(),
            pressure: None,
            dfsph: None,
            xpbd: Some(XpbdSolverMetadata {
                kind: kind.to_string(),
                constraint_iterations_per_substep: XPBD_CONSTRAINT_ITERATIONS,
            }),
        },
    }
}

fn backend_for_solver(
    solver: SolverSpec,
    device: &wgpu::Device,
    sim: &MpmSim3D,
) -> Box<dyn ProfileBackend> {
    match solver {
        SolverSpec::Mpm { .. } => Box::new(MpmProfileBackend),
        SolverSpec::Dfsph => Box::new(DfsphProfileBackend),
        SolverSpec::Xpbd { solver: kind } => Box::new(XpbdProfileBackend {
            kind,
            pipelines: XpbdPipelines::new(device, &sim.buffers),
        }),
    }
}

fn profile_solver_backend(
    ctx: &ProfilerDeviceContext<'_>,
    run: &ProfilerRunConfig<'_>,
    solver: SolverSpec,
) {
    let settings = settings_for_solver(solver, run);
    let grid_dims = settings.grid_dims;
    let total_cells = grid_dims[0] * grid_dims[1] * grid_dims[2];
    let max_particles = settings.max_particles;
    let substeps = settings.substeps.max(1);
    let mut sim = MpmSim3D::new(ctx.device, ctx.queue, settings);
    let mut backend = backend_for_solver(solver, ctx.device, &sim);

    for _ in 0..run.warmup {
        backend.calibration_step(&mut sim, ctx.device, ctx.queue);
    }
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let bed_particles = sim.num_bed;
    let mut production_ms = Vec::with_capacity(run.calibration as usize);
    for _ in 0..run.calibration {
        let t = Instant::now();
        backend.calibration_step(&mut sim, ctx.device, ctx.queue);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        production_ms.push(ms(t));
    }

    let timer = ctx.timestamps_supported.then(|| {
        GpuTimer::new(
            ctx.device,
            ctx.period_ns,
            backend.timestamp_query_capacity(&sim),
        )
    });
    let mut measurements = ProfileMeasurements::with_capacity(run.measured);

    let water_particles_start = sim.num_water;
    for _ in 0..run.measured {
        let t = Instant::now();
        let sums = backend.measured_step(&mut sim, ctx.device, ctx.queue, timer.as_ref());
        measurements.record(ms(t), sums);
    }
    let water_particles_end = sim.num_water;
    let (frame_timings, gpu_passes) = measurements.finish(production_ms, run.measured);

    let mut bottlenecks = top_pass_bottlenecks(&gpu_passes);
    backend.append_bottlenecks(&mut bottlenecks, &frame_timings);

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
            solver: backend.solver_metadata(solver, &sim),
            solver_run: solver_run_metadata(run, solver),
            pressure_rbgs_pairs: backend.pressure_rbgs_pairs(&sim),
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

pub fn run_profile_from_env_args() {
    let cli = ProfilerCliArgs::from_env_args();
    if cli.help.unwrap_or(false) {
        print!("{}", profiler_usage());
        return;
    }
    if cli.list_solvers.unwrap_or(false) {
        print!("{}", solver_list_summary());
        return;
    }
    let selection = ProfileSelection::from_cli(&cli);
    if cli.dry_run_json.unwrap_or(false) {
        println!("{}", selection.dry_run_json());
        return;
    }
    if cli.dry_run.unwrap_or(false) {
        print!("{}", selection.dry_run_summary());
        return;
    }
    let Some(adapter) = request_adapter() else {
        eprintln!("profile_solvers: no GPU adapter available; skipping.");
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
    let period_ns = queue.get_timestamp_period();

    if !timestamps_supported {
        eprintln!(
            "profile_solvers: adapter lacks TIMESTAMP_QUERY; \
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
        scene: selection.scene,
        warmup: selection.warmup,
        measured: selection.measured,
        calibration: selection.calibration,
        cg_iterations: selection.cg_iterations,
        pressure_operator: selection.pressure_operator,
        solver_ids: selection.solver_ids,
        base_output_path: &selection.base_output_path,
        multiple_outputs: selection.multiple_outputs,
    };

    for &solver in &selection.solvers {
        solver.profile(&device_ctx, &run);
    }
}

#[test]
#[ignore = "profiling harness; run explicitly with --ignored --release"]
fn profile_solvers() {
    run_profile_from_env_args();
}

#[test]
#[ignore = "compatibility wrapper; prefer profile_solvers"]
fn profile_mpm_pipeline() {
    run_profile_from_env_args();
}

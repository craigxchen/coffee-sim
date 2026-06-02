//! PBF water core under `SolverId::Xpbd` — water incompressibility only (no bed/coupling).
//!
//! Classic Position Based Fluids (Macklin & Müller 2013): a constant-density constraint +
//! `s_corr` artificial pressure (anti-clumping), with XSPH viscosity damping the surface
//! jitter that artificial pressure injects. Adaptive constraint iterations decide
//! convergence GPU-side (no per-iteration readback). See `docs/plans/solver_xpbd.md`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::emission::EmissionInput;
use crate::engine::scene::Species;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::solvers::base::Solver;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;
use crate::utils::kernels;

const WG: u32 = 256;
/// Rebuild the neighbor grid every this-many solver iterations (anti-stale-grid).
const REGRID_INTERVAL: u32 = 4;

fn groups(n: u32) -> u32 {
    n.div_ceil(WG)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    box_min: [f32; 4],
    box_max: [f32; 4],
    gravity: [f32; 4],
    grid_origin: [f32; 4],
    grid_dims: [u32; 4], // nx, ny, nz, num_cells
    dt: f32,
    h: f32,
    rest_density: f32,
    particle_mass: f32,
    s_corr_k: f32,
    s_corr_n: f32,
    s_corr_wq: f32,
    relaxation_eps: f32,
    position_relaxation: f32,
    xsph_c: f32,
    max_speed: f32,
    spiky_r_min: f32,
    cell_size: f32,
    particle_count: u32,
    bucket_capacity: u32,
    min_iters: u32,
    max_iters: u32,
    residual_tolerance: f32,
    lambda_noncohesive: u32,
    max_correction: f32,
    velocity_damping: f32,
    // --- granular bed (grain phase) ---
    grain_diameter: f32,
    friction_mu: f32,
    floor_mu: f32,
    dry_cohesion: f32,
    cohesion_range: f32,
    rolling_damping: f32,
    freeze_speed: f32,
    freeze_pen: f32,
    thaw_pen: f32,
    freeze_frames: u32,
    _pad0: u32,
}

/// CPU mirror of the WGSL `Status` struct (8 × u32).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct StatusRaw {
    overflow: u32,
    max_occupancy: u32,
    converged: u32,
    iters_done: u32,
    effective_iters: u32,
    residual_bits: u32,
    frozen_count: u32,
    _pad1: u32,
}

struct Pipelines {
    predict: wgpu::ComputePipeline,
    grid_clear: wgpu::ComputePipeline,
    grid_fill: wgpu::ComputePipeline,
    compute_lambda: wgpu::ComputePipeline,
    residual_reduce: wgpu::ComputePipeline,
    compute_dp: wgpu::ComputePipeline,
    bed_project: wgpu::ComputePipeline,
    apply_dp: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    xsph: wgpu::ComputePipeline,
}

struct BindGroups {
    predict: wgpu::BindGroup,
    grid_clear: wgpu::BindGroup,
    grid_fill: wgpu::BindGroup,
    compute_lambda: wgpu::BindGroup,
    residual_reduce: wgpu::BindGroup,
    compute_dp: wgpu::BindGroup,
    bed_project: wgpu::BindGroup,
    apply_dp: wgpu::BindGroup,
    finalize: wgpu::BindGroup,
    xsph: wgpu::BindGroup,
}

struct Timestamps {
    qset: wgpu::QuerySet,
    capacity: u32,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    labels: Vec<String>,
    count: u32,
}

/// Diagnostics sampled from the GPU (dev/test only; the read-back stalls).
#[derive(Clone, Copy, Debug, Default)]
pub struct XpbdDiagnostics {
    pub overflow: bool,
    pub max_occupancy: u32,
    pub effective_iters: u32,
    /// Final max constraint residual: over-density (water) or normalized penetration (grain).
    pub residual: f32,
    /// Number of frozen (settled, static) grains last frame.
    pub frozen_count: u32,
}

pub struct XpbdSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Params,
    particle_count: u32,
    num_cells: u32,
    substeps: u32,
    /// Constraint-iteration cap for this scene's species (water or bed).
    max_iters: u32,
    /// Neighbor-grid rebuild interval for this scene's species.
    regrid_interval: u32,
    /// Which species this scene seeds; selects the per-frame pass sequence.
    species: Species,

    // Buffers referenced every frame only through the bind groups (which retain them) are
    // not held here. We keep the ones we touch directly: params (write), pos/vel (expose +
    // readback/copy), vel_smoothed (copy src), status (+ readbacks), phase/frozen (expose/reset).
    params_buf: wgpu::Buffer,
    pos: Arc<wgpu::Buffer>,
    vel: Arc<wgpu::Buffer>,
    vel_smoothed: wgpu::Buffer,
    phase: Arc<wgpu::Buffer>,
    frozen: wgpu::Buffer,
    status: wgpu::Buffer,
    status_readback: wgpu::Buffer,
    pos_readback: wgpu::Buffer,

    pipelines: Pipelines,
    bind_groups: BindGroups,
    ts: Option<Timestamps>,

    dispatches: u32,
    cached_diag: XpbdDiagnostics,
    cached_passes: Vec<(String, f32)>,

    // retained for reset (exact re-seed of the initial block + phase tags)
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
}

/// Seed the scene's initial particle block on a jittered lattice. Returns positions and the
/// matching per-particle phase tags (all water or all grain — scenes are single-species for now).
fn seed_block(scene: &Scene, mats: &Materials, cfg: &Config) -> (Vec<[f32; 4]>, Vec<u32>) {
    let s = mats.particle_spacing;
    let jitter = cfg.seed_jitter * s;
    let mut rng = crate::utils::rng::Rng::new(cfg.seed);
    let lo = scene.water_block_min;
    let hi = scene.water_block_max;
    let tag = match scene.species {
        Species::Water => 0u32,
        Species::Grain => 1u32,
    };
    let mut pos = Vec::new();
    let mut phase = Vec::new();
    let nx = (((hi[0] - lo[0]) / s).floor() as i32).max(0);
    let ny = (((hi[1] - lo[1]) / s).floor() as i32).max(0);
    let nz = (((hi[2] - lo[2]) / s).floor() as i32).max(0);
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let jx = (rng.next_f32() * 2.0 - 1.0) * jitter;
                let jy = (rng.next_f32() * 2.0 - 1.0) * jitter;
                let jz = (rng.next_f32() * 2.0 - 1.0) * jitter;
                pos.push([
                    lo[0] + i as f32 * s + jx,
                    lo[1] + j as f32 * s + jy,
                    lo[2] + k as f32 * s + jz,
                    0.0,
                ]);
                phase.push(tag);
            }
        }
    }
    (pos, phase)
}

impl XpbdSolver {
    fn storage(
        device: &wgpu::Device,
        label: &str,
        bytes: u64,
        extra: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        })
    }

    /// Re-seed the particles to the original block (exact, deterministic): positions, phase
    /// tags, zeroed velocities, and thawed (zeroed) freeze counters.
    fn seed(&mut self) {
        self.queue
            .write_buffer(&self.pos, 0, bytemuck::cast_slice(&self.initial_positions));
        self.queue
            .write_buffer(&self.phase, 0, bytemuck::cast_slice(&self.initial_phases));
        let zeros = vec![[0.0f32; 4]; self.particle_count as usize];
        self.queue
            .write_buffer(&self.vel, 0, bytemuck::cast_slice(&zeros));
        let frozen_zeros = vec![0u32; self.particle_count as usize];
        self.queue
            .write_buffer(&self.frozen, 0, bytemuck::cast_slice(&frozen_zeros));
    }

    /// Blocking GPU→CPU read-back of a `vec4` particle buffer (dev/test only — stalls).
    fn read_vec4(&self, src: &wgpu::Buffer) -> Vec<[f32; 4]> {
        let size = (self.particle_count as u64) * 16;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vec4-readback"),
            });
        enc.copy_buffer_to_buffer(src, 0, &self.pos_readback, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = self.pos_readback.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.pos_readback.unmap();
        out
    }

    /// Read back current particle positions (dev/test only — stalls the GPU).
    pub fn read_positions(&self) -> Vec<[f32; 4]> {
        self.read_vec4(self.pos.as_ref())
    }

    /// Read back current particle velocities (dev/test only — stalls the GPU).
    pub fn read_velocities(&self) -> Vec<[f32; 4]> {
        self.read_vec4(self.vel.as_ref())
    }

    /// Sample GPU diagnostics (status + per-pass timestamps) into the caches that
    /// `metrics()`/`profile()` return. Blocks (dev/test/periodic only).
    pub fn sample_diagnostics(&mut self) {
        // --- status ---
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("status-readback"),
            });
        enc.copy_buffer_to_buffer(
            &self.status,
            0,
            &self.status_readback,
            0,
            std::mem::size_of::<StatusRaw>() as u64,
        );
        self.queue.submit(Some(enc.finish()));
        let slice = self.status_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let raw: StatusRaw = *bytemuck::from_bytes(&slice.get_mapped_range());
        self.status_readback.unmap();
        self.cached_diag = XpbdDiagnostics {
            overflow: raw.overflow != 0,
            max_occupancy: raw.max_occupancy,
            effective_iters: raw.effective_iters,
            residual: f32::from_bits(raw.residual_bits),
            frozen_count: raw.frozen_count,
        };

        // --- timestamps ---
        if let Some(ts) = &self.ts {
            if ts.count >= 2 {
                let size = (ts.count as u64) * 8;
                let slice = ts.readback.slice(0..size);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                rx.recv().unwrap().unwrap();
                let times: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
                ts.readback.unmap();
                let mut passes = Vec::new();
                for (idx, label) in ts.labels.iter().enumerate() {
                    let b = times[idx * 2];
                    let e = times[idx * 2 + 1];
                    let us = (e.saturating_sub(b)) as f32 * ts.period_ns / 1000.0;
                    passes.push((label.clone(), us));
                }
                self.cached_passes = passes;
            }
        }
    }

    pub fn diagnostics(&self) -> XpbdDiagnostics {
        self.cached_diag
    }
}

impl Solver for XpbdSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let s = mats.particle_spacing;
        let h = mats.support_radius;
        let m = mats.particle_mass;

        let (positions, phases) = seed_block(scene, mats, cfg);
        let particle_count = positions.len() as u32;

        // The grain bed solves contacts (harder to converge than water): its own iteration cap,
        // contact tolerance, and a more frequent grid rebuild. Single-species scenes pick one set.
        let grain = scene.species == Species::Grain;
        let solve_iters = if grain {
            cfg.bed_max_iters
        } else {
            cfg.max_iters
        };
        let regrid_interval = if grain {
            cfg.bed_regrid_interval.max(1)
        } else {
            REGRID_INTERVAL
        };
        let residual_tolerance = if grain {
            cfg.bed_residual_tolerance
        } else {
            cfg.residual_tolerance
        };

        let cell_size = h;
        let nx = (((scene.box_max[0] - scene.box_min[0]) / cell_size).ceil() as u32).max(1);
        let ny = (((scene.box_max[1] - scene.box_min[1]) / cell_size).ceil() as u32).max(1);
        let nz = (((scene.box_max[2] - scene.box_min[2]) / cell_size).ceil() as u32).max(1);
        let num_cells = nx * ny * nz;

        let rest_density = kernels::rest_density(s, h, m);
        let dq = cfg.s_corr_dq_ratio * h;
        let s_corr_wq = kernels::w_poly6(dq, h);

        let params = Params {
            box_min: [scene.box_min[0], scene.box_min[1], scene.box_min[2], 0.0],
            box_max: [scene.box_max[0], scene.box_max[1], scene.box_max[2], 0.0],
            gravity: [scene.gravity[0], scene.gravity[1], scene.gravity[2], 0.0],
            grid_origin: [scene.box_min[0], scene.box_min[1], scene.box_min[2], 0.0],
            grid_dims: [nx, ny, nz, num_cells],
            dt: 1.0 / 60.0,
            h,
            rest_density,
            particle_mass: m,
            s_corr_k: cfg.s_corr_k,
            s_corr_n: cfg.s_corr_n,
            s_corr_wq,
            relaxation_eps: cfg.relaxation_eps,
            position_relaxation: cfg.position_relaxation,
            xsph_c: cfg.xsph_viscosity_c,
            max_speed: cfg.max_speed,
            spiky_r_min: cfg.spiky_r_min_ratio * h,
            cell_size,
            particle_count,
            bucket_capacity: cfg.bucket_capacity,
            min_iters: cfg.min_iters,
            max_iters: solve_iters,
            residual_tolerance,
            lambda_noncohesive: cfg.lambda_clamp_noncohesive as u32,
            max_correction: cfg.max_correction_ratio * h,
            velocity_damping: cfg.velocity_damping,
            grain_diameter: mats.grain_diameter,
            friction_mu: mats.friction_mu,
            floor_mu: mats.floor_mu,
            dry_cohesion: mats.dry_cohesion,
            cohesion_range: crate::models::cohesion::COHESION_RANGE_RATIO * mats.grain_diameter,
            rolling_damping: mats.rolling_damping,
            freeze_speed: cfg.freeze_speed,
            freeze_pen: cfg.freeze_pen,
            thaw_pen: cfg.thaw_pen,
            freeze_frames: cfg.freeze_frames,
            _pad0: 0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xpbd-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let n = particle_count.max(1) as u64;
        let vec4 = n * 16;
        let f32s = n * 4;
        let pos = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("xpbd-pos"),
                contents: bytemuck::cast_slice(&positions),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
        );
        let pred = Self::storage(&device, "xpbd-pred", vec4, wgpu::BufferUsages::empty());
        let vel = Arc::new(Self::storage(
            &device,
            "xpbd-vel",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let vel_smoothed = Self::storage(
            &device,
            "xpbd-vel-smoothed",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let lambda = Self::storage(&device, "xpbd-lambda", f32s, wgpu::BufferUsages::empty());
        let dp = Self::storage(&device, "xpbd-dp", vec4, wgpu::BufferUsages::empty());
        let c_residual =
            Self::storage(&device, "xpbd-cresidual", f32s, wgpu::BufferUsages::empty());
        // Per-particle species tag (exposed to the renderer for color-by-phase) and the
        // per-grain freeze/settle counter (zero-initialized by wgpu = all thawed).
        let phase = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("xpbd-phase"),
                contents: bytemuck::cast_slice(&phases),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
        );
        let frozen = Self::storage(&device, "xpbd-frozen", f32s, wgpu::BufferUsages::empty());
        let cell_count = Self::storage(
            &device,
            "xpbd-cellcount",
            (num_cells as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        let cell_bucket = Self::storage(
            &device,
            "xpbd-cellbucket",
            (num_cells as u64) * (cfg.bucket_capacity as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        let status = Self::storage(
            &device,
            "xpbd-status",
            std::mem::size_of::<StatusRaw>() as u64,
            wgpu::BufferUsages::COPY_SRC,
        );

        let status_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xpbd-status-readback"),
            size: std::mem::size_of::<StatusRaw>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pos_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xpbd-pos-readback"),
            size: vec4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // WGSL has no imports: assemble the one module from the three concern files. `common`
        // declares Params/Status/bindings + shared kernels; `water` and `bed` add the
        // per-species solves. Module-scope declarations are order-independent.
        let shader_src = format!(
            "{}\n{}\n{}",
            include_str!("common.wgsl"),
            include_str!("water.wgsl"),
            include_str!("bed.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xpbd"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            predict: make("predict"),
            grid_clear: make("grid_clear"),
            grid_fill: make("grid_fill"),
            compute_lambda: make("compute_lambda"),
            residual_reduce: make("residual_reduce"),
            compute_dp: make("compute_dp"),
            bed_project: make("bed_project"),
            apply_dp: make("apply_dp"),
            finalize: make("finalize"),
            xsph: make("xsph"),
        };

        // Per-pipeline bind groups providing exactly the bindings each entry point uses.
        let bg = |pipe: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let layout = pipe.get_bind_group_layout(0);
            let e: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(b, buf)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &e,
            })
        };
        let bind_groups = BindGroups {
            predict: bg(
                &pipelines.predict,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &pred),
                    (3, &vel),
                    (10, &status),
                    (11, &phase),
                    (12, &frozen),
                ],
            ),
            grid_clear: bg(&pipelines.grid_clear, &[(0, &params_buf), (8, &cell_count)]),
            grid_fill: bg(
                &pipelines.grid_fill,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (8, &cell_count),
                    (9, &cell_bucket),
                    (10, &status),
                ],
            ),
            compute_lambda: bg(
                &pipelines.compute_lambda,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (5, &lambda),
                    (7, &c_residual),
                    (8, &cell_count),
                    (9, &cell_bucket),
                    (10, &status),
                    (11, &phase),
                ],
            ),
            residual_reduce: bg(
                &pipelines.residual_reduce,
                &[(0, &params_buf), (7, &c_residual), (10, &status)],
            ),
            compute_dp: bg(
                &pipelines.compute_dp,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (5, &lambda),
                    (6, &dp),
                    (8, &cell_count),
                    (9, &cell_bucket),
                    (10, &status),
                    (11, &phase),
                ],
            ),
            bed_project: bg(
                &pipelines.bed_project,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &pred),
                    (6, &dp),
                    (7, &c_residual),
                    (8, &cell_count),
                    (9, &cell_bucket),
                    (11, &phase),
                    (12, &frozen),
                ],
            ),
            apply_dp: bg(
                &pipelines.apply_dp,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &pred),
                    (6, &dp),
                    (10, &status),
                    (11, &phase),
                ],
            ),
            finalize: bg(
                &pipelines.finalize,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &pred),
                    (3, &vel),
                    (7, &c_residual),
                    (10, &status),
                    (11, &phase),
                    (12, &frozen),
                ],
            ),
            xsph: bg(
                &pipelines.xsph,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (3, &vel),
                    (4, &vel_smoothed),
                    (8, &cell_count),
                    (9, &cell_bucket),
                ],
            ),
        };

        let ts = if gpu.timestamps_supported {
            let passes_per_step = 5 + 6 * solve_iters;
            let capacity = (2 * passes_per_step * cfg.substeps).clamp(2, 512);
            let qset = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("xpbd-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xpbd-ts-resolve"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("xpbd-ts-readback"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Some(Timestamps {
                qset,
                capacity,
                resolve,
                readback,
                period_ns: queue.get_timestamp_period(),
                labels: Vec::new(),
                count: 0,
            })
        } else {
            None
        };

        Self {
            device,
            queue,
            params,
            particle_count,
            num_cells,
            substeps: cfg.substeps.max(1),
            max_iters: solve_iters,
            regrid_interval,
            species: scene.species,
            params_buf,
            pos,
            vel,
            vel_smoothed,
            phase,
            frozen,
            status,
            status_readback,
            pos_readback,
            pipelines,
            bind_groups,
            ts,
            dispatches: 0,
            cached_diag: XpbdDiagnostics::default(),
            cached_passes: Vec::new(),
            initial_positions: positions,
            initial_phases: phases,
        }
    }

    fn reset(&mut self, _scene: &Scene) {
        self.seed();
    }

    fn step(&mut self, dt: f32, _input: &EmissionInput) {
        self.params.dt = dt / self.substeps as f32;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let np = groups(self.particle_count);
        let nc = groups(self.num_cells);
        let mut cursor = 0u32;
        let mut labels: Vec<String> = Vec::new();
        let mut dispatches = 0u32;

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("xpbd-frame"),
            });

        // A scene is single-species: run the water density solve OR the granular contact solve.
        // Both share predict / grid build / apply_dp / finalize; the constraint passes differ.
        let grain = self.species == Species::Grain;
        for _ in 0..self.substeps {
            dispatch_pass(
                &mut enc,
                &self.pipelines.predict,
                &self.bind_groups.predict,
                self.ts.as_ref(),
                &mut cursor,
                &mut labels,
                "predict",
                np,
                &mut dispatches,
            );
            for it in 0..self.max_iters {
                // Rebuild the neighbor grid every few iterations so corrections never act on
                // a stale grid — particles can drift > 1 cell over the solve, and stale
                // neighbors are the source of the stochastic squeeze-out eruptions. The bed is
                // contact-heavy and rebuilds more often (smaller interval).
                if it % self.regrid_interval == 0 {
                    dispatch_pass(
                        &mut enc,
                        &self.pipelines.grid_clear,
                        &self.bind_groups.grid_clear,
                        self.ts.as_ref(),
                        &mut cursor,
                        &mut labels,
                        "grid_clear",
                        nc,
                        &mut dispatches,
                    );
                    dispatch_pass(
                        &mut enc,
                        &self.pipelines.grid_fill,
                        &self.bind_groups.grid_fill,
                        self.ts.as_ref(),
                        &mut cursor,
                        &mut labels,
                        "grid_fill",
                        np,
                        &mut dispatches,
                    );
                }
                if grain {
                    // Granular projection: non-penetration + Coulomb friction + light cohesion.
                    dispatch_pass(
                        &mut enc,
                        &self.pipelines.bed_project,
                        &self.bind_groups.bed_project,
                        self.ts.as_ref(),
                        &mut cursor,
                        &mut labels,
                        "bed_project",
                        np,
                        &mut dispatches,
                    );
                } else {
                    dispatch_pass(
                        &mut enc,
                        &self.pipelines.compute_lambda,
                        &self.bind_groups.compute_lambda,
                        self.ts.as_ref(),
                        &mut cursor,
                        &mut labels,
                        "compute_lambda",
                        np,
                        &mut dispatches,
                    );
                    dispatch_pass(
                        &mut enc,
                        &self.pipelines.compute_dp,
                        &self.bind_groups.compute_dp,
                        self.ts.as_ref(),
                        &mut cursor,
                        &mut labels,
                        "compute_dp",
                        np,
                        &mut dispatches,
                    );
                }
                dispatch_pass(
                    &mut enc,
                    &self.pipelines.apply_dp,
                    &self.bind_groups.apply_dp,
                    self.ts.as_ref(),
                    &mut cursor,
                    &mut labels,
                    "apply_dp",
                    np,
                    &mut dispatches,
                );
                // Check convergence AFTER applying the correction, so every frame relieves
                // gravity's compression at least once. (Checking before — the old order —
                // let "converged" frames skip the correction, so sub-tolerance compression
                // accumulated silently until it detonated into a global eruption.)
                dispatch_pass(
                    &mut enc,
                    &self.pipelines.residual_reduce,
                    &self.bind_groups.residual_reduce,
                    self.ts.as_ref(),
                    &mut cursor,
                    &mut labels,
                    "residual_reduce",
                    1,
                    &mut dispatches,
                );
            }
            dispatch_pass(
                &mut enc,
                &self.pipelines.finalize,
                &self.bind_groups.finalize,
                self.ts.as_ref(),
                &mut cursor,
                &mut labels,
                "finalize",
                np,
                &mut dispatches,
            );
            // XSPH viscosity is a fluid term; the dry bed has none. Skipping it for grains also
            // avoids the vel_smoothed copy clobbering grain velocities (xsph doesn't write them).
            if !grain {
                dispatch_pass(
                    &mut enc,
                    &self.pipelines.xsph,
                    &self.bind_groups.xsph,
                    self.ts.as_ref(),
                    &mut cursor,
                    &mut labels,
                    "xsph",
                    np,
                    &mut dispatches,
                );
                enc.copy_buffer_to_buffer(
                    &self.vel_smoothed,
                    0,
                    &self.vel,
                    0,
                    (self.particle_count as u64) * 16,
                );
            }
        }

        if let Some(ts) = &self.ts {
            if cursor >= 2 {
                enc.resolve_query_set(&ts.qset, 0..cursor, &ts.resolve, 0);
                enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, (cursor as u64) * 8);
            }
        }

        self.queue.submit(Some(enc.finish()));

        self.dispatches = dispatches;
        if let Some(ts) = self.ts.as_mut() {
            ts.labels = labels;
            ts.count = cursor;
        }
    }

    fn particles(&self) -> ParticleBuffers {
        ParticleBuffers {
            particle_count: self.particle_count,
            position: Some(Arc::clone(&self.pos)),
            velocity: Some(Arc::clone(&self.vel)),
            phase_tag: Some(Arc::clone(&self.phase)),
            ..Default::default()
        }
    }

    fn metrics(&self) -> Metrics {
        Metrics {
            particle_count: self.particle_count,
            iteration_count: self.cached_diag.effective_iters,
            ..Default::default()
        }
    }

    fn profile(&self) -> Profile {
        Profile {
            passes: self.cached_passes.clone(),
            dispatches_per_frame: self.dispatches,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_pass(
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    ts: Option<&Timestamps>,
    cursor: &mut u32,
    labels: &mut Vec<String>,
    label: &str,
    groups: u32,
    dispatches: &mut u32,
) {
    let tw = match ts {
        Some(t) if *cursor + 1 < t.capacity => {
            let b = *cursor;
            let e = *cursor + 1;
            *cursor += 2;
            labels.push(label.to_string());
            Some(wgpu::ComputePassTimestampWrites {
                query_set: &t.qset,
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            })
        }
        _ => None,
    };
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: tw,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, Some(bind_group), &[]);
    pass.dispatch_workgroups(groups, 1, 1);
    *dispatches += 1;
}

//! Seam-blend composed solver (M0): PB-MPM owns all free water, two-field owns the bed.
//!
//! Plan: docs/plans/2026-07-09-002-feat-seam-blend-m0-plan.md. The two inner solvers are
//! built from ONE shared (Scene bounds, Materials) pair, so their grids are co-registered
//! by construction (both derive h = 2·spacing, origin = box_min − h, dims = ceil(extent/h)+3
//! from the same inputs — `pbmpm::grid_spec_for` is a verbatim mirror of twofield's).
//!
//! KTD1 composition contract:
//! - The pbmpm inner receives the scene as-is (it ignores `Species::Grain` regions) and the
//!   full `EmissionInput` — the pour goes through pbmpm, never twofield.
//! - The twofield inner receives a Grain-only scene clone with `pour_water_ml = 0` (it seeds
//!   EVERY region it is given, and sizes a dead water pool off the pour declaration) and
//!   `solid_dynamics` forced on, stepped with a zeroed emission every frame. That keeps it on
//!   the cheap dry-dynamic path (4 passes/substep); a single live twofield water particle
//!   would re-enable its ~200-dispatch water pipeline — pinned by the scaffold gate.
//! - U1 carries NO seam physics: the inners do not interact yet. The bed field / porous BC /
//!   reaction / infiltration land in U2–U4.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::emission::EmissionInput;
use crate::engine::scene::Species;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::solvers::base::Solver;
use crate::solvers::pbmpm::PbmpmSolver;
use crate::solvers::twofield::TwofieldSolver;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

/// Bytes per canonical vec4 particle lane (pos/vel/chem).
const VEC4: u64 = 16;
/// Bytes per canonical u32 particle lane (phase).
const U32S: u64 = 4;
/// Seam GPU passes per frame when a bed is present (clear + scatter; the U3 hook adds more).
const SEAM_PASSES: u32 = 2;
const WG: u32 = 256;

/// Rust mirror of `seam.wgsl`'s `SeamParams` (byte-identical).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SeamParams {
    grid_origin: [f32; 4], // .xyz = node (0,0,0) world position; .w = cell size h
    grid_dims: [u32; 4],   // nx, ny, nz, num_nodes
    solid: [u32; 4],       // .x = solid range start in bed_pos, .y = count
    wet: [f32; 4],         // .x = V_dry, .y = V_cap
}
const _: () = assert!(std::mem::size_of::<SeamParams>() == 64);

pub struct SeamSolver {
    water: PbmpmSolver,
    bed: TwofieldSolver,

    /// Held scene splits, so a future rebuild path re-splits identically. Both inner
    /// `reset()`s ignore their scene argument and replay cached seeds, so these are
    /// documentation-of-record more than live inputs today.
    #[allow(dead_code)]
    water_scene: Scene,
    #[allow(dead_code)]
    bed_scene: Scene,

    /// Solid count is fixed at build (grains neither emit nor die in M0).
    solid_count: u32,
    /// The twofield inner's water-pool capacity — the element offset where its solid
    /// range starts (`[water_capacity, water_capacity + solid_count)`).
    bed_solid_offset: u32,

    // Seam-owned canonical render buffers (KTD6): per-frame region copies of each inner's
    // LIVE ranges only — pbmpm's dormant pour tail and twofield's parked slots never render.
    render_pos: Arc<wgpu::Buffer>,
    render_vel: Arc<wgpu::Buffer>,
    render_phase: Arc<wgpu::Buffer>,
    render_chem: Arc<wgpu::Buffer>,
    exposed_count: u32,

    // Seam bed-field passes (U2): clear + trilinear scatter of the bed solver's grains into
    // pbmpm's `bed_occupancy` (KTD2 — cleared every frame; the scatter reads the bed's
    // PERSISTENT grain buffer, never its per-substep scratch).
    seam_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    seam_scatter: (wgpu::ComputePipeline, wgpu::BindGroup),
    /// pbmpm's reaction ledger (U3): accumulated by the bed BC across the water step,
    /// consumed by the bed inner's `seam_inject` each substep, zeroed here after the frame.
    reaction: Arc<wgpu::Buffer>,
    num_nodes: u32,
    seam_dispatches: u32,

    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl SeamSolver {
    /// The twofield inner's scene: Grain regions only, no pour declaration (KTD1).
    fn split_bed_scene(scene: &Scene) -> Scene {
        let mut bed = scene.clone();
        bed.regions.retain(|r| r.species == Species::Grain);
        bed.pour_water_ml = 0.0;
        bed
    }

    fn render_buffer(device: &wgpu::Device, label: &str, size: u64) -> Arc<wgpu::Buffer> {
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    /// Copy both inners' live particle ranges into the seam's canonical buffers.
    /// Runs inside `step()` (its own small submit); `particles()` stays sync-free.
    fn merge_render_buffers(&mut self) {
        let wc = self.water.active_count() as u64;
        let sc = self.solid_count as u64;
        let s_off = self.bed_solid_offset as u64;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("seam-render-merge"),
            });
        // U3: the bed inner consumed the reaction ledger this frame (merge runs after
        // bed.step()); zero it so next frame's BC accumulates fresh.
        enc.clear_buffer(&self.reaction, 0, None);
        let wp = self.water.particles();
        let bp = self.bed.particles();
        let lanes: [(&Option<Arc<wgpu::Buffer>>, &Option<Arc<wgpu::Buffer>>, &Arc<wgpu::Buffer>, u64); 4] = [
            (&wp.position, &bp.position, &self.render_pos, VEC4),
            (&wp.velocity, &bp.velocity, &self.render_vel, VEC4),
            (&wp.phase_tag, &bp.phase_tag, &self.render_phase, U32S),
            (&wp.concentration, &bp.concentration, &self.render_chem, VEC4),
        ];
        for (water_src, bed_src, dst, stride) in lanes {
            if wc > 0 {
                if let Some(src) = water_src {
                    enc.copy_buffer_to_buffer(src, 0, dst, 0, wc * stride);
                }
            }
            if sc > 0 {
                if let Some(src) = bed_src {
                    enc.copy_buffer_to_buffer(src, s_off * stride, dst, wc * stride, sc * stride);
                }
            }
        }
        self.queue.submit(Some(enc.finish()));
        self.exposed_count = (wc + sc) as u32;
    }

    /// Pre-saturate the bed (docs/plans/2026-07-09-002 U2): grain V_abs = sat_frac·V_cap on
    /// both the live buffer and the cached seed, so `reset()` replays the wet bed. Part of
    /// the pre-registered scene setup for every M0 arm; call before stepping.
    pub fn prewet_bed(&mut self, sat_frac: f32) {
        self.bed.prewet_grains(sat_frac);
    }

    /// Read-only handle on the water (pbmpm) inner, for gates and diagnostics.
    pub fn water_solver(&self) -> &PbmpmSolver {
        &self.water
    }

    /// Read-only handle on the bed (twofield) inner, for gates and diagnostics.
    pub fn bed_solver(&self) -> &TwofieldSolver {
        &self.bed
    }

    /// Live water particles currently simulated by the pbmpm inner.
    pub fn water_count(&self) -> u32 {
        self.water.active_count()
    }

    /// Bed grains (fixed at build).
    pub fn solid_count(&self) -> u32 {
        self.solid_count
    }

    /// The scaffold invariant: the twofield inner must never hold live water — one stray
    /// particle re-enables its full water pipeline (the count-keyed cost cliff).
    pub fn bed_water_count(&self) -> u32 {
        self.bed.active_count() - self.solid_count
    }

    /// Blocking GPU→CPU readback of the first `bytes` of a seam render buffer
    /// (dev/test only — stalls; transient staging buffer, not a hot path).
    fn read_render_bytes(&self, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        if bytes == 0 {
            return Vec::new();
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seam-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("seam-readback"),
            });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    /// Read back the merged render phase lane (dev/test only — stalls the GPU).
    pub fn read_render_phases_for_test(&self) -> Vec<u32> {
        let bytes = self.exposed_count as u64 * U32S;
        bytemuck::cast_slice(&self.read_render_bytes(&self.render_phase, bytes)).to_vec()
    }

    /// Read back the merged render positions (dev/test only — stalls the GPU).
    pub fn read_render_positions_for_test(&self) -> Vec<[f32; 4]> {
        let bytes = self.exposed_count as u64 * VEC4;
        bytemuck::cast_slice(&self.read_render_bytes(&self.render_pos, bytes)).to_vec()
    }
}

impl Solver for SeamSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        let water_scene = scene.clone();
        let bed_scene = Self::split_bed_scene(scene);

        // The bed inner runs its deformable dynamics unconditionally (the dry-dynamic
        // elision requires it); the water inner arms the bed BC + reaction lanes. Everything
        // else in cfg passes through.
        let mut bed_cfg = cfg.clone();
        bed_cfg.solid_dynamics = true;
        let mut water_cfg = cfg.clone();
        water_cfg.pbmpm_seam_bed = true;

        let water = PbmpmSolver::build(&water_scene, mats, &water_cfg, gpu);
        let mut bed = TwofieldSolver::build(&bed_scene, mats, &bed_cfg, gpu);

        // KTD1: co-registration holds by construction (one shared scene-bounds + Materials
        // pair, verbatim-mirrored grid derivations) — a future caller diverging the inputs
        // must fail loudly, not resample silently.
        assert_eq!(
            water.grid_spec(),
            bed.grid_spec(),
            "seam inner grids must be co-registered (same origin, cell size, dims)"
        );

        let solid_count = bed.active_count(); // bed scene seeds no water, so live == solids
        let bed_solid_offset = bed.water_pool_capacity();
        let capacity = (water.capacity() + solid_count) as u64;

        let device = gpu.device.clone();

        // Seam bed-field passes (U2). The scatter binds the bed's persistent grain positions
        // and pbmpm's bed_occupancy lanes; params are static in M0 (grains neither emit nor
        // die, the grid is fixed at build).
        let (origin, h, dims) = water.grid_spec();
        let num_nodes = dims[0] * dims[1] * dims[2];
        let v_dry = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
        let v_cap = crate::models::wetting::capacity(v_dry, mats.r_max, mats.rho_ratio);
        let seam_params = SeamParams {
            grid_origin: [origin[0], origin[1], origin[2], h],
            grid_dims: [dims[0], dims[1], dims[2], num_nodes],
            solid: [bed_solid_offset, solid_count, 0, 0],
            wet: [v_dry, v_cap, 0.0, 0.0],
        };
        let seam_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seam-params"),
            contents: bytemuck::bytes_of(&seam_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seam"),
            source: wgpu::ShaderSource::Wgsl(include_str!("seam.wgsl").into()),
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
        let (bed_occupancy, reaction) = water.seam_buffers();
        // U3: the bed inner consumes the water inner's reaction ledger inside its dry-branch
        // substep loop (one full ledger per frame); the seam zeroes it after each frame.
        bed.attach_seam_reaction(&reaction);
        let bed_pos = bed
            .particles()
            .position
            .expect("twofield always exposes a position buffer");
        let seam_clear_pipe = make("seam_clear_bed");
        let seam_clear_bind = bg(
            &seam_clear_pipe,
            &[(0, &seam_params_buf), (2, &bed_occupancy)],
        );
        let seam_scatter_pipe = make("seam_scatter_bed");
        let seam_scatter_bind = bg(
            &seam_scatter_pipe,
            &[(0, &seam_params_buf), (1, &bed_pos), (2, &bed_occupancy)],
        );
        let render_pos = Self::render_buffer(&device, "seam-render-pos", capacity * VEC4);
        let render_vel = Self::render_buffer(&device, "seam-render-vel", capacity * VEC4);
        let render_phase = Self::render_buffer(&device, "seam-render-phase", capacity * U32S);
        let render_chem = Self::render_buffer(&device, "seam-render-chem", capacity * VEC4);

        let mut seam = Self {
            water,
            bed,
            water_scene,
            bed_scene,
            solid_count,
            bed_solid_offset,
            render_pos,
            render_vel,
            render_phase,
            render_chem,
            exposed_count: 0,
            seam_clear: (seam_clear_pipe, seam_clear_bind),
            seam_scatter: (seam_scatter_pipe, seam_scatter_bind),
            reaction,
            num_nodes,
            seam_dispatches: 0,
            device,
            queue: gpu.queue.clone(),
        };
        seam.merge_render_buffers();
        seam
    }

    fn reset(&mut self, scene: &Scene) {
        // Both inners replay their cached seeds (their scene argument is ignored today);
        // re-splitting therefore needs a rebuild, not a reset — hold the splits as-is.
        self.water.reset(scene);
        self.bed.reset(scene);
        self.merge_render_buffers();
    }

    fn step(&mut self, dt: f32, input: &EmissionInput) {
        // KTD2 frame order: refresh the bed-occupancy field (clear, then trilinear scatter
        // of the bed's persistent grain positions) BEFORE the water step, so pbmpm's bed BC
        // sees this frame's bed. Skipped without a bed — water-only scenes stay pure pbmpm.
        self.seam_dispatches = 0;
        if self.solid_count > 0 {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("seam-bed-field"),
                });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("seam-bed-field"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.seam_clear.0);
                cpass.set_bind_group(0, &self.seam_clear.1, &[]);
                cpass.dispatch_workgroups(self.num_nodes.div_ceil(WG).max(1), 1, 1);
                cpass.set_pipeline(&self.seam_scatter.0);
                cpass.set_bind_group(0, &self.seam_scatter.1, &[]);
                cpass.dispatch_workgroups(self.solid_count.div_ceil(WG).max(1), 1, 1);
            }
            self.queue.submit(Some(enc.finish()));
            self.seam_dispatches = SEAM_PASSES;
        }

        // The pour goes through pbmpm only. The bed inner gets a zeroed flow (its emit()
        // runs unconditionally at step() top) with the discrete event forwarded so a
        // Reset never leaves a stale emitter backlog on either side.
        self.water.step(dt, input);
        let bed_input = EmissionInput {
            event: input.event,
            ..EmissionInput::default()
        };
        self.bed.step(dt, &bed_input);
        self.merge_render_buffers();
    }

    fn particles(&self) -> ParticleBuffers {
        ParticleBuffers {
            particle_count: self.exposed_count,
            position: Some(Arc::clone(&self.render_pos)),
            velocity: Some(Arc::clone(&self.render_vel)),
            phase_tag: Some(Arc::clone(&self.render_phase)),
            concentration: Some(Arc::clone(&self.render_chem)),
            temperature: Some(Arc::clone(&self.render_chem)),
            ..Default::default()
        }
    }

    fn metrics(&self) -> Metrics {
        // Water-side count + the bed's diagnostics; extraction/TDS stay 0 in M0.
        let bed = self.bed.metrics();
        Metrics {
            particle_count: self.water.active_count() + bed.particle_count,
            drawdown_time: bed.drawdown_time,
            evenness: bed.evenness,
            iteration_count: bed.iteration_count,
            ..Default::default()
        }
    }

    fn profile(&self) -> Profile {
        // Namespaced merge: both inners use the same raw pass labels ("p2g", "g2p", …) and
        // the perf harness aggregates by label string — unprefixed merging would silently
        // fuse them.
        let w = self.water.profile();
        let b = self.bed.profile();
        let mut passes = Vec::with_capacity(w.passes.len() + b.passes.len());
        passes.extend(w.passes.into_iter().map(|(l, us)| (format!("water/{l}"), us)));
        passes.extend(b.passes.into_iter().map(|(l, us)| (format!("bed/{l}"), us)));
        Profile {
            passes,
            dispatches_per_frame: w.dispatches_per_frame
                + b.dispatches_per_frame
                + self.seam_dispatches,
        }
    }
}

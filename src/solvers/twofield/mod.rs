//! Two-field grid solver under `SolverId::Twofield` — rung L0 of
//! `docs/plans/2026-06-09-001-feat-unified-twofield-solver-plan.md`.
//!
//! U1 pinned the seam contract (six `Solver` methods, GPU-cached getters), the two-field
//! particle-range layout (water `[0, w)`, then solids — KTD-1), the canonical `ParticleBuffers`
//! exposure, and the dispatch/budget bookkeeping.
//!
//! U2 adds the APIC water transfers: quadratic B-spline P2G with fixed-point atomics, grid
//! gravity + boundary conditions, APIC G2P with a per-particle affine C matrix
//! (`transfers.wgsl`). Per frame: `grid_clear → p2g_water → grid_update → g2p_water`, one
//! substep (the CFL substep policy is a later-unit decision). No pressure yet — U2 water is
//! momentum-correct splashing dust; incompressibility lands in U3.

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

const WG: u32 = 256;

fn groups(n: u32) -> u32 {
    n.div_ceil(WG)
}

/// U2 GPU-budget bookkeeping (R8): the widest entry point's storage-buffer count, derived by
/// inspection of the bind groups in `build` (the params uniform doesn't count against the
/// storage limit). Per pass: `p2g_water` 4 (pos, vel, cmat, grid_fp), `grid_update` 3
/// (grid_fp, grid_vel, solids), `g2p_water` 5 (pos, vel, cmat, grid_vel, solids),
/// `grid_clear` 1 (grid_fp). Re-derive when passes are added. The device requests 9 storage
/// buffers per stage (`src/utils/gpu.rs::NEEDED_STORAGE_BUFFERS`) — still NOT raised (KTD-7):
/// grid mass+momentum share one `array<atomic<i32>>` (stride 4) so the grid costs two bindings,
/// not five.
pub const MAX_STORAGE_BUFFERS_PER_ENTRY_POINT: u32 = 5;

/// Compute dispatches per frame: grid_clear, p2g_water, grid_update, g2p_water.
pub const DISPATCHES_PER_FRAME: u32 = 4;

/// Fine-grid cell size as a multiple of the particle spacing. ~2× spacing gives the quadratic
/// B-spline support (1.5 cells each way) a ≈3-spacing reach with ≈8 particles per cell at rest
/// — the standard APIC loading and the same scale as the SPH `support_radius` default. The plan
/// defers this to implementation; revisit when the U3 pressure stencil picks its resolution.
pub const CELL_SIZE_FACTOR: f32 = 2.0;

/// Fixed-point scale for the grid atomics — mirrors `FP_SCALE` in `common.wgsl` (2^18,
/// KEEP.md §3). The overflow-headroom derivation (coupled to `Config::max_speed`) lives next to
/// the WGSL constant; the overflow-probe gate in `tests/twofield_water.rs` exercises it.
pub const FP_SCALE: f64 = 262144.0;

/// Uniform parameters, byte-mirrored by the WGSL `Params` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    box_min: [f32; 4],
    box_max: [f32; 4],
    gravity: [f32; 4],
    grid_origin: [f32; 4], // .xyz = node (0,0,0) world position; .w = cell size h
    grid_dims: [u32; 4],   // nx, ny, nz, num_nodes
    dt: f32,
    particle_mass: f32,
    max_speed: f32,
    pic_mode: u32,       // test-only PIC variant flag (APIC-vs-PIC discrimination gate)
    water_count: u32,    // particles [0, water_count) are water (KTD-1 range layout)
    solid_count: u32,    // particles [water_count, water_count + solid_count) are solid grains
    particle_count: u32, // = water_count + solid_count (kernel live-set guard)
    num_solids: u32,     // count of static SDF solids in the `solids` buffer
}

// Params is uploaded as a uniform and must stay byte-identical to the WGSL `Params`.
const _: () = assert!(std::mem::size_of::<Params>() == 112);

/// GPU record for one static SDF solid — byte-identical to the WGSL `Primitive` (64 bytes,
/// vec4-aligned; mirrors the xpbd packing of `utils::sdf` primitives). Cone radii in `a` are
/// OUTER wall radii; the cavity surface is `outer − thickness`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Primitive {
    kind: u32,         // 0 = cone, 1 = cylinder
    species_mask: u32, // MASK_* bits
    friction: f32,
    flags: u32,  // bit0 = apex_open
    a: [f32; 4], // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)
    b: [f32; 4], // cone:(thickness, hole_radius, center_x, center_z)  cyl:(center_x, center_z, _, _)
    c: [f32; 4], // reserved
}

// Primitive is a storage-buffer element and must stay byte-identical to the WGSL `Primitive`.
const _: () = assert!(std::mem::size_of::<Primitive>() == 64);

/// Pack a scene's analytic solids into the GPU `Primitive` layout (mirrors utils/sdf.rs).
fn pack_solids(solids: &[crate::utils::sdf::SdfPrimitive]) -> Vec<Primitive> {
    use crate::utils::sdf::SolidKind;
    solids
        .iter()
        .map(|s| match s.kind {
            SolidKind::Cone {
                center,
                apex_y,
                top_y,
                apex_r,
                top_r,
                thickness,
                hole_radius,
                apex_open,
            } => Primitive {
                kind: 0,
                species_mask: s.species_mask,
                friction: s.friction,
                flags: u32::from(apex_open),
                a: [apex_y, apex_r, top_y, top_r],
                b: [thickness, hole_radius, center.x, center.z],
                c: [0.0; 4],
            },
            SolidKind::Cylinder {
                center,
                floor_y,
                rim_y,
                radius,
            } => Primitive {
                kind: 1,
                species_mask: s.species_mask,
                friction: s.friction,
                flags: 0,
                a: [floor_y, rim_y, radius, 0.0],
                b: [center.x, center.z, 0.0, 0.0],
                c: [0.0; 4],
            },
        })
        .collect()
}

/// Per-pass `timestamp-query` capture (native only; the web context never enables the
/// feature). Same resolve-into-readback-of-the-previous-frame shape as xpbd, so `profile()`
/// never stalls — `sample_diagnostics` is the explicit blocking cache point.
struct Timestamps {
    qset: wgpu::QuerySet,
    capacity: u32,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    labels: Vec<String>,
    count: u32,
}

pub struct TwofieldSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Params,
    water_count: u32,
    solid_count: u32,
    num_nodes: u32,

    params_buf: wgpu::Buffer,
    // Canonical particle state, exposed through `ParticleBuffers`: pos.w carries the moisture
    // lane (water = remaining fraction, grain = absorbed volume), chem = (c, T, _, _).
    pos: Arc<wgpu::Buffer>,
    vel: Arc<wgpu::Buffer>,
    phase: Arc<wgpu::Buffer>,
    chem: Arc<wgpu::Buffer>,
    // Per-particle APIC affine matrix C: 3 vec4 rows per particle (see common.wgsl binding 5).
    cmat: wgpu::Buffer,
    // WATER grid field: fixed-point atomic<i32>, 4 lanes per node (mass, mom.xyz). The float
    // grid-velocity and solids buffers live only inside the bind groups (no CPU-side access).
    grid_fp: wgpu::Buffer,
    readback: wgpu::Buffer,

    pipelines: Pipelines,
    ts: Option<Timestamps>,

    // Cached per-frame results returned by the getters (never a GPU sync there).
    dispatches: u32,
    cached_passes: Vec<(String, f32)>,

    // Retained for reset (exact, deterministic re-seed).
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
}

struct Pipelines {
    grid_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g_water: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p_water: (wgpu::ComputePipeline, wgpu::BindGroup),
}

/// Seed the scene's regions into the two-field range layout: ALL water particles first
/// (`[0, water_count)`), then all solids. Lattice + seeded jitter per species spacing, with
/// cone-aware SDF rejection (mirrors xpbd's `seed_block`; RNG draws happen per lattice point
/// regardless of rejection, so the layout is deterministic for a given `cfg.seed`). The 4th
/// position lane carries moisture: water = remaining fraction (1 = full), grain = absorbed
/// volume (0 = dry). Returns `(positions, phases, water_count)`.
fn seed_ranges(scene: &Scene, mats: &Materials, cfg: &Config) -> (Vec<[f32; 4]>, Vec<u32>, u32) {
    const SEED_CLEARANCE: f32 = 0.4;
    let mut rng = crate::utils::rng::Rng::new(cfg.seed);
    let mut water: Vec<[f32; 4]> = Vec::new();
    let mut solid: Vec<[f32; 4]> = Vec::new();
    for region in &scene.regions {
        let (lo, hi) = (region.min, region.max);
        let (s, tag, w0) = match region.species {
            Species::Water => (mats.particle_spacing, 0u32, 1.0f32),
            Species::Grain => (mats.grain_diameter, 1u32, 0.0f32),
        };
        let jitter = cfg.seed_jitter * s;
        let nx = (((hi[0] - lo[0]) / s).floor() as i32).max(0);
        let ny = (((hi[1] - lo[1]) / s).floor() as i32).max(0);
        let nz = (((hi[2] - lo[2]) / s).floor() as i32).max(0);
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let jx = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let jy = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let jz = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let px = lo[0] + i as f32 * s + jx;
                    let py = lo[1] + j as f32 * s + jy;
                    let pz = lo[2] + k as f32 * s + jz;
                    if !scene.solids.is_empty() {
                        let c = crate::utils::sdf::nearest(
                            &scene.solids,
                            glam::Vec3::new(px, py, pz),
                            tag,
                        );
                        if c.signed < SEED_CLEARANCE {
                            continue;
                        }
                    }
                    let p = [px, py, pz, w0];
                    if tag == 0 {
                        water.push(p);
                    } else {
                        solid.push(p);
                    }
                }
            }
        }
    }
    let water_count = water.len() as u32;
    let mut pos = water;
    pos.extend_from_slice(&solid);
    let mut phase = vec![0u32; water_count as usize];
    phase.extend(std::iter::repeat_n(1u32, solid.len()));
    (pos, phase, water_count)
}

/// Grid dimensions for a scene: cell size `h = CELL_SIZE_FACTOR · particle_spacing`, node 0 one
/// cell OUTSIDE `box_min` (a pad layer), `ceil(extent/h) + 3` nodes per axis. With the pad, a
/// particle anywhere in the box — including exactly on a face after the G2P clamp — has its
/// full 3-node B-spline support in range: `xl = (x − origin)/h ∈ [1, 1 + extent/h]`, so
/// `base = floor(xl − 0.5) ∈ [0, dims − 3]`. The node layer at exactly `box_min` always exists
/// (origin + h = box_min), which is what the grid box-face BC keys on.
fn grid_spec_for(scene: &Scene, mats: &Materials) -> ([f32; 3], f32, [u32; 3]) {
    let h = CELL_SIZE_FACTOR * mats.particle_spacing;
    let origin = [
        scene.box_min[0] - h,
        scene.box_min[1] - h,
        scene.box_min[2] - h,
    ];
    let mut dims = [0u32; 3];
    for (a, d) in dims.iter_mut().enumerate() {
        let extent = (scene.box_max[a] - scene.box_min[a]).max(0.0);
        *d = (extent / h).ceil() as u32 + 3;
    }
    (origin, h, dims)
}

impl TwofieldSolver {
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

    /// Blocking GPU→CPU read-back of the first `bytes` of `src` (dev/test only — stalls).
    fn read_bytes(&self, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        if bytes == 0 {
            return Vec::new();
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-readback"),
            });
        enc.copy_buffer_to_buffer(src, 0, &self.readback, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = self.readback.slice(0..bytes);
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
        self.readback.unmap();
        data
    }

    /// Read back current particle positions (dev/test only — stalls the GPU).
    pub fn read_positions(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.pos, bytes)).to_vec()
    }

    /// Read back current particle velocities (dev/test only — stalls the GPU).
    pub fn read_velocities(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.vel, bytes)).to_vec()
    }

    /// Read back the per-particle phase tags (dev/test only — stalls the GPU).
    pub fn read_phases(&self) -> Vec<u32> {
        let bytes = (self.params.particle_count as u64) * 4;
        bytemuck::cast_slice(&self.read_bytes(&self.phase, bytes)).to_vec()
    }

    /// Water/solid range sizes (the KTD-1 layout: water `[0, w)`, solids `[w, w + s)`).
    pub fn phase_counts(&self) -> (u32, u32) {
        (self.water_count, self.solid_count)
    }

    /// Grid layout `(origin, cell_size, dims)` for CPU twins (dev/test only).
    pub fn grid_spec(&self) -> ([f32; 3], f32, [u32; 3]) {
        let o = self.params.grid_origin;
        let d = self.params.grid_dims;
        ([o[0], o[1], o[2]], o[3], [d[0], d[1], d[2]])
    }

    /// Raw fixed-point grid totals `(mass_counts, momentum_counts)` summed in i64 — exact
    /// integer sums, so two scatters of the same state are bit-identical (dev/test only).
    /// Valid after `step()`: the grid holds that frame's P2G result until the next frame's
    /// `grid_clear`.
    pub fn read_grid_counts(&self) -> (i64, [i64; 3]) {
        let bytes = (self.num_nodes as u64) * 16;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_fp, bytes)).to_vec();
        let mut mass = 0i64;
        let mut mom = [0i64; 3];
        for n in raw.chunks_exact(4) {
            mass += n[0] as i64;
            mom[0] += n[1] as i64;
            mom[1] += n[2] as i64;
            mom[2] += n[3] as i64;
        }
        (mass, mom)
    }

    /// Decoded grid totals `(total_mass, total_momentum)` (dev/test only — stalls).
    pub fn read_grid_mass_momentum(&self) -> (f64, [f64; 3]) {
        let (mass, mom) = self.read_grid_counts();
        (
            mass as f64 / FP_SCALE,
            [
                mom[0] as f64 / FP_SCALE,
                mom[1] as f64 / FP_SCALE,
                mom[2] as f64 / FP_SCALE,
            ],
        )
    }

    /// Read back the maximum per-node fixed-point lane magnitude (overflow-probe gate support;
    /// dev/test only — stalls).
    pub fn read_grid_max_count(&self) -> i64 {
        let bytes = (self.num_nodes as u64) * 16;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_fp, bytes)).to_vec();
        raw.iter().map(|&c| (c as i64).abs()).max().unwrap_or(0)
    }

    /// Overwrite particle velocities (dev/test only). Length must equal the particle count.
    pub fn write_velocities_for_test(&self, velocities: &[[f32; 4]]) {
        assert_eq!(
            velocities.len(),
            self.params.particle_count as usize,
            "velocity seed length must match particle count"
        );
        self.queue
            .write_buffer(&self.vel, 0, bytemuck::cast_slice(velocities));
    }

    /// Overwrite the per-particle APIC C matrices (dev/test only). Three vec4 rows per
    /// particle, row-major: `rows[3p + r] = (C[r][0], C[r][1], C[r][2], 0)`.
    pub fn write_affine_for_test(&self, rows: &[[f32; 4]]) {
        assert_eq!(
            rows.len(),
            3 * self.params.particle_count as usize,
            "affine seed length must be 3 rows per particle"
        );
        self.queue
            .write_buffer(&self.cmat, 0, bytemuck::cast_slice(rows));
    }

    /// Switch G2P to the PIC variant (C zeroed each step) — ONLY for the APIC-vs-PIC
    /// discrimination gate; never set in production paths.
    pub fn set_pic_for_test(&mut self, pic: bool) {
        self.params.pic_mode = u32::from(pic);
    }

    /// Sample per-pass GPU timestamps into the cache that `profile()` returns. Blocks
    /// (dev/test/periodic only) — the explicit cache point, so the getters never stall.
    pub fn sample_diagnostics(&mut self) {
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
}

impl Solver for TwofieldSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let (positions, phases, water_count) = seed_ranges(scene, mats, cfg);
        let particle_count = positions.len() as u32;
        let solid_count = particle_count - water_count;

        let (origin, cell, dims) = grid_spec_for(scene, mats);
        let num_nodes = dims[0] * dims[1] * dims[2];

        let packed = {
            let mut packed = pack_solids(&scene.solids);
            if packed.is_empty() {
                packed.push(Primitive::zeroed()); // non-empty binding; num_solids = 0 skips it
            }
            packed
        };

        let params = Params {
            box_min: [scene.box_min[0], scene.box_min[1], scene.box_min[2], 0.0],
            box_max: [scene.box_max[0], scene.box_max[1], scene.box_max[2], 0.0],
            gravity: [scene.gravity[0], scene.gravity[1], scene.gravity[2], 0.0],
            grid_origin: [origin[0], origin[1], origin[2], cell],
            grid_dims: [dims[0], dims[1], dims[2], num_nodes],
            dt: 1.0 / 60.0,
            particle_mass: mats.particle_mass,
            max_speed: cfg.max_speed,
            pic_mode: 0,
            water_count,
            solid_count,
            particle_count,
            num_solids: scene.solids.len() as u32,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twofield-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let n = particle_count.max(1) as u64;
        let vec4 = n * 16;
        let pos = Arc::new(Self::storage(
            &device,
            "twofield-pos",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let vel = Arc::new(Self::storage(
            &device,
            "twofield-vel",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let phase = Arc::new(Self::storage(
            &device,
            "twofield-phase",
            n * 4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        // Chem/thermal lanes (c, T, _, _); zero-initialized by wgpu until the extraction units.
        let chem = Arc::new(Self::storage(
            &device,
            "twofield-chem",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        // APIC C matrices: 3 vec4 rows per particle, zero-initialized (C = 0 at rest).
        let cmat = Self::storage(
            &device,
            "twofield-cmat",
            n * 48,
            wgpu::BufferUsages::empty(),
        );
        let grid_bytes = (num_nodes.max(1) as u64) * 16;
        let grid_fp = Self::storage(
            &device,
            "twofield-grid-fp",
            grid_bytes,
            wgpu::BufferUsages::COPY_SRC,
        );
        let grid_vel = Self::storage(
            &device,
            "twofield-grid-vel",
            grid_bytes,
            wgpu::BufferUsages::empty(),
        );
        let solids_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twofield-solids"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        if particle_count > 0 {
            queue.write_buffer(&pos, 0, bytemuck::cast_slice(&positions));
            queue.write_buffer(&phase, 0, bytemuck::cast_slice(&phases));
        }
        // One readback scratch big enough for the largest readable buffer (particles or grid).
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("twofield-readback"),
            size: vec4.max(grid_bytes),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // WGSL has no imports: assemble the one module from the concern files. `common` declares
        // Params/bindings + shared helpers; `transfers` adds the APIC water passes.
        let shader_src = format!(
            "{}\n{}",
            include_str!("common.wgsl"),
            include_str!("transfers.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("twofield"),
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
        // Per-pipeline bind groups providing exactly the bindings each entry point uses
        // (auto layout drops declared-but-unused globals; binding indices stay global).
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
        let pipelines = {
            let grid_clear = make("grid_clear");
            let grid_clear_bind = bg(&grid_clear, &[(0, &params_buf), (6, &grid_fp)]);
            let p2g = make("p2g_water");
            let p2g_bind = bg(
                &p2g,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (5, &cmat),
                    (6, &grid_fp),
                ],
            );
            let grid_update = make("grid_update");
            let grid_update_bind = bg(
                &grid_update,
                &[
                    (0, &params_buf),
                    (6, &grid_fp),
                    (7, &grid_vel),
                    (8, &solids_buf),
                ],
            );
            let g2p = make("g2p_water");
            let g2p_bind = bg(
                &g2p,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (5, &cmat),
                    (7, &grid_vel),
                    (8, &solids_buf),
                ],
            );
            Pipelines {
                grid_clear: (grid_clear, grid_clear_bind),
                p2g_water: (p2g, p2g_bind),
                grid_update: (grid_update, grid_update_bind),
                g2p_water: (g2p, g2p_bind),
            }
        };

        let ts = if gpu.timestamps_supported {
            // 4 passes per frame today; headroom for the U3+ pipeline growth.
            let capacity = 16u32;
            let qset = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("twofield-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("twofield-ts-resolve"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("twofield-ts-readback"),
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
            water_count,
            solid_count,
            num_nodes,
            params_buf,
            pos,
            vel,
            phase,
            chem,
            cmat,
            grid_fp,
            readback,
            pipelines,
            ts,
            dispatches: 0,
            cached_passes: Vec::new(),
            initial_positions: positions,
            initial_phases: phases,
        }
    }

    fn reset(&mut self, _scene: &Scene) {
        // Exact, deterministic re-seed: positions + phase tags back to the seed; velocities,
        // chem lanes, and APIC C matrices re-zeroed. (The grid is cleared at the start of every
        // frame, so it needs no reset.)
        if !self.initial_positions.is_empty() {
            self.queue
                .write_buffer(&self.pos, 0, bytemuck::cast_slice(&self.initial_positions));
            self.queue
                .write_buffer(&self.phase, 0, bytemuck::cast_slice(&self.initial_phases));
            let zeros = vec![[0.0f32; 4]; self.initial_positions.len()];
            self.queue
                .write_buffer(&self.vel, 0, bytemuck::cast_slice(&zeros));
            self.queue
                .write_buffer(&self.chem, 0, bytemuck::cast_slice(&zeros));
            let czeros = vec![[0.0f32; 4]; 3 * self.initial_positions.len()];
            self.queue
                .write_buffer(&self.cmat, 0, bytemuck::cast_slice(&czeros));
        }
        self.cached_passes.clear();
        self.dispatches = 0;
    }

    fn step(&mut self, dt: f32, _input: &EmissionInput) {
        self.params.dt = dt;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut cursor = 0u32;
        let mut labels: Vec<String> = Vec::new();
        let mut dispatches = 0u32;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-frame"),
            });

        // The U2 APIC pipeline, one substep per frame. Water passes dispatch at least one
        // workgroup so the dispatch/profiling path is real even on an empty scene (threads
        // early-out on the water_count guard).
        let node_groups = groups(self.num_nodes).max(1);
        let water_groups = groups(self.water_count).max(1);
        let seq: [(&str, &(wgpu::ComputePipeline, wgpu::BindGroup), u32); 4] = [
            ("grid_clear", &self.pipelines.grid_clear, node_groups),
            ("p2g_water", &self.pipelines.p2g_water, water_groups),
            ("grid_update", &self.pipelines.grid_update, node_groups),
            ("g2p_water", &self.pipelines.g2p_water, water_groups),
        ];
        for (label, (pipe, bind), g) in seq {
            dispatch_pass(
                &mut enc,
                pipe,
                bind,
                self.ts.as_ref(),
                &mut cursor,
                &mut labels,
                label,
                g,
                &mut dispatches,
            );
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
            particle_count: self.params.particle_count,
            position: Some(Arc::clone(&self.pos)),
            velocity: Some(Arc::clone(&self.vel)),
            phase_tag: Some(Arc::clone(&self.phase)),
            // The chem buffer carries both lanes: concentration (.x) and temperature (.y).
            concentration: Some(Arc::clone(&self.chem)),
            temperature: Some(Arc::clone(&self.chem)),
            ..Default::default()
        }
    }

    fn metrics(&self) -> Metrics {
        // All values CPU-resident — never a GPU sync here (trait contract).
        Metrics {
            particle_count: self.params.particle_count,
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

#[cfg(test)]
mod seed_tests {
    use super::*;

    /// The range layout is water-first: every water particle precedes every solid.
    #[test]
    fn seeds_water_first_then_solids() {
        let scene = Scene::pour_over();
        let (pos, phase, water_count) =
            seed_ranges(&scene, &Materials::default(), &Config::default());
        assert_eq!(pos.len(), phase.len());
        assert!(water_count > 0, "pour_over seeds water");
        assert!(
            (water_count as usize) < phase.len(),
            "pour_over seeds solids too"
        );
        for (i, &ph) in phase.iter().enumerate() {
            let expect = if (i as u32) < water_count { 0 } else { 1 };
            assert_eq!(ph, expect, "particle {i} out of range order");
        }
        // Moisture lane per species: water full (1), grain dry (0).
        for (p, &ph) in pos.iter().zip(&phase) {
            assert_eq!(p[3], if ph == 0 { 1.0 } else { 0.0 });
        }
    }

    /// Seeding is deterministic for a fixed seed (R6).
    #[test]
    fn seeding_is_deterministic() {
        let scene = Scene::pour_over();
        let a = seed_ranges(&scene, &Materials::default(), &Config::default());
        let b = seed_ranges(&scene, &Materials::default(), &Config::default());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }

    /// The grid pads one node layer below box_min and covers the full B-spline support of any
    /// in-box particle: base = floor((x − origin)/h − 0.5) ∈ [0, dims − 3] at both extremes.
    #[test]
    fn grid_spec_covers_box_with_padding() {
        let scene = Scene::default();
        let mats = Materials::default();
        let (origin, h, dims) = grid_spec_for(&scene, &mats);
        assert_eq!(h, CELL_SIZE_FACTOR * mats.particle_spacing);
        for a in 0..3 {
            assert_eq!(origin[a], scene.box_min[a] - h);
            for x in [scene.box_min[a], scene.box_max[a]] {
                let xl = (x - origin[a]) / h;
                let base = (xl - 0.5).floor() as i64;
                assert!(base >= 0, "axis {a}: base {base} below grid at x={x}");
                assert!(
                    base + 2 <= dims[a] as i64 - 1,
                    "axis {a}: support exceeds grid at x={x} (base {base}, dims {})",
                    dims[a]
                );
            }
        }
    }
}

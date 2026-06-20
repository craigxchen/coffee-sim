//! Position-Based MPM (PB-MPM) solver under `SolverId::Pbmpm` — the U1 scaffold of
//! `docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md`.
//!
//! U1 pins the seam contract (the six `Solver` methods), the per-particle PB-MPM state layout
//! (KTD8: `deformation_displacement` D + `deformation_gradient` F as mat3x3 floats, the liquid
//! constraint scalars — all in the PARTICLE buffer; the GRID carries ONLY [mass, mom.xyz] as
//! fixed-point `atomic<i32>`), the `Params` Rust↔WGSL ABI lock, the `FP_SCALE` fixed-point
//! encoding, the WGSL-concatenation module assembly, and the registry seam. It mirrors
//! `src/solvers/twofield/` structurally (KTD2). U3 (this revision) replaces the U1 gravity
//! `advect` placeholder with a single APIC fixed-point transfer cycle per substep:
//! `grid_clear -> p2g -> grid_update -> g2p` (`transfers.wgsl`). Water now scatters mass + APIC
//! momentum to the 3x3x3 fixed-point grid, gains gravity on the grid, and gathers velocity +
//! reconstructs the per-particle affine matrix D, so it falls and accumulates (compressibly; the
//! compliant density constraint that gives incompressibility/bounce is U4). Collider BC/restitution
//! is U5.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::emission::EmissionInput;
use crate::engine::scene::Species;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::{Profile, Profiler};
use crate::solvers::base::Solver;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

const WG: u32 = 256;

/// Y-coordinate at which un-emitted (dormant) water-pool slots are parked: far below any scene so
/// the renderer culls them (`ui/particles.wgsl` matches this with `p.y <= -1.0e8`). Activated slots
/// get a real position from `emit()`. Kept finite (not NaN) so any accidental read stays defined.
/// (Mirrors twofield's `DORMANT_PARK_Y`.)
const DORMANT_PARK_Y: f32 = -1.0e9;

/// Reduced-units volume calibration (KEEP.md §2; mirrors twofield's constant): mL per scene unit³ —
/// sizes the pour pool from a scene's declared `pour_water_ml`.
const ML_PER_SIM_UNIT3: f32 = 5.20;

fn groups(n: u32) -> u32 {
    n.div_ceil(WG)
}

/// Storage-buffer budget (KTD5): per U3 entry point (the params uniform does NOT count against
/// the storage limit) —
///   `grid_clear`  1 (grid_fp);
///   `p2g`         4 (pos, vel, deform_disp, grid_fp) — reads pos/vel/deform_disp, scatters grid_fp;
///   `grid_update` 2 (grid_fp, grid_vel);
///   `g2p`         4 (pos, vel, deform_disp, grid_vel) — the OTHER widest entry point.
/// Widest = `p2g`/`g2p` at 4 storage buffers. Well within the 9 grant
/// the device requests (`src/utils/gpu.rs::NEEDED_STORAGE_BUFFERS`). Re-derive when U4's constraint
/// pass lands (it adds deform_grad reads). The compliant-density constraint + iteration loop is U4.
pub const MAX_STORAGE_BUFFERS_PER_ENTRY_POINT: u32 = 4;

/// Fixed-point scale for the grid atomics — mirrors `FP_SCALE` in `common.wgsl` (2^18,
/// KEEP.md §3). Coupled to `Config::max_speed` (the overflow-headroom derivation lives next to
/// the WGSL constant); the light U3 overflow probe is in `tests/pbmpm_transfers.rs` (the pinned
/// perf/range gate is U6).
pub const FP_SCALE: f64 = 262144.0;

/// Default PB-MPM liquid-constraint knobs (KTD1). Inert in U1 — the compliant density
/// constraint that reads them lands in U4; carried in `Params` now so the ABI is fixed up front.
pub const LIQUID_DENSITY_DEFAULT: f32 = 1.0;
pub const LIQUID_RELAXATION_DEFAULT: f32 = 1.0;
pub const LIQUID_VISCOSITY_DEFAULT: f32 = 0.0;
pub const ITERATION_COUNT_DEFAULT: u32 = 2;

/// Fine-grid cell size as a multiple of the particle spacing — ~2× spacing gives the quadratic
/// B-spline support a ≈3-spacing reach with ≈8 particles per cell at rest (mirrors twofield's
/// CELL_SIZE_FACTOR; the transfers in U3 pick the final resolution).
pub const CELL_SIZE_FACTOR: f32 = 2.0;

/// Dispatches per substep at the U3 knobs: `grid_clear → p2g → grid_update → g2p`. One substep per
/// frame for now (U4 adds the iteration_count loop; a CFL substep policy is a later decision).
pub const DISPATCHES_PER_FRAME: u32 = 4;

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
    water_count: u32,    // particles [0, water_count) are live water
    particle_count: u32, // allocated pool size (kernel live-set guard)
    // PB-MPM liquid constraint knobs (KTD1; inert in U1, read by the U4 constraint).
    liquid_density: f32,
    liquid_relaxation: f32,
    liquid_viscosity: f32,
    // .x = iteration_count (PB-MPM constraint→grid rebuild loop); .yzw = tail pad. Packed as one
    // vec4 so the WGSL vec4<u32> alignment matches Rust's `[u32; 4]` exactly (a bare u32 followed
    // by a vec3 pad disagrees: WGSL aligns the vec3 to 16, Rust does not).
    iter_pad: [u32; 4],
}

// Params is uploaded as a uniform and must stay byte-identical to the WGSL `Params`.
const _: () = assert!(std::mem::size_of::<Params>() == 128);

/// Compute grid dimensions for a scene: cell size `h = CELL_SIZE_FACTOR · particle_spacing`,
/// node 0 one cell OUTSIDE `box_min` (a pad layer), `ceil(extent/h) + 3` nodes per axis (mirrors
/// twofield's `grid_spec_for`, so the B-spline support of any in-box particle stays in range).
pub fn grid_spec_for(scene: &Scene, mats: &Materials) -> ([f32; 3], f32, [u32; 3]) {
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

/// Seed the scene's water regions as a lattice with seeded jitter (mirrors twofield's
/// `seed_ranges`, water-only for the single-phase prototype). RNG draws happen per lattice point
/// regardless of rejection, so the layout is deterministic for a given `cfg.seed`. The 4th
/// position lane carries the moisture fraction (1 = full). Returns `(positions, phases)`.
fn seed_water(scene: &Scene, mats: &Materials, cfg: &Config) -> (Vec<[f32; 4]>, Vec<u32>) {
    const SEED_CLEARANCE: f32 = 0.4;
    let mut rng = crate::utils::rng::Rng::new(cfg.seed);
    let mut pos: Vec<[f32; 4]> = Vec::new();
    for region in &scene.regions {
        // Single-phase water prototype: only water regions seed particles.
        if region.species != Species::Water {
            continue;
        }
        let (lo, hi) = (region.min, region.max);
        let s = mats.particle_spacing;
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
                            0,
                        );
                        if c.signed < SEED_CLEARANCE {
                            continue;
                        }
                    }
                    pos.push([px, py, pz, 1.0]);
                }
            }
        }
    }
    let phases = vec![0u32; pos.len()];
    (pos, phases)
}

/// Pour-emission state + spout parameters (host-side; ported verbatim from twofield's `Inflow`).
/// Turns `EmissionInput` into activated water-pool particles via the volume-consistent
/// arclength-credit emitter: the volume accumulator (flow/V_w·dt) is the master count budget;
/// layers release one `particle_spacing` of stream travel apart, each filling a golden-angle disc,
/// so the inlet packs to the fluid's rest density.
struct Inflow {
    // Static (from Config/Materials at build).
    nozzle_radius: f32,
    discharge_coeff: f32,
    spacing: f32,
    v_w: f32,
    pour_t: f32,
    // State.
    accumulator: f32, // volume budget in particles (carries the sub-particle fraction)
    axial: f32,       // arclength credit (scene units) toward the next layer
    last_exit_speed: f32, // drains backlog at the last cadence when flow drops to 0
    cursor: u64,      // golden-angle determinism across all emitted particles
}

/// Orthonormal disc basis perpendicular to a (unit) pour direction `dir` (mirrors twofield).
fn disc_basis(dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1.0e-9);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let refv = if dir[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = norm(cross(dir, refv));
    let w = cross(dir, u);
    (u, w)
}

/// Per-particle identity deformation gradients (F = I), 3 vec4 rows each — the U1 seed for the
/// KTD8 state (D is left zero-initialized by wgpu).
fn identity_rows(count: u32) -> Vec<[f32; 4]> {
    let mut rows = Vec::with_capacity(3 * count as usize);
    for _ in 0..count {
        rows.push([1.0, 0.0, 0.0, 0.0]);
        rows.push([0.0, 1.0, 0.0, 0.0]);
        rows.push([0.0, 0.0, 1.0, 0.0]);
    }
    rows
}

pub struct PbmpmSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Params,
    water_count: u32,
    // Pre-allocated water pool: buffers + readbacks span `[0, water_capacity)`; the live water
    // range is `[0, water_count)` and grows toward `water_capacity` as `emit()` activates dormant
    // slots. Dormant slots `[water_count, water_capacity)` are parked off-scene and never dispatched
    // (the kernels guard on `water_count`).
    water_capacity: u32,
    inflow: Inflow,
    num_nodes: u32,

    params_buf: wgpu::Buffer,
    // Canonical particle state, exposed through `ParticleBuffers`.
    pos: Arc<wgpu::Buffer>,
    vel: Arc<wgpu::Buffer>,
    phase: Arc<wgpu::Buffer>,
    chem: Arc<wgpu::Buffer>,
    // Per-particle PB-MPM state (KTD8): deformation displacement D and gradient F, 3 vec4 rows
    // each. Floats in the particle buffer — never on the grid.
    deform_disp: wgpu::Buffer,
    deform_grad: wgpu::Buffer,
    // LIQUID grid field: fixed-point atomic<i32>, 4 lanes per node [mass, mom.xyz] (KTD8).
    grid_fp: wgpu::Buffer,
    // Decoded grid velocity (.xyz) + node mass (.w) after grid_update — a FLOAT transient (KTD8:
    // the grid carries only the atomic mass/momentum; this is the decode scratch g2p gathers from).
    grid_vel: wgpu::Buffer,
    readback: wgpu::Buffer,

    pipelines: Pipelines,
    profiler: Profiler,

    // Retained for reset (exact, deterministic re-seed).
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
    // Live water count at the seed (the count `reset()` returns to before any pour activates slots).
    initial_water: u32,
}

struct Pipelines {
    grid_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p: (wgpu::ComputePipeline, wgpu::BindGroup),
}

impl PbmpmSolver {
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
                label: Some("pbmpm-readback"),
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

    /// Raw fixed-point grid totals `(mass_counts, momentum_counts)` summed in i64 — exact
    /// integer sums, so two scatters of the same state are bit-identical (dev/test only). Valid
    /// after `step()`: the grid holds that frame's P2G result until the next frame's `grid_clear`.
    pub fn read_grid_counts(&self) -> (i64, [i64; 3]) {
        let bytes = (self.num_nodes as u64) * 16;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_fp, bytes)).to_vec();
        let mut mass = 0i64;
        let mut mom = [0i64; 3];
        for node in raw.chunks_exact(4) {
            mass += node[0] as i64;
            mom[0] += node[1] as i64;
            mom[1] += node[2] as i64;
            mom[2] += node[3] as i64;
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

    /// Maximum per-node fixed-point lane magnitude (overflow-probe support; dev/test only).
    pub fn read_grid_max_count(&self) -> i64 {
        let bytes = (self.num_nodes as u64) * 16;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_fp, bytes)).to_vec();
        raw.iter().map(|&c| (c as i64).abs()).max().unwrap_or(0)
    }

    /// Read back the decoded float grid velocity (.xyz) + node mass (.w) after `grid_update`
    /// (dev/test only — stalls). Valid until the next frame's `grid_update` overwrites it.
    pub fn read_grid_velocities(&self) -> Vec<[f32; 4]> {
        let bytes = (self.num_nodes as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.grid_vel, bytes)).to_vec()
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

    /// Grid layout `(origin, cell_size, dims)` for CPU twins (dev/test only).
    pub fn grid_spec(&self) -> ([f32; 3], f32, [u32; 3]) {
        let o = self.params.grid_origin;
        let d = self.params.grid_dims;
        ([o[0], o[1], o[2]], o[3], [d[0], d[1], d[2]])
    }

    /// Live water count currently simulated.
    pub fn active_count(&self) -> u32 {
        self.water_count
    }

    /// Pre-allocated water-pool capacity (the live count never exceeds this).
    pub fn capacity(&self) -> u32 {
        self.water_capacity
    }

    /// Activate pour-emitted water particles for this frame from `EmissionInput` — the
    /// volume-consistent arclength-credit emitter, ported verbatim from twofield's `emit`: the
    /// volume accumulator (flow/V_w·dt) is the master count budget; layers release one
    /// `particle_spacing` of stream travel apart, each filling a golden-angle disc with up to
    /// `N_layer = ceil(A_eff·spacing/V_w)` particles, so the inlet packs to rest density. Activated
    /// slots are written into the water pool's live range (real position/velocity, identity F via
    /// the dormant-slot seed, zero D) and `water_count` grows. No-ops when not pouring and no
    /// backlog remains. (Newly activated slots inherit the seed's identity F + zero D; the
    /// dormant-slot writes only overwrite pos/vel/chem.)
    fn emit(&mut self, input: &EmissionInput, dt: f32) {
        // A Reset event clears the emitter's backlog/credit (the emitter contract; a full sim
        // restart is `reset()`). Done before the gate so a Reset with zero flow clears.
        if input.event == crate::emission::PourEvent::Reset {
            self.inflow.accumulator = 0.0;
            self.inflow.axial = 0.0;
            self.inflow.last_exit_speed = 0.0;
        }
        let flow = input.flow_rate.max(0.0);
        // Gate: pour active, or a whole particle of backlog still to drain.
        if flow <= 0.0 && self.inflow.accumulator < 1.0 {
            return;
        }
        let a_eff = std::f32::consts::PI
            * self.inflow.nozzle_radius
            * self.inflow.nozzle_radius
            * self.inflow.discharge_coeff;
        let a_eff = a_eff.max(1.0e-9);
        // Orifice relation: exit speed from flow + effective area. While draining a backlog at zero
        // flow, keep the last cadence so the stream tail stays correctly spaced.
        let exit_speed = if flow > 0.0 {
            let es = flow / a_eff;
            self.inflow.last_exit_speed = es;
            self.inflow.accumulator += flow / self.inflow.v_w * dt;
            es
        } else {
            self.inflow.last_exit_speed
        };
        if exit_speed <= 0.0 {
            return;
        }
        // Volume-consistent layer capacity (ceil ⇒ throughput ≥ flow/V_w, no backlog).
        let n_layer = ((a_eff * self.inflow.spacing / self.inflow.v_w).ceil() as u32).max(1);

        // Pour direction (downward, tilted by pour_angle toward +x) + a disc basis.
        let a = input.pour_angle;
        let dir = [a.sin(), -a.cos(), 0.0];
        let (u, w) = disc_basis(dir);
        let r_eff = self.inflow.nozzle_radius * self.inflow.discharge_coeff.sqrt();
        const GOLDEN: f32 = 2.399_963_2;

        self.inflow.axial += exit_speed * dt;
        let kettle = input.kettle_pos;
        let mut want = self.inflow.accumulator.floor() as u32;
        let mut new_pos: Vec<[f32; 4]> = Vec::new();
        let mut new_vel: Vec<[f32; 4]> = Vec::new();
        let mut new_chem: Vec<[f32; 4]> = Vec::new();
        let mut clamped = false;
        while self.inflow.axial >= self.inflow.spacing && want > 0 {
            // Capacity check BEFORE spending arclength credit, so a full pool doesn't silently
            // consume a layer's axial.
            let avail = self.water_capacity - (self.water_count + new_pos.len() as u32);
            if avail == 0 {
                clamped = true;
                break;
            }
            self.inflow.axial -= self.inflow.spacing;
            let depth = self.inflow.axial; // residual stream travel below the nozzle
            let this_layer = n_layer.min(want).min(avail);
            for _ in 0..this_layer {
                // Radial shell cycles with the cursor (mod N_layer) so partial layers still cover
                // the whole disc over time; golden angle fills it uniformly.
                let ri = (self.inflow.cursor % n_layer as u64) as f32;
                let r = r_eff * ((ri + 0.5) / n_layer as f32).sqrt();
                let theta = self.inflow.cursor as f32 * GOLDEN;
                self.inflow.cursor += 1;
                let (ct, st) = (theta.cos(), theta.sin());
                let off = [
                    u[0] * r * ct + w[0] * r * st,
                    u[1] * r * ct + w[1] * r * st,
                    u[2] * r * ct + w[2] * r * st,
                ];
                new_pos.push([
                    kettle[0] + dir[0] * depth + off[0],
                    kettle[1] + dir[1] * depth + off[1],
                    kettle[2] + dir[2] * depth + off[2],
                    1.0, // moisture lane: full water
                ]);
                new_vel.push([
                    dir[0] * exit_speed,
                    dir[1] * exit_speed,
                    dir[2] * exit_speed,
                    0.0,
                ]);
                new_chem.push([0.0, self.inflow.pour_t, 0.0, 0.0]); // c = 0, T = pour temp
            }
            want -= this_layer;
        }
        if clamped {
            eprintln!(
                "pbmpm pour: water pool {} reached; emission clamped (a recipe scene must declare \
                 enough pour_water_ml to size the pool to its dose)",
                self.water_capacity
            );
        }
        let emit_n = new_pos.len() as u32;
        if emit_n == 0 {
            return;
        }
        // Clamp-before-decrement: subtract only what was actually emitted — unspent budget stays as
        // backlog rather than being silently burned.
        self.inflow.accumulator -= emit_n as f32;
        let off_v4 = (self.water_count as u64) * 16;
        self.queue
            .write_buffer(&self.pos, off_v4, bytemuck::cast_slice(&new_pos));
        self.queue
            .write_buffer(&self.vel, off_v4, bytemuck::cast_slice(&new_vel));
        self.queue
            .write_buffer(&self.chem, off_v4, bytemuck::cast_slice(&new_chem));
        // Dormant pool slots already carry phase 0 (water), identity F, and zero D (the build seed);
        // activation only overwrites pos/vel/chem, so the KTD8 per-particle state stays consistent.
        self.water_count += emit_n;
        self.params.water_count = self.water_count;
    }
}

impl Solver for PbmpmSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let (seed_positions, seed_phases) = seed_water(scene, mats, cfg);
        let water_seed = seed_positions.len() as u32;

        // Pre-allocated water pool (KTD2, mirroring twofield): seed + dose headroom for a declared
        // pour (KEEP §2 calibration; V_w = spacing³ at this solver's rest density). The live water
        // range `[0, water_count)` grows toward `water_capacity` as `emit()` activates dormant slots
        // each frame; dormant slots `[water_count, water_capacity)` are never dispatched (the kernels
        // guard on the LIVE count) and are parked at a far below-domain sentinel (y = DORMANT_PARK_Y)
        // so the renderer culls them. `emit()` overwrites a slot's pos/vel/chem when it activates it.
        let v_w = mats.particle_spacing.powi(3);
        let dose_headroom = if scene.declares_pour() {
            (scene.pour_water_ml / ML_PER_SIM_UNIT3 / v_w).ceil() as u32
        } else {
            0
        };
        let water_capacity = water_seed + dose_headroom;
        let particle_count = water_capacity; // pool size (buffers + readbacks)
        let water_count = water_seed; // live count
                                      // Park the dormant tail far below the domain (matched by the cull in ui/particles.wgsl).
        let park = [scene.box_min[0], DORMANT_PARK_Y, scene.box_min[2], 0.0];
        let mut positions = seed_positions;
        positions.resize(water_capacity as usize, park);
        let mut phases = seed_phases;
        phases.resize(water_capacity as usize, 0);

        let (origin, cell, dims) = grid_spec_for(scene, mats);
        let num_nodes = dims[0] * dims[1] * dims[2];

        let params = Params {
            box_min: [scene.box_min[0], scene.box_min[1], scene.box_min[2], 0.0],
            box_max: [scene.box_max[0], scene.box_max[1], scene.box_max[2], 0.0],
            gravity: [scene.gravity[0], scene.gravity[1], scene.gravity[2], 0.0],
            grid_origin: [origin[0], origin[1], origin[2], cell],
            grid_dims: [dims[0], dims[1], dims[2], num_nodes],
            dt: 1.0 / 60.0,
            particle_mass: mats.particle_mass,
            max_speed: cfg.max_speed,
            water_count,
            particle_count,
            liquid_density: LIQUID_DENSITY_DEFAULT,
            liquid_relaxation: LIQUID_RELAXATION_DEFAULT,
            liquid_viscosity: LIQUID_VISCOSITY_DEFAULT,
            iter_pad: [ITERATION_COUNT_DEFAULT, 0, 0, 0],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbmpm-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let n = particle_count.max(1) as u64;
        let vec4 = n * 16;
        let pos = Arc::new(Self::storage(
            &device,
            "pbmpm-pos",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let vel = Arc::new(Self::storage(
            &device,
            "pbmpm-vel",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let phase = Arc::new(Self::storage(
            &device,
            "pbmpm-phase",
            n * 4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let chem = Arc::new(Self::storage(
            &device,
            "pbmpm-chem",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        // KTD8 per-particle state: D (zero) + F (identity), 3 vec4 rows per particle.
        let deform_disp = Self::storage(
            &device,
            "pbmpm-deform-disp",
            n * 48,
            wgpu::BufferUsages::COPY_SRC,
        );
        let deform_grad = Self::storage(
            &device,
            "pbmpm-deform-grad",
            n * 48,
            wgpu::BufferUsages::COPY_SRC,
        );
        // KTD8 grid field: 4 fixed-point lanes per node [mass, mom.xyz].
        let grid_fp = Self::storage(
            &device,
            "pbmpm-grid-fp",
            (num_nodes.max(1) as u64) * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        // Decoded grid velocity (.xyz) + node mass (.w) — float transient (KTD8). One vec4 per node.
        let grid_vel = Self::storage(
            &device,
            "pbmpm-grid-vel",
            (num_nodes.max(1) as u64) * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbmpm-readback"),
            size: vec4.max(grid_fp.size()),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Seed the FULL pool: live water positions + parked dormant tail, all phase 0, identity F
        // across the whole capacity so a slot `emit()` later activates is already F = I (activation
        // only overwrites pos/vel/chem). D/vel/chem stay zero.
        if water_capacity > 0 {
            queue.write_buffer(&pos, 0, bytemuck::cast_slice(&positions));
            queue.write_buffer(&phase, 0, bytemuck::cast_slice(&phases));
            queue.write_buffer(
                &deform_grad,
                0,
                bytemuck::cast_slice(&identity_rows(water_capacity)),
            );
        }

        // One shader module from the concatenated WGSL (WGSL has no imports): common.wgsl declares
        // the shared bindings/helpers, transfers.wgsl adds the U3 p2g/grid_update/g2p passes. The
        // U4 constraint file concatenates after these.
        let shader_src = format!(
            "{}\n{}",
            include_str!("common.wgsl"),
            include_str!("transfers.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pbmpm"),
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
        // Per-pipeline bind groups providing exactly the bindings each entry point uses (auto
        // layout drops declared-but-unused globals; binding indices stay global).
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
            let grid_clear_bind = bg(&grid_clear, &[(0, &params_buf), (7, &grid_fp)]);
            // p2g: pos, vel, deform_disp (read affine D), grid_fp (scatter). 4 storage buffers.
            let p2g = make("p2g");
            let p2g_bind = bg(
                &p2g,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (5, &deform_disp),
                    (7, &grid_fp),
                ],
            );
            // grid_update: grid_fp (decode), grid_vel (write). 2 storage buffers.
            let grid_update = make("grid_update");
            let grid_update_bind = bg(
                &grid_update,
                &[(0, &params_buf), (7, &grid_fp), (8, &grid_vel)],
            );
            // g2p: pos, vel, deform_disp (write D), grid_vel (gather). 4 storage buffers.
            let g2p = make("g2p");
            let g2p_bind = bg(
                &g2p,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (5, &deform_disp),
                    (8, &grid_vel),
                ],
            );
            Pipelines {
                grid_clear: (grid_clear, grid_clear_bind),
                p2g: (p2g, p2g_bind),
                grid_update: (grid_update, grid_update_bind),
                g2p: (g2p, g2p_bind),
            }
        };

        Self {
            device,
            queue,
            params,
            water_count,
            water_capacity,
            inflow: Inflow {
                nozzle_radius: cfg.nozzle_radius,
                discharge_coeff: cfg.discharge_coeff,
                spacing: mats.particle_spacing,
                v_w,
                pour_t: mats.pour_t,
                accumulator: 0.0,
                axial: 0.0,
                last_exit_speed: 0.0,
                cursor: 0,
            },
            num_nodes,
            params_buf,
            pos,
            vel,
            phase,
            chem,
            deform_disp,
            deform_grad,
            grid_fp,
            grid_vel,
            readback,
            pipelines,
            profiler: Profiler::new(gpu.timestamps_supported),
            initial_positions: positions,
            initial_phases: phases,
            initial_water: water_seed,
        }
    }

    fn reset(&mut self, _scene: &Scene) {
        // Exact, deterministic re-seed of the FULL pool: positions + phase tags + identity F back to
        // the seed (pour-activated slots return to the parked dormant state); velocities, chem lanes,
        // and D re-zeroed; the live count drops back to the seed; the emitter restarts. (The grid is
        // cleared at the start of every frame, so it needs no reset.)
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
            let dzeros = vec![[0.0f32; 4]; 3 * self.initial_positions.len()];
            self.queue
                .write_buffer(&self.deform_disp, 0, bytemuck::cast_slice(&dzeros));
            self.queue.write_buffer(
                &self.deform_grad,
                0,
                bytemuck::cast_slice(&identity_rows(self.water_capacity)),
            );
        }
        self.water_count = self.initial_water;
        self.params.water_count = self.initial_water;
        self.inflow.accumulator = 0.0;
        self.inflow.axial = 0.0;
        self.inflow.last_exit_speed = 0.0;
        self.inflow.cursor = 0;
    }

    fn step(&mut self, dt: f32, input: &EmissionInput) {
        // Pour emission first (grows water_count for this frame); no-op when not pouring. Newly
        // activated slots are written into the live range so the transfer dispatches below (guarded
        // on water_count) pick them up.
        self.emit(input, dt);
        debug_assert!(self.water_count <= self.water_capacity);

        self.profiler.begin_frame();
        // One substep per frame for now (U4 adds the iteration_count loop; a CFL substep policy is
        // a later decision — twofield also runs one substep in its U2 transfer revision).
        self.params.dt = dt;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbmpm-frame"),
            });
        // The single APIC transfer cycle: grid_clear -> p2g -> grid_update -> g2p. Each pass
        // dispatches at least one workgroup so the dispatch/profiling path is real even on an empty
        // scene (threads early-out on the live-set guards — over-dispatch + early-out, never
        // indirect dispatch). Fixed-point P2G is order-independent, so the cycle is deterministic.
        let node_groups = groups(self.num_nodes).max(1);
        let water_groups = groups(self.water_count).max(1);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pbmpm-frame"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.grid_clear.0);
            pass.set_bind_group(0, Some(&self.pipelines.grid_clear.1), &[]);
            pass.dispatch_workgroups(node_groups, 1, 1);
            self.profiler.record_dispatch();
            pass.set_pipeline(&self.pipelines.p2g.0);
            pass.set_bind_group(0, Some(&self.pipelines.p2g.1), &[]);
            pass.dispatch_workgroups(water_groups, 1, 1);
            self.profiler.record_dispatch();
            pass.set_pipeline(&self.pipelines.grid_update.0);
            pass.set_bind_group(0, Some(&self.pipelines.grid_update.1), &[]);
            pass.dispatch_workgroups(node_groups, 1, 1);
            self.profiler.record_dispatch();
            pass.set_pipeline(&self.pipelines.g2p.0);
            pass.set_bind_group(0, Some(&self.pipelines.g2p.1), &[]);
            pass.dispatch_workgroups(water_groups, 1, 1);
            self.profiler.record_dispatch();
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn particles(&self) -> ParticleBuffers {
        ParticleBuffers {
            particle_count: self.water_count,
            position: Some(Arc::clone(&self.pos)),
            velocity: Some(Arc::clone(&self.vel)),
            phase_tag: Some(Arc::clone(&self.phase)),
            concentration: Some(Arc::clone(&self.chem)),
            temperature: Some(Arc::clone(&self.chem)),
            ..Default::default()
        }
    }

    fn metrics(&self) -> Metrics {
        Metrics {
            particle_count: self.water_count,
            ..Default::default()
        }
    }

    fn profile(&self) -> Profile {
        self.profiler.snapshot()
    }
}

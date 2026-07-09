//! Position-Based MPM (PB-MPM) solver under `SolverId::Pbmpm` — the U1 scaffold of
//! `docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md`.
//!
//! U1 pins the seam contract (the six `Solver` methods), the per-particle PB-MPM state layout
//! (KTD8: `deformation_displacement` D + `deformation_gradient` F as mat3x3 floats, the liquid
//! constraint scalars — all in the PARTICLE buffer; the GRID carries ONLY [mass, mom.xyz] as
//! fixed-point `atomic<i32>`), the `Params` Rust↔WGSL ABI lock, the `FP_SCALE` fixed-point
//! encoding, the WGSL-concatenation module assembly, and the registry seam. It mirrors
//! `src/solvers/twofield/` structurally (KTD2). U3 added the APIC fixed-point transfer cycle
//! (`grid_clear -> p2g -> grid_update -> g2p`, `transfers.wgsl`). U4 (this revision) adds the
//! compliant density constraint that makes the water STIFF so it bounces (`constraint.wgsl`): per
//! substep the bundle `particle_update -> grid_clear -> p2g -> grid_update -> g2p` repeats
//! `iteration_count` times (the grid rebuilds each iteration so the per-particle correction
//! propagates spatially), then `particle_integrate` advects ONCE on the converged velocity (the
//! advection was moved out of g2p). D is zeroed per substep (`deform_clear`) and accumulated across
//! the iterations; F carries across substeps. The four knobs (iteration_count, liquid_density,
//! liquid_relaxation, liquid_viscosity) are plumbed from `Config` and have `set_*_for_test`
//! setters. Collider BC/restitution is U5.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::emission::EmissionInput;
use crate::engine::scene::Species;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::solvers::base::Solver;
use crate::solvers::pass_recorder::{PassRecorder, TimestampSink};
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

/// Storage-buffer budget (KTD5): per entry point (the params uniform does NOT count against the
/// storage limit) —
///   `grid_clear`         1 (grid_fp);
///   `particle_update`    2 (deform_disp, deform_grad) — U4 constraint, reads+writes D, reads the
///                          U6 per-particle liquidDensity lane (deform_grad[3p+0].x);
///   `p2g`                4 (pos, vel, deform_disp, grid_fp) — reads pos/vel/deform_disp, scatters;
///   `grid_decode_old`    2 (grid_fp, grid_vel_old) — SPLASH FLIP snapshot decode (no gravity/BC);
///   `grid_update`        5 (grid_fp, grid_vel, solids, bed_occupancy, seam_reaction) — U5 adds
///                          the collider node BC; the seam bed BC adds the two seam lanes
///                          (1-element dummies when `pbmpm_seam_bed` is off);
///   `g2p`                4 (pos, vel, deform_disp, grid_vel);
///   `particle_integrate` 8 (pos, vel, deform_disp, deform_grad, solids, grid_vel_old, vel_prev,
///                          bed_occupancy) — the SPLASH FLIP blend (gather grid_vel_old + read
///                          vel_prev) + advect once + the U5 SDF push-out/restitution + the U6
///                          per-particle liquidDensity accumulation + the seam bed-band PIC
///                          override (reads bed_occupancy; dummy when the seam BC is off).
/// Widest = `particle_integrate` at 8 storage buffers.
/// Within the 9 grant the device requests (`src/utils/gpu.rs::NEEDED_STORAGE_BUFFERS`).
pub const MAX_STORAGE_BUFFERS_PER_ENTRY_POINT: u32 = 8;

/// Fixed-point scale for the grid atomics — mirrors `FP_SCALE` in `common.wgsl` (2^18,
/// KEEP.md §3). Coupled to `Config::max_speed` (the overflow-headroom derivation lives next to
/// the WGSL constant); the light U3 overflow probe is in `tests/pbmpm_transfers.rs` (the pinned
/// perf/range gate is U6).
pub const FP_SCALE: f64 = 262144.0;

/// Default PB-MPM liquid-constraint knobs (KTD1). These mirror the `Config` defaults
/// (`pbmpm_*`) and are the fallback if a caller hands a `Config` that left them at zero; the
/// live values come from `Config` in `build()`. The compliant density constraint that reads them
/// is the U4 `particle_update` pass.
pub const LIQUID_DENSITY_DEFAULT: f32 = 1.0;
pub const LIQUID_RELAXATION_DEFAULT: f32 = 0.5;
pub const LIQUID_VISCOSITY_DEFAULT: f32 = 0.01;
pub const ITERATION_COUNT_DEFAULT: u32 = 2;

/// Fine-grid cell size as a multiple of the particle spacing — ~2× spacing gives the quadratic
/// B-spline support a ≈3-spacing reach with ≈8 particles per cell at rest (mirrors twofield's
/// CELL_SIZE_FACTOR; the transfers in U3 pick the final resolution).
pub const CELL_SIZE_FACTOR: f32 = 2.0;

/// Dispatches per substep (U4 structure + SPLASH snapshot): `deform_clear` once, then the SPLASH
/// FLIP snapshot `grid_clear → p2g → grid_decode_old` (3 passes), then the iteration loop runs
/// `iteration_count` × the 5-pass bundle `particle_update → grid_clear → p2g → grid_update → g2p`,
/// then `particle_integrate` once, i.e. `1 + 3 + 5·iteration_count + 1` = `5 + 5·iteration_count` per
/// substep. One substep per frame for now (a CFL substep policy is a later decision). The profiler
/// reports the real per-frame dispatch count.
pub const DISPATCHES_PER_SUBSTEP_BUNDLE: u32 = 5;

/// Fine nodes per coarse cell per axis (U8 coarse pressure pre-pass; mirrors `COARSE_FACTOR`
/// in coarse.wgsl). Coarse cell size H = COARSE_FACTOR·h; a full interior coarse cell holds
/// COARSE_FACTOR³ fine nodes.
pub const COARSE_FACTOR: u32 = 4;

/// Plain-Jacobi sweep count of the U8 coarse solve. EVEN by contract: the ping-pong ends with
/// the final potential back in `coarse_phi_a`, which is the buffer `coarse_apply` binds. The
/// active pool region is a handful of coarse cells across, so 16 sweeps propagate the
/// low-frequency correction wall-to-wall with margin (each sweep reaches one more cell).
pub const COARSE_SWEEPS: u32 = 16;

/// U8 dispatch increment when the coarse pre-pass is enabled (`pbmpm_coarse_strength > 0`):
/// clear + restrict + source + the sweeps + apply, once per substep.
pub fn coarse_dispatches() -> u32 {
    3 + COARSE_SWEEPS + 1
}

/// Seam bed BC thresholds (U2, docs/plans/2026-07-09-002): the minimum node solid fraction
/// that counts as "in the bed" (below it the BC is inert — open water), and the saturation
/// ratio treated as fully saturated → full entry block (mirrors the wetting contract's
/// `wet_sat_cutoff = V_cap·(1 − absorb_roundoff)` at the default roundoff 1e-3).
pub const SEAM_PHI_MIN: f32 = 0.15;
pub const SEAM_SAT_FULL: f32 = 0.999;
/// Fixed-point scale of the `seam_reaction` impulse lanes. Deliberately COARSER than
/// FP_SCALE (2^18): the ledger accumulates up to iteration_count node-mass × velocity
/// impulses per frame, which would overflow the 2^13 value ceiling at 2^18; at 2^12 the
/// worst case (16 × ~30 mass × 50 cap) sits ~20× under i32 range. Telemetry only — R2
/// gates on measured momentum deltas, and the U3 headroom probe gates this scale.
pub const SEAM_IMPULSE_SCALE: f32 = 4096.0;

/// The recorded dispatch formula (see `DISPATCHES_PER_SUBSTEP_BUNDLE` docs): `5 +
/// 5·iteration_count` per frame at one substep/frame (1 deform_clear + 3 snapshot passes +
/// the iteration bundles + 1 particle_integrate), plus the U8 coarse family when enabled.
/// The U6 dispatch-budget gate pins `profile().dispatches_per_frame` to exactly this.
pub fn dispatches_per_frame_for(iterations: u32, coarse_enabled: bool) -> u32 {
    let coarse = if coarse_enabled {
        coarse_dispatches()
    } else {
        0
    };
    5 + DISPATCHES_PER_SUBSTEP_BUNDLE * iterations.max(1) + coarse
}

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
    // .x = iteration_count (PB-MPM constraint→grid rebuild loop); .y = num_solids (count of static
    // SDF solids in the `solids` buffer, 0 = none → BC skipped); .z = restitution as f32 bits
    // (U5 collider normal-velocity reflection coefficient, read via `bitcast<f32>`); .w =
    // flip_fraction as f32 bits (SPLASH FLIP blend weight, read via `bitcast<f32>`).
    // Packed as one vec4 so the WGSL vec4<u32> alignment matches Rust's `[u32; 4]` exactly (a bare
    // u32 followed by a vec3 pad disagrees: WGSL aligns the vec3 to 16, Rust does not).
    iter_pad: [u32; 4],
    // U8 coarse pressure pre-pass (coarse.wgsl): dims (cx, cy, cz, num_ccells) and
    // (strength κ, interior rest mass per coarse cell, coarse cell size H, kick cap).
    coarse_dims: [u32; 4],
    coarse: [f32; 4],
    // Seam-blend bed BC (docs/plans/2026-07-09-002 U2): .x = enabled (1.0/0.0), .y = minimum
    // node solid fraction that counts as "in the bed", .z = saturation ratio treated as fully
    // saturated (full block; mirrors wet_sat_cutoff's 1 − absorb_roundoff), .w = reserved.
    seam: [f32; 4],
}

// Params is uploaded as a uniform and must stay byte-identical to the WGSL `Params`.
const _: () = assert!(std::mem::size_of::<Params>() == 176);

/// GPU record for one static SDF solid — byte-identical to the WGSL `Primitive` (64 bytes,
/// vec4-aligned; mirrors twofield's `Primitive` packing of `utils::sdf` primitives, U5). Cone
/// radii in `a` are OUTER wall radii; the cavity surface is `outer − thickness`. Each solid is a
/// CAVITY (interior allowed): `sample > 0` inside the free space, `< 0` through the wall material,
/// and the gradient points INTO the cavity — so the collider BC contains water inside the cup.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Primitive {
    kind: u32,         // 0 = cone, 1 = cylinder, 2 = poly-cup
    species_mask: u32, // MASK_* bits
    friction: f32,
    flags: u32,  // bit0 = apex_open
    a: [f32; 4], // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)  poly:(floor_y, rim_y, apothem, sides)
    b: [f32; 4], // cone:(thickness, hole_radius, center_x, center_z)  cyl/poly:(center_x, center_z, _, _)
    c: [f32; 4], // reserved
}

// Primitive is a storage-buffer element and must stay byte-identical to the WGSL `Primitive`.
const _: () = assert!(std::mem::size_of::<Primitive>() == 64);

/// Pack a scene's analytic solids into the GPU `Primitive` layout (mirrors twofield's `pack_solids`
/// and `utils/sdf.rs`). U5: the single-phase prototype only collides WATER, but the packed
/// `species_mask` is preserved so the WGSL union filters by species exactly as twofield does.
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
            SolidKind::PolyCup {
                center,
                floor_y,
                rim_y,
                apothem,
                sides,
            } => Primitive {
                kind: 2,
                species_mask: s.species_mask,
                friction: s.friction,
                flags: 0,
                a: [floor_y, rim_y, apothem, sides as f32],
                b: [center.x, center.z, 0.0, 0.0],
                c: [0.0; 4],
            },
        })
        .collect()
}

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
    emitted_mass: f32, // conservation accounting: emit_n × particle_mass (twofield convention)
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
    // SPLASH (FLIP) snapshot: each particle's substep-start velocity, copied from `vel` via
    // copy_buffer_to_buffer before the iteration loop. particle_integrate reads it to form the FLIP
    // blend. (The companion PRE-FORCE grid snapshot `grid_vel_old` is not stored on the struct — like
    // `solids`, the snapshot/integrate bind groups retain it; it is a per-substep transient.)
    vel_prev: wgpu::Buffer,
    // (Static SDF solids the water collides with (U5) are created in build() and retained by the
    // collider-pass bind groups — like twofield's `solids_buf`, not stored on the struct.)
    // Seam-blend bed coupling (U2): the bed-occupancy field the seam's scatter pass writes and
    // grid_update reads, and the reaction impulse ledger the bed BC accumulates. Real-sized only
    // when `pbmpm_seam_bed` is on; 1-element dummies otherwise (the WGSL branch is params-dead).
    bed_occupancy: Arc<wgpu::Buffer>,
    seam_reaction: Arc<wgpu::Buffer>,
    readback: wgpu::Buffer,

    pipelines: Pipelines,
    // Per-pass timestamp wiring (native profiling only; `None` on the web / no TIMESTAMP_QUERY).
    // Mirrors twofield: step() records per-pass timestamps, `sample_diagnostics()` decodes them
    // into `cached_passes`, and `profile()` returns the cache (never a GPU sync).
    ts: Option<Timestamps>,
    dispatches: u32,
    cached_passes: Vec<(String, f32)>,

    // Retained for reset (exact, deterministic re-seed).
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
    // Live water count at the seed (the count `reset()` returns to before any pour activates slots).
    initial_water: u32,
}

/// Per-pass timestamp query wiring (native profiling only). Mirrors twofield's `Timestamps`.
struct Timestamps {
    qset: wgpu::QuerySet,
    capacity: u32,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    labels: Vec<String>,
    count: u32,
}

struct Pipelines {
    deform_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    particle_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_decode_old: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p: (wgpu::ComputePipeline, wgpu::BindGroup),
    particle_integrate: (wgpu::ComputePipeline, wgpu::BindGroup),
    // U8 coarse pressure family (coarse.wgsl). coarse_jacobi carries the two ping-pong bind
    // groups ([0] reads phi_a → writes phi_b, [1] the swap); COARSE_SWEEPS is even so the final
    // potential lands in phi_a, the buffer coarse_apply binds.
    coarse_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    coarse_restrict: (wgpu::ComputePipeline, wgpu::BindGroup),
    coarse_source: (wgpu::ComputePipeline, wgpu::BindGroup),
    coarse_jacobi: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    coarse_apply: (wgpu::ComputePipeline, wgpu::BindGroup),
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

    /// Instantaneous bulk density of the pool INTERIOR from the fine grid mass field
    /// (dev/test only — stalls): mean of `node_mass / rest_node_mass − 1` over nodes carrying
    /// at least `interior_frac` of the rest loading (excludes surface/air nodes). This is the
    /// PHYSICAL volume-loss observable — unlike the per-particle `liquidDensity` memory, it
    /// carries no multiplicative random-walk noise (the memory is a ∏(tr(D)+1) integral whose
    /// MEAN inflates with accumulated variance even at rest; see the U8 drift-gate notes).
    /// Returns `(mean_error, interior_node_count)`.
    pub fn read_fine_interior_density(&self, interior_frac: f32) -> (f64, u32) {
        let bytes = (self.num_nodes as u64) * 16;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_fp, bytes)).to_vec();
        let spacing_ratio = self.params.grid_origin[3] / self.inflow.spacing;
        let rest_node = (self.params.particle_mass * spacing_ratio.powi(3)) as f64;
        let mut sum = 0.0f64;
        let mut count = 0u32;
        for node in raw.chunks_exact(4) {
            let mass = node[0] as f64 / FP_SCALE;
            if mass >= interior_frac as f64 * rest_node {
                sum += mass / rest_node - 1.0;
                count += 1;
            }
        }
        (if count > 0 { sum / count as f64 } else { 0.0 }, count)
    }

    /// Read back the decoded float grid velocity (.xyz) + node mass (.w) after `grid_update`
    /// (dev/test only — stalls). Valid until the next frame's `grid_update` overwrites it.
    pub fn read_grid_velocities(&self) -> Vec<[f32; 4]> {
        let bytes = (self.num_nodes as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.grid_vel, bytes)).to_vec()
    }

    /// Read back the per-particle deformation displacement `D` rows (3 vec4 rows per particle,
    /// dev/test only — stalls). The U6 float-range probe reads this: `D` is the float state
    /// surface that can blow up without ever touching the fixed-point grid lanes.
    pub fn read_deform_disp_rows(&self) -> Vec<[f32; 4]> {
        let bytes = (self.water_capacity as u64) * 48;
        bytemuck::cast_slice(&self.read_bytes(&self.deform_disp, bytes)).to_vec()
    }

    /// Read back the per-particle deformation gradient `F` rows (3 vec4 rows per particle,
    /// dev/test only — stalls). Row 0's `.x` lane carries the per-particle `liquidDensity`
    /// (the running ∏(tr(D)+1) volume product, `constraint.wgsl`) — the U6 bulk-density drift
    /// measurement reads it directly.
    pub fn read_deform_grad_rows(&self) -> Vec<[f32; 4]> {
        let bytes = (self.water_capacity as u64) * 48;
        bytemuck::cast_slice(&self.read_bytes(&self.deform_grad, bytes)).to_vec()
    }

    /// Sample per-pass GPU timestamps into the cache that `profile()` returns. Blocks
    /// (dev/test/periodic only) — the explicit cache point, so the getters never stall.
    /// Mirrors twofield's `sample_diagnostics` so perf harnesses drive both identically.
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

    /// Set the PB-MPM iteration count (the `particle_update → grid_zero → p2g → grid_update → g2p`
    /// bundle repeats this many times per substep) — dev/test only (the bounce/stability sweep and
    /// the pure-transfer round-trip arm, which sets it to 1). Clamped to ≥ 1.
    pub fn set_iteration_count_for_test(&mut self, count: u32) {
        self.params.iter_pad[0] = count.max(1);
    }

    /// Set the INIT rest liquid density (U6: the constraint reads the PER-PARTICLE accumulated
    /// density from the `deform_grad[3p+0].x` lane, not this Params lane — the lane is seeded to 1.0
    /// by the identity F seed at build/reset/emit). Kept for the Params ABI + future re-seed hooks;
    /// changing it alone no longer changes the running constraint. Dev/test only.
    pub fn set_liquid_density_for_test(&mut self, density: f32) {
        self.params.liquid_density = density;
    }

    /// Set the compliant volume-correction relaxation `∈ [0,1]` — dev/test only (the stiffness ⇒
    /// bounce sweep, and the `0.0` disabled arm that makes the iteration loop a pure transfer
    /// check: alpha is still computed but the correction is scaled to zero).
    pub fn set_liquid_relaxation_for_test(&mut self, relaxation: f32) {
        self.params.liquid_relaxation = relaxation;
    }

    /// Set the viscous (deviatoric/shear) correction weight — dev/test only. `0.0` disables the
    /// shear term.
    pub fn set_liquid_viscosity_for_test(&mut self, viscosity: f32) {
        self.params.liquid_viscosity = viscosity;
    }

    /// Set the collider normal-velocity restitution (U5) — dev/test only. `0.0` = free-slip stop
    /// (the plan's constraint-only arm: the into-solid normal velocity is killed, no rebound);
    /// `>0` reflects `v_n_out = −restitution·v_n_in` on penetration (a bouncier floor/wall).
    /// Clamped to `[0, 1]`. Stored as f32 bits in `iter_pad.z` to keep the 128-byte Params ABI.
    pub fn set_restitution_for_test(&mut self, restitution: f32) {
        self.params.iter_pad[2] = restitution.clamp(0.0, 1.0).to_bits();
    }

    /// Set the SPLASH FLIP fraction (the per-substep output-velocity blend `v = mix(v_pic, v_flip,
    /// flip_fraction)`) — dev/test only. `0.0` = pure APIC/PIC (the byte-identical off-switch the
    /// round-trip transfer check uses); `1.0` = full FLIP (maximally preserves the impact-generated
    /// grid-velocity change). Clamped to `[0, 1]`. Stored as f32 bits in `iter_pad.w`.
    pub fn set_flip_fraction_for_test(&mut self, flip_fraction: f32) {
        self.params.iter_pad[3] = flip_fraction.clamp(0.0, 1.0).to_bits();
    }

    /// Set the U8 coarse pre-pass strength κ — dev/test only. `0.0` disables the pass entirely
    /// (its dispatches are skipped; the byte-identical off-switch the A/B tests key on).
    pub fn set_coarse_strength_for_test(&mut self, strength: f32) {
        self.params.coarse[0] = strength.max(0.0);
    }

    /// Live water count currently simulated.
    /// The seam-blend coupling buffers `(bed_occupancy, seam_reaction)` — 4 fixed-point
    /// lanes per node each when `pbmpm_seam_bed` is on, 1-element dummies otherwise. The
    /// seam's scatter pass writes bed_occupancy; its hook consumes/zeroes seam_reaction.
    pub fn seam_buffers(&self) -> (Arc<wgpu::Buffer>, Arc<wgpu::Buffer>) {
        (
            Arc::clone(&self.bed_occupancy),
            Arc::clone(&self.seam_reaction),
        )
    }

    /// Total water mass emitted by the pour so far (conservation accounting; the
    /// twofield/xpbd convention — emit_n × particle_mass per emission).
    pub fn total_emitted_water_mass(&self) -> f32 {
        self.inflow.emitted_mass
    }

    /// Remove marked live water particles (the seam's absorption handoff, docs/plans/
    /// 2026-07-09-002 U4): swap-with-last over the FULL live per-particle state set —
    /// pos, vel, phase, chem, deform_disp (3 rows), deform_grad (3 rows), vel_prev — a
    /// partial swap silently corrupts the survivor (review r1.2). The swap plan is built
    /// host-side (marks deduped, sorted descending, tail-collision-safe) and executed as
    /// GPU buffer copies; the live count then drops. Freed tail slots keep stale data —
    /// every kernel guards `p >= water_count`, and emit() overwrites on reuse.
    pub fn remove_water_for_seam(&mut self, marked: &[u32]) {
        let mut marks: Vec<u32> = marked
            .iter()
            .copied()
            .filter(|&m| m < self.water_count)
            .collect();
        marks.sort_unstable();
        marks.dedup();
        if marks.is_empty() {
            return;
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbmpm-seam-remove"),
            });
        // WebGPU forbids same-buffer copies: bounce each lane through a scratch buffer
        // (copies within one encoder execute in submission order, so serial reuse is safe).
        let scratch = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbmpm-seam-remove-scratch"),
            size: 48,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut new_count = self.water_count;
        // Descending: each removal's swap source is the current last live slot, which can
        // never itself be a not-yet-processed mark (those are all smaller indices).
        for &m in marks.iter().rev() {
            new_count -= 1;
            if m != new_count {
                let (dst, src) = (m as u64, new_count as u64);
                let lanes: [(&wgpu::Buffer, u64); 7] = [
                    (&self.pos, 16),
                    (&self.vel, 16),
                    (&self.chem, 16),
                    (&self.vel_prev, 16),
                    (&self.phase, 4),
                    (&self.deform_disp, 48),
                    (&self.deform_grad, 48),
                ];
                for (buf, stride) in lanes {
                    enc.copy_buffer_to_buffer(buf, src * stride, &scratch, 0, stride);
                    enc.copy_buffer_to_buffer(&scratch, 0, buf, dst * stride, stride);
                }
            }
        }
        self.queue.submit(Some(enc.finish()));
        self.water_count = new_count;
        self.params.water_count = new_count;
    }

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
        // activation only overwrites pos/vel/chem. Re-seed identity F for the activated slots so the
        // U6 per-particle liquidDensity lane (deform_grad[3p+0].x) is exactly 1.0 at activation (the
        // identity seed already puts 1.0 there, but re-writing it is defensive against a reused slot
        // carrying stale accumulation; 3 contiguous vec4 rows per particle = one strided write).
        let off_f = (self.water_count as u64) * 48;
        self.queue.write_buffer(
            &self.deform_grad,
            off_f,
            bytemuck::cast_slice(&identity_rows(emit_n)),
        );
        self.inflow.emitted_mass += emit_n as f32 * self.params.particle_mass;
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
        // U8 coarse grid: COARSE_FACTOR fine nodes per coarse cell per axis, rounded up so the
        // last partial cells still cover the node grid.
        let cdims = [
            dims[0].div_ceil(COARSE_FACTOR),
            dims[1].div_ceil(COARSE_FACTOR),
            dims[2].div_ceil(COARSE_FACTOR),
        ];
        let num_ccells = cdims[0] * cdims[1] * cdims[2];

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
            // PB-MPM liquid constraint knobs from Config (U4); tunable live via the setters below.
            liquid_density: cfg.pbmpm_liquid_density,
            liquid_relaxation: cfg.pbmpm_liquid_relaxation,
            liquid_viscosity: cfg.pbmpm_liquid_viscosity,
            iter_pad: [
                cfg.pbmpm_iteration_count.max(1),
                scene.solids.len() as u32,
                cfg.pbmpm_restitution.to_bits(),
                cfg.pbmpm_flip_fraction.clamp(0.0, 1.0).to_bits(),
            ],
            coarse_dims: [cdims[0], cdims[1], cdims[2], num_ccells],
            coarse: [
                cfg.pbmpm_coarse_strength.max(0.0),
                // Interior rest mass per FULL coarse cell: rest node mass m·(h/spacing)³ times
                // COARSE_FACTOR³ nodes. The 0.5× surface classifier in coarse_source keys on it.
                mats.particle_mass
                    * (cell / mats.particle_spacing).powi(3)
                    * (COARSE_FACTOR.pow(3) as f32),
                cell * COARSE_FACTOR as f32,
                cfg.pbmpm_coarse_kick_cap.max(0.0),
            ],
            seam: [
                if cfg.pbmpm_seam_bed { 1.0 } else { 0.0 },
                SEAM_PHI_MIN,
                SEAM_SAT_FULL,
                0.0,
            ],
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
        // Seam-blend bed coupling lanes (U2): 4 fixed-point lanes per node each —
        // bed_occupancy [V_eff, V_abs, V_cap, unused] (seam-scattered, read here);
        // seam_reaction [impulse.xyz, unused] (accumulated here, consumed by the seam).
        let seam_nodes = if cfg.pbmpm_seam_bed {
            num_nodes as u64
        } else {
            1
        };
        let bed_occupancy = Arc::new(Self::storage(
            &device,
            "pbmpm-seam-bed-occupancy",
            seam_nodes * 16,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let seam_reaction = Arc::new(Self::storage(
            &device,
            "pbmpm-seam-reaction",
            seam_nodes * 16,
            wgpu::BufferUsages::COPY_SRC,
        ));
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
        // SPLASH (FLIP) snapshots: the PRE-FORCE pure-transfer grid velocity (one vec4 per node) and
        // each particle's substep-start velocity (one vec4 per particle). vel_prev is a
        // copy_buffer_to_buffer destination each substep (the `storage()` helper grants COPY_DST).
        let grid_vel_old = Self::storage(
            &device,
            "pbmpm-grid-vel-old",
            (num_nodes.max(1) as u64) * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        let vel_prev = Self::storage(
            &device,
            "pbmpm-vel-prev",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        );
        // U8 coarse pressure state: fixed-point mass, the compacted active-cell list
        // ([0] = count + capacity entries — capacity == num_ccells so the atomic append can
        // never overflow), the (rhs, kind) source, and the Jacobi ping-pong potentials.
        let ncc = num_ccells.max(1) as u64;
        let coarse_fp = Self::storage(
            &device,
            "pbmpm-coarse-fp",
            ncc * 4,
            wgpu::BufferUsages::empty(),
        );
        let coarse_list = Self::storage(
            &device,
            "pbmpm-coarse-list",
            (1 + ncc) * 4,
            wgpu::BufferUsages::empty(),
        );
        let coarse_src = Self::storage(
            &device,
            "pbmpm-coarse-src",
            ncc * 8,
            wgpu::BufferUsages::empty(),
        );
        let coarse_phi_a = Self::storage(
            &device,
            "pbmpm-coarse-phi-a",
            ncc * 4,
            wgpu::BufferUsages::empty(),
        );
        let coarse_phi_b = Self::storage(
            &device,
            "pbmpm-coarse-phi-b",
            ncc * 4,
            wgpu::BufferUsages::empty(),
        );
        // Static SDF solids the water collides with (U5). An empty scene still needs a non-empty
        // binding, so push one zeroed primitive; `num_solids = 0` (in Params) makes every BC loop
        // skip it. The packed `Primitive`s mirror twofield/utils::sdf (interior-positive cavities).
        let packed = {
            let mut packed = pack_solids(&scene.solids);
            if packed.is_empty() {
                packed.push(Primitive::zeroed());
            }
            packed
        };
        let solids = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbmpm-solids"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pbmpm-readback"),
            // Must cover the widest dev/test read: the 3-vec4-row deform buffers (48 B/particle),
            // not just the vec4 particle lanes or the grid.
            size: (vec4 * 3).max(grid_fp.size()),
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
        // the shared bindings/helpers, transfers.wgsl adds the U3 p2g/grid_update/g2p passes, and
        // constraint.wgsl adds the U4 particle_update (compliant density constraint) + the moved-out
        // particle_integrate (advect).
        let shader_src = format!(
            "{}\n{}\n{}\n{}",
            include_str!("common.wgsl"),
            include_str!("transfers.wgsl"),
            include_str!("constraint.wgsl"),
            include_str!("coarse.wgsl"),
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
            // deform_clear: deform_disp (write). 1 storage buffer. Zeroes D once per substep.
            let deform_clear = make("deform_clear");
            let deform_clear_bind = bg(&deform_clear, &[(0, &params_buf), (5, &deform_disp)]);
            // particle_update: deform_disp (read+write D), deform_grad (READ the per-particle
            // liquidDensity lane [3p+0].x). 2 storage buffers. Runs BEFORE p2g each iteration so the
            // corrected D propagates through the scatter.
            let particle_update = make("particle_update");
            let particle_update_bind = bg(
                &particle_update,
                &[(0, &params_buf), (5, &deform_disp), (6, &deform_grad)],
            );
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
            // grid_decode_old: grid_fp (decode), grid_vel_old (write). 2 storage buffers. Runs ONCE
            // per substep before the iteration loop to snapshot the PRE-FORCE pure-transfer velocity
            // (no gravity/BC) for the FLIP blend.
            let grid_decode_old = make("grid_decode_old");
            let grid_decode_old_bind = bg(
                &grid_decode_old,
                &[(0, &params_buf), (7, &grid_fp), (10, &grid_vel_old)],
            );
            // grid_update: grid_fp (decode), grid_vel (write), solids (read SDF BC), plus the
            // seam bed lanes (U2 — dummies when the seam BC is off). 5 storage buffers.
            let grid_update = make("grid_update");
            let grid_update_bind = bg(
                &grid_update,
                &[
                    (0, &params_buf),
                    (7, &grid_fp),
                    (8, &grid_vel),
                    (9, &solids),
                    (17, &bed_occupancy),
                    (18, &seam_reaction),
                ],
            );
            // g2p: pos (read), vel (write), deform_disp (write D), grid_vel (gather). 4 storage
            // buffers. No longer writes pos — advection moved to particle_integrate (U4).
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
            // particle_integrate: pos (write), vel (read+wall clamp), deform_disp (READ the converged
            // D for the liquidDensity accumulation), deform_grad (READ-WRITE the per-particle
            // liquidDensity lane), grid_vel_old (gather the PRE-FORCE velocity for the FLIP blend),
            // vel_prev (the substep-start velocity), solids (SDF push-out + restitution). 7 storage
            // buffers. Runs ONCE per substep after the iteration loop.
            let particle_integrate = make("particle_integrate");
            let particle_integrate_bind = bg(
                &particle_integrate,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (5, &deform_disp),
                    (6, &deform_grad),
                    (9, &solids),
                    (10, &grid_vel_old),
                    (11, &vel_prev),
                    (17, &bed_occupancy),
                ],
            );
            // U8 coarse pressure family. coarse_jacobi ping-pongs the phi buffers at the
            // BIND-GROUP level: variant [0] reads binding 15 = phi_a / writes 16 = phi_b,
            // variant [1] the swap. COARSE_SWEEPS is even, so the final potential lands in the
            // real phi_a buffer — which is what coarse_apply binds at 15.
            let coarse_clear = make("coarse_clear");
            let coarse_clear_bind = bg(
                &coarse_clear,
                &[
                    (0, &params_buf),
                    (12, &coarse_fp),
                    (13, &coarse_list),
                    (14, &coarse_src),
                    (15, &coarse_phi_a),
                    (16, &coarse_phi_b),
                ],
            );
            let coarse_restrict = make("coarse_restrict");
            let coarse_restrict_bind = bg(
                &coarse_restrict,
                &[
                    (0, &params_buf),
                    (7, &grid_fp),
                    (12, &coarse_fp),
                    (13, &coarse_list),
                ],
            );
            let coarse_source = make("coarse_source");
            let coarse_source_bind = bg(
                &coarse_source,
                &[
                    (0, &params_buf),
                    (12, &coarse_fp),
                    (13, &coarse_list),
                    (14, &coarse_src),
                ],
            );
            let coarse_jacobi = make("coarse_jacobi");
            let cj = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
                bg(
                    &coarse_jacobi,
                    &[
                        (0, &params_buf),
                        (13, &coarse_list),
                        (14, &coarse_src),
                        (15, src),
                        (16, dst),
                    ],
                )
            };
            let coarse_jacobi_binds = [
                cj(&coarse_phi_a, &coarse_phi_b),
                cj(&coarse_phi_b, &coarse_phi_a),
            ];
            let coarse_apply = make("coarse_apply");
            let coarse_apply_bind = bg(
                &coarse_apply,
                &[(0, &params_buf), (1, &pos), (2, &vel), (15, &coarse_phi_a)],
            );
            Pipelines {
                deform_clear: (deform_clear, deform_clear_bind),
                particle_update: (particle_update, particle_update_bind),
                grid_clear: (grid_clear, grid_clear_bind),
                p2g: (p2g, p2g_bind),
                grid_decode_old: (grid_decode_old, grid_decode_old_bind),
                grid_update: (grid_update, grid_update_bind),
                g2p: (g2p, g2p_bind),
                particle_integrate: (particle_integrate, particle_integrate_bind),
                coarse_clear: (coarse_clear, coarse_clear_bind),
                coarse_restrict: (coarse_restrict, coarse_restrict_bind),
                coarse_source: (coarse_source, coarse_source_bind),
                coarse_jacobi: (coarse_jacobi, coarse_jacobi_binds),
                coarse_apply: (coarse_apply, coarse_apply_bind),
            }
        };

        let ts = if gpu.timestamps_supported {
            // 5 + 5·iteration_count dispatches per frame (2 queries each); 384 queries cover
            // iteration counts up to 37 — headroom over the frozen default (16) and the sweeps.
            let capacity = 384u32;
            let qset = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pbmpm-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbmpm-ts-resolve"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let ts_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pbmpm-ts-readback"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Some(Timestamps {
                qset,
                capacity,
                resolve,
                readback: ts_readback,
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
                emitted_mass: 0.0,
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
            vel_prev,
            bed_occupancy,
            seam_reaction,
            readback,
            pipelines,
            ts,
            dispatches: 0,
            cached_passes: Vec::new(),
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
        self.inflow.emitted_mass = 0.0;
    }

    fn step(&mut self, dt: f32, input: &EmissionInput) {
        // Pour emission first (grows water_count for this frame); no-op when not pouring. Newly
        // activated slots are written into the live range so the transfer dispatches below (guarded
        // on water_count) pick them up.
        self.emit(input, dt);
        debug_assert!(self.water_count <= self.water_capacity);

        // One substep per frame for now; the iteration_count loop runs WITHIN the substep (below).
        // A CFL substep policy is a later decision (twofield also runs one substep in its U2 form).
        self.params.dt = dt;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbmpm-frame"),
            });
        // SPLASH (FLIP) snapshot (a): copy the substep-start particle velocity into vel_prev BEFORE
        // the iteration loop overwrites `vel`. An encoder copy (no shader — cheapest, deterministic);
        // must sit OUTSIDE the compute pass.
        if self.water_count > 0 {
            let bytes = (self.water_count as u64) * 16;
            enc.copy_buffer_to_buffer(&self.vel, 0, &self.vel_prev, 0, bytes);
        }
        // PB-MPM iteration structure (U4 + SPLASH snapshot; one substep per frame for now):
        //   copy vel → vel_prev                    // (encoder copy above) substep-start velocity
        //   deform_clear                          // zero D once at the substep's iteration loop start
        //   grid_clear ; p2g ; grid_decode_old     // FLIP snapshot (b): PRE-FORCE pure-transfer grid
        //                                          //   velocity (no gravity/BC) → grid_vel_old
        //   repeat iteration_count {
        //     particle_update                     // compliant density constraint (writes D)
        //     grid_clear                          // clear the fixed-point grid lanes
        //     p2g                                 // scatter m, m·(v + D·d) (fixed-point atomicAdd)
        //     grid_update                         // decode, gravity, domain BC → grid_vel
        //     g2p                                 // gather velocity + reconstruct D (NO advect)
        //   }
        //   particle_integrate                    // FLIP blend + advect ONCE on the converged velocity
        // deform_clear runs BEFORE the snapshot p2g so D = 0 there: the snapshot scatters PURE `vel`
        // (no affine/constraint term), making grid_vel_old the pure-transfer velocity exactly. The
        // snapshot perturbs nothing else — its grid_fp is overwritten by the loop's first grid_clear,
        // and D is already zero for the loop's first particle_update. FLIP is applied ONCE in
        // particle_integrate so the loop's g2p stays pure-PIC and the incompressibility solve is
        // unaffected. D is zeroed per substep and accumulated across the iterations; F carries across
        // substeps. Each pass dispatches at least one workgroup so the dispatch/profiling path is real
        // even on an empty scene (threads early-out on the live-set guards — over-dispatch +
        // early-out, never indirect dispatch). Fixed-point P2G is order-independent and the host-side
        // loop count is uniform, so the structure is deterministic (Tint-safe: no in-shader
        // barriers/loops over the iterations).
        //
        // Dispatches go through the shared PassRecorder: with timestamps (native profiling) each
        // dispatch gets its own timestamped pass (all of them — one substep per frame, so the
        // whole frame IS the substep the perf gate sums); without timestamps the frame collapses
        // into one batched compute pass (the web path, identical work).
        let node_groups = groups(self.num_nodes).max(1);
        let water_groups = groups(self.water_count).max(1);
        let iterations = self.params.iter_pad[0].max(1);
        let sink = self.ts.as_ref().map(|t| TimestampSink {
            qset: &t.qset,
            capacity: t.capacity,
        });
        let mut rec = PassRecorder::new(sink);
        let p = &self.pipelines;
        rec.dispatch(
            &mut enc,
            &p.deform_clear.0,
            &p.deform_clear.1,
            "deform_clear",
            water_groups,
        );
        // FLIP snapshot (b): clear the grid, scatter the substep-start velocity (D = 0 now), decode
        // it to the PRE-FORCE pure-transfer velocity (no gravity/BC) into grid_vel_old.
        rec.dispatch(
            &mut enc,
            &p.grid_clear.0,
            &p.grid_clear.1,
            "grid_clear",
            node_groups,
        );
        rec.dispatch(&mut enc, &p.p2g.0, &p.p2g.1, "p2g", water_groups);
        rec.dispatch(
            &mut enc,
            &p.grid_decode_old.0,
            &p.grid_decode_old.1,
            "grid_decode_old",
            node_groups,
        );
        // U8 coarse pressure pre-pass (once per substep, before the constraint loop): restrict
        // the snapshot mass to the coarse grid + compacted active list, solve the low-frequency
        // Poisson over the list, kick the particle velocities with +∇φ. Skipped entirely at
        // strength 0 — the byte-identical off-switch (no dispatches, no cost).
        if self.params.coarse[0] > 0.0 {
            let ccell_groups = groups(self.params.coarse_dims[3]).max(1);
            rec.dispatch(
                &mut enc,
                &p.coarse_clear.0,
                &p.coarse_clear.1,
                "coarse_clear",
                ccell_groups,
            );
            rec.dispatch(
                &mut enc,
                &p.coarse_restrict.0,
                &p.coarse_restrict.1,
                "coarse_restrict",
                node_groups,
            );
            rec.dispatch(
                &mut enc,
                &p.coarse_source.0,
                &p.coarse_source.1,
                "coarse_source",
                ccell_groups,
            );
            for s in 0..COARSE_SWEEPS {
                rec.dispatch(
                    &mut enc,
                    &p.coarse_jacobi.0,
                    &p.coarse_jacobi.1[(s % 2) as usize],
                    "coarse_jacobi",
                    ccell_groups,
                );
            }
            rec.dispatch(
                &mut enc,
                &p.coarse_apply.0,
                &p.coarse_apply.1,
                "coarse_apply",
                water_groups,
            );
        }
        for _ in 0..iterations {
            rec.dispatch(
                &mut enc,
                &p.particle_update.0,
                &p.particle_update.1,
                "particle_update",
                water_groups,
            );
            rec.dispatch(
                &mut enc,
                &p.grid_clear.0,
                &p.grid_clear.1,
                "grid_clear",
                node_groups,
            );
            rec.dispatch(&mut enc, &p.p2g.0, &p.p2g.1, "p2g", water_groups);
            rec.dispatch(
                &mut enc,
                &p.grid_update.0,
                &p.grid_update.1,
                "grid_update",
                node_groups,
            );
            rec.dispatch(&mut enc, &p.g2p.0, &p.g2p.1, "g2p", water_groups);
        }
        rec.dispatch(
            &mut enc,
            &p.particle_integrate.0,
            &p.particle_integrate.1,
            "particle_integrate",
            water_groups,
        );
        let (dispatches, cursor, labels) = rec.finish();
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
        Profile {
            passes: self.cached_passes.clone(),
            dispatches_per_frame: self.dispatches,
        }
    }
}

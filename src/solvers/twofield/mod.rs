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
//! substep (the CFL substep policy is a later-unit decision).
//!
//! U3 adds incompressibility (`pressure.wgsl`, KTD-2): a coarse-grid pressure seed plus a
//! fixed budget of fine damped-Jacobi sweeps on ONE discretely consistent operator family
//! (pressure at cell centers, corner-trilinear D, G = −Dᵀ by construction, A only ever the
//! composition −D·M̃⁻¹·G) — no converged Poisson anywhere (R7). The full operator-family and
//! coarse-seed construction is documented at the top of `pressure.wgsl`; the pre-registered
//! gates live in `tests/twofield_pressure.rs`. The KTD-2 adaptive-Tait predictor is a STUB
//! DECISION: deliberately not implemented (knob grid pins Tait ∈ {off}); if a later unit
//! demonstrates an iteration-budget win it ships behind an opt-in `Config` gate, otherwise it
//! is removed rather than shipped dormant.
//!
//! U4 (the L0 exit) adds pour emission and the pour-cavity machinery (`surface.wgsl`,
//! KTD-6): `EmissionInput` drives the xpbd-shaped volume-consistent arclength-credit emitter
//! into a pre-allocated water pool (water live range `[0, water_count)` grows toward
//! `water_capacity`; solids sit AFTER the full pool so the KTD-1 range layout survives
//! emission — dormant slots are never dispatched); enclosed air pockets are detected by a
//! fixed-budget grid flood fill from the open (top) boundary and carried as ONE constraint
//! bubble: a single Lagrange-multiplier scalar λ_b whose value IS the pocket pressure,
//! represented identically in the coarse solve and the fine sweeps (see surface.wgsl's
//! BUBBLE REPRESENTATION header). Gates: `tests/twofield_cavity.rs`.

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

/// U2+U3+U4 GPU-budget bookkeeping (R8): the widest entry point's storage-buffer count,
/// derived by inspection of the bind groups in `build` (the params uniform doesn't count
/// against the storage limit). Per pass: `grid_clear` 2 (grid_fp, cell_cnt), `p2g_water` 5
/// (pos, vel, cmat, grid_fp, cell_cnt), `grid_update` 3 (grid_fp, grid_vel, solids),
/// `g2p_water` 5 (pos, vel, cmat, grid_vel, solids); U3 pressure family — `node_setup` 3
/// (grid_vel, solids, nm),
/// `cell_classify` 6 (grid_vel, solids, cell_meta, pf_a, pf_b, cell_cnt), `coarse_node_setup` 2,
/// `coarse_cell_setup` 4 (cell_meta, cmeta, pc_a, pc_b), `jacobi_fine`/`jacobi_coarse` 5
/// (+bubble), `prolong_add` 6 (cell_meta, cmeta, pc, pf×2, bubble) — the widest, `project` 4,
/// debug taps ≤ 3; U4 surface family — `flood_init` 2, `flood_sweep` 3, `pocket_mark` 5
/// (grid_vel, cell_meta, pf×2, bubble), `bubble_fine` 4, `bubble_coarse` 4. Re-derive when
/// passes are added. The device requests 9 storage buffers per stage
/// (`src/utils/gpu.rs::NEEDED_STORAGE_BUFFERS`) — still NOT raised (KTD-7).
pub const MAX_STORAGE_BUFFERS_PER_ENTRY_POINT: u32 = 6;

/// U3 pressure knobs — defaults of the declared knob grid (KTD-9; the grid itself is pinned
/// in the header of `tests/twofield_pressure.rs`). The coarse ratio is fine cells per coarse
/// cell per axis (4 → 1/4³ cell count); sweep counts are dispatches, so they are plain Rust
/// fields driven from `step()` (a Jacobi sweep cannot loop inside one dispatch — WGSL has no
/// global sync).
pub const COARSE_RATIO_DEFAULT: u32 = 4;
pub const COARSE_SWEEPS_DEFAULT: u32 = 8;
pub const FINE_SWEEPS_DEFAULT: u32 = 8;

/// Damped-Jacobi relaxation factor — a fixed algorithmic constant of the family (mirrors
/// `JACOBI_OMEGA` in `pressure.wgsl`; the CPU twins import it), NOT a gate knob. Derivation:
/// the Fourier symbol of A = −D·M⁻¹·G on the corner family gives
/// λ(diag⁻¹A)(θ) = (8/3)·[sin²(θx/2)cos²(θy/2)cos²(θz/2) + cyc], max 8/3 at θ = (π,0,0), so
/// Jacobi requires ω < 2/(8/3) = 0.75; ω = 2/3 is the classic weighted-Jacobi choice and
/// keeps ω·λ_max = 16/9 < 2 with healthy high-frequency smoothing.
pub const JACOBI_OMEGA: f32 = 2.0 / 3.0;

/// Density-relief time constant in frames (mirrors `DENSITY_RELAX_FRAMES` in pressure.wgsl).
/// A fixed structural constant like `JACOBI_OMEGA`: it closes the volume-conservation loop
/// (velocity-only projection cannot see accumulated positional compression), it is not a
/// gate-tuning knob. Half a second (30 frames): fast enough that equilibrium compaction
/// drift stays ≈ residual·N·dt ≈ 1% (inside the ±5% band), slow enough that relieving a
/// seeded over-density (the lattice double-counts the wall layers by ~30%) injects v ~
/// Δx/τ ≈ 0.6 instead of ~4.5 of slosh — the relief must correct volume, not detonate it.
pub const DENSITY_RELAX_FRAMES: f32 = 30.0;

/// Free-surface fill-fraction constants (mirror `SURF_FULL_FRAC`/`SURF_MIN_CORNER` in
/// pressure.wgsl — the ghost-fluid-style fraction weighting documented in its FREE SURFACE
/// header; refined, not replaced, by U4). Fixed structural constants of the operator family,
/// not gate knobs: a cell's constraint row is weighted by f = clamp(ρ̄/(0.5·ρ_rest), 0, 1),
/// so interior rows are exactly f = 1 (a flat surface cuts a cell at ρ̄ ≈ 0.5·ρ_rest — the
/// taper lives strictly above the mean surface line) and the implicit p = 0 Dirichlet acts
/// at the surface instead of one cell inside it. ACTIVITY is the min corner-node density ≥
/// `SURF_MIN_CORNER`·ρ_rest — the particle-presence discriminator that excludes B-spline
/// smear cells (particle-free, far corner ≤ ~0.05·ρ_rest), whose pressure rows otherwise
/// hover their dust against gravity and block settling (see the pressure.wgsl FREE SURFACE
/// header for the observed failure).
pub const SURF_FULL_FRAC: f32 = 0.5;
pub const SURF_MIN_CORNER: f32 = 0.1;

/// U3 dispatch increment over U2's 4, at the default knobs: node_setup + cell_classify +
/// pre-smooth/post-smooth FINE_SWEEPS + residual + coarse_node_setup +
/// coarse_cell_setup(+restrict) + COARSE_SWEEPS + prolong_add + project. Each Jacobi sweep is
/// necessarily its own dispatch (no global sync within a dispatch), which is why this lands
/// above the plan's rough +6–12 guess — recorded honestly per R8.
pub const U3_PRESSURE_DISPATCHES: u32 = 7 + COARSE_SWEEPS_DEFAULT + FINE_SWEEPS_DEFAULT;

/// U4 flood-fill sweep budget (fixed structural constant, like JACOBI_OMEGA — NOT a gate
/// knob): each sweep propagates the OUTSIDE label one 6-neighbor cell, so the budget bounds
/// the reachable open-air path length in cells. 24 covers every current scene with margin
/// (tallest air column: the 32-unit tank → 16 cells; the V60 cone detour ≈ 20); cells beyond
/// the budget would degrade conservatively (extra pocket members with massless nodes — zero
/// row coupling). Kept EVEN so the final labels land back in the pf_a slot (parity).
pub const FLOOD_SWEEPS: u32 = 24;

/// U4 dispatch increment at the default knobs: flood_init + FLOOD_SWEEPS + pocket_mark, plus
/// one single-workgroup bubble-row solve preceding EVERY fine and coarse Jacobi sweep (the
/// KTD-6 identical-representation requirement — the multiplier relaxes WITH the smoother at
/// both levels, so neither level can erode the constraint).
pub const U4_SURFACE_DISPATCHES: u32 =
    2 + FLOOD_SWEEPS + FINE_SWEEPS_DEFAULT + COARSE_SWEEPS_DEFAULT;

/// Compute dispatches per frame at the default knobs: the U2 transfer pipeline (grid_clear,
/// p2g_water, grid_update, g2p_water) + the U3 pressure stack + the U4 surface stack.
pub const DISPATCHES_PER_FRAME: u32 = 4 + U3_PRESSURE_DISPATCHES + U4_SURFACE_DISPATCHES;

/// Reduced-units volume calibration (KEEP.md §2; mirrors the xpbd constant): mL per scene
/// unit³ — sizes the pour pool from a scene's declared `pour_water_ml`.
const ML_PER_SIM_UNIT3: f32 = 5.20;

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
    coarse_dims: [u32; 4], // coarse CELLS per axis (= ceil(fine_cells/ratio)); .w = ratio
    extra: [f32; 4],     // (rest_density, rho_floor, mass_eps, unused)
}

// Params is uploaded as a uniform and must stay byte-identical to the WGSL `Params`.
const _: () = assert!(std::mem::size_of::<Params>() == 144);

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
    /// LIVE water count (grows as the pour activates pool slots); the seed value is
    /// `initial_water`.
    water_count: u32,
    /// Allocated water-pool size = seed + dose headroom (KTD-1 layout with emission: water
    /// pool `[0, water_capacity)`, live `[0, water_count)`, solids
    /// `[water_capacity, water_capacity + solid_count)` — dormant slots are never dispatched).
    water_capacity: u32,
    solid_count: u32,
    num_nodes: u32,
    num_cells: u32,
    /// Pour-emission state (xpbd-shaped volume-consistent emitter).
    inflow: Inflow,

    // U3 pressure knobs (KTD-9 grid points; sweep counts = dispatch counts, driven in step).
    coarse_ratio: u32,
    coarse_sweeps: u32,
    fine_sweeps: u32,

    params_buf: wgpu::Buffer,
    // Canonical particle state, exposed through `ParticleBuffers`: pos.w carries the moisture
    // lane (water = remaining fraction, grain = absorbed volume), chem = (c, T, _, _).
    pos: Arc<wgpu::Buffer>,
    vel: Arc<wgpu::Buffer>,
    phase: Arc<wgpu::Buffer>,
    chem: Arc<wgpu::Buffer>,
    // Per-particle APIC affine matrix C: 3 vec4 rows per particle (see common.wgsl binding 5).
    cmat: wgpu::Buffer,
    // WATER grid field: fixed-point atomic<i32>, 4 lanes per node (mass, mom.xyz).
    grid_fp: wgpu::Buffer,
    // Float grid velocity (.xyz) + node mass (.w) after grid_update; post-projection after
    // the U3 pressure stack. CPU-readable for the divergence/volume gates.
    grid_vel: wgpu::Buffer,
    // U3 pressure-family state (layouts documented in pressure.wgsl): per-node M̃⁻¹, per-cell
    // (rhs, active, dbg) meta, and the fine pressure ping-pong pair. The coarse mirrors live
    // only inside the bind groups.
    nm: wgpu::Buffer,
    cell_meta: wgpu::Buffer,
    pf_a: wgpu::Buffer,
    pf_b: wgpu::Buffer,
    // U4 bubble state: [λ_b, δλ_b] (surface.wgsl binding 17).
    bubble: wgpu::Buffer,
    readback: wgpu::Buffer,

    pipelines: Pipelines,
    ts: Option<Timestamps>,

    // Cached per-frame results returned by the getters (never a GPU sync there).
    dispatches: u32,
    cached_passes: Vec<(String, f32)>,

    // Retained for reset (exact, deterministic re-seed; pool-padded layout).
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
    initial_water: u32,
}

/// Pour-emission state + spout parameters (host-side; mirrors the xpbd `Inflow`). Turns
/// `EmissionInput` into activated water-pool particles via the volume-consistent
/// arclength-credit emitter: the volume accumulator (flow/V_w·dt) is the master count budget;
/// layers release one `particle_spacing` of stream travel apart, each filling a golden-angle
/// disc, so the inlet packs to the fluid's rest density.
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
    emitted_mass: f32, // running total emitted water mass (conservation accounting)
}

/// Orthonormal disc basis perpendicular to a (unit) pour direction `dir` (mirrors xpbd).
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

struct Pipelines {
    grid_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g_water: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p_water: (wgpu::ComputePipeline, wgpu::BindGroup),
    // U3 pressure family. The Jacobi sweeps ping-pong the pressure pair by swapping which
    // buffer sits at the src/dst binding indices (two bind-group variants, index = sweep % 2);
    // prolong/project carry parity variants selecting the final-parity buffer.
    node_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    cell_classify: (wgpu::ComputePipeline, wgpu::BindGroup),
    residual: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    coarse_node_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    coarse_cell_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    jacobi_coarse: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    // prolong_add variants indexed [current-p parity][coarse-final parity].
    prolong_add: (wgpu::ComputePipeline, [[wgpu::BindGroup; 2]; 2]),
    jacobi_fine: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    project: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    // U4 surface family: flood fill (label ping-pong in the pressure slots), pocket marking,
    // and the single-workgroup bubble-row solves at each level (parity variants like the
    // sweeps they precede).
    flood_init: (wgpu::ComputePipeline, wgpu::BindGroup),
    flood_sweep: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    pocket_mark: (wgpu::ComputePipeline, wgpu::BindGroup),
    bubble_fine: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    bubble_coarse: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    // Test-only operator taps (never dispatched in step; zero budget impact).
    dbg_div: (wgpu::ComputePipeline, wgpu::BindGroup),
    dbg_grad: (wgpu::ComputePipeline, wgpu::BindGroup),
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

/// Coarse CELL counts at `ratio` fine cells per coarse cell per axis (fine cells = node dims
/// − 1; partial coarse cells at the high end clamp their children to the fine range).
fn coarse_dims_for(dims: [u32; 3], ratio: u32) -> [u32; 3] {
    [
        (dims[0] - 1).div_ceil(ratio),
        (dims[1] - 1).div_ceil(ratio),
        (dims[2] - 1).div_ceil(ratio),
    ]
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

    /// Set the U3 pressure-budget knobs (KTD-9 grid points; dev/test only). `coarse_sweeps =
    /// 0` is the A/B seed-off arm (the whole coarse stage is skipped, the fine sweeps start
    /// from p = 0); `coarse_sweeps = fine_sweeps = 0` makes the frame U2-equivalent (the
    /// projection applies a zero pressure field).
    pub fn set_pressure_budget_for_test(
        &mut self,
        coarse_ratio: u32,
        coarse_sweeps: u32,
        fine_sweeps: u32,
    ) {
        assert!(
            coarse_ratio >= COARSE_RATIO_DEFAULT,
            "coarse buffers are sized for ratio {COARSE_RATIO_DEFAULT} (the densest grid point)"
        );
        let d = self.params.grid_dims;
        let cd = coarse_dims_for([d[0], d[1], d[2]], coarse_ratio);
        self.params.coarse_dims = [cd[0], cd[1], cd[2], coarse_ratio];
        self.coarse_ratio = coarse_ratio;
        self.coarse_sweeps = coarse_sweeps;
        self.fine_sweeps = fine_sweeps;
    }

    /// Set the live water count (dev/test only — state-replay support for the decay gates;
    /// `n` must not exceed the allocated water pool).
    pub fn set_live_water_for_test(&mut self, n: u32) {
        assert!(
            n <= self.water_capacity,
            "live water {n} exceeds the allocated pool {}",
            self.water_capacity
        );
        self.water_count = n;
        self.params.water_count = n;
    }

    /// Read back the bubble state `[λ_b, δλ_b]` (dev/test only — stalls).
    pub fn read_bubble(&self) -> [f32; 2] {
        let raw: Vec<f32> = bytemuck::cast_slice(&self.read_bytes(&self.bubble, 8)).to_vec();
        [raw[0], raw[1]]
    }

    /// Total water mass emitted by the pour so far (conservation accounting; dev/test).
    pub fn total_emitted_water_mass(&self) -> f32 {
        self.inflow.emitted_mass
    }

    /// Activate pour-emitted water particles for this frame from `EmissionInput` — the
    /// volume-consistent arclength-credit emitter, ported from the xpbd solver (see its
    /// `emit` for the derivations): the volume accumulator (flow/V_w·dt) is the master count
    /// budget; layers release one `particle_spacing` of stream travel apart, each filling a
    /// golden-angle disc with up to `N_layer = ceil(A_eff·spacing/V_w)` particles, so the
    /// inlet packs to rest density. Particles are written into the water pool's live range
    /// and `water_count` grows. No-ops when not pouring and no backlog remains.
    fn emit(&mut self, input: &EmissionInput, dt: f32) {
        // A Reset event clears the emitter's backlog/credit (the emitter contract; a full
        // sim restart is `reset()`). Done before the gate so a Reset with zero flow clears.
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
        // Orifice relation: exit speed from flow + effective area. While draining a backlog
        // at zero flow, keep the last cadence so the stream tail stays correctly spaced.
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
            // Capacity check BEFORE spending arclength credit, so a full pool doesn't
            // silently consume a layer's axial.
            let avail = self.water_capacity - (self.water_count + new_pos.len() as u32);
            if avail == 0 {
                clamped = true;
                break;
            }
            self.inflow.axial -= self.inflow.spacing;
            let depth = self.inflow.axial; // residual stream travel below the nozzle
            let this_layer = n_layer.min(want).min(avail);
            for _ in 0..this_layer {
                // Radial shell cycles with the cursor (mod N_layer) so partial layers still
                // cover the whole disc over time; golden angle fills it uniformly.
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
                "twofield pour: water pool {} reached; emission clamped (a recipe scene must \
                 declare enough pour_water_ml to size the pool to its dose)",
                self.water_capacity
            );
        }
        let emit_n = new_pos.len() as u32;
        if emit_n == 0 {
            return;
        }
        // Clamp-before-decrement: subtract only what was actually emitted — unspent budget
        // stays as backlog rather than being silently burned.
        self.inflow.accumulator -= emit_n as f32;
        self.inflow.emitted_mass += emit_n as f32 * self.params.particle_mass;
        let off_v4 = (self.water_count as u64) * 16;
        self.queue
            .write_buffer(&self.pos, off_v4, bytemuck::cast_slice(&new_pos));
        self.queue
            .write_buffer(&self.vel, off_v4, bytemuck::cast_slice(&new_vel));
        self.queue
            .write_buffer(&self.chem, off_v4, bytemuck::cast_slice(&new_chem));
        // Phase tags of dormant pool slots are already 0 (water) and the APIC C rows are
        // zero (each slot is activated at most once per run; reset re-zeroes them).
        self.water_count += emit_n;
        self.params.water_count = self.water_count;
    }

    /// Overwrite particle positions (dev/test only; the .w lane carries moisture).
    pub fn write_positions_for_test(&self, positions: &[[f32; 4]]) {
        assert_eq!(
            positions.len(),
            self.params.particle_count as usize,
            "position seed length must match particle count"
        );
        self.queue
            .write_buffer(&self.pos, 0, bytemuck::cast_slice(positions));
    }

    /// Overwrite the float grid-velocity field (dev/test only — operator gates feed random
    /// node fields to the dbg taps). Length must equal the node count; .w is the mass lane.
    pub fn write_grid_velocities_for_test(&self, v: &[[f32; 4]]) {
        assert_eq!(v.len(), self.num_nodes as usize, "one vec4 per grid node");
        self.queue
            .write_buffer(&self.grid_vel, 0, bytemuck::cast_slice(v));
    }

    /// Overwrite the fine pressure field (slot pf_a, which `dbg_grad` reads; dev/test only).
    pub fn write_pressure_for_test(&self, p: &[f32]) {
        assert_eq!(p.len(), self.num_cells as usize, "one f32 per fine cell");
        self.queue
            .write_buffer(&self.pf_a, 0, bytemuck::cast_slice(p));
    }

    /// Read back the float grid velocities (.xyz) + node masses (.w). After `step()` this is
    /// the post-projection field (dev/test only — stalls).
    pub fn read_grid_velocities(&self) -> Vec<[f32; 4]> {
        let bytes = (self.num_nodes as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.grid_vel, bytes)).to_vec()
    }

    /// Read back the per-cell (rhs, active, dbg_div, _) meta (dev/test only — stalls).
    pub fn read_cell_meta(&self) -> Vec<[f32; 4]> {
        let bytes = (self.num_cells as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.cell_meta, bytes)).to_vec()
    }

    /// Read back the per-node M̃⁻¹ matrices, 8 floats per node:
    /// (xx, xy, xz, yy, yz, zz, massy_flag, 0) (dev/test only — stalls).
    pub fn read_node_matrices(&self) -> Vec<[f32; 8]> {
        let bytes = (self.num_nodes as u64) * 32;
        bytemuck::cast_slice(&self.read_bytes(&self.nm, bytes)).to_vec()
    }

    /// Read back the frame's solved pressure (the final-parity ping-pong buffer the project
    /// pass consumed; dev/test only — stalls).
    pub fn read_pressure(&self) -> Vec<f32> {
        let buf = if self.fine_sweeps.is_multiple_of(2) {
            &self.pf_a
        } else {
            &self.pf_b
        };
        let bytes = (self.num_cells as u64) * 4;
        bytemuck::cast_slice(&self.read_bytes(buf, bytes)).to_vec()
    }

    /// Read back the per-particle APIC C rows (3 vec4 per particle; dev/test only — stalls).
    pub fn read_affine_rows(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 48;
        bytemuck::cast_slice(&self.read_bytes(&self.cmat, bytes)).to_vec()
    }

    fn run_dbg(&self, pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, n_groups: u32) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-dbg"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("twofield-dbg"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, Some(bind), &[]);
            pass.dispatch_workgroups(n_groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Dispatch the test-only `dbg_div` tap: writes the raw masked divergence of the current
    /// grid-velocity field into `cell_meta.z`. Masks/nm must exist (call after a `step()`).
    pub fn run_div_for_test(&self) {
        self.run_dbg(
            &self.pipelines.dbg_div.0,
            &self.pipelines.dbg_div.1,
            groups(self.num_cells).max(1),
        );
    }

    /// Dispatch the test-only `dbg_grad` tap: writes the raw masked gradient of pf_a into the
    /// grid-velocity buffer (overwrites it — test sequences only).
    pub fn run_grad_for_test(&self) {
        self.run_dbg(
            &self.pipelines.dbg_grad.0,
            &self.pipelines.dbg_grad.1,
            groups(self.num_nodes).max(1),
        );
    }

    /// Fine-sweep split around the coarse correction: pre-smooth half the budget (≥1 — the
    /// residual restriction needs the boundary-spike content absorbed first, see
    /// pressure.wgsl), post-smooth the rest. Seed-off runs everything as post-smoothing.
    fn smooth_split(&self) -> (u32, u32) {
        let pre = if self.coarse_sweeps > 0 && self.fine_sweeps > 0 {
            (self.fine_sweeps / 2).max(1)
        } else {
            0
        };
        (pre, self.fine_sweeps - pre)
    }

    /// Which ping-pong buffer holds the final pressure (0 = pf_a) — mirrors step()'s parity.
    fn pressure_parity(&self) -> usize {
        let (pre, post) = self.smooth_split();
        let mut par = (pre % 2) as usize;
        if self.coarse_sweeps > 0 {
            par ^= 1; // prolong_add flips
        }
        par ^= (post % 2) as usize;
        par
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

        let (seed_positions, seed_phases, water_seed) = seed_ranges(scene, mats, cfg);
        let solid_count = seed_positions.len() as u32 - water_seed;

        // Water pool: seed + dose headroom for a declared pour (KEEP §2 calibration; V_w =
        // spacing³ at this solver's rest density). Solids sit AFTER the full pool so the
        // KTD-1 range layout survives emission; dormant slots `[water_count, water_capacity)`
        // are never dispatched (the water kernels guard on the LIVE count) and are parked at
        // the box corner with zero moisture until activated.
        let v_w = mats.particle_spacing.powi(3);
        let dose_headroom = if scene.declares_pour() {
            (scene.pour_water_ml / ML_PER_SIM_UNIT3 / v_w).ceil() as u32
        } else {
            0
        };
        let water_capacity = water_seed + dose_headroom;
        let particle_count = water_capacity + solid_count; // pool size (buffers + readbacks)
        let park = [scene.box_min[0], scene.box_min[1], scene.box_min[2], 0.0];
        let mut positions = seed_positions[..water_seed as usize].to_vec();
        positions.resize(water_capacity as usize, park);
        positions.extend_from_slice(&seed_positions[water_seed as usize..]);
        let mut phases = seed_phases[..water_seed as usize].to_vec();
        phases.resize(water_capacity as usize, 0);
        phases.extend_from_slice(&seed_phases[water_seed as usize..]);
        let water_count = water_seed; // live count

        let (origin, cell, dims) = grid_spec_for(scene, mats);
        let num_nodes = dims[0] * dims[1] * dims[2];
        let num_cells = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1);
        // Coarse buffers are sized once for the densest knob-grid ratio (4 → most coarse
        // cells); the ratio-8 grid point uses a prefix of them.
        let cdims = coarse_dims_for(dims, COARSE_RATIO_DEFAULT);
        let num_ccells = cdims[0] * cdims[1] * cdims[2];
        let num_cnodes = (cdims[0] + 1) * (cdims[1] + 1) * (cdims[2] + 1);
        // ρ_rest = particle_mass/spacing³ (the rest node density the M̃⁻¹ floor is keyed to);
        // mass_eps gates "this node carries water" well above fixed-point decode noise.
        // The density floor is ρ_rest itself: a dilute (free-surface/splash) node otherwise
        // receives an invρ-amplified pressure kick (a 0.25·ρ_rest node would get 4× the bulk
        // Δv), which measurably pumps energy into the surface every frame until eruption.
        // Flooring at ρ_rest under-kicks dilute nodes instead — they sag, compact, and the
        // floor releases: stable. The floor enters A and the projection through the SAME M̃⁻¹,
        // so the operator family stays consistent; proper free-surface treatment is U4.
        let rest_density = mats.particle_mass / mats.particle_spacing.powi(3);
        let rho_floor = rest_density;
        let mass_eps = 1.0e-4 * mats.particle_mass;

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
            coarse_dims: [cdims[0], cdims[1], cdims[2], COARSE_RATIO_DEFAULT],
            extra: [rest_density, rho_floor, mass_eps, 0.0],
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
        // COPY_SRC for the snapshot/replay readback in the U3 divergence-decay gate.
        let cmat = Self::storage(
            &device,
            "twofield-cmat",
            n * 48,
            wgpu::BufferUsages::COPY_SRC,
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
            wgpu::BufferUsages::COPY_SRC,
        );
        // U3 pressure-family buffers (layouts in pressure.wgsl).
        let nm = Self::storage(
            &device,
            "twofield-nm",
            (num_nodes.max(1) as u64) * 32,
            wgpu::BufferUsages::COPY_SRC,
        );
        let cell_meta = Self::storage(
            &device,
            "twofield-cell-meta",
            (num_cells.max(1) as u64) * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        let pf_a = Self::storage(
            &device,
            "twofield-pf-a",
            (num_cells.max(1) as u64) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let pf_b = Self::storage(
            &device,
            "twofield-pf-b",
            (num_cells.max(1) as u64) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let nm_c = Self::storage(
            &device,
            "twofield-nm-c",
            (num_cnodes.max(1) as u64) * 32,
            wgpu::BufferUsages::empty(),
        );
        let cmeta = Self::storage(
            &device,
            "twofield-cmeta",
            (num_ccells.max(1) as u64) * 16,
            wgpu::BufferUsages::empty(),
        );
        let pc_a = Self::storage(
            &device,
            "twofield-pc-a",
            (num_ccells.max(1) as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        let pc_b = Self::storage(
            &device,
            "twofield-pc-b",
            (num_ccells.max(1) as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        // U4 bubble state: [λ_b, δλ_b].
        let bubble = Self::storage(&device, "twofield-bubble", 8, wgpu::BufferUsages::COPY_SRC);
        // U4 per-cell particle counts (the surface-classification particle-presence census).
        let cell_cnt = Self::storage(
            &device,
            "twofield-cell-cnt",
            (num_cells.max(1) as u64) * 4,
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
        // One readback scratch big enough for the largest readable buffer (particle vec4s,
        // affine rows at 48 B/particle, grid lanes, or the 32 B/node M̃⁻¹ matrices).
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("twofield-readback"),
            size: (n * 48).max(grid_bytes).max((num_nodes.max(1) as u64) * 32),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // WGSL has no imports: assemble the one module from the concern files. `common` declares
        // Params/bindings + shared helpers; `transfers` adds the APIC water passes; `pressure`
        // adds the U3 incompressibility family (bindings 9–16); `surface` adds the U4 flood
        // fill + constraint bubble (binding 17).
        let shader_src = format!(
            "{}\n{}\n{}\n{}",
            include_str!("common.wgsl"),
            include_str!("transfers.wgsl"),
            include_str!("pressure.wgsl"),
            include_str!("surface.wgsl"),
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
            let grid_clear_bind = bg(
                &grid_clear,
                &[(0, &params_buf), (6, &grid_fp), (18, &cell_cnt)],
            );
            let p2g = make("p2g_water");
            let p2g_bind = bg(
                &p2g,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (5, &cmat),
                    (6, &grid_fp),
                    (18, &cell_cnt),
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
            // --- U3 pressure family (bindings per pass derived in the budget comment) ---
            let node_setup = make("node_setup");
            let node_setup_bind = bg(
                &node_setup,
                &[(0, &params_buf), (7, &grid_vel), (8, &solids_buf), (9, &nm)],
            );
            let cell_classify = make("cell_classify");
            let cell_classify_bind = bg(
                &cell_classify,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (8, &solids_buf),
                    (10, &cell_meta),
                    (11, &pf_a),
                    (12, &pf_b),
                    (18, &cell_cnt),
                ],
            );
            let coarse_node_setup = make("coarse_node_setup");
            let coarse_node_setup_bind = bg(
                &coarse_node_setup,
                &[(0, &params_buf), (9, &nm), (13, &nm_c)],
            );
            let coarse_cell_setup = make("coarse_cell_setup");
            let coarse_cell_setup_bind = bg(
                &coarse_cell_setup,
                &[
                    (0, &params_buf),
                    (10, &cell_meta),
                    (14, &cmeta),
                    (15, &pc_a),
                    (16, &pc_b),
                ],
            );
            let jacobi_coarse = make("jacobi_coarse");
            let jacobi_coarse_binds = [
                bg(
                    &jacobi_coarse,
                    &[
                        (0, &params_buf),
                        (13, &nm_c),
                        (14, &cmeta),
                        (15, &pc_a),
                        (16, &pc_b),
                        (17, &bubble),
                    ],
                ),
                bg(
                    &jacobi_coarse,
                    &[
                        (0, &params_buf),
                        (13, &nm_c),
                        (14, &cmeta),
                        (15, &pc_b),
                        (16, &pc_a),
                        (17, &bubble),
                    ],
                ),
            ];
            let residual = make("residual");
            let residual_binds = [
                bg(
                    &residual,
                    &[(0, &params_buf), (9, &nm), (10, &cell_meta), (11, &pf_a)],
                ),
                bg(
                    &residual,
                    &[(0, &params_buf), (9, &nm), (10, &cell_meta), (11, &pf_b)],
                ),
            ];
            // prolong_add reads the current pressure (11) + the coarse final-parity buffer
            // (15) and writes the corrected pressure (12) — variants [p parity][pc parity].
            let prolong_add = make("prolong_add");
            let pa = |src: &wgpu::Buffer, dst: &wgpu::Buffer, pc: &wgpu::Buffer| {
                bg(
                    &prolong_add,
                    &[
                        (0, &params_buf),
                        (10, &cell_meta),
                        (14, &cmeta),
                        (15, pc),
                        (11, src),
                        (12, dst),
                        (17, &bubble),
                    ],
                )
            };
            let prolong_add_binds = [
                [pa(&pf_a, &pf_b, &pc_a), pa(&pf_a, &pf_b, &pc_b)],
                [pa(&pf_b, &pf_a, &pc_a), pa(&pf_b, &pf_a, &pc_b)],
            ];
            let jacobi_fine = make("jacobi_fine");
            let jacobi_fine_binds = [
                bg(
                    &jacobi_fine,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_a),
                        (12, &pf_b),
                        (17, &bubble),
                    ],
                ),
                bg(
                    &jacobi_fine,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_b),
                        (12, &pf_a),
                        (17, &bubble),
                    ],
                ),
            ];
            // Project consumes the fine final-parity buffer (index = fine_sweeps % 2).
            let project = make("project");
            let project_binds = [
                bg(
                    &project,
                    &[
                        (0, &params_buf),
                        (7, &grid_vel),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_a),
                    ],
                ),
                bg(
                    &project,
                    &[
                        (0, &params_buf),
                        (7, &grid_vel),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_b),
                    ],
                ),
            ];
            // --- U4 surface family --------------------------------------------------------
            let flood_init = make("flood_init");
            let flood_init_bind = bg(
                &flood_init,
                &[(0, &params_buf), (10, &cell_meta), (11, &pf_a)],
            );
            let flood_sweep = make("flood_sweep");
            let flood_sweep_binds = [
                bg(
                    &flood_sweep,
                    &[(0, &params_buf), (10, &cell_meta), (11, &pf_a), (12, &pf_b)],
                ),
                bg(
                    &flood_sweep,
                    &[(0, &params_buf), (10, &cell_meta), (11, &pf_b), (12, &pf_a)],
                ),
            ];
            // pocket_mark reads the final labels from pf_a (FLOOD_SWEEPS is even) and
            // re-zeroes both pressure slots for the solve.
            let pocket_mark = make("pocket_mark");
            let pocket_mark_bind = bg(
                &pocket_mark,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (10, &cell_meta),
                    (11, &pf_a),
                    (12, &pf_b),
                    (17, &bubble),
                ],
            );
            let bubble_fine = make("bubble_fine");
            let bf = |pf: &wgpu::Buffer| {
                bg(
                    &bubble_fine,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, pf),
                        (17, &bubble),
                    ],
                )
            };
            let bubble_fine_binds = [bf(&pf_a), bf(&pf_b)];
            let bubble_coarse = make("bubble_coarse");
            let bc = |pc: &wgpu::Buffer| {
                bg(
                    &bubble_coarse,
                    &[
                        (0, &params_buf),
                        (13, &nm_c),
                        (14, &cmeta),
                        (15, pc),
                        (17, &bubble),
                    ],
                )
            };
            let bubble_coarse_binds = [bc(&pc_a), bc(&pc_b)];
            let dbg_div = make("dbg_div");
            let dbg_div_bind = bg(
                &dbg_div,
                &[(0, &params_buf), (7, &grid_vel), (10, &cell_meta)],
            );
            let dbg_grad = make("dbg_grad");
            let dbg_grad_bind = bg(
                &dbg_grad,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (10, &cell_meta),
                    (11, &pf_a),
                ],
            );
            Pipelines {
                grid_clear: (grid_clear, grid_clear_bind),
                p2g_water: (p2g, p2g_bind),
                grid_update: (grid_update, grid_update_bind),
                g2p_water: (g2p, g2p_bind),
                node_setup: (node_setup, node_setup_bind),
                cell_classify: (cell_classify, cell_classify_bind),
                residual: (residual, residual_binds),
                coarse_node_setup: (coarse_node_setup, coarse_node_setup_bind),
                coarse_cell_setup: (coarse_cell_setup, coarse_cell_setup_bind),
                jacobi_coarse: (jacobi_coarse, jacobi_coarse_binds),
                prolong_add: (prolong_add, prolong_add_binds),
                jacobi_fine: (jacobi_fine, jacobi_fine_binds),
                project: (project, project_binds),
                flood_init: (flood_init, flood_init_bind),
                flood_sweep: (flood_sweep, flood_sweep_binds),
                pocket_mark: (pocket_mark, pocket_mark_bind),
                bubble_fine: (bubble_fine, bubble_fine_binds),
                bubble_coarse: (bubble_coarse, bubble_coarse_binds),
                dbg_div: (dbg_div, dbg_div_bind),
                dbg_grad: (dbg_grad, dbg_grad_bind),
            }
        };

        let ts = if gpu.timestamps_supported {
            // 69 passes per frame at the U4 defaults (2 queries each); headroom for U5+.
            let capacity = 192u32;
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
            water_capacity,
            solid_count,
            num_nodes,
            num_cells,
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
            coarse_ratio: COARSE_RATIO_DEFAULT,
            coarse_sweeps: COARSE_SWEEPS_DEFAULT,
            fine_sweeps: FINE_SWEEPS_DEFAULT,
            params_buf,
            pos,
            vel,
            phase,
            chem,
            cmat,
            grid_fp,
            grid_vel,
            nm,
            cell_meta,
            pf_a,
            pf_b,
            bubble,
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
        // Exact, deterministic re-seed: positions + phase tags back to the seed (pool-padded
        // layout — pour-activated slots return to the parked dormant state); velocities, chem
        // lanes, and APIC C matrices re-zeroed; the emitter restarts from t = 0. (The grid is
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
            let czeros = vec![[0.0f32; 4]; 3 * self.initial_positions.len()];
            self.queue
                .write_buffer(&self.cmat, 0, bytemuck::cast_slice(&czeros));
        }
        self.water_count = self.initial_water;
        self.params.water_count = self.initial_water;
        self.inflow.accumulator = 0.0;
        self.inflow.axial = 0.0;
        self.inflow.last_exit_speed = 0.0;
        self.inflow.cursor = 0;
        self.inflow.emitted_mass = 0.0;
        self.cached_passes.clear();
        self.dispatches = 0;
    }

    fn step(&mut self, dt: f32, input: &EmissionInput) {
        // Pour emission first (grows water_count for this frame); no-op when not pouring.
        self.emit(input, dt);
        debug_assert!(self.water_count <= self.water_capacity);
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

        // The U2 transfer pipeline + U3 pressure stack, one substep per frame. Passes dispatch
        // at least one workgroup so the dispatch/profiling path is real even on an empty scene
        // (threads early-out on the live-set / mask guards — over-dispatch + early-out, never
        // indirect dispatch, R8).
        let node_groups = groups(self.num_nodes).max(1);
        let water_groups = groups(self.water_count).max(1);
        let cell_groups = groups(self.num_cells).max(1);
        let cd = self.params.coarse_dims;
        let ccell_groups = groups(cd[0] * cd[1] * cd[2]).max(1);
        let cnode_groups = groups((cd[0] + 1) * (cd[1] + 1) * (cd[2] + 1)).max(1);

        // (label, pipeline, bind group, workgroups) — pressure order per pressure.wgsl header:
        // node_setup → cell_classify → pre-smooth → residual → coarse correction →
        // prolong_add → post-smooth → project. `par` tracks which ping-pong buffer holds the
        // current pressure (0 = pf_a); sweeps and prolong_add flip it.
        let (pre, post) = self.smooth_split();
        let mut par = 0usize;
        let mut seq: Vec<(&str, &wgpu::ComputePipeline, &wgpu::BindGroup, u32)> = vec![
            (
                "grid_clear",
                &self.pipelines.grid_clear.0,
                &self.pipelines.grid_clear.1,
                node_groups,
            ),
            (
                "p2g_water",
                &self.pipelines.p2g_water.0,
                &self.pipelines.p2g_water.1,
                water_groups,
            ),
            (
                "grid_update",
                &self.pipelines.grid_update.0,
                &self.pipelines.grid_update.1,
                node_groups,
            ),
            (
                "node_setup",
                &self.pipelines.node_setup.0,
                &self.pipelines.node_setup.1,
                node_groups,
            ),
            (
                "cell_classify",
                &self.pipelines.cell_classify.0,
                &self.pipelines.cell_classify.1,
                cell_groups,
            ),
        ];
        // U4 pocket detection (surface.wgsl): flood the OUTSIDE label through the air cells
        // (labels ping-pong through the pressure slots, which cell_classify just zeroed and
        // pocket_mark re-zeroes), then mark the enclosed remainder as the pocket. The sweep
        // budget is fixed and even, so the final labels land back in pf_a (the pocket_mark
        // binding).
        const _: () = assert!(FLOOD_SWEEPS.is_multiple_of(2));
        seq.push((
            "flood_init",
            &self.pipelines.flood_init.0,
            &self.pipelines.flood_init.1,
            cell_groups,
        ));
        for s in 0..FLOOD_SWEEPS {
            seq.push((
                "flood_sweep",
                &self.pipelines.flood_sweep.0,
                &self.pipelines.flood_sweep.1[(s % 2) as usize],
                cell_groups,
            ));
        }
        seq.push((
            "pocket_mark",
            &self.pipelines.pocket_mark.0,
            &self.pipelines.pocket_mark.1,
            cell_groups,
        ));
        // Every Jacobi sweep is preceded by the single-workgroup bubble-row solve at its own
        // level (KTD-6 identical representation): the sweep then writes the fresh multiplier
        // into the pocket slots while relaxing the fluid rows against the previous one.
        let push_sweeps = |seq: &mut Vec<_>, n: u32, par: &mut usize| {
            for _ in 0..n {
                seq.push((
                    "bubble_fine",
                    &self.pipelines.bubble_fine.0,
                    &self.pipelines.bubble_fine.1[*par],
                    1,
                ));
                seq.push((
                    "jacobi_fine",
                    &self.pipelines.jacobi_fine.0,
                    &self.pipelines.jacobi_fine.1[*par],
                    cell_groups,
                ));
                *par ^= 1;
            }
        };
        push_sweeps(&mut seq, pre, &mut par);
        if self.coarse_sweeps > 0 {
            seq.push((
                "residual",
                &self.pipelines.residual.0,
                &self.pipelines.residual.1[par],
                cell_groups,
            ));
            seq.push((
                "coarse_node_setup",
                &self.pipelines.coarse_node_setup.0,
                &self.pipelines.coarse_node_setup.1,
                cnode_groups,
            ));
            seq.push((
                "coarse_cell_setup",
                &self.pipelines.coarse_cell_setup.0,
                &self.pipelines.coarse_cell_setup.1,
                ccell_groups,
            ));
            for s in 0..self.coarse_sweeps {
                seq.push((
                    "bubble_coarse",
                    &self.pipelines.bubble_coarse.0,
                    &self.pipelines.bubble_coarse.1[(s % 2) as usize],
                    1,
                ));
                seq.push((
                    "jacobi_coarse",
                    &self.pipelines.jacobi_coarse.0,
                    &self.pipelines.jacobi_coarse.1[(s % 2) as usize],
                    ccell_groups,
                ));
            }
            let pc_par = (self.coarse_sweeps % 2) as usize;
            seq.push((
                "prolong_add",
                &self.pipelines.prolong_add.0,
                &self.pipelines.prolong_add.1[par][pc_par],
                cell_groups,
            ));
            par ^= 1;
        }
        push_sweeps(&mut seq, post, &mut par);
        debug_assert_eq!(par, self.pressure_parity());
        seq.push((
            "project",
            &self.pipelines.project.0,
            &self.pipelines.project.1[par],
            node_groups,
        ));
        seq.push((
            "g2p_water",
            &self.pipelines.g2p_water.0,
            &self.pipelines.g2p_water.1,
            water_groups,
        ));
        for (label, pipe, bind, g) in seq {
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
        // Exposure of the pool layout: with no solids the live water range `[0, water_count)`
        // is a contiguous prefix, so only the live set is exposed (dormant tail hidden —
        // the xpbd shape). With solids present the dormant gap sits between the water pool
        // and the solid range; the full pool is exposed and dormant slots stay parked at the
        // box corner with zero velocity (no twofield scene renders solids + pour before U5).
        let exposed = if self.solid_count == 0 {
            self.water_count
        } else {
            self.params.particle_count
        };
        ParticleBuffers {
            particle_count: exposed,
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
            particle_count: self.water_count + self.solid_count, // live set
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

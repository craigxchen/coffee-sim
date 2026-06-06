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
use crate::models::permeability::{drag_rate, kozeny_carman};
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
/// Volume scale (KEEP.md §27): mL per sim-unit³. Converts a scene's pour dose (mL) into a particle
/// capacity (and the emission rate, in the pour-emission unit).
const ML_PER_SIM_UNIT3: f32 = 5.20;

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
    num_solids: u32, // count of static SDF solids in the `solids` storage buffer (0 = none)
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
    grain_sleep_speed: f32,
    // --- water/bed coupling (mixed scenes) ---
    grain_mass: f32,
    grain_volume: f32, // (π/6)·grain_diameter³ — effective volume for the α_s sum
    packing_limit: f32,
    exclusion_relax: f32,
    drag_gamma: f32,
    drag_beta_max: f32,
    buoyancy_scale: f32,
    wake_threshold: f32,
    water_grain_distance: f32,
    // --- wetting / cohesion (Phase 1.4) ---
    r_max: f32,           // moisture ratio at saturation (mass water / mass dry grain)
    rho_ratio: f32,       // ρ_s/ρ_w — converts absorbed water mass → swelling volume
    s_peak: f32,          // saturation at the cohesion-curve peak
    c_max: f32,           // peak wet cohesion strength (0 until calibrated)
    k_abs: f32,           // absorption rate constant (1/s)
    absorb_roundoff: f32, // f_w deactivation floor (exact-conservation; ≪ pbf_eps)
    pbf_eps: f32,         // PBF skips water with remaining fraction ≤ this
    // --- extraction + thermal (Phase 1.5) ---
    extract_rate: f32, // opt-in gate (0 = extraction/thermal off; existing scenes byte-unchanged)
    k0_fast: f32,      // base fast-pool dissolution rate (1/s)
    k0_slow: f32,      // base slow-pool dissolution rate (1/s)
    ea_over_r: f32,    // Arrhenius Ea/R (temperature sensitivity)
    t_ref: f32,        // Arrhenius reference temperature (k_T(t_ref)=1)
    c_sat: f32,        // max solute concentration (driving force → 0 here)
    d_ref: f32,        // reference grind diameter for surface area ∝ 1/d
    u_half: f32,       // flux half-saturation (saturating flow→extraction bridge)
    s_on: f32,         // moisture-gate onset saturation (dry grain doesn't extract)
    kappa: f32,        // thermal conductance (pair heat κ)
    cp_water: f32,     // water specific heat (C_water = particle_mass·f_w·cp_water)
    cp_grain: f32,     // grain specific heat (C_grain = grain_eff_mass·cp_grain)
    h_amb: f32,        // ambient heat-loss rate
    t_amb: f32,        // ambient temperature
    _pad_chem0: f32,
    _pad_chem1: f32,
    // --- fines migration (Phase 6); vec4-aligned tail (rate/gate, seed, crit_flux, _) ---
    fines: [f32; 4],
}

// Params is uploaded as a uniform and must stay byte-identical to the WGSL `Params`.
const _: () = assert!(std::mem::size_of::<Params>() == 336);

/// GPU record for one static SDF solid — byte-identical to the WGSL `Primitive` (64 bytes,
/// vec4-aligned). Cone radii in `a` are OUTER wall radii; the cavity surface is `outer − thickness`.
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

/// Pack a scene's analytic solids into the GPU `Primitive` layout.
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
    _pad0: u32,
    _pad1: u32,
}

struct Pipelines {
    predict: wgpu::ComputePipeline,
    grid_clear: wgpu::ComputePipeline,
    grid_count: wgpu::ComputePipeline,
    grid_scan: wgpu::ComputePipeline,
    grid_scatter: wgpu::ComputePipeline,
    grid_reorder_a: wgpu::ComputePipeline,
    grid_reorder_b: wgpu::ComputePipeline,
    grid_reorder_identity: wgpu::ComputePipeline,
    compute_boundary: wgpu::ComputePipeline,
    compute_lambda: wgpu::ComputePipeline,
    residual_reduce: wgpu::ComputePipeline,
    compute_dp: wgpu::ComputePipeline,
    bed_project: wgpu::ComputePipeline,
    compute_fractions: wgpu::ComputePipeline,
    exclude_water: wgpu::ComputePipeline,
    exclude_grain: wgpu::ComputePipeline,
    compute_coupling_scale: wgpu::ComputePipeline,
    drag_water: wgpu::ComputePipeline,
    drag_grain: wgpu::ComputePipeline,
    buoyancy_grain: wgpu::ComputePipeline,
    buoyancy_water: wgpu::ComputePipeline,
    apply_drag_pred: wgpu::ComputePipeline,
    wet_count: wgpu::ComputePipeline,
    wet_water: wgpu::ComputePipeline,
    wet_grain: wgpu::ComputePipeline,
    diss_count: wgpu::ComputePipeline,
    dissolve_grain: wgpu::ComputePipeline,
    dissolve_water: wgpu::ComputePipeline,
    thermal_exchange: wgpu::ComputePipeline,
    fines_count: wgpu::ComputePipeline,
    fines_grain: wgpu::ComputePipeline,
    fines_water: wgpu::ComputePipeline,
    apply_dp: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
    xsph: wgpu::ComputePipeline,
}

struct BindGroups {
    predict: wgpu::BindGroup,
    grid_clear: wgpu::BindGroup,
    grid_count: wgpu::BindGroup,
    grid_scan: wgpu::BindGroup,
    grid_scatter: wgpu::BindGroup,
    grid_reorder_a: wgpu::BindGroup,
    grid_reorder_b: wgpu::BindGroup,
    grid_reorder_identity: wgpu::BindGroup,
    compute_boundary: wgpu::BindGroup,
    compute_lambda: wgpu::BindGroup,
    residual_reduce: wgpu::BindGroup,
    compute_dp: wgpu::BindGroup,
    bed_project: wgpu::BindGroup,
    compute_fractions: wgpu::BindGroup,
    exclude_water: wgpu::BindGroup,
    exclude_grain: wgpu::BindGroup,
    compute_coupling_scale: wgpu::BindGroup,
    drag_water: wgpu::BindGroup,
    drag_grain: wgpu::BindGroup,
    buoyancy_grain: wgpu::BindGroup,
    buoyancy_water: wgpu::BindGroup,
    apply_drag_pred: wgpu::BindGroup,
    wet_count: wgpu::BindGroup,
    wet_water: wgpu::BindGroup,
    wet_grain: wgpu::BindGroup,
    diss_count: wgpu::BindGroup,
    dissolve_grain: wgpu::BindGroup,
    dissolve_water: wgpu::BindGroup,
    thermal_exchange: wgpu::BindGroup,
    fines_count: wgpu::BindGroup,
    fines_grain: wgpu::BindGroup,
    fines_water: wgpu::BindGroup,
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
}

pub struct XpbdSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Params,
    /// Allocated particle-pool size (buffers are sized to this). `= seed_count + pour headroom`;
    /// equals `active_count` for non-pour scenes.
    capacity: u32,
    /// Live particle count actually simulated this frame (dispatch size + `params.particle_count`).
    /// Starts at the seed count; grows as the pour activates dormant pool slots.
    active_count: u32,
    num_cells: u32,
    substeps: u32,
    /// Which species are present (from the scene's seed regions) — selects the per-frame passes.
    has_water: bool,
    has_grain: bool,
    /// Whether the scene has SDF solids — gates the water-boundary density-compensation pass.
    has_solids: bool,
    /// Water density-solve iteration cap + grid-rebuild interval.
    water_iters: u32,
    water_regrid: u32,
    /// Bed contact-solve iteration cap + grid-rebuild interval.
    bed_iters: u32,
    bed_regrid: u32,
    /// Jacobi sub-iterations for implicit water↔grain velocity drag (mixed scenes only).
    drag_subiters: u32,

    // Buffers referenced every frame only through the bind groups (which retain them) are
    // not held here. We keep the ones we touch directly: params (write), pos/vel (expose +
    // readback/copy), vel_smoothed (copy src), lambda (test seeding), status (+ readbacks),
    // phase (expose + re-seed).
    params_buf: wgpu::Buffer,
    pos: Arc<wgpu::Buffer>,
    vel: Arc<wgpu::Buffer>,
    pred: wgpu::Buffer, // retained for read_pred (test: pred.w mirror survives the frame)
    vel_smoothed: wgpu::Buffer,
    vel_frozen: wgpu::Buffer,
    lambda: wgpu::Buffer,
    phase: Arc<wgpu::Buffer>,
    status: wgpu::Buffer,
    status_readback: wgpu::Buffer,
    pos_readback: wgpu::Buffer,
    // The `solids` buffer (binding 19) is not held here: like dp / c_residual it lives only in the
    // collision bind groups, which keep its GPU resource alive (geometry is static, never re-uploaded).
    // Phase 1.5 chem/thermal state: `chem` (b20, live) is retained for readback (yield/TDS), the
    // snapshot copy, re-seeding, and exposure as concentration/temperature; `chem_frozen` (b22) is the
    // per-substep snapshot the chem passes read.
    chem: Arc<wgpu::Buffer>,
    // The per-substep snapshot the dissolution/thermal passes read while writing the live `chem`.
    chem_frozen: wgpu::Buffer,
    // Cell-order reorder scratch (gather targets for the water-loop locality reorder; copied back into
    // the live buffers each rebuild). Sized to capacity; only the active range is touched.
    pos_scratch: wgpu::Buffer,
    pred_scratch: wgpu::Buffer,
    vel_scratch: wgpu::Buffer,
    chem_scratch: wgpu::Buffer,
    phase_scratch: wgpu::Buffer,
    pipelines: Pipelines,
    bind_groups: BindGroups,
    ts: Option<Timestamps>,

    dispatches: u32,
    cached_diag: XpbdDiagnostics,
    cached_passes: Vec<(String, f32)>,

    // --- extraction yield/TDS readout (Phase 1.5 U7) ---
    /// Total dry-coffee dose `grain_mass · n_grain` — the yield denominator (real-coffee convention:
    /// yield = dissolved solute / total dry mass, so yield ≤ `soluble_fraction`). Set at build.
    total_dose: f32,
    /// Catch-cup region (from the scene's cylinder solid) for the TDS readout; `None` ⇒ TDS = 0.
    cup: Option<CupRegion>,
    /// Cached yield/TDS, refreshed by `sample_diagnostics` (the stalling cache point) and returned
    /// by `metrics()` without a GPU sync.
    cached_yield: f32,
    cached_tds: f32,

    /// Pour-emission state (water injected over the brew). Inactive when no pour drives it.
    inflow: Inflow,

    // retained for reset (exact re-seed of the initial block + phase tags + chem state)
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
    initial_chem: Vec<[f32; 4]>,
    /// Host mirror of the per-particle phase tag for ALL `active_count` particles (the seed plus any
    /// pour-emitted water), grown as particles are activated. Host readouts (`sample_extraction`,
    /// `read_temperature`) index this rather than the seed-only `initial_phases`.
    phase_mirror: Vec<u32>,
}

/// Catch-cup geometry (a cylinder cavity) used to scope the TDS readout to pooled cup water.
#[derive(Clone, Copy)]
struct CupRegion {
    cx: f32,
    cz: f32,
    radius: f32,
    y_min: f32,
    y_max: f32,
}

/// Pour-emission state + spout parameters (host-side). Turns `EmissionInput` into activated water
/// particles via the volume-consistent arclength-credit emitter (KTD-3): the volume accumulator is
/// the master particle-count budget; layers release one `particle_spacing` of stream travel apart
/// (the axial credit) so the inlet packs to the fluid's rest density (PBF-safe — no spike, no void).
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
    emitted_mass: f32, // running total emitted water mass (conservation accounting, R7)
}

/// Orthonormal disc basis perpendicular to a (unit) pour direction `dir`.
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
    // Reference axis not parallel to dir.
    let refv = if dir[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = norm(cross(dir, refv));
    let w = cross(dir, u);
    (u, w)
}

/// Seed the scene's initial particles on a jittered lattice, one block per region. Returns
/// positions and the matching per-particle phase tags (0=water, 1=grain).
fn seed_block(scene: &Scene, mats: &Materials, cfg: &Config) -> (Vec<[f32; 4]>, Vec<u32>) {
    // Cone-aware rejection clearance: when the scene has solids, a seeded particle must sit at least
    // this far inside a solid's cavity for its species (KEEP.md §4 uses 0.4–0.6 units) so a bed
    // starts inside the dripper, not through its wall.
    const SEED_CLEARANCE: f32 = 0.4;
    let mut rng = crate::utils::rng::Rng::new(cfg.seed);
    let mut pos = Vec::new();
    let mut phase = Vec::new();
    for region in &scene.regions {
        let (lo, hi) = (region.min, region.max);
        // Water seeds at the (fine) particle spacing; grains seed at their (possibly coarser)
        // contact diameter, so a sand wall can have pores larger than the water that flows through
        // it. When grain_diameter == particle_spacing (the default), both seed identically.
        // The 4th lane carries Phase-1.4 moisture state, seeded per species: water = remaining
        // volume fraction f_w (1 = full), grain = absorbed volume V_abs (0 = dry).
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
                    // Cone-aware rejection: drop lattice points that fall outside (or too near) a
                    // solid's cavity for this species. RNG draws happen above regardless, so the
                    // lattice stays deterministic whether or not a point is kept.
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
                    pos.push([px, py, pz, w0]);
                    phase.push(tag);
                }
            }
        }
    }
    (pos, phase)
}

/// Seed the per-particle chem/thermal state (Phase 1.5), one vec4 per particle matching the phase
/// tags: grain = (s_f, s_s, T_g, _) with the soluble dose `soluble_fraction·grain_mass` split into the
/// fast/slow pools; water = (c=0, T_w, _, _). Both species start at the pour temperature.
fn seed_chem(phases: &[u32], mats: &Materials) -> Vec<[f32; 4]> {
    let extractable = mats.soluble_fraction * mats.grain_mass;
    let (s_f, s_s) = crate::models::extraction::split(extractable, mats.fast_fraction);
    // chem.w carries lodged fines (grain) / suspended fines (water); seeded uniformly onto grains
    // so only the migration-induced deviation from this baseline changes permeability (KTD-2).
    // fines_fraction = 0 (default) ⇒ seed 0.0, identical to the pre-fines layout.
    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    let fines_seed = crate::models::fines::fines_seed(grain_volume, mats.fines_fraction);
    phases
        .iter()
        .map(|&ph| {
            if ph == 1 {
                [s_f, s_s, mats.pour_t, fines_seed] // grain: two pools + grain temp + lodged fines
            } else {
                [0.0, mats.pour_t, 0.0, 0.0] // water: concentration 0, water temp, _, suspended fines 0
            }
        })
        .collect()
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
    /// tags, and zeroed velocities. (`normal_impulse` is reset every frame in `predict`.)
    fn seed(&mut self) {
        // Reset the live set to the seed (drops any pour-emitted particles); writes only the seed
        // range — dormant pool slots stay inert. `params.particle_count = active_count` is refreshed
        // each frame in step().
        self.active_count = self.initial_positions.len() as u32;
        self.queue
            .write_buffer(&self.pos, 0, bytemuck::cast_slice(&self.initial_positions));
        self.queue
            .write_buffer(&self.phase, 0, bytemuck::cast_slice(&self.initial_phases));
        let zeros = vec![[0.0f32; 4]; self.initial_positions.len()];
        self.queue
            .write_buffer(&self.vel, 0, bytemuck::cast_slice(&zeros));
        // Re-seed chem/thermal state (pools, c=0, pour temperature) so a reset restarts the brew.
        self.queue
            .write_buffer(&self.chem, 0, bytemuck::cast_slice(&self.initial_chem));
        // Restore the host phase mirror to the seed baseline.
        self.phase_mirror.clear();
        self.phase_mirror.extend_from_slice(&self.initial_phases);
        // Reset the pour-emission state (the recipe restarts from t=0 on reset).
        self.inflow.accumulator = 0.0;
        self.inflow.axial = 0.0;
        self.inflow.last_exit_speed = 0.0;
        self.inflow.cursor = 0;
        self.inflow.emitted_mass = 0.0;
        // Clear the yield/TDS cache so metrics() reports 0 after a reset (matching the re-zeroed
        // chem) rather than stale values until the next sample_diagnostics.
        self.cached_yield = 0.0;
        self.cached_tds = 0.0;
    }

    /// Blocking GPU→CPU read-back of a `vec4` particle buffer (dev/test only — stalls).
    fn read_vec4(&self, src: &wgpu::Buffer) -> Vec<[f32; 4]> {
        let size = (self.active_count as u64) * 16;
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

    /// Read back the per-particle moisture lane `pos.w` (dev/test only — stalls the GPU):
    /// water = remaining-volume fraction `f_w` (1 = full), grain = absorbed volume `V_abs`.
    pub fn read_moisture(&self) -> Vec<f32> {
        self.read_vec4(self.pos.as_ref())
            .iter()
            .map(|p| p[3])
            .collect()
    }

    /// Read back the per-particle chem/thermal lanes (dev/test only — stalls the GPU): grain =
    /// (s_f, s_s, T_g, _), water = (c, T_w, _, _). Used by the extraction tests + yield/TDS readout.
    pub fn read_chem(&self) -> Vec<[f32; 4]> {
        self.read_vec4(self.chem.as_ref())
    }

    /// Read back the predicted-position buffer (dev/test only — stalls the GPU). `pred.w` mirrors
    /// the moisture snapshot; a test asserts it survives every `pred` writer through a full step.
    pub fn read_pred(&self) -> Vec<[f32; 4]> {
        self.read_vec4(&self.pred)
    }

    /// Single water-particle volume `V_w = particle_mass / rest_density` (dev/test only). Lets a
    /// conservation test convert the water moisture lane `f_w` into an absolute volume.
    pub fn water_particle_volume(&self) -> f32 {
        self.params.particle_mass / self.params.rest_density
    }

    /// Overwrite current particle velocities (dev/test only).
    pub fn write_velocities_for_test(&self, velocities: &[[f32; 4]]) {
        assert_eq!(
            velocities.len(),
            self.active_count as usize,
            "velocity seed length must match particle count"
        );
        self.queue
            .write_buffer(&self.vel, 0, bytemuck::cast_slice(velocities));
    }

    /// Overwrite the per-particle chem/thermal lanes (dev/test only): grain = (s_f, s_s, T_g, _),
    /// water = (c, T_w, _, _). Lets a test pre-seed a saturated water field or a known thermal state.
    pub fn write_chem_for_test(&self, chem: &[[f32; 4]]) {
        assert_eq!(
            chem.len(),
            self.active_count as usize,
            "chem seed length must match particle count"
        );
        self.queue
            .write_buffer(self.chem.as_ref(), 0, bytemuck::cast_slice(chem));
    }

    /// Overwrite the per-particle moisture lane `pos.w` (dev/test only): water `f_w`, grain `V_abs`.
    /// Lets a test pre-wet grains so the dissolution pass runs with `absorb_rate=0` (constant `f_w`),
    /// isolating solute conservation from the wetting volume transfer.
    pub fn write_moisture_for_test(&self, moisture: &[f32]) {
        let mut pos = self.read_vec4(self.pos.as_ref());
        assert_eq!(
            pos.len(),
            moisture.len(),
            "moisture seed length must match particle count"
        );
        for (p, &m) in pos.iter_mut().zip(moisture) {
            p[3] = m;
        }
        self.queue
            .write_buffer(self.pos.as_ref(), 0, bytemuck::cast_slice(&pos));
    }

    /// Overwrite current water Lagrange multipliers (dev/test only).
    pub fn write_lambdas_for_test(&self, lambdas: &[f32]) {
        assert_eq!(
            lambdas.len(),
            self.active_count as usize,
            "lambda seed length must match particle count"
        );
        self.queue
            .write_buffer(&self.lambda, 0, bytemuck::cast_slice(lambdas));
    }

    /// Read back per-particle phase tags (0=water, 1=grain) (dev/test only — stalls the GPU).
    pub fn read_phases(&self) -> Vec<u32> {
        let size = (self.active_count as u64) * 4;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("phase-readback"),
            });
        enc.copy_buffer_to_buffer(self.phase.as_ref(), 0, &self.pos_readback, 0, size);
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
        let out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.pos_readback.unmap();
        out
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

        self.sample_extraction();
    }

    /// Refresh the cached yield/TDS from the current chem + positions (a blocking readback, folded
    /// into `sample_diagnostics`). Yield = dissolved solute / total soluble dose; TDS = cup solute /
    /// cup water mass, over the catch-cup region. All water volumes scale by the remaining fraction
    /// `f_w` (deactivated water `f_w ≤ roundoff` is excluded). Phase is the immutable seed tag.
    fn sample_extraction(&mut self) {
        if self.total_dose <= 1.0e-9 {
            return; // no coffee dose (e.g. water-only scene) ⇒ yield/TDS stay 0
        }
        let chem = self.read_vec4(self.chem.as_ref());
        let pos = self.read_vec4(self.pos.as_ref());
        let v_w = self.params.particle_mass / self.params.rest_density;
        let rho_w = self.params.rest_density;
        let mut dissolved = 0.0f32;
        let mut cup_solute = 0.0f32;
        let mut cup_water_mass = 0.0f32;
        for (i, (c, p)) in chem.iter().zip(&pos).enumerate() {
            if self.phase_mirror[i] != 0 {
                continue; // water only (mirror covers seed + pour-emitted particles)
            }
            let f_w = p[3];
            if f_w <= self.params.absorb_roundoff {
                continue; // deactivated water carries no solute
            }
            let solute = c[0] * f_w * v_w; // c·(f_w·V_w)
            dissolved += solute;
            if let Some(cup) = self.cup {
                let dx = p[0] - cup.cx;
                let dz = p[2] - cup.cz;
                if dx * dx + dz * dz < cup.radius * cup.radius
                    && p[1] >= cup.y_min
                    && p[1] <= cup.y_max
                {
                    cup_solute += solute;
                    cup_water_mass += f_w * v_w * rho_w; // = f_w · particle_mass
                }
            }
        }
        self.cached_yield = dissolved / self.total_dose;
        self.cached_tds = if cup_water_mass > 1.0e-9 {
            cup_solute / cup_water_mass
        } else {
            0.0
        };
    }

    pub fn diagnostics(&self) -> XpbdDiagnostics {
        self.cached_diag
    }

    /// Live particle count currently simulated (grows as the pour activates pool slots) (dev/test).
    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    /// Allocated particle-pool capacity (buffer size in particles) (dev/test).
    pub fn pool_capacity(&self) -> u32 {
        self.capacity
    }

    /// Total spatial-hash cell count `nx·ny·nz` (dev/test scaling diagnostics).
    pub fn grid_cell_count(&self) -> u32 {
        self.params.grid_dims[3]
    }

    /// Whether the per-frame water/coupling/wetting/extraction passes run (true if a water region is
    /// seeded OR the scene declares a pour) (dev/test).
    pub fn has_water_passes(&self) -> bool {
        self.has_water
    }

    /// Total water mass emitted by the pour so far (conservation accounting, R7) (dev/test).
    pub fn total_emitted_water_mass(&self) -> f32 {
        self.inflow.emitted_mass
    }

    /// Activate pour-emitted water particles for this frame from `EmissionInput` (KTD-3): a
    /// volume-consistent arclength-credit emitter. The volume accumulator (flow_rate/V_w·dt) is the
    /// master count budget; layers release one `particle_spacing` of stream travel apart, each filling
    /// a golden-angle disc with up to `N_layer = ceil(A_eff·spacing/V_w)` particles, so the inlet packs
    /// to the fluid's rest density (no PBF spike). Particles are written into the active pool range and
    /// `active_count` grows. No-ops when not pouring and no backlog remains.
    fn emit(&mut self, input: &EmissionInput, dt: f32) {
        // A Reset event clears the emitter's backlog/credit (the emitter contract; a full sim restart
        // is `reset()`). Done before the gate so a Reset with zero flow still clears.
        if input.event == crate::emission::PourEvent::Reset {
            self.inflow.accumulator = 0.0;
            self.inflow.axial = 0.0;
            self.inflow.last_exit_speed = 0.0;
        }
        let flow = input.flow_rate.max(0.0);
        // Gate: pour active, or a whole particle of backlog still to drain (R4 — drawdown drain).
        if flow <= 0.0 && self.inflow.accumulator < 1.0 {
            return;
        }
        let a_eff = std::f32::consts::PI
            * self.inflow.nozzle_radius
            * self.inflow.nozzle_radius
            * self.inflow.discharge_coeff;
        let a_eff = a_eff.max(1.0e-9);
        // Orifice relation: exit_speed derived from flow + effective area (KTD-3). While draining a
        // backlog at zero flow, keep the last cadence so the stream tail stays correctly spaced.
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
        // Volume-consistent layer capacity (ceil ⇒ throughput ≥ flow/V_w, no permanent backlog).
        let n_layer = ((a_eff * self.inflow.spacing / self.inflow.v_w).ceil() as u32).max(1);

        // Pour direction (downward, tilted by pour_angle toward +x) + a perpendicular disc basis.
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
        let mut new_phase: Vec<u32> = Vec::new();
        let mut clamped = false;
        while self.inflow.axial >= self.inflow.spacing && want > 0 {
            // Check capacity BEFORE spending arclength credit, so a full pool doesn't silently consume
            // a layer's axial (Codex U2). The clamp is surfaced below regardless of emit_n.
            let avail = self.capacity - (self.active_count + new_pos.len() as u32);
            if avail == 0 {
                clamped = true;
                break;
            }
            self.inflow.axial -= self.inflow.spacing;
            let depth = self.inflow.axial; // residual stream travel below the nozzle ⇒ layer depth
            let this_layer = n_layer.min(want).min(avail);
            for _ in 0..this_layer {
                // Radial shell cycles with the cursor (mod N_layer) so PARTIAL layers still cover the
                // whole disc over time — no center bias (Codex U2). Golden angle fills it uniformly.
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
                    1.0, // moisture lane f_w = 1 (full water)
                ]);
                new_vel.push([
                    dir[0] * exit_speed,
                    dir[1] * exit_speed,
                    dir[2] * exit_speed,
                    0.0,
                ]);
                new_chem.push([0.0, self.inflow.pour_t, 0.0, 0.0]); // c=0, T=pour temperature
                new_phase.push(0); // water
            }
            want -= this_layer;
        }
        // Surface the capacity clamp whether or not anything emitted (Codex U2 — the warning must not
        // be skipped by the zero-emission early return below).
        if clamped {
            eprintln!(
                "xpbd pour: pool capacity {} reached; emission clamped (a recipe scene must declare \
                 enough pour_water_ml to size the pool to its dose)",
                self.capacity
            );
        }
        let emit_n = new_pos.len() as u32;
        if emit_n == 0 {
            return;
        }
        // Clamp-before-decrement: subtract only what was actually emitted (R4) — unspent budget stays
        // as backlog rather than being silently burned.
        self.inflow.accumulator -= emit_n as f32;
        self.inflow.emitted_mass += emit_n as f32 * self.params.particle_mass;
        let off_v4 = (self.active_count as u64) * 16;
        let off_u32 = (self.active_count as u64) * 4;
        self.queue
            .write_buffer(&self.pos, off_v4, bytemuck::cast_slice(&new_pos));
        self.queue
            .write_buffer(self.chem.as_ref(), off_v4, bytemuck::cast_slice(&new_chem));
        self.queue
            .write_buffer(&self.vel, off_v4, bytemuck::cast_slice(&new_vel));
        self.queue
            .write_buffer(&self.phase, off_u32, bytemuck::cast_slice(&new_phase));
        self.active_count += emit_n;
        self.phase_mirror.extend_from_slice(&new_phase);
    }

    /// Read back per-particle water concentration `c` (chem `.x`; for a grain `.x` is its fast pool).
    /// Dev/test/inspection only — stalls the GPU.
    pub fn read_concentration(&self) -> Vec<f32> {
        self.read_vec4(self.chem.as_ref())
            .iter()
            .map(|c| c[0])
            .collect()
    }

    /// Read back per-particle temperature (phase-aware: water `T` is chem `.y`, grain `T` is `.z`).
    /// Dev/test/inspection only — stalls the GPU.
    pub fn read_temperature(&self) -> Vec<f32> {
        self.read_vec4(self.chem.as_ref())
            .iter()
            .zip(&self.phase_mirror)
            .map(|(c, &ph)| if ph == 0 { c[1] } else { c[2] })
            .collect()
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
        let seed_count = positions.len() as u32;

        // Which species are present drives the per-frame passes. The grain bed solves contacts
        // (harder to converge than water): its own iteration cap + more frequent grid rebuild.
        // A declared pour means water WILL be present even if no water region is seeded, so the
        // water/coupling/wetting/extraction passes must be selected.
        let has_water =
            scene.regions.iter().any(|r| r.species == Species::Water) || scene.declares_pour();
        let has_grain = scene.regions.iter().any(|r| r.species == Species::Grain);
        let has_solids = !scene.solids.is_empty();
        let water_iters = cfg.max_iters;
        let water_regrid = REGRID_INTERVAL;
        let bed_iters = cfg.bed_max_iters;
        let bed_regrid = cfg.bed_regrid_interval.max(1);
        let drag_subiters = cfg.drag_subiters;
        // The shared `status` residual machinery (single-species early-exit) tracks the sole
        // species; a mixed scene runs fixed iteration counts (no early-exit), so it doesn't use it.
        let (param_iters, residual_tolerance) = if has_grain && !has_water {
            (bed_iters, cfg.bed_residual_tolerance)
        } else {
            (water_iters, cfg.residual_tolerance)
        };
        let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);

        // Grid cell must cover the largest neighbor query: the water support radius h, OR the
        // grain contact diameter when grains are coarser than the water (so grain neighbors aren't
        // missed). For single-resolution scenes (grain_diameter ≤ h) this is just h.
        let cell_size = h.max(mats.grain_diameter);
        let nx = (((scene.box_max[0] - scene.box_min[0]) / cell_size).ceil() as u32).max(1);
        let ny = (((scene.box_max[1] - scene.box_min[1]) / cell_size).ceil() as u32).max(1);
        let nz = (((scene.box_max[2] - scene.box_min[2]) / cell_size).ceil() as u32).max(1);
        let num_cells = nx * ny * nz;

        let rest_density = kernels::rest_density(s, h, m);
        // Particle pool: buffers are sized to `capacity` = the seed plus headroom for a declared pour
        // dose (KTD-7: water_ml / 5.20 ml·unit⁻³ / V_w particles). `active_count` (the live, dispatched
        // set) starts at the seed and grows as the pour activates dormant slots. Non-pour scenes get
        // `capacity == active_count == seed_count`, so the layout/dispatch is byte-identical to before.
        let v_w = m / rest_density;
        let dose_headroom = if scene.declares_pour() {
            (scene.pour_water_ml / ML_PER_SIM_UNIT3 / v_w).ceil() as u32
        } else {
            0
        };
        let capacity = seed_count + dose_headroom;
        let active_count = seed_count;
        let dq = cfg.s_corr_dq_ratio * h;
        let s_corr_wq = kernels::w_poly6(dq, h);
        // Resolve the user-facing drag scale through Kozeny-Carman once at build time, stored in
        // `drag_gamma` (live-porosity drag in a later unit makes this per-particle).
        let permeability = kozeny_carman(mats.grain_diameter, mats.porosity);
        let resolved_drag_gamma = drag_rate(permeability, cfg.drag_gamma);

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
            particle_count: active_count, // live count; grows as the pour activates pool slots
            num_solids: scene.solids.len() as u32,
            min_iters: cfg.min_iters,
            max_iters: param_iters,
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
            grain_sleep_speed: cfg.grain_sleep_speed,
            grain_mass: mats.grain_mass,
            grain_volume,
            packing_limit: cfg.packing_limit,
            exclusion_relax: cfg.exclusion_relax,
            drag_gamma: resolved_drag_gamma,
            drag_beta_max: cfg.drag_beta_max,
            buoyancy_scale: cfg.buoyancy_scale,
            wake_threshold: cfg.wake_threshold,
            water_grain_distance: mats.water_grain_distance,
            r_max: mats.r_max,
            rho_ratio: mats.rho_ratio,
            s_peak: mats.s_peak,
            c_max: mats.c_max,
            k_abs: cfg.absorb_rate,
            absorb_roundoff: cfg.absorb_roundoff,
            pbf_eps: cfg.pbf_eps,
            // --- extraction + thermal (Phase 1.5) ---
            extract_rate: cfg.extract_rate,
            k0_fast: mats.k0_fast,
            k0_slow: mats.k0_slow,
            ea_over_r: mats.ea_over_r,
            t_ref: mats.t_ref,
            c_sat: mats.c_sat,
            d_ref: mats.d_ref,
            u_half: mats.u_half,
            s_on: mats.s_on,
            kappa: mats.kappa,
            cp_water: mats.cp_water,
            cp_grain: mats.cp_grain,
            h_amb: mats.h_amb,
            t_amb: mats.t_amb,
            _pad_chem0: 0.0,
            _pad_chem1: 0.0,
            // Fines: gate/scale from Config, per-grain seed + critical flux from Materials. When
            // fines_rate == 0 (default) no fines kernel runs and the seed lane stays inert.
            fines: [
                cfg.fines_rate,
                crate::models::fines::fines_seed(grain_volume, mats.fines_fraction),
                mats.fines_crit_flux,
                0.0,
            ],
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xpbd-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Buffers are sized to `capacity` (the pool); the seed data is written into the first
        // `seed_count` slots below (dormant slots `[seed_count, capacity)` are never dispatched).
        let n = capacity.max(1) as u64;
        let vec4 = n * 16;
        let f32s = n * 4;
        let pos = Arc::new(Self::storage(
            &device,
            "xpbd-pos",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        queue.write_buffer(&pos, 0, bytemuck::cast_slice(&positions));
        // COPY_SRC so read_pred can read back pred.w (test: moisture mirror survives the frame).
        let pred = Self::storage(&device, "xpbd-pred", vec4, wgpu::BufferUsages::COPY_SRC);
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
        let vel_frozen = Self::storage(
            &device,
            "xpbd-vel-frozen",
            vec4,
            wgpu::BufferUsages::empty(),
        );
        let lambda = Self::storage(&device, "xpbd-lambda", f32s, wgpu::BufferUsages::empty());
        let dp = Self::storage(&device, "xpbd-dp", vec4, wgpu::BufferUsages::empty());
        let c_residual =
            Self::storage(&device, "xpbd-cresidual", f32s, wgpu::BufferUsages::empty());
        // Per-particle species tag (exposed to the renderer for color-by-phase) and the per-grain
        // accumulated-normal-impulse buffer (the friction budget; reset each frame in predict).
        let phase = Arc::new(Self::storage(
            &device,
            "xpbd-phase",
            f32s,
            wgpu::BufferUsages::COPY_SRC,
        ));
        queue.write_buffer(&phase, 0, bytemuck::cast_slice(&phases));
        let normal_impulse = Self::storage(
            &device,
            "xpbd-normal-impulse",
            f32s,
            wgpu::BufferUsages::empty(),
        );
        // Per-particle solid fraction α_s (coupling). Zero-initialized by wgpu, so single-species
        // scenes (which never run compute_fractions) read α_s=0 → the water solve is unmodulated.
        let alpha_s = Self::storage(&device, "xpbd-alpha-s", f32s, wgpu::BufferUsages::empty());
        let fluid_impulse = Self::storage(
            &device,
            "xpbd-fluid-impulse",
            f32s,
            wgpu::BufferUsages::empty(),
        );
        let coupling_scale = Self::storage(
            &device,
            "xpbd-coupling-scale",
            f32s,
            wgpu::BufferUsages::empty(),
        );
        // Per-particle eligible-opposite-species neighbor count for the wetting allocation (u32).
        let wet_count = Self::storage(&device, "xpbd-wet-count", f32s, wgpu::BufferUsages::empty());
        // Counting-sort grid: transient per-cell counter, exclusive-prefix-sum offsets (num_cells+1),
        // and a contiguous particle-index array (num_particles). No fixed buckets / overflow.
        let cell_count = Self::storage(
            &device,
            "xpbd-cellcount",
            (num_cells as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        let cell_start = Self::storage(
            &device,
            "xpbd-cellstart",
            (num_cells as u64 + 1) * 4,
            wgpu::BufferUsages::empty(),
        );
        let sorted_indices = Self::storage(
            &device,
            "xpbd-sorted",
            (capacity.max(1) as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        // Static SDF geometry (binding 19, read-only). Built-time-immutable; an empty scene gets a
        // 1-element dummy (params.num_solids = 0 makes the union skip it). Bound by the kernels that
        // query the geometry — the position-solve collision passes and `finalize` (wall velocity
        // response). Each stays within the 8-storage-buffer budget (finalize is at 7).
        let mut packed = pack_solids(&scene.solids);
        if packed.is_empty() {
            packed.push(Primitive::zeroed());
        }
        let solids = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("xpbd-solids"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Per-particle chem/thermal state (Phase 1.5), one vec4 per particle: grain lanes
        // (s_f, s_s, T_g, _), water lanes (c, T_w, _, _). `chem` is the live buffer (binding 20,
        // readback for yield/TDS + the snapshot copy); `chem_frozen` (binding 22) is the per-substep
        // snapshot the dissolution/thermal passes read while writing the live buffer (no read/write
        // race). Bound only by the chem passes (U5/U6); other kernels never touch it. Seeded per
        // species: grains carry the soluble dose split into the two pools, water starts at c=0, pour T.
        let initial_chem = seed_chem(&phases, mats);
        // Yield denominator: the total dry-coffee dose = grain_mass · n_grain (real-coffee
        // convention; yield ≤ soluble_fraction). Cup region for TDS: the scene's catch cylinder
        // (if any), scoping TDS to pooled cup water.
        let n_grain = phases.iter().filter(|&&ph| ph == 1).count() as f32;
        let total_dose = mats.grain_mass * n_grain;
        let cup = scene.solids.iter().find_map(|p| match p.kind {
            crate::utils::sdf::SolidKind::Cylinder {
                center,
                floor_y,
                rim_y,
                radius,
            } => Some(CupRegion {
                cx: center.x,
                cz: center.z,
                radius,
                y_min: floor_y,
                y_max: rim_y,
            }),
            _ => None,
        });
        let chem = Arc::new(Self::storage(
            &device,
            "xpbd-chem",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        queue.write_buffer(&chem, 0, bytemuck::cast_slice(&initial_chem));
        let chem_frozen = Self::storage(
            &device,
            "xpbd-chem-frozen",
            vec4,
            wgpu::BufferUsages::empty(),
        );
        // Cell-order reorder scratch: gather targets for the water-loop locality reorder. COPY_SRC so
        // each is copied back into its live buffer after the gather.
        let pos_scratch = Self::storage(
            &device,
            "xpbd-pos-scratch",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let pred_scratch = Self::storage(
            &device,
            "xpbd-pred-scratch",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let vel_scratch = Self::storage(
            &device,
            "xpbd-vel-scratch",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let chem_scratch = Self::storage(
            &device,
            "xpbd-chem-scratch",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let phase_scratch = Self::storage(
            &device,
            "xpbd-phase-scratch",
            f32s,
            wgpu::BufferUsages::COPY_SRC,
        );
        // Per-particle dissolution scratch (binding 21), one vec2<f32> each: grain → (N_w, flux_g),
        // water → (N_g, _). Written by diss_count, read by both transfer passes. Bound only by the
        // dissolution passes (U5).
        let diss_neighbors = Self::storage(
            &device,
            "xpbd-diss-neighbors",
            n * 8,
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

        // WGSL has no imports: assemble the one module from the concern files. `common` declares
        // Params/Status/bindings + shared kernels; `water`/`bed`/`coupling`/`wetting` add the
        // per-species + interphase solves. Module-scope declarations are order-independent.
        let shader_src = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("common.wgsl"),
            include_str!("water.wgsl"),
            include_str!("bed.wgsl"),
            include_str!("coupling.wgsl"),
            include_str!("wetting.wgsl"),
            include_str!("extraction.wgsl"),
            include_str!("fines.wgsl"),
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
            grid_count: make("grid_count"),
            grid_scan: make("grid_scan"),
            grid_scatter: make("grid_scatter"),
            grid_reorder_a: make("grid_reorder_a"),
            grid_reorder_b: make("grid_reorder_b"),
            grid_reorder_identity: make("grid_reorder_identity"),
            compute_boundary: make("compute_boundary"),
            compute_lambda: make("compute_lambda"),
            residual_reduce: make("residual_reduce"),
            compute_dp: make("compute_dp"),
            bed_project: make("bed_project"),
            compute_fractions: make("compute_fractions"),
            exclude_water: make("exclude_water"),
            exclude_grain: make("exclude_grain"),
            compute_coupling_scale: make("compute_coupling_scale"),
            drag_water: make("drag_water"),
            drag_grain: make("drag_grain"),
            buoyancy_grain: make("buoyancy_grain"),
            buoyancy_water: make("buoyancy_water"),
            apply_drag_pred: make("apply_drag_pred"),
            wet_count: make("wet_count"),
            wet_water: make("wet_water"),
            wet_grain: make("wet_grain"),
            diss_count: make("diss_count"),
            dissolve_grain: make("dissolve_grain"),
            dissolve_water: make("dissolve_water"),
            thermal_exchange: make("thermal_exchange"),
            fines_count: make("fines_count"),
            fines_grain: make("fines_grain"),
            fines_water: make("fines_water"),
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
                    (12, &normal_impulse),
                    (14, &fluid_impulse),
                ],
            ),
            grid_clear: bg(
                &pipelines.grid_clear,
                &[(0, &params_buf), (18, &cell_count)],
            ),
            grid_count: bg(
                &pipelines.grid_count,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (10, &status),
                    (18, &cell_count),
                ],
            ),
            grid_scan: bg(
                &pipelines.grid_scan,
                &[(0, &params_buf), (8, &cell_start), (18, &cell_count)],
            ),
            grid_scatter: bg(
                &pipelines.grid_scatter,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (18, &cell_count),
                ],
            ),
            grid_reorder_a: bg(
                &pipelines.grid_reorder_a,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &pred),
                    (3, &vel),
                    (9, &sorted_indices),
                    (23, &pos_scratch),
                    (24, &pred_scratch),
                    (25, &vel_scratch),
                ],
            ),
            grid_reorder_b: bg(
                &pipelines.grid_reorder_b,
                &[
                    (0, &params_buf),
                    (9, &sorted_indices),
                    (11, &phase),
                    (20, &chem),
                    (26, &chem_scratch),
                    (27, &phase_scratch),
                ],
            ),
            grid_reorder_identity: bg(
                &pipelines.grid_reorder_identity,
                &[(0, &params_buf), (9, &sorted_indices)],
            ),
            compute_boundary: bg(
                &pipelines.compute_boundary,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (7, &c_residual),
                    (10, &status),
                    (11, &phase),
                    (19, &solids),
                ],
            ),
            compute_lambda: bg(
                &pipelines.compute_lambda,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (5, &lambda),
                    (7, &c_residual),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (10, &status),
                    (11, &phase),
                    (13, &alpha_s),
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
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (10, &status),
                    (11, &phase),
                    (13, &alpha_s),
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
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (12, &normal_impulse),
                ],
            ),
            compute_fractions: bg(
                &pipelines.compute_fractions,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (13, &alpha_s),
                ],
            ),
            exclude_water: bg(
                &pipelines.exclude_water,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (6, &dp),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (10, &status),
                    (11, &phase),
                ],
            ),
            exclude_grain: bg(
                &pipelines.exclude_grain,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (6, &dp),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (10, &status),
                    (11, &phase),
                ],
            ),
            compute_coupling_scale: bg(
                &pipelines.compute_coupling_scale,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (13, &alpha_s),
                    (16, &coupling_scale),
                ],
            ),
            drag_water: bg(
                &pipelines.drag_water,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (3, &vel),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (14, &fluid_impulse),
                    (15, &vel_frozen),
                    (16, &coupling_scale),
                ],
            ),
            drag_grain: bg(
                &pipelines.drag_grain,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (3, &vel),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (14, &fluid_impulse),
                    (15, &vel_frozen),
                    (16, &coupling_scale),
                ],
            ),
            buoyancy_grain: bg(
                &pipelines.buoyancy_grain,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (3, &vel),
                    (5, &lambda),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (14, &fluid_impulse),
                    (15, &vel_frozen),
                ],
            ),
            buoyancy_water: bg(
                &pipelines.buoyancy_water,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (3, &vel),
                    (5, &lambda),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (15, &vel_frozen),
                ],
            ),
            apply_drag_pred: bg(
                &pipelines.apply_drag_pred,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (3, &vel),
                    (11, &phase),
                    (15, &vel_frozen),
                    (19, &solids),
                ],
            ),
            wet_count: bg(
                &pipelines.wet_count,
                &[
                    (0, &params_buf),
                    (2, &pred),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (17, &wet_count),
                ],
            ),
            wet_water: bg(
                &pipelines.wet_water,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &pred),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (17, &wet_count),
                ],
            ),
            wet_grain: bg(
                &pipelines.wet_grain,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &pred),
                    (3, &vel),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (17, &wet_count),
                ],
            ),
            diss_count: bg(
                &pipelines.diss_count,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (3, &vel),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (21, &diss_neighbors),
                ],
            ),
            dissolve_grain: bg(
                &pipelines.dissolve_grain,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (20, &chem),
                    (21, &diss_neighbors),
                    (22, &chem_frozen),
                ],
            ),
            dissolve_water: bg(
                &pipelines.dissolve_water,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (20, &chem),
                    (21, &diss_neighbors),
                    (22, &chem_frozen),
                ],
            ),
            thermal_exchange: bg(
                &pipelines.thermal_exchange,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (20, &chem),
                    (22, &chem_frozen),
                ],
            ),
            // Fines transfer (Phase 6): reuses the wetting neighbor-count buffer (binding 17) — the
            // wetting block has finished for the substep. Count is eligibility-only (5 storage);
            // the transfers read live vel for the per-pair flux and write only chem.w (8 storage).
            fines_count: bg(
                &pipelines.fines_count,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (17, &wet_count),
                ],
            ),
            fines_grain: bg(
                &pipelines.fines_grain,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (3, &vel),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (17, &wet_count),
                    (20, &chem),
                    (22, &chem_frozen),
                ],
            ),
            fines_water: bg(
                &pipelines.fines_water,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (3, &vel),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                    (17, &wet_count),
                    (20, &chem),
                    (22, &chem_frozen),
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
                    (19, &solids),
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
                    (11, &phase),
                    (14, &fluid_impulse),
                    (19, &solids),
                ],
            ),
            xsph: bg(
                &pipelines.xsph,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (3, &vel),
                    (4, &vel_smoothed),
                    (8, &cell_start),
                    (9, &sorted_indices),
                    (11, &phase),
                ],
            ),
        };

        let ts = if gpu.timestamps_supported {
            // Generous upper bound: water density loop (≈6 passes/iter) + bed contact loop
            // (≈4 passes/iter) + coupling/finalize overhead; clamped to the query-set cap.
            // Water loop is ≈7 passes/iter with the boundary pass (solid scenes); 6 covers it loosely.
            // Baseline covers coupling/finalize + the wetting/extraction/fines blocks (~8 passes
            // each, gated); water loop ≈7 passes/iter, bed ≈5.
            let passes_per_step = 20 + 7 * water_iters + 5 * bed_iters;
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
            capacity,
            active_count,
            num_cells,
            substeps: cfg.substeps.max(1),
            has_water,
            has_grain,
            has_solids,
            water_iters,
            water_regrid,
            bed_iters,
            bed_regrid,
            drag_subiters,
            params_buf,
            pos,
            vel,
            pred,
            vel_smoothed,
            vel_frozen,
            lambda,
            phase,
            status,
            status_readback,
            pos_readback,
            chem,
            chem_frozen,
            pos_scratch,
            pred_scratch,
            vel_scratch,
            chem_scratch,
            phase_scratch,
            pipelines,
            bind_groups,
            ts,
            dispatches: 0,
            cached_diag: XpbdDiagnostics::default(),
            cached_passes: Vec::new(),
            total_dose,
            cup,
            cached_yield: 0.0,
            cached_tds: 0.0,
            inflow: Inflow {
                nozzle_radius: cfg.nozzle_radius,
                discharge_coeff: cfg.discharge_coeff,
                spacing: s,
                v_w,
                pour_t: mats.pour_t,
                accumulator: 0.0,
                axial: 0.0,
                last_exit_speed: 0.0,
                cursor: 0,
                emitted_mass: 0.0,
            },
            initial_positions: positions,
            phase_mirror: phases.clone(),
            initial_phases: phases,
            initial_chem,
        }
    }

    fn reset(&mut self, _scene: &Scene) {
        self.seed();
    }

    fn step(&mut self, dt: f32, input: &EmissionInput) {
        // Pour emission first (grows active_count for this frame); no-op when not pouring. Uses the
        // full frame dt (flow integrated once per frame, then simulated by all substeps).
        self.emit(input, dt);
        // Pool invariant: the live set never exceeds the allocated capacity (emit() clamps to it).
        debug_assert!(self.active_count <= self.capacity);
        self.params.dt = dt / self.substeps as f32;
        // The live set the kernels guard against (i >= particle_count returns) — refreshed each frame
        // so pour-grown active_count takes effect (emit() above ran first).
        self.params.particle_count = self.active_count;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let np = groups(self.active_count);
        let nc = groups(self.num_cells);
        let mut cursor = 0u32;
        let mut labels: Vec<String> = Vec::new();
        let mut dispatches = 0u32;

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("xpbd-frame"),
            });

        // Pass sequence is chosen from which species are present. Water runs the density solve
        // (+ grain exclusion when mixed); grain runs the contact subcycle; mixed runs both, with
        // the bed contact AFTER the water/fluid passes so grains respond. Single-species scenes
        // keep their adaptive early-exit (residual_reduce); mixed runs fixed iteration counts.
        let mixed = self.has_water && self.has_grain;
        let p = &self.pipelines;
        let b = &self.bind_groups;
        let ts_ref = self.ts.as_ref();
        {
            let mut pass = |enc: &mut wgpu::CommandEncoder,
                            pipe: &wgpu::ComputePipeline,
                            bg: &wgpu::BindGroup,
                            label: &str,
                            groups: u32| {
                dispatch_pass(
                    enc,
                    pipe,
                    bg,
                    ts_ref,
                    &mut cursor,
                    &mut labels,
                    label,
                    groups,
                    &mut dispatches,
                );
            };
            for _ in 0..self.substeps {
                pass(&mut enc, &p.predict, &b.predict, "predict", np);

                if self.has_water {
                    for it in 0..self.water_iters {
                        if it % self.water_regrid == 0 {
                            pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                            pass(&mut enc, &p.grid_count, &b.grid_count, "grid_count", np);
                            pass(&mut enc, &p.grid_scan, &b.grid_scan, "grid_scan", 1);
                            pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                            pass(
                                &mut enc,
                                &p.grid_scatter,
                                &b.grid_scatter,
                                "grid_scatter",
                                np,
                            );
                            // Cell-order reorder (memory locality): gather the persistent payload
                            // (pos/pred/vel/chem/phase) into cell order, then set sorted_indices to
                            // identity so the density gathers below read contiguous memory. Water-loop
                            // only — later blocks inherit this cell-sorted layout (reads stay
                            // consistent, gathers stay near-contiguous). Single shared mapping
                            // (sorted_indices read-only until the identity write) → no chunk hazard.
                            let nbytes_v4 = (self.active_count as u64) * 16;
                            let nbytes_u32 = (self.active_count as u64) * 4;
                            pass(
                                &mut enc,
                                &p.grid_reorder_a,
                                &b.grid_reorder_a,
                                "grid_reorder_a",
                                np,
                            );
                            enc.copy_buffer_to_buffer(
                                &self.pos_scratch,
                                0,
                                &self.pos,
                                0,
                                nbytes_v4,
                            );
                            enc.copy_buffer_to_buffer(
                                &self.pred_scratch,
                                0,
                                &self.pred,
                                0,
                                nbytes_v4,
                            );
                            enc.copy_buffer_to_buffer(
                                &self.vel_scratch,
                                0,
                                &self.vel,
                                0,
                                nbytes_v4,
                            );
                            pass(
                                &mut enc,
                                &p.grid_reorder_b,
                                &b.grid_reorder_b,
                                "grid_reorder_b",
                                np,
                            );
                            enc.copy_buffer_to_buffer(
                                &self.chem_scratch,
                                0,
                                &self.chem,
                                0,
                                nbytes_v4,
                            );
                            enc.copy_buffer_to_buffer(
                                &self.phase_scratch,
                                0,
                                &self.phase,
                                0,
                                nbytes_u32,
                            );
                            pass(
                                &mut enc,
                                &p.grid_reorder_identity,
                                &b.grid_reorder_identity,
                                "grid_reorder_identity",
                                np,
                            );
                        }
                        if mixed {
                            // Solid fraction (live from current pred) → pore-modulated water target.
                            pass(
                                &mut enc,
                                &p.compute_fractions,
                                &b.compute_fractions,
                                "compute_fractions",
                                np,
                            );
                        }
                        if self.has_solids {
                            // Wall density compensation into c_residual (consumed by compute_lambda,
                            // then overwritten with the convergence residual). Solid scenes only.
                            pass(
                                &mut enc,
                                &p.compute_boundary,
                                &b.compute_boundary,
                                "compute_boundary",
                                np,
                            );
                        }
                        pass(
                            &mut enc,
                            &p.compute_lambda,
                            &b.compute_lambda,
                            "compute_lambda",
                            np,
                        );
                        pass(&mut enc, &p.compute_dp, &b.compute_dp, "compute_dp", np);
                        if mixed {
                            // Grain exclusion (A.2), two-way: water out of grain bodies + reaction.
                            pass(
                                &mut enc,
                                &p.exclude_water,
                                &b.exclude_water,
                                "exclude_water",
                                np,
                            );
                            pass(
                                &mut enc,
                                &p.exclude_grain,
                                &b.exclude_grain,
                                "exclude_grain",
                                np,
                            );
                        }
                        pass(&mut enc, &p.apply_dp, &b.apply_dp, "apply_dp", np);
                        // Adaptive early-exit only for single-species water (mixed runs fixed iters).
                        if !mixed {
                            pass(
                                &mut enc,
                                &p.residual_reduce,
                                &b.residual_reduce,
                                "residual_reduce",
                                1,
                            );
                        }
                    }
                }

                if mixed && self.drag_subiters > 0 {
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(&mut enc, &p.grid_count, &b.grid_count, "grid_count", np);
                    pass(&mut enc, &p.grid_scan, &b.grid_scan, "grid_scan", 1);
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(
                        &mut enc,
                        &p.grid_scatter,
                        &b.grid_scatter,
                        "grid_scatter",
                        np,
                    );
                    pass(
                        &mut enc,
                        &p.compute_coupling_scale,
                        &b.compute_coupling_scale,
                        "compute_coupling_scale",
                        np,
                    );
                    for _ in 0..self.drag_subiters {
                        enc.copy_buffer_to_buffer(
                            self.vel.as_ref(),
                            0,
                            &self.vel_frozen,
                            0,
                            (self.active_count as u64) * 16,
                        );
                        pass(&mut enc, &p.drag_water, &b.drag_water, "drag_water", np);
                        pass(&mut enc, &p.drag_grain, &b.drag_grain, "drag_grain", np);
                        pass(
                            &mut enc,
                            &p.apply_drag_pred,
                            &b.apply_drag_pred,
                            "apply_drag_pred",
                            np,
                        );
                    }
                }

                if mixed && self.params.buoyancy_scale > 0.0 {
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(&mut enc, &p.grid_count, &b.grid_count, "grid_count", np);
                    pass(&mut enc, &p.grid_scan, &b.grid_scan, "grid_scan", 1);
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(
                        &mut enc,
                        &p.grid_scatter,
                        &b.grid_scatter,
                        "grid_scatter",
                        np,
                    );
                    enc.copy_buffer_to_buffer(
                        self.vel.as_ref(),
                        0,
                        &self.vel_frozen,
                        0,
                        (self.active_count as u64) * 16,
                    );
                    pass(
                        &mut enc,
                        &p.buoyancy_grain,
                        &b.buoyancy_grain,
                        "buoyancy_grain",
                        np,
                    );
                    pass(
                        &mut enc,
                        &p.buoyancy_water,
                        &b.buoyancy_water,
                        "buoyancy_water",
                        np,
                    );
                    pass(
                        &mut enc,
                        &p.apply_drag_pred,
                        &b.apply_drag_pred,
                        "apply_drag_pred",
                        np,
                    );
                }

                if self.has_grain {
                    for it in 0..self.bed_iters {
                        if it % self.bed_regrid == 0 {
                            pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                            pass(&mut enc, &p.grid_count, &b.grid_count, "grid_count", np);
                            pass(&mut enc, &p.grid_scan, &b.grid_scan, "grid_scan", 1);
                            pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                            pass(
                                &mut enc,
                                &p.grid_scatter,
                                &b.grid_scatter,
                                "grid_scatter",
                                np,
                            );
                        }
                        pass(&mut enc, &p.bed_project, &b.bed_project, "bed_project", np);
                        pass(&mut enc, &p.apply_dp, &b.apply_dp, "apply_dp", np);
                        if !mixed {
                            pass(
                                &mut enc,
                                &p.residual_reduce,
                                &b.residual_reduce,
                                "residual_reduce",
                                1,
                            );
                        }
                    }
                }

                pass(&mut enc, &p.finalize, &b.finalize, "finalize", np);

                // XSPH viscosity is a fluid term; runs only when water is present.
                if self.has_water {
                    pass(&mut enc, &p.xsph, &b.xsph, "xsph", np);
                    enc.copy_buffer_to_buffer(
                        &self.vel_smoothed,
                        0,
                        &self.vel,
                        0,
                        (self.active_count as u64) * 16,
                    );
                }

                // Wetting / absorption (mixed scenes; opt-in via absorb_rate>0). Runs last in the
                // substep — after finalize/xsph so it reads final positions/velocities and its grain
                // momentum merge isn't clobbered by the xsph velocity copy. Reads the frozen pred.w
                // snapshot, writes the new moisture to pos.w; the next predict mirrors it.
                if mixed && self.params.k_abs > 0.0 {
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(&mut enc, &p.grid_count, &b.grid_count, "grid_count", np);
                    pass(&mut enc, &p.grid_scan, &b.grid_scan, "grid_scan", 1);
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(
                        &mut enc,
                        &p.grid_scatter,
                        &b.grid_scatter,
                        "grid_scatter",
                        np,
                    );
                    pass(&mut enc, &p.wet_count, &b.wet_count, "wet_count", np);
                    pass(&mut enc, &p.wet_water, &b.wet_water, "wet_water", np);
                    pass(&mut enc, &p.wet_grain, &b.wet_grain, "wet_grain", np);
                }

                // Extraction / dissolution (mixed scenes; opt-in via extract_rate>0). Wet grains
                // dissolve two-pool solute into overlapping water's concentration, conserving the
                // solute inventory exactly. Reads the frozen pred.w moisture (same snapshot wetting
                // used) + a frozen chem copy; writes only the live chem buffer. Rebuilds the grid
                // itself so it's independent of the wetting block's gate.
                if mixed && self.params.extract_rate > 0.0 {
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(&mut enc, &p.grid_count, &b.grid_count, "grid_count", np);
                    pass(&mut enc, &p.grid_scan, &b.grid_scan, "grid_scan", 1);
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(
                        &mut enc,
                        &p.grid_scatter,
                        &b.grid_scatter,
                        "grid_scatter",
                        np,
                    );
                    // Snapshot the live chem so both transfer passes read a frozen state while
                    // writing disjoint slots of the live buffer (race-free conservation).
                    enc.copy_buffer_to_buffer(
                        self.chem.as_ref(),
                        0,
                        &self.chem_frozen,
                        0,
                        (self.active_count as u64) * 16,
                    );
                    pass(&mut enc, &p.diss_count, &b.diss_count, "diss_count", np);
                    pass(
                        &mut enc,
                        &p.dissolve_grain,
                        &b.dissolve_grain,
                        "dissolve_grain",
                        np,
                    );
                    pass(
                        &mut enc,
                        &p.dissolve_water,
                        &b.dissolve_water,
                        "dissolve_water",
                        np,
                    );
                    // Thermal exchange + ambient loss (U6). Re-snapshot the post-dissolution chem so
                    // the pass preserves the updated c/pools (it writes only the T lane), then run on
                    // the still-valid grid. T evolved here feeds the NEXT substep's k_T (one-step lag).
                    enc.copy_buffer_to_buffer(
                        self.chem.as_ref(),
                        0,
                        &self.chem_frozen,
                        0,
                        (self.active_count as u64) * 16,
                    );
                    pass(
                        &mut enc,
                        &p.thermal_exchange,
                        &b.thermal_exchange,
                        "thermal_exchange",
                        np,
                    );
                }

                // Fines migration (mixed scenes; opt-in via fines_rate>0). Grains shed fines into
                // fast-flowing water and water deposits them where flow slackens — a massless,
                // volume-conserving scalar transfer that writes only chem.w (no velocity writes, so
                // momentum is untouched). Reads live post-wetting pos/pos.w + a frozen chem copy;
                // rebuilds the grid itself so it's independent of the wetting/extraction gates.
                if mixed && self.params.fines[0] > 0.0 {
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(&mut enc, &p.grid_count, &b.grid_count, "grid_count", np);
                    pass(&mut enc, &p.grid_scan, &b.grid_scan, "grid_scan", 1);
                    pass(&mut enc, &p.grid_clear, &b.grid_clear, "grid_clear", nc);
                    pass(
                        &mut enc,
                        &p.grid_scatter,
                        &b.grid_scatter,
                        "grid_scatter",
                        np,
                    );
                    // Snapshot live chem so both transfer passes read a frozen state while writing
                    // disjoint chem.w slots (race-free, conservation-safe).
                    enc.copy_buffer_to_buffer(
                        self.chem.as_ref(),
                        0,
                        &self.chem_frozen,
                        0,
                        (self.active_count as u64) * 16,
                    );
                    pass(&mut enc, &p.fines_count, &b.fines_count, "fines_count", np);
                    pass(&mut enc, &p.fines_grain, &b.fines_grain, "fines_grain", np);
                    pass(&mut enc, &p.fines_water, &b.fines_water, "fines_water", np);
                }
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
            particle_count: self.active_count,
            position: Some(Arc::clone(&self.pos)),
            velocity: Some(Arc::clone(&self.vel)),
            phase_tag: Some(Arc::clone(&self.phase)),
            // The chem buffer carries both lanes: water concentration `c` (.x) and temperature `T`
            // (.y). The renderer reads the relevant lane per field; both share the one buffer.
            concentration: Some(Arc::clone(&self.chem)),
            temperature: Some(Arc::clone(&self.chem)),
            ..Default::default()
        }
    }

    fn metrics(&self) -> Metrics {
        Metrics {
            particle_count: self.active_count,
            iteration_count: self.cached_diag.effective_iters,
            extraction_yield: self.cached_yield,
            tds: self.cached_tds,
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

    /// The V60 scene's cone-aware rejection seeds both species strictly inside their cavities.
    #[test]
    fn v60_seeds_inside_the_cavity() {
        let scene = Scene::v60();
        let (pos, phase) = seed_block(&scene, &Materials::default(), &Config::default());
        assert!(!pos.is_empty(), "v60 seeds particles");
        let (mut nw, mut ng) = (0u32, 0u32);
        for (p, &ph) in pos.iter().zip(&phase) {
            let c =
                crate::utils::sdf::nearest(&scene.solids, glam::Vec3::new(p[0], p[1], p[2]), ph);
            assert!(
                c.signed >= 0.0,
                "seed (phase {ph}) at {:?} is outside its cavity: signed {}",
                [p[0], p[1], p[2]],
                c.signed
            );
            if ph == 0 {
                nw += 1
            } else {
                ng += 1
            }
        }
        assert!(
            nw > 0 && ng > 0,
            "both species seeded (water {nw}, grain {ng})"
        );
    }

    /// Existing AABB scenes carry no solids, so rejection is a no-op for them.
    #[test]
    fn aabb_scenes_have_no_solids() {
        for s in [
            Scene::default(),
            Scene::dam_break(),
            Scene::bed_drop(),
            Scene::dam_through_sand(),
            Scene::pour_over(),
        ] {
            assert!(s.solids.is_empty());
        }
    }
}

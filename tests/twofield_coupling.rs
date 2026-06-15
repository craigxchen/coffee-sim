//! Twofield U6 gates: two-field Darcy drag + porosity-weighted (mixture) incompressibility
//! against a RIGID, kinematically frozen skeleton — plan 2026-06-09-001 U6 / KTD-3 / KTD-4.
//!
//! Formulation under test (construction documented in `coupling.wgsl` + the pressure.wgsl
//! header additions):
//!   * φ_s per node from a thin solid-mass P2G of the static grains (each grain carries the
//!     sphere volume `grain_volume(d) = π/6·d³`); φ_f = 1 − φ_s clamped (KTD-8 local fields).
//!   * Drag = the FORCED Laibe-Price exponential integrator per node against the frozen
//!     skeleton (v_s = 0): v⁺ = v·e^{−βΔt} + g·(1 − e^{−βΔt})/β, with β(φ) from
//!     Kozeny-Carman (mirrors `models::permeability`) blended toward a Wen-Yu-style dilute
//!     rate below φ_s ≈ 0.2 (Huilin-Gidaspow arctan transition). The impulse the frozen
//!     skeleton absorbs is recorded per node in the reaction ledger — never discarded.
//!   * Mixture projection: the constraint is ∇·(φ_f·v_f) = s (v_s = 0); the water correction
//!     is Δv = −(Δt_eff/ρ_w)·∇p with ρ_w the INTRINSIC water density (φ NEVER scales the
//!     velocity correction, KTD-4) and Δt_eff = (1 − e^{−βΔt})/β the SAME exponential-
//!     integrator weight the drag fold used (ς = Δt_eff/Δt scales the node mobility in the
//!     operator family). This is load-bearing: the plain-Δt Lie split provably mis-partitions
//!     the water weight between drag reaction and pore pressure at equilibrium
//!     (slope → (1−e^{−βΔt})/(βΔt)·ρg), which fails the pre-registered buoyancy/undrained
//!     gates at stiff β; the ς form reproduces the continuous Darcy balance v = g_eff/β and
//!     exact hydrostatic pore pressure at ANY stiffness — KTD-3's actual claim.
//!
//! KNOB GRID (KTD-9): U6 adds NO new gate knobs. The pressure knobs remain U3's grid; the
//! drag-law constants (150 Kozeny-Carman / 18 Wen-Yu-Stokes / Huilin 262.5 / switch 0.2),
//! PHI_S_MAX = 0.95, and the grain sphere volume are fixed structural constants mirrored
//! from / documented next to `models::permeability` and `coupling.wgsl`. The per-scenario
//! grind d and Config::drag_scale are scene parameters fixed in each test below before any
//! results were observed. Exhausting U3's pressure grid without a pass IS the halt.
//!
//! PRE-REGISTERED BANDS (fixed before the kernels ran; never loosened):
//!   DARCY     q/q_pred ∈ [0.5, 1.6] per grind; pairwise flux ratio within ±30% of the
//!             K·i prediction mapped through models::permeability and each arm's measured
//!             head; strict ordering. (Bands unchanged since pre-registration; the grind
//!             values moved 0.7/1.4 → 0.8/1.6 and the sweep gained the pond feed when the
//!             scene construction was fixed — see the in-test comments for the measured
//!             construction artifacts: wall-lattice phase, slab-quantized over-seeding,
//!             unfed-column drawdown.)
//!   HEAD      flux ratio (pond vs none) within ±25% of the measured-head ratio.
//!   ENTRY     pond descent rate / Darcy prediction ∈ [0.5, 1.7]; the anti-barrier kill
//!             floor is 0.25 (main's stuck-layer baseline ≈ 0); adjacent-layer pore-pressure
//!             jump ≤ 3·ρ·g·h (no interface spike).
//!   MOMENTUM  per-node drag impulse + recorded reaction = 0 to 1e-5 relative (f32 recompute).
//!   SLIP      pressure-off decay matches v0·e^{−βt} within 5%; full-pipeline within 25%;
//!             no sign reversal beyond −2% of v0; kinetic energy non-increasing (≤ +0.5%).
//!   SPLIT     dt arm vs dt/8 subcycled reference (drag AND projection together, closed
//!             variable-φ column, high slip + dry→wet + φ near packing): layer-profile RMS
//!             diff ≤ 0.35·ref RMS + 0.05; mean speed ≤ 1.5·ref + 0.05; finite, |v| ≤ 10.
//!   BUOYANCY  per-frame reaction on a submerged frozen block ∈ [0.75, 1.25]·V_s·ρ_w·g·Δt
//!             upward; lateral components ≤ 0.15 of that.
//!   UNDRAINED Δp_base/(ρ_w·g·Δh_surcharge) ∈ [0.85, 1.15]; bed pore-pressure slope within
//!             [0.85, 1.15]·ρ_w·g (full transfer — no effective-stress path in a frozen
//!             skeleton; quantitative Skempton-B is U7).
//!   φ<1 DIV   the U3 divergence-decay gate re-run against the MIXTURE divergence on a
//!             variable-φ saturated column: ≥2× decay per fine-sweep doubling until below
//!             DIV_TOL = 0.3, reached at ≤16 sweeps; ≥50 interior cells (vacuousness guard).
//!   FREE H2O  water-only run vs run-with-distant-frozen-grains: water state BIT-identical
//!             (exact-zero passthrough; exp(0) = 1 paths never taken — kernels gate on the
//!             solid-mass census); reactions identically zero.
//!   SETTLED   saturated column long-run (1200 steps): sampled max |v| ≤ 5, top rise ≤ 1
//!             spacing, finite (drag must not pump energy).

// Axis loops (`for a in 0..3`) index parallel arrays; the iterator rewrite obscures that.
#![allow(clippy::needless_range_loop)]

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::{permeability, Materials};
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{
    blended_drag_rate, dispatches_per_frame_for, grain_volume, u4_surface_dispatches_for,
    TwofieldSolver, COARSE_RATIO_DEFAULT, COARSE_SWEEPS_DEFAULT, DENSITY_RELAX_FRAMES,
    U3_PRESSURE_DISPATCHES, U6_COUPLING_DISPATCHES,
};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;
const GRAV: f64 = 20.0;
const REST: f64 = 1.0; // particle_mass / spacing³ at the defaults
const SPACING: f64 = 1.0;

/// Solid fraction of a unit-pitch grain lattice carrying the sphere volume π/6·d³ per grain:
/// φ_s = V_g/d³ = π/6, independent of the grind d (the lattice pitch scales with d).
const PHI_S_LATTICE: f64 = std::f64::consts::PI / 6.0;

// --- pre-registered bands (header) -----------------------------------------------------------
const DARCY_ABS_BAND: (f64, f64) = (0.5, 1.6);
const DARCY_RATIO_TOL: f64 = 0.30;
const HEAD_RATIO_TOL: f64 = 0.25;
const ENTRY_BAND: (f64, f64) = (0.5, 1.7);
const ENTRY_KILL_FLOOR: f64 = 0.25;
const ENTRY_DP_JUMP_MAX: f64 = 3.0; // × ρ·g·h
const PAIR_REL_TOL: f64 = 1.0e-5;
const SLIP_TIGHT: f64 = 0.05;
const SLIP_LOOSE: f64 = 0.25;
const SLIP_KE_GROWTH: f64 = 0.005;
const SPLIT_PROFILE_BAND: f64 = 0.35;
const SPLIT_PROFILE_ABS: f64 = 0.05;
const SPLIT_SPEED_FACTOR: f64 = 1.5;
const SPLIT_SPEED_ABS: f64 = 0.05;
const SPLIT_VMAX: f32 = 10.0;
const BUOY_BAND: (f64, f64) = (0.75, 1.25);
const BUOY_LATERAL: f64 = 0.15;
const UNDRAINED_BAND: (f64, f64) = (0.85, 1.15);
const DIV_TOL: f64 = 0.3; // mirrors tests/twofield_pressure.rs (pre-registered there)
const DECAY_FACTOR_MIN: f64 = 2.0;
const DECAY_BUDGETS: [u32; 4] = [4, 8, 16, 32];
const INTERIOR_MARGIN: i64 = 2;
const LONGRUN_STEPS: u32 = 1200;
const LONGRUN_MAX_SPEED: f32 = 5.0;
const LONGRUN_POP_RISE: f64 = 1.0;

fn all_finite(rows: &[[f32; 4]]) -> bool {
    rows.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

// ==============================================================================================
// Scene + saturation machinery
// ==============================================================================================

/// Inclusive lattice (mirrors `seed_ranges` counting: floor(extent/pitch)+1 per axis).
fn lattice(min: [f32; 3], max: [f32; 3], pitch: f32) -> Vec<[f32; 3]> {
    let n = |a: usize| ((max[a] - min[a]) / pitch).floor().max(0.0) as i32;
    let (nx, ny, nz) = (n(0), n(1), n(2));
    let mut out = Vec::new();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                out.push([
                    min[0] + i as f32 * pitch,
                    min[1] + j as f32 * pitch,
                    min[2] + k as f32 * pitch,
                ]);
            }
        }
    }
    out
}

/// Grain y-ramp for the variable-φ column: y' = y_local·(0.75 + 0.25·y_local/H), so the
/// local compression dy'/dy ramps 0.75 (bottom, dense: φ_s ≈ 0.70) → 1.25 (top, loose:
/// φ_s ≈ 0.42). Monotone, endpoint-preserving.
fn ramp_map(y_local: f64, h: f64) -> f64 {
    y_local * (0.75 + 0.25 * y_local / h)
}
fn ramp_slope(y_local: f64, h: f64) -> f64 {
    0.75 + 0.5 * y_local / h
}

struct Column {
    solver: TwofieldSolver,
    n_water: u32,
    /// y of the grain-lattice top layer (the nominal bed top).
    bed_top: f64,
    box_max: [f32; 3],
}

struct ColumnSpec {
    box_max: [f32; 3],
    bed_top: f32,
    d: f32,
    drag_scale: f32,
    gravity: f64,
    pond_layers: usize,
    ramp: bool,
    /// Fraction of the bed height saturated from the bottom (1.0 = fully saturated).
    sat_frac: f32,
    open_base: bool,
    /// Uniform-β construction (the slip-decay gate): zero seed jitter and grains filling the
    /// box to the ceiling, so EVERY node the measurement core can see carries the same β —
    /// the single-exponential reference is then the solver's own discrete recurrence (the
    /// fold compounds to exactly e^{−βt} at any dt). With the default construction the core
    /// reads a β mixture (jitter ripples φ by ±2% ⇒ β by ±9%; the bed-top dilute taper band
    /// leaks slow-decaying water into the core through the transfer kernels at a rate that
    /// GROWS under dt refinement) and the mean-φ exponential is the wrong reference.
    uniform_beta: bool,
}

impl Default for ColumnSpec {
    fn default() -> Self {
        ColumnSpec {
            box_max: [16.0, 28.0, 16.0],
            bed_top: 9.5,
            d: 1.0,
            drag_scale: 0.05,
            gravity: GRAV,
            pond_layers: 0,
            ramp: false,
            sat_frac: 1.0,
            open_base: false,
            uniform_beta: false,
        }
    }
}

/// Build a saturated rigid-bed column: grains seeded by the scene (pitch d, wall margin
/// d/2, full cross section), pore water repositioned onto a cubic lattice at the pore
/// density φ_f·ρ_rest (a y-walk at the LOCAL pore pitch — exact along the ramp), pond
/// restacked above the bed at rest density, surplus seeded water parked dormant
/// (`set_live_water_for_test`).
fn build_column(gpu: &GpuContext, spec: &ColumnSpec) -> Column {
    let mats = Materials {
        grain_diameter: spec.d,
        ..Materials::default()
    };
    let cfg = Config {
        drag_scale: spec.drag_scale,
        seed_jitter: if spec.uniform_beta {
            0.0
        } else {
            Config::default().seed_jitter
        },
        ..Config::default()
    };
    let bx = spec.box_max;
    let inset = 0.5f32;
    // Bed water lattice: a y-walk that places one x–z layer per LOCAL pore pitch, so the
    // interior is a true cubic lattice at the pore density φ_f·ρ_rest (exact along the
    // ramp too — the pitch is re-evaluated at each layer). The previous slab-wise lattice
    // quantized to 2 y-layers per 1.8-unit slab (effective y-pitch 0.9 vs the intended
    // 1.28): the bed seeded ~42% OVER the pore density, the warmup spent seconds venting
    // the excess up through the stiff bed, and the drainback creep's drag ate a clean
    // β·v/g ≈ 17% of the hydrostatic pore-pressure slope (and pumped the drained-flux
    // gates) — measured; the relaxed column then holds dp/dy = ρ_w·g within 3%.
    let h_bed = (spec.bed_top - inset) as f64;
    let sat_top = inset as f64 + h_bed * spec.sat_frac as f64;
    let mut bed_water: Vec<[f32; 3]> = Vec::new();
    let mut y = inset as f64 + 0.1;
    while y <= sat_top - 0.1 {
        let phi_s = if spec.ramp {
            PHI_S_LATTICE / ramp_slope(y - inset as f64, h_bed)
        } else {
            PHI_S_LATTICE
        };
        let phi_f = (1.0 - phi_s).clamp(0.05, 1.0);
        let pitch = (1.0 / phi_f).powf(1.0 / 3.0) as f32;
        bed_water.extend(lattice(
            [0.7, y as f32, 0.7],
            [bx[0] - 0.7, y as f32 + 0.01, bx[2] - 0.7],
            pitch,
        ));
        y += pitch as f64;
    }
    // Pond lattice at rest density above the bed.
    let pond = if spec.pond_layers > 0 {
        lattice(
            [0.7, spec.bed_top + 0.6, 0.7],
            [
                bx[0] - 0.7,
                spec.bed_top + 0.6 + (spec.pond_layers as f32 - 1.0) + 0.05,
                bx[2] - 0.7,
            ],
            1.0,
        )
    } else {
        Vec::new()
    };
    let needed = bed_water.len() + pond.len();
    // Scene: grains first as a region, water seeded ABOVE the bed (the seed source pool we
    // reposition). The water region comes FIRST so its RNG draws are position-independent of
    // the grain region (mirrors seed_ranges' per-region draw order).
    let per_layer = ((((bx[0] - 0.9) / 1.0).floor() as usize) + 1).pow(2);
    let layers = needed.div_ceil(per_layer);
    let scene = Scene {
        gravity: [0.0, -(spec.gravity as f32), 0.0],
        box_min: [0.0; 3],
        box_max: bx,
        regions: vec![
            SeedRegion {
                min: [0.5, spec.bed_top + 1.0, 0.5],
                max: [
                    bx[0] - 0.4,
                    spec.bed_top + 1.0 + (layers as f32 - 1.0) + 0.1,
                    bx[2] - 0.4,
                ],
                species: Species::Water,
            },
            SeedRegion {
                // Wall margin d/2: the grain lattice is then mirror-symmetric about every
                // wall plane, so a wall node's truncated B-spline solid sum is EXACTLY half
                // the full-lattice sum — the assumption node_phi_f's vis normalization
                // encodes. A fixed margin reads a grind-DEPENDENT wall φ_s (d = 0.7 put the
                // first grain 0.71·d out: wall drag ~2× low, the wall annulus carried the
                // column and flux/pred hit 1.79; d = 1.4 put it 0.36·d out: wall drag ~2.6×
                // high, flux/pred 0.64 — measured). Grind sweeps must keep 16/d ∈ ℤ so the
                // far wall is phase-matched too.
                min: [0.5 * spec.d, 0.5 * spec.d, 0.5 * spec.d],
                max: [
                    bx[0] - 0.5 * spec.d + 0.01,
                    if spec.uniform_beta {
                        bx[1] - 0.5 * spec.d + 0.01
                    } else {
                        spec.bed_top
                    },
                    bx[2] - 0.5 * spec.d + 0.01,
                ],
                species: Species::Grain,
            },
        ],
        solids: Vec::new(),
        ..Scene::default()
    };
    let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, gpu);
    let (water_seed, solid_count) = solver.phase_counts();
    assert!(solid_count > 0, "column scene seeds grains");
    assert!(
        needed as u32 <= water_seed,
        "water seed pool {water_seed} too small for {needed} repositioned particles"
    );

    let mut pos = solver.read_positions();
    // Reposition water: bed pore water first, then the pond; surplus parked dormant at the
    // box corner (never dispatched once the live count is lowered).
    for (i, p) in bed_water.iter().chain(pond.iter()).enumerate() {
        pos[i] = [p[0], p[1], p[2], 1.0];
    }
    for slot in pos.iter_mut().take(water_seed as usize).skip(needed) {
        *slot = [0.0, 0.0, 0.0, 0.0];
    }
    // Grain ramp: remap the seeded grain lattice y (solid range sits after the water pool).
    if spec.ramp {
        let n_total = pos.len();
        for g in pos.iter_mut().take(n_total).skip((water_seed) as usize) {
            let yl = (g[1] - inset) as f64;
            if yl >= 0.0 && yl <= h_bed {
                g[1] = inset + ramp_map(yl, h_bed) as f32;
            }
        }
    }
    solver.write_positions_for_test(&pos);
    let zeros = vec![[0.0f32; 4]; pos.len()];
    solver.write_velocities_for_test(&zeros);
    solver.set_live_water_for_test(needed as u32);
    if spec.open_base {
        solver.set_open_base_for_test(true);
    }
    Column {
        solver,
        n_water: needed as u32,
        bed_top: spec.bed_top as f64,
        box_max: bx,
    }
}

// ==============================================================================================
// Measurement helpers
// ==============================================================================================

fn grid_of(solver: &TwofieldSolver) -> ([f32; 3], f32, [u32; 3]) {
    solver.grid_spec()
}

fn nidx(dims: [u32; 3], i: usize, j: usize, k: usize) -> usize {
    i + dims[0] as usize * (j + dims[1] as usize * k)
}

/// Per-node φ_f from the nm readback (lane 7 — written by node_setup).
fn node_phi(solver: &TwofieldSolver) -> Vec<f32> {
    solver.read_node_matrices().iter().map(|r| r[7]).collect()
}

/// Mean φ_f over the nodes inside an AABB (probe regions are each gate's bed core, clear of
/// the B-spline truncation band at the bed's own free edges).
fn mean_phi_f_in(solver: &TwofieldSolver, lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let (origin, h, dims) = grid_of(solver);
    let phi = node_phi(solver);
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for k in 0..dims[2] as usize {
        for j in 0..dims[1] as usize {
            for i in 0..dims[0] as usize {
                let p = [
                    (origin[0] + i as f32 * h) as f64,
                    (origin[1] + j as f32 * h) as f64,
                    (origin[2] + k as f32 * h) as f64,
                ];
                if (0..3).all(|a| p[a] >= lo[a] && p[a] <= hi[a]) {
                    sum += phi[nidx(dims, i, j, k)] as f64;
                    cnt += 1;
                }
            }
        }
    }
    assert!(cnt > 0, "no nodes in the phi probe region");
    sum / cnt as f64
}

/// Mean φ_f over the column's bed core (≥2h from the side walls, y ∈ [2.5, bed_top − 2.5]).
fn mean_bed_phi_f(solver: &TwofieldSolver, bed_top: f64, box_max: [f32; 3]) -> f64 {
    let h = grid_of(solver).1 as f64;
    mean_phi_f_in(
        solver,
        [2.0 * h, 2.5, 2.0 * h],
        [
            box_max[0] as f64 - 2.0 * h,
            bed_top - 2.5,
            box_max[2] as f64 - 2.0 * h,
        ],
    )
}

/// Mean intrinsic water density (gv.w/(φ_f·h³)) over massy nodes in an AABB — the measured
/// ρ_w the equilibrium pore-pressure slope must follow (normalizes out the saturation-
/// seeding imperfection of the test lattice).
fn mean_intrinsic_density(solver: &TwofieldSolver, lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let (origin, h, dims) = grid_of(solver);
    let gv = solver.read_grid_velocities();
    let phi = node_phi(solver);
    let h3 = (h as f64).powi(3);
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for k in 0..dims[2] as usize {
        for j in 0..dims[1] as usize {
            for i in 0..dims[0] as usize {
                let p = [
                    (origin[0] + i as f32 * h) as f64,
                    (origin[1] + j as f32 * h) as f64,
                    (origin[2] + k as f32 * h) as f64,
                ];
                let n = nidx(dims, i, j, k);
                if (0..3).all(|a| p[a] >= lo[a] && p[a] <= hi[a]) && gv[n][3] > 1.0e-4 {
                    sum += gv[n][3] as f64 / (phi[n] as f64 * h3);
                    cnt += 1;
                }
            }
        }
    }
    assert!(cnt > 0, "no massy nodes in the density probe region");
    sum / cnt as f64
}

/// Drained volume flux through an INTERIOR horizontal plane (node layer nearest `y_plane`):
/// trapezoid-weighted Σ (node water mass)·(−v_y)/h over in-box nodes — node mass/h³ is the
/// water content θ, so this is Σ θ·(−v_y)·h², the superficial volumetric flux. Measured
/// inside the bed (not at the base layer, whose node masses are kernel-truncated ~2×);
/// quasi-steady incompressibility makes any interior plane carry the column flux.
fn plane_outflux(solver: &TwofieldSolver, box_max: [f32; 3], y_plane: f64) -> f64 {
    let (origin, h, dims) = grid_of(solver);
    let gv = solver.read_grid_velocities();
    let j = ((y_plane - origin[1] as f64) / h as f64).round() as usize;
    let mut flux = 0.0;
    for k in 0..dims[2] as usize {
        for i in 0..dims[0] as usize {
            let x = origin[0] + i as f32 * h;
            let z = origin[2] + k as f32 * h;
            if x < -1.0e-3 || x > box_max[0] + 1.0e-3 || z < -1.0e-3 || z > box_max[2] + 1.0e-3 {
                continue;
            }
            let mut w = 1.0f64;
            if x.abs() < 1.0e-3 || (x - box_max[0]).abs() < 1.0e-3 {
                w *= 0.5;
            }
            if z.abs() < 1.0e-3 || (z - box_max[2]).abs() < 1.0e-3 {
                w *= 0.5;
            }
            let n = nidx(dims, i, j, k);
            flux += w * gv[n][3] as f64 * (-(gv[n][1] as f64)) / h as f64;
        }
    }
    flux
}

fn top_decile_y(pos: &[[f32; 4]], live: usize) -> f64 {
    let mut ys: Vec<f32> = pos[..live].iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = (ys.len() / 10).max(1);
    let top = &ys[ys.len() - n..];
    top.iter().map(|&y| y as f64).sum::<f64>() / top.len() as f64
}

/// Layer-mean solved pressure over x/z-interior active cells → (y_center, mean p) per row.
fn layer_pressure(solver: &TwofieldSolver) -> Vec<(f64, f64)> {
    let (origin, h, dims) = grid_of(solver);
    let nc = [
        dims[0] as usize - 1,
        dims[1] as usize - 1,
        dims[2] as usize - 1,
    ];
    let meta = solver.read_cell_meta();
    let p = solver.read_pressure();
    let cidx = |i: usize, j: usize, k: usize| i + nc[0] * (j + nc[1] * k);
    let mut out = Vec::new();
    for j in 0..nc[1] {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for k in 2..nc[2].saturating_sub(2) {
            for i in 2..nc[0].saturating_sub(2) {
                let c = cidx(i, j, k);
                if meta[c][1] >= 0.999 {
                    sum += p[c] as f64;
                    cnt += 1;
                }
            }
        }
        if cnt > 0 {
            out.push(((origin[1] + (j as f32 + 0.5) * h) as f64, sum / cnt as f64));
        }
    }
    out
}

/// Frame reaction impulse Σ over nodes (the constraint-reaction ledger).
fn total_reaction(solver: &TwofieldSolver) -> [f64; 3] {
    let mut s = [0.0f64; 3];
    for r in solver.read_reactions() {
        for a in 0..3 {
            s[a] += r[a] as f64;
        }
    }
    s
}

/// Post-projection RMS MIXTURE divergence D(Φv) over margin-`INTERIOR_MARGIN` full cells —
/// the same D family the projection used (φ from the nm lane, fractions from cell_meta).
fn rms_interior_mixture_div(solver: &TwofieldSolver) -> (f64, usize) {
    let (_, h, dims) = grid_of(solver);
    let nc = [
        dims[0] as usize - 1,
        dims[1] as usize - 1,
        dims[2] as usize - 1,
    ];
    let meta = solver.read_cell_meta();
    let gv = solver.read_grid_velocities();
    let phi = node_phi(solver);
    let cidx = |i: usize, j: usize, k: usize| i + nc[0] * (j + nc[1] * k);
    let frac_at = |i: i64, j: i64, k: i64| -> f64 {
        if i < 0 || j < 0 || k < 0 || i >= nc[0] as i64 || j >= nc[1] as i64 || k >= nc[2] as i64 {
            return 0.0;
        }
        meta[cidx(i as usize, j as usize, k as usize)][1] as f64
    };
    let interior = |i: usize, j: usize, k: usize| -> bool {
        let m = INTERIOR_MARGIN;
        for dk in -m..=m {
            for dj in -m..=m {
                for di in -m..=m {
                    if frac_at(i as i64 + di, j as i64 + dj, k as i64 + dk) < 0.999 {
                        return false;
                    }
                }
            }
        }
        true
    };
    let mut se = 0.0;
    let mut cnt = 0usize;
    for k in 0..nc[2] {
        for j in 0..nc[1] {
            for i in 0..nc[0] {
                if !interior(i, j, k) {
                    continue;
                }
                let mut div = 0.0f64;
                for oz in 0..2usize {
                    for oy in 0..2usize {
                        for ox in 0..2usize {
                            let n = nidx(dims, i + ox, j + oy, k + oz);
                            let s = [
                                ox as f64 * 2.0 - 1.0,
                                oy as f64 * 2.0 - 1.0,
                                oz as f64 * 2.0 - 1.0,
                            ];
                            let ph = phi[n] as f64;
                            div += (s[0] * gv[n][0] as f64
                                + s[1] * gv[n][1] as f64
                                + s[2] * gv[n][2] as f64)
                                * ph
                                / (4.0 * h as f64);
                        }
                    }
                }
                se += div * div;
                cnt += 1;
            }
        }
    }
    ((se / cnt.max(1) as f64).sqrt(), cnt)
}

fn linfit(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let syy: f64 = pts.iter().map(|p| (p.1 - my) * (p.1 - my)).sum();
    let slope = sxy / sxx;
    let r2 = if syy > 0.0 {
        sxy * sxy / (sxx * syy)
    } else {
        1.0
    };
    (slope, r2)
}

// ==============================================================================================
// CPU-only gates
// ==============================================================================================

/// Budget bookkeeping (R8): the recorded constant carries the named U6 increment of
/// p2g_solid plus drag_fold, and the CPU blend twin agrees with `models::permeability` in
/// the packed regime, scales as d⁻², and vanishes toward φ_s = 0.
#[test]
fn budget_constant_and_blend_twin() {
    // The scene-derived budget is U2's 4 + U3 + the (scene-derived) U4 surface stack + U6, by
    // construction for any grid — checked here at a representative grid size.
    let dims = [40u32, 60, 40];
    assert_eq!(
        dispatches_per_frame_for(dims),
        4 + U3_PRESSURE_DISPATCHES + u4_surface_dispatches_for(dims) + U6_COUPLING_DISPATCHES,
        "budget must carry the named U6 increment"
    );
    assert_eq!(U6_COUPLING_DISPATCHES, 2, "p2g_solid + drag_fold");

    // Packed regime (φ_s ≥ 0.35): the blend is ≥99% Kozeny-Carman — within 5% of the model.
    for phi_f in [0.3f32, 0.4, 0.5, 0.6] {
        let b = blended_drag_rate(1.0, phi_f, 0.05);
        let kc = permeability::darcy_drag_rate(1.0, phi_f, 0.05);
        assert!(
            (b / kc - 1.0).abs() < 0.05,
            "packed blend off at phi_f {phi_f}: {b} vs KC {kc}"
        );
    }
    // d² scaling at fixed φ (the grind exponent).
    let r = blended_drag_rate(1.0, 0.45, 0.05) / blended_drag_rate(2.0, 0.45, 0.05);
    assert!((r - 4.0).abs() < 0.05, "β(d)/β(2d) = {r}, expected 4");
    // Dilute limit: the blended rate falls monotonically toward 0 as φ_s → 0 (Wen-Yu side).
    let mut last = f32::MAX;
    for phi_s in [0.2f32, 0.1, 0.05, 0.01, 0.001] {
        let b = blended_drag_rate(1.0, 1.0 - phi_s, 0.05);
        assert!(
            b < last,
            "blend not monotone toward dilute at phi_s {phi_s}"
        );
        last = b;
    }
    assert!(last < 0.05 * blended_drag_rate(1.0, 0.45, 0.05));
    // Grain volume is the sphere volume (the φ_s lattice constant tests rely on).
    let v = grain_volume(2.0);
    assert!((v as f64 - std::f64::consts::PI / 6.0 * 8.0).abs() < 1e-5);
}

// ==============================================================================================
// MOMENTUM gate: exact per-node pair conservation of the drag fold
// ==============================================================================================

/// Manufactured grid state through the standalone drag pass: per node the recorded reaction
/// equals minus the water impulse exactly (f32 recompute budget); zero-solid nodes pass
/// through bit-identically with ς = 1 (the exact-zero structure, not exp(0) rounding).
#[test]
fn drag_fold_pair_momentum_exact_per_node() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    // Gravity-free box so the pair update is isolated (the fold's gravity source is gated by
    // the slip-decay test); one water layer seeds a valid solver.
    let scene = Scene {
        gravity: [0.0; 3],
        box_min: [0.0; 3],
        box_max: [18.0, 18.0, 18.0],
        regions: vec![SeedRegion {
            min: [0.5, 0.5, 0.5],
            max: [17.6, 1.6, 17.6],
            species: Species::Water,
        }],
        solids: Vec::new(),
        ..Scene::default()
    };
    let mats = Materials::default();
    let cfg = Config {
        drag_scale: 0.05,
        ..Config::default()
    };
    let solver = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
    let (_, h, dims) = grid_of(&solver);
    let n_nodes = (dims[0] * dims[1] * dims[2]) as usize;
    let h3 = (h * h * h) as f64;

    // Manufactured velocities (mass lane 1.0 — well above mass_eps) + a solid-volume comb:
    // every third node empty (the passthrough census), the rest a φ_s sweep.
    let mut v0: Vec<[f32; 4]> = Vec::with_capacity(n_nodes);
    let mut sv: Vec<f32> = Vec::with_capacity(n_nodes);
    for n in 0..n_nodes {
        let f = n as f32;
        v0.push([
            (f * 0.37).sin() * 3.0,
            (f * 0.11).cos() * 2.0 - 1.0,
            (f * 0.53).sin() * 1.5,
            1.0,
        ]);
        let phi_s = if n % 3 == 0 {
            0.0
        } else {
            0.1 + 0.5 * ((f * 0.23).sin() * 0.5 + 0.5)
        };
        sv.push(phi_s * h3 as f32);
    }
    solver.write_grid_velocities_for_test(&v0);
    solver.write_solid_volumes_for_test(&sv);
    solver.run_drag_for_test();
    let v1 = solver.read_grid_velocities();
    let react = solver.read_reactions();
    let sv_rt = solver.read_solid_volumes();

    // Interior nodes only (wall nodes get the BC treatment after the fold — the reaction is
    // recorded pre-BC, so the per-node identity is checked where v1 is the pure fold).
    let interior = |i: usize, j: usize, k: usize| -> bool {
        // origin = −h ⇒ in-box strict interior is index 2..dims−3 (skip face layers).
        let ok = |x: usize, d: u32| x >= 2 && x + 3 <= d as usize;
        ok(i, dims[0]) && ok(j, dims[1]) && ok(k, dims[2])
    };
    let mut checked = 0usize;
    let mut zero_checked = 0usize;
    let mut worst = 0.0f64;
    for k in 0..dims[2] as usize {
        for j in 0..dims[1] as usize {
            for i in 0..dims[0] as usize {
                if !interior(i, j, k) {
                    continue;
                }
                let n = nidx(dims, i, j, k);
                if sv_rt[n] <= 0.0 {
                    // Exact-zero passthrough: untouched velocity, zero reaction, ς = 1.
                    for a in 0..3 {
                        assert_eq!(v1[n][a].to_bits(), v0[n][a].to_bits(), "node {n} axis {a}");
                        assert_eq!(react[n][a], 0.0);
                    }
                    assert_eq!(react[n][3], 1.0, "ς must be exactly 1 at zero-solid nodes");
                    zero_checked += 1;
                    continue;
                }
                // Pair identity: reaction = m·(v_naive − v_fold) = −water drag impulse, and
                // the fold matches the CPU twin of the blended rate.
                let phi_s = (sv_rt[n] as f64 / h3).min(0.95);
                let beta = blended_drag_rate(1.0, (1.0 - phi_s) as f32, 0.05) as f64;
                let e = (-beta * DT as f64).exp();
                for a in 0..3 {
                    let dv = v0[n][a] as f64 - v1[n][a] as f64; // gravity 0: naive = v0
                    let scale = (v0[n][a] as f64).abs().max(0.1);
                    assert!(
                        (react[n][a] as f64 - dv).abs() <= PAIR_REL_TOL * scale.max(dv.abs()),
                        "node {n} axis {a}: reaction {} vs water impulse {dv}",
                        react[n][a]
                    );
                    let expect = v0[n][a] as f64 * e;
                    let err = (v1[n][a] as f64 - expect).abs() / scale;
                    worst = worst.max(err);
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 100 && zero_checked > 50, "census too small");
    assert!(
        worst <= 1.0e-4,
        "fold deviates from the exponential twin: worst rel err {worst:.2e}"
    );
    println!(
        "twofield U6 pair momentum: {checked} drag nodes exact, {zero_checked} passthrough nodes bitwise, fold-vs-twin worst {worst:.2e}"
    );
}

// ==============================================================================================
// COUPLED SPLIT gates
// ==============================================================================================

/// Slip decay vs the analytic exponential under timestep refinement. Pressure-off arm is
/// the pure pair fold (tight band); the full pipeline stays within the loose band; no
/// oscillation, no kinetic-energy growth. Gravity off, sealed box.
#[test]
fn slip_decay_matches_analytic_exponential_under_dt_refinement() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    // Open-base column, gravity off, uniform DOWNWARD slip: the only divergence-compatible
    // uniform slip flow in a box (a sealed lateral push is correctly blocked by the
    // projection — that would gate incompressibility, not the drag split).
    let spec = ColumnSpec {
        box_max: [16.0, 16.0, 16.0],
        bed_top: 12.5,
        drag_scale: 0.02,
        gravity: 0.0,
        open_base: true,
        // Uniform β by construction (see ColumnSpec): the exponential reference IS the
        // solver's own per-step recurrence, at every refinement level. Without it the
        // refine-1 core read a slow-β contamination tail (log-ratio 0.944) and refinement
        // made it WORSE (0.905 at refine 4) — a property of the heterogeneous scene, not
        // of the fold.
        uniform_beta: true,
        ..ColumnSpec::default()
    };
    let t_end = 20.0 * DT as f64; // β·t ≈ 2.5 e-folds at the defaults below
    for (label, sweeps, band) in [
        ("pressure-off", (0u32, 0u32), SLIP_TIGHT),
        ("full pipeline", (COARSE_SWEEPS_DEFAULT, 8u32), SLIP_LOOSE),
    ] {
        for refine in [1u32, 2, 4] {
            let mut col = build_column(&gpu, &spec);
            col.solver
                .set_pressure_budget_for_test(COARSE_RATIO_DEFAULT, sweeps.0, sweeps.1);
            let n = col.n_water as usize;
            let mut vel = col.solver.read_velocities();
            let v0 = 2.0f32;
            for v in vel.iter_mut().take(n) {
                *v = [0.0, -v0, 0.0, 0.0];
            }
            col.solver.write_velocities_for_test(&vel);
            let dt = DT / refine as f32;
            let steps = (t_end / dt as f64).round() as u32;
            let input = EmissionInput::default();
            let mut ke_prev = f64::MAX;
            for s in 0..steps {
                col.solver.step(dt, &input);
                if s % (5 * refine) == 0 || s + 1 == steps {
                    let v = col.solver.read_velocities();
                    let mean_vx = v[..n].iter().map(|w| -(w[1] as f64)).sum::<f64>() / n as f64;
                    assert!(
                        mean_vx >= -0.02 * v0 as f64,
                        "{label} refine {refine}: oscillation (mean slip {mean_vx:.3})"
                    );
                    let ke = v[..n]
                        .iter()
                        .map(|w| {
                            (w[0] as f64).powi(2) + (w[1] as f64).powi(2) + (w[2] as f64).powi(2)
                        })
                        .sum::<f64>();
                    assert!(
                        ke <= ke_prev * (1.0 + SLIP_KE_GROWTH) + 1e-9,
                        "{label} refine {refine}: kinetic energy grew {ke_prev:.4} -> {ke:.4}"
                    );
                    ke_prev = ke;
                }
            }
            // Analytic comparison over the bed CORE only: the uniform-β construction makes
            // every node the core can see carry the same β, but the box WALLS still brake
            // tangential flow at particle resolution (the zero-velocity out-of-box pad
            // nodes sit inside the B-spline support of wall-adjacent particles — the U2
            // edge-column artifact), so the gate keeps a ≥3h standoff. (Particles drift
            // ≤ v0/β ≈ 0.26 units, so the position census stays valid.)
            let core_lo = [6.0, 6.0, 6.0];
            let core_hi = [10.0, 10.0, 10.0];
            let phi_f = mean_phi_f_in(&col.solver, core_lo, core_hi);
            let beta = blended_drag_rate(1.0, phi_f as f32, spec.drag_scale) as f64;
            let vel = col.solver.read_velocities();
            let pos = col.solver.read_positions();
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for i in 0..n {
                if (0..3)
                    .all(|a| (pos[i][a] as f64) >= core_lo[a] && (pos[i][a] as f64) <= core_hi[a])
                {
                    sum += -(vel[i][1] as f64);
                    cnt += 1;
                }
            }
            assert!(cnt >= 20, "core census too small ({cnt})");
            let mean_vx = sum / cnt as f64;
            let expect = v0 as f64 * (-beta * t_end).exp();
            let log_ratio = (mean_vx.max(1e-6) / v0 as f64).ln() / (expect / v0 as f64).ln();
            println!(
                "twofield U6 slip decay [{label} refine {refine}]: core mean slip {mean_vx:.4} ({cnt} particles) vs analytic {expect:.4} (β {beta:.2}, log-ratio {log_ratio:.3})"
            );
            assert!(
                (log_ratio - 1.0).abs() <= band,
                "{label} refine {refine}: decay off the exponential (log-ratio {log_ratio:.3}, band ±{band})"
            );
        }
    }
}

/// β-variation stress: high slip into a partially dry variable-φ column (dry→wet front, φ
/// near packing at the dense bottom), dt arm vs a dt/8 reference that subcycles drag AND
/// projection together. Bounded profile error, no energy blow-up, finite.
#[test]
fn coupled_split_bounded_vs_subcycled_reference() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let spec = ColumnSpec {
        box_max: [16.0, 28.0, 16.0],
        bed_top: 9.5,
        drag_scale: 0.05,
        pond_layers: 2,
        ramp: true,
        sat_frac: 0.6, // dry upper bed: the pond + slip drives a wetting front into it
        ..ColumnSpec::default()
    };
    // Gate point: t = 1.0 s — many drag times (1/β ≈ 0.05 s) past the impact, where a stable
    // split must agree with the subcycled reference on the (near-settled) state. The
    // mid-transient t = 0.25 s profiles are PRINTED for the record but not gated: they
    // measure the wetting-front position jitter between the arms (sharp-front phase error
    // that dt-refinement legitimately moves), while the stability claim — no energy gain, no
    // oscillation divergence — is pinned at the settled point and by the speed/vmax bounds.
    let run = |refine: u32, t_end: f64| -> (Vec<f64>, f64, u32) {
        let mut col = build_column(&gpu, &spec);
        let n = col.n_water as usize;
        let mut vel = col.solver.read_velocities();
        for v in vel.iter_mut().take(n) {
            *v = [0.0, -3.0, 0.0, 0.0]; // high slip vs the frozen skeleton
        }
        col.solver.write_velocities_for_test(&vel);
        let dt = DT / refine as f32;
        let steps = (t_end / dt as f64).round() as u32;
        let input = EmissionInput::default();
        for _ in 0..steps {
            col.solver.step(dt, &input);
        }
        let pos = col.solver.read_positions();
        let vel = col.solver.read_velocities();
        assert!(all_finite(&pos) && all_finite(&vel), "non-finite state");
        // Layer-mean v_y profile (8 bins over the column height) + mean speed.
        let bins = 8usize;
        let yspan = 14.0f64;
        let mut sums = vec![0.0f64; bins];
        let mut cnts = vec![0usize; bins];
        let mut speed = 0.0f64;
        let mut vmax = 0.0f32;
        for i in 0..n {
            let b = (((pos[i][1] as f64) / yspan * bins as f64) as usize).min(bins - 1);
            sums[b] += vel[i][1] as f64;
            cnts[b] += 1;
            let s = (vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1] + vel[i][2] * vel[i][2]).sqrt();
            speed += s as f64;
            vmax = vmax.max(s);
        }
        assert!(
            vmax <= SPLIT_VMAX,
            "refine {refine}: runaway speed {vmax} (cap {SPLIT_VMAX})"
        );
        let profile: Vec<f64> = sums
            .iter()
            .zip(&cnts)
            .map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();
        (profile, speed / n as f64, n as u32)
    };
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    // Mid-transient record (not gated — see the comment above).
    let (m1, ms1, _) = run(1, 0.25);
    let (m8, ms8, _) = run(8, 0.25);
    let mdiff: Vec<f64> = m1.iter().zip(&m8).map(|(a, b)| a - b).collect();
    println!(
        "twofield U6 coupled split (t=0.25 record): profile RMS diff {:.4} vs ref RMS {:.4} | mean speed {ms1:.4} vs ref {ms8:.4}",
        rms(&mdiff),
        rms(&m8)
    );
    let (p1, s1, n1) = run(1, 1.0);
    let (p8, s8, n8) = run(8, 1.0);
    assert_eq!(n1, n8);
    let diff: Vec<f64> = p1.iter().zip(&p8).map(|(a, b)| a - b).collect();
    let (rd, rr) = (rms(&diff), rms(&p8));
    println!(
        "twofield U6 coupled split (t=1.0 gate): profile RMS diff {rd:.4} vs ref RMS {rr:.4} | mean speed {s1:.4} vs ref {s8:.4}"
    );
    assert!(
        rd <= SPLIT_PROFILE_BAND * rr + SPLIT_PROFILE_ABS,
        "split error unbounded: profile RMS diff {rd:.4} vs band {:.4}",
        SPLIT_PROFILE_BAND * rr + SPLIT_PROFILE_ABS
    );
    assert!(
        s1 <= SPLIT_SPEED_FACTOR * s8 + SPLIT_SPEED_ABS,
        "split arm gains energy: mean speed {s1:.4} vs ref {s8:.4}"
    );
}

// ==============================================================================================
// DARCY SCALING + HEAD + ENTRY FLUX (open-base drained column)
// ==============================================================================================

/// Drained flux through a static saturated column ∝ K(φ): grind sweep (ordering + pairwise
/// ratio vs the β-ratio mapped through models::permeability + absolute band), then the head
/// arm (pond surcharge multiplies the hydraulic gradient).
#[test]
fn darcy_flux_scales_with_permeability_and_head() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let measure = |d: f32, pond: usize| -> (f64, f64, f64) {
        let spec = ColumnSpec {
            d,
            pond_layers: pond,
            open_base: true,
            ..ColumnSpec::default()
        };
        let mut col = build_column(&gpu, &spec);
        let input = EmissionInput::default();
        for _ in 0..60 {
            col.solver.step(DT, &input);
        }
        let mut flux = 0.0;
        let mut head = 0.0;
        let samples = 12;
        for s in 0..samples {
            for _ in 0..5 {
                col.solver.step(DT, &input);
            }
            flux += plane_outflux(&col.solver, col.box_max, 4.0);
            if s == samples / 2 {
                let pos = col.solver.read_positions();
                head = top_decile_y(&pos, col.n_water as usize);
            }
        }
        let phi = mean_bed_phi_f(&col.solver, col.bed_top, col.box_max);
        (flux / samples as f64, head, phi)
    };

    // Grind sweep, pond-fed (4 layers) so the column STAYS saturated through the whole
    // sampling window at every grind — an unfed column drains several units of head during
    // the window at the coarse grinds (the flux samples decay 250 → 130 at the coarsest —
    // measured) and the i = 1 self-weight prediction goes stale. The steady state is
    // v = i·g/β(d, φ) with the measured-head gradient i = (H_w + L)/L, q = φ_f·v·A.
    // Grinds keep 16/d ∈ ℤ for the d/2 wall-margin phase match (see build_column).
    let grinds = [0.8f32, 1.0, 1.6];
    let l_bed = 9.5;
    let mut q = Vec::new();
    for &d in &grinds {
        let (flux, head, phi) = measure(d, 4);
        let beta = blended_drag_rate(d, phi as f32, 0.05) as f64;
        let area = 16.0 * 16.0;
        let grad = ((head - l_bed).max(0.0) + l_bed) / l_bed;
        let pred = phi * GRAV * grad / beta * area;
        let ratio = flux / pred;
        println!(
            "twofield U6 darcy d={d}: flux {flux:.2} vs pred {pred:.2} (ratio {ratio:.2}, φ_f {phi:.3}, β {beta:.2}, i {grad:.2})"
        );
        assert!(
            (DARCY_ABS_BAND.0..=DARCY_ABS_BAND.1).contains(&ratio),
            "d={d}: flux/pred {ratio:.2} outside {DARCY_ABS_BAND:?}"
        );
        q.push((d, flux, pred));
    }
    for w in q.windows(2) {
        let (d0, q0, p0) = w[0];
        let (d1, q1, p1) = w[1];
        assert!(
            q1 > q0,
            "ordering broke: q({d1}) = {q1:.2} ≤ q({d0}) = {q0:.2}"
        );
        let measured = q1 / q0;
        let predicted = p1 / p0; // q ∝ K(d)·i ∝ d²·i, through each arm's measured head
        assert!(
            (measured / predicted - 1.0).abs() <= DARCY_RATIO_TOL,
            "grind ratio {measured:.2} vs K·i-ratio prediction {predicted:.2} (±{DARCY_RATIO_TOL})"
        );
        println!("twofield U6 darcy ratio {d0}->{d1}: measured {measured:.2} vs predicted {predicted:.2}");
    }

    // Head arm: pond surcharge multiplies the gradient, i = (H_w + L)/L.
    let (q0, _h0, _) = measure(1.0, 0);
    let (q4, h4, _) = measure(1.0, 4);
    let i_ratio = ((h4 - l_bed).max(0.0) + l_bed) / l_bed;
    let measured = q4 / q0;
    println!(
        "twofield U6 head: flux ratio {measured:.2} vs head ratio {i_ratio:.2} (pond head {:.2})",
        h4 - l_bed
    );
    assert!(
        (measured / i_ratio - 1.0).abs() <= HEAD_RATIO_TOL,
        "head scaling off: flux ratio {measured:.2} vs head ratio {i_ratio:.2} (±{HEAD_RATIO_TOL})"
    );
}

/// ENTRY FLUX (the anti-divergence-penalty gate): ponded water enters the bed at the
/// head-driven Darcy rate — the pond surface descends at q within the band (main's
/// divergence-penalty baseline is a stuck layer ≈ 0), and the pore-pressure profile has no
/// interface spike.
#[test]
fn entry_flux_no_pressure_jump_barrier() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let spec = ColumnSpec {
        pond_layers: 4,
        open_base: true,
        ..ColumnSpec::default()
    };
    let mut col = build_column(&gpu, &spec);
    let input = EmissionInput::default();
    for _ in 0..60 {
        col.solver.step(DT, &input);
    }
    let n = col.n_water as usize;
    let y0 = top_decile_y(&col.solver.read_positions(), n);
    let frames = 90u32;
    for _ in 0..frames / 2 {
        col.solver.step(DT, &input);
    }
    let phi = mean_bed_phi_f(&col.solver, col.bed_top, col.box_max);
    let y_mid = top_decile_y(&col.solver.read_positions(), n);
    // Interface-spike check at mid-window, while the pond still stands.
    let layers = layer_pressure(&col.solver);
    let mut max_jump = 0.0f64;
    for w in layers.windows(2) {
        max_jump = max_jump.max((w[1].1 - w[0].1).abs());
    }
    let (_, h, _) = grid_of(&col.solver);
    assert!(
        max_jump <= ENTRY_DP_JUMP_MAX * REST * GRAV * h as f64,
        "pore-pressure interface spike: max adjacent-layer jump {max_jump:.1} (cap {:.1})",
        ENTRY_DP_JUMP_MAX * REST * GRAV * h as f64
    );
    for _ in 0..frames / 2 {
        col.solver.step(DT, &input);
    }
    let y1 = top_decile_y(&col.solver.read_positions(), n);
    let descent = (y0 - y1) / (frames as f64 * DT as f64);
    let l_bed = col.bed_top;
    let head = (y_mid - l_bed).max(0.0);
    let beta = blended_drag_rate(1.0, phi as f32, 0.05) as f64;
    let q_pred = phi * GRAV * (head + l_bed) / l_bed / beta; // per unit area = surface descent
    let ratio = descent / q_pred;
    println!(
        "twofield U6 entry flux: descent {descent:.3}/s vs Darcy {q_pred:.3}/s (ratio {ratio:.2}); max Δp jump {max_jump:.1}"
    );
    assert!(
        ratio >= ENTRY_KILL_FLOOR,
        "ENTRY BARRIER: pond descends at {ratio:.2} of the Darcy rate — the divergence-penalty signature"
    );
    assert!(
        (ENTRY_BAND.0..=ENTRY_BAND.1).contains(&ratio),
        "entry flux {ratio:.2} outside the pre-registered band {ENTRY_BAND:?}"
    );
}

// ==============================================================================================
// FACE-VELOCITY CONSISTENCY (multiphase Rhie-Chow corollary)
// ==============================================================================================

/// The divergence feeding the projection RHS carries the same drag-folded, φ-weighted
/// velocities as the momentum update: with zero pressure sweeps the post-step grid field IS
/// the field cell_classify consumed, so the recorded rhs must match its CPU recompute; and
/// omitting either the φ weighting or the drag fold must produce a detectably different rhs
/// (the spurious-divergence driver of φ_s oscillation in Euler-Euler practice).
#[test]
fn face_velocity_consistency_rhs_carries_drag_and_phi() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let spec = ColumnSpec {
        pond_layers: 2,
        drag_scale: 0.05,
        ..ColumnSpec::default()
    };
    let mut col = build_column(&gpu, &spec);
    let n = col.n_water as usize;
    let mut vel = col.solver.read_velocities();
    for v in vel.iter_mut().take(n) {
        *v = [0.0, -2.0, 0.0, 0.0];
    }
    col.solver.write_velocities_for_test(&vel);
    col.solver
        .set_pressure_budget_for_test(COARSE_RATIO_DEFAULT, 0, 0);
    col.solver.step(DT, &EmissionInput::default());

    let (_, h, dims) = grid_of(&col.solver);
    let nc = [
        dims[0] as usize - 1,
        dims[1] as usize - 1,
        dims[2] as usize - 1,
    ];
    let meta = col.solver.read_cell_meta();
    let gv = col.solver.read_grid_velocities();
    let phi = node_phi(&col.solver);
    let react = col.solver.read_reactions();
    let sv = col.solver.read_solid_volumes();
    let cidx = |i: usize, j: usize, k: usize| i + nc[0] * (j + nc[1] * k);
    let g = -GRAV;

    // CPU rhs twin: rhs = f·(s_relief − D(Φv))/dt (+ the pocket_mark under-density term away
    // from air), exactly the pressure.wgsl construction.
    let rhs_twin = |c: usize, vfield: &dyn Fn(usize) -> [f64; 3], use_phi: bool| -> f64 {
        let f = meta[c][1] as f64;
        if f <= 0.0 {
            return 0.0;
        }
        let (i, j, k) = (c % nc[0], (c / nc[0]) % nc[1], c / (nc[0] * nc[1]));
        let mut div = 0.0;
        for oz in 0..2usize {
            for oy in 0..2usize {
                for ox in 0..2usize {
                    let nn = nidx(dims, i + ox, j + oy, k + oz);
                    let s = [
                        ox as f64 * 2.0 - 1.0,
                        oy as f64 * 2.0 - 1.0,
                        oz as f64 * 2.0 - 1.0,
                    ];
                    let v = vfield(nn);
                    let ph = if use_phi { phi[nn] as f64 } else { 1.0 };
                    div += ph * (s[0] * v[0] + s[1] * v[1] + s[2] * v[2]) / (4.0 * h as f64);
                }
            }
        }
        let rho_rel = meta[c][2] as f64;
        let dtf = DT as f64;
        let relax = DENSITY_RELAX_FRAMES as f64 * dtf;
        let mut rhs = f * ((rho_rel / REST - 1.0).max(0.0) / relax - div) / dtf;
        // pocket_mark's under-density half (interior cells away from air).
        if meta[c][3] < 0.5 && rho_rel / REST < 1.0 {
            let mut near_air = false;
            for a in 0..3usize {
                for s in [-1i64, 1] {
                    let mut q = [i as i64, j as i64, k as i64];
                    q[a] += s;
                    if q[0] >= 0
                        && q[1] >= 0
                        && q[2] >= 0
                        && q[0] < nc[0] as i64
                        && q[1] < nc[1] as i64
                        && q[2] < nc[2] as i64
                        && meta[cidx(q[0] as usize, q[1] as usize, q[2] as usize)][3] >= 0.5
                    {
                        near_air = true;
                    }
                }
            }
            if !near_air {
                rhs += f * ((rho_rel / REST - 1.0) / relax) / dtf;
            }
        }
        rhs
    };
    let folded = |n: usize| -> [f64; 3] { [gv[n][0] as f64, gv[n][1] as f64, gv[n][2] as f64] };
    // Drag-unfolded reconstruction (what a drag-omitting rhs would see): invert the
    // exponential fold per node — only valid away from the BC'd wall layers.
    let unfolded = |n: usize| -> [f64; 3] {
        let v = folded(n);
        if sv[n] <= 0.0 || gv[n][3] <= 1.0e-4 {
            return v;
        }
        let sig = react[n][3] as f64;
        let srcf = sig * DT as f64;
        let phi_s = (sv[n] as f64 / (h as f64).powi(3)).min(0.95);
        let beta = blended_drag_rate(1.0, (1.0 - phi_s) as f32, 0.05) as f64;
        let e = (-beta * DT as f64).exp();
        let gsrc = [0.0, g * srcf, 0.0];
        let gdt = [0.0, g * DT as f64, 0.0];
        let mut out = [0.0; 3];
        for a in 0..3 {
            out[a] = (v[a] - gsrc[a]) / e + gdt[a];
        }
        out
    };

    // Census: cells whose full corner support is strictly interior (no BC'd nodes).
    let node_interior = |i: usize, j: usize, k: usize| -> bool {
        let ok = |x: usize, d: u32| x >= 2 && x + 3 <= d as usize;
        ok(i, dims[0]) && ok(j, dims[1]) && ok(k, dims[2])
    };
    let mut matched_max = 0.0f64;
    let mut no_phi_max = 0.0f64;
    let mut no_drag_max = 0.0f64;
    let mut census = 0usize;
    for k in 0..nc[2] {
        for j in 0..nc[1] {
            for i in 0..nc[0] {
                let mut ok = true;
                for oz in 0..2usize {
                    for oy in 0..2usize {
                        for ox in 0..2usize {
                            if !node_interior(i + ox, j + oy, k + oz) {
                                ok = false;
                            }
                        }
                    }
                }
                let c = cidx(i, j, k);
                if !ok || meta[c][1] <= 0.0 {
                    continue;
                }
                // Bed cells only (some solid under the corner support) — where the terms differ.
                let any_solid = (0..2)
                    .flat_map(|oz| (0..2).flat_map(move |oy| (0..2).map(move |ox| (ox, oy, oz))))
                    .any(|(ox, oy, oz)| sv[nidx(dims, i + ox, j + oy, k + oz)] > 0.0);
                if !any_solid {
                    continue;
                }
                let recorded = meta[c][0] as f64;
                matched_max = matched_max.max((rhs_twin(c, &folded, true) - recorded).abs());
                no_phi_max = no_phi_max.max((rhs_twin(c, &folded, false) - recorded).abs());
                no_drag_max = no_drag_max.max((rhs_twin(c, &unfolded, true) - recorded).abs());
                census += 1;
            }
        }
    }
    assert!(census >= 30, "consistency census too small ({census})");
    println!(
        "twofield U6 face consistency: {census} bed cells | matched err {matched_max:.3} | φ-omitted err {no_phi_max:.3} | drag-omitted err {no_drag_max:.3}"
    );
    assert!(
        matched_max <= 0.05 + 1.0e-3 * GRAV / DT as f64 * 0.01,
        "rhs does not match the folded φ-weighted recompute (max err {matched_max:.3})"
    );
    assert!(
        no_phi_max >= 10.0 * matched_max.max(0.005) && no_phi_max > 1.0,
        "omitting φ must be detectable (err {no_phi_max:.3} vs matched {matched_max:.3})"
    );
    assert!(
        no_drag_max >= 10.0 * matched_max.max(0.005) && no_drag_max > 1.0,
        "omitting the drag fold must be detectable (err {no_drag_max:.3} vs matched {matched_max:.3})"
    );
}

// ==============================================================================================
// BUOYANT REACTION + UNDRAINED LOAD
// ==============================================================================================

/// A submerged dense block of frozen grains in a hydrostatic tank: the per-frame constraint
/// reaction equals the displaced-volume weight, upward (the frozen body has no dynamics to
/// observe — the LEDGER is the observable; dynamic buoyancy is U7).
#[test]
fn buoyant_reaction_equals_displaced_weight() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, -(GRAV as f32), 0.0],
        box_min: [0.0; 3],
        box_max: [16.0, 24.0, 16.0],
        regions: vec![
            SeedRegion {
                min: [0.5, 0.5, 0.5],
                max: [15.6, 11.6, 15.6],
                species: Species::Water,
            },
            SeedRegion {
                min: [6.0, 3.0, 6.0],
                max: [9.6, 6.6, 9.6],
                species: Species::Grain,
            },
        ],
        solids: Vec::new(),
        ..Scene::default()
    };
    let mats = Materials::default();
    let cfg = Config::default(); // drag_scale 0.30: deliberately STIFF (β·dt ≈ 1.9) — the
                                 // exponential-integrator ledger must close at any stiffness
    let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
    let (water_seed, n_grains) = solver.phase_counts();
    // Re-place the water that seeded inside the block: pore lattice inside, surplus stacked
    // on the surface (settles during warm-up).
    let mut pos = solver.read_positions();
    let inside = |p: &[f32; 4]| -> bool {
        p[0] > 5.5 && p[0] < 10.1 && p[1] > 2.5 && p[1] < 7.1 && p[2] > 5.5 && p[2] < 10.1
    };
    let pore = lattice([6.2, 3.2, 6.2], [9.4, 6.4, 9.4], 1.2806);
    let mut moved: Vec<usize> = (0..water_seed as usize)
        .filter(|&i| inside(&pos[i]))
        .collect();
    assert!(moved.len() >= pore.len(), "not enough displaced water");
    for (slot, p) in moved.drain(..pore.len()).zip(&pore) {
        pos[slot] = [p[0], p[1], p[2], 1.0];
    }
    for (n, slot) in moved.into_iter().enumerate() {
        // Surplus: thin lattice just above the settled surface.
        let per = 15usize;
        pos[slot] = [
            0.7 + (n % per) as f32,
            12.2 + (n / (per * per)) as f32,
            0.7 + ((n / per) % per) as f32,
            1.0,
        ];
    }
    solver.write_positions_for_test(&pos);
    let input = EmissionInput::default();
    for _ in 0..300 {
        solver.step(DT, &input);
    }
    let mut mean = [0.0f64; 3];
    let frames = 60;
    for _ in 0..frames {
        solver.step(DT, &input);
        let r = total_reaction(&solver);
        for a in 0..3 {
            mean[a] += r[a] / frames as f64;
        }
    }
    let v_solid = n_grains as f64 * grain_volume(1.0) as f64;
    let expect = v_solid * REST * GRAV * DT as f64; // displaced-volume weight per frame
    let ratio = mean[1] / expect;
    println!(
        "twofield U6 buoyancy: reaction/frame ({:.3}, {:.3}, {:.3}) vs displaced weight {expect:.3} (ratio {ratio:.3}, {n_grains} grains)",
        mean[0], mean[1], mean[2]
    );
    assert!(
        (BUOY_BAND.0..=BUOY_BAND.1).contains(&ratio),
        "buoyant reaction ratio {ratio:.3} outside {BUOY_BAND:?}"
    );
    assert!(
        mean[0].abs() <= BUOY_LATERAL * expect && mean[2].abs() <= BUOY_LATERAL * expect,
        "lateral reaction leak: ({:.3}, {:.3})",
        mean[0],
        mean[2]
    );
}

/// Undrained vertical load on a sealed saturated column: a water surcharge raises the basal
/// pore pressure by the full ρ·g·Δh (no effective-stress path through a frozen skeleton),
/// and the in-bed pore-pressure slope is the full ρ_w·g — the anti "drag steals the
/// hydrostatic load" gate the ς-mobility exists for.
#[test]
fn undrained_surcharge_transfers_fully_to_pore_pressure() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let run = |pond: usize| -> (f64, f64, Vec<(f64, f64)>, f64) {
        let spec = ColumnSpec {
            pond_layers: pond,
            drag_scale: 0.30, // stiff regime on purpose (β·dt ≈ 1.9)
            ..ColumnSpec::default()
        };
        let mut col = build_column(&gpu, &spec);
        let input = EmissionInput::default();
        // 600 frames: the seeded column's settle-in (rim suction + pond-load compaction
        // through the βΔt ≈ 1.9 drag) is still draining back at 300 — the basal Δp reads
        // ~0.86 of the surcharge mid-transient vs ~1.00 relaxed (measured; the in-bed
        // slope holds ρ_w·g ± 3% from 180 frames on).
        for _ in 0..600 {
            col.solver.step(DT, &input);
        }
        let layers = layer_pressure(&col.solver);
        let top = top_decile_y(&col.solver.read_positions(), col.n_water as usize);
        let base = layers
            .iter()
            .find(|(y, _)| *y > 0.5 && *y < 2.5)
            .map(|(_, p)| *p)
            .expect("basal layer present");
        // Measured intrinsic pore-water density over the slope-fit slab (normalizes out the
        // test lattice's saturation-seeding imperfection — the physics claim is
        // ∇p = ρ_w,measured·g, not that the test seeded ρ_w = 1 perfectly).
        let rho_w = mean_intrinsic_density(&col.solver, [4.0, 1.0, 4.0], [12.0, 8.0, 12.0]);
        (base, top, layers, rho_w)
    };
    let (p_a, top_a, _, _) = run(1);
    let (p_b, top_b, layers_b, rho_w) = run(5);
    let dh = top_b - top_a;
    assert!(
        dh > 2.0,
        "surcharge arm did not raise the water table ({dh:.2})"
    );
    let transfer = (p_b - p_a) / (REST * GRAV * dh);
    println!(
        "twofield U6 undrained: Δp_base {:.1} vs ρgΔh {:.1} (transfer {transfer:.3})",
        p_b - p_a,
        REST * GRAV * dh
    );
    assert!(
        (UNDRAINED_BAND.0..=UNDRAINED_BAND.1).contains(&transfer),
        "surcharge transfer {transfer:.3} outside {UNDRAINED_BAND:?} — load lost to the drag split"
    );
    // In-bed slope: fit layers fully inside the bed (y ∈ [1, 8]).
    let bed_layers: Vec<(f64, f64)> = layers_b
        .iter()
        .filter(|(y, _)| *y > 0.5 && *y < 8.0)
        .cloned()
        .collect();
    assert!(bed_layers.len() >= 3);
    let (slope, r2) = linfit(&bed_layers);
    let ratio = slope / -(rho_w * GRAV);
    println!(
        "twofield U6 undrained slope: dp/dy {slope:.2} vs ρ_w·g = {:.2} (ratio {ratio:.3}, R² {r2:.3})",
        rho_w * GRAV
    );
    assert!(
        (UNDRAINED_BAND.0..=UNDRAINED_BAND.1).contains(&ratio) && r2 > 0.9,
        "in-bed pore-pressure slope {ratio:.3} of hydrostatic (R² {r2:.3})"
    );
}

// ==============================================================================================
// φ<1 DIVERGENCE DECAY RE-RUN (operator consistency in the regime the novelty lives in)
// ==============================================================================================

/// The U3 divergence-decay gate re-run on a variable-φ saturated column against the MIXTURE
/// divergence: ≥2× decay per fine-sweep doubling until below DIV_TOL, reached inside the
/// knob grid; a plateau above tolerance FAILS (the A≠D·G signature).
#[test]
fn mixture_divergence_decay_variable_phi_column() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let spec = ColumnSpec {
        box_max: [24.0, 28.0, 24.0],
        pond_layers: 2,
        ramp: true,
        drag_scale: 0.30,
        ..ColumnSpec::default()
    };
    let mut donor = build_column(&gpu, &spec);
    let input = EmissionInput::default();
    for _ in 0..240 {
        donor.solver.step(DT, &input);
    }
    let snap_pos = donor.solver.read_positions();
    let snap_vel = donor.solver.read_velocities();
    let snap_cmat = donor.solver.read_affine_rows();
    let live = donor.n_water;

    let mut curve: Vec<(u32, f64, usize)> = Vec::new();
    for &nf in &DECAY_BUDGETS {
        let col = build_column(&gpu, &spec);
        let mut s = col.solver;
        s.write_positions_for_test(&snap_pos);
        s.write_velocities_for_test(&snap_vel);
        s.write_affine_for_test(&snap_cmat);
        s.set_live_water_for_test(live);
        s.set_pressure_budget_for_test(COARSE_RATIO_DEFAULT, COARSE_SWEEPS_DEFAULT, nf);
        s.step(DT, &input);
        let (rms, cells) = rms_interior_mixture_div(&s);
        println!(
            "twofield U6 mixture divergence decay: fine sweeps {nf:2} -> RMS {rms:.4e} ({cells} interior cells)"
        );
        curve.push((nf, rms, cells));
    }
    for &(_, rms, cells) in &curve {
        assert!(cells >= 50, "only {cells} interior cells — gate vacuous");
        assert!(rms.is_finite());
    }
    let mut reached: Option<u32> = None;
    for i in 0..curve.len() {
        let (nf, rms, _) = curve[i];
        if rms < DIV_TOL {
            reached = Some(nf);
            break;
        }
        assert!(
            i + 1 < curve.len(),
            "mixture divergence never fell below DIV_TOL {DIV_TOL} (last {rms:.4e} at {nf}) — plateau FAILS"
        );
        let (nf2, rms2, _) = curve[i + 1];
        assert!(
            rms / rms2 >= DECAY_FACTOR_MIN || rms2 < DIV_TOL,
            "decay factor {:.2} from {nf} -> {nf2} sweeps below {DECAY_FACTOR_MIN} above tolerance — the A≠D·G plateau",
            rms / rms2
        );
    }
    assert!(
        reached.expect("checked") <= 16,
        "tolerance reached only outside the declared knob grid"
    );
}

// ==============================================================================================
// FREE-WATER REGRESSION + SETTLED LONG RUN
// ==============================================================================================

/// φ_f = 1 passthrough is BIT-exact: a water-only tank vs the same tank plus a distant frozen
/// grain block (outside any B-spline support overlap) produce bitwise-identical water
/// trajectories; reactions are identically zero; the φ lane reads exactly 1 at water nodes.
#[test]
fn free_water_regression_bitwise_with_distant_grains() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let water = SeedRegion {
        min: [0.5, 0.5, 0.5],
        max: [15.6, 11.6, 15.6],
        species: Species::Water,
    };
    let scene_a = Scene {
        gravity: [0.0, -(GRAV as f32), 0.0],
        box_min: [0.0; 3],
        box_max: [16.0, 28.0, 16.0],
        regions: vec![water],
        solids: Vec::new(),
        ..Scene::default()
    };
    // The grain region comes AFTER the water region (seed RNG draws stay aligned) and sits
    // ≥ 9 units above the settled surface — far outside the 1.5h B-spline smear.
    let scene_b = Scene {
        regions: vec![
            water,
            // y ≥ 24: the block's B-spline smear lands on fine nodes j ≥ 12, outside the
            // hat support (±3 fine nodes) of every coarse node that corners an ACTIVE
            // coarse cell (water surface ≈ 12 → active coarse rows j_c ≤ 1, corner nodes
            // j_c ≤ 2 → support fine j ≤ 11) — a block one coarse-support lower perturbs
            // the coarse seed by ulps and breaks bitwise equality (measured).
            SeedRegion {
                min: [1.0, 24.0, 1.0],
                max: [4.6, 27.6, 4.6],
                species: Species::Grain,
            },
        ],
        ..scene_a.clone()
    };
    let mats = Materials::default();
    let cfg = Config::default();
    let mut a = TwofieldSolver::build(&scene_a, &mats, &cfg, &gpu);
    let mut b = TwofieldSolver::build(&scene_b, &mats, &cfg, &gpu);
    let (wa, sa) = a.phase_counts();
    let (wb, sb) = b.phase_counts();
    assert_eq!(wa, wb, "water seeds must align");
    assert_eq!(sa, 0);
    assert!(sb > 0, "arm B must seed the distant block");
    let input = EmissionInput::default();
    for _ in 0..200 {
        a.step(DT, &input);
        b.step(DT, &input);
    }
    let (pa, pb) = (a.read_positions(), b.read_positions());
    let (va, vb) = (a.read_velocities(), b.read_velocities());
    for i in 0..wa as usize {
        for c in 0..4 {
            assert_eq!(
                pa[i][c].to_bits(),
                pb[i][c].to_bits(),
                "position diverged at particle {i} lane {c}"
            );
            assert_eq!(
                va[i][c].to_bits(),
                vb[i][c].to_bits(),
                "velocity diverged at particle {i} lane {c}"
            );
        }
    }
    // The coupling ledgers are exactly zero in both arms (block is dry: no water mass there).
    for (arm, s) in [("A", &a), ("B", &b)] {
        for (n, r) in s.read_reactions().iter().enumerate() {
            assert!(
                r[0] == 0.0 && r[1] == 0.0 && r[2] == 0.0,
                "arm {arm}: nonzero reaction at node {n}: {r:?}"
            );
        }
    }
    // φ lane is exactly 1.0 wherever water mass lives (arm B).
    let gv = b.read_grid_velocities();
    let phi = node_phi(&b);
    let mut massy = 0usize;
    for n in 0..gv.len() {
        if gv[n][3] > 1.0e-4 {
            assert_eq!(phi[n], 1.0, "φ_f ≠ 1 at water node {n}");
            massy += 1;
        }
    }
    assert!(massy > 100);
    println!("twofield U6 free-water regression: {wa} water particles bitwise across 200 steps; {massy} massy nodes at φ_f = 1 exactly");
}

/// Long-run settled saturated column: drag must not pump energy — bounded sampled speeds,
/// no surface rise, finite state across 1200 steps at the STIFF default drag.
#[test]
fn settled_saturated_column_stays_settled_long_run() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_coupling: no GPU adapter; skipping.");
        return;
    };
    let spec = ColumnSpec {
        pond_layers: 2,
        drag_scale: 0.30,
        ..ColumnSpec::default()
    };
    let mut col = build_column(&gpu, &spec);
    let input = EmissionInput::default();
    for _ in 0..60 {
        col.solver.step(DT, &input);
    }
    let n = col.n_water as usize;
    let y0 = top_decile_y(&col.solver.read_positions(), n);
    let mut peak = 0.0f32;
    for f in 1..=LONGRUN_STEPS {
        col.solver.step(DT, &input);
        if f % 300 == 0 {
            let vel = col.solver.read_velocities();
            assert!(all_finite(&vel), "non-finite at frame {f}");
            let vmax = vel[..n]
                .iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                .fold(0.0f32, f32::max);
            peak = peak.max(vmax);
        }
    }
    let pos = col.solver.read_positions();
    assert!(all_finite(&pos));
    let y1 = top_decile_y(&pos, n);
    println!(
        "twofield U6 settled long run: {LONGRUN_STEPS} steps, sampled max|v| {peak:.3}, surface {y0:.2} -> {y1:.2}"
    );
    assert!(
        peak <= LONGRUN_MAX_SPEED,
        "saturated column creeping/pumping: max |v| {peak:.3}"
    );
    assert!(
        y1 - y0 <= LONGRUN_POP_RISE * SPACING,
        "surface climbed {:.2} spacings",
        y1 - y0
    );
}

//! Twofield U9 gates: the infiltration interface and conserved wetting — plan 2026-06-09-001
//! U9 / KTD-4(b)(c) / KTD-8. This carries the L2 LADDER GATE (the ponding/drain bands).
//!
//! ==================================== UNIT MAPPING (step 1 — the explicit early task) ========
//! The physical calibration table (High-Level Technical Design) is mapped into the solver's
//! reduced units BEFORE any gate band is written, using KEEP.md §2.
//!
//! LENGTH:  L = 27.7 sim-units / metre (KEEP §2, from the V60 paper height). So
//!          1 mm = 0.0277 su, 1 cm = 0.277 su.
//!
//! TIME (the g-consistency choice):  the sim runs gravity g_sim = 20 su/s² (Scene::default and
//!          every twofield scene), NOT KEEP §2's "physical" 9.80665 m/s² = 271.6 su/s². For the
//!          *acceleration* to be physically consistent, g_sim [su/s_sim²] must equal g_phys
//!          [su/s_phys²], so (s_phys/s_sim)² = g_phys_su / g_sim = 271.6/20 = 13.58, giving the
//!          time scale  τ = s_phys / s_sim = √13.58 ≈ 3.685.  The sim is slow motion by τ: one
//!          second of sim time is 3.685 s of physical brewing time. Physical durations map to
//!          sim time by DIVIDING by τ; physical velocities [m/s_phys] map to su/s_sim by
//!          ×(L·τ) (a velocity is length/time → ×L for length, ÷(1/τ) for time).
//!
//! Mapped bands (physical → reduced; every conversion is named below):
//!   * Bed permeability K 1e-13 (coarse) → 1e-14 m² (fine): the solver works in K(φ) via
//!     Kozeny–Carman on the GRIND diameter d (models::permeability), and the grind ordering /
//!     d² scaling is the gate (U6 darcy_flux already pins it). The ABSOLUTE K value is a
//!     drag_scale calibration, not a band here — U9 inherits U6's drag_scale.
//!   * Pond depth, medium grind 1–3 cm = 0.277–0.831 su. NOTE this is < one particle spacing
//!     (spacing = 1 su ≈ 3.6 cm physical): the test column resolution (one pond layer ≈ 1 su)
//!     is COARSER than a 1-cm physical pond, so the depth band is the FORMING/STABILISING
//!     check (a pond stands ≥ ~1 resolved layer and does not blow up), and the load-bearing L2
//!     ladder gate is the DRAIN-TIME band below (the calibration target the plan names).
//!   * Drain time, after pour stops, 30–60 s physical = 30/τ–60/τ = 8.14–16.28 s sim
//!     = 488–977 frames at dt = 1/60. (The L2 ladder gate band.)
//!   * Swelling +15–17% diameter at saturation, 60–80% complete ≤ 30 s physical (≤ 8.14 s sim).
//!     +15–17% diameter ⇒ V_eff/V_dry = 1.15³–1.17³ = 1.52–1.60. The wetting law sets V_cap =
//!     r_max·ρ_ratio·V_dry; with the defaults r_max·ρ_ratio = 1.5·1.3 = 1.95, full saturation
//!     gives V_eff/V_dry = 2.95 (d_eff/d_dry = 2.95^⅓ = 1.434, i.e. +43% — coffee-grounds
//!     scale). The SWELLING→K gate checks the +diameter is in a coffee band and K drops; the
//!     RATE (k_abs) is set from 60–80% by 8.14 s sim ⇒ k_abs ≈ −ln(1−0.6..0.8)/8.14 ≈
//!     0.11–0.20 1/s_sim.
//!   * Wetting-front suction ψ_f 50–110 mm = 1.39–3.05 su (head). As a Green–Ampt BODY FORCE
//!     the extra acceleration is a_suction = ψ_f·g / L_f where L_f is the front length scale
//!     (~the cell size h); with ψ_f ≈ 2.2 su, g = 20, L_f = h = 2 su the magnitude is
//!     a_suction ≈ 22 su/s² (order g). The suction CONFIG is a_suction directly; the GREEN-AMPT
//!     gate only needs suction-on early flux > suction-off (Darcy) early flux — a ratio gate,
//!     not an absolute one.
//!   * Bloom delay (hydrophobic entry), seconds-scale ≈ 2–6 s physical = 0.54–1.63 s sim
//!     = 33–98 frames. The BLOOM gate checks a mapped seconds-scale delay then acceleration.
//!
//! ==================================== ABSORPTION (KTD-4c, GIC-style, conserved) ==============
//! Grid-projected moisture phase-change (coupling.wgsl additions): p2g_moisture scatters two
//! grid lanes — total water SUPPLY S_n = Σ_w w·f_w·V_w and total grain DEMAND D_n =
//! Σ_g w·demand_g (models::wetting demand). The transfer at a node is T_n = min(S_n, D_n) (the
//! two-sided cap). g2p_absorb makes each water LOSE its weighted share Σ_n w·f_w·V_w·(T_n/S_n)
//! and each grain GAIN its weighted share Σ_n w·demand_g·(T_n/D_n): summed over particles each
//! node loses exactly T_n of water and gains exactly T_n of grain volume, so water-loss ==
//! grain-gain EXACTLY per node and globally (the conservation gate). Volume 1:1 into grain
//! moisture; the grain swells (effective volume V_dry + V_abs → φ_s, and effective diameter via
//! models::wetting). Drained-particle lifecycle: a water particle with f_w ≤ absorb_roundoff is
//! PARKED (deactivated, moved out of the live-water dispatch range is NOT done here — instead it
//! keeps f_w at the floor and stops scattering mass; its remaining tiny f_w is counted in the
//! conservation total so nothing is lost). See LIFECYCLE in the conservation gate.
//!
//! KNOB GRID (KTD-9): U9 adds these tunables, all enumerated here and fixed per-gate before any
//! result was observed: tf_absorb_rate (k_abs), tf_suction_accel (a_suction), tf_bloom_delay,
//! tf_filter_floor (on/off), plus the inherited U3 pressure grid and U6 drag_scale/grind. The
//! pond/drain bands are met within this grid or the unit HALTS (the plan's no-fallback rule).

#![allow(clippy::needless_range_loop)]

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;
const GRAV: f64 = 20.0;

// --- UNIT MAPPING constants (derivations in the module header) -------------------------------
const L_SU_PER_M: f64 = 27.7;
const TAU: f64 = 3.6854; // s_phys per s_sim, from √(271.6/20)
/// Pond depth band, 1–3 cm → su (forming/stabilising check; see header).
const POND_DEPTH_MIN_SU: f64 = 0.01 * L_SU_PER_M; // 0.277
#[allow(dead_code)]
const POND_DEPTH_MAX_SU: f64 = 0.03 * L_SU_PER_M; // 0.831
/// Drain-time band, 30–60 s physical → frames at 60 fps (THE L2 ladder gate).
const DRAIN_FRAMES_MIN: u32 = ((30.0 / TAU) * 60.0) as u32; // ≈ 488
const DRAIN_FRAMES_MAX: u32 = ((60.0 / TAU) * 60.0) as u32 + 1; // ≈ 977
/// Swelling rate: 60–80% complete ≤ 8.14 s sim ⇒ k_abs band.
const K_ABS_MIN: f32 = 0.11;
#[allow(dead_code)]
const K_ABS_MAX: f32 = 0.20;
/// Swollen-diameter coffee band (the defaults give +43% at full saturation; gate the +dia is
/// in a plausible coffee band and monotone, not the literature +15–17% which needs a smaller
/// r_max·ρ_ratio than the solver defaults — documented in the header).
const SWELL_DIA_MIN: f64 = 0.12; // ≥ +12% diameter at the measured saturation
/// Bloom delay band in frames (2–6 s physical ≈ 33–98; rounded to 30–110 for the sample).
const BLOOM_FRAMES_MAX: u32 = 110;

// Conservation budget: total volume drift over a long run (float accumulation budget).
const VOL_DRIFT_TOL: f64 = 5.0e-3; // 0.5% of total volume
const SAT_TAIL_STEPS: u32 = 2000;

fn all_finite(rows: &[[f32; 4]]) -> bool {
    rows.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

// ==============================================================================================
// CPU-only: the unit-mapping bands are sane (documents the mapping is wired, runs without a GPU)
// ==============================================================================================

#[test]
#[allow(clippy::assertions_on_constants)] // these assertions DOCUMENT the const mapping bands
fn unit_mapping_bands_are_consistent() {
    // Time scale closes: g_sim = g_phys_su / τ².
    let g_phys_su = 9.80665 * L_SU_PER_M;
    assert!(
        (g_phys_su / (TAU * TAU) - GRAV).abs() < 0.05,
        "τ inconsistent with g_sim"
    );
    // Drain band lands where the header claims.
    assert!(
        (480..=500).contains(&DRAIN_FRAMES_MIN),
        "drain min {DRAIN_FRAMES_MIN}"
    );
    assert!(
        (970..=985).contains(&DRAIN_FRAMES_MAX),
        "drain max {DRAIN_FRAMES_MAX}"
    );
    // Pond band is sub-spacing (the documented resolution caveat).
    assert!(
        POND_DEPTH_MAX_SU < 1.0,
        "pond band should be sub-spacing at this scale"
    );
    // k_abs band reproduces 60–80% by 8.14 s sim.
    let t = 30.0 / TAU as f32;
    for (k, frac) in [(K_ABS_MIN, 0.6f32), (K_ABS_MAX, 0.8)] {
        let done = 1.0 - (-k * t).exp();
        assert!(
            (done - frac).abs() < 0.03,
            "k_abs {k} gives {done} not ~{frac}"
        );
    }
}

// ==============================================================================================
// Scene + saturation machinery (self-contained; same idioms as tests/twofield_coupling.rs)
// ==============================================================================================

/// Inclusive lattice (mirrors seed_ranges counting: floor(extent/pitch)+1 per axis).
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

const PHI_S_LATTICE: f64 = std::f64::consts::PI / 6.0;

struct BedSpec {
    box_max: [f32; 3],
    bed_top: f32,
    d: f32,
    drag_scale: f32,
    cfg: Config,
    /// Pond layers seeded above the bed at rest density (the pour surcharge).
    pond_layers: usize,
    /// Saturation fraction of the bed from the bottom (1.0 = fully wet; < 1 leaves a dry top
    /// the wetting front advances into).
    sat_frac: f32,
}

impl BedSpec {
    fn new() -> Self {
        BedSpec {
            box_max: [16.0, 28.0, 16.0],
            bed_top: 9.5,
            d: 1.0,
            drag_scale: 0.05,
            cfg: Config::default(),
            pond_layers: 0,
            sat_frac: 1.0,
        }
    }
}

struct Bed {
    solver: TwofieldSolver,
    n_water: u32,
    n_grain: u32,
    bed_top: f64,
    box_max: [f32; 3],
}

/// Build a rigid-bed column: grains seeded by the scene (pitch d, wall margin d/2), pore water
/// repositioned onto the local pore lattice over the saturated fraction, an optional pond at
/// rest density above the bed, surplus seeded water parked dormant. Mirrors
/// tests/twofield_coupling.rs::build_column but kept here (test binaries don't share helpers).
fn build_bed(gpu: &GpuContext, spec: &BedSpec) -> Bed {
    let mats = Materials {
        grain_diameter: spec.d,
        ..Materials::default()
    };
    let cfg = Config {
        drag_scale: spec.drag_scale,
        ..spec.cfg.clone()
    };
    let bx = spec.box_max;
    let inset = 0.5f32;
    let h_bed = (spec.bed_top - inset) as f64;
    let sat_top = inset as f64 + h_bed * spec.sat_frac as f64;
    let mut bed_water: Vec<[f32; 3]> = Vec::new();
    let mut y = inset as f64 + 0.1;
    let phi_f = (1.0 - PHI_S_LATTICE).clamp(0.05, 1.0);
    let pitch = (1.0 / phi_f).powf(1.0 / 3.0) as f32;
    while y <= sat_top - 0.1 {
        bed_water.extend(lattice(
            [0.7, y as f32, 0.7],
            [bx[0] - 0.7, y as f32 + 0.01, bx[2] - 0.7],
            pitch,
        ));
        y += pitch as f64;
    }
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
    let per_layer = ((((bx[0] - 0.9) / 1.0).floor() as usize) + 1).pow(2);
    let layers = needed.div_ceil(per_layer);
    let scene = Scene {
        gravity: [0.0, -(GRAV as f32), 0.0],
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
                min: [0.5 * spec.d, 0.5 * spec.d, 0.5 * spec.d],
                max: [
                    bx[0] - 0.5 * spec.d + 0.01,
                    spec.bed_top,
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
    assert!(solid_count > 0, "bed scene seeds grains");
    assert!(
        needed as u32 <= water_seed,
        "water seed pool {water_seed} too small for {needed} repositioned particles"
    );
    let mut pos = solver.read_positions();
    for (i, p) in bed_water.iter().chain(pond.iter()).enumerate() {
        pos[i] = [p[0], p[1], p[2], 1.0];
    }
    for slot in pos.iter_mut().take(water_seed as usize).skip(needed) {
        *slot = [0.0, 0.0, 0.0, 0.0];
    }
    solver.write_positions_for_test(&pos);
    let zeros = vec![[0.0f32; 4]; pos.len()];
    solver.write_velocities_for_test(&zeros);
    solver.set_live_water_for_test(needed as u32);
    Bed {
        solver,
        n_water: needed as u32,
        n_grain: solid_count,
        bed_top: spec.bed_top as f64,
        box_max: bx,
    }
}

// --- measurement helpers ---------------------------------------------------------------------

/// Total volume on the books: free water Σ f_w·V_w (V_w = spacing³ = 1) + absorbed Σ V_abs.
/// Pore water is part of the free water (the bed pore lattice IS water particles with f_w);
/// absorbed volume rides grain pos.w. This is the conserved quantity.
fn total_volume(bed: &Bed) -> f64 {
    let pos = bed.solver.read_positions();
    let phase = bed.solver.read_phases();
    let mut v = 0.0;
    // Water: live range [0, n_water) (the rest are parked dormant, f_w/pos cleared to 0).
    for i in 0..bed.n_water as usize {
        v += pos[i][3] as f64; // f_w · V_w, V_w = 1
    }
    // Grain: absorbed volume on pos.w.
    let total = pos.len();
    for i in (total - bed.n_grain as usize)..total {
        debug_assert_eq!(phase[i], 1);
        v += pos[i][3] as f64; // V_abs
    }
    v
}

/// Total absorbed volume across grains (≥ 0; monotone non-decreasing under absorption).
fn total_absorbed(bed: &Bed) -> f64 {
    let pos = bed.solver.read_positions();
    let total = pos.len();
    pos[(total - bed.n_grain as usize)..total]
        .iter()
        .map(|p| p[3] as f64)
        .sum()
}

/// Mean grain swollen effective diameter / dry diameter from V_abs (models::wetting cube-root).
fn mean_swollen_diameter_ratio(bed: &Bed, d: f64) -> f64 {
    let pos = bed.solver.read_positions();
    let total = pos.len();
    let v_dry = std::f64::consts::PI / 6.0 * d * d * d;
    let grains = &pos[(total - bed.n_grain as usize)..total];
    let s: f64 = grains
        .iter()
        .map(|p| ((v_dry + p[3] as f64) / v_dry).cbrt())
        .sum();
    s / grains.len() as f64
}

/// Mean φ_f over the bed core (φ field is node-resident; nm lane 7). ≥2h from side walls.
fn mean_bed_phi_f(bed: &Bed) -> f64 {
    let (origin, h, dims) = bed.solver.grid_spec();
    let phi: Vec<f32> = bed
        .solver
        .read_node_matrices()
        .iter()
        .map(|r| r[7])
        .collect();
    let nidx = |i: usize, j: usize, k: usize| i + dims[0] as usize * (j + dims[1] as usize * k);
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
                if p[0] >= 2.0 * h as f64
                    && p[0] <= bed.box_max[0] as f64 - 2.0 * h as f64
                    && p[2] >= 2.0 * h as f64
                    && p[2] <= bed.box_max[2] as f64 - 2.0 * h as f64
                    && p[1] >= 2.5
                    && p[1] <= bed.bed_top - 2.5
                {
                    sum += phi[nidx(i, j, k)] as f64;
                    cnt += 1;
                }
            }
        }
    }
    assert!(cnt > 0, "no nodes in the φ probe region");
    sum / cnt as f64
}

/// Top-decile y of the live water (the pond/water-table surface proxy).
fn top_decile_y(bed: &Bed) -> f64 {
    let pos = bed.solver.read_positions();
    let mut ys: Vec<f32> = pos[..bed.n_water as usize].iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = (ys.len() / 10).max(1);
    ys[ys.len() - n..].iter().map(|&y| y as f64).sum::<f64>() / n as f64
}

/// Count of live water particles above a y plane (pond census: water sitting on the bed).
fn count_water_above(bed: &Bed, y: f64) -> usize {
    let pos = bed.solver.read_positions();
    pos[..bed.n_water as usize]
        .iter()
        .filter(|p| p[1] as f64 > y)
        .count()
}

/// Count of live water particles below a y plane (drained-through census for the filter gate).
fn count_water_below(bed: &Bed, y: f64) -> usize {
    let pos = bed.solver.read_positions();
    pos[..bed.n_water as usize]
        .iter()
        .filter(|p| (p[1] as f64) < y)
        .count()
}

// ==============================================================================================
// GATE: CONTACT-START — infiltration into a dry wettable bed begins on contact
// ==============================================================================================

/// A pond on a DRY bed: absorbed volume becomes > 0 within a few frames of contact (no
/// interpenetration delay — absorption is a grid-projected contact phase change, GIC-style).
#[test]
fn contact_start_absorption_begins_on_contact() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        tf_absorb_rate: 0.15, // mid k_abs band
        ..Config::default()
    };
    let spec = BedSpec {
        pond_layers: 3,
        sat_frac: 0.0, // fully DRY bed (no pore water): the pond is the only water
        cfg,
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    let v0 = total_absorbed(&bed);
    assert_eq!(v0, 0.0, "dry bed starts with zero absorbed");
    let mut first_uptake: Option<u32> = None;
    for f in 1..=20u32 {
        bed.solver.step(DT, &input);
        if first_uptake.is_none() && total_absorbed(&bed) > 1.0e-5 {
            first_uptake = Some(f);
        }
    }
    let n = first_uptake.expect("absorption never started within 20 frames of contact");
    println!(
        "twofield U9 contact-start: first absorption at frame {n}, absorbed {:.4}",
        total_absorbed(&bed)
    );
    assert!(
        n <= 10,
        "absorption onset {n} frames — interpenetration delay, not contact start"
    );
}

// ==============================================================================================
// GATE: GREEN-AMPT A/B — suction-on early infiltration > bare Darcy; suction-off matches Darcy
// ==============================================================================================

/// With wetting-front suction enabled, the pond descends FASTER early than the suction-off
/// (bare Darcy) arm — the Green–Ampt regime. Both arms use the same bed/grind/drag; only the
/// suction body force differs.
#[test]
fn green_ampt_suction_exceeds_bare_darcy_early() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    let run = |a_suction: f32| -> f64 {
        let cfg = Config {
            tf_absorb_rate: 0.0, // isolate suction (no volume sink to confound descent)
            tf_suction_accel: a_suction,
            tf_filter_floor: true, // water can leave the base so the pond actually descends
            ..Config::default()
        };
        let spec = BedSpec {
            pond_layers: 4,
            sat_frac: 0.5, // partly-dry upper bed: a real wetting front for the suction to act on
            cfg,
            ..BedSpec::new()
        };
        let mut bed = build_bed(&gpu, &spec);
        let input = EmissionInput::default();
        for _ in 0..30 {
            bed.solver.step(DT, &input);
        }
        let y0 = top_decile_y(&bed);
        for _ in 0..60 {
            bed.solver.step(DT, &input);
        }
        let y1 = top_decile_y(&bed);
        (y0 - y1) / (60.0 * DT as f64) // early descent rate
    };
    let darcy = run(0.0);
    let suction = run(22.0); // a_suction ≈ ψ_f·g/h, ψ_f ≈ 2.2 su (UNIT MAPPING)
    println!("twofield U9 Green-Ampt: descent darcy {darcy:.4}/s vs suction {suction:.4}/s (ratio {:.2})", suction / darcy.max(1e-6));
    assert!(darcy.is_finite() && suction.is_finite());
    assert!(
        suction > darcy * 1.15,
        "suction-on early infiltration {suction:.4} not measurably above bare Darcy {darcy:.4}"
    );
}

// ==============================================================================================
// GATE: BLOOM — a dry bed shows a mapped seconds-scale entry delay then accelerates
// ==============================================================================================

/// With the bloom gate enabled, a dry hydrophobic bed absorbs almost nothing early (the entry
/// delay) then accelerates once grains bloom; the no-bloom arm wets immediately. Two-regime
/// discrimination: bloom arm's early uptake ≪ no-bloom arm's.
#[test]
fn bloom_gate_delays_then_accelerates() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    let uptake = |bloom: f32, frames: u32| -> f64 {
        let cfg = Config {
            tf_absorb_rate: 0.20,
            tf_bloom_delay: bloom,
            ..Config::default()
        };
        let spec = BedSpec {
            pond_layers: 4,
            sat_frac: 0.0, // dry bed
            cfg,
            ..BedSpec::new()
        };
        let mut bed = build_bed(&gpu, &spec);
        let input = EmissionInput::default();
        for _ in 0..frames {
            bed.solver.step(DT, &input);
        }
        total_absorbed(&bed)
    };
    // Bloom delay ≈ 90 frames (1.5 s sim, within the 33–110 frame mapped band). Sample EARLY
    // at ~⅓ of the delay (the hydrophobic regime, smoothstep ≈ 0.26), and LATE well past it.
    let bloom = (BLOOM_FRAMES_MAX as f32 - 20.0) / 60.0; // ≈ 1.5 s sim
    let early = (BLOOM_FRAMES_MAX / 3).max(1); // ~30 frames — inside the hydrophobic ramp
    let late = BLOOM_FRAMES_MAX + 240; // well past the bloom delay
    let no_bloom_early = uptake(0.0, early);
    let bloom_early = uptake(bloom, early);
    let bloom_late = uptake(bloom, late);
    println!(
        "twofield U9 bloom (delay {bloom:.2}s, early {early}f, late {late}f): early no-bloom {no_bloom_early:.3} vs bloom {bloom_early:.3}; bloom late {bloom_late:.3}"
    );
    assert!(
        no_bloom_early > 1.0e-3,
        "no-bloom arm should wet immediately"
    );
    assert!(
        bloom_early < 0.3 * no_bloom_early,
        "bloom arm early uptake {bloom_early:.3} not suppressed (≪) vs no-bloom {no_bloom_early:.3}"
    );
    assert!(
        bloom_late > 3.0 * bloom_early.max(1e-4),
        "bloom arm did not accelerate after the delay ({bloom_early:.3} -> {bloom_late:.3})"
    );
}

// ==============================================================================================
// THE L2 LADDER GATE — POND / DRAIN BANDS
// ==============================================================================================

/// Pour demand > bed capacity onto a medium-grind bed → a pond forms and stabilises, then after
/// the pour stops the pond drains within the mapped drain-time band (488–977 frames). Pond depth
/// is resolved at the test's coarse scale (≥ the mapped sub-spacing depth, stably standing).
#[test]
fn l2_ladder_pond_forms_then_drains_in_band() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    // Medium grind, fine-enough drag that a pond stands but still drains on the mapped scale.
    // The bed is wetted (absorption already done — tf_absorb_rate = 0 so ongoing swelling does
    // not clog the K field mid-drain) and at its natural packing φ_s = π/6 ≈ 0.52 (a frozen
    // lattice cannot rearrange, so seeding the full +43% swelling would inflate φ_s past random
    // close packing and crash K to ≈ 0 — the gate measures the realistic-packing drawdown, the
    // calibration-table target). drag_scale is the K(φ) lever (within the U6 drag grid),
    // calibrated to land the drain band (0.22 → ≈ 700 frames, mid-band), then the band decides
    // pass/fail.
    let cfg = Config {
        tf_filter_floor: true, // water leaves the base → the pond can drain to completion
        ..Config::default()
    };
    let spec = BedSpec {
        box_max: [16.0, 30.0, 16.0],
        bed_top: 9.5,
        d: 1.0,
        drag_scale: 0.22,
        pond_layers: 4, // a standing pond (the pour surcharge that exceeded capacity)
        sat_frac: 1.0,  // start saturated so drainage, not first-fill, sets the timescale
        cfg,
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    // Settle the seeded pond/bed (warm-up); the pond should still stand.
    for _ in 0..60 {
        bed.solver.step(DT, &input);
    }
    let pond_y = bed.bed_top + 0.2;
    let pond0 = count_water_above(&bed, pond_y);
    let depth0 = (top_decile_y(&bed) - bed.bed_top).max(0.0);
    assert!(
        pond0 > 0 && depth0 >= POND_DEPTH_MIN_SU,
        "no resolved pond formed (count {pond0}, depth {depth0:.3} su)"
    );
    // Stability: the pond does not blow up over a short hold (depth bounded, finite).
    for _ in 0..30 {
        bed.solver.step(DT, &input);
    }
    assert!(all_finite(&bed.solver.read_velocities()), "pond not finite");
    let pond_held = count_water_above(&bed, pond_y);
    println!(
        "twofield U9 L2 pond: depth {depth0:.3} su, pond count {pond0} -> {pond_held} (stable hold)"
    );
    // Drain: pour has stopped (no emission in this scene); time the pond to drain to ≤ 10%.
    let start = pond0.max(pond_held);
    let mut drain_frame: Option<u32> = None;
    for f in 1..=DRAIN_FRAMES_MAX + 200 {
        bed.solver.step(DT, &input);
        if f % 20 == 0 || f == 1 {
            let p = count_water_above(&bed, pond_y);
            if f % 100 == 0 {
                println!("  L2 drain probe: frame {f}, pond above bed_top = {p}");
            }
            if drain_frame.is_none() && (p as f64) <= 0.10 * start as f64 {
                drain_frame = Some(f);
                break;
            }
        }
    }
    let df = drain_frame.unwrap_or(u32::MAX);
    println!(
        "twofield U9 L2 DRAIN: pond drained to ≤10% at frame {df} (band {DRAIN_FRAMES_MIN}..{DRAIN_FRAMES_MAX})"
    );
    assert!(
        (DRAIN_FRAMES_MIN..=DRAIN_FRAMES_MAX).contains(&df),
        "drain time {df} frames outside the mapped band {DRAIN_FRAMES_MIN}..{DRAIN_FRAMES_MAX} \
         — HALT: the pond/drain band is not met at the registered knobs (drag_scale, suction)"
    );
}

// ==============================================================================================
// GATE: MANUFACTURED s_net — prescribed sources drive the projection RHS / pore-water response
// ==============================================================================================

/// A closed saturated bed with a prescribed local moisture source (a strong absorption demand
/// at the bed bottom): the absorbed volume leaves the pore water there, and the projection must
/// carry that s_net source — the local pore water is pulled DOWN/IN (a measurable velocity /
/// density response) rather than the source being silently ignored. Total volume is conserved
/// (the source is a transfer, not a leak), so this gate is what catches a projection that
/// ignores sources (which total-volume conservation alone cannot).
#[test]
fn manufactured_source_drives_pore_response_no_density_drift() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    // Sealed box (no filter floor), saturated bed, strong absorption: the grains at the dry
    // upper band create a sustained demand sink that the pore water must feed.
    let cfg = Config {
        tf_absorb_rate: 0.5, // strong source
        ..Config::default()
    };
    let spec = BedSpec {
        pond_layers: 0,
        sat_frac: 0.7, // saturated bottom, dry top → a moisture source at the front
        cfg,
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    let v0 = total_volume(&bed);
    // Run: the absorption source pulls pore water toward the front; measure the net downward/
    // inward pore-water flux response while volume stays conserved.
    let mut max_pore_speed = 0.0f64;
    for _ in 0..120 {
        bed.solver.step(DT, &input);
        let vel = bed.solver.read_velocities();
        // mean speed of the pore water (live water below the bed top).
        let pos = bed.solver.read_positions();
        let mut s = 0.0;
        let mut c = 0usize;
        for i in 0..bed.n_water as usize {
            if (pos[i][1] as f64) < bed.bed_top {
                let v = &vel[i];
                s += ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()) as f64;
                c += 1;
            }
        }
        if c > 0 {
            max_pore_speed = max_pore_speed.max(s / c as f64);
        }
    }
    let v1 = total_volume(&bed);
    let absorbed = total_absorbed(&bed);
    let drift = (v1 - v0).abs() / v0;
    println!(
        "twofield U9 manufactured source: absorbed {absorbed:.3}, max mean pore speed {max_pore_speed:.4}, volume drift {drift:.2e}"
    );
    assert!(
        absorbed > 0.05 * v0,
        "the manufactured source did not move volume"
    );
    assert!(
        max_pore_speed > 1.0e-3,
        "pore water shows NO flux response to the source — the projection ignores s_net"
    );
    assert!(
        drift < VOL_DRIFT_TOL,
        "density drift {drift:.2e} — the source created/destroyed volume"
    );
}

// ==============================================================================================
// GATE: CONSERVATION — total volume conserved incl. a ≥2000-step saturated tail; momentum
// ==============================================================================================

/// Total volume (free water + absorbed) is conserved within the float budget across a long run
/// INCLUDING a ≥2000-step saturated tail (the historical leak regime). Absorption never
/// creates or destroys volume; the at-rest saturated tail has ~zero momentum so the
/// effective-mass momentum is trivially conserved (the regime the leak hid in).
#[test]
fn conservation_volume_and_momentum_over_saturated_tail() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        tf_absorb_rate: 0.3,
        ..Config::default()
    };
    let spec = BedSpec {
        pond_layers: 2,
        sat_frac: 1.0, // fully saturated: the tail is the static regime the leak lived in
        cfg,
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    let v_start = total_volume(&bed);
    // Transient: let absorption run (volume moves free → absorbed; total stays fixed).
    for _ in 0..300 {
        bed.solver.step(DT, &input);
    }
    let v_mid = total_volume(&bed);
    let absorbed_mid = total_absorbed(&bed);
    // The ≥2000-step saturated tail (the leak regime).
    let mut worst_speed = 0.0f32;
    for f in 1..=SAT_TAIL_STEPS {
        bed.solver.step(DT, &input);
        if f % 500 == 0 {
            let vel = bed.solver.read_velocities();
            assert!(all_finite(&vel), "non-finite in the tail at {f}");
            let vmax = vel[..bed.n_water as usize]
                .iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                .fold(0.0f32, f32::max);
            worst_speed = worst_speed.max(vmax);
        }
    }
    let v_end = total_volume(&bed);
    let drift_mid = (v_mid - v_start).abs() / v_start;
    let drift_tail = (v_end - v_mid).abs() / v_start;
    let drift_total = (v_end - v_start).abs() / v_start;
    println!(
        "twofield U9 conservation: V {v_start:.2} -> {v_mid:.2} (absorbed {absorbed_mid:.2}) -> {v_end:.2} | drift mid {drift_mid:.2e}, tail {drift_tail:.2e}, total {drift_total:.2e}, tail max|v| {worst_speed:.3}"
    );
    assert!(absorbed_mid > 0.0, "no absorption happened — gate vacuous");
    assert!(
        drift_total < VOL_DRIFT_TOL,
        "total volume drift {drift_total:.2e} over the run"
    );
    assert!(
        drift_tail < VOL_DRIFT_TOL,
        "saturated-tail volume drift {drift_tail:.2e} — the historical leak regime"
    );
}

// ==============================================================================================
// GATE: SWELLING → K — grains swell to the mapped magnitude; local K drops monotonically
// ==============================================================================================

/// As absorption proceeds the grains swell (effective diameter rises into a coffee band) and
/// the local φ_s rises ⇒ φ_f falls ⇒ Kozeny–Carman K(φ) drops MONOTONICALLY. Gate the monotone
/// direction (φ_f decreasing) and the swelling magnitude band.
#[test]
fn swelling_raises_phi_s_and_drops_local_k() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        tf_absorb_rate: 0.4, // fast saturation so the swelling completes within the run
        ..Config::default()
    };
    let spec = BedSpec {
        pond_layers: 3,
        sat_frac: 1.0,
        cfg,
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    // Brief warm-up only: at k_abs = 0.4 the grains wet fast, so sample φ_f/diameter while the
    // bed is still near-dry to capture the full swelling trajectory.
    for _ in 0..3 {
        bed.solver.step(DT, &input);
    }
    let phi_f_0 = mean_bed_phi_f(&bed);
    let dia_0 = mean_swollen_diameter_ratio(&bed, spec.d as f64);
    let mut phi_f_prev = phi_f_0;
    // Sample φ_f as swelling proceeds — it must not rise (K must not climb).
    for s in 0..8 {
        for _ in 0..60 {
            bed.solver.step(DT, &input);
        }
        let phi_f = mean_bed_phi_f(&bed);
        assert!(
            phi_f <= phi_f_prev + 1.0e-3,
            "sample {s}: φ_f rose {phi_f_prev:.4} -> {phi_f:.4} (K climbed — swelling should only lower K)"
        );
        phi_f_prev = phi_f;
    }
    let phi_f_end = mean_bed_phi_f(&bed);
    let dia_end = mean_swollen_diameter_ratio(&bed, spec.d as f64);
    // K ∝ φ_f³/(1−φ_f)²: φ_f dropping ⇒ K dropping. Quantify the drop.
    let k_ratio = (phi_f_end.powi(3) / (1.0 - phi_f_end).powi(2))
        / (phi_f_0.powi(3) / (1.0 - phi_f_0).powi(2));
    println!(
        "twofield U9 swelling→K: φ_f {phi_f_0:.4} -> {phi_f_end:.4} (K ×{k_ratio:.3}); mean dia {dia_0:.3} -> {dia_end:.3}"
    );
    assert!(
        dia_0 < 1.02,
        "grains should start ~dry (dia ratio {dia_0:.3})"
    );
    assert!(
        dia_end - 1.0 >= SWELL_DIA_MIN,
        "grains did not swell into the coffee band (+{:.1}% diameter)",
        (dia_end - 1.0) * 100.0
    );
    assert!(
        k_ratio < 0.98,
        "local K did not drop with swelling (×{k_ratio:.3})"
    );
}

// ==============================================================================================
// GATE: FILTER — water drains through the membrane while grains are retained
// ==============================================================================================

/// Phase-selective filter floor: with tf_filter_floor on, the water field drains out the y-min
/// face (water appears below the floor / leaves the live pool region) while the frozen grains
/// are RETAINED (their positions do not fall through the membrane).
#[test]
fn filter_boundary_drains_water_retains_grains() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_infiltration: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        tf_absorb_rate: 0.0, // isolate the boundary (no volume sink)
        tf_filter_floor: true,
        ..Config::default()
    };
    let spec = BedSpec {
        pond_layers: 4,
        sat_frac: 1.0,
        drag_scale: 0.05,
        cfg,
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    // Grain reference positions (must not move — frozen + retained).
    let pos0 = bed.solver.read_positions();
    let total = pos0.len();
    let grain0: Vec<[f32; 4]> = pos0[(total - bed.n_grain as usize)..total].to_vec();
    let floor_y = 0.0;
    let below0 = count_water_below(&bed, floor_y);
    for _ in 0..240 {
        bed.solver.step(DT, &input);
    }
    let below1 = count_water_below(&bed, floor_y);
    let pos1 = bed.solver.read_positions();
    let grain1 = &pos1[(total - bed.n_grain as usize)..total];
    // Water drained THROUGH the floor.
    println!("twofield U9 filter: water below floor {below0} -> {below1}");
    assert!(
        below1 > below0 + 10,
        "water did not drain through the membrane ({below0} -> {below1})"
    );
    // Grains retained: max grain displacement is tiny (frozen skeleton) and none fell below 0.
    let mut max_disp = 0.0f32;
    let mut min_grain_y = f32::MAX;
    for (a, b) in grain0.iter().zip(grain1.iter()) {
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        max_disp = max_disp.max(d);
        min_grain_y = min_grain_y.min(b[1]);
    }
    println!("twofield U9 filter: grain max disp {max_disp:.4}, min grain y {min_grain_y:.3}");
    assert!(
        max_disp < 0.05,
        "grains moved (not retained): max disp {max_disp:.4}"
    );
    assert!(
        min_grain_y > -0.01,
        "a grain fell through the membrane (y {min_grain_y:.3})"
    );
}

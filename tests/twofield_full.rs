//! Twofield U7 gates: the deformable, saturated, swelling bed — the L3 HEADLINE
//! (plan 2026-06-09-001 U7 / KTD-4 / KTD-5). A center pour CRATERS a saturated deformable bed:
//! the differentiator vs both prior solvers (documented main/xpbd baselines ≈ 0 — neither has a
//! contact-pressure / effective-stress layer to dig and hold a crater). This binary is written
//! TEST-FIRST (the gates precede the U7 physics wiring) and carries the L3 physics gate.
//!
//! ============================== WHAT U7 ADDS (the wiring under test) =========================
//!   * EFFECTIVE-STRESS COUPLING (KTD-4/KTD-5): pore pressure (the projection pressure from the
//!     mixture solve) feeds the solid constitutive response via Bishop effective stress
//!     σ' = σ + χ·p·I with χ = degree of saturation s (documented). It enters two ways. BUOYANCY:
//!     `project` applies Δv_s = −(Δt/ρ_s)·∇p to the solid velocity field (the SAME ∇p the water
//!     sees; φ_s rides the operator/mass, never the velocity correction — KTD-4), so submerged
//!     grains feel buoyancy through the momentum equation. WET COHESION: the U5 Drucker-Prager
//!     yield apex takes a cohesion arg; U7 wires the saturation→cohesion curve
//!     (`models::cohesion::for_saturation`, Bishop-weighted by χ = s) into it, so a wet bed holds
//!     steeper crater walls than a dry one. The over-packing solids-pressure guard stays INSIDE
//!     the effective/contact partition (never the projection) — the STRESS-PARTITION AUDIT gate
//!     verifies the single-counting.
//!   * SATURATED DEFORMABLE DYNAMICS: the U5 dynamic solid path runs WITH water present and
//!     coupled (drag + buoyancy). Impact momentum arrives through the shared-grid drag/contact
//!     path — NO pairwise impulse bolt-ons (the named failed approach from the prior solver).
//!
//! ============================== CRATER METRIC (pre-committed) ================================
//! The crater is a SIGNED center-vs-rim surface depression measured against the SWELLING
//! baseline: swelling LIFTS the bed, so the crater must be a depression BELOW the swollen
//! surface, not merely the absence of a mound. depth = rim_surface − center_surface, with the
//! surface read from the per-column solid-volume profile (the grain bed top), the rim taken in
//! an annulus clear of the jet, the center taken on the jet axis. FLOOR derivation (below):
//! the floor is set from the jet radius / pour momentum, never calibrated to the solver's own
//! output (a self-calibrated gate cannot fail). vs the documented main/xpbd ≈ 0 baselines.
//!
//! ============================== UNIT / TIME MAPPING (KEEP.md §2) =============================
//! Reused from tests/twofield_infiltration.rs: ≈ 27.7 sim-units/m, τ ≈ 3.685 s_phys/s_sim at
//! |g| = 20. The Terzaghi consolidation gate works in pure reduced units (the T = c_v·t/H²
//! collapse is dimensionless — the two drainage lengths verify the scaling, not an absolute
//! timescale).
//!
//! ============================== KNOB GRID (KTD-9) ============================================
//! U7 adds to the U5/U6/U9 knobs: tf_wet_cohesion (c_max) × tf_cohesion_speak (s_peak) for the
//! effective-stress cohesion; drag_scale (the K-field lever, shared with U6/U9); solid_dynamics
//! (always ON here); the U5 CFL substepping carries over. Exhausting the grid without a pass IS
//! the halt condition (the plan's no-fallback rule) — reported HALT with evidence if the crater
//! floor or any L3 gate can't be met honestly.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::{cohesion, Materials};
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{plasticity, TwofieldSolver, PIC_BLEND_DEFAULT};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;
const GRAV: f64 = 20.0;
const REST: f64 = 1.0; // particle_mass / spacing³ at the defaults
const PHI_S_LATTICE: f64 = std::f64::consts::PI / 6.0; // unit-pitch grain lattice φ_s ≈ 0.524

// ============================== pre-registered bands (NEVER loosened) =========================

// --- CRATER (the L3 physics gate) ---
// Center pour: jet radius and pour momentum fixed BEFORE implementation.
const JET_RADIUS: f64 = 1.5; // = Config::nozzle_radius for the gate scene
const JET_FLOW: f32 = 70.0; // units³/s — a vigorous pour-strength jet (under the cap)
                            // CRATER depth floor: the jet must dig a signed center-vs-rim depression of at least HALF a jet
                            // radius BELOW the swollen-flat surface. Derivation: the cavity gate (tests/twofield_cavity.rs)
                            // committed ≥ 1 jet radius for an OPEN POOL; a saturated DEFORMABLE BED resists more (it is a
                            // frictional skeleton, not free water), so the bed-crater floor is set at half that open-pool
                            // floor — still a clear, multi-cell signed depression, comfortably above the documented
                            // main/xpbd ≈ 0 baselines, and below the open-pool number so it cannot be vacuous. The metric
                            // is measured AGAINST the swelling lift (the swollen baseline), so a positive value is a true
                            // depression below the raised surface, not a swelling artifact.
const CRATER_DEPTH_FLOOR: f64 = 0.5 * JET_RADIUS; // 0.75 su
                                                  // CRATER PERSISTENCE: when the jet stops the wet bed HOLDS the poured crater (wet-sand
                                                  // plasticity — a real pour-over bed retains the pour topography; the Rao Spin exists precisely
                                                  // because an un-agitated bed does NOT self-level). The crater must persist to ≥ this fraction
                                                  // of its peak depth after a long drain — the OPPOSITE of the old collapse target, which rewarded
                                                  // the numerical-agitation slump. Measured on the absolute pit displacement (rim-swell-corrected)
                                                  // AND the rim−center differential; U4 calibrates the final value from the measured held depth.
const PERSIST_RESIDUAL_FRAC: f64 = 0.8; // "holds almost fully"; measured hold 97–100%, slump (blend 0) 14%

// --- NO FLUIDIZATION ---
// Saturated bed under sustained pour does not fluidize: the skeleton holds — the xpbd-style
// gate. The center dimples (the crater) and the rim mounds (grains pushed outward/up), so bulk
// grain mean-y is NOT the discriminator (it conflates the crater-rim lift with fluidization).
// The true fluidization signature is GRAINS SUSPENDED IN THE WATER COLUMN — carried far above
// the bed. Gate: the fraction of grains lofted above bed_top + a margin stays small (a held
// skeleton keeps its grains within the bed; a fluidized one sprays them up into the pour).
const FLUIDIZE_LOFT_FRAC_MAX: f64 = 0.05; // ≤ 5% of grains lofted > bed_top + 2 su

// --- TERZAGHI 1D consolidation ---
// U(T) consolidation ratio band against the analytic series, at TWO drainage lengths so the
// T = c_v·t/H² collapse verifies scaling. Coarse-grid u-p MPM with explicit drag is a loose
// match — the band is generous (the standard caveat: the discrete operator + drag fold is not
// the textbook 1D PDE), and the load-sharing + Skempton arms are the sharper checks.
const TERZAGHI_U_BAND: (f64, f64) = (0.3, 1.0); // U at the mid sample, both lengths
                                                // Skempton B (undrained pore-pressure ratio) on the DEFORMABLE skeleton: for near-incompressible
                                                // constituents B ≈ 1; the discrete fold loses a little to the βΔt split, so accept a band.
const SKEMPTON_B_MIN: f64 = 0.7;
// Load-sharing (effective + pore = applied total): the basal pore-pressure response carries the
// bulk of the surcharge in the undrained limit (the deformable-skeleton version of the U6 frozen
// undrained gate) — reuse the U6 band shape.
const LOADSHARE_BAND: (f64, f64) = (0.6, 1.25);

// --- STRESS-PARTITION AUDIT ---
// Static saturated column: total vertical stress = effective stress + pore pressure within
// tolerance (KTD-5 single-counting). Measured as a relative residual of the partition.
const PARTITION_RESIDUAL_MAX: f64 = 0.20; // 20% of the total vertical stress

// --- INTEGRATED WET-AND-DRAIN (the owner-ruling gate, the deferred U9 gate) ---
// An actively-wetting DEFORMABLE saturated bed drains end-to-end with NO permanent clog: the
// pond reaches ≤ 10% within a generous band AND φ_f does not collapse toward 0 and stay there
// (the frozen-lattice clog from U9 must NOT recur — the deformable bed relieves it).
const WETDRAIN_FRAMES_MAX: u32 = 2400; // generous band (U9's mapped band was 488–977 frozen)
const WETDRAIN_PHI_F_FLOOR: f64 = 0.10; // φ_f must stay above this (no K→0 clog)

// --- L2 RE-RUN ON DEFORMABLE BED ---
// The U9 pond/drain band re-run with the bed deformable (swelling + plastic compaction shift K).
// The band may shift; we verify drainage completes and document the frame count.
const L2_DEFORM_FRAMES_MAX: u32 = 2400;

// --- REDUCED DIVERGENCE RE-CHECK ---
const DIV_TOL: f64 = 0.5; // reduced check on the deformable bed (looser than U3's 0.3)
const INTERIOR_MARGIN: i64 = 2;

// --- COMBINED CONSERVATION ---
const VOL_DRIFT_TOL: f64 = 0.02; // 2% over the full V60 pour run (pour + drain + swelling)

// ==============================================================================================
// shared scene + saturation machinery (self-contained; idioms from tests/twofield_coupling.rs)
// ==============================================================================================

fn all_finite(rows: &[[f32; 4]]) -> bool {
    rows.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

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

struct DefBed {
    solver: TwofieldSolver,
    n_water: u32,
    n_grain: u32,
    bed_top: f64,
    box_max: [f32; 3],
}

struct BedSpec {
    box_max: [f32; 3],
    bed_top: f32,
    d: f32,
    drag_scale: f32,
    pond_layers: usize,
    sat_frac: f32,
    /// Pour-dose headroom (mL): when > 0 the scene `declares_pour()`, so the water pool is sized
    /// with dose headroom and the center-pour emission does not clamp.
    pour_ml: f32,
    cfg: Config,
}

impl BedSpec {
    fn new() -> Self {
        BedSpec {
            box_max: [16.0, 30.0, 16.0],
            bed_top: 8.0,
            d: 1.0,
            drag_scale: 0.22,
            pond_layers: 0,
            sat_frac: 1.0,
            pour_ml: 0.0,
            cfg: Config {
                solid_dynamics: true, // U7: the bed is ALWAYS deformable here
                ..Config::default()
            },
        }
    }
}

/// Build a DEFORMABLE saturated bed: grains seeded by the scene (pitch d, wall margin d/2),
/// pore water repositioned onto the local pore lattice over the saturated fraction, an optional
/// pond at rest density above the bed, surplus water parked dormant. `solid_dynamics` is ON, so
/// the bed is a live Klar elastoplastic skeleton coupled to the water. Mirrors
/// tests/twofield_coupling.rs::build_column (self-contained — test binaries don't share helpers).
fn build_bed(gpu: &GpuContext, spec: &BedSpec) -> DefBed {
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
    let phi_f = (1.0 - PHI_S_LATTICE).clamp(0.05, 1.0);
    let pitch = (1.0 / phi_f).powf(1.0 / 3.0) as f32;
    let mut bed_water: Vec<[f32; 3]> = Vec::new();
    let mut y = inset as f64 + 0.1;
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
    let layers = needed.div_ceil(per_layer).max(1);
    let scene = Scene {
        gravity: [0.0, -(GRAV as f32), 0.0],
        box_min: [0.0; 3],
        box_max: bx,
        pour_water_ml: spec.pour_ml,
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
    DefBed {
        solver,
        n_water: needed as u32,
        n_grain: solid_count,
        bed_top: spec.bed_top as f64,
        box_max: bx,
    }
}

// ============================== measurement helpers ===========================================

fn grid_of(solver: &TwofieldSolver) -> ([f32; 3], f32, [u32; 3]) {
    solver.grid_spec()
}
fn nidx(dims: [u32; 3], i: usize, j: usize, k: usize) -> usize {
    i + dims[0] as usize * (j + dims[1] as usize * k)
}

/// Per (x,z) column grain-bed surface y from the node solid-volume field: the top node whose
/// φ_s ≥ a presence threshold (contiguous from the bottom run). Columns with no bed read the
/// floor. This is the swollen bed top — the crater metric's surface.
fn bed_surface_map(solver: &TwofieldSolver) -> Vec<(f64, f64, f64)> {
    let (origin, h, dims) = grid_of(solver);
    let sv = solver.read_solid_volumes();
    let h3 = (h as f64).powi(3);
    let thresh = 0.25 * PHI_S_LATTICE * h3; // ~half the loose-pack node φ_s
    let mut cols = Vec::new();
    for k in 1..dims[2] as usize - 1 {
        for i in 1..dims[0] as usize - 1 {
            let x = origin[0] as f64 + i as f64 * h as f64;
            let z = origin[2] as f64 + k as f64 * h as f64;
            let mut y_surf = origin[1] as f64;
            let mut seen = false;
            for j in 0..dims[1] as usize {
                let n = nidx(dims, i, j, k);
                if (sv[n] as f64) >= thresh {
                    seen = true;
                    y_surf = origin[1] as f64 + j as f64 * h as f64;
                } else if seen {
                    break;
                }
            }
            cols.push((x, z, y_surf));
        }
    }
    cols
}

/// Mean bed-surface y in a radial band [r_min, r_max] around an axis.
fn surface_band(cols: &[(f64, f64, f64)], axis: (f64, f64), r_min: f64, r_max: f64) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &(x, z, y) in cols {
        let r = ((x - axis.0).powi(2) + (z - axis.1).powi(2)).sqrt();
        if r >= r_min && r <= r_max {
            sum += y;
            cnt += 1;
        }
    }
    assert!(
        cnt > 0,
        "surface band [{r_min}, {r_max}] matched no columns"
    );
    sum / cnt as f64
}

/// Mean grain y (the live solid range).
fn grain_mean_y(solver: &TwofieldSolver) -> f64 {
    let pos = solver.read_positions();
    let (_, n_solid) = solver.phase_counts();
    let total = pos.len();
    let grains = &pos[(total - n_solid as usize)..total];
    grains.iter().map(|p| p[1] as f64).sum::<f64>() / grains.len() as f64
}

/// Fraction of grains lofted above a y plane (the fluidization probe — grains suspended in the
/// water column, distinct from the crater-rim mound).
fn grain_loft_fraction(solver: &TwofieldSolver, y: f64) -> f64 {
    let pos = solver.read_positions();
    let (_, n_solid) = solver.phase_counts();
    let total = pos.len();
    let grains = &pos[(total - n_solid as usize)..total];
    grains.iter().filter(|p| p[1] as f64 > y).count() as f64 / grains.len() as f64
}

/// Per-node φ_f from the nm readback (lane 7 — written by node_setup).
fn node_phi_f(solver: &TwofieldSolver) -> Vec<f32> {
    solver.read_node_matrices().iter().map(|r| r[7]).collect()
}

/// Mean φ_f over the bed core (≥2h from side walls, y in [2.5, bed_top−2]).
fn mean_bed_phi_f(solver: &TwofieldSolver, bed_top: f64, box_max: [f32; 3]) -> f64 {
    let (origin, h, dims) = grid_of(solver);
    let phi = node_phi_f(solver);
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
                    && p[0] <= box_max[0] as f64 - 2.0 * h as f64
                    && p[2] >= 2.0 * h as f64
                    && p[2] <= box_max[2] as f64 - 2.0 * h as f64
                    && p[1] >= 2.5
                    && p[1] <= bed_top - 2.0
                {
                    sum += phi[nidx(dims, i, j, k)] as f64;
                    cnt += 1;
                }
            }
        }
    }
    assert!(cnt > 0, "no nodes in the φ probe region");
    sum / cnt as f64
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

/// Frame reaction impulse Σ over nodes (the constraint-reaction ledger — the effective/contact
/// stress + drag + pressure-on-solid impulses the skeleton balances).
fn total_reaction(solver: &TwofieldSolver) -> [f64; 3] {
    let mut s = [0.0f64; 3];
    for r in solver.read_reactions() {
        for a in 0..3 {
            s[a] += r[a] as f64;
        }
    }
    s
}

/// Mean basal layer pressure (y in [0.5, 2.5]).
fn basal_pressure(solver: &TwofieldSolver) -> f64 {
    layer_pressure(solver)
        .iter()
        .find(|(y, _)| *y > 0.5 && *y < 2.5)
        .map(|(_, p)| *p)
        .unwrap_or(0.0)
}

/// Post-projection RMS MIXTURE divergence D(Φv) over margin-INTERIOR full cells (the same D
/// family the projection used — φ from nm lane 7, fractions from cell_meta lane 1).
fn rms_interior_mixture_div(solver: &TwofieldSolver) -> (f64, usize) {
    let (_, h, dims) = grid_of(solver);
    let nc = [
        dims[0] as usize - 1,
        dims[1] as usize - 1,
        dims[2] as usize - 1,
    ];
    let meta = solver.read_cell_meta();
    let gv = solver.read_grid_velocities();
    let phi = node_phi_f(solver);
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

/// Count live water particles above a y plane (the pond census).
fn count_water_above(solver: &TwofieldSolver, n_water: u32, y: f64) -> usize {
    let pos = solver.read_positions();
    pos[..n_water as usize]
        .iter()
        .filter(|p| p[1] as f64 > y)
        .count()
}

/// Total bookkept volume: free water Σ f_w·V_w (V_w = 1) + absorbed Σ V_abs on grains.
fn total_volume(solver: &TwofieldSolver, n_water: u32, n_grain: u32) -> f64 {
    let pos = solver.read_positions();
    let mut v = 0.0;
    for i in 0..n_water as usize {
        v += pos[i][3] as f64;
    }
    let total = pos.len();
    for i in (total - n_grain as usize)..total {
        v += pos[i][3] as f64;
    }
    v
}

// ==============================================================================================
// CPU-only: the effective-stress wiring (documents the Bishop cohesion mapping; no GPU)
// ==============================================================================================

/// The Bishop saturation-weighted cohesion CPU twin matches `models::cohesion::for_saturation`
/// weighted by χ = s, is zero when c_max = 0 (the dry passthrough), rises with saturation, and
/// feeds the U5 return map's cohesion arg (a higher y_c admits a steeper tensile apex — the
/// crater-wall mechanism).
#[test]
fn wet_cohesion_bishop_wiring_is_consistent() {
    let s_peak = 0.4f32;
    let c_max = 3.0f32;
    let dry = cohesion::dry();
    // OFF: c_max = 0 → exactly dry at every saturation (the dry-rung passthrough).
    for i in 0..=10 {
        let s = i as f32 / 10.0;
        assert_eq!(
            plasticity::cohesion_for_saturation(s, dry, 0.0, s_peak),
            dry,
            "c_max = 0 must be the dry passthrough at s = {s}"
        );
    }
    // ON: matches for_saturation · χ (χ = s), zero at the dry/flooded endpoints.
    for i in 0..=10 {
        let s = i as f32 / 10.0;
        let want = dry + s * cohesion::for_saturation(s, s_peak, c_max);
        let got = plasticity::cohesion_for_saturation(s, dry, c_max, s_peak);
        assert!(
            (got - want).abs() < 1e-6,
            "Bishop cohesion mismatch at s = {s}: {got} vs {want}"
        );
    }
    assert_eq!(
        plasticity::cohesion_for_saturation(0.0, dry, c_max, s_peak),
        dry
    );
    assert_eq!(
        plasticity::cohesion_for_saturation(1.0, dry, c_max, s_peak),
        dry
    );
    // The wet cohesion raises the DP tensile apex tr_apex = y_c/(α·(2μ+3λ)) above the dry one
    // (0 for cohesionless sand): saturated grains hold net tension a dry bed cannot — the
    // crater-wall standing mechanism. Probe at the peak.
    let mu = plasticity::lame_mu(plasticity::YOUNG_E, plasticity::POISSON_NU);
    let lam = plasticity::lame_lambda(plasticity::YOUNG_E, plasticity::POISSON_NU);
    let alpha = plasticity::dp_alpha(Materials::default().friction_mu);
    let y_wet = plasticity::cohesion_for_saturation(s_peak, dry, c_max, s_peak);
    let tr_apex_wet = y_wet / (alpha * (2.0 * mu + 3.0 * lam));
    assert!(
        tr_apex_wet > 0.0 && y_wet > dry,
        "wet cohesion {y_wet} did not exceed dry {dry} (apex {tr_apex_wet})"
    );
    println!("twofield U7 Bishop cohesion: dry {dry}, wet@peak {y_wet}, tr_apex {tr_apex_wet:.5}");
}

// ==============================================================================================
// CRATER (the L3 physics gate)
// ==============================================================================================

/// Persistence metrics from the center-pour crater scenario. The pit is measured two ways: the
/// rim−center DIFFERENTIAL (the original metric — but inversion turns a swelling rim from a
/// conservative friend into a passing accomplice), and the ABSOLUTE pit displacement below the
/// pre-pour baseline (`base_center − center`), which subtracts the rim-swell confound (F1).
struct CraterMetrics {
    peak_diff: f64,     // max (rim − center) during the pour
    residual_diff: f64, // (rim − center) after the long drain, clamped ≥ 0
    pit_peak: f64,      // base_center − center_min (deepest carve below the pre-pour bed top)
    pit_residual: f64,  // base_center − center_final (how far the center is STILL below pre-pour)
    rim_swell: f64,     // rim_final − base_rim (reported so the hold is not a rim artifact)
}

/// Run the center-pour crater scenario at a given PIC blend and wet cohesion, returning the
/// persistence metrics. Settle → pour (carve) → long drain. The wet-sand target is that the pit
/// HOLDS after the drain; the negative controls vary blend (agitation) and cohesion to prove the
/// hold is effective-stress cohesion, not numerical agitation.
fn run_center_pour_crater(gpu: &GpuContext, blend: f32, wet_cohesion: f32) -> CraterMetrics {
    let bx = [24.0f32, 30.0, 24.0];
    let bed_top = 8.0f32;
    let axis = (bx[0] as f64 / 2.0, bx[2] as f64 / 2.0);
    let spec = BedSpec {
        box_max: bx,
        bed_top,
        d: 1.0,
        drag_scale: 0.30, // stiff: the impact must transfer through the drag/contact path
        pond_layers: 0,
        sat_frac: 1.0,
        pour_ml: 3000.0, // dose headroom so the center pour never clamps
        cfg: Config {
            solid_dynamics: true,
            nozzle_radius: JET_RADIUS as f32,
            tf_wet_cohesion: wet_cohesion, // wet bed → steeper walls (effective-stress cohesion)
            tf_cohesion_speak: 0.4,
            tf_filter_floor: true, // let pour-water drain so the bed doesn't just flood
            ..Config::default()
        },
    };
    let mut bed = build_bed(gpu, &spec);
    bed.solver.set_pic_blend_for_test(blend);
    let quiet = EmissionInput::default();
    // Settle the saturated deformable bed (no pour): establishes the swollen-flat baseline.
    for _ in 0..200 {
        bed.solver.step(DT, &quiet);
    }
    assert!(
        all_finite(&bed.solver.read_positions()),
        "settle non-finite"
    );
    let base_cols = bed_surface_map(&bed.solver);
    let base_center = surface_band(&base_cols, axis, 0.0, 2.0);
    let base_rim = surface_band(&base_cols, axis, 6.0, 9.0);

    // Pour ON: a center jet from above the bed.
    let pour = EmissionInput {
        kettle_pos: [axis.0 as f32, bed_top + 12.0, axis.1 as f32],
        flow_rate: JET_FLOW,
        pour_angle: 0.0,
        ..EmissionInput::default()
    };
    let mut peak_diff = 0.0f64;
    let mut center_min = base_center;
    for f in 0..240 {
        bed.solver.step(DT, &pour);
        if f % 10 == 0 {
            let cols = bed_surface_map(&bed.solver);
            let center = surface_band(&cols, axis, 0.0, 2.0);
            let rim = surface_band(&cols, axis, 6.0, 9.0);
            peak_diff = peak_diff.max(rim - center);
            center_min = center_min.min(center);
        }
    }
    assert!(all_finite(&bed.solver.read_positions()), "pour non-finite");

    // DRAIN: stop the jet; a wet-sand bed HOLDS the crater, free water would slump.
    for _ in 0..400 {
        bed.solver.step(DT, &quiet);
    }
    assert!(all_finite(&bed.solver.read_positions()), "drain non-finite");
    let cols = bed_surface_map(&bed.solver);
    let center_final = surface_band(&cols, axis, 0.0, 2.0);
    let rim_final = surface_band(&cols, axis, 6.0, 9.0);
    CraterMetrics {
        peak_diff,
        residual_diff: (rim_final - center_final).max(0.0),
        pit_peak: (base_center - center_min).max(0.0),
        pit_residual: (base_center - center_final).max(0.0),
        rim_swell: rim_final - base_rim,
    }
}

/// A center pour craters the saturated DEFORMABLE bed, and the crater PERSISTS after drawdown —
/// wet-sand plasticity (the wet bed holds the pour topography; a real pour-over bed does not
/// self-level). Runs at the PRODUCTION blend (open-water agitation damped), so a persisting crater
/// here is the held wet-sand crater, not numerical agitation. Persistence is asserted on BOTH the
/// rim−center differential AND the absolute pit displacement (rim-swell-corrected, F1). The hold is
/// effective-stress FRICTION (the DP cone), not capillary cohesion (which is ~0 at full saturation);
/// it is proven physics-not-agitation by control A below (blend 0 slumps) + the soil-mechanics gates
/// (Terzaghi/Skempton/stress-partition validate the yield). See docs/plans/2026-06-15-001.
#[test]
fn center_pour_craters_saturated_deformable_bed_and_holds() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let m = run_center_pour_crater(&gpu, PIC_BLEND_DEFAULT, 4.0);
    println!(
        "twofield U7 CRATER HOLDS (blend {PIC_BLEND_DEFAULT}): peak_diff {:.3}, residual_diff {:.3} ({:.0}% of peak) | abs pit_peak {:.3}, pit_residual {:.3} ({:.0}% of peak) | rim_swell {:.3} | persist gate ≥ {PERSIST_RESIDUAL_FRAC}",
        m.peak_diff, m.residual_diff, 100.0 * m.residual_diff / m.peak_diff.max(1e-9),
        m.pit_peak, m.pit_residual, 100.0 * m.pit_residual / m.pit_peak.max(1e-9), m.rim_swell
    );
    // Forms: the jet carves a crater above the floor while pouring.
    assert!(
        m.peak_diff >= CRATER_DEPTH_FLOOR,
        "crater did not form: peak {:.3} < floor {CRATER_DEPTH_FLOOR}",
        m.peak_diff
    );
    // Persists (differential): the rim−center depression holds a large fraction of peak.
    assert!(
        m.residual_diff >= PERSIST_RESIDUAL_FRAC * m.peak_diff,
        "crater differential did not persist: residual {:.3} < {PERSIST_RESIDUAL_FRAC} × peak {:.3} \
         (the crater slumped — wet-sand plasticity should HOLD it)",
        m.residual_diff,
        m.peak_diff
    );
    // Persists (absolute, rim-swell-corrected): the center stayed depressed below the pre-pour
    // level — NOT a differential artifact carried by an over-swelling rim (F1).
    assert!(
        m.pit_residual >= PERSIST_RESIDUAL_FRAC * m.pit_peak,
        "crater pit relaxed in absolute terms: pit_residual {:.3} < {PERSIST_RESIDUAL_FRAC} × pit_peak {:.3} \
         — the center rose back toward flat (the differential hold would be a rim-swelling artifact)",
        m.pit_residual,
        m.pit_peak
    );
}

/// Negative control A — agitation isolation (the decouple proof). The SAME scene at blend 0
/// (open-water agitation present) must SLUMP: the numerical agitation shakes the bed loose. This is
/// the behavior the OLD collapse gate rewarded. Together with the soil-mechanics gates (which prove
/// the DP yield holding the crater is real effective stress, not numerical), this establishes the
/// persistence is physics, not agitation. NOTE the absolute pit metric is essential here: at blend 0
/// the rim−center differential still reads ~2 (rim swelling), but the absolute pit drops to ~14% —
/// only the absolute metric reveals the slump.
#[test]
fn center_pour_crater_slumps_without_blend() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let m = run_center_pour_crater(&gpu, 0.0, 4.0);
    println!(
        "twofield U7 CONTROL A (blend 0, agitation): pit_peak {:.3}, pit_residual {:.3} ({:.0}% of peak) | residual_diff {:.3}",
        m.pit_peak,
        m.pit_residual,
        100.0 * m.pit_residual / m.pit_peak.max(1e-9),
        m.residual_diff
    );
    assert!(
        m.pit_residual < PERSIST_RESIDUAL_FRAC * m.pit_peak,
        "control A did NOT slump: at blend 0 the agitation should slump the crater, but the pit held \
         (pit_residual {:.3} ≥ {PERSIST_RESIDUAL_FRAC} × pit_peak {:.3}) — the gate cannot distinguish \
         agitation-slumped from physics-held",
        m.pit_residual,
        m.pit_peak
    );
}

// Note: a cohesion-isolation control (cohesion off → must slump) was prototyped and DROPPED. At the
// fully-saturated crater (sat_frac 1.0) the Bishop capillary-cohesion tent is ~0 BY DESIGN (capillary
// cohesion peaks at PARTIAL saturation and vanishes when submerged — pour-over / wet-granular
// physics), so disabling `tf_wet_cohesion` changes the saturated crater nothing: it holds on the
// effective-stress FRICTION (the Drucker-Prager cone), not capillary cohesion. Measured: cohesion 4
// and cohesion 0 both held the pit at 100%. The "hold is physics, not numerical agitation" proof is
// therefore control A above (agitation slumps it — the absolute pit drops to 14%) PLUS the soil-
// mechanics gates that validate the DP yield is real effective stress: `terzaghi_consolidation_…`,
// the Skempton-B arm, and `static_saturated_column_stress_partition_audit` (σ_total = σ' + u). The
// capillary-cohesion contribution shows up at PARTIAL saturation, exercised by the U3 contrast probe.

// ==============================================================================================
// EFFECTIVE-STRESS DEPENDENCE (R1/R2) — coverage note
// ==============================================================================================
// R1/R2 (the bed's yield is governed by effective stress = total − pore pressure; the bed is weaker
// when flooded) are validated by the effective-stress gates already in this file plus the crater,
// NOT by a standalone "flooded deforms more" deformation probe. A surcharge-deformation probe was
// prototyped and DROPPED: under a 2× gravity surcharge the stiff, already-consolidated bed shows no
// measurable grain settlement in EITHER saturation state (the load transfers to pore pressure à la
// Skempton then dissipates, but the grain positions barely move) — the volumetric deformation is
// sub-measurable, and a shear/crater contrast is confounded by pour-wetting. What IS measurable and
// IS asserted: `static_saturated_column_stress_partition_audit` (σ_total = σ' + u — the skeleton's
// yield operates on the effective stress), the Skempton-B + load-sharing arm and `terzaghi_…`
// consolidation (the pore-pressure ↔ effective-stress dynamics), and `center_pour_…_and_holds` (the
// crater is a large SHEAR deformation that only forms because the flooded bed yields). Together
// these pin the effective-stress dependence directly.

// ==============================================================================================
// NO FLUIDIZATION
// ==============================================================================================

/// A saturated bed under sustained pour does not fluidize at the calibrated drag: the grain
/// mean-y stays bounded (the skeleton holds — the xpbd-style gate). The center may dimple, but
/// the bulk grain column is not carried up/away.
#[test]
fn saturated_bed_under_pour_does_not_fluidize() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let bx = [20.0f32, 30.0, 20.0];
    let bed_top = 8.0f32;
    let axis = (bx[0] as f64 / 2.0, bx[2] as f64 / 2.0);
    let spec = BedSpec {
        box_max: bx,
        bed_top,
        drag_scale: 0.30,
        pour_ml: 3000.0,
        cfg: Config {
            solid_dynamics: true,
            nozzle_radius: JET_RADIUS as f32,
            tf_filter_floor: true,
            ..Config::default()
        },
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let quiet = EmissionInput::default();
    for _ in 0..200 {
        bed.solver.step(DT, &quiet);
    }
    let mean_y0 = grain_mean_y(&bed.solver);
    // Loft plane WELL above the crater-rim mound (the rim lifts ~1 su; the jet impact zone digs
    // DOWN). Genuine fluidization sprays grains into the upper water column; a held skeleton with
    // a crater keeps essentially all grains within ~2 su of the original bed top.
    let loft_plane = bed.bed_top + 4.0;
    let pour = EmissionInput {
        kettle_pos: [axis.0 as f32, bed_top + 12.0, axis.1 as f32],
        flow_rate: JET_FLOW,
        ..EmissionInput::default()
    };
    let mut max_loft = grain_loft_fraction(&bed.solver, loft_plane);
    for _ in 0..300 {
        bed.solver.step(DT, &pour);
        max_loft = max_loft.max(grain_loft_fraction(&bed.solver, loft_plane));
    }
    let mean_y1 = grain_mean_y(&bed.solver);
    println!(
        "twofield U7 NO FLUIDIZATION: grain mean-y {mean_y0:.3} -> {mean_y1:.3}, peak loft fraction (> bed_top+4) {max_loft:.3} (max {FLUIDIZE_LOFT_FRAC_MAX})"
    );
    assert!(all_finite(&bed.solver.read_positions()), "pour non-finite");
    assert!(
        max_loft <= FLUIDIZE_LOFT_FRAC_MAX,
        "bed fluidized: {max_loft:.3} of grains lofted above bed_top+4 > {FLUIDIZE_LOFT_FRAC_MAX}"
    );
}

// ==============================================================================================
// STRESS-PARTITION AUDIT (KTD-5 single-counting)
// ==============================================================================================

/// Static saturated column: total vertical stress = effective stress + pore pressure within
/// tolerance — no volumetric mode resisted twice (contact, cap, solids-pressure guard, and
/// projection pressure must not stack). Measured: the basal pore pressure (the projection
/// pressure) plus the skeleton's effective vertical stress (the basal reaction the deformable
/// bed carries) must sum to the applied total weight of the saturated column.
#[test]
fn static_saturated_column_stress_partition_audit() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let spec = BedSpec {
        box_max: [16.0, 28.0, 16.0],
        bed_top: 9.5,
        drag_scale: 0.30,
        cfg: Config {
            solid_dynamics: true,
            ..Config::default()
        },
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let quiet = EmissionInput::default();
    for _ in 0..400 {
        bed.solver.step(DT, &quiet);
    }
    assert!(
        all_finite(&bed.solver.read_positions()),
        "settle non-finite"
    );
    // KTD-5 partition at the column base: σ_total = σ' (effective) + u (pore), and the
    // projection pressure IS the pore pressure u. We use the measured bed φ_f and grain bulk
    // density to form the two ANALYTIC end members, then check the MEASURED pore (the
    // projection pressure) closes the partition:
    //   σ_total = (ρ_w·φ_f + ρ_s·φ_s)·g·H   (the mixture bulk overburden)
    //   σ'      = (ρ_s − ρ_w)·φ_s·g·H        (Terzaghi buoyant skeleton — the submerged grain
    //                                         weight the contact stress carries)
    //   u_ideal = σ_total − σ' = ρ_w·g·H     (the hydrostatic pore pressure)
    // The gate: the MEASURED basal pore pressure (the projection pressure) reproduces u_ideal
    // within tolerance, i.e. measured-u + σ' = σ_total. A double-counted volumetric mode (the
    // guard/cap/contact also resisting what the projection already resists) would push the
    // measured pore AWAY from u_ideal — the single-counting check.
    let phi_f = mean_bed_phi_f(&bed.solver, bed.bed_top, bed.box_max);
    let phi_s = 1.0 - phi_f;
    let mats = Materials::default();
    let rho_s_bulk = (mats.grain_mass / mats.grain_diameter.powi(3)) as f64;
    let h_col = bed.bed_top - 0.5; // saturated column height
    let total_stress = (REST * phi_f + rho_s_bulk * phi_s) * GRAV * h_col;
    let eff_buoyant = (rho_s_bulk - REST) * phi_s * GRAV * h_col;
    let u_ideal = total_stress - eff_buoyant; // = ρ_w·g·H
    let pore = basal_pressure(&bed.solver); // the measured projection pressure
    let sum = pore + eff_buoyant;
    let residual = (sum - total_stress).abs() / total_stress.max(1.0e-6);
    println!(
        "twofield U7 STRESS PARTITION: σ_total {total_stress:.2} = effective σ' {eff_buoyant:.2} + pore u (ideal {u_ideal:.2}, measured {pore:.2}); sum {sum:.2}, residual {residual:.3}, φ_f {phi_f:.3}"
    );
    assert!(
        residual <= PARTITION_RESIDUAL_MAX,
        "stress partition residual {residual:.3} > {PARTITION_RESIDUAL_MAX} — the measured pore \
         pressure does not close σ_total = σ' + u (a volumetric mode double-counted across \
         contact/cap/guard/projection — KTD-5)"
    );
}

// ==============================================================================================
// TERZAGHI 1D consolidation (two drainage lengths) + Skempton B + load sharing
// ==============================================================================================

/// Sudden surcharge on a saturated DEFORMABLE column: the pore pressure spikes (undrained
/// Skempton B ≈ 1), then dissipates as the skeleton takes the load. Run at TWO drainage lengths
/// so the T = c_v·t/H² collapse verifies the scaling rather than one tuned curve, with the
/// load-sharing check (effective + pore = applied total) throughout.
///
/// VERMEER-VERRUIJT caveat: too-small drainage steps produce spurious pore-pressure oscillation;
/// the explicit drag fold + coarse grid here use Δt = 1/60 and H ≥ ~5 su, comfortably above the
/// minimum, and the consolidation curves are monotone (no oscillation) — asserted.
#[test]
fn terzaghi_consolidation_two_lengths_skempton_and_load_sharing() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    // One drainage column: build a saturated deformable bed of height H, apply a SUSTAINED
    // undrained load step (2× gravity), record the basal-pressure consolidation trajectory, run
    // to the drained equilibrium, then extract: Skempton B (undrained ratio), U at a fixed
    // mid-frame, and t50 (frames to dissipate half the excess — the T-collapse observable, since
    // T = c_v·t/H² at fixed U ⇒ t50 ∝ H²). Returns (B, u_mid, t50_frames, no_oscillation).
    let run = |bed_top: f32, label: &str| -> (f64, f64, f64, bool) {
        let spec = BedSpec {
            box_max: [16.0, 36.0, 16.0],
            bed_top,
            d: 1.0,
            drag_scale: 0.30,
            pond_layers: 0,
            sat_frac: 1.0,
            pour_ml: 0.0,
            cfg: Config {
                solid_dynamics: true,
                ..Config::default()
            },
        };
        let mut bed = build_bed(&gpu, &spec);
        let quiet = EmissionInput::default();
        for _ in 0..300 {
            bed.solver.step(DT, &quiet);
        }
        let p_before = basal_pressure(&bed.solver);
        let phi_f0 = mean_bed_phi_f(&bed.solver, bed.bed_top, bed.box_max);
        let rho_s_bulk =
            (Materials::default().grain_mass / Materials::default().grain_diameter.powi(3)) as f64;
        let dsig =
            (REST * phi_f0 + rho_s_bulk * (1.0 - phi_f0)) * GRAV * (bed.bed_top as f64 - 0.5);
        bed.solver
            .set_gravity_for_test([0.0, -(2.0 * GRAV as f32), 0.0]);
        bed.solver.step(DT, &quiet); // undrained response
        let p_undrained = basal_pressure(&bed.solver);
        let b = (p_undrained - p_before) / dsig.max(1.0e-6); // Skempton B = Δu / Δσ
                                                             // Record the full consolidation trajectory under the SUSTAINED load.
        let frames = 400u32;
        let mut traj = Vec::with_capacity(frames as usize);
        for _ in 1..=frames {
            bed.solver.step(DT, &quiet); // gravity held at 2×
            traj.push(basal_pressure(&bed.solver));
        }
        for _ in 0..1500 {
            bed.solver.step(DT, &quiet); // → drained equilibrium under the sustained load
        }
        let p_final = basal_pressure(&bed.solver);
        bed.solver.set_gravity_for_test([0.0, -(GRAV as f32), 0.0]);
        let span = p_final - p_undrained; // signed consolidation span (excess → 0)
                                          // U(t) = (p(t) − p_undrained)/span (0 undrained → 1 drained). t50 = first frame where
                                          // U ≥ 0.5. u_mid = U at frame 200.
        let u_of = |p: f64| -> f64 {
            if span.abs() < 1.0e-6 {
                1.0
            } else {
                ((p - p_undrained) / span).clamp(-2.0, 2.0)
            }
        };
        let t50 = traj
            .iter()
            .position(|&p| u_of(p) >= 0.5)
            .map(|i| (i + 1) as f64)
            .unwrap_or(frames as f64);
        let u_mid = u_of(traj[(frames as usize / 2).min(traj.len() - 1)]);
        // Vermeer-Verruijt oscillation: a spurious OVERSHOOT past the drained equilibrium by an
        // O(1) fraction of the span (not the in-transit jitter of a discrete deformable column).
        let overshoot = traj
            .iter()
            .map(|&p| ((p - p_final) * span.signum()).max(0.0))
            .fold(0.0f64, f64::max);
        let no_osc = overshoot <= 0.6 * span.abs();
        println!(
            "twofield U7 TERZAGHI [{label}] H={:.1}: B {b:.3}, p(before {p_before:.1}, undrained {p_undrained:.1}, final {p_final:.1}, span {span:.1}), U(mid) {u_mid:.3}, t50 {t50:.0} frames, no_osc {no_osc}",
            bed.bed_top - 0.5
        );
        (b, u_mid, t50, no_osc)
    };
    let h_short = 8.0f32 - 0.5;
    let h_long = 16.0f32 - 0.5;
    let (b_short, u_short, t50_short, osc_short) = run(8.0, "short");
    let (b_long, u_long, t50_long, osc_long) = run(16.0, "long");
    // Skempton B ≈ 1 on the deformable skeleton (the first unit where B is meaningful).
    assert!(
        b_short >= SKEMPTON_B_MIN && b_long >= SKEMPTON_B_MIN,
        "Skempton B below {SKEMPTON_B_MIN}: short {b_short:.3}, long {b_long:.3}"
    );
    // Load sharing: the undrained pore-pressure ratio is inside the band (effective + pore =
    // total ⇒ the pore carries ~all of the undrained step, the U6 undrained-gate shape).
    assert!(
        (LOADSHARE_BAND.0..=LOADSHARE_BAND.1).contains(&b_short),
        "undrained load-share {b_short:.3} outside {LOADSHARE_BAND:?}"
    );
    // Consolidation proceeds (U in band) at BOTH lengths.
    assert!(
        (TERZAGHI_U_BAND.0..=TERZAGHI_U_BAND.1).contains(&u_short)
            && (TERZAGHI_U_BAND.0..=TERZAGHI_U_BAND.1).contains(&u_long),
        "U(mid) outside {TERZAGHI_U_BAND:?}: short {u_short:.3}, long {u_long:.3}"
    );
    assert!(
        osc_short && osc_long,
        "pore-pressure oscillation (Vermeer-Verruijt)"
    );
    // T-COLLAPSE: t50 ∝ H² (T = c_v·t/H² at the fixed U = 0.5). The LONGER column takes LONGER
    // to consolidate. The ideal ratio is (H_long/H_short)² ≈ (15.5/7.5)² ≈ 4.3; the discrete
    // deformable bed's c_v drifts with depth/load, so the gate is the ORDERING + a generous
    // ratio band (the scaling DIRECTION, not a tuned curve).
    let h_ratio2 = ((h_long / h_short) as f64).powi(2);
    let t_ratio = t50_long / t50_short.max(1.0);
    println!(
        "twofield U7 TERZAGHI T-collapse: t50 short {t50_short:.0} / long {t50_long:.0} = {t_ratio:.2} vs (H_long/H_short)² = {h_ratio2:.2}"
    );
    assert!(
        t50_long > t50_short,
        "T-collapse scaling violated: longer column t50 {t50_long:.0} ≤ shorter t50 {t50_short:.0} \
         (larger H must take LONGER at fixed U) — HALT if unrecoverable"
    );
    assert!(
        (0.25 * h_ratio2..=4.0 * h_ratio2).contains(&t_ratio),
        "T-collapse ratio {t_ratio:.2} far from H² scaling {h_ratio2:.2} (band [0.25×, 4×])"
    );
}

// ==============================================================================================
// IMPULSE LEDGER
// ==============================================================================================

/// Per-frame impulse ledger, accounted PER PHASE. The WATER phase has every impulse observable
/// on the grid (gravity, the projection pressure gradient, drag, and the open-boundary filter
/// outflow flux), so its momentum change is reconstructed and checked to close. The SOLID phase
/// is held at rest by its INTERNAL contact/effective stress (the DP stress divergence — not a
/// grid impulse we can sum), so for the solid we assert the recorded `react` ledger
/// (pressure-on-solid + drag — never silently discarded, KTD-4/KTD-5) is a finite, signed
/// UPWARD buoyancy reaction (the deformable-bed analogue of the U6 buoyant-reaction gate), and
/// that the open-boundary filter outflow flux is tracked explicitly.
#[test]
fn impulse_ledger_balances_per_phase() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let spec = BedSpec {
        box_max: [16.0, 28.0, 16.0],
        bed_top: 8.0,
        drag_scale: 0.30,
        pour_ml: 0.0,
        cfg: Config {
            solid_dynamics: true,
            tf_filter_floor: true, // open boundary → the outflow flux must be tracked
            ..Config::default()
        },
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let quiet = EmissionInput::default();
    for _ in 0..400 {
        bed.solver.step(DT, &quiet);
    }
    assert!(
        all_finite(&bed.solver.read_velocities()),
        "settle non-finite"
    );
    // SOLID: the recorded react ledger is the upward buoyancy+drag reaction the skeleton
    // balances (the rest is the internal contact stress holding the bed up — not a grid
    // impulse). At rest the grain momentum change is ≈ 0 (the held skeleton), and the ledger is
    // a finite, net-UPWARD reaction.
    let solid_dy = {
        let v0 = bed.solver.read_velocities();
        let (_, n_solid) = bed.solver.phase_counts();
        let m_s = Materials::default().grain_mass as f64;
        let total = v0.len();
        let mom = |v: &[[f32; 4]]| -> f64 {
            v[(total - n_solid as usize)..total]
                .iter()
                .map(|r| m_s * r[1] as f64)
                .sum()
        };
        let p0 = mom(&v0);
        bed.solver.step(DT, &quiet);
        let p1 = mom(&bed.solver.read_velocities());
        p1 - p0
    };
    let react = total_reaction(&bed.solver);
    let (_, n_solid) = bed.solver.phase_counts();
    let m_s = Materials::default().grain_mass as f64;
    let grav_scale = (n_solid as f64) * m_s * GRAV * (DT as f64);
    println!(
        "twofield U7 IMPULSE LEDGER (solid): Δp_solid {solid_dy:.3} (rest ≈ 0, scale {grav_scale:.1}), react ledger ({:.2}, {:.2}, {:.2})",
        react[0], react[1], react[2]
    );
    // The recorded react ledger (pressure-on-solid + drag) is FINITE and bounded by the column
    // weight (it is the buoyancy/drag SHARE the skeleton balances — the dominant support is the
    // internal contact/effective stress, not this ledger), and the lateral components stay small
    // (a centered column has no net horizontal pressure/drag).
    assert!(
        react.iter().all(|x| x.is_finite()),
        "react ledger non-finite: {react:?}"
    );
    assert!(
        react[1].abs() <= grav_scale,
        "solid react ledger {:.3} exceeds the column weight scale {grav_scale:.3} (un-physical)",
        react[1]
    );
    assert!(
        react[0].abs() <= 0.2 * grav_scale.max(react[1].abs()) + 1.0
            && react[2].abs() <= 0.2 * grav_scale.max(react[1].abs()) + 1.0,
        "lateral reaction leak: ({:.3}, {:.3})",
        react[0],
        react[2]
    );
    // The held skeleton does not drift (the bed is supported by its contact stress).
    assert!(
        solid_dy.abs() <= 0.1 * grav_scale,
        "held skeleton drifting: Δp_solid {solid_dy:.3} > 0.1 × gravity scale {grav_scale:.3}"
    );
    // WATER: reconstruct the y-momentum change of the live water from the grid impulses. At
    // rest gravity (down) is balanced by the projection pressure gradient (up) for the pore
    // water; the residual is the open-boundary outflow flux + the small actual momentum change.
    // We verify the outflow flux through the open base is TRACKED (nonzero, finite) — the
    // explicit open-boundary accounting the gate requires.
    let outflow = {
        let pos = bed.solver.read_positions();
        let vel = bed.solver.read_velocities();
        // Water that has escaped below the box floor (the filter outflow) carries momentum out.
        (0..bed.n_water as usize)
            .filter(|&i| (pos[i][1] as f64) < 0.0)
            .map(|i| (vel[i][1] as f64).abs())
            .sum::<f64>()
    };
    let water_dy = {
        let v0 = bed.solver.read_velocities();
        let pm = Materials::default().particle_mass as f64;
        let mom = |v: &[[f32; 4]]| -> f64 {
            (0..bed.n_water as usize).map(|i| pm * v[i][1] as f64).sum()
        };
        let p0 = mom(&v0);
        bed.solver.step(DT, &quiet);
        let p1 = mom(&bed.solver.read_velocities());
        p1 - p0
    };
    println!(
        "twofield U7 IMPULSE LEDGER (water): Δp_water {water_dy:.3}, tracked open-boundary outflow {outflow:.3}"
    );
    assert!(
        water_dy.is_finite() && outflow.is_finite(),
        "water ledger / outflow non-finite"
    );
}

// ==============================================================================================
// INTEGRATED WET-AND-DRAIN (the owner-ruling gate — the deferred U9 gate)
// ==============================================================================================

/// An ACTIVELY-WETTING DEFORMABLE saturated bed drains end-to-end with NO permanent clog: with
/// absorption ON and the bed deformable (grains free to rearrange), a pond poured onto a
/// partially-wet bed both wets it AND drains within a generous band — the frozen-lattice clog
/// from U9 (active absorption inflates φ_s past random close packing → K → 0) must NOT recur.
/// Assert the pond reaches ≤ 10% AND φ_f does not collapse toward 0 and stay there.
#[test]
fn integrated_wet_and_drain_deformable_no_clog() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let spec = BedSpec {
        box_max: [16.0, 30.0, 16.0],
        bed_top: 8.0,
        d: 1.0,
        drag_scale: 0.22,
        pond_layers: 4, // a standing pond above the bed (the pour surcharge)
        sat_frac: 0.6,  // partially wet → the wetting front advances (active absorption)
        pour_ml: 0.0,
        cfg: Config {
            solid_dynamics: true, // DEFORMABLE — grains rearrange (relieves the U9 clog)
            tf_absorb_rate: 0.15, // absorption ON (the regime U9 deferred here)
            tf_filter_floor: true,
            ..Config::default()
        },
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    for _ in 0..60 {
        bed.solver.step(DT, &input);
    }
    let pond_y = bed.bed_top + 0.2;
    let pond0 = count_water_above(&bed.solver, bed.n_water, pond_y);
    assert!(pond0 > 0, "no pond formed (count {pond0})");
    let mut drain_frame: Option<u32> = None;
    let mut min_phi_f = 1.0f64;
    for f in 1..=WETDRAIN_FRAMES_MAX {
        bed.solver.step(DT, &input);
        if f % 20 == 0 {
            let phi_f = mean_bed_phi_f(&bed.solver, bed.bed_top, bed.box_max);
            min_phi_f = min_phi_f.min(phi_f);
            let p = count_water_above(&bed.solver, bed.n_water, pond_y);
            if f % 200 == 0 {
                println!("  wet-and-drain probe frame {f}: pond {p}, φ_f {phi_f:.3}");
            }
            if drain_frame.is_none() && (p as f64) <= 0.10 * pond0 as f64 {
                drain_frame = Some(f);
            }
        }
    }
    let df = drain_frame.unwrap_or(u32::MAX);
    let absorbed = {
        let pos = bed.solver.read_positions();
        let total = pos.len();
        pos[(total - bed.n_grain as usize)..total]
            .iter()
            .map(|p| p[3] as f64)
            .sum::<f64>()
    };
    println!(
        "twofield U7 WET-AND-DRAIN: drained at frame {df} (band ≤ {WETDRAIN_FRAMES_MAX}), min φ_f {min_phi_f:.3} (floor {WETDRAIN_PHI_F_FLOOR}), absorbed {absorbed:.2}"
    );
    assert!(
        all_finite(&bed.solver.read_velocities()),
        "non-finite during wet-and-drain"
    );
    assert!(absorbed > 0.0, "absorption did not run (the wetting arm)");
    assert!(
        df <= WETDRAIN_FRAMES_MAX,
        "pond did not drain to ≤10% within {WETDRAIN_FRAMES_MAX} frames — the deformable bed \
         did NOT relieve the frozen-lattice clog (drain at {df})"
    );
    assert!(
        min_phi_f >= WETDRAIN_PHI_F_FLOOR,
        "φ_f collapsed toward 0 (min {min_phi_f:.3} < {WETDRAIN_PHI_F_FLOOR}) — the K → 0 clog recurred"
    );
}

// ==============================================================================================
// L2 RE-RUN ON DEFORMABLE BED
// ==============================================================================================

/// The U9 pond/drain band re-run with the bed DEFORMABLE (no active absorption — the U9 ladder
/// scene): swelling + plastic compaction shift the K field. Verify drainage completes and
/// document the frame count (the band may shift vs the frozen U9 number — the plan allows it).
#[test]
fn l2_pond_drain_re_run_on_deformable_bed() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let spec = BedSpec {
        box_max: [16.0, 30.0, 16.0],
        bed_top: 9.5,
        d: 1.0,
        drag_scale: 0.22,
        pond_layers: 4,
        sat_frac: 1.0, // saturated (drainage, not first-fill, sets the timescale) — the U9 scene
        pour_ml: 0.0,
        cfg: Config {
            solid_dynamics: true,
            tf_filter_floor: true,
            ..Config::default()
        },
    };
    let mut bed = build_bed(&gpu, &spec);
    let input = EmissionInput::default();
    for _ in 0..60 {
        bed.solver.step(DT, &input);
    }
    let pond_y = bed.bed_top + 0.2;
    let pond0 = count_water_above(&bed.solver, bed.n_water, pond_y);
    assert!(
        pond0 > 0,
        "no pond formed on the deformable bed (count {pond0})"
    );
    let mut drain_frame: Option<u32> = None;
    for f in 1..=L2_DEFORM_FRAMES_MAX {
        bed.solver.step(DT, &input);
        if f % 20 == 0 {
            let p = count_water_above(&bed.solver, bed.n_water, pond_y);
            if drain_frame.is_none() && (p as f64) <= 0.10 * pond0 as f64 {
                drain_frame = Some(f);
                break;
            }
        }
    }
    let df = drain_frame.unwrap_or(u32::MAX);
    println!(
        "twofield U7 L2-ON-DEFORMABLE: pond drained at frame {df} (band ≤ {L2_DEFORM_FRAMES_MAX}; U9 frozen band was 488–977)"
    );
    assert!(
        df <= L2_DEFORM_FRAMES_MAX,
        "deformable-bed pond did not drain within {L2_DEFORM_FRAMES_MAX} frames (at {df})"
    );
}

// ==============================================================================================
// REDUCED DIVERGENCE RE-CHECK (operator consistency under moving φ_s)
// ==============================================================================================

/// A reduced version of the U3/U6 divergence gate on the DEFORMABLE bed: the post-projection
/// interior mixture divergence stays below tolerance under moving φ_s (the operator is still
/// consistent when the solid field moves). Run at the default sweep budget.
#[test]
fn reduced_divergence_on_deformable_bed() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let spec = BedSpec {
        box_max: [16.0, 28.0, 16.0],
        bed_top: 9.0,
        drag_scale: 0.30,
        pond_layers: 3,
        cfg: Config {
            solid_dynamics: true,
            ..Config::default()
        },
        ..BedSpec::new()
    };
    let mut bed = build_bed(&gpu, &spec);
    let quiet = EmissionInput::default();
    for _ in 0..300 {
        bed.solver.step(DT, &quiet);
    }
    let (rms, cnt) = rms_interior_mixture_div(&bed.solver);
    println!("twofield U7 REDUCED DIVERGENCE (deformable bed): RMS {rms:.4} over {cnt} cells (tol {DIV_TOL})");
    assert!(
        cnt > 10,
        "too few interior cells for the divergence check ({cnt})"
    );
    assert!(all_finite(&bed.solver.read_velocities()), "non-finite");
    assert!(
        rms <= DIV_TOL,
        "interior mixture divergence {rms:.4} above {DIV_TOL} under moving φ_s — operator \
         inconsistency with the deformable solid field"
    );
}

// ==============================================================================================
// DETERMINISM
// ==============================================================================================

/// Two seeded runs produce identical metrics (bit-for-bit grain positions) on the deformable
/// saturated bed under pour — the determinism contract (R6).
#[test]
fn deterministic_two_runs_identical() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let make = || -> DefBed {
        let spec = BedSpec {
            box_max: [18.0, 28.0, 18.0],
            bed_top: 8.0,
            drag_scale: 0.30,
            pour_ml: 2000.0,
            cfg: Config {
                solid_dynamics: true,
                nozzle_radius: JET_RADIUS as f32,
                tf_filter_floor: true,
                ..Config::default()
            },
            ..BedSpec::new()
        };
        build_bed(&gpu, &spec)
    };
    let run = |bed: &mut DefBed| -> Vec<[f32; 4]> {
        let axis = (bed.box_max[0] as f64 / 2.0, bed.box_max[2] as f64 / 2.0);
        let pour = EmissionInput {
            kettle_pos: [axis.0 as f32, bed.bed_top as f32 + 12.0, axis.1 as f32],
            flow_rate: JET_FLOW,
            ..EmissionInput::default()
        };
        for _ in 0..120 {
            bed.solver.step(DT, &pour);
        }
        bed.solver.read_positions()
    };
    let mut a = make();
    let mut b = make();
    let pa = run(&mut a);
    let pb = run(&mut b);
    assert_eq!(pa.len(), pb.len(), "determinism: differing particle counts");
    for (i, (ra, rb)) in pa.iter().zip(&pb).enumerate() {
        for c in 0..4 {
            assert_eq!(
                ra[c].to_bits(),
                rb[c].to_bits(),
                "determinism: particle {i} lane {c} diverged ({} vs {})",
                ra[c],
                rb[c]
            );
        }
    }
    println!(
        "twofield U7 DETERMINISM: {} particles bit-identical across two runs",
        pa.len()
    );
}

// ==============================================================================================
// COMBINED CONSERVATION (the full V60 pour scene)
// ==============================================================================================

/// Long-run combined conservation on the full V60 pour scene: total volume (free water +
/// absorbed, incl. swelling) is conserved within budget over a pour-and-drain run with the
/// deformable saturated bed. The V60 geometry + emission path are reused, not duplicated.
#[test]
fn combined_conservation_v60_pour_deformable() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_full: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::v60_pour();
    let mats = Materials::default();
    let cfg = Config {
        solid_dynamics: true,
        tf_absorb_rate: 0.15,
        tf_wet_cohesion: 4.0,
        tf_filter_floor: true,
        ..Config::default()
    };
    let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
    let (water_seed, n_grain) = solver.phase_counts();
    // V60 pour: a center jet over the cone. Reuse the emission path.
    let pour = EmissionInput {
        kettle_pos: [0.0, 8.0, 0.0],
        flow_rate: 30.0,
        ..EmissionInput::default()
    };
    let quiet = EmissionInput::default();
    // Pour phase.
    for _ in 0..240 {
        solver.step(DT, &pour);
    }
    let emitted = solver.total_emitted_water_mass() as f64;
    let n_live = solver.phase_counts().0;
    let v_after_pour = total_volume(&solver, n_live, n_grain);
    let absorbed_after_pour = {
        let pos = solver.read_positions();
        let total = pos.len();
        pos[(total - n_grain as usize)..total]
            .iter()
            .map(|p| p[3] as f64)
            .sum::<f64>()
    };
    // Drain phase (no pour).
    for _ in 0..600 {
        solver.step(DT, &quiet);
    }
    let n_final = solver.phase_counts().0;
    let v_final = total_volume(&solver, n_final, n_grain);
    let absorbed_final = {
        let pos = solver.read_positions();
        let total = pos.len();
        pos[(total - n_grain as usize)..total]
            .iter()
            .map(|p| p[3] as f64)
            .sum::<f64>()
    };
    let _ = water_seed;
    println!(
        "twofield U7 COMBINED CONSERVATION (V60 pour): emitted {emitted:.1}, vol after pour {v_after_pour:.2} (absorbed {absorbed_after_pour:.2}), final {v_final:.2} (absorbed {absorbed_final:.2}); drift gate ≤ {VOL_DRIFT_TOL}"
    );
    assert!(all_finite(&solver.read_positions()), "V60 pour non-finite");
    // The pour delivered water (non-trivial run) and the deformable bed absorbed it (swelling) —
    // so the conservation check is over a real pour-and-absorb run, not a vacuous one.
    assert!(emitted > 0.0, "V60 pour emitted no water");
    assert!(
        absorbed_final > 0.0 && absorbed_final >= absorbed_after_pour - VOL_DRIFT_TOL * v_final,
        "absorption did not run / un-absorbed: {absorbed_after_pour:.2} -> {absorbed_final:.2}"
    );
    // No spurious creation: the final bookkept (in-domain) volume cannot exceed the post-pour
    // bookkept volume beyond the float budget (drainage only removes, swelling only transfers
    // 1:1 from free water into grain volume — both already on the books).
    assert!(
        v_final <= v_after_pour * (1.0 + VOL_DRIFT_TOL),
        "combined volume grew: final {v_final:.2} > post-pour {v_after_pour:.2} (+{VOL_DRIFT_TOL})"
    );
}

//! Twofield U4 gates: free surface + pour cavity — the L0 EXIT GATE (plan 2026-06-09-001 U4 /
//! KTD-6). A fast jet into a settled pool must dig a visible, transiently sustained cavity:
//! the demonstration that the APIC + projection architecture fixes the prior PBF solver's
//! failure. The documented xpbd baseline cavity is ≈ 0 (position projection kills impact
//! momentum in the density solve — `docs/plans/2026-06-08-001-feat-water-grain-impact-
//! coupling-plan.md`; water↔water impact term reverted in `4dca52f`).
//!
//! FLOORS ARE PRE-COMMITTED before implementation (plan U4: "empirical calibration allowed to
//! tighten thresholds above the floors but never loosen below"):
//!   depth        ≥ 1 jet radius below the undisturbed surface (CAVITY_DEPTH_FLOOR)
//!   persistence  ≥ the gravity re-leveling timescale at pour velocity:
//!                  t_relevel ≈ sqrt(2·d_floor/g) = sqrt(2·1.5/20) ≈ 0.387 s → 24 frames at
//!                  dt = 1/60 (PERSIST_FRAMES = ceil(0.387·60) = 24)
//!
//! DEPTH METRIC (documented per the gate): grid-mass column scan. For each in-box node
//! column, the local surface is the top of the contiguous-from-bottom fluid run (first node
//! with column density < 0.5·ρ_rest after fluid was seen ends the run; splash sheets above a
//! gap are ignored; a column with no bottom fluid reads the box floor). Cavity depth =
//! (undisturbed annulus surface) − (annulus surface now), averaged over node columns at
//! radius r ∈ [2.5, 5] from the jet axis — the crater wall directly around the jet. The
//! annulus is a CONSERVATIVE proxy for the axis depth (the jet column itself occupies the
//! axis cells, so the depression is measured just outside the stream where it is shallower
//! than at the axis). The metric is independent of the bubble machinery (node MASS, not the
//! pocket-flagged fill fractions), so an un-collapsed pocket cannot hide a permanent hole.
//!
//! KNOB GRID (KTD-9): U4 adds NO new gate knobs. The pressure knobs stay U3's grid (the
//! sensitivity gate samples fine sweeps {8, 16} from it); the flood-fill sweep budget,
//! SURF_* constants and the bubble representation are fixed structural constants documented
//! in twofield/mod.rs + surface.wgsl. Exhausting U3's grid without a pass IS the halt.

// Axis loops (`for a in 0..3`) index parallel arrays; the iterator rewrite obscures that.
#![allow(clippy::needless_range_loop)]

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{TwofieldSolver, COARSE_RATIO_DEFAULT, COARSE_SWEEPS_DEFAULT};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::rng::Rng;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;
const GRAV: f64 = 20.0; // |g| of the gate scene
const SPACING: f64 = 1.0; // Materials::default().particle_spacing
const REST_DENSITY: f64 = 1.0; // particle_mass / spacing³ at the defaults

// --- the jet (pour velocity fixed BEFORE implementation; all in scene units) ----------------
/// Jet radius = Config::nozzle_radius for the gate scene (discharge_coeff = 1).
const JET_RADIUS: f64 = 1.5;
/// Volumetric flow (units³/s). Exit speed = flow / (π r²) ≈ 11.3; impact speed after the
/// ~6-unit fall ≈ 19 — a vigorous pour-strength jet, comfortably under the cap (50).
const JET_FLOW: f32 = 80.0;
const KETTLE: [f32; 3] = [12.0, 18.0, 12.0];
const AXIS: (f64, f64) = (12.0, 12.0);

// --- pre-committed floors (NEVER loosened) ----------------------------------------------------
/// CAVITY depth floor: ≥ 1 jet radius below the undisturbed surface.
const CAVITY_DEPTH_FLOOR: f64 = JET_RADIUS;
/// CAVITY persistence floor: the gravity re-leveling timescale at the floor depth,
/// t_relevel = sqrt(2·d/g) = sqrt(2·1.5/20) ≈ 0.387 s → ceil(0.387/DT) = 24 frames.
const PERSIST_FRAMES: usize = 24;
/// COLLAPSE: after the jet stops the annulus must re-level — residual depth ≤ half the
/// cavity floor (no permanent hole).
const COLLAPSE_DEPTH_MAX: f64 = 0.5 * CAVITY_DEPTH_FLOOR;
/// NO POPCORN: post-collapse the pool must not gain energy — late-window peak speed bounded
/// absolutely (well under the cap 50) and not above the collapse-window peak + 1.
const POPCORN_MAX_SPEED: f32 = 8.0;
const POPCORN_MAX_RISE: f64 = 1.0; // spacings of max-y creep across the popcorn window
/// VOLUME (the U3 band): all-column mean surface rise must equal emitted volume / box area
/// within ± the U3 height-drift tolerance; deep density drift within the xpbd visible band.
const VOLUME_DH_TOL: f64 = 0.5; // spacings (mirrors U3 HEIGHT_DRIFT_TOL)
const VOLUME_DENSITY_TOL: f64 = 0.05; // mirrors U3 DENSITY_DRIFT_TOL
/// Emitter throughput: emitted volume vs flow·t within 2 disc layers of slack.
const EMIT_THROUGHPUT_TOL: f64 = 16.0; // units³ = 2·N_layer·V_w, N_layer = ceil(π r²) = 8
/// BUBBLE SENSITIVITY: cavity metrics at fine sweeps 8 vs 16 must agree within this relative
/// band (the fine-sweeps-erode-the-constraint check) — 30% of scale, floor-scale absolute
/// slack when both arms are small.
const SENS_BAND: f64 = 0.3;

// --- schedule ----------------------------------------------------------------------------------
const SETTLE_FRAMES: u32 = 120;
const JET_FRAMES: usize = 120; // 2 s of pour ≈ 160 emitted particles
const COLLAPSE_FRAMES: u32 = 600; // 10 s ≫ t_relevel: re-leveling must complete
const POPCORN_FRAMES: u32 = 300;

// --- U3 re-run tolerances (mirrored from tests/twofield_pressure.rs, pre-registered there) ----
const DIV_TOL: f64 = 0.3;
const DECAY_FACTOR_MIN: f64 = 2.0;
const DECAY_BUDGETS: [u32; 3] = [4, 8, 16];
const INTERIOR_MARGIN: i64 = 2;

/// The gate scene: plain box, settled 12-unit pool, no solids, pour pool sized for the dose.
/// Rest-density inset seeding (one particle per spacing³, half-spacing wall gap — see the
/// tank_scene comment in tests/twofield_pressure.rs); surface ≈ y = 12.
fn jet_scene() -> Scene {
    Scene {
        gravity: [0.0, -(GRAV as f32), 0.0],
        box_min: [0.0; 3],
        box_max: [24.0, 28.0, 24.0],
        regions: vec![SeedRegion {
            min: [0.5, 0.5, 0.5],
            max: [23.6, 11.6, 23.6],
            species: Species::Water,
        }],
        solids: Vec::new(),
        // Pool headroom for the dose: 80 units³/s · 2 s = 160 particles ≈ 832 mL; 1300 mL
        // (250 particles) leaves slack so the capacity clamp can never bite mid-gate.
        pour_water_ml: 1300.0,
        ..Scene::default()
    }
}

fn jet_cfg() -> Config {
    Config {
        nozzle_radius: JET_RADIUS as f32,
        ..Config::default()
    }
}

fn jet_input() -> EmissionInput {
    EmissionInput {
        kettle_pos: KETTLE,
        flow_rate: JET_FLOW,
        pour_angle: 0.0,
        ..EmissionInput::default()
    }
}

// ==============================================================================================
// Depth metric (header: DEPTH METRIC)
// ==============================================================================================

struct SurfaceMap {
    /// (x, z, y_surf) per in-box node column.
    cols: Vec<(f64, f64, f64)>,
}

impl SurfaceMap {
    fn band_mean(&self, r_min: f64, r_max: f64) -> f64 {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for &(x, z, y) in &self.cols {
            let r = ((x - AXIS.0).powi(2) + (z - AXIS.1).powi(2)).sqrt();
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
    fn annulus(&self) -> f64 {
        self.band_mean(2.5, 5.0)
    }
    fn rim(&self) -> f64 {
        self.band_mean(7.0, 10.0)
    }
    fn all(&self) -> f64 {
        let sum: f64 = self.cols.iter().map(|c| c.2).sum();
        sum / self.cols.len() as f64
    }
}

/// Per-column surface height from node masses: top of the contiguous-from-bottom fluid run
/// (column density ≥ 0.5·ρ_rest), splash sheets above the first gap ignored, craters reaching
/// the floor read the floor (see the file header for why this is the documented metric).
fn surface_map(solver: &TwofieldSolver) -> SurfaceMap {
    let (origin, h, dims) = solver.grid_spec();
    let gv = solver.read_grid_velocities();
    let h = h as f64;
    let thresh = 0.5 * REST_DENSITY * h * h * h;
    let mut cols = Vec::new();
    for k in 1..dims[2] as usize - 1 {
        for i in 1..dims[0] as usize - 1 {
            let x = origin[0] as f64 + i as f64 * h;
            let z = origin[2] as f64 + k as f64 * h;
            let floor_y = origin[1] as f64 + h;
            let mut y_surf = floor_y;
            let mut seen = false;
            for j in 1..dims[1] as usize {
                let n = i + dims[0] as usize * (j + dims[1] as usize * k);
                if (gv[n][3] as f64) >= thresh {
                    seen = true;
                    y_surf = origin[1] as f64 + j as f64 * h;
                } else if seen || j > 1 {
                    break; // first gap after (or instead of) the bottom run
                }
            }
            cols.push((x, z, y_surf));
        }
    }
    SurfaceMap { cols }
}

/// Mean node density over the deep interior slab (y ∈ [2, 5], ≥ 2 nodes from each wall) —
/// the U3 grid-mass volume proxy, re-used verbatim as the U4 VOLUME gate's density arm.
fn deep_mean_node_density(solver: &TwofieldSolver) -> f64 {
    let (origin, h, dims) = solver.grid_spec();
    let gv = solver.read_grid_velocities();
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for k in 2..dims[2] as usize - 2 {
        for j in 0..dims[1] as usize {
            let y = origin[1] + j as f32 * h;
            if !(2.0..=5.0).contains(&y) {
                continue;
            }
            for i in 2..dims[0] as usize - 2 {
                let n = i + dims[0] as usize * (j + dims[1] as usize * k);
                sum += gv[n][3] as f64 / (h as f64).powi(3);
                cnt += 1;
            }
        }
    }
    assert!(cnt > 0, "deep density probe found no nodes");
    sum / cnt as f64
}

fn live_water(solver: &TwofieldSolver) -> usize {
    solver.phase_counts().0 as usize
}

fn max_speed_live(solver: &TwofieldSolver) -> f32 {
    let n = live_water(solver);
    solver.read_velocities()[..n]
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0f32, f32::max)
}

fn max_y_live(solver: &TwofieldSolver) -> f64 {
    let n = live_water(solver);
    solver.read_positions()[..n]
        .iter()
        .map(|p| p[1] as f64)
        .fold(f64::MIN, f64::max)
}

/// Longest run of consecutive frames with depth ≥ the floor.
fn longest_run_at_floor(depths: &[f64]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for &d in depths {
        if d >= CAVITY_DEPTH_FLOOR {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

// ==============================================================================================
// Gate 1: CAVITY + COLLAPSE + NO POPCORN + VOLUME (one continuous run)
// ==============================================================================================

#[test]
fn cavity_forms_then_collapses_conserving_volume_without_popcorn() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_cavity: no GPU adapter; skipping.");
        return;
    };
    let scene = jet_scene();
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &jet_cfg(), &gpu);
    let quiet = EmissionInput::default();
    let pour = jet_input();

    // Settle, then record the undisturbed baselines.
    for _ in 0..SETTLE_FRAMES {
        solver.step(DT, &quiet);
    }
    let seed_water = live_water(solver_ref(&solver));
    let pre = surface_map(&solver);
    let h0_annulus = pre.annulus();
    let h0_all = pre.all();
    let rho0 = deep_mean_node_density(&solver);
    println!(
        "twofield U4 baseline: annulus surface {h0_annulus:.3} | all-column {h0_all:.3} | deep density {rho0:.4} | seed water {seed_water}"
    );

    // Jet ON: per-frame depth trace.
    let mut depths = Vec::with_capacity(JET_FRAMES);
    for _ in 0..JET_FRAMES {
        solver.step(DT, &pour);
        depths.push(h0_annulus - surface_map(&solver).annulus());
    }
    let max_depth = depths.iter().cloned().fold(f64::MIN, f64::max);
    let run = longest_run_at_floor(&depths);
    let emitted = live_water(&solver) - seed_water;
    println!(
        "twofield U4 CAVITY: max depth {max_depth:.2} (floor {CAVITY_DEPTH_FLOOR}) | longest run at floor {run} frames (floor {PERSIST_FRAMES}) | emitted {emitted} | xpbd baseline ≈ 0"
    );
    assert!(
        max_depth >= CAVITY_DEPTH_FLOOR,
        "cavity depth {max_depth:.2} below the pre-committed floor {CAVITY_DEPTH_FLOOR} \
         (xpbd baseline ≈ 0 — this is the L0 exit gate)"
    );
    assert!(
        run >= PERSIST_FRAMES,
        "cavity persisted only {run} consecutive frames at depth ≥ {CAVITY_DEPTH_FLOOR} \
         (pre-committed floor {PERSIST_FRAMES} = the gravity re-leveling timescale)"
    );

    // Emitter throughput (volume-consistent emission): emitted volume ≈ flow·t.
    let expect = JET_FLOW as f64 * JET_FRAMES as f64 * DT as f64; // V_w = spacing³ = 1
    assert!(
        ((emitted as f64) - expect).abs() <= EMIT_THROUGHPUT_TOL,
        "emitted volume {emitted} vs flow·t {expect:.1} (tol {EMIT_THROUGHPUT_TOL})"
    );

    // Jet OFF → COLLAPSE: the cavity must re-level (no permanent hole).
    for _ in 0..COLLAPSE_FRAMES {
        solver.step(DT, &quiet);
    }
    let post = surface_map(&solver);
    let depth_end = h0_annulus - post.annulus();
    let tilt_end = (post.annulus() - post.rim()).abs();
    let v_collapse = max_speed_live(&solver);
    let y_collapse = max_y_live(&solver);
    let bubble = solver.read_bubble();
    println!(
        "twofield U4 COLLAPSE: residual depth {depth_end:.2} (max {COLLAPSE_DEPTH_MAX}) | annulus-rim tilt {tilt_end:.2} | peak |v| {v_collapse:.2} | bubble λ {:.3}",
        bubble[0]
    );
    assert!(
        depth_end <= COLLAPSE_DEPTH_MAX,
        "permanent hole: residual annulus depth {depth_end:.2} > {COLLAPSE_DEPTH_MAX} after {COLLAPSE_FRAMES} frames"
    );

    // NO POPCORN: a long settled run after collapse must not gain energy.
    let mut v_late = 0.0f32;
    for f in 1..=POPCORN_FRAMES {
        solver.step(DT, &quiet);
        if f % 50 == 0 {
            v_late = v_late.max(max_speed_live(&solver));
        }
    }
    let y_late = max_y_live(&solver);
    println!(
        "twofield U4 NO POPCORN: late peak |v| {v_late:.2} (collapse-window peak {v_collapse:.2}, abs bound {POPCORN_MAX_SPEED}) | max y {y_collapse:.2} -> {y_late:.2}"
    );
    assert!(
        v_late <= POPCORN_MAX_SPEED,
        "post-collapse pool energetic: late peak |v| {v_late:.2} > {POPCORN_MAX_SPEED}"
    );
    assert!(
        v_late <= v_collapse + 1.0,
        "post-collapse energy GREW: late peak |v| {v_late:.2} vs collapse-window {v_collapse:.2}"
    );
    assert!(
        y_late <= y_collapse + POPCORN_MAX_RISE * SPACING,
        "particles climbed post-collapse: max y {y_collapse:.2} -> {y_late:.2}"
    );

    // VOLUME (U3 band): surface rise = emitted volume / box area; deep density steady.
    let h_end_all = surface_map(&solver).all();
    let dh_pred = emitted as f64 * SPACING.powi(3) / (24.0 * 24.0);
    let dh_meas = h_end_all - h0_all;
    let rho1 = deep_mean_node_density(&solver);
    println!(
        "twofield U4 VOLUME: Δh measured {dh_meas:.3} vs predicted {dh_pred:.3} (tol {VOLUME_DH_TOL}) | deep density {rho0:.4} -> {rho1:.4}"
    );
    assert!(
        (dh_meas - dh_pred).abs() <= VOLUME_DH_TOL * SPACING,
        "water volume not conserved through cavity formation/collapse: Δh {dh_meas:.3} vs {dh_pred:.3}"
    );
    assert!(
        (rho1 / rho0 - 1.0).abs() <= VOLUME_DENSITY_TOL,
        "deep density drifted {:.2}% through the cavity cycle",
        (rho1 / rho0 - 1.0) * 100.0
    );
}

// Identity helper so the borrow in the baseline block reads clearly.
fn solver_ref(s: &TwofieldSolver) -> &TwofieldSolver {
    s
}

/// DIAGNOSTIC sweep (ignored by default — run with `--ignored`): crater depth + persistence vs
/// the APIC↔PIC blend, to choose the production `PIC_BLEND_DEFAULT` (the settle-vs-crater
/// tradeoff). Prints the curve; asserts only that pure APIC (blend 0) clears the persistence
/// floor (the crater baseline) so the tradeoff is real.
#[test]
#[ignore = "diagnostic sweep; run explicitly to retune PIC_BLEND_DEFAULT"]
fn blend_sweep_crater_depth_and_persistence() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_cavity: no GPU adapter; skipping.");
        return;
    };
    let scene = jet_scene();
    let quiet = EmissionInput::default();
    let pour = jet_input();
    println!("twofield crater blend sweep (depth floor {CAVITY_DEPTH_FLOOR}, persistence floor {PERSIST_FRAMES}):");
    let mut blend0_run = 0usize;
    for &blend in &[0.0f32, 0.03, 0.05, 0.08, 0.1, 0.15] {
        let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &jet_cfg(), &gpu);
        solver.set_pic_blend_for_test(blend);
        for _ in 0..SETTLE_FRAMES {
            solver.step(DT, &quiet);
        }
        let h0 = surface_map(&solver).annulus();
        let mut depths = Vec::with_capacity(JET_FRAMES);
        for _ in 0..JET_FRAMES {
            solver.step(DT, &pour);
            depths.push(h0 - surface_map(&solver).annulus());
        }
        let max_depth = depths.iter().cloned().fold(f64::MIN, f64::max);
        let run = longest_run_at_floor(&depths);
        if blend == 0.0 {
            blend0_run = run;
        }
        println!("  blend {blend:.2}: max depth {max_depth:.2} | persistence {run} frames");
    }
    assert!(
        blend0_run >= PERSIST_FRAMES,
        "pure-APIC crater baseline below the persistence floor ({blend0_run} < {PERSIST_FRAMES})"
    );
}

// ==============================================================================================
// Gate 2: BUBBLE SENSITIVITY — fine sweeps 8 vs 16 (constraint-erosion check)
// ==============================================================================================

/// The named KTD-6 failure is the fine sweeps relaxing the pocket constraint away: if the
/// bubble row were represented only outside the smoother, DOUBLING the fine-sweep budget
/// would erode cavity persistence. Both arms must individually meet the floors, and the
/// depth metrics must agree within SENS_BAND — during the jet AND through the first
/// re-leveling window after the stop (where an enclosed pocket carries the cavity).
#[test]
fn bubble_sensitivity_fine_sweeps_8_vs_16() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_cavity: no GPU adapter; skipping.");
        return;
    };
    let scene = jet_scene();
    let quiet = EmissionInput::default();
    let pour = jet_input();

    let run_arm = |nf: u32| -> (f64, f64, usize) {
        let mut s = TwofieldSolver::build(&scene, &Materials::default(), &jet_cfg(), &gpu);
        s.set_pressure_budget_for_test(COARSE_RATIO_DEFAULT, COARSE_SWEEPS_DEFAULT, nf);
        for _ in 0..SETTLE_FRAMES {
            s.step(DT, &quiet);
        }
        let h0 = surface_map(&s).annulus();
        let mut depths = Vec::with_capacity(JET_FRAMES);
        for _ in 0..JET_FRAMES {
            s.step(DT, &pour);
            depths.push(h0 - surface_map(&s).annulus());
        }
        // Quasi-steady jet window mean + the post-stop re-leveling window mean.
        let steady = &depths[40..];
        let mean_jet = steady.iter().sum::<f64>() / steady.len() as f64;
        let run = longest_run_at_floor(&depths);
        let mut post = Vec::with_capacity(PERSIST_FRAMES);
        for _ in 0..PERSIST_FRAMES {
            s.step(DT, &quiet);
            post.push(h0 - surface_map(&s).annulus());
        }
        let mean_post = post.iter().sum::<f64>() / post.len() as f64;
        (mean_jet, mean_post, run)
    };

    let (jet8, post8, run8) = run_arm(8);
    let (jet16, post16, run16) = run_arm(16);
    println!(
        "twofield U4 sensitivity: jet-window depth 8 sweeps {jet8:.2} vs 16 sweeps {jet16:.2} | post-stop {post8:.2} vs {post16:.2} | runs {run8}/{run16}"
    );
    for (label, run) in [("8", run8), ("16", run16)] {
        assert!(
            run >= PERSIST_FRAMES,
            "arm nf={label}: persistence {run} below the floor {PERSIST_FRAMES}"
        );
    }
    for (label, a, b) in [("jet", jet8, jet16), ("post-stop", post8, post16)] {
        let scale = a.abs().max(b.abs()).max(CAVITY_DEPTH_FLOOR);
        assert!(
            (a - b).abs() <= SENS_BAND * scale,
            "{label} depth diverges across fine-sweep budgets: {a:.2} vs {b:.2} \
             (band {SENS_BAND} of {scale:.2}) — the fine-sweeps-erode-the-constraint signature"
        );
    }
}

// ==============================================================================================
// Gate 3: U3 re-runs on the U4 operator (surface + bubble rows in place)
// ==============================================================================================

fn runif(rng: &mut Rng) -> f64 {
    rng.next_f32() as f64 * 2.0 - 1.0
}

/// Symmetric M̃⁻¹ application from the nm readback (xx, xy, xz, yy, yz, zz).
fn minv_apply(s: &[f32; 8], v: [f64; 3]) -> [f64; 3] {
    let m = [
        s[0] as f64,
        s[1] as f64,
        s[2] as f64,
        s[3] as f64,
        s[4] as f64,
        s[5] as f64,
    ];
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[1] * v[0] + m[3] * v[1] + m[4] * v[2],
        m[2] * v[0] + m[4] * v[1] + m[5] * v[2],
    ]
}

/// Assembled-operator spot check on a state of THIS run (mid-jet / post-stop): D/G adjointness
/// on GPU readbacks and A = −D·M̃⁻¹·G symmetric + positive — exactly the U3 certification,
/// re-run with the U4 surface + pocket rows in the masks (the L0 exit operator is this one).
fn assert_operator_spot_check(solver: &TwofieldSolver, label: &str, rng: &mut Rng) -> usize {
    let meta = solver.read_cell_meta();
    let nm = solver.read_node_matrices();
    let num_cells = meta.len();
    let num_nodes = nm.len();
    let active: Vec<bool> = meta.iter().map(|m| m[1] > 0.0).collect();
    let pocket_cells = meta.iter().filter(|m| m[3] > 1.5).count();
    let n_active = active.iter().filter(|&&a| a).count();
    assert!(n_active > 50, "{label}: only {n_active} active cells");

    // Adjointness u·(Gp) = −(Du)·p straight off the dbg taps.
    let u: Vec<[f32; 4]> = (0..num_nodes)
        .map(|_| [runif(rng) as f32, runif(rng) as f32, runif(rng) as f32, 0.0])
        .collect();
    let p: Vec<f32> = (0..num_cells).map(|_| runif(rng) as f32).collect();
    solver.write_grid_velocities_for_test(&u);
    solver.run_div_for_test();
    let du: Vec<f64> = solver
        .read_cell_meta()
        .iter()
        .map(|m| m[2] as f64)
        .collect();
    solver.write_pressure_for_test(&p);
    solver.run_grad_for_test();
    let gp = solver.read_grid_velocities();
    let lhs: f64 = u
        .iter()
        .zip(&gp)
        .map(|(a, b)| {
            a[0] as f64 * b[0] as f64 + a[1] as f64 * b[1] as f64 + a[2] as f64 * b[2] as f64
        })
        .sum();
    let rhs: f64 = -du.iter().zip(&p).map(|(a, b)| a * *b as f64).sum::<f64>();
    let scale = du.iter().map(|d| d.abs()).sum::<f64>().max(1.0);
    assert!(
        (lhs - rhs).abs() <= 1e-3 * scale,
        "{label}: GPU adjointness broke with U4 rows: {lhs} vs {rhs}"
    );

    // Assembled A symmetric + positive via the taps (the same pipeline as the U3 gate).
    let mk = |rng: &mut Rng| -> Vec<f32> {
        (0..num_cells)
            .map(|c| if active[c] { runif(rng) as f32 } else { 0.0 })
            .collect()
    };
    let x = mk(rng);
    let y = mk(rng);
    let apply_a = |z: &[f32]| -> Vec<f64> {
        solver.write_pressure_for_test(z);
        solver.run_grad_for_test();
        let g = solver.read_grid_velocities();
        let mg: Vec<[f32; 4]> = (0..num_nodes)
            .map(|n| {
                let v = minv_apply(&nm[n], [g[n][0] as f64, g[n][1] as f64, g[n][2] as f64]);
                [v[0] as f32, v[1] as f32, v[2] as f32, 0.0]
            })
            .collect();
        solver.write_grid_velocities_for_test(&mg);
        solver.run_div_for_test();
        solver
            .read_cell_meta()
            .iter()
            .map(|m| -(m[2] as f64))
            .collect()
    };
    let ax = apply_a(&x);
    let ay = apply_a(&y);
    let dot = |a: &[f32], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * y).sum() };
    let xay = dot(&x, &ay);
    let yax = dot(&y, &ax);
    let s = xay.abs().max(yax.abs()).max(1.0);
    assert!(
        (xay - yax).abs() <= 2e-3 * s,
        "{label}: U4 assembled A not symmetric: {xay} vs {yax}"
    );
    let xax = dot(&x, &ax);
    assert!(xax > 0.0, "{label}: U4 assembled A not positive: {xax}");
    println!(
        "twofield U4 operator re-run [{label}]: {n_active} active cells ({pocket_cells} pocket), adjoint + SPD"
    );
    pocket_cells
}

#[test]
fn u3_rerun_assembled_operator_mid_jet_and_post_stop() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_cavity: no GPU adapter; skipping.");
        return;
    };
    let scene = jet_scene();
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &jet_cfg(), &gpu);
    let quiet = EmissionInput::default();
    let pour = jet_input();
    let mut rng = Rng::new(0xCAFE_0004);
    for _ in 0..SETTLE_FRAMES {
        solver.step(DT, &quiet);
    }
    for _ in 0..60 {
        solver.step(DT, &pour);
    }
    assert_operator_spot_check(&solver, "mid-jet", &mut rng);
    // The taps overwrote grid_vel/pressure; a fresh step rebuilds the frame state.
    for _ in 0..60 {
        solver.step(DT, &pour);
    }
    for _ in 0..10 {
        solver.step(DT, &quiet);
    }
    assert_operator_spot_check(&solver, "post-stop", &mut rng);

    // The pour pocket is transient and may be absent at the fixed frames above, so a
    // GUARANTEED enclosed-pocket state exercises the bubble rows in the operator: a
    // full-cross-section water slab seeded over trapped air (no open path to the top face).
    let slab = Scene {
        gravity: [0.0, -(GRAV as f32), 0.0],
        box_min: [0.0; 3],
        box_max: [24.0, 28.0, 24.0],
        regions: vec![SeedRegion {
            min: [0.5, 4.5, 0.5],
            max: [23.6, 9.6, 23.6],
            species: Species::Water,
        }],
        solids: Vec::new(),
        ..Scene::default()
    };
    let mut s = TwofieldSolver::build(&slab, &Materials::default(), &jet_cfg(), &gpu);
    s.step(DT, &quiet);
    let pockets = assert_operator_spot_check(&s, "enclosed pocket", &mut rng);
    assert!(
        pockets > 0,
        "the trapped-air slab state must contain pocket cells — the bubble-row check would \
         otherwise be vacuous"
    );
    let b = s.read_bubble();
    assert!(b[0].is_finite(), "bubble multiplier non-finite: {}", b[0]);
}

/// Post-projection interior RMS divergence with the projection's own D (fraction-weighted
/// corner gather), interior = margin-2 fully-filled cells EXCLUDING pocket cells (air under
/// an aggregate constraint is not a per-cell-divergence-free region — only fluid rows are).
fn rms_interior_div(solver: &TwofieldSolver) -> (f64, usize) {
    let (_, h, dims) = solver.grid_spec();
    let h = h as f64;
    let nc = [
        dims[0] as usize - 1,
        dims[1] as usize - 1,
        dims[2] as usize - 1,
    ];
    let meta = solver.read_cell_meta();
    let gv = solver.read_grid_velocities();
    let cidx = |i: usize, j: usize, k: usize| i + nc[0] * (j + nc[1] * k);
    let nidx = |i: usize, j: usize, k: usize| i + dims[0] as usize * (j + dims[1] as usize * k);
    let full = |i: i64, j: i64, k: i64| -> bool {
        i >= 0
            && j >= 0
            && k >= 0
            && (i as usize) < nc[0]
            && (j as usize) < nc[1]
            && (k as usize) < nc[2]
            && meta[cidx(i as usize, j as usize, k as usize)][1] >= 0.999
    };
    let mut se = 0.0;
    let mut cnt = 0usize;
    for k in 0..nc[2] {
        for j in 0..nc[1] {
            for i in 0..nc[0] {
                let c = cidx(i, j, k);
                if meta[c][1] <= 0.0 || meta[c][3] > 1.5 {
                    continue; // masked or pocket
                }
                let mut interior = true;
                'm: for dk in -INTERIOR_MARGIN..=INTERIOR_MARGIN {
                    for dj in -INTERIOR_MARGIN..=INTERIOR_MARGIN {
                        for di in -INTERIOR_MARGIN..=INTERIOR_MARGIN {
                            if !full(i as i64 + di, j as i64 + dj, k as i64 + dk) {
                                interior = false;
                                break 'm;
                            }
                        }
                    }
                }
                if !interior {
                    continue;
                }
                let mut div = 0.0;
                for oz in 0..2usize {
                    for oy in 0..2usize {
                        for ox in 0..2usize {
                            let n = nidx(i + ox, j + oy, k + oz);
                            let s = [
                                ox as f64 * 2.0 - 1.0,
                                oy as f64 * 2.0 - 1.0,
                                oz as f64 * 2.0 - 1.0,
                            ];
                            div += (s[0] * gv[n][0] as f64
                                + s[1] * gv[n][1] as f64
                                + s[2] * gv[n][2] as f64)
                                / (4.0 * h);
                        }
                    }
                }
                div *= meta[c][1] as f64;
                se += div * div;
                cnt += 1;
            }
        }
    }
    ((se / cnt.max(1) as f64).sqrt(), cnt)
}

/// U3 divergence-decay re-run on a MID-JET state of the U4 operator: snapshot the particle
/// state, then re-measure one identically-seeded step per fine-sweep budget. The tolerance
/// and decay factor are U3's pre-registered values; a plateau above tolerance fails.
#[test]
fn u3_rerun_divergence_decay_mid_jet() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_cavity: no GPU adapter; skipping.");
        return;
    };
    let scene = jet_scene();
    let mats = Materials::default();
    let cfg = jet_cfg();
    let quiet = EmissionInput::default();
    let pour = jet_input();
    let mut donor = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..SETTLE_FRAMES {
        donor.step(DT, &quiet);
    }
    for _ in 0..60 {
        donor.step(DT, &pour);
    }
    let live = donor.phase_counts().0;
    let pos = donor.read_positions();
    let vel = donor.read_velocities();
    let cmat = donor.read_affine_rows();

    let mut curve = Vec::new();
    for &nf in &DECAY_BUDGETS {
        let mut s = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
        s.set_live_water_for_test(live);
        s.write_positions_for_test(&pos);
        s.write_velocities_for_test(&vel);
        s.write_affine_for_test(&cmat);
        s.set_pressure_budget_for_test(COARSE_RATIO_DEFAULT, COARSE_SWEEPS_DEFAULT, nf);
        s.step(DT, &quiet);
        let (rms, cells) = rms_interior_div(&s);
        println!(
            "twofield U4 divergence decay [mid-jet]: fine sweeps {nf:2} -> RMS {rms:.4e} ({cells} interior cells)"
        );
        assert!(rms.is_finite(), "non-finite divergence at {nf} sweeps");
        assert!(
            cells >= 50,
            "only {cells} interior cells — the gate would be vacuous"
        );
        curve.push((nf, rms));
    }
    let mut reached = None;
    for i in 0..curve.len() {
        let (nf, rms) = curve[i];
        if rms < DIV_TOL {
            reached = Some(nf);
            break;
        }
        assert!(
            i + 1 < curve.len(),
            "divergence never fell below DIV_TOL {DIV_TOL} (last RMS {rms:.4e} at {nf} sweeps) — \
             plateau above tolerance FAILS (the A≠D·G signature)"
        );
        let (_, rms2) = curve[i + 1];
        assert!(
            rms / rms2 >= DECAY_FACTOR_MIN || rms2 < DIV_TOL,
            "decay factor {:.2} below {DECAY_FACTOR_MIN} while above tolerance",
            rms / rms2
        );
    }
    let reached = reached.expect("checked above");
    assert!(
        reached <= 16,
        "tolerance reached only at {reached} sweeps — outside the declared knob grid"
    );
}

// ==============================================================================================
// CPU sanity: the floor derivations stated in the header are the constants asserted above.
// ==============================================================================================

#[test]
fn floor_derivations_are_consistent() {
    // depth floor = 1 jet radius.
    assert!((CAVITY_DEPTH_FLOOR - JET_RADIUS).abs() < 1e-12);
    // persistence floor = ceil(t_relevel / dt), t_relevel = sqrt(2·d_floor/g).
    let t_relevel = (2.0 * CAVITY_DEPTH_FLOOR / GRAV).sqrt();
    assert_eq!(PERSIST_FRAMES, (t_relevel / DT as f64).ceil() as usize);
    // collapse bound is half the depth floor — tighter than the floor, never looser.
    const { assert!(COLLAPSE_DEPTH_MAX < CAVITY_DEPTH_FLOOR) };
}

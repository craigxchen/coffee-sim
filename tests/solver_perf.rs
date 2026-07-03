//! U6 — the SHARED solver-parameterized perf harness (KTD10) and the PB-MPM real-time gate.
//!
//! The phenomenon under test — real-time ≤ 33 ms/frame median at ~200k particles on the
//! reference device (Apple M5) — is SOLVER-INDEPENDENT, so this harness is parameterized over
//! `SolverId` rather than living in a per-solver silo. It runs against `Pbmpm` now (the U6
//! deliverable of `docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md`); twofield/xpbd
//! enter by adding a `GateSolver`/`gate_setup` match arm and extending `GATED` — their
//! existing per-solver gates (`tests/twofield_perf.rs`) can converge here over time (KTD10:
//! not a required refactor now).
//!
//! THE PER-FRAME COST. Per-pass `timestamp-query` sum (`profile().total_micros()`) captures ONE
//! substep; PB-MPM runs exactly one substep per frame, so per-frame = 1 × the sum. The substep
//! multiplier is still carried explicitly so a substepping solver slots in without changing the
//! measurement (mirroring `tests/twofield_perf.rs`).
//!
//! DECISION INTEGRITY (R8/KTD9). Perf is measured at the FROZEN bounce-winning configuration
//! from Phase A — `Config::default()`'s pbmpm knobs — and the frozen values are asserted below
//! so a config drift cannot silently re-tune the gate. The gate constants (33 ms, the scene
//! composition, warmup/measure windows, the linearity factor) are PRE-REGISTERED here before
//! any 200k number is read into them.
//!
//! Verdict posture (NO-FALLBACK). If the 200k median misses 33 ms, the gate asserts the REAL
//! number and goes RED, naming the dominant pass — a failing-but-honest gate is the correct
//! outcome of a bet that doesn't make budget (the U7 go/no-go consumes exactly that number).
//! The gate is NOT loosened and the composition is NOT made cheaper.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::registry::SolverId;
use coffee_sim::engine::scene::{Scene, SeedRegion, Species};
use coffee_sim::models::Materials;
use coffee_sim::profiling::Profile;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::{dispatches_per_frame_for, PbmpmSolver};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;

// ======================= PRE-REGISTERED gate constants (FIXED before any run) =================

/// The hard real-time gate — ≤ 33 ms/frame median at ~200k on the reference device (Apple M5).
/// The same absolute number every solver in this harness gates against.
const REALTIME_MS_GATE: f32 = 33.0;

/// FROZEN bounce-winning configuration (R8/KTD9): the Phase A visual/bounce work settled on
/// `Config::default()`'s pbmpm knobs. Asserted against the live defaults so the perf number
/// provably reflects the bounce-winning config, not a cheaper one.
const FROZEN_ITERATION_COUNT: u32 = 16;
const FROZEN_RESTITUTION: f32 = 0.4;
const FROZEN_FLIP_FRACTION: f32 = 0.95;

/// The pre-registered gate-scene composition: single-phase water-only IMPACT scene composed by
/// box edge. A deep water column seeded ABOVE the floor free-falls and impacts during warmup
/// (the impact transient — contested P2G atomics, capped velocities — is the expensive regime
/// the measure window samples), with the center pour running from warmup/3 onward. Same scene
/// grows with N; the ~200k point cannot be passed with a cheaper mix (composition guard below).
const WATER_COL: f32 = 52.0; // column height (the whole scene is water)
const DROP_GAP: f32 = 6.0; // column bottom starts this far above the floor → real impact
const EDGE_200K: f32 = 61.0; // box edge landing ~191k seeded at spacing 1.0
const EDGE_40K: f32 = 28.0; // box edge landing ~40k (the linearity low anchor)

/// Linearity gate: max acceptable µs/Kparticle degradation 40k → 200k, PRE-REGISTERED from the
/// linear thesis (mirrors `tests/twofield_perf.rs`): a linear solver is 1.0×; small-N launch
/// amortization fades by ~40k; PB-MPM's grid passes carry a sub-linear fixed-grid component.
/// 2.5× covers a genuinely-near-linear solver while still failing a super-linear one.
const LINEARITY_FACTOR_MAX: f32 = 2.5;

const WARMUP_FRAMES: u32 = 50; // free-fall + impact + pour spin-up
const MEASURE_FRAMES: u32 = 30; // median window
const JET_FLOW: f32 = 70.0;

// ======================= the SolverId seam (KTD10) ============================================

/// A gated solver behind the `SolverId` seam. Twofield/xpbd enter as new variants with their
/// own pre-registered scene composition in `gate_setup` — the measurement below is shared.
enum GateSolver {
    Pbmpm(PbmpmSolver),
}

impl GateSolver {
    fn step(&mut self, dt: f32, input: &EmissionInput) {
        match self {
            GateSolver::Pbmpm(s) => s.step(dt, input),
        }
    }
    /// Resolve per-pass timestamps into the profile cache (blocking; dev/test only).
    fn sample_diagnostics(&mut self) {
        match self {
            GateSolver::Pbmpm(s) => s.sample_diagnostics(),
        }
    }
    fn profile(&self) -> Profile {
        match self {
            GateSolver::Pbmpm(s) => s.profile(),
        }
    }
    fn active_count(&self) -> u32 {
        match self {
            GateSolver::Pbmpm(s) => s.active_count(),
        }
    }
    /// Substeps folded into one frame (the timestamp sum covers one substep).
    fn substeps_for_dt(&self, _dt: f32) -> u32 {
        match self {
            GateSolver::Pbmpm(_) => 1, // one substep per frame (pbmpm/mod.rs step())
        }
    }
}

/// Per-solver pre-registered gate scene + build (the ONLY solver-specific code in the harness).
fn gate_setup(id: SolverId, edge: f32, gpu: &GpuContext) -> GateSolver {
    match id {
        SolverId::Pbmpm => {
            let cfg = Config::default();
            // R8/KTD9: the gate measures the FROZEN bounce-winning config — assert, don't trust.
            assert_eq!(
                cfg.pbmpm_iteration_count, FROZEN_ITERATION_COUNT,
                "pbmpm_iteration_count drifted from the frozen bounce-winning count — \
                 re-freeze deliberately (this re-baselines U6 AND the U7 cost projection)"
            );
            assert_eq!(
                cfg.pbmpm_restitution, FROZEN_RESTITUTION,
                "frozen restitution drifted"
            );
            assert_eq!(
                cfg.pbmpm_flip_fraction, FROZEN_FLIP_FRACTION,
                "frozen flip_fraction drifted"
            );
            let scene = water_impact_scene(edge);
            let mats = Materials::default();
            GateSolver::Pbmpm(PbmpmSolver::build(&scene, &mats, &cfg, gpu))
        }
        other => unimplemented!(
            "gate_setup({other:?}): add this solver's pre-registered composition to enter the \
             shared harness (KTD10)"
        ),
    }
}

/// Single-phase water-only impact scene, composed by box edge (see the constants above).
fn water_impact_scene(edge: f32) -> Scene {
    let water_lo = DROP_GAP;
    let water_hi = water_lo + WATER_COL;
    let y = (water_hi + 16.0).round();
    Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 6000.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [edge, y, edge],
        solids: Vec::new(),
        regions: vec![SeedRegion {
            min: [0.7, water_lo, 0.7],
            max: [edge - 0.7, water_hi, edge - 0.7],
            species: Species::Water,
        }],
    }
}

struct Measured {
    n: u32,
    substeps: u32,
    dispatches: u32,
    median_ms: f32,
    per_kpart: f32,
    breakdown: Vec<(String, f32)>,
}

/// Build the pre-registered scene at `edge`, warm up through the impact into the pour, and
/// return the median per-frame GPU cost (`substeps × timestamp sum`) over the measure window.
fn measure(id: SolverId, gpu: &GpuContext, edge: f32) -> Measured {
    let mut solver = gate_setup(id, edge, gpu);

    let axis = edge / 2.0;
    let spout_y = DROP_GAP + WATER_COL + 6.0;
    let quiet = EmissionInput::default();
    let pour = EmissionInput {
        kettle_pos: [axis, spout_y, axis],
        flow_rate: JET_FLOW,
        pour_angle: 0.0,
        ..EmissionInput::default()
    };
    for f in 0..WARMUP_FRAMES {
        let drive = if f < WARMUP_FRAMES / 3 { &quiet } else { &pour };
        solver.step(DT, drive);
    }

    let substeps = solver.substeps_for_dt(DT);
    let mut samples: Vec<f32> = Vec::new();
    let mut agg: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let mut dispatches = 0u32;
    for _ in 0..MEASURE_FRAMES {
        solver.step(DT, &pour);
        solver.sample_diagnostics();
        let prof = solver.profile();
        dispatches = prof.dispatches_per_frame;
        let sub = prof.total_micros();
        if sub > 0.0 {
            samples.push(sub * substeps as f32);
            for (label, us) in &prof.passes {
                *agg.entry(label.clone()).or_insert(0.0) += us * substeps as f32;
            }
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = if samples.is_empty() {
        0.0
    } else {
        samples[samples.len() / 2]
    };
    let n = solver.active_count().max(1);
    let denom = samples.len().max(1) as f32;
    let mut breakdown: Vec<(String, f32)> = agg.into_iter().map(|(k, v)| (k, v / denom)).collect();
    breakdown.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Measured {
        n,
        substeps,
        dispatches,
        median_ms: median_us / 1000.0,
        per_kpart: median_us / (n as f32 / 1000.0),
        breakdown,
    }
}

// =================================== the real-time gate ========================================

/// THE U6 GATE (pbmpm): the ~200k single-phase water impact scene runs ≤ 33 ms/frame median at
/// the FROZEN bounce-winning iteration count. NO-FALLBACK: on a miss this asserts the real
/// number and goes RED, naming the dominant pass — that number feeds the U7 go/no-go directly.
#[test]
fn pbmpm_realtime_gate_200k_water_impact() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("solver_perf: no GPU adapter; skipping.");
        return;
    };
    let m = measure(SolverId::Pbmpm, &gpu, EDGE_200K);

    println!(
        "U6 PB-MPM REAL-TIME GATE (Apple M5, frozen iteration_count {FROZEN_ITERATION_COUNT}):"
    );
    println!(
        "  composition: {} water (water-only impact scene, pre-registered ~200k)",
        m.n
    );
    println!(
        "  substeps {}, dispatches/frame {}, median {:.3} ms ({:.1} µs/Kpart)",
        m.substeps, m.dispatches, m.median_ms, m.per_kpart
    );
    println!("  per-pass breakdown (largest first):");
    let total: f32 = m.breakdown.iter().map(|(_, us)| us).sum();
    for (label, us) in m.breakdown.iter().take(10) {
        println!(
            "    {:<18} {:>9.1} µs  ({:>4.1}%)",
            label,
            us,
            100.0 * us / total.max(1.0e-6)
        );
    }

    // Composition guard: the ~200k point is pinned by the scene, not the threshold.
    assert!(
        (185_000..=215_000).contains(&m.n),
        "gate scene N={} drifted from the pre-registered ~200k window",
        m.n
    );

    assert!(
        m.median_ms <= REALTIME_MS_GATE,
        "U6 RED: 200k median {:.3} ms/frame EXCEEDS the {REALTIME_MS_GATE} ms real-time gate \
         (N={}, frozen iteration_count {FROZEN_ITERATION_COUNT}). This is the honest input the \
         U7 go/no-go consumes: the dominant pass is `{}` ({:.1} µs, {:.0}% of the frame) — the \
         iterated per-constraint-loop grid rebuild is the structural suspect (P2G atomics × \
         iteration_count; KTD6 cell-binning is the named-but-unbuilt lever). The gate is NOT \
         loosened and the composition is NOT made cheaper; record the number in the U7 decision \
         note.",
        m.median_ms,
        m.n,
        m.breakdown.first().map(|(l, _)| l.as_str()).unwrap_or("?"),
        m.breakdown.first().map(|(_, u)| *u).unwrap_or(0.0),
        100.0 * m.breakdown.first().map(|(_, u)| *u).unwrap_or(0.0) / total.max(1.0e-6),
    );
}

// =================================== linearity gate ===========================================

/// µs/Kparticle stays flat within the PRE-REGISTERED factor from 40k → 200k. The factor was
/// fixed from the linear thesis, not derived from the 200k number.
#[test]
fn pbmpm_linearity_40k_to_200k_within_preregistered_factor() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("solver_perf: no GPU adapter; skipping.");
        return;
    };
    let lo = measure(SolverId::Pbmpm, &gpu, EDGE_40K);
    let hi = measure(SolverId::Pbmpm, &gpu, EDGE_200K);
    let factor = hi.per_kpart / lo.per_kpart.max(1.0e-6);
    println!(
        "U6 pbmpm linearity: 40k N={} {:.1} µs/Kpart  →  200k N={} {:.1} µs/Kpart  =  {:.2}× \
         (pre-registered max {LINEARITY_FACTOR_MAX}×)",
        lo.n, lo.per_kpart, hi.n, hi.per_kpart, factor,
    );
    assert!(
        factor <= LINEARITY_FACTOR_MAX,
        "linearity RED: µs/Kpart degraded {factor:.2}× from 40k to 200k, above the pre-registered \
         {LINEARITY_FACTOR_MAX}× — super-linear scaling (P2G atomic contention is the structural \
         suspect; KTD6). NOT tuned to the result."
    );
}

// =================================== dispatch budget ==========================================

/// Dispatches/frame are EXACTLY the recorded formula `5 + 5·iteration_count` (no hidden growth).
#[test]
fn pbmpm_dispatch_budget_matches_recorded_formula() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("solver_perf: no GPU adapter; skipping.");
        return;
    };
    let mut solver = gate_setup(SolverId::Pbmpm, EDGE_40K, &gpu);
    solver.step(DT, &EmissionInput::default());
    let expected = dispatches_per_frame_for(FROZEN_ITERATION_COUNT);
    let got = solver.profile().dispatches_per_frame;
    println!("U6 pbmpm dispatch budget: {got} dispatches/frame (expected {expected})");
    assert_eq!(
        got, expected,
        "dispatch count drifted from 5 + 5·iteration_count at the frozen count"
    );
}

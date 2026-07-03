//! U7 — the SHARED bounce + conservation harness (KTD10), parameterized over the solver arms.
//!
//! Every arm runs the ONE pinned physical config of
//! `docs/plans/2026-07-02-001-pbmpm-u7-preregistration.md` (committed before any number here
//! was read): a settled shallow pool, a hard center-jet burst, and the localized mass-weighted
//! rebound/spread metric with its cap-hit / no-popcorn / conservation guards. The arms:
//! pbmpm constraint-only (THE decision arm), pbmpm tuned, twofield production, twofield+DensU
//! swept over K (candidate-D proxy, compared at its best arm), xpbd reference.
//!
//! Verdict posture (NO-FALLBACK): the pre-registered decision asserts live here — a RED is the
//! honest NO-GO evidence the U7 decision note records, not a tuning failure. Thresholds are
//! NOT adjusted to results; the pre-registration file is the contract.
//!
//! GPU-gated headless idiom — skips gracefully without an adapter.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::scene::{Scene, SeedRegion, Species};
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::PbmpmSolver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;

// ================= PRE-REGISTERED pinned config (see the pre-registration doc) =================

const BOX_MAX: [f32; 3] = [24.0, 40.0, 24.0];
const POOL_MIN: [f32; 3] = [1.0, 1.0, 1.0];
const POOL_MAX: [f32; 3] = [23.0, 6.0, 23.0];
const GRAVITY: [f32; 3] = [0.0, -20.0, 0.0];
const MAX_SPEED: f32 = 25.0;
const NOZZLE_RADIUS: f32 = 0.25;
const JET_POS: [f32; 3] = [12.0, 26.0, 12.0];
const JET_FLOW: f32 = 30.0;
/// Exit speed is flow/A_eff capped to MAX_SPEED; at flow 30 / r 0.25 the cap binds → 25.
const V_EXIT: f32 = MAX_SPEED;

const SETTLE_FRAMES: u32 = 240;
const BURST_FRAMES: u32 = 30;
const AFTER_FRAMES: u32 = 30;

/// Ejecta selection: one particle diameter above the pre-burst surface, outside the jet column.
const EJECTA_Y_MARGIN: f32 = 1.0;
const JET_COLUMN_R: f32 = 1.5;

// Guards (pre-registered).
const CAP_HIT_CEILING: f32 = 0.02;
const NO_POPCORN_MIN_FRAC: f32 = 0.002;
const NO_POPCORN_MAX_SHARE: f32 = 0.20; // amended pre-run, see the pre-registration doc
const KE_POST_TOLERANCE: f32 = 1.05;

// Decision thresholds (pre-registered).
const MATERIAL_BOUNCE_FACTOR: f32 = 2.0;
const DENSU_K_SWEEP: &[f32] = &[1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0];

// ======================= the arm seam (KTD10) ==================================================

#[allow(clippy::large_enum_variant)]
enum Arm {
    Pbmpm(PbmpmSolver),
    Twofield(TwofieldSolver),
    Xpbd(XpbdSolver),
}

impl Arm {
    fn step(&mut self, input: &EmissionInput) {
        match self {
            Arm::Pbmpm(s) => s.step(DT, input),
            Arm::Twofield(s) => s.step(DT, input),
            Arm::Xpbd(s) => s.step(DT, input),
        }
    }
    fn positions(&self) -> Vec<[f32; 4]> {
        match self {
            Arm::Pbmpm(s) => s.read_positions(),
            Arm::Twofield(s) => s.read_positions(),
            Arm::Xpbd(s) => s.read_positions(),
        }
    }
    fn velocities(&self) -> Vec<[f32; 4]> {
        match self {
            Arm::Pbmpm(s) => s.read_velocities(),
            Arm::Twofield(s) => s.read_velocities(),
            Arm::Xpbd(s) => s.read_velocities(),
        }
    }
    fn live(&self) -> u32 {
        match self {
            Arm::Pbmpm(s) => s.active_count(),
            Arm::Twofield(s) => s.active_count(),
            Arm::Xpbd(s) => s.active_count(),
        }
    }
}

fn pinned_scene() -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 600.0, // dose headroom for the 30-frame burst
        gravity: GRAVITY,
        box_min: [0.0, 0.0, 0.0],
        box_max: BOX_MAX,
        solids: Vec::new(),
        regions: vec![SeedRegion {
            min: POOL_MIN,
            max: POOL_MAX,
            species: Species::Water,
        }],
    }
}

/// The pinned physics, constructed directly here (R5) — never a solver-specific setup path.
fn pinned_config() -> Config {
    Config {
        nozzle_radius: NOZZLE_RADIUS,
        max_speed: MAX_SPEED,
        ..Config::default()
    }
}

struct ArmResult {
    label: String,
    rebound: f32,
    spread: f32,
    valid_frames: u32,
    window_frames: u32,
    live: u32,
    guard_note: String,
}

/// Run one arm through settle → burst → after, applying the metric each window frame.
fn run_arm(label: &str, mut arm: Arm) -> ArmResult {
    let quiet = EmissionInput::default();
    let burst = EmissionInput {
        kettle_pos: JET_POS,
        flow_rate: JET_FLOW,
        pour_angle: 0.0,
        ..EmissionInput::default()
    };
    for _ in 0..SETTLE_FRAMES {
        arm.step(&quiet);
    }
    // Baseline free surface: p95 of live particle heights on the frame before the burst.
    let live0 = arm.live() as usize;
    let mut ys: Vec<f32> = arm.positions()[..live0].iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y_s = ys[(ys.len() * 95) / 100];

    let mut rebound = 0.0f32;
    let mut spread = 0.0f32;
    let mut valid_frames = 0u32;
    let mut ke_burst_end = 0.0f32;
    let mut ke_final = 0.0f32;
    let mut guard_note = String::new();
    let window = BURST_FRAMES + AFTER_FRAMES;
    for f in 0..window {
        let drive = if f < BURST_FRAMES { &burst } else { &quiet };
        arm.step(drive);
        let live = arm.live() as usize;
        let pos = arm.positions();
        let vel = arm.velocities();
        let ke: f32 = vel[..live]
            .iter()
            .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            .sum();
        if f + 1 == BURST_FRAMES {
            ke_burst_end = ke;
        }
        if f + 1 == window {
            ke_final = ke;
        }

        // Guards + metric on this frame.
        let cap_hits = vel[..live]
            .iter()
            .filter(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() >= 0.95 * MAX_SPEED)
            .count();
        let cap_frac = cap_hits as f32 / live.max(1) as f32;

        let mut flux = 0.0f32;
        let mut max_contrib = 0.0f32;
        let mut radii: Vec<f32> = Vec::new();
        for i in 0..live {
            let p = pos[i];
            let dx = p[0] - JET_POS[0];
            let dz = p[2] - JET_POS[2];
            let r = (dx * dx + dz * dz).sqrt();
            if p[1] > y_s + EJECTA_Y_MARGIN && r > JET_COLUMN_R {
                let up = vel[i][1].max(0.0);
                flux += up;
                max_contrib = max_contrib.max(up);
                radii.push(r);
            }
        }
        let ejecta = radii.len();
        let min_ejecta = ((live as f32 * NO_POPCORN_MIN_FRAC).ceil() as usize).max(10);
        let popcorn_ok =
            ejecta >= min_ejecta && (flux <= 0.0 || max_contrib <= NO_POPCORN_MAX_SHARE * flux);
        if cap_frac < CAP_HIT_CEILING && popcorn_ok {
            valid_frames += 1;
            let r_frame = flux / (live.max(1) as f32 * V_EXIT);
            if r_frame > rebound {
                rebound = r_frame;
            }
            radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let s_frame = radii[(radii.len() * 95) / 100];
            if s_frame > spread {
                spread = s_frame;
            }
        } else if guard_note.is_empty() && f < BURST_FRAMES {
            guard_note = format!(
                "first invalid frame {f}: cap_frac {cap_frac:.3}, ejecta {ejecta} (min {min_ejecta})"
            );
        }
    }
    // Conservation guard: no energy manufactured after forcing stops.
    assert!(
        ke_final <= ke_burst_end.max(1e-6) * KE_POST_TOLERANCE,
        "{label}: KE grew after the burst ({ke_burst_end:.2} → {ke_final:.2}) — conservation \
         guard failed; the metric window is invalid for this arm"
    );

    ArmResult {
        label: label.to_string(),
        rebound,
        spread,
        valid_frames,
        window_frames: window,
        live: arm.live(),
        guard_note,
    }
}

fn print_table(rows: &[&ArmResult]) {
    println!(
        "{:<28} {:>9} {:>8} {:>12} {:>8}",
        "arm", "R", "S", "valid/window", "live"
    );
    for r in rows {
        println!(
            "{:<28} {:>9.5} {:>8.2} {:>7}/{:<4} {:>8}  {}",
            r.label, r.rebound, r.spread, r.valid_frames, r.window_frames, r.live, r.guard_note
        );
    }
}

/// THE U7 COMPARISON: every arm on the pinned config, one measurement pass, the pre-registered
/// decision asserts applied to the printed table. RED here = the honest NO-GO evidence.
#[test]
fn u7_pinned_bounce_comparison() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("solver_physics: no GPU adapter; skipping.");
        return;
    };
    let scene = pinned_scene();
    let mats = Materials::default();
    let cfg = pinned_config();

    // Arm 1: pbmpm constraint-only (restitution = 0) — THE decision arm.
    let constraint_only = {
        let mut s = PbmpmSolver::build(&scene, &mats, &cfg, &gpu);
        s.set_restitution_for_test(0.0);
        run_arm("pbmpm constraint-only", Arm::Pbmpm(s))
    };
    // Arm 2: pbmpm tuned (frozen restitution from Config::default) — recorded, not decisive.
    let tuned = run_arm(
        "pbmpm tuned",
        Arm::Pbmpm(PbmpmSolver::build(&scene, &mats, &cfg, &gpu)),
    );
    // Arm 3: twofield production — the baseline.
    let twofield = run_arm(
        "twofield production",
        Arm::Twofield(TwofieldSolver::build(&scene, &mats, &cfg, &gpu)),
    );
    // Arm 4: twofield + DensU proxy, swept over K, compared at its best (highest valid R) arm.
    let mut best_densu: Option<ArmResult> = None;
    for &k in DENSU_K_SWEEP {
        let mut s = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
        s.set_density_target_mode_for_test(true);
        s.set_density_rate_k_for_test(k);
        let r = run_arm(&format!("twofield DensU K={k}"), Arm::Twofield(s));
        if best_densu.as_ref().map(|b| r.rebound > b.rebound).unwrap_or(true) {
            best_densu = Some(r);
        }
    }
    let best_densu = best_densu.expect("DensU sweep is non-empty");
    // Arm 5: xpbd reference.
    let xpbd = run_arm(
        "xpbd production",
        Arm::Xpbd(XpbdSolver::build(&scene, &mats, &cfg, &gpu)),
    );

    println!("\nU7 PINNED BOUNCE COMPARISON (pre-registered config; R normalized by N·v_exit):");
    print_table(&[&constraint_only, &tuned, &twofield, &best_densu, &xpbd]);

    // Metric validity: the decision arm must have produced valid guarded samples at all.
    assert!(
        constraint_only.valid_frames > 0,
        "decision arm produced NO guard-valid frames — the metric window is unusable \
         ({})",
        constraint_only.guard_note
    );

    // PRE-REGISTERED decision asserts (docs/plans/2026-07-02-001):
    // (1) material bounce — constraint-only ≥ 2× twofield production.
    let material = constraint_only.rebound >= MATERIAL_BOUNCE_FACTOR * twofield.rebound
        && constraint_only.rebound > 0.0;
    // (2) beats the best DensU proxy arm (sign).
    let beats_densu = constraint_only.rebound > best_densu.rebound;
    assert!(
        material,
        "U7 RED (NO-GO evidence): constraint-only pbmpm R {:.5} is NOT material vs twofield R \
         {:.5} (pre-registered factor {MATERIAL_BOUNCE_FACTOR}×). Record in the decision note; \
         the threshold is NOT adjusted.",
        constraint_only.rebound, twofield.rebound
    );
    assert!(
        beats_densu,
        "U7 RED (NO-GO evidence): constraint-only pbmpm R {:.5} does not beat the best DensU \
         proxy arm ({} R {:.5}). Record in the decision note.",
        constraint_only.rebound, best_densu.label, best_densu.rebound
    );
}

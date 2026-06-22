//! HEADLESS settle-calibration harness for the PB-MPM solver (`SolverId::Pbmpm`).
//!
//! Empirically calibrates the FLIP dissipation knob (`set_flip_fraction_for_test`) by measuring how
//! long the water takes to SETTLE after the finite pour stops. The pour scene
//! (`Scene::debug_high_velocity_jet_impact`) emits a fixed 150 mL dose, so `water_count` plateaus on
//! its own = "faucet off". From faucet-off we keep stepping and watch the water speed decay.
//!
//! METRIC (corrected): "settled" is NOT the global MEAN speed. A calm bulk pool dilutes the mean while
//! a subset (recirculating column, splash ejecta, stuck/fast particles) keeps moving — the mean reads
//! "settled" while real motion persists. "Settled" must be gated on the SLOWEST-to-calm particle, i.e.
//! the MAX speed (and high percentiles p95/p99). Each settle frame we compute the FULL speed
//! distribution {mean, p50, p95, p99, max} over the live water plus the FRACTION of particles still
//! moving faster than 1.0 / 2.0 su/s. The headline metric is `max < 0.5` and `p99 < 0.5` su/s
//! ("essentially all at rest"). Stats are also split inside-cup vs outside so we can tell real
//! recirculation in the cup from spilled water settling in the box.
//!
//! Time scale: physical_seconds = sim_seconds × τ (τ = 3.685; the sim is slow-motion). dt = 1/60
//! sim-second/frame ⇒ frames = sim_seconds × 60 ⇒ phys_s = frames/60 × τ.
//!
//! Run: `cargo run --example pbmpm_settle --release`  (GPU readbacks every frame; offline, no perf
//! gate). Skips cleanly when no GPU adapter is present.

use coffee_sim::emission::{EmissionInput, PourEvent};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::PbmpmSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

/// Slow-motion time scale: physical_seconds = sim_seconds × TAU.
const TAU: f32 = 3.685;
/// One sim-second is 60 frames at dt = 1/60.
const DT: f32 = 1.0 / 60.0;

/// Rest thresholds (sim-units/s) reported per distribution metric.
const REST_THRESHOLDS: [f32; 2] = [1.0, 0.5];
/// Still-moving fraction cutoffs (sim-units/s): fraction of live water faster than each.
const MOVING_CUTOFFS: [f32; 2] = [1.0, 2.0];

/// Catch-cup cylinder from `v60_dripper()`: floor_y=-8.0, rim_y=-3.5, radius=3.0. "Inside cup" =
/// horizontal r < CUP_R and y in [CUP_FLOOR_Y, CUP_RIM_Y]; everything else is "outside" (spilled into
/// the box, splash ejecta, or still in the dripper above).
const CUP_R: f32 = 3.0;
const CUP_FLOOR_Y: f32 = -8.0;
const CUP_RIM_Y: f32 = -3.5;

/// FILL = a deterministic high-velocity jet BURST: pour hard for this many frames, then cut the flow
/// to zero. A short hard burst makes faucet-off a sharp, known event with a real post-impact speed
/// transient to settle from.
const BURST_FRAMES: usize = 60;
/// Volumetric flow during the burst (exit speed = flow / A_eff; capped to max_speed = 25 — a genuine
/// high-velocity impacting jet).
const BURST_FLOW: f32 = 30.0;
/// Generous cap (frames) on the settle phase = +1800 frames = 30 sim-s ≈ 110 phys-s, so a run that
/// NEVER calms (permanent recirculation cycle) is visible rather than truncated early.
const SETTLE_CAP: usize = 1800;

/// frames → physical seconds.
fn phys_s(frames: usize) -> f32 {
    frames as f32 / 60.0 * TAU
}

/// Build the v60 water-only materials/config the `HighVelocityJetImpact` web scene uses.
fn water_setup() -> (Materials, Config) {
    let r = 0.16_f32;
    let mats = Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    };
    let cfg = Config {
        nozzle_radius: 0.25,
        max_speed: 25.0,
        xsph_viscosity_c: 0.02,
        ..Config::default()
    };
    (mats, cfg)
}

/// Full speed-distribution snapshot over the live water for one frame, split by cup region.
struct Stats {
    mean: f32,
    p50: f32,
    p95: f32,
    p99: f32,
    max: f32,
    /// fraction of live water with speed > MOVING_CUTOFFS[i].
    moving_frac: [f32; 2],
    /// max speed of water INSIDE the cup cylinder.
    max_in_cup: f32,
    /// max speed of water OUTSIDE the cup cylinder.
    max_outside: f32,
    /// fraction (of ALL live water) moving > 1.0 su/s and inside the cup.
    moving_frac_in_cup: f32,
    /// fraction (of ALL live water) moving > 1.0 su/s and outside the cup.
    moving_frac_outside: f32,
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((q * (sorted.len() as f32 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Read positions + velocities over the live water `[0, active_count)` and compute the full
/// distribution + region split (two GPU read-backs per call).
fn snapshot(solver: &PbmpmSolver) -> Stats {
    let n = solver.active_count() as usize;
    if n == 0 {
        return Stats {
            mean: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            max: 0.0,
            moving_frac: [0.0; 2],
            max_in_cup: 0.0,
            max_outside: 0.0,
            moving_frac_in_cup: 0.0,
            moving_frac_outside: 0.0,
        };
    }
    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    let mut speeds: Vec<f32> = Vec::with_capacity(n);
    let mut sum = 0.0f64;
    let mut moving = [0usize; 2];
    let mut max_in_cup = 0.0f32;
    let mut max_outside = 0.0f32;
    let mut moving_in_cup = 0usize;
    let mut moving_outside = 0usize;
    for i in 0..n {
        let v = vel[i];
        let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        speeds.push(s);
        sum += s as f64;
        for (j, &c) in MOVING_CUTOFFS.iter().enumerate() {
            if s > c {
                moving[j] += 1;
            }
        }
        let p = pos[i];
        let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
        let in_cup = r < CUP_R && p[1] >= CUP_FLOOR_Y && p[1] <= CUP_RIM_Y;
        if in_cup {
            max_in_cup = max_in_cup.max(s);
            if s > MOVING_CUTOFFS[0] {
                moving_in_cup += 1;
            }
        } else {
            max_outside = max_outside.max(s);
            if s > MOVING_CUTOFFS[0] {
                moving_outside += 1;
            }
        }
    }
    speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let nf = n as f32;
    Stats {
        mean: (sum / n as f64) as f32,
        p50: percentile(&speeds, 0.50),
        p95: percentile(&speeds, 0.95),
        p99: percentile(&speeds, 0.99),
        max: *speeds.last().unwrap(),
        moving_frac: [moving[0] as f32 / nf, moving[1] as f32 / nf],
        max_in_cup,
        max_outside,
        moving_frac_in_cup: moving_in_cup as f32 / nf,
        moving_frac_outside: moving_outside as f32 / nf,
    }
}

/// A sampled point on the decay curve (frames since faucet-off + the stats at that frame).
struct DecaySample {
    k: usize,
    mean: f32,
    p50: f32,
    p95: f32,
    p99: f32,
    max: f32,
    moving1: f32,
    moving2: f32,
    max_in_cup: f32,
    max_outside: f32,
    moving_in_cup: f32,
    moving_outside: f32,
}

impl DecaySample {
    fn from_stats(k: usize, s: &Stats) -> Self {
        DecaySample {
            k,
            mean: s.mean,
            p50: s.p50,
            p95: s.p95,
            p99: s.p99,
            max: s.max,
            moving1: s.moving_frac[0],
            moving2: s.moving_frac[1],
            max_in_cup: s.max_in_cup,
            max_outside: s.max_outside,
            moving_in_cup: s.moving_frac_in_cup,
            moving_outside: s.moving_frac_outside,
        }
    }
}

/// Outcome of one flip sweep run.
struct RunResult {
    flip: f32,
    faucet_off_frame: usize,
    water_at_off: u32,
    /// stats snapshot at faucet-off.
    at_off: SpeedSummary,
    /// per (metric, threshold) settle frame relative to faucet-off; None = never within the cap.
    /// metrics order: [mean, p95, p99, max]; thresholds order: [1.0, 0.5].
    settle: [[Option<usize>; 2]; 4],
    /// residual distribution at the end of the settle phase.
    residual: SpeedSummary,
    /// residual still-moving fractions at the end.
    residual_moving: [f32; 2],
    /// decay curve samples.
    decay: Vec<DecaySample>,
}

/// Compact summary of the four headline metrics for printing.
#[derive(Clone, Copy)]
struct SpeedSummary {
    mean: f32,
    p95: f32,
    p99: f32,
    max: f32,
}

impl From<&Stats> for SpeedSummary {
    fn from(s: &Stats) -> Self {
        SpeedSummary {
            mean: s.mean,
            p95: s.p95,
            p99: s.p99,
            max: s.max,
        }
    }
}

/// Run one fill+settle measurement at a fixed flip fraction (and viscosity).
fn run_settle(
    gpu: &GpuContext,
    scene: &Scene,
    mats: &Materials,
    cfg: &Config,
    flip: f32,
    viscosity: Option<f32>,
) -> RunResult {
    let mut solver = PbmpmSolver::build(scene, mats, cfg, gpu);
    solver.set_flip_fraction_for_test(flip);
    if let Some(v) = viscosity {
        solver.set_liquid_viscosity_for_test(v);
    }

    let pour = EmissionInput {
        kettle_pos: [0.0, scene.box_max[1] - 1.0, 0.0],
        flow_rate: BURST_FLOW,
        pour_angle: 0.0,
        event: PourEvent::None,
    };
    let off = EmissionInput {
        flow_rate: 0.0,
        ..pour
    };

    // --- FILL phase: a deterministic high-velocity burst, then cut the flow. ---
    for _ in 0..BURST_FRAMES {
        solver.step(DT, &pour);
    }
    let faucet_off_frame = BURST_FRAMES;
    let water_at_off = solver.active_count();
    let at_off_stats = snapshot(&solver);
    let at_off = SpeedSummary::from(&at_off_stats);

    // --- SETTLE phase: flow off; watch the full distribution decay over a generous window. ---
    // metrics order: [mean, p95, p99, max].
    let mut settle: [[Option<usize>; 2]; 4] = [[None; 2]; 4];
    let mut decay: Vec<DecaySample> = vec![DecaySample::from_stats(0, &at_off_stats)];
    let mut residual = at_off;
    let mut residual_moving = at_off_stats.moving_frac;
    for k in 1..=SETTLE_CAP {
        solver.step(DT, &off);
        let s = snapshot(&solver);
        residual = SpeedSummary::from(&s);
        residual_moving = s.moving_frac;
        let metrics = [s.mean, s.p95, s.p99, s.max];
        for (mi, &mv) in metrics.iter().enumerate() {
            for (ti, &thr) in REST_THRESHOLDS.iter().enumerate() {
                if settle[mi][ti].is_none() && mv < thr {
                    settle[mi][ti] = Some(k);
                }
            }
        }
        // Decay sampling: ~every 0.25 sim-s (15 frames) for the first 5 sim-s, then every 1 sim-s.
        let sample = if k <= 300 { k % 15 == 0 } else { k % 60 == 0 };
        if sample {
            decay.push(DecaySample::from_stats(k, &s));
        }
        // Early-out only once EVERY metric (incl. max) has crossed its strictest threshold.
        if settle.iter().all(|m| m.iter().all(|t| t.is_some())) {
            break;
        }
    }

    RunResult {
        flip,
        faucet_off_frame,
        water_at_off,
        at_off,
        settle,
        residual,
        residual_moving,
        decay,
    }
}

const METRIC_NAMES: [&str; 4] = ["mean", "p95", "p99", "max "];

/// Print the per-flip settle table: for each metric, frame/sim-s/phys-s at <1.0 and <0.5.
fn print_settle_table(r: &RunResult) {
    println!(
        "  flip {:.2}  (faucet off @ frame {}, {} water particles)",
        r.flip, r.faucet_off_frame, r.water_at_off
    );
    println!(
        "    speed@off: mean {:.3}  p95 {:.3}  p99 {:.3}  max {:.3}",
        r.at_off.mean, r.at_off.p95, r.at_off.p99, r.at_off.max
    );
    println!("    metric | <1.0 (frame / sim-s / phys-s)        | <0.5 (frame / sim-s / phys-s)");
    let fmt = |s: Option<usize>| match s {
        Some(f) => format!("{f:>4} / {:>6.3} / {:>6.2}", f as f32 / 60.0, phys_s(f)),
        None => " --- never (within 30 sim-s) ---".to_string(),
    };
    for (mi, name) in METRIC_NAMES.iter().enumerate() {
        println!(
            "    {name}   | {:<37} | {}",
            fmt(r.settle[mi][0]),
            fmt(r.settle[mi][1])
        );
    }
    println!(
        "    residual @ end: mean {:.3}  p95 {:.3}  p99 {:.3}  max {:.3}  (moving>1.0 {:.1}%, >2.0 {:.1}%)",
        r.residual.mean,
        r.residual.p95,
        r.residual.p99,
        r.residual.max,
        r.residual_moving[0] * 100.0,
        r.residual_moving[1] * 100.0
    );
}

/// Print the per-flip decay curve: mean / p95 / p99 / max + still-moving fractions + region maxes.
fn print_decay(r: &RunResult) {
    println!("  flip {:.2}:", r.flip);
    println!(
        "    {:>5} {:>6} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7}",
        "frame",
        "phys-s",
        "mean",
        "p50",
        "p95",
        "p99",
        "max",
        "mv>1.0%",
        "mv>2.0%",
        "max_cup",
        "max_box",
        "mv_cup%",
        "mv_box%"
    );
    for d in &r.decay {
        println!(
            "    {:>5} {:>6.2} | {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} | {:>6.1}% {:>6.1}% | {:>7.3} {:>7.3} | {:>6.1}% {:>6.1}%",
            d.k,
            phys_s(d.k),
            d.mean,
            d.p50,
            d.p95,
            d.p99,
            d.max,
            d.moving1 * 100.0,
            d.moving2 * 100.0,
            d.max_in_cup,
            d.max_outside,
            d.moving_in_cup * 100.0,
            d.moving_outside * 100.0
        );
    }
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_settle: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::debug_high_velocity_jet_impact();
    let (mats, cfg) = water_setup();

    println!("=== PB-MPM settle calibration (debug_high_velocity_jet_impact, 150 mL dose) ===");
    println!(
        "  time scale TAU = {TAU} (phys_s = frames/60 x TAU); dt = 1/60 sim-s; settle cap = {SETTLE_CAP} frames = {:.1} phys-s",
        phys_s(SETTLE_CAP)
    );
    println!(
        "  METRIC: full speed distribution (mean/p95/p99/max) + still-moving fraction; headline = max<0.5 & p99<0.5"
    );
    println!(
        "  region split: inside cup = r<{CUP_R} & y in [{CUP_FLOOR_Y}, {CUP_RIM_Y}]; outside = box/splash/dripper"
    );
    println!();

    // --- FLIP SWEEP ---
    let flips = [0.0f32, 0.5, 0.7, 0.8, 0.9, 0.95];
    let mut results: Vec<RunResult> = Vec::new();
    for &flip in &flips {
        let r = run_settle(&gpu, &scene, &mats, &cfg, flip, None);
        results.push(r);
    }

    println!("--- FLIP SWEEP: per-metric settle times (frame / sim-s / phys-s to first cross) ---");
    for r in &results {
        print_settle_table(r);
        println!();
    }

    println!("--- FLIP SWEEP: HEADLINE summary (max<0.5 & p99<0.5 = essentially all at rest) ---");
    println!(
        "  {:>4} | {:>22} | {:>22} | {:>10}",
        "flip", "p99<0.5 (frame/phys-s)", "max<0.5 (frame/phys-s)", "resid max"
    );
    for r in &results {
        let fmt = |s: Option<usize>| match s {
            Some(f) => format!("{f:>5} / {:>6.2}", phys_s(f)),
            None => "  never".to_string(),
        };
        println!(
            "  {:>4.2} | {:>22} | {:>22} | {:>10.3}",
            r.flip,
            fmt(r.settle[2][1]),
            fmt(r.settle[3][1]),
            r.residual.max
        );
    }
    println!();

    println!(
        "--- FLIP SWEEP: decay curves (mean/p95/p99/max + still-moving frac + region max) ---"
    );
    for r in &results {
        print_decay(r);
        println!();
    }
}

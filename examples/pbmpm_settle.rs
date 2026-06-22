//! HEADLESS settle-calibration harness for the PB-MPM solver (`SolverId::Pbmpm`).
//!
//! Empirically calibrates the FLIP dissipation knob (`set_flip_fraction_for_test`) by measuring how
//! long the water takes to SETTLE after the finite pour stops. The pour scene
//! (`Scene::debug_high_velocity_jet_impact`) emits a fixed 150 mL dose, so `water_count` plateaus on
//! its own = "faucet off". From faucet-off we keep stepping and watch the GLOBAL mean water speed
//! decay; "settled" = mean speed below a rest threshold.
//!
//! Time scale: physical_seconds = sim_seconds × τ (τ = 3.685; the sim is slow-motion). dt = 1/60
//! sim-second/frame ⇒ frames = sim_seconds × 60 ⇒ phys_s = frames/60 × τ. A realistic 2–3 PHYSICAL-
//! second settle = 0.54–0.81 sim-s = ~33–49 frames after the faucet stops.
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

/// Rest thresholds on the global mean water speed (sim-units/s) — "settled" is reported at several so
/// the choice isn't arbitrary. The 2–3 phys-s target is keyed to the middle one (< 1.0).
const ABS_THRESHOLDS: [f32; 3] = [2.0, 1.0, 0.5];
/// Relative-rest gate: also report when the speed falls below this fraction of the post-impact PEAK.
const REL_THRESHOLD: f32 = 0.05;

/// FILL = a deterministic high-velocity jet BURST: pour hard for this many frames, then cut the flow
/// to zero. A short hard burst makes faucet-off a sharp, known event with a real post-impact speed
/// transient to settle from (a long steady pour reaches near-rest while pouring, so there is no
/// transient — and faucet-off speed approaches the rest level itself, which the < thresholds and the
/// relative gate cannot resolve). The burst delivers a partial dose; the dose never fully exhausts,
/// so the flow is cut explicitly rather than detected from a count plateau.
const BURST_FRAMES: usize = 60;
/// Volumetric flow during the burst (exit speed = flow / A_eff; with A_eff = pi*0.25^2 this is a
/// ~150 su/s jet, capped to max_speed = 25 — a genuine high-velocity impacting jet).
const BURST_FLOW: f32 = 30.0;
/// Safety cap (frames) on the settle phase so a non-settling run still terminates.
const SETTLE_CAP: usize = 1200;

/// frames → physical seconds.
fn phys_s(frames: usize) -> f32 {
    frames as f32 / 60.0 * TAU
}

/// Build the v60 water-only materials/config the `HighVelocityJetImpact` web scene uses (mirrors
/// `web::v60_water_setup` for the non-twofield path).
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

/// Mean over the live water `[0, water_count)` of `length(vel.xyz)` (one full velocity read-back).
fn mean_water_speed(solver: &PbmpmSolver) -> f32 {
    let n = solver.active_count() as usize;
    if n == 0 {
        return 0.0;
    }
    let vel = solver.read_velocities();
    let mut sum = 0.0f64;
    for v in vel.iter().take(n) {
        sum += ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()) as f64;
    }
    (sum / n as f64) as f32
}

/// Outcome of one flip sweep run.
struct RunResult {
    flip: f32,
    speed_at_off: f32,
    faucet_off_frame: usize,
    water_at_off: u32,
    /// post-impact peak mean speed (over the first ~0.5 sim-s of the settle phase).
    peak: f32,
    /// settle frame (relative to faucet-off) per absolute threshold; None = never reached.
    settle_frames: [Option<usize>; 3],
    /// frame the speed first fell below REL_THRESHOLD × peak; None = never reached.
    rel_settle_frame: Option<usize>,
    /// residual mean speed at the end of the settle phase.
    residual: f32,
    /// coarse decay curve: (frames_since_off, mean_speed).
    decay: Vec<(usize, f32)>,
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

    // Active-pour input (mirrors the scaffold test + web pour): kettle above the cup, pouring
    // straight down a high-velocity jet. FILL pours hard for BURST_FRAMES, then the flow is cut to
    // zero (faucet off) so emit() stops.
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
    let speed_at_off = mean_water_speed(&solver);

    // --- SETTLE phase: flow is off (emit() no-ops); watch the impact transient decay. The relative
    // gate is keyed to the post-impact PEAK speed (the airborne burst column is still arriving for a
    // few frames after the cut, so the peak is slightly past faucet-off, not at it). ---
    let mut settle_frames: [Option<usize>; 3] = [None; 3];
    let mut rel_settle_frame: Option<usize> = None;
    let mut decay: Vec<(usize, f32)> = vec![(0, speed_at_off)];
    let mut peak = speed_at_off;
    let mut residual = speed_at_off;
    for k in 1..=SETTLE_CAP {
        solver.step(DT, &off);
        let mean = mean_water_speed(&solver);
        residual = mean;
        // Peak tracks the impact transient over the first ~0.5 sim-s (the arriving burst column),
        // then freezes so the relative gate measures decay below that peak.
        if k <= 30 {
            peak = peak.max(mean);
        }
        // Coarse decay sample: every ~6 frames (0.1 sim-s) for the first ~2 sim-s, then every 30.
        let sample = if k <= 120 { k % 6 == 0 } else { k % 30 == 0 };
        if sample {
            decay.push((k, mean));
        }
        for (i, &thr) in ABS_THRESHOLDS.iter().enumerate() {
            if settle_frames[i].is_none() && mean < thr {
                settle_frames[i] = Some(k);
            }
        }
        if rel_settle_frame.is_none() && mean < REL_THRESHOLD * peak {
            rel_settle_frame = Some(k);
        }
        if settle_frames.iter().all(|s| s.is_some()) && rel_settle_frame.is_some() {
            break;
        }
    }

    RunResult {
        flip,
        speed_at_off,
        faucet_off_frame,
        water_at_off,
        peak,
        settle_frames,
        rel_settle_frame,
        residual,
        decay,
    }
}

/// Print the per-flip summary table row + the threshold breakdown.
fn print_row(r: &RunResult) {
    let fmt = |s: Option<usize>| match s {
        Some(f) => format!("{f:>4} / {:>5.3} / {:>5.2}", f as f32 / 60.0, phys_s(f)),
        None => "  -- did not settle --".to_string(),
    };
    let settled = if r.settle_frames[1].is_some() {
        "yes"
    } else {
        "NO"
    };
    println!(
        "{:>4.2} | {:>9.3} | {} | {}",
        r.flip,
        r.speed_at_off,
        fmt(r.settle_frames[1]),
        settled
    );
    // Full threshold breakdown for this flip (absolute thresholds + the relative-to-peak gate).
    for (i, &thr) in ABS_THRESHOLDS.iter().enumerate() {
        println!(
            "       <{thr:>3.1}      : {}",
            match r.settle_frames[i] {
                Some(f) => format!(
                    "frame {f:>4} | {:>5.3} sim-s | {:>5.2} phys-s",
                    f as f32 / 60.0,
                    phys_s(f)
                ),
                None => format!("did not settle (residual {:>6.3} su/s)", r.residual),
            }
        );
    }
    println!(
        "       <5%*peak  : {}  (peak {:.3} su/s -> floor {:.3})",
        match r.rel_settle_frame {
            Some(f) => format!(
                "frame {f:>4} | {:>5.3} sim-s | {:>5.2} phys-s",
                f as f32 / 60.0,
                phys_s(f)
            ),
            None => format!("did not settle (residual {:>6.3} su/s)", r.residual),
        },
        r.peak,
        REL_THRESHOLD * r.peak
    );
}

/// Print the coarse decay curve for one run.
fn print_decay(r: &RunResult) {
    println!(
        "  flip {:.2}  (faucet off @ frame {}, {} water particles, speed@off {:.3}, peak {:.3} su/s):",
        r.flip, r.faucet_off_frame, r.water_at_off, r.speed_at_off, r.peak
    );
    print!("    ");
    for (i, (k, mean)) in r.decay.iter().enumerate() {
        print!("{:>4}f:{:>6.2}", k, mean);
        if (i + 1) % 6 == 0 {
            println!();
            print!("    ");
        } else {
            print!("   ");
        }
    }
    println!();
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
        "  time scale TAU = {TAU} (phys_s = frames/60 x TAU); dt = 1/60 sim-s; target settle 2-3 phys-s = ~33-49 frames @ <1.0"
    );
    println!();

    // --- FLIP SWEEP ---
    let flips = [0.0f32, 0.5, 0.7, 0.8, 0.9, 0.95];
    let mut results: Vec<RunResult> = Vec::new();
    for &flip in &flips {
        let r = run_settle(&gpu, &scene, &mats, &cfg, flip, None);
        results.push(r);
    }

    println!("--- FLIP SWEEP: settle table (settle is at the <1.0 su/s threshold) ---");
    println!("flip | speed@off | settle(<1.0): frame / sim_s / phys_s | settled?");
    for r in &results {
        print_row(r);
    }
    println!();

    println!("--- FLIP SWEEP: coarse mean-speed decay curves (frames since faucet-off) ---");
    for r in &results {
        print_decay(r);
    }
    println!();

    // Pick the flip whose <1.0 settle lands closest to the 2-3 phys-s target (~33-49 frames).
    const TARGET_LO: usize = 33;
    const TARGET_HI: usize = 49;
    let target_mid = (TARGET_LO + TARGET_HI) as f32 / 2.0;
    let best = results
        .iter()
        .filter_map(|r| r.settle_frames[1].map(|f| (r.flip, f)))
        .min_by(|a, b| {
            let da = (a.1 as f32 - target_mid).abs();
            let db = (b.1 as f32 - target_mid).abs();
            da.partial_cmp(&db).unwrap()
        });
    match best {
        Some((flip, f)) => {
            let on_target = (TARGET_LO..=TARGET_HI).contains(&f);
            println!(
                "==> Closest to 2-3 phys-s target: flip {flip:.2} settles (<1.0) at frame {f} = {:.2} phys-s {}",
                phys_s(f),
                if on_target { "(IN target band)" } else { "(outside the 33-49 band)" }
            );
        }
        None => println!("==> No flip settled below 1.0 su/s within the settle cap."),
    }
    println!();

    // --- VISCOSITY SWEEP at the flip nearest the target (the literal diffusion term) ---
    if let Some((flip, _)) = best {
        println!("--- VISCOSITY SWEEP (flip fixed at {flip:.2}; settle is at <1.0 su/s) ---");
        println!("visc | speed@off | settle(<1.0): frame / sim_s / phys_s | settled?");
        for &visc in &[0.0f32, 0.05, 0.1, 0.2] {
            let r = run_settle(&gpu, &scene, &mats, &cfg, flip, Some(visc));
            let fmt = match r.settle_frames[1] {
                Some(f) => format!("{f:>4} / {:>5.3} / {:>5.2}", f as f32 / 60.0, phys_s(f)),
                None => format!("-- did not settle (residual {:.3}) --", r.residual),
            };
            let settled = if r.settle_frames[1].is_some() {
                "yes"
            } else {
                "NO"
            };
            println!(
                "{visc:>4.2} | {:>9.3} | {} | {}",
                r.speed_at_off, fmt, settled
            );
        }
    }
}

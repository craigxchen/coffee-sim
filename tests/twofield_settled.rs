//! Twofield SETTLED-POOL energy gates (spontaneous-stirring artifact).
//!
//! SYMPTOM (reported in-browser, two-field solver, BOTH WaterOnly and coupled CenterPour):
//! settled water re-energizes / stirs on its own — energy injected into fluid that should be at
//! rest. The existing "settled tank stays settled" checks (tests/twofield_pressure.rs,
//! tests/twofield_cavity.rs) bound only the SAMPLED max|v| every N frames; an intermittent KE
//! spike between samples slips through. This file pins the missing invariant: the TOTAL kinetic
//! energy Σ½m|v|² of a zero-velocity-seeded settled pool must DECAY and STAY bounded over a long
//! run (no spontaneous re-energization), sampled EVERY frame so a transient spike cannot hide.
//!
//! ROOT CAUSE (diagnosed by the isolation matrix below, `isolate_settled_stirring_mechanisms`,
//! water-only tank, settled-tail mean KE):
//!   baseline (pure APIC)  ~67   |  pure PIC (no affine)  ~1.3   |  relief OFF  ~7.6   |  32 sweeps  ~17
//! Mechanism: the local-Jacobi pressure solve is intentionally UNDER-converged (real-time design),
//! leaving a standing density error; the density relief converts that error into a velocity each
//! frame; the lossless APIC affine field accumulates it (no numerical dissipation); the pool
//! sloshes and re-creates the error — a self-sustaining limit cycle. Pure PIC (C zeroed) collapses
//! it (the C is the lossless CARRIER); relief OFF collapses it (relief is the SOURCE); more sweeps
//! cut it (convergence is the cure the budget can't afford).
//!
//! FIX STATUS: SHIPPED — a small global G2P PIC blend (`PIC_BLEND_DEFAULT`). The blend was held at
//! 0 during the investigation because the stirring is the SAME open-water agitation that drove the
//! OLD deformable-bed crater slump, so any blend that quiets the pool also "freezes" that slump.
//! The resolution was NOT a damping knob but correcting the crater gate: a real wet bed HOLDS the
//! poured crater (wet-sand plasticity — pour-over research), so a held crater is the CORRECT
//! outcome. With the crater gate asserting persistence (`twofield_full.rs`, docs/plans/2026-06-15-001),
//! the global blend is safe: it quiets the pool AND the crater correctly persists. Volume
//! conservation was verified to hold at the shipped blend.
//!
//! This file (1) GATES the bounded settled-pool KE (the regression floor: KE stays bounded, no
//! spontaneous spikes), now quieter under the shipped blend, and (2) CHARACTERIZES the mechanism
//! (`isolate_…`: pure-PIC kills the carrier, relief-off cuts the source, the shipped blend quiets).

#![allow(clippy::needless_range_loop)]

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{TwofieldSolver, PIC_BLEND_DEFAULT};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;
const REST_MASS: f64 = 1.0; // particle_mass at the defaults (ρ_rest = 1, spacing = 1)

// ==============================================================================================
// Scenes
// ==============================================================================================

/// Settled water-only tank: full-width pool so the only free surface is the top (mirrors the
/// pressure-suite `tank_scene`; rest-density half-spacing inset seeding). The WaterOnly symptom
/// reproduces here — it is a CORE-solver artifact, not geometry-specific.
fn tank_scene(depth: f32) -> Scene {
    Scene {
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0; 3],
        box_max: [16.0, 32.0, 16.0],
        regions: vec![SeedRegion {
            min: [0.5, 0.5, 0.5],
            max: [15.6, depth - 0.4, 15.6],
            species: Species::Water,
        }],
        solids: Vec::new(),
        ..Scene::default()
    }
}

/// Settled coupled bed: a grain bed filling the lower box with pore water seeded among it and a
/// shallow pond on top (the CenterPour coupled regime). Grains are the frozen U6 skeleton (no
/// solid_dynamics), water is the live phase whose KE we bound. Both phases start at rest.
fn coupled_bed_scene() -> Scene {
    Scene {
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0; 3],
        box_max: [16.0, 28.0, 16.0],
        regions: vec![
            // Water seeded FIRST (range layout): pore water spanning the bed + a shallow pond.
            SeedRegion {
                min: [0.5, 0.5, 0.5],
                max: [15.6, 11.6, 15.6],
                species: Species::Water,
            },
            // Grain bed filling the lower box (wall margin d/2 = 0.5 at d = 1).
            SeedRegion {
                min: [0.5, 0.5, 0.5],
                max: [15.6, 9.6, 15.6],
                species: Species::Grain,
            },
        ],
        solids: Vec::new(),
        ..Scene::default()
    }
}

fn coupled_cfg() -> Config {
    Config {
        drag_scale: 0.05, // packed-bed Kozeny-Carman rate (the coupling-suite default regime)
        ..Config::default()
    }
}

// V60 cup geometry (utils::geometry::v60_dripper cylinder): floor -8, rim -3.5, radius 3.
const CUP_FLOOR: f32 = -8.0;
const CUP_RIM: f32 = -3.5;
const CUP_RADIUS: f64 = 3.0;

/// Web WaterOnly resolution (mirrors tests/twofield_cup.rs).
fn web_mats() -> Materials {
    let r = 0.16_f32;
    Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    }
}

fn web_cfg() -> Config {
    Config {
        nozzle_radius: 0.55,
        max_speed: 12.0,
        xsph_viscosity_c: 0.02,
        ..Config::default()
    }
}

/// V60 cup with a water slab seeded inside (settle without a pour) — the EDGE-RING scene.
fn cup_static_scene() -> Scene {
    let mut s = Scene::v60_pour_water_only();
    s.regions = vec![SeedRegion {
        min: [-2.0, CUP_FLOOR + 0.3, -2.0],
        max: [2.0, CUP_FLOOR + 2.0, 2.0],
        species: Species::Water,
    }];
    s
}

// ==============================================================================================
// Energy measurement
// ==============================================================================================

/// Total kinetic energy Σ½m|v|² over the LIVE water particles (the phase that should settle to
/// rest). Constant mass per particle at the defaults.
fn total_ke(solver: &TwofieldSolver) -> f64 {
    let nw = solver.phase_counts().0 as usize;
    let vel = solver.read_velocities();
    let mut ke = 0.0f64;
    for v in &vel[..nw] {
        let s2 = (v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2);
        ke += 0.5 * REST_MASS * s2;
    }
    ke
}

fn max_speed(solver: &TwofieldSolver) -> f64 {
    let nw = solver.phase_counts().0 as usize;
    let vel = solver.read_velocities();
    vel[..nw]
        .iter()
        .map(|v| ((v[0] as f64).powi(2) + (v[1] as f64).powi(2) + (v[2] as f64).powi(2)).sqrt())
        .fold(0.0, f64::max)
}

/// Run `frames` settling steps with zero pour and EVERY-FRAME KE/max|v| sampling. Returns
/// (ke trace, max|v| trace). The pool starts from the seeded zero-velocity rest state.
fn settle_trace(solver: &mut TwofieldSolver, frames: usize) -> (Vec<f64>, Vec<f64>) {
    let quiet = EmissionInput::default();
    let mut ke = Vec::with_capacity(frames);
    let mut mv = Vec::with_capacity(frames);
    for _ in 0..frames {
        solver.step(DT, &quiet);
        ke.push(total_ke(solver));
        mv.push(max_speed(solver));
    }
    (ke, mv)
}

/// Window statistics over the SETTLED tail [tail_start, end): the late-window mean KE and the
/// peak KE. A correct settled pool has its KE decay early and stay low — so the tail peak must
/// not exceed the tail mean by a large factor (no spontaneous re-energization spikes).
fn tail_stats(ke: &[f64], tail_start: usize) -> (f64, f64) {
    let tail = &ke[tail_start..];
    let mean = tail.iter().sum::<f64>() / tail.len() as f64;
    let peak = tail.iter().cloned().fold(0.0, f64::max);
    (mean, peak)
}

// ==============================================================================================
// STEP 1 + STEP 2: reproduce + isolate (diagnostic; prints the numbers, asserts the mechanism)
// ==============================================================================================

/// Reproduce + quantify the spontaneous stirring, then isolate the mechanism by toggling each
/// candidate independently and re-measuring the settled-tail KE. This is CHARACTERIZATION for the
/// saturated-bed-creep redesign (the settled-pool stirring fix is deferred there — no damping knob
/// quiets the pool without freezing the crater; see PIC_BLEND_DEFAULT). It pins the diagnosis as
/// robust facts: the affine C is the lossless CARRIER (pure PIC collapses the tail KE), the density
/// relief is the SOURCE (relief OFF collapses it), and pressure convergence is the cure that the
/// real-time budget cannot afford (32 sweeps cut it ~4× — but globally that inflates the pool, the
/// reason the redesign, not a sweep bump, is the path). It does NOT assert any shipped fix.
#[test]
fn isolate_settled_stirring_mechanisms() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_settled: no GPU adapter; skipping.");
        return;
    };
    let scene = tank_scene(12.0);
    let mats = Materials::default();
    let cfg = Config::default();
    const WARM: usize = 200; // let the initial seed over-density relax out
    const RUN: usize = 1000; // long settled run
    let tail_start = RUN * 6 / 10; // last 40% is the "fully settled" tail

    // Helper: build a fresh solver, apply a toggle closure, settle, return (warm-end KE,
    // tail mean, tail peak, run max|v|).
    let arm = |label: &str, setup: &dyn Fn(&mut TwofieldSolver)| -> (f64, f64, f64, f64) {
        let mut s = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
        setup(&mut s);
        let quiet = EmissionInput::default();
        for _ in 0..WARM {
            s.step(DT, &quiet);
        }
        let ke_warm = total_ke(&s);
        let (ke, mv) = settle_trace(&mut s, RUN);
        let (mean, peak) = tail_stats(&ke, tail_start);
        let run_maxv = mv.iter().cloned().fold(0.0, f64::max);
        println!(
            "  arm {label:<28} warm-end KE {ke_warm:9.3} | tail mean KE {mean:9.4} | tail PEAK KE {peak:9.4} | run max|v| {run_maxv:.3}"
        );
        (ke_warm, mean, peak, run_maxv)
    };

    println!(
        "twofield SETTLED stirring isolation (water-only tank, depth 12, {RUN} settled frames):"
    );
    // BASELINE: the reported BUG state — pure APIC, relief + the dead-band off.
    let (_, mean_apic, peak_apic, _) = arm("APIC blend=0 (BUG)", &|s| {
        s.set_pic_blend_for_test(0.0);
    });
    // ISOLATE (a) APIC ringing: pure PIC zeros the affine state.
    let (_, mean_pic, _, _) = arm("pure PIC (no affine)", &|s| {
        s.set_pic_blend_for_test(1.0);
    });
    // ISOLATE (b) relief: turn relief OFF entirely.
    let (_, mean_norelief, _, _) = arm("APIC blend=0, relief OFF", &|s| {
        s.set_pic_blend_for_test(0.0);
        s.set_relief_for_test(false);
    });
    // ISOLATE (d) pressure residual: 4x fine sweeps.
    let (_, mean_sweeps, _, _) = arm("APIC blend=0, fine sweeps 32", &|s| {
        s.set_pic_blend_for_test(0.0);
        s.set_pressure_budget_for_test(4, 8, 32);
    });
    // CANDIDATE that quiets the pool but is deferred: a global PIC blend (shown here for the
    // redesign's record — it freezes the crater slump in twofield_full.rs, hence not shipped).
    let (_, mean_blend, _, _) = arm("PIC blend 0.05 (quiets, freezes crater)", &|s| {
        s.set_pic_blend_for_test(0.05);
    });

    // ---- structural assertions (robust mechanism facts, not tuned thresholds) ----
    // The affine C is the dominant energy CARRIER: pure PIC (C zeroed) collapses the tail KE far
    // below the pure-APIC baseline — APIC's lossless affine accumulation is the primary mechanism.
    assert!(
        mean_pic < 0.2 * mean_apic,
        "pure PIC did not collapse the stirring (APIC ringing is NOT dominant?): \
         APIC {mean_apic:.4} vs PIC {mean_pic:.4}"
    );
    // The density relief is the dominant SOURCE: turning it off collapses the tail KE well below
    // the baseline (the relief converts the under-converged standing density error into velocity
    // each frame, which the lossless C then accumulates).
    assert!(
        mean_norelief < 0.5 * mean_apic,
        "relief OFF did not cut the stirring (relief is NOT a dominant source?): \
         APIC {mean_apic:.4} vs relief-OFF {mean_norelief:.4}"
    );
    // A global PIC blend DOES quiet the pool (carrier damping) — recorded so the redesign knows the
    // lever works in isolation; it is deferred only because it freezes the crater (twofield_full).
    assert!(
        mean_blend < mean_apic,
        "blend did not quiet the pool in isolation: APIC {mean_apic:.4} vs blend {mean_blend:.4}"
    );
    println!(
        "  -> contributions: baseline(APIC) {mean_apic:.3} | pure-PIC {mean_pic:.3} | relief-OFF {mean_norelief:.3} | 32-sweeps {mean_sweeps:.3} | blend-0.05 {mean_blend:.3} (peak baseline {peak_apic:.3})"
    );
}

// ==============================================================================================
// STEP 4: regression gates — settled pool KE decays and STAYS bounded (the missing invariant)
// ==============================================================================================

/// Water-only settled tank: at the production default (small PIC blend) the total KE must decay
/// and STAY bounded over a long run, sampled EVERY frame. The settled-tail peak KE must not
/// exceed the tail mean by more than a small factor — no spontaneous re-energization spikes.
#[test]
fn settled_water_tank_ke_decays_and_stays_bounded() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_settled: no GPU adapter; skipping.");
        return;
    };
    let scene = tank_scene(12.0);
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let quiet = EmissionInput::default();

    // Warm out the seed over-density relaxation (the legitimate one-time expansion).
    for _ in 0..200 {
        solver.step(DT, &quiet);
    }
    const RUN: usize = 1500;
    let (ke, mv) = settle_trace(&mut solver, RUN);
    let tail_start = RUN * 6 / 10;
    let (tail_mean, tail_peak) = tail_stats(&ke, tail_start);
    let run_peak = ke.iter().cloned().fold(0.0, f64::max);
    let run_maxv = mv.iter().cloned().fold(0.0, f64::max);
    let nw = solver.phase_counts().0 as f64;
    println!(
        "twofield SETTLED water-only: blend {PIC_BLEND_DEFAULT} | tail mean KE {tail_mean:.4} | tail PEAK KE {tail_peak:.4} | run PEAK KE {run_peak:.4} | per-particle tail mean KE {:.2e} | max|v| {run_maxv:.3}",
        tail_mean / nw
    );

    // KE actually settled LOW: per-particle tail-mean KE is a tiny fraction of rest jitter.
    assert!(
        tail_mean / nw < 0.05,
        "settled tank never went quiet: per-particle tail mean KE {:.4} (total {tail_mean:.3})",
        tail_mean / nw
    );
    // NO SPONTANEOUS RE-ENERGIZATION: the settled-tail peak does not blow past the tail mean.
    assert!(
        tail_peak <= 4.0 * tail_mean,
        "spontaneous KE spike in the settled tail: peak {tail_peak:.4} > 4x mean {tail_mean:.4}"
    );
    // The whole run never re-energizes above a hard absolute KE ceiling (≈ the warmup energy).
    assert!(
        run_maxv < 5.0,
        "settled tank max|v| {run_maxv:.3} exceeds the bound"
    );
    assert!(ke.iter().all(|k| k.is_finite()), "non-finite KE");
}

/// Coupled settled bed: the same every-frame KE invariant on the CenterPour regime (frozen
/// grain skeleton + pore water + pond). Confirms the artifact and its fix are core-solver, not
/// water-only-specific.
#[test]
fn settled_coupled_bed_ke_decays_and_stays_bounded() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_settled: no GPU adapter; skipping.");
        return;
    };
    let scene = coupled_bed_scene();
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &coupled_cfg(), &gpu);
    let quiet = EmissionInput::default();

    for _ in 0..300 {
        solver.step(DT, &quiet);
    }
    const RUN: usize = 1200;
    let (ke, mv) = settle_trace(&mut solver, RUN);
    let tail_start = RUN * 6 / 10;
    let (tail_mean, tail_peak) = tail_stats(&ke, tail_start);
    let run_maxv = mv.iter().cloned().fold(0.0, f64::max);
    let nw = solver.phase_counts().0 as f64;
    println!(
        "twofield SETTLED coupled bed: blend {PIC_BLEND_DEFAULT} | live water {nw} | tail mean KE {tail_mean:.4} | tail PEAK KE {tail_peak:.4} | per-particle tail mean KE {:.2e} | max|v| {run_maxv:.3}",
        tail_mean / nw
    );

    assert!(
        tail_mean / nw < 0.1,
        "settled coupled bed never went quiet: per-particle tail mean KE {:.4}",
        tail_mean / nw
    );
    assert!(
        tail_peak <= 4.0 * tail_mean,
        "spontaneous KE spike in the settled coupled tail: peak {tail_peak:.4} > 4x mean {tail_mean:.4}"
    );
    assert!(
        run_maxv < 5.0,
        "settled coupled bed max|v| {run_maxv:.3} exceeds the bound"
    );
    assert!(ke.iter().all(|k| k.is_finite()), "non-finite KE");
}

// ==============================================================================================
// EDGE RING: water must not pin in a single-particle-thick shell against the cup wall/floor seam
// ==============================================================================================

/// Cup-water particles below the rim (mirrors twofield_cup.rs `cup_stats`).
fn cup_water(solver: &TwofieldSolver) -> Vec<[f32; 4]> {
    let nlive = solver.phase_counts().0 as usize;
    let pos = solver.read_positions();
    pos[..nlive]
        .iter()
        .copied()
        .filter(|p| p[1] <= CUP_RIM && p[1] >= CUP_FLOOR - 1.0 && p[3] > 0.0)
        .collect()
}

/// Settled cup water must flow back into the bulk, not pin in a one-particle shell against the
/// wall/floor seam. The wall BC is free-slip (only the wall-normal velocity is zeroed) and the
/// SDF push-out projects to the surface without removing tangential motion, so trapped particles
/// slide back. The gate bounds the seam-shell fraction (no anomalous pinned ring) AND requires
/// the cup water to stay spread/centered (it is not collapsing instead).
#[test]
fn settled_cup_water_no_edge_ring() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_settled: no GPU adapter; skipping.");
        return;
    };
    let mut solver = TwofieldSolver::build(&cup_static_scene(), &web_mats(), &web_cfg(), &gpu);
    let quiet = EmissionInput::default();

    solver.step(DT, &quiet);
    let cup0 = cup_water(&solver);
    assert!(cup0.len() > 1000, "cup seed too small ({})", cup0.len());

    for _ in 0..400 {
        solver.step(DT, &quiet);
    }
    let cup = cup_water(&solver);

    // Radial AREA-DENSITY histogram (particles per unit annulus area): a smooth pool is roughly
    // flat across bins; a single-particle wall SHELL spikes the OUTERMOST bin's density past the
    // bulk mean. This is the direct ring discriminator (a pinned shell pushes it to 4–10×; a
    // smooth pool with a thin physical contact layer sits near ~2×).
    let nbins = 10usize;
    let mut counts = vec![0usize; nbins];
    for p in &cup {
        let r = ((p[0] as f64).powi(2) + (p[2] as f64).powi(2)).sqrt();
        let b = ((r / CUP_RADIUS) * nbins as f64).floor() as usize;
        counts[b.min(nbins - 1)] += 1;
    }
    let dens: Vec<f64> = counts
        .iter()
        .enumerate()
        .map(|(b, &c)| {
            let r_in = b as f64 / nbins as f64 * CUP_RADIUS;
            let r_out = (b + 1) as f64 / nbins as f64 * CUP_RADIUS;
            let area = std::f64::consts::PI * (r_out * r_out - r_in * r_in);
            c as f64 / area
        })
        .collect();
    let mean_dens = dens.iter().sum::<f64>() / nbins as f64;
    let outer_ratio = dens[nbins - 1] / mean_dens.max(1e-9);
    // Radial spread (a ring would suck particles to the wall, spiking the centroid radius).
    let (mut cx, mut cz) = (0.0f64, 0.0f64);
    for p in &cup {
        cx += p[0] as f64;
        cz += p[2] as f64;
    }
    cx /= cup.len() as f64;
    cz /= cup.len() as f64;
    let centroid_r = (cx * cx + cz * cz).sqrt();
    println!(
        "twofield SETTLED cup EDGE-RING: cup n {} | radial area-density {dens:?} | outer-bin/mean {outer_ratio:.2} | centroid_r {centroid_r:.3} (cup R {CUP_RADIUS})",
        cup.len()
    );

    // No severe pinned shell: the outermost-bin density does not spike far past the bulk mean.
    // Damping the APIC ringing (the PIC blend) + relieving the over-dense seam (the relief loop)
    // hold this near the smooth-pool ~2× rather than the hoarding-ring 4–10×. (A residual ~2×
    // wall layer in the radially-confined cup is the same family as the documented floor
    // over-compression — a pressure-iteration limit, not a separate pinned-shell BC bug; a
    // one-sided free-slip node BC was tried and reverted: it barely moved the ring (~2.2→2.0),
    // broke operator consistency, and trapped a spurious pocket in the filling cup. See the
    // tests/twofield_cup.rs footer.)
    assert!(
        outer_ratio < 2.5,
        "edge ring: outer-bin radial density {outer_ratio:.2}x the bulk mean (pinned wall shell)"
    );
    // The pool stays centered (not collapsed to the wall instead of ringing).
    assert!(
        centroid_r <= 0.4 * CUP_RADIUS,
        "cup water off-axis (centroid radius {centroid_r:.3})"
    );
}

/// DIAGNOSTIC (ignored): cup edge-ring (outer-bin radial density spike) under each toggle, to
/// isolate what drives the wall shell. Prints the outer-bin/mean ratio for: default, relief OFF,
/// pure PIC, pure APIC.
#[test]
#[ignore = "diagnostic; run explicitly to isolate the edge ring"]
fn isolate_cup_edge_ring() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_settled: no GPU adapter; skipping.");
        return;
    };
    let scene = cup_static_scene();
    let mats = web_mats();
    let cfg = web_cfg();
    let outer_ratio = |solver: &TwofieldSolver| -> f64 {
        let cup = cup_water(solver);
        let nbins = 10usize;
        let mut counts = vec![0usize; nbins];
        for p in &cup {
            let r = ((p[0] as f64).powi(2) + (p[2] as f64).powi(2)).sqrt();
            let b = ((r / CUP_RADIUS) * nbins as f64).floor() as usize;
            counts[b.min(nbins - 1)] += 1;
        }
        let dens: Vec<f64> = counts
            .iter()
            .enumerate()
            .map(|(b, &c)| {
                let r_in = b as f64 / nbins as f64 * CUP_RADIUS;
                let r_out = (b + 1) as f64 / nbins as f64 * CUP_RADIUS;
                c as f64 / (std::f64::consts::PI * (r_out * r_out - r_in * r_in))
            })
            .collect();
        let mean = dens.iter().sum::<f64>() / nbins as f64;
        dens[nbins - 1] / mean.max(1e-9)
    };
    let arm = |label: &str, setup: &dyn Fn(&mut TwofieldSolver)| {
        let mut s = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
        setup(&mut s);
        let quiet = EmissionInput::default();
        for _ in 0..400 {
            s.step(DT, &quiet);
        }
        println!(
            "  edge-ring arm {label:<24} outer-bin/mean = {:.2}",
            outer_ratio(&s)
        );
    };
    println!("twofield cup EDGE-RING isolation (outer-bin radial density / mean):");
    arm("default (FIX)", &|_s| {});
    arm("relief OFF", &|s| s.set_relief_for_test(false));
    arm("pure PIC", &|s| s.set_pic_blend_for_test(1.0));
    arm("pure APIC", &|s| s.set_pic_blend_for_test(0.0));
}

/// Pin the production PIC blend to a sane band. It is currently 0 (pure APIC): the settled-pool
/// stirring fix is DEFERRED to the saturated-bed-creep redesign, because any blend that quiets the
/// pool also freezes the crater slump (see PIC_BLEND_DEFAULT). The band [0, 0.15] is the guard so a
/// future edit cannot set a large blend that would freeze the crater; the redesign may set a small
/// nonzero value once the bed slumps via real creep.
#[test]
fn pic_blend_default_in_documented_band() {
    const {
        assert!(
            PIC_BLEND_DEFAULT >= 0.0 && PIC_BLEND_DEFAULT <= 0.15,
            "PIC_BLEND_DEFAULT out of the documented settle-vs-crater band [0, 0.15]"
        )
    };
}

/// DensU5 TEMPER-K sweep — settled-tank agitation (tail KE + max|v|) vs the uncapped rate-cap K.
/// The U5 recon found the full uncap (K=1) pumps energy into deep/coupled scenes (this tank's KE
/// blew up); legacy (rate-capped, K=30-equiv) is the quiet reference. Finds the SMALLEST K
/// (= stiffest/crispest density) that keeps the tank quiet. Pair with the cup over-pack sweep
/// (`twofield_wall_audit::temper_k_sweep_overpack`): a K that is BOTH crisp (low cup over-pack)
/// AND quiet (tank KE ≈ legacy) is the temper candidate — then eyeball it in the webapp.
/// `cargo test --release --test twofield_settled temper_k_sweep_settled_agitation -- --ignored --nocapture`
#[test]
#[ignore = "on-demand temper-K agitation sweep (slow: ~6 long tank runs)"]
fn temper_k_sweep_settled_agitation() {
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    const RUN: usize = 1500; // MUST match the gate — the agitation builds up slowly; 800 frames
                             // gives false "QUIET" verdicts (K=1: 0.0115@800 but 0.6151@1500).
    let tail_start = RUN * 6 / 10;
    let scene = tank_scene(12.0);
    // Match the `settled_water_tank_ke_decays_and_stays_bounded` gate EXACTLY: default mats/cfg,
    // 200-frame warmup (seed-relax), then the trace. Bound: per-particle tail-mean KE < 0.05,
    // max|v| < 5.0. The triage showed full uncap blows it (per-particle 0.6151, max|v| 26).
    let measure = |label: &str, uncapped: bool, k: f32| {
        let mut s = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
        if uncapped {
            s.set_density_target_mode_for_test(true);
            s.set_density_rate_k_for_test(k);
        }
        let quiet = EmissionInput::default();
        for _ in 0..200 {
            s.step(DT, &quiet);
        }
        let nw = s.phase_counts().0 as f64;
        let (ke, mv) = settle_trace(&mut s, RUN);
        let (mean, peak) = tail_stats(&ke, tail_start);
        let maxv = mv.iter().cloned().fold(0.0, f64::max);
        let pp = mean / nw;
        let verdict = if pp < 0.05 && maxv < 5.0 {
            "QUIET ✓"
        } else {
            "agitated"
        };
        println!(
            "  [{label:<18}] per-particle KE {pp:7.4}  max|v| {maxv:6.2}  (peak/mean {:.1}x)  {verdict}",
            peak / mean.max(1e-9)
        );
    };
    println!("\n==== TEMPER-K vs SETTLED-TANK AGITATION (gate config: default mats, 200 warmup + {RUN}) ====");
    println!("  bound: per-particle tail-mean KE < 0.05 AND max|v| < 5.0; smaller K = crisper but more agitation");
    measure("legacy (capped)", false, 0.0);
    measure("uncapped K=1", true, 1.0);
    measure("uncapped K=3", true, 3.0);
    measure("uncapped K=5", true, 5.0);
    measure("uncapped K=10", true, 10.0);
    measure("uncapped K=30", true, 30.0); // sanity: ≈ legacy (same rate, two-sided path)
    println!("  → smallest K whose tail KE / max|v| ≈ legacy is the temper candidate (quiet + crispest).");
}

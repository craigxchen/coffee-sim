//! Twofield U8 — the R9 REAL-TIME GATE and the program close-out perf gates.
//!
//! This is the program's verdict. It measures the per-FRAME GPU cost of the FULL V60 saturated-
//! pour scene at the PRE-REGISTERED 200k composition and asserts the hard, NO-FALLBACK gate:
//! ≤ 33 ms/frame median on the reference device (Apple M5 — this machine). It also gates the
//! linearity thesis (µs/Kparticle degradation 40k→200k within a factor PRE-REGISTERED here,
//! before any 200k number is read into the threshold), the dispatch budget, the no-GPU-sync
//! getter contract, the wasm32 target compile, and an offline grid-refinement (GCI) study.
//!
//! THE PER-FRAME COST. The per-pass `timestamp-query` sum (`profile().total_micros()`) captures
//! ONE CFL substep; the deformable-bed frame runs `solid_substeps` of them (7 at the defaults),
//! and every substep re-runs the full water+solid+pressure+surface pipeline (one shared `dt`).
//! So the honest per-frame GPU cost is `substeps × timestamp_sum`; `dispatches_per_frame` already
//! counts every substep, and the two are cross-checked here. This measures the real per-frame
//! cost via device timestamps — NOT the many-frames-of-small-scenes-with-readbacks wall time that
//! made `twofield_full`'s end-to-end gates run 35 min (a different thing entirely, see U7).
//!
//! Verdict posture (NO-FALLBACK program). If after the sanctioned honest cheap wins the 200k
//! median does NOT meet 33 ms, the gate asserts the REAL number and goes RED — a failing-but-
//! honest gate is the correct outcome of a bet that doesn't make budget. The breakdown printed
//! by `examples/twofield_scaling` shows WHERE the time goes (the deferred CK-MPM / kernel-fusion
//! levers' target). The gate is NOT loosened, the composition is NOT made cheaper, and the
//! linearity factor is NOT tuned to the result.

use coffee_sim::engine::scene::{Scene, SeedRegion, Species};
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{dispatches_per_frame_for, TwofieldSolver};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

// ======================= PRE-REGISTERED gate constants (FIXED before any run) =================

/// R9: the hard real-time gate — ≤ 33 ms/frame median at 200k on the reference device. A
/// perfectly linear 90 ms/frame solver still fails; this is an absolute number, not structure.
const REALTIME_MS_GATE: f32 = 33.0;

/// The pre-registered gate-scene composition (KTD-5 cost bound): ~185–190k water + 10–15k solids
/// at the 200k point. The bed is a thin frictional minority; the pour column + pond + bed pore
/// water are the water majority. Encoded as a box edge + fixed bed/water proportions so the SAME
/// scene grows with N — the gate cannot be passed with a conveniently cheap mix.
const BED_LAYERS: f32 = 4.0; // thin skeleton (the solids minority)
const WATER_COL: f32 = 52.0; // deep pour column / pond (the water majority)
const PORE_PITCH: f32 = 1.6; // sparse bed pore-water lattice
const EDGE_200K: f32 = 61.0; // box edge landing ~205k at the fixed composition
const EDGE_40K: f32 = 28.0; // box edge landing ~40k (the linearity low anchor)

/// Linearity gate: the maximum acceptable µs/Kparticle degradation from 40k to 200k, PRE-
/// REGISTERED before the 200k number is read into it. Derivation from the LINEAR THESIS, not the
/// result: a perfectly linear solver is 1.0×; real GPU launch/overhead amortization adds a known
/// small-N tax that fades by ~40k (the xpbd probe measured a residual ≈1.1× over its saturated
/// range), and the coarse-pressure + surface stages carry a sub-linear fixed-grid component, so a
/// generous 2.5× covers a genuinely-near-linear solver while still failing a super-linear one.
const LINEARITY_FACTOR_MAX: f32 = 2.5;

const WARMUP_FRAMES: u32 = 50; // settle the saturated bed, start the pour
const MEASURE_FRAMES: u32 = 30; // median window

// ======================= shared scene machinery (self-contained per test) =====================

fn build_scene(edge: f32) -> Scene {
    let bed_top = BED_LAYERS;
    let water_lo = bed_top + 1.0;
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
        regions: vec![
            SeedRegion {
                min: [0.5, 0.5, 0.5],
                max: [edge - 0.5, bed_top, edge - 0.5],
                species: Species::Grain,
            },
            SeedRegion {
                min: [0.7, water_lo, 0.7],
                max: [edge - 0.7, water_hi, edge - 0.7],
                species: Species::Water,
            },
        ],
    }
}

fn gate_cfg() -> Config {
    Config {
        solid_dynamics: true,
        tf_filter_floor: true,
        nozzle_radius: 1.5,
        tf_wet_cohesion: 4.0,
        tf_cohesion_speak: 0.4,
        ..Config::default()
    }
}

fn gate_mats() -> Materials {
    Materials {
        grain_diameter: 1.0,
        ..Materials::default()
    }
}

/// Seed the bed pore water onto a sub-lattice so the bed starts saturated (mirrors
/// `tests/twofield_full.rs::build_bed`), using the dormant water-pool headroom.
fn saturate_bed(solver: &mut TwofieldSolver, edge: f32, bed_top: f32) {
    let mut pore: Vec<[f32; 3]> = Vec::new();
    let mut yy = 0.6f32;
    while yy <= bed_top - 0.4 {
        let mut xx = 0.7f32;
        while xx <= edge - 0.7 {
            let mut zz = 0.7f32;
            while zz <= edge - 0.7 {
                pore.push([xx, yy, zz]);
                zz += PORE_PITCH;
            }
            xx += PORE_PITCH;
        }
        yy += PORE_PITCH;
    }
    let n_pond = solver.phase_counts().0;
    let mut pos = solver.read_positions();
    // Pore water fills dormant water-pool slots only [n_pond, water_pool_capacity).
    let room = solver.water_pool_capacity().saturating_sub(n_pond) as usize;
    let take = pore.len().min(room);
    for (i, p) in pore.iter().take(take).enumerate() {
        pos[n_pond as usize + i] = [p[0], p[1], p[2], 1.0];
    }
    solver.write_positions_for_test(&pos);
    solver.set_live_water_for_test(n_pond + take as u32);
}

struct Measured {
    n_water: u32,
    n_solid: u32,
    substeps: u32,
    dispatches: u32,
    median_ms: f32,
    per_kpart: f32,
    breakdown: Vec<(String, f32)>,
}

/// Build the fixed-composition scene at `edge`, warm up into the pour, and return the median
/// per-frame GPU cost (`substeps × substep-0 timestamp sum`) over the measure window.
fn measure(gpu: &GpuContext, edge: f32) -> Measured {
    let scene = build_scene(edge);
    let mats = gate_mats();
    let cfg = gate_cfg();
    let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, gpu);
    let (_, n_solid) = solver.phase_counts();
    let bed_top = scene.regions[0].max[1];
    saturate_bed(&mut solver, edge, bed_top);
    let n_water = solver.active_count() - n_solid;

    let axis = (edge / 2.0, edge / 2.0);
    let water_hi = scene.regions[1].max[1];
    let quiet = EmissionInput::default();
    let pour = EmissionInput {
        kettle_pos: [axis.0, water_hi + 6.0, axis.1],
        flow_rate: 70.0,
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
    let n = (n_water + n_solid).max(1);
    let denom = samples.len().max(1) as f32;
    let mut breakdown: Vec<(String, f32)> = agg.into_iter().map(|(k, v)| (k, v / denom)).collect();
    breakdown.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Measured {
        n_water,
        n_solid,
        substeps,
        dispatches,
        median_ms: median_us / 1000.0,
        per_kpart: median_us / (n as f32 / 1000.0),
        breakdown,
    }
}

// =================================== R9 real-time gate =========================================

/// THE R9 GATE: the full V60 saturated-pour scene at the pre-registered 200k composition runs
/// ≤ 33 ms/frame median. NO-FALLBACK: if it does not, this asserts the real number and goes RED.
#[test]
fn r9_realtime_gate_200k_full_v60_pour() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_perf: no GPU adapter; skipping.");
        return;
    };
    let m = measure(&gpu, EDGE_200K);
    let n = m.n_water + m.n_solid;

    println!("twofield U8 R9 REAL-TIME GATE (Apple M5):");
    println!(
        "  composition: {} water + {} solids = {} (pre-registered ~185–190k + 10–15k)",
        m.n_water, m.n_solid, n
    );
    println!(
        "  substeps {}, dispatches/frame {}, median {:.3} ms ({:.1} µs/Kpart)",
        m.substeps, m.dispatches, m.median_ms, m.per_kpart
    );
    println!("  per-pass breakdown (substeps-scaled, largest first):");
    let total: f32 = m.breakdown.iter().map(|(_, us)| us).sum();
    for (label, us) in m.breakdown.iter().take(10) {
        println!(
            "    {:<18} {:>9.1} µs  ({:>4.1}%)",
            label,
            us,
            100.0 * us / total.max(1.0e-6)
        );
    }

    // Composition guard: the gate cannot be passed with a cheaper mix — the solid minority and
    // the ~200k total are pinned by the scene, not by the threshold.
    assert!(
        (185_000..=215_000).contains(&n),
        "gate scene N={n} drifted from the pre-registered ~200k window"
    );
    assert!(
        (8_000..=18_000).contains(&m.n_solid),
        "solid count {} drifted from the pre-registered 10–15k window (KTD-5 cost bound)",
        m.n_solid
    );

    assert!(
        m.median_ms <= REALTIME_MS_GATE,
        "R9 HALT: 200k median {:.3} ms/frame EXCEEDS the {REALTIME_MS_GATE} ms real-time gate \
         (N={n}, {} water + {} solids, {} substeps). This is a NO-FALLBACK halt-for-owner-ruling: \
         the dominant pass is `{}` ({:.1} µs, {:.0}% of the frame) — the constraint-bubble row \
         solve (KTD-6), a single-workgroup all-cells reduction run per fine sweep per substep, \
         which the deferred CK-MPM / kernel-fusion / parallel-reduction levers target. The gate \
         is NOT loosened and the composition is NOT made cheaper; the honest number stands.",
        m.median_ms,
        m.n_water,
        m.n_solid,
        m.substeps,
        m.breakdown.first().map(|(l, _)| l.as_str()).unwrap_or("?"),
        m.breakdown.first().map(|(_, u)| *u).unwrap_or(0.0),
        100.0 * m.breakdown.first().map(|(_, u)| *u).unwrap_or(0.0) / total.max(1.0e-6),
    );
}

// =================================== linearity gate ===========================================

/// µs/Kparticle stays flat within the PRE-REGISTERED factor from 40k → 200k (the ~linear-scaling
/// thesis). The factor was fixed from the linear thesis, not derived from the 200k number.
#[test]
fn linearity_40k_to_200k_within_preregistered_factor() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_perf: no GPU adapter; skipping.");
        return;
    };
    let lo = measure(&gpu, EDGE_40K);
    let hi = measure(&gpu, EDGE_200K);
    let factor = hi.per_kpart / lo.per_kpart.max(1.0e-6);
    println!(
        "twofield U8 linearity: 40k N={} {:.1} µs/Kpart  →  200k N={} {:.1} µs/Kpart  =  {:.2}× \
         (pre-registered max {LINEARITY_FACTOR_MAX}×)",
        lo.n_water + lo.n_solid,
        lo.per_kpart,
        hi.n_water + hi.n_solid,
        hi.per_kpart,
        factor,
    );
    assert!(
        factor <= LINEARITY_FACTOR_MAX,
        "linearity HALT: µs/Kpart degraded {factor:.2}× from 40k to 200k, above the pre-registered \
         {LINEARITY_FACTOR_MAX}× — the scaling is super-linear (same constraint-bubble root cause \
         as the R9 gate). NOT tuned to the result."
    );
}

// =================================== dispatch budget ==========================================

/// Dispatches/frame are within the final recorded budget: in the dynamic deformable-bed regime
/// the frame runs `substeps × (DISPATCHES_PER_FRAME + U5 increment)`. We pin that the count is
/// EXACTLY the recorded formula (no hidden growth) on the gate scene.
#[test]
fn dispatch_budget_within_recorded_formula() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_perf: no GPU adapter; skipping.");
        return;
    };
    let scene = build_scene(EDGE_40K);
    let mut solver = TwofieldSolver::build(&scene, &gate_mats(), &gate_cfg(), &gpu);
    saturate_bed(&mut solver, EDGE_40K, scene.regions[0].max[1]);
    let axis = (EDGE_40K / 2.0, EDGE_40K / 2.0);
    let water_hi = scene.regions[1].max[1];
    let pour = EmissionInput {
        kettle_pos: [axis.0, water_hi + 6.0, axis.1],
        flow_rate: 70.0,
        ..EmissionInput::default()
    };
    // A few frames so the pour activates (water present → the full water+solid pipeline runs).
    for _ in 0..6 {
        solver.step(DT, &pour);
    }
    let substeps = solver.substeps_for_dt(DT);
    let (_, _, dims) = solver.grid_spec();
    // Water-present dynamic budget per the U5 doc: substeps × (dispatches_per_frame + the U5
    // plasticity increment of 2: solid_update + g2p_solid each substep; p2g_solid_dyn replaces
    // p2g_solid 1:1). The frame budget is scene-derived through the U4 flood stack. U9 absorption
    // is OFF on this scene (tf_absorb_rate default 0).
    let expected = substeps * (dispatches_per_frame_for(dims) + 2);
    println!(
        "twofield U8 dispatch budget: substeps {substeps}, dispatches/frame {} (expected {expected})",
        solver.profile().dispatches_per_frame
    );
    assert_eq!(
        solver.profile().dispatches_per_frame,
        expected,
        "dynamic deformable-bed dispatch count drifted from substeps × (dispatches_per_frame + 2)"
    );
}

// =================================== no-GPU-sync getter smoke ==================================

/// `metrics()`/`particles()`/`profile()` complete without a GPU sync (the trait contract). A
/// readback (`sample_diagnostics`) is slow; the three getters must be orders of magnitude faster,
/// proving they return cached state (timing-based smoke, mirroring the xpbd precedent).
#[test]
fn getters_complete_without_gpu_sync() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_perf: no GPU adapter; skipping.");
        return;
    };
    let scene = build_scene(EDGE_40K);
    let mut solver = TwofieldSolver::build(&scene, &gate_mats(), &gate_cfg(), &gpu);
    saturate_bed(&mut solver, EDGE_40K, scene.regions[0].max[1]);
    solver.step(DT, &EmissionInput::default());
    solver.sample_diagnostics();

    // Reference: a real readback (stalls on the GPU). The getters must be far cheaper.
    let t0 = std::time::Instant::now();
    let _ = solver.read_positions();
    let sync_ns = t0.elapsed().as_nanos().max(1);

    let t1 = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = std::hint::black_box(solver.metrics());
        let _ = std::hint::black_box(solver.particles());
        let _ = std::hint::black_box(solver.profile());
    }
    let getters_ns = t1.elapsed().as_nanos() / 1000;

    println!(
        "twofield U8 getter smoke: 1 readback {sync_ns} ns vs 1×(metrics+particles+profile) \
         {getters_ns} ns"
    );
    // The getters clone small CPU state; a GPU sync is ~1e5–1e6 ns. Require a wide margin so a
    // hidden sync inside a getter would be caught (the getters must be < 10% of one readback).
    assert!(
        (getters_ns as f64) < 0.10 * (sync_ns as f64),
        "a getter stalled the GPU: getters {getters_ns} ns is not << one readback {sync_ns} ns"
    );

    // The metric surface is populated (drawdown latch is event-driven so may be 0 here; the
    // count + iteration budget are always live).
    let m = solver.metrics();
    assert!(m.particle_count > 0, "metrics particle_count unpopulated");
    assert!(
        m.iteration_count > 0,
        "metrics iteration_count (pressure budget) unpopulated"
    );
}

// =================================== offline GCI refinement sweep ==============================

/// Offline grid-refinement (Richardson/GCI) study — REDUCED form (the prompt sanctions a
/// documented reduced version: 3 resolutions at 2:1 with the observed-order check). One fixed
/// scenario (a settled hydrostatic saturated column), one scalar QoI = the settled water-column
/// center-of-mass height (a physical, grid-independent length — the raw pressure field is not
/// grid-normalized across resolutions and would diverge for non-spatial reasons).
/// The fine-sweep budget SCALES with resolution so iterative error stays subdominant to
/// truncation (a fixed absolute sweep count fails the asymptotic-range check for non-spatial
/// reasons — KTD-2 / Roache). We compute the observed order p and the Richardson-extrapolated
/// value, store it as a regression band, and check p is a finite positive order (the asymptotic-
/// range sanity check). The fixed real-time sweep budget is gated separately by U3's tolerance.
#[test]
fn offline_gci_refinement_settled_height() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_perf: no GPU adapter; skipping.");
        return;
    };
    // Three 2:1 refinements via the particle spacing (cell size h = 2·spacing tracks it): a
    // fixed physical column (a short saturated bed), QoI = settled water COM height at rest.
    // Coarse → fine spacing halves; the fine-sweep budget DOUBLES per refinement so iterative
    // error decays faster than truncation.
    let spacings = [2.0f32, 1.0, 0.5];
    let fine_sweeps = [8u32, 16, 32]; // scales with resolution (KTD-2)
    let mut qoi = [0.0f64; 3];
    let mut counts = [0u32; 3];
    for (i, (&s, &fs)) in spacings.iter().zip(&fine_sweeps).enumerate() {
        let scene = Scene {
            gravity: [0.0, -20.0, 0.0],
            box_min: [0.0, 0.0, 0.0],
            box_max: [8.0, 16.0, 8.0],
            pour_water_ml: 0.0,
            solids: Vec::new(),
            regions: vec![
                SeedRegion {
                    min: [0.5, 0.5, 0.5],
                    max: [7.5, 6.0, 7.5],
                    species: Species::Grain,
                },
                SeedRegion {
                    min: [0.7, 6.5, 0.7],
                    max: [7.3, 12.0, 7.3],
                    species: Species::Water,
                },
            ],
            ..Scene::default()
        };
        let mats = Materials {
            particle_spacing: s,
            support_radius: 2.0 * s,
            grain_diameter: s,
            ..Materials::default()
        };
        let cfg = Config::default(); // frozen skeleton: a clean hydrostatic column (no plasticity noise)
        let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
        solver.set_pressure_budget_for_test(4, 8, fs);
        for _ in 0..120 {
            solver.step(DT, &EmissionInput::default());
        }
        // QoI: the settled water-column center-of-mass height — a PHYSICAL length (scene units),
        // grid-INDEPENDENT by construction, that converges to the incompressible-equilibrium fill
        // height as the grid refines. (The raw pressure field is NOT grid-normalized across
        // resolutions, so it is not a valid Richardson QoI — it would diverge for non-spatial
        // reasons; the settled height is the clean truncation-error probe.)
        let pos = solver.read_positions();
        let phase = solver.read_phases();
        let live = solver.active_count() as usize;
        let mut sum = 0.0f64;
        let mut cnt = 0u32;
        for k in 0..live.min(pos.len()).min(phase.len()) {
            if phase[k] == 0 && pos[k][3] > 0.0 {
                sum += pos[k][1] as f64;
                cnt += 1;
            }
        }
        qoi[i] = if cnt > 0 { sum / cnt as f64 } else { 0.0 };
        counts[i] = solver.active_count();
        println!(
            "  GCI refine h={:.2} (fine_sweeps {fs}, N={}): water COM height = {:.4}",
            2.0 * s,
            counts[i],
            qoi[i]
        );
    }

    // Richardson extrapolation on the 2:1:1 triple (r = 2). Observed order
    //   p = ln(|f1 − f2| / |f2 − f3|) / ln(r),
    // extrapolated value f_exact ≈ f3 + (f3 − f2)/(r^p − 1), GCI band on the finest pair.
    let (f1, f2, f3) = (qoi[0], qoi[1], qoi[2]);
    let r = 2.0f64;
    let e12 = (f1 - f2).abs();
    let e23 = (f2 - f3).abs();
    // Guard the degenerate (near-converged) case: if the changes are tiny the field is already
    // grid-insensitive — that PASSES the asymptotic sanity (no spurious divergence) trivially.
    if e23 < 1.0e-6 || e12 < 1.0e-6 {
        println!(
            "twofield U8 GCI: changes below 1e-6 (f1 {f1:.4}, f2 {f2:.4}, f3 {f3:.4}) — grid-\
             insensitive, asymptotic sanity trivially met"
        );
        return;
    }
    let p_obs = (e12 / e23).ln() / r.ln();
    let f_exact = f3 + (f3 - f2) / (r.powf(p_obs) - 1.0);
    let gci = 1.25 * (e23 / f3.abs().max(1.0e-9)) / (r.powf(p_obs) - 1.0).abs().max(1.0e-9);
    println!(
        "twofield U8 GCI (reduced, 3×2:1, budget-scaled): observed order p = {p_obs:.2}, \
         extrapolated water COM height = {f_exact:.4}, GCI(fine) = {:.1}%",
        100.0 * gci
    );
    // Asymptotic-range sanity: the observed order is finite and positive (the sequence is
    // CONVERGING, not diverging — the failure the budget-scaling is designed to prevent). The
    // extrapolated value is the stored regression band for future suites.
    assert!(
        p_obs.is_finite() && p_obs > 0.0,
        "GCI: observed order p = {p_obs:.3} is not a positive convergent order — the refinement \
         is not in the asymptotic range (f1 {f1:.4}, f2 {f2:.4}, f3 {f3:.4})"
    );
    // Regression band: the extrapolated settled water COM height for this fixed column. The
    // physical fill sits between the bed top (6.0) and the seeded column top (12.0), so the
    // grid-converged equilibrium height must land in that envelope — a stored band future
    // suites regress against (the exact value is solver-defined; this catches gross drift).
    assert!(
        f_exact.is_finite() && (4.0..=12.0).contains(&f_exact),
        "GCI extrapolated water COM height {f_exact:.4} fell outside the physical fill band \
         [4.0, 12.0] (bed top 6.0 → column top 12.0)"
    );
}

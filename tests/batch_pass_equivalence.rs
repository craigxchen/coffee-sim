//! Equivalence gate for the single-compute-pass dispatch batching (the browser CPU-overhead
//! refactor). Both GPU solvers normally open one `begin_compute_pass`/`end` pair PER dispatch;
//! the refactor batches *consecutive* dispatches into ONE pass when per-pass timestamps are off
//! (the web / no-TIMESTAMP_QUERY path). WebGPU/Dawn inserts the storage-buffer read-after-write
//! barriers between dependent dispatches in the same pass automatically, so the batched frame
//! must produce the same result.
//!
//! Two regimes, two gate strengths:
//!
//! * **twofield is DETERMINISTIC** — its grid atomics are fixed-point integers (commutative, so
//!   the accumulation order does not matter). The batched run must be BIT-IDENTICAL to the
//!   per-pass run.
//!
//! * **xpbd is NONDETERMINISTIC** — its grid/density/coupling passes accumulate via *float*
//!   atomics, whose result depends on GPU scheduling order, so two identical per-pass runs from
//!   the same seed already differ by a per-particle noise band (measured here). The batching
//!   cannot be graded per-particle. Instead we assert that the batched-vs-per-pass deviation of a
//!   *stable aggregate* (the particle centroid, which averages out the per-particle atomic
//!   jitter) stays WITHIN that intrinsic per-pass-vs-per-pass noise band — i.e. batching adds no
//!   systematic deviation beyond xpbd's own run-to-run float-atomic noise.
//!
//! A real intra-pass barrier failure would push the batched arm cleanly outside the noise band
//! (or break twofield's bit-identity). GPU-gated: skips cleanly without an adapter.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

fn bit_identical(a: &[[f32; 4]], b: &[[f32; 4]]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(ra, rb)| (0..4).all(|c| ra[c].to_bits() == rb[c].to_bits()))
}

fn max_abs_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    let mut m = 0.0f32;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for c in 0..4 {
            let d = (ra[c] - rb[c]).abs();
            m = if d.is_nan() { f32::INFINITY } else { m.max(d) };
        }
    }
    m
}

/// Centroid of the xyz positions — a stable aggregate that averages out per-particle float-atomic
/// jitter, so batched-vs-per-pass shifts of the *bulk* are visible without per-particle noise.
fn centroid(a: &[[f32; 4]]) -> [f64; 3] {
    let mut s = [0.0f64; 3];
    for p in a {
        for c in 0..3 {
            s[c] += p[c] as f64;
        }
    }
    let n = a.len().max(1) as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

fn centroid_dist(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    let (x, y) = (centroid(a), centroid(b));
    ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
}

/// twofield: the center-pour cavity scene (full pipeline — P2G, the U3 pressure stack, the U4
/// cavity flood + bubble, coupling BCs, G2P). Settled pool, then a vigorous jet, then quiet.
/// twofield is deterministic, so the batched frame must be BIT-IDENTICAL.
#[test]
fn twofield_batched_is_bit_identical_to_per_pass() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("batch_pass_equivalence (twofield): no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0; 3],
        box_max: [24.0, 28.0, 24.0],
        regions: vec![SeedRegion {
            min: [0.5, 0.5, 0.5],
            max: [23.6, 11.6, 23.6],
            species: Species::Water,
        }],
        solids: Vec::new(),
        pour_water_ml: 1300.0,
        ..Scene::default()
    };
    let mats = Materials::default();
    let cfg = Config {
        nozzle_radius: 1.5,
        ..Config::default()
    };
    let pour = EmissionInput {
        kettle_pos: [12.0, 18.0, 12.0],
        flow_rate: 80.0,
        pour_angle: 0.0,
        ..EmissionInput::default()
    };
    let quiet = EmissionInput::default();
    let schedule: Vec<(EmissionInput, u32)> = vec![(quiet, 10), (pour, 40), (quiet, 30)];

    let run = |batched: bool| -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
        let mut s = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
        if batched {
            s.force_batched_passes_for_test();
        }
        for (input, n) in &schedule {
            for _ in 0..*n {
                s.step(DT, input);
            }
        }
        (s.read_positions(), s.read_velocities())
    };

    let (pos_ref, vel_ref) = run(false);
    let (pos_bat, vel_bat) = run(true);

    let bit = bit_identical(&pos_ref, &pos_bat) && bit_identical(&vel_ref, &vel_bat);
    eprintln!(
        "twofield batch equivalence: {} (max |Δpos| = {:.3e}, max |Δvel| = {:.3e}, N = {})",
        if bit { "BIT-IDENTICAL" } else { "DIFFERS" },
        max_abs_diff(&pos_ref, &pos_bat),
        max_abs_diff(&vel_ref, &vel_bat),
        pos_ref.len()
    );
    assert!(
        bit,
        "twofield batched vs per-pass is NOT bit-identical (Δpos {:.3e}, Δvel {:.3e}) — an \
         intra-pass barrier assumption is wrong for some deterministic buffer/stage",
        max_abs_diff(&pos_ref, &pos_bat),
        max_abs_diff(&vel_ref, &vel_bat),
    );
}

/// xpbd: a mixed water+grain pour-over with the copy-interleaved blocks ON (impact, wetting,
/// extraction, fines) plus drag/buoyancy and the per-iteration cell-order reorder — every block
/// that splits the batched pass with an `enc.copy_buffer_to_buffer`. xpbd's float atomics make it
/// nondeterministic, so we gate the centroid (a stable aggregate): the batched-vs-per-pass shift
/// must stay within the intrinsic per-pass-vs-per-pass noise band (batching adds no systematic
/// deviation).
#[test]
fn xpbd_batched_within_nondeterminism_band_of_per_pass() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("batch_pass_equivalence (xpbd): no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::pour_over();
    let mats = Materials::default();
    // Every optional block ON so the batched run exercises each copy-induced pass split.
    let cfg = Config {
        impact_scale: 1.0,
        absorb_rate: 0.5,
        extract_rate: 0.5,
        fines_rate: 0.5,
        ..Config::default()
    };
    let input = EmissionInput::default();

    let run = |batched: bool| -> Vec<[f32; 4]> {
        let mut s = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        if batched {
            s.force_batched_passes_for_test();
        }
        for _ in 0..60 {
            s.step(DT, &input);
        }
        s.read_positions()
    };

    // Intrinsic per-pass run-to-run noise (float-atomic ordering), measured on the centroid.
    let a = run(false);
    let b = run(false);
    let noise = centroid_dist(&a, &b);

    // Batched vs per-pass on the same aggregate.
    let batched = run(true);
    let drift = centroid_dist(&batched, &a);

    eprintln!(
        "xpbd batch equivalence: centroid drift batched-vs-per-pass = {drift:.3e}, intrinsic \
         per-pass noise = {noise:.3e} (N = {})",
        a.len()
    );
    assert!(
        a.iter()
            .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite())
            && batched
                .iter()
                .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite()),
        "xpbd produced non-finite positions"
    );
    // Batched must not drift the bulk beyond the intrinsic noise (plus a small absolute floor for
    // when noise happens to be tiny). A real missing barrier would shift the centroid far outside
    // this band.
    let tol = (4.0 * noise).max(5.0e-3);
    assert!(
        drift <= tol,
        "xpbd batched centroid drifted {drift:.3e} > {tol:.3e} (4x the {noise:.3e} per-pass \
         noise) — batching introduced a systematic deviation, so an intra-pass barrier is missing"
    );
}

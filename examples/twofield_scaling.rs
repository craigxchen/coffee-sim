//! Two-field solver scaling probe (U8, R7/R8/R9): how does the per-FRAME GPU cost of the FULL
//! V60 saturated-pour scene scale with particle count, from 5k toward the 200k real-time target?
//!
//! Mirrors `examples/scaling_probe.rs` (the xpbd probe), but for the deformable saturated bed:
//! a center-pour crater scenario (seeded grain bed + pore water + a pond, `solid_dynamics` ON,
//! a phase-selective filter floor draining the water). Each point sweeps the box edge to grow
//! the lattice; the bed/pond proportions are FIXED (the composition is pre-registered, not tuned
//! to the result — see `COMPOSITION` below), so a bigger N is a bigger version of the SAME scene.
//!
//! THE PER-FRAME COST. The per-pass `timestamp-query` sum (`profile().total_micros()`) captures
//! ONE CFL substep; the dynamic-solid frame runs `solid_substeps` of them (7 at the defaults).
//! Every substep re-runs the full water+solid+pressure pipeline, so the honest per-frame GPU cost
//! is `substeps × timestamp_sum`. `dispatches_per_frame` already counts every substep — the probe
//! prints both so the multiplier is auditable.
//!
//! Reports per point: active particles (water / solid split), median ms/frame, µs/Kparticle,
//! dispatches/frame, and the per-pass breakdown (largest passes first). The headline gates live in
//! `tests/twofield_perf.rs`; this probe is the human-readable table behind them.
//!
//! Run: `cargo run --release --example twofield_scaling`.

use coffee_sim::engine::scene::{Scene, SeedRegion, Species};
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

// ====================== PRE-REGISTERED gate-scene composition (FIXED) =========================
// Documented as named constants BEFORE the run (R9/KTD-5 cost bound): the 200k point is
// ~185–190k water + 10–15k solids. The bed is a thin frictional skeleton (solids are the small
// minority per KTD-5); the pour column + pond + bed pore water are the water majority. Holding
// these ratios fixed across the sweep is what makes µs/Kparticle a fair linearity metric and
// stops the 200k gate being passed by a conveniently cheap mix.
//
// Geometry (unit lattice, spacing = grain_diameter = 1.0):
//   box = [L, Y, L]. The bed is a THIN fixed-height frictional skeleton (a V60 bed is shallow
//   relative to the pour column — KTD-5 makes solids the small minority); the water column above
//   it is the majority phase. Both have footprint ≈ L², so N scales as L² and the SPLIT is
//   constant across the sweep (water : solid fixed by the height ratio, independent of L):
//   grain bed:  footprint L², height BED_LAYERS  ⇒  N_grain ≈ L²·BED_LAYERS
//   pore water: the bed's pore lattice (pitch PORE_PITCH) over the bed volume
//   column:     footprint L², height WATER_COL above the bed ⇒ the water majority
// WATER_COL / BED_LAYERS sets the water:solid ratio → ~12.5 here, landing 200k in the
// pre-registered ~185–190k water + 10–15k solids window. Pinned BEFORE the run.
const BED_LAYERS: f32 = 4.0; // bed height in unit lattice layers (thin skeleton)
const WATER_COL: f32 = 52.0; // water-column height above the bed (the majority phase)
const PORE_PITCH: f32 = 1.6; // pore-water lattice pitch inside the bed (sparse — solids dominate the bed)

// Swept box edges (scene units). Tuned so N spans ~5k → ~200k at the fixed composition.
const EDGES: &[f32] = &[8.0, 14.0, 22.0, 32.0, 44.0, 56.0, 61.0];

// Fixed pressure budget (the U3 default knob grid point) and pour strength — pinned, not swept.
const JET_FLOW: f32 = 70.0;
const JET_RADIUS: f32 = 1.5;
const WARMUP_FRAMES: u32 = 60; // settle the saturated bed + start the pour
const MEASURE_FRAMES: u32 = 40; // median window (matches the xpbd probe)

struct Row {
    edge: f32,
    n_water: u32,
    n_solid: u32,
    substeps: u32,
    dispatches: u32,
    median_ms: f32,
    per_kpart: f32,
    breakdown: Vec<(String, f32)>, // per-substep pass µs, aggregated by label, largest first
}

/// Build the FIXED-composition V60 saturated-pour scene at box edge `edge`.
fn build_scene(edge: f32) -> Scene {
    let bed_top = BED_LAYERS;
    let water_lo = bed_top + 1.0;
    let water_hi = water_lo + WATER_COL;
    let y = (water_hi + 16.0).round(); // headroom above the column for the kettle + pour cavity
    Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 6000.0, // generous dose headroom so the center pour never clamps
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [edge, y, edge],
        solids: Vec::new(),
        regions: vec![
            // Grain bed across the footprint (wall margin d/2 = 0.5).
            SeedRegion {
                min: [0.5, 0.5, 0.5],
                max: [edge - 0.5, bed_top, edge - 0.5],
                species: Species::Grain,
            },
            // Water column / pond above the bed (the majority phase).
            SeedRegion {
                min: [0.7, water_lo, 0.7],
                max: [edge - 0.7, water_hi, edge - 0.7],
                species: Species::Water,
            },
        ],
    }
}

/// Seed the bed pore water onto a sub-lattice inside the bed so the bed starts saturated (mirrors
/// the `tests/twofield_full.rs` build_bed pore-fill), using the dormant water-pool headroom.
fn saturate_bed(solver: &mut TwofieldSolver, edge: f32, bed_top: f32, n_water_seed: u32) -> u32 {
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
    let mut pos = solver.read_positions();
    // The pond column seed lives at the water prefix [0, n_pond). Append pore water after it,
    // into dormant water-pool slots [n_pond, water_pool_capacity); drop any overflow.
    let n_pond = solver.phase_counts().0;
    let _ = n_water_seed;
    let room = solver.water_pool_capacity().saturating_sub(n_pond) as usize;
    let take = pore.len().min(room);
    for (i, p) in pore.iter().take(take).enumerate() {
        let slot = n_pond as usize + i;
        pos[slot] = [p[0], p[1], p[2], 1.0];
    }
    solver.write_positions_for_test(&pos);
    let live = n_pond + take as u32;
    solver.set_live_water_for_test(live);
    live
}

fn measure(gpu: &GpuContext, edge: f32) -> Row {
    let scene = build_scene(edge);
    let mats = Materials {
        grain_diameter: 1.0,
        ..Materials::default()
    };
    let cfg = Config {
        solid_dynamics: true,  // the bed is deformable (the U7 regime)
        tf_filter_floor: true, // water drains so the bed doesn't just flood
        nozzle_radius: JET_RADIUS,
        tf_wet_cohesion: 4.0, // wet-bed cohesion (the crater regime)
        tf_cohesion_speak: 0.4,
        ..Config::default()
    };
    let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, gpu);
    let (n_water_seed, n_solid) = solver.phase_counts();
    let bed_top = scene.regions[0].max[1];
    let n_water = saturate_bed(&mut solver, edge, bed_top, n_water_seed);

    let axis = (edge / 2.0, edge / 2.0);
    // Kettle ABOVE the pond/column surface (a real pour-over: the jet enters through open air,
    // not submerged) — water_hi is the seeded column top.
    let water_hi = scene.regions[1].max[1];
    let quiet = EmissionInput::default();
    let pour = EmissionInput {
        kettle_pos: [axis.0, water_hi + 6.0, axis.1],
        flow_rate: JET_FLOW,
        pour_angle: 0.0,
        ..EmissionInput::default()
    };
    // Warm up: a few quiet frames to settle, then start the pour (the measured regime).
    for f in 0..WARMUP_FRAMES {
        let drive = if f < WARMUP_FRAMES / 3 { &quiet } else { &pour };
        solver.step(DT, drive);
    }

    let substeps = solver.substeps_for_dt(DT);
    let mut samples: Vec<f32> = Vec::new();
    let mut agg: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let mut dispatches = 0u32;
    let mut pocket_frames = 0u32;
    for _ in 0..MEASURE_FRAMES {
        solver.step(DT, &pour);
        if solver.pocket_present() {
            pocket_frames += 1;
        }
        solver.sample_diagnostics();
        let prof = solver.profile();
        dispatches = prof.dispatches_per_frame;
        let sub = prof.total_micros();
        if sub > 0.0 {
            // substep-0 timestamp sum × substeps = honest per-frame GPU cost.
            samples.push(sub * substeps as f32);
            for (label, us) in &prof.passes {
                *agg.entry(label.clone()).or_insert(0.0) += us * substeps as f32;
            }
        }
    }
    eprintln!(
        "    [edge {edge:.0}: pocket present in {pocket_frames}/{MEASURE_FRAMES} measured frames]"
    );
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = if samples.is_empty() {
        0.0
    } else {
        samples[samples.len() / 2]
    };
    let n = (n_water + n_solid).max(1);
    let per_kpart = median_us / (n as f32 / 1000.0);
    // Average the breakdown over the measured frames, sort largest-first.
    let denom = samples.len().max(1) as f32;
    let mut breakdown: Vec<(String, f32)> = agg.into_iter().map(|(k, v)| (k, v / denom)).collect();
    breakdown.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    Row {
        edge,
        n_water,
        n_solid,
        substeps,
        dispatches,
        median_ms: median_us / 1000.0,
        per_kpart,
        breakdown,
    }
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_scaling: no GPU adapter; cannot run.");
        return;
    };

    println!("\n=== twofield V60 saturated-pour scaling (Apple M5) ===");
    println!(
        "  PRE-REGISTERED composition: ~185–190k water + 10–15k solids @ 200k (KTD-5 cost bound)"
    );
    println!(
        "  per-frame GPU = substeps × (substep-0 timestamp sum); dispatches counts every substep\n"
    );
    println!(
        "{:>5} {:>9} {:>8} {:>7} {:>6} {:>10} {:>11} {:>5}",
        "edge", "N_water", "N_solid", "N", "subs", "ms/frame", "µs/Kpart", "disp"
    );

    let mut rows = Vec::new();
    for &e in EDGES {
        let r = measure(&gpu, e);
        println!(
            "{:>5.0} {:>9} {:>8} {:>7} {:>6} {:>10.3} {:>11.2} {:>5}",
            r.edge,
            r.n_water,
            r.n_solid,
            r.n_water + r.n_solid,
            r.substeps,
            r.median_ms,
            r.per_kpart,
            r.dispatches,
        );
        rows.push(r);
    }

    // Per-pass breakdown of the largest (closest-to-200k) point — where the time goes.
    if let Some(top) = rows.last() {
        println!(
            "\n  per-pass breakdown @ N={} ({} µs/frame total, largest passes):",
            top.n_water + top.n_solid,
            (top.median_ms * 1000.0).round() as u32
        );
        let total: f32 = top.breakdown.iter().map(|(_, us)| us).sum();
        for (label, us) in top.breakdown.iter().take(12) {
            println!(
                "    {:<18} {:>9.1} µs  ({:>4.1}%)",
                label,
                us,
                100.0 * us / total.max(1.0e-6)
            );
        }
    }

    // Linearity over the 40k→200k saturated range (the R9 linearity indicator): µs/Kpart
    // max/min, mirroring the xpbd probe's climb metric.
    let saturated: Vec<&Row> = rows
        .iter()
        .filter(|r| r.n_water + r.n_solid >= 40_000)
        .collect();
    if saturated.len() >= 2 {
        let lo = saturated
            .iter()
            .map(|r| r.per_kpart)
            .fold(f32::INFINITY, f32::min);
        let hi = saturated.iter().map(|r| r.per_kpart).fold(0.0, f32::max);
        println!(
            "\n  linearity (µs/Kpart climb over N≥40k): {:.2}× (flat ⇒ ~1.0×, linear scaling)",
            hi / lo.max(1.0e-6)
        );
    }
}

//! Twofield regression gate: pure water in the V60 cup must POOL (stay spread across the cup,
//! centered, pocket-free) — NOT collapse into the corner. This is the missing gate for the
//! reported WaterOnly symptom: with the two-field solver, water "compresses into the corner of
//! the cup and never fills up any volume". Mirrors the web WaterOnly scene
//! (`Scene::v60_pour_water_only`) at the web resolution (spacing 0.16). XPBD is unaffected.
//!
//! ROOT CAUSE (fixed here): the V60 cup+cone air is OPEN to the atmosphere, but the U4 pocket
//! detection misread it as one enclosed bubble and the bubble constraint (λ ≈ −3000) crushed the
//! standing water into a corner. Two mechanisms, both fixed in src/solvers/twofield/:
//!   1. The flood-fill OUTSIDE label could not reach the cup air — the fixed 24-sweep budget was
//!      far short of the V60 air path, and a 6-neighbor flood cannot squeeze the label through
//!      the ~1-cell-wide cone apex hole. Fix: scene-derived flood budget (grid Manhattan
//!      diameter) + seed OUTSIDE from every wall-adjacent air cell (a sub-resolution void at an
//!      SDF wall is open, never trapped gas) so the flood conducts through the apex.
//!   2. The SDF wall no-penetration projector only fired strictly inside the wall material
//!      (`dist < 0`); the supporting fluid layer of the non-grid-aligned cup floor sits OUTSIDE
//!      the wall (`dist ∈ [0, h)`) and was left y-free, so water free-fell into the floor. Fix:
//!      band the projector by one cell (`dist < h`) in node_setup + drag_fold.
//!
//! These eliminate the catastrophic collapse (the bubble suction, the free-fall) and keep the
//! water pocket-free, centered, and spread. A residual sub-rest over-compression of the floor
//! water in the confined cup remains (see the file footer) — these gates pin the ANTI-COLLAPSE
//! invariants the fix restores, not bit-perfect hydrostatic rest.

#![allow(clippy::needless_range_loop)]

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

// Cup geometry (utils::geometry::v60_dripper cylinder): floor -8, rim -3.5, radius 3.
const CUP_FLOOR: f32 = -8.0;
const CUP_RIM: f32 = -3.5;
const CUP_RADIUS: f64 = 3.0;

/// Web WaterOnly resolution.
fn web_mats() -> Materials {
    let r = 0.16_f32;
    Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    }
}

/// Web WaterOnly twofield config — mirrors `src/web.rs::setup_for(WaterOnly, Twofield)`: the
/// XPBD-tuned base plus the twofield-only jet widening (a grid solver needs a grid-resolvable
/// jet or the thin plunging stream traps air the pocket machinery carries as a crushing bubble).
fn web_cfg() -> Config {
    Config {
        nozzle_radius: 0.55,
        max_speed: 12.0,
        xsph_viscosity_c: 0.02,
        ..Config::default()
    }
}

/// V60 cone+cup with a water slab seeded INSIDE the cup (settle without waiting for a pour).
fn cup_static_scene() -> Scene {
    let mut s = Scene::v60_pour_water_only();
    s.regions = vec![SeedRegion {
        min: [-2.0, CUP_FLOOR + 0.3, -2.0],
        max: [2.0, CUP_FLOOR + 2.0, 2.0],
        species: Species::Water,
    }];
    s
}

struct CupStats {
    n: usize,
    radial_rms: f64,
    centroid_r: f64,
}

/// Cup-water = live water particles below the rim.
fn cup_stats(solver: &TwofieldSolver) -> CupStats {
    let nlive = solver.phase_counts().0 as usize;
    let pos = solver.read_positions();
    let cup: Vec<[f32; 4]> = pos[..nlive]
        .iter()
        .copied()
        .filter(|p| p[1] <= CUP_RIM && p[1] >= CUP_FLOOR - 1.0)
        .collect();
    let n = cup.len();
    if n == 0 {
        return CupStats {
            n: 0,
            radial_rms: 0.0,
            centroid_r: 0.0,
        };
    }
    let (mut cx, mut cz, mut rr) = (0.0f64, 0.0f64, 0.0f64);
    for p in &cup {
        cx += p[0] as f64;
        cz += p[2] as f64;
        rr += (p[0] as f64).powi(2) + (p[2] as f64).powi(2);
    }
    cx /= n as f64;
    cz /= n as f64;
    CupStats {
        n,
        radial_rms: (rr / n as f64).sqrt(),
        centroid_r: (cx * cx + cz * cz).sqrt(),
    }
}

/// Static cup water must STAY spread across the cup and centered, with NO spurious enclosed
/// pocket — the bubble that crushed it to a corner in the bug. (The original bug: water pulled
/// off-axis, an ~10k-cell pocket with λ ≈ −3000, particle density spiking past 500× rest.)
#[test]
fn static_cup_water_pools_not_corner() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_cup: no GPU adapter; skipping.");
        return;
    };
    let mut solver = TwofieldSolver::build(&cup_static_scene(), &web_mats(), &web_cfg(), &gpu);
    let quiet = EmissionInput::default();

    solver.step(DT, &quiet);
    let s0 = cup_stats(&solver);
    assert!(s0.n > 1000, "seed too small ({})", s0.n);

    let mut max_pocket_lambda = 0.0f32;
    for _ in 0..400 {
        solver.step(DT, &quiet);
        // The cup/cone air is OPEN: no enclosed pocket may form (the bug's crushing bubble).
        let lambda = solver.read_bubble()[0];
        max_pocket_lambda = max_pocket_lambda.max(lambda.abs());
    }
    let sf = cup_stats(&solver);
    println!(
        "twofield cup STATIC: radial_rms {:.3} -> {:.3} | centroid_r {:.3} -> {:.3} | |λ|_max {max_pocket_lambda:.2} | pocket_present {}",
        s0.radial_rms, sf.radial_rms, s0.centroid_r, sf.centroid_r, solver.pocket_present()
    );

    // No spurious crushing bubble: the open cup/cone air is never an enclosed pocket. (The bug
    // sustained |λ| ≈ 3000; a genuine transient surface gap is orders of magnitude smaller.)
    assert!(
        !solver.pocket_present() && max_pocket_lambda < 100.0,
        "the open cup/cone air was misclassified as an enclosed pocket: |λ|_max {max_pocket_lambda:.1}"
    );
    // Water stays SPREAD across the cup, not pulled into a thin core/corner (the bug shrank the
    // radial spread as it collapsed): keep most of the seeded radial extent.
    assert!(
        sf.radial_rms >= 0.75 * s0.radial_rms,
        "COLLAPSE: radial spread shrank {:.3} -> {:.3} (water pulled toward a core/corner)",
        s0.radial_rms,
        sf.radial_rms
    );
    // Water stays CENTERED — it does not drift into a corner (the bug drifted the centroid out
    // past 0.6 toward the wall).
    assert!(
        sf.centroid_r <= 0.4 * CUP_RADIUS,
        "OFF-AXIS COLLAPSE: cup-water centroid radius {:.3} (cup radius {CUP_RADIUS})",
        sf.centroid_r
    );
}

/// Pour into the empty V60 cup: water reaches the cup, fills it (rises) while pouring, and stays
/// centered + pocket-free — it does not collapse into the corner.
#[test]
fn poured_cup_water_fills_not_corner() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_cup: no GPU adapter; skipping.");
        return;
    };
    let mut solver =
        TwofieldSolver::build(&Scene::v60_pour_water_only(), &web_mats(), &web_cfg(), &gpu);
    let pour = EmissionInput {
        kettle_pos: [0.0, 5.0, 0.0],
        flow_rate: 8.0,
        pour_angle: 0.0,
        ..EmissionInput::default()
    };
    let quiet = EmissionInput::default();

    // Pour, tracking how high the cup water rises (the cup must FILL, not crush to the floor).
    let mut max_fill_height = f64::MIN;
    for _ in 0..400 {
        solver.step(DT, &pour);
        let nlive = solver.phase_counts().0 as usize;
        let pos = solver.read_positions();
        let top = pos[..nlive]
            .iter()
            .filter(|p| p[1] <= CUP_RIM && p[1] >= CUP_FLOOR - 1.0)
            .map(|p| p[1] as f64)
            .fold(f64::MIN, f64::max);
        max_fill_height = max_fill_height.max(top);
    }
    // Stop pouring and let the standing pool settle: with the pour off, the cup pool is standing
    // water — no plunge to trap air — so any enclosed pocket here would be the bug's spurious
    // crushing bubble, not a legitimate transient pour cavity.
    let mut settled_pocket_lambda = 0.0f32;
    for _ in 0..200 {
        solver.step(DT, &quiet);
        settled_pocket_lambda = settled_pocket_lambda.max(solver.read_bubble()[0].abs());
    }
    let sf = cup_stats(&solver);
    println!(
        "twofield cup POUR: cup n {} | radial_rms {:.3} | centroid_r {:.3} | max fill y {:.2} (floor {CUP_FLOOR}, rim {CUP_RIM}) | settled |λ|_max {settled_pocket_lambda:.2}",
        sf.n, sf.radial_rms, sf.centroid_r, max_fill_height
    );
    assert!(sf.n > 1000, "almost no water reached the cup ({})", sf.n);
    // The cup FILLS: water rises well above the floor toward the rim (the bug never filled — it
    // crushed everything into the floor corner).
    assert!(
        max_fill_height >= CUP_FLOOR as f64 + 2.0,
        "cup never filled: max water height {max_fill_height:.2} barely above floor {CUP_FLOOR}"
    );
    // The settled pool stays CENTERED — no off-axis collapse into a corner.
    assert!(
        sf.centroid_r <= 0.4 * CUP_RADIUS,
        "OFF-AXIS COLLAPSE: cup-water centroid radius {:.3} (cup radius {CUP_RADIUS})",
        sf.centroid_r
    );
    // The STANDING pool (pour off) has no spurious crushing bubble (the bug sustained |λ| ≈ 3000
    // on the open cup air; a legitimate transient pour cavity is gone once the pour stops).
    assert!(
        settled_pocket_lambda < 100.0,
        "the standing cup pool grew a spurious crushing pocket: settled |λ|_max {settled_pocket_lambda:.1}"
    );
}

// ---------------------------------------------------------------------------------------------
// KNOWN RESIDUAL (documented, not gated here): in the confined V60 cup the floor water still
// over-compresses to several × rest over many seconds of settling — the collocated pressure
// solve does not reach hydrostatic rest in the radially-confined SDF cup at the web resolution
// (h = 2× spacing). This is independent of the pocket (verified pocket-free) and of floor/grid
// alignment (verified: an exactly node-aligned cup floor over-compresses identically while an
// open box floor stays bounded). It is a deeper free-surface / pressure-iteration limitation,
// not the corner-collapse bug these gates lock down.

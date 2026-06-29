//! U2a AUDIT (plan docs/plans/2026-06-17-001) — localize the aligned-square SDF-wall over-pack
//! residual by comparing configs that differ ONLY in which surface is an SDF wall vs a box face:
//!   A) box floor + box side walls           (pure flat box — the 0.98× reference)
//!   B) box floor + SDF square side walls     (isolates the SDF SIDE wall)
//!   C) SDF square floor + SDF square walls    (adds the gravity-loaded SDF FLOOR)
//! Density is reported ABSOLUTELY as ρ/ρ_rest, calibrated so the box interior = 1.0 (the box is
//! at rest per the original diagnosis). A wall/interior RATIO would wash out a uniform over-pack,
//! so we report wall-shell, floor-shell, and interior separately. DIAGNOSTIC only — no solver
//! change, no pinned gate (`feedback_no_premature_tests`); it PRINTS the localization.
//!
//! Run: `cargo test --release --test twofield_wall_audit -- --nocapture`

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{TwofieldSolver, WALL_BC_MULTI, WALL_BC_SINGLE};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::sdf::{SdfPrimitive, SolidKind, MASK_WATER};
use coffee_sim::EmissionInput;
use glam::Vec3;

const DT: f32 = 1.0 / 60.0;
const SPACING: f32 = 0.16;
const SETTLE: usize = 250;
const APOTHEM: f32 = 2.4; // ±2.4 faces land on nodes for box_min.x=-4, h=0.32
const TOP_Y: f32 = 3.0;

fn web_mats() -> Materials {
    Materials {
        particle_spacing: SPACING,
        support_radius: 2.0 * SPACING,
        ..Materials::default()
    }
}
fn web_cfg() -> Config {
    Config {
        max_speed: 12.0,
        ..Config::default()
    }
}

fn slab(floor: f32) -> SeedRegion {
    SeedRegion {
        min: [-2.3, floor + 0.2, -2.3],
        max: [2.3, floor + 2.5, 2.3],
        species: Species::Water,
    }
}

fn poly_wall(floor_y: f32) -> SdfPrimitive {
    SdfPrimitive {
        kind: SolidKind::PolyCup {
            center: Vec3::ZERO,
            floor_y,
            rim_y: TOP_Y,
            apothem: APOTHEM,
            sides: 4,
        },
        species_mask: MASK_WATER,
        friction: 0.0,
    }
}

/// A) pure flat box: walls + floor are box faces. box_min.x=-2.4 (faces node-aligned by construction).
fn box_scene() -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 100.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-APOTHEM, -5.0, -APOTHEM],
        box_max: [APOTHEM, TOP_Y, APOTHEM],
        regions: vec![slab(-5.0)],
        solids: vec![],
    }
}

/// B) SDF square SIDE walls, water rests on the BOX floor (cup floor pushed below the domain).
fn side_wall_scene() -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 100.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-4.0, -5.0, -4.0],
        box_max: [4.0, TOP_Y, 4.0],
        regions: vec![slab(-5.0)],
        solids: vec![poly_wall(-6.0)], // floor below domain → never the resting surface
    }
}

/// C) full SDF square cup: water rests on the SDF FLOOR (node-aligned at -4.8; box floor far below).
fn sdf_floor_scene() -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 100.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-4.0, -8.0, -4.0], // origin.y = -8.32; floor -4.8 → (3.52/0.32)=11 → node-aligned
        box_max: [4.0, TOP_Y, 4.0],
        regions: vec![slab(-4.8)],
        solids: vec![poly_wall(-4.8)],
    }
}

/// Mean neighbor count (within 2*spacing, excl self) over a selected box of particles. (mean, n).
fn mean_nb(pos: &[[f32; 4]], xr: (f32, f32), yr: (f32, f32), zband: f32) -> (f64, usize) {
    let r2 = (2.0 * SPACING) * (2.0 * SPACING);
    let sel: Vec<[f32; 4]> = pos
        .iter()
        .copied()
        .filter(|p| {
            (xr.0..=xr.1).contains(&p[0]) && (yr.0..=yr.1).contains(&p[1]) && p[2].abs() < zband
        })
        .collect();
    if sel.is_empty() {
        return (0.0, 0);
    }
    let mut total = 0u64;
    for a in &sel {
        for b in pos {
            let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            if d2 <= r2 && d2 > 0.0 {
                total += 1;
            }
        }
    }
    (total as f64 / sel.len() as f64, sel.len())
}

/// Mean nearest-neighbor distance over `sel` (against `all`).
fn dnn_mean(sel: &[[f32; 4]], all: &[[f32; 4]]) -> f32 {
    if sel.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for a in sel {
        let mut best = f32::INFINITY;
        for b in all {
            let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            if d2 > 1e-9 {
                best = best.min(d2);
            }
        }
        sum += best.sqrt();
    }
    sum / sel.len() as f32
}

struct Probe {
    label: &'static str,
    interior_nb: f64,
    side_nb: f64,
    floor_nb: f64,
    corner_nb: f64,
    interior_dnn: f32,
    nm_floor: [f32; 8],
    floor_dist: f32,
    floor_wallish: bool,
}

fn probe(
    label: &'static str,
    scene: &Scene,
    gpu: &GpuContext,
    rest_floor: f32,
    is_sdf_floor: bool,
) -> Probe {
    let mut s = TwofieldSolver::build(scene, &web_mats(), &web_cfg(), gpu);
    let quiet = EmissionInput::default();
    for _ in 0..SETTLE {
        s.step(DT, &quiet);
    }
    let pos = s.read_positions();
    let nlive = s.phase_counts().0 as usize;
    let live = &pos[..nlive];
    let ys: Vec<f32> = live.iter().map(|p| p[1]).collect();
    let ymax = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let ymid = 0.5 * (rest_floor + ymax);

    // interior column (away from all walls), side-wall shell (+x), floor shell (just above floor).
    let interior_nb = mean_nb(live, (-0.6, 0.6), (ymid - 0.5, ymid + 0.5), 0.6).0;
    let side_nb = mean_nb(
        live,
        (APOTHEM - 2.0 * SPACING, APOTHEM),
        (ymid - 0.5, ymid + 0.5),
        0.6,
    )
    .0;
    let floor_nb = mean_nb(
        live,
        (-0.6, 0.6),
        (rest_floor, rest_floor + 2.0 * SPACING),
        0.6,
    )
    .0;
    // CORNER = floor∩wall seam (+x bottom edge): the multi-face-incidence locus.
    let corner_nb = mean_nb(
        live,
        (APOTHEM - 2.0 * SPACING, APOTHEM),
        (rest_floor, rest_floor + 2.0 * SPACING),
        0.6,
    )
    .0;
    let interior_sel: Vec<[f32; 4]> = live
        .iter()
        .copied()
        .filter(|p| {
            p[0].abs() < 0.6 && p[2].abs() < 0.6 && p[1] >= ymid - 0.5 && p[1] <= ymid + 0.5
        })
        .collect();
    let interior_dnn = dnn_mean(&interior_sel, live);

    // Floor node at (0, rest_floor, 0): its M̃⁻¹ + SDF classification.
    let (origin, h, dims) = s.grid_spec();
    let idx = |i: i32, j: i32, k: i32| {
        (i as usize) + (dims[0] as usize) * ((j as usize) + (dims[1] as usize) * (k as usize))
    };
    let i0 = ((0.0 - origin[0]) / h).round() as i32;
    let jf = ((rest_floor - origin[1]) / h).round() as i32;
    let k0 = ((0.0 - origin[2]) / h).round() as i32;
    let nm_floor = s.read_node_matrices()[idx(i0, jf, k0)];
    let xp = Vec3::new(
        origin[0] + i0 as f32 * h,
        origin[1] + jf as f32 * h,
        origin[2] + k0 as f32 * h,
    );
    let floor_dist = if is_sdf_floor {
        scene.solids[0].cavity(xp).0
    } else {
        f32::NAN
    };

    Probe {
        label,
        interior_nb,
        side_nb,
        floor_nb,
        corner_nb,
        interior_dnn,
        nm_floor,
        floor_dist,
        floor_wallish: is_sdf_floor && floor_dist < 0.0,
    }
}

/// VALIDATION (high-level, #[ignore] diagnostic): do the three water-side symptoms — corner/wall
/// over-pack, settled stirring, and not-reaching-hydrostatic-rest (floor over-compression) — share
/// the SAME root: the ∇·v-only under-converged pressure with no density constraint? Decisive test:
/// if the root is UNDER-CONVERGENCE, cranking the fine-sweep budget fixes them; if the root is the
/// MODEL (a compacted static pool has ∇·v=0, so converging ∇·v→0 leaves the wrong density), more
/// sweeps change nothing (or detonate via the one-sided relief). Measures all three on the real
/// v60_cup_static_full in SINGLE mode (the base model, no corner-BC patch) across sweep/relief
/// interventions. `cargo test ... validate_pressure_model_root -- --ignored --nocapture`.
#[test]
#[ignore = "on-demand high-level diagnostic (slow); validates the shared pressure-model root"]
fn validate_pressure_model_root() {
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    let quiet = EmissionInput::default();
    // ρ_rest calibration: a flat box interior at rest (same spacing).
    let rest = {
        let mut s = TwofieldSolver::build(&box_scene(), &web_mats(), &web_cfg(), &gpu);
        for _ in 0..SETTLE {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let n = s.phase_counts().0 as usize;
        mean_nb(&pos[..n], (-0.6, 0.6), (-4.0, -3.0), 0.6)
            .0
            .max(1.0)
    };
    let measure = |label: &str, fine_sweeps: u32, relief: bool| {
        let mut s =
            TwofieldSolver::build(&Scene::v60_cup_static_full(), &web_mats(), &web_cfg(), &gpu);
        s.set_pressure_budget_for_test(4, 8, fine_sweeps);
        s.set_relief_for_test(relief);
        for _ in 0..400 {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let nlive = s.phase_counts().0 as usize;
        let cup: Vec<[f32; 4]> = pos[..nlive]
            .iter()
            .copied()
            .filter(|p| p[1] <= -3.5 && p[1] >= -8.5)
            .collect();
        let ymin = cup.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let band = (ymin + 0.05, ymin + 0.05 + 2.0 * SPACING);
        // (1) over-pack: wall shell + interior. (3) hydrostatic-rest proxy: floor-shell over-pack.
        let wall = mean_nb_radial(&cup, 3.0 - 2.0 * SPACING, 3.0, band).0 / rest;
        let interior = mean_nb_radial(&cup, 0.0, 1.0, band).0 / rest;
        // (2) stirring: mean speed over massy cup nodes (the post-project settled field).
        let gv = s.read_grid_velocities();
        let (mut ssum, mut sn) = (0.0f64, 0u64);
        for v in &gv {
            if v[3] > 1.0e-6 {
                ssum += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() as f64;
                sn += 1;
            }
        }
        let mean_speed = ssum / sn.max(1) as f64;
        println!(
            "[{label:<22}] wall ρ/ρ_rest={wall:.2}  interior ρ/ρ_rest={interior:.2}  settled mean|v|={mean_speed:.3}"
        );
    };
    println!(
        "\n==== PRESSURE-MODEL ROOT VALIDATION (v60_cup_static_full, SINGLE, 400 frames) ===="
    );
    println!("  ρ_rest (box interior) = {rest:.2}\n");
    measure("baseline fine=8", 8, true);
    measure("cranked fine=32", 32, true);
    measure("cranked fine=64", 64, true);
    measure("relief OFF fine=8", 8, false);
    println!("  → if cranking does NOT reduce over-pack, the root is the MODEL (∇·v≠density), not convergence.");
    println!("================================================================================\n");
}

/// Radial-shell neighbor mean for the real cylindrical V60 cup. (mean, n).
fn mean_nb_radial(pos: &[[f32; 4]], rlo: f32, rhi: f32, yr: (f32, f32)) -> (f64, usize) {
    let r2 = (2.0 * SPACING) * (2.0 * SPACING);
    let sel: Vec<[f32; 4]> = pos
        .iter()
        .copied()
        .filter(|p| {
            let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
            rr >= rlo && rr <= rhi && p[1] >= yr.0 && p[1] <= yr.1
        })
        .collect();
    if sel.is_empty() {
        return (0.0, 0);
    }
    let mut total = 0u64;
    for a in &sel {
        for b in pos {
            let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            if d2 <= r2 && d2 > 0.0 {
                total += 1;
            }
        }
    }
    (total as f64 / sel.len() as f64, sel.len())
}

/// Measure the documented `v60_cup_static_full` over-pack with the SAME probe + ρ_rest calibration.
/// Cup: floor -8, rim -3.5, radius 3 (utils::geometry::v60_dripper). Returns (interior, wall, floor).
fn probe_v60_cup(gpu: &GpuContext, rest: f64) -> (f64, f64, f64, usize) {
    let mut s = TwofieldSolver::build(&Scene::v60_cup_static_full(), &web_mats(), &web_cfg(), gpu);
    let quiet = EmissionInput::default();
    for _ in 0..400 {
        s.step(DT, &quiet);
    }
    let pos = s.read_positions();
    let nlive = s.phase_counts().0 as usize;
    let live = &pos[..nlive];
    // Characterize WHERE the water is: live count, y-extent, and a radial ρ/ρ_rest profile.
    let cup: Vec<[f32; 4]> = live
        .iter()
        .copied()
        .filter(|p| p[1] <= -3.5 && p[1] >= -8.5)
        .collect();
    let ys: Vec<f32> = cup.iter().map(|p| p[1]).collect();
    let (ymin, ymax) = (
        ys.iter().cloned().fold(f32::INFINITY, f32::min),
        ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    // Match the band to where the water actually settled (it pancakes low).
    let yband = (ymin + 0.05, ymin + 0.05 + 2.0 * SPACING);
    println!(
        "  [v60 cup chars] live={} cup-water={} y-extent=[{:.2},{:.2}] (rim -3.5, floor -8)",
        nlive,
        cup.len(),
        ymin,
        ymax
    );
    print!("  radial ρ/ρ_rest by bin (r=0..3, mid-height): ");
    for bin in 0..6 {
        let rlo = bin as f32 * 0.5;
        let rhi = rlo + 0.5;
        let (m, nb) = mean_nb_radial(&cup, rlo, rhi, yband);
        print!("[{:.1}-{:.1}]={:.2}(n{}) ", rlo, rhi, m / rest, nb);
    }
    println!();
    let interior = mean_nb_radial(&cup, 0.0, 1.0, yband).0;
    let (wall, nwall) = mean_nb_radial(&cup, 3.0 - 2.0 * SPACING, 3.0, yband);
    let floor = mean_nb_radial(&cup, 0.0, 1.5, (-8.0, -8.0 + 2.0 * SPACING)).0;

    // RECONCILIATION: nearest-neighbor distance directly tests physical cramming (it can't be
    // averaged away like a 2s-radius count can). A true 25× volume over-pack ⇒ d_nn ≈ s/2.9.
    // d_nn stats + per-particle ρ=(s/d_nn)³ mean (the likely original "25×" metric) over a subset.
    let dnn_stats = |subset: &[[f32; 4]]| -> (f32, f32, f32, f64) {
        let mut dnn: Vec<f32> = Vec::with_capacity(subset.len());
        for a in subset {
            let mut best = f32::INFINITY;
            for b in &cup {
                let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
                if d2 > 1e-9 {
                    best = best.min(d2);
                }
            }
            dnn.push(best.sqrt());
        }
        dnn.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mean = dnn.iter().sum::<f32>() / dnn.len().max(1) as f32;
        let p05 = dnn.get(dnn.len() / 20).copied().unwrap_or(0.0);
        let min = dnn.first().copied().unwrap_or(0.0);
        // per-particle (s/d_nn)^3 mean — clump-dominated, the metric that yields the "25×".
        let rho_mean = dnn
            .iter()
            .map(|d| (SPACING / d.max(1e-6)).powi(3) as f64)
            .sum::<f64>()
            / dnn.len().max(1) as f64;
        (mean, p05, min, rho_mean)
    };
    let in_band = |rlo: f32, rhi: f32| -> Vec<[f32; 4]> {
        cup.iter()
            .copied()
            .filter(|p| {
                let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
                rr >= rlo && rr <= rhi && p[1] >= yband.0 && p[1] <= yband.1
            })
            .collect()
    };
    let (wm, wp05, wmin, wrho) = dnn_stats(&in_band(2.5, 3.0));
    let (im, _ip05, _imin, irho) = dnn_stats(&in_band(0.0, 1.0));
    println!(
        "  [RECONCILE] d_nn (rest spacing {:.3}):  WALL mean={:.4} p05={:.4} min={:.4}  |  INTERIOR mean={:.4}",
        SPACING, wm, wp05, wmin, im
    );
    println!(
        "  [RECONCILE] per-particle ρ=(s/d_nn)³ MEAN (the likely original metric):  WALL={:.1}×  INTERIOR={:.1}×",
        wrho, irho
    );

    // CORNER = floor∩wall seam (low-y ∩ high-r): the visible "corner packing" locus, where the
    // floor band + wall band compound and the single SDF normal is ambiguous (multi-face seam).
    let corner: Vec<[f32; 4]> = cup
        .iter()
        .copied()
        .filter(|p| {
            let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
            (3.0 - 2.0 * SPACING..=3.0).contains(&rr)
                && (-8.0..=-8.0 + 2.0 * SPACING).contains(&p[1])
        })
        .collect();
    let (corner_coarse, ncorner) =
        mean_nb_radial(&cup, 3.0 - 2.0 * SPACING, 3.0, (-8.0, -8.0 + 2.0 * SPACING));
    let (cm, cp05, cmin, crho) = dnn_stats(&corner);
    println!(
        "  [CORNER seam] n={} coarse ρ/ρ_rest={:.2}  d_nn mean={:.4} p05={:.4} min={:.4} ({:.2}× rest)  (s/d_nn)³_mean={:.1}×",
        ncorner,
        corner_coarse / rest,
        cm,
        cp05,
        cmin,
        cm / SPACING,
        crho
    );

    (interior / rest, wall / rest, floor / rest, nwall)
}

/// DIG-DEEPER: is the cup clumping APIC-affine-driven? Sweep the PIC blend (0 = pure APIC,
/// 1 = pure PIC) on the real cup and watch mean d_nn (clumping) + settled depth (pancaking) +
/// the (s/d_nn)³ tail. If d_nn climbs toward the at-rest box (~0.114) as blend→1, the existing
/// PIC-blend knob is the anti-clumping lever and no new mechanism is needed.
#[test]
fn cup_clumping_pic_blend_sweep() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_wall_audit: no GPU adapter; skipping.");
        return;
    };
    println!("\n========== CLUMPING vs PIC BLEND (v60_cup_static_full, 400 frames) ==========");
    println!("  reference: at-rest box interior d_nn ≈ 0.114 (rest spacing 0.160)\n");
    for &blend in &[0.0f32, 0.05, 0.2, 0.5, 1.0] {
        let mut s =
            TwofieldSolver::build(&Scene::v60_cup_static_full(), &web_mats(), &web_cfg(), &gpu);
        s.set_pic_blend_for_test(blend);
        let quiet = EmissionInput::default();
        for _ in 0..400 {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let nlive = s.phase_counts().0 as usize;
        let cup: Vec<[f32; 4]> = pos[..nlive]
            .iter()
            .copied()
            .filter(|p| p[1] <= -3.5 && p[1] >= -8.5)
            .collect();
        let ys: Vec<f32> = cup.iter().map(|p| p[1]).collect();
        let (ymin, ymax) = (
            ys.iter().cloned().fold(f32::INFINITY, f32::min),
            ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
        // overall cup clumping
        let mean_dnn = dnn_mean(&cup, &cup);
        // (s/d_nn)³ mean (tail-sensitive) over a center column at the settled band
        let band = (ymin + 0.05, ymin + 0.05 + 2.0 * SPACING);
        let center: Vec<[f32; 4]> = cup
            .iter()
            .copied()
            .filter(|p| {
                (p[0] * p[0] + p[2] * p[2]).sqrt() < 1.0 && p[1] >= band.0 && p[1] <= band.1
            })
            .collect();
        let mut rho_tail = 0.0f64;
        for a in &center {
            let mut best = f32::INFINITY;
            for b in &cup {
                let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
                if d2 > 1e-9 {
                    best = best.min(d2);
                }
            }
            rho_tail += (SPACING / best.sqrt().max(1e-6)).powi(3) as f64;
        }
        rho_tail /= center.len().max(1) as f64;
        println!(
            "  blend={:.2}  cup_water={}  settled_depth={:.2} (seed 1.65)  mean_d_nn={:.4} ({:.2}× rest)  (s/d_nn)³_mean={:.1}×",
            blend,
            cup.len(),
            ymax - ymin,
            mean_dnn,
            mean_dnn / SPACING,
            rho_tail
        );
    }
    println!("==============================================================================\n");
}

#[test]
fn aligned_square_localize_overpack() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_wall_audit: no GPU adapter; skipping.");
        return;
    };

    let a = probe("A box floor + box walls", &box_scene(), &gpu, -5.0, false);
    let b = probe(
        "B box floor + SDF side walls",
        &side_wall_scene(),
        &gpu,
        -5.0,
        false,
    );
    let c = probe(
        "C SDF floor + SDF walls",
        &sdf_floor_scene(),
        &gpu,
        -4.8,
        true,
    );

    // Calibrate ρ_rest = box interior (the box is at rest per the original diagnosis).
    let rest = a.interior_nb.max(1.0);
    let r = |nb: f64| nb / rest;
    let fmt_nm = |m: &[f32; 8]| {
        format!(
            "xx={:.4} yy={:.4} zz={:.4} (massy={:.0})",
            m[0], m[3], m[5], m[6]
        )
    };

    println!("\n========== U2a OVER-PACK LOCALIZATION (ρ/ρ_rest, box interior = 1.0) ==========");
    println!(
        "  ρ_rest calibration (box interior neighbor count) = {:.2}\n",
        rest
    );
    for p in [&a, &b, &c] {
        println!(
            "[{}]\n  interior ρ/ρ_rest={:.3}  SIDE-wall ρ/ρ_rest={:.3}  FLOOR ρ/ρ_rest={:.3}",
            p.label,
            r(p.interior_nb),
            r(p.side_nb),
            r(p.floor_nb),
        );
        print!("  floor-node M̃⁻¹: {}", fmt_nm(&p.nm_floor));
        if p.floor_dist.is_nan() {
            println!("   (box-face floor)");
        } else {
            println!(
                "   SDF floor dist={:.4} wallish(dist<0)?={}",
                p.floor_dist, p.floor_wallish
            );
        }
    }
    println!("\n--- LOCALIZATION ---");
    println!(
        "  INTERIOR d_nn (rest spacing {:.3}) A/B/C = {:.4} / {:.4} / {:.4}  [control: is 0.068 normal?]",
        SPACING, a.interior_dnn, b.interior_dnn, c.interior_dnn
    );
    println!(
        "  CORNER ρ/ρ_rest   A/B/C = {:.3} / {:.3} / {:.3}  [box multi-axis vs SDF single-normal seam]",
        r(a.corner_nb),
        r(b.corner_nb),
        r(c.corner_nb)
    );
    println!(
        "  FLOOR ρ/ρ_rest    A/B/C = {:.3} / {:.3} / {:.3}",
        r(a.floor_nb),
        r(b.floor_nb),
        r(c.floor_nb)
    );
    println!(
        "  SIDE  ρ/ρ_rest    A/B/C = {:.3} / {:.3} / {:.3}",
        r(a.side_nb),
        r(b.side_nb),
        r(c.side_nb)
    );
    println!(
        "  INTERIOR ρ/ρ_rest A/B/C = {:.3} / {:.3} / {:.3}",
        r(a.interior_nb),
        r(b.interior_nb),
        r(c.interior_nb)
    );
    println!("================================================================================\n");

    // Reality check: does the DOCUMENTED v60_cup_static_full reproduce the 16-25× over-pack with
    // this same probe + ρ_rest? (If ~1×, the metric differs from the session's original probe.)
    let (cup_int, cup_wall, cup_floor, n) = probe_v60_cup(&gpu, rest);
    println!(
        "[REAL v60_cup_static_full] (same ρ_rest={:.2}, 400 frames, n_wall={})",
        rest, n
    );
    println!(
        "  interior ρ/ρ_rest={:.3}  WALL(r≈3) ρ/ρ_rest={:.3}  FLOOR(y≈-8) ρ/ρ_rest={:.3}",
        cup_int, cup_wall, cup_floor
    );
    println!("================================================================================\n");

    assert!(
        a.interior_nb > 0.0 && b.side_nb > 0.0 && c.floor_nb > 0.0,
        "no particles measured in a shell"
    );
}

// ---- U2: multi-normal operator-consistency gate -------------------------------------------------

/// CornerU3 corner-parity measurement: the SDF square cup floor∩wall corner ρ/ρ_rest in SINGLE
/// (bug, ~2.4×) vs MULTI (fix → should reach box parity ~1.4×), with the flat box as the live
/// reference. Print-first (calibrate-then-pin) + a two-sided parity assertion.
#[test]
fn corner_parity_multi_drops_to_box() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_wall_audit: no GPU adapter; skipping.");
        return;
    };
    let quiet = EmissionInput::default();
    let corner_and_interior = |scene: &Scene, floor: f32, mode: f32| -> (f64, f64) {
        let mut s = TwofieldSolver::build(scene, &web_mats(), &web_cfg(), &gpu);
        s.set_wall_bc_mode_for_test(mode);
        for _ in 0..SETTLE {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let nlive = s.phase_counts().0 as usize;
        let live = &pos[..nlive];
        let corner = mean_nb(
            live,
            (APOTHEM - 2.0 * SPACING, APOTHEM),
            (floor, floor + 2.0 * SPACING),
            0.6,
        )
        .0;
        let interior = mean_nb(live, (-0.6, 0.6), (floor + 0.8, floor + 1.8), 0.6).0;
        (corner, interior)
    };

    // Flat box reference (no solids → box-face BC; the mode is irrelevant there).
    let (box_corner, box_interior) = corner_and_interior(&box_scene(), -5.0, WALL_BC_SINGLE);
    let rest = box_interior.max(1.0);
    let box_c = box_corner / rest;
    // SDF SQUARE cup: single (the bug) vs multi (the fix).
    let sq_single = corner_and_interior(&sdf_floor_scene(), -4.8, WALL_BC_SINGLE).0 / rest;
    let sq_multi = corner_and_interior(&sdf_floor_scene(), -4.8, WALL_BC_MULTI).0 / rest;
    // ROUND (cylinder) cup: same orthogonal floor+radial seam.
    let rd_single = corner_and_interior(&cyl_cup_scene(), -4.8, WALL_BC_SINGLE).0 / rest;
    let rd_multi = corner_and_interior(&cyl_cup_scene(), -4.8, WALL_BC_MULTI).0 / rest;

    println!("CORNER ρ/ρ_rest:  box(ref)={box_c:.3}");
    println!("  square: SINGLE(bug)={sq_single:.3}  MULTI(fix)={sq_multi:.3}");
    println!("  round:  SINGLE(bug)={rd_single:.3}  MULTI(fix)={rd_multi:.3}");

    // Two-sided acceptance band for the MULTI corner: no over-pack (≤ box + margin) AND no
    // depletion/void (≥ void_floor, well below rest). The band is NOT centered on the box —
    // the box's own corner is mildly hydrostatically elevated (1.4×); a curved cup legitimately
    // settles to rest (~1.0) there, which is correct, not a void.
    let over_pack_cap = box_c + 0.35;
    let void_floor = 0.65;
    for (name, single, multi) in [
        ("square", sq_single, sq_multi),
        ("round", rd_single, rd_multi),
    ] {
        // The bug must be present in SINGLE mode (corner notably over the box).
        assert!(
            single > box_c + 0.3,
            "{name}: expected corner over-pack in SINGLE mode: {single:.3} vs box {box_c:.3}"
        );
        // MULTI removes the over-pack without carving a void, and improves on SINGLE.
        assert!(
            multi <= over_pack_cap && multi >= void_floor,
            "{name}: MULTI corner must land in [{void_floor:.2}, {over_pack_cap:.3}] (no over-pack, \
             no void): got {multi:.3}"
        );
        assert!(
            multi < single - 0.3,
            "{name}: MULTI must reduce the corner vs SINGLE: {multi:.3} vs {single:.3}"
        );
    }
}

fn cyl_cup_scene() -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 100.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-4.0, -8.0, -4.0],
        box_max: [4.0, TOP_Y, 4.0],
        regions: vec![slab(-4.8)],
        solids: vec![SdfPrimitive {
            kind: SolidKind::Cylinder {
                center: Vec3::ZERO,
                floor_y: -4.8,
                rim_y: TOP_Y,
                radius: APOTHEM,
            },
            species_mask: MASK_WATER,
            friction: 0.0,
        }],
    }
}

/// Octagon SDF cup: its vertical edges meet adjacent side faces at 45° → NON-orthogonal binding
/// normals (dot ≈ 0.707). A raw Σ n̂n̂ᵀ projector would be non-PSD there; the orthonormal-basis
/// P = I − QQᵀ must stay PSD.
fn octagon_cup_scene() -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 100.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-4.0, -8.0, -4.0],
        box_max: [4.0, TOP_Y, 4.0],
        regions: vec![slab(-4.8)],
        solids: vec![SdfPrimitive {
            kind: SolidKind::PolyCup {
                center: Vec3::ZERO,
                floor_y: -4.8,
                rim_y: TOP_Y,
                apothem: APOTHEM,
                sides: 8,
            },
            species_mask: MASK_WATER,
            friction: 0.0,
        }],
    }
}

/// Smallest eigenvalue of a symmetric 3×3 packed as (xx, xy, xz, yy, yz, zz) — Smith's closed form.
fn min_eig_sym3(m: [f32; 6]) -> f64 {
    let (xx, xy, xz, yy, yz, zz) = (
        m[0] as f64,
        m[1] as f64,
        m[2] as f64,
        m[3] as f64,
        m[4] as f64,
        m[5] as f64,
    );
    let p1 = xy * xy + xz * xz + yz * yz;
    if p1 == 0.0 {
        return xx.min(yy).min(zz);
    }
    let q = (xx + yy + zz) / 3.0;
    let p2 = (xx - q).powi(2) + (yy - q).powi(2) + (zz - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let (bxx, byy, bzz) = ((xx - q) / p, (yy - q) / p, (zz - q) / p);
    let (bxy, bxz, byz) = (xy / p, xz / p, yz / p);
    let detb = bxx * (byy * bzz - byz * byz) - bxy * (bxy * bzz - byz * bxz)
        + bxz * (bxy * byz - byy * bxz);
    let r = (detb / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    let e2 = 3.0 * q - e1 - e3;
    e1.min(e2).min(e3)
}

/// CornerU2 operator-consistency gate: in WALL_BC_MULTI mode every per-node M̃⁻¹ stays symmetric
/// POSITIVE-SEMIDEFINITE — even at the octagon's non-orthogonal vertical edges. This is the guard
/// that the orthonormal basis (not a raw dyad sum) is used; a `Σ n̂n̂ᵀ` projector fails here
/// (relative min eigenvalue ≈ −0.7). PSD M̃⁻¹ ⇒ A = D·M̃⁻¹·Dᵀ stays SPD (operator consistency).
#[test]
fn multi_normal_nm_is_psd_on_octagon_cup() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_wall_audit: no GPU adapter; skipping.");
        return;
    };
    let mut s = TwofieldSolver::build(&octagon_cup_scene(), &web_mats(), &web_cfg(), &gpu);
    s.set_wall_bc_mode_for_test(WALL_BC_MULTI);
    let quiet = EmissionInput::default();
    for _ in 0..60 {
        s.step(DT, &quiet);
    }
    let nm = s.read_node_matrices();
    // Global matrix scale (≈ sig·invr ~ 1e-2). Fully-constrained nodes (P=0 ⇒ M̃⁻¹=0, e.g. floor +
    // two octagon side faces spanning R³, like a box bottom-corner) are legitimately ~0; judge PSD
    // by an ABSOLUTE eigenvalue floor scaled by this global scale, NOT a per-node relative ratio
    // (which would amplify float noise at the zero nodes).
    let mut global_scale = 0.0f64;
    for m in &nm {
        global_scale = global_scale.max(m[0].abs().max(m[3].abs()).max(m[5].abs()) as f64);
    }
    let mut min_eig = f64::INFINITY;
    for m in &nm {
        min_eig = min_eig.min(min_eig_sym3([m[0], m[1], m[2], m[3], m[4], m[5]]));
    }
    let floor = -1.0e-3 * global_scale;
    println!(
        "octagon WALL_BC_MULTI: min eigenvalue of M̃⁻¹ = {min_eig:.3e} (global scale {global_scale:.3e}, PSD floor {floor:.3e})"
    );
    assert!(
        min_eig > floor,
        "M̃⁻¹ went non-PSD under non-orthogonal octagon-edge normals (min eig {min_eig:.3e} < {floor:.3e}) \
         — the projector must be orthonormal I−QQᵀ, not a raw Σ n̂n̂ᵀ"
    );
}

/// DensU4 R9 — the durable over-pack regression gate for the uncapped two-sided density relief
/// (plan 2026-06-17-003). Two bars, kept SEPARATE (do NOT collapse / over-tighten the wall):
///   (a) clean-model bar: cup INTERIOR ρ/ρ_rest ≲ 1.2 (the strong incompressibility claim);
///   (b) V60 WALL-shell regression bar: the fix must materially beat legacy (wall < legacy − 0.3),
///       but is NOT held to ≲1.2 — the residual wall over-pack is partly the deferred corner-seam
///       BC (this runs WALL_BC_SINGLE), not the model (see R11 / the corner follow-up).
/// Sanity: the clean-model bar genuinely separates fix from legacy (legacy interior > 1.2).
/// ρ_rest = a flat box interior at rest (box-interior = 1.0 calibration).
#[test]
fn uncapped_density_reduces_overpack() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_wall_audit: no GPU adapter; skipping.");
        return;
    };
    let quiet = EmissionInput::default();
    let rest = {
        let mut s = TwofieldSolver::build(&box_scene(), &web_mats(), &web_cfg(), &gpu);
        for _ in 0..SETTLE {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let n = s.phase_counts().0 as usize;
        mean_nb(&pos[..n], (-0.6, 0.6), (-4.0, -3.0), 0.6)
            .0
            .max(1.0)
    };
    let measure = |uncapped: bool| -> (f64, f64) {
        let mut s =
            TwofieldSolver::build(&Scene::v60_cup_static_full(), &web_mats(), &web_cfg(), &gpu);
        if uncapped {
            s.set_density_target_mode_for_test(true);
        }
        for _ in 0..400 {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let nlive = s.phase_counts().0 as usize;
        let cup: Vec<[f32; 4]> = pos[..nlive]
            .iter()
            .copied()
            .filter(|p| p[1] <= -3.5 && p[1] >= -8.5)
            .collect();
        let ymin = cup.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let band = (ymin + 0.05, ymin + 0.05 + 2.0 * SPACING);
        let wall = mean_nb_radial(&cup, 3.0 - 2.0 * SPACING, 3.0, band).0 / rest;
        let interior = mean_nb_radial(&cup, 0.0, 1.0, band).0 / rest;
        (wall, interior)
    };
    let (lw, li) = measure(false);
    let (uw, ui) = measure(true);
    println!(
        "DensU4 R9 over-pack (ρ_rest={rest:.2}): legacy wall {lw:.2}/interior {li:.2}  →  \
         uncapped wall {uw:.2}/interior {ui:.2}"
    );
    // (a) clean-model bar: interior near rest, and it genuinely separates fix from legacy.
    assert!(
        ui <= 1.20,
        "clean-model bar: uncapped interior ρ/ρ_rest {ui:.2} must be ≲ 1.20 (near rest)"
    );
    assert!(
        li > 1.20,
        "sanity: legacy interior {li:.2} should exceed the 1.20 bar (else the bar is vacuous)"
    );
    // (b) V60 wall-shell regression bar: fix materially beats legacy; NOT held to ≲1.2 (corner BC).
    assert!(
        uw < li.max(lw) && uw < lw - 0.3,
        "wall-shell regression: uncapped wall {uw:.2} must materially beat legacy wall {lw:.2} (by ≥ 0.3)"
    );
    assert!(
        uw <= 1.55,
        "wall-shell bar: uncapped wall {uw:.2} ≤ 1.55 (residual over 1.0 is the deferred corner BC, \
         not the model — do NOT tighten to ≲1.2)"
    );
}

/// DensU5 TEMPER-K sweep — cup over-pack (crispness) vs the uncapped rate-cap K, the partner to
/// `twofield_settled::temper_k_sweep_settled_agitation`. The over-pack should improve as K→1
/// (stiffer); the settled sweep showed K≥3 stays quiet while K=1 agitates. This finds the LARGEST
/// K (gentlest/quietest) that still de-mushes the cup (interior near rest) — the temper candidate
/// to eyeball in the webapp. ρ_rest = flat box-interior at rest (box-interior = 1.0).
/// `cargo test --release --test twofield_wall_audit temper_k_sweep_overpack -- --ignored --nocapture`
#[test]
#[ignore = "on-demand temper-K cup over-pack (crispness) sweep"]
fn temper_k_sweep_overpack() {
    let Some(gpu) = GpuContext::new_headless() else {
        return;
    };
    let quiet = EmissionInput::default();
    let rest = {
        let mut s = TwofieldSolver::build(&box_scene(), &web_mats(), &web_cfg(), &gpu);
        for _ in 0..SETTLE {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let n = s.phase_counts().0 as usize;
        mean_nb(&pos[..n], (-0.6, 0.6), (-4.0, -3.0), 0.6)
            .0
            .max(1.0)
    };
    let measure = |label: &str, uncapped: bool, k: f32| {
        let mut s =
            TwofieldSolver::build(&Scene::v60_cup_static_full(), &web_mats(), &web_cfg(), &gpu);
        if uncapped {
            s.set_density_target_mode_for_test(true);
            s.set_density_rate_k_for_test(k);
        }
        for _ in 0..400 {
            s.step(DT, &quiet);
        }
        let pos = s.read_positions();
        let nlive = s.phase_counts().0 as usize;
        let cup: Vec<[f32; 4]> = pos[..nlive]
            .iter()
            .copied()
            .filter(|p| p[1] <= -3.5 && p[1] >= -8.5)
            .collect();
        let ymin = cup.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let band = (ymin + 0.05, ymin + 0.05 + 2.0 * SPACING);
        let wall = mean_nb_radial(&cup, 3.0 - 2.0 * SPACING, 3.0, band).0 / rest;
        let interior = mean_nb_radial(&cup, 0.0, 1.0, band).0 / rest;
        println!("  [{label:<18}] interior ρ/ρ_rest {interior:.2}  wall {wall:.2}  (rest 1.0; mushy = high interior)");
    };
    println!("\n==== TEMPER-K vs CUP OVER-PACK / CRISPNESS (v60_cup_static_full, 400f, ρ_rest={rest:.2}) ====");
    println!("  smaller K = stiffer/crisper (interior → rest); legacy is mushy (interior ~1.3)");
    measure("legacy (capped)", false, 0.0);
    measure("uncapped K=1", true, 1.0);
    measure("uncapped K=3", true, 3.0);
    measure("uncapped K=5", true, 5.0);
    measure("uncapped K=10", true, 10.0);
    measure("uncapped K=30", true, 30.0); // sanity: ≈ legacy
    println!(
        "  → cross with the settled sweep: K≥3 is QUIET; pick the smallest-still-quiet K that"
    );
    println!(
        "    de-mushes most (interior nearest rest). Candidate ≈ K=3. Eyeball it in the webapp."
    );
}

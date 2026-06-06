//! Water-phase static validation against closed forms. GPU-gated; skips without
//! an adapter.
//!
//! - `rest_lattice_is_a_quiet_fixed_point_across_resolutions`: a block seeded on
//!   the exact lattice that DEFINES ρ₀ reads C ≈ 0 and, under zero gravity, stays
//!   at rest — at several resolutions. Encodes the fine-spacing-eruption lesson
//!   (commit 93eb9ba): the rest residual must not blow up as the lattice refines.

mod common;

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::kernels;
use coffee_sim::EmissionInput;

use common::reconstruct_density;

fn speed(v: [f32; 4]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn centroid(pos: &[[f32; 4]]) -> [f32; 3] {
    let mut c = [0.0f64; 3];
    for p in pos {
        for k in 0..3 {
            c[k] += p[k] as f64;
        }
    }
    let n = pos.len() as f64;
    [(c[0] / n) as f32, (c[1] / n) as f32, (c[2] / n) as f32]
}

/// Self-similar materials at a given spacing: support radius scales with spacing
/// (h = 2·spacing) so the rest lattice has the same neighbourhood topology at
/// every resolution — the meaningful refinement axis for the eruption lesson.
fn materials_at_spacing(spacing: f32) -> Materials {
    Materials {
        particle_spacing: spacing,
        support_radius: 2.0 * spacing,
        ..Materials::default()
    }
}

fn rest_block_scene(spacing: f32, n_cells: i32) -> Scene {
    let hi = n_cells as f32 * spacing;
    let pad = 4.0 * spacing;
    Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, 0.0, 0.0],
        box_min: [-pad, -pad, -pad],
        box_max: [hi + pad, hi + pad, hi + pad],
        regions: vec![SeedRegion {
            min: [0.0, 0.0, 0.0],
            max: [hi, hi, hi],
            species: Species::Water,
        }],
        solids: Vec::new(),
    }
}

#[test]
fn rest_lattice_is_a_quiet_fixed_point_across_resolutions() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_statics: no GPU adapter; skipping.");
        return;
    };

    let spacings = [1.25f32, 1.0, 0.8]; // coarse → fine
    let n_cells = 6; // 7³ lattice; interior (full ±1 neighbourhood = 26) = indices 1..5
    let interior_min_neighbours = 24usize;
    let frames = 120u32;

    let mut residual_by_spacing: Vec<(f32, f32)> = Vec::new(); // (spacing, max interior |C| after run)

    for &s in &spacings {
        let mats = materials_at_spacing(s);
        let h = mats.support_radius;
        let m = mats.particle_mass;
        let rho0 = kernels::rest_density(s, h, m);
        let scene = rest_block_scene(s, n_cells);
        let cfg = Config {
            seed_jitter: 0.0,
            ..Config::default()
        };
        let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        let input = EmissionInput::default();

        // Frame 0: interior reconstructed density == ρ₀ ⇒ C ≈ 0 by construction
        // (this is the cross-check that rest_density and the kernel agree).
        let pos0 = solver.read_positions();
        let dens0 = reconstruct_density(&pos0, h, m);
        let interior0: Vec<f32> = dens0
            .iter()
            .filter(|(_, nbr)| *nbr >= interior_min_neighbours)
            .map(|(d, _)| d / rho0 - 1.0)
            .collect();
        assert!(
            !interior0.is_empty(),
            "no interior particles at spacing {s}"
        );
        let max_c0 = interior0.iter().map(|c| c.abs()).fold(0.0, f32::max);

        let centroid0 = centroid(&pos0);
        for _ in 0..frames {
            solver.step(1.0 / 60.0, &input);
        }
        let pos1 = solver.read_positions();
        let vel1 = solver.read_velocities();
        let dens1 = reconstruct_density(&pos1, h, m);

        let mut max_c = 0.0f32;
        let mut max_v = 0.0f32;
        for (i, (d, nbr)) in dens1.iter().enumerate() {
            if *nbr >= interior_min_neighbours {
                max_c = max_c.max((d / rho0 - 1.0).abs());
                max_v = max_v.max(speed(vel1[i]));
            }
        }
        let drift = {
            let c1 = centroid(&pos1);
            let d = [
                c1[0] - centroid0[0],
                c1[1] - centroid0[1],
                c1[2] - centroid0[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };

        // Frame 0 is the kernel/ρ₀ cross-check: an interior rest-lattice site sums
        // to exactly ρ₀, so C is zero to float precision.
        assert!(
            max_c0 < 1e-5,
            "spacing {s}: rest lattice not at C≈0: {max_c0}"
        );
        // After 120 zero-gravity steps the lattice is a quiet fixed point. Observed
        // worst across resolutions: maxV ~2.5e-4, drift ~1.6e-4, maxC ~1.1e-3 — the
        // bounds below sit ~5–8× above that, far below any real instability (the
        // 93eb9ba eruption hit vmax ~1e5).
        assert!(
            max_v < 2e-3,
            "spacing {s}: spurious velocity {max_v} (lattice should stay at rest)"
        );
        assert!(drift < 2e-3, "spacing {s}: block drifted {drift}");
        assert!(
            max_c < 5e-3,
            "spacing {s}: interior density drifted from ρ₀: max|C|={max_c}"
        );

        residual_by_spacing.push((s, max_c));
    }

    // Spacing-invariance (the eruption-bug lesson): refining the lattice must not
    // amplify the rest residual. A 1/h³-style normalization bug would blow this up
    // by orders of magnitude; the real solver grows it only ~2.5× over a 1.56×
    // refinement, so a 4× factor (plus a small floor) is a real, non-trivial guard.
    let coarsest = residual_by_spacing[0].1;
    for &(s, c) in &residual_by_spacing {
        assert!(
            c <= coarsest * 4.0 + 1e-3,
            "spacing {s}: residual {c} grew vs coarsest {coarsest} under refinement"
        );
    }
}

fn column_scene(gravity_y: f32) -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, gravity_y, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [10.0, 30.0, 10.0],
        regions: vec![SeedRegion {
            min: [0.5, 0.5, 0.5],
            max: [9.5, 24.0, 9.5],
            species: Species::Water,
        }],
        solids: Vec::new(),
    }
}

/// Bin interior particles by depth and return the per-bin mean density excess
/// `mean(ρ/ρ₀ − 1)`, ordered bottom → top, for bins with enough samples.
fn depth_profile(pos: &[[f32; 4]], dens: &[(f32, usize)], rho0: f32, bins: usize) -> Vec<f32> {
    let interior: Vec<(f32, f32)> = pos
        .iter()
        .zip(dens)
        .filter(|(p, (_, nbr))| *nbr >= 24 && p[0] > 2.5 && p[0] < 7.5 && p[2] > 2.5 && p[2] < 7.5)
        .map(|(p, (d, _))| (p[1], d / rho0 - 1.0))
        .collect();
    let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
    for &(y, _) in &interior {
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    let span = (ymax - ymin).max(1e-3);
    let mut sum = vec![0.0f32; bins];
    let mut cnt = vec![0usize; bins];
    for &(y, c) in &interior {
        let b = (((y - ymin) / span) * bins as f32).floor() as usize;
        let b = b.min(bins - 1);
        sum[b] += c;
        cnt[b] += 1;
    }
    (0..bins)
        .filter(|&b| cnt[b] >= 8)
        .map(|b| sum[b] / cnt[b] as f32)
        .collect()
}

/// Under hydrostatic load the fluid stays incompressible: ρ(z) = ρ₀ at every
/// depth (the closed-form incompressible limit), even at 2× gravity, where a
/// compressible fluid would stratify as ρ ∝ g·depth. This is the testable
/// replacement for "pressure ∝ depth": PBF carries the hydrostatic balance in
/// the implicit constraint (no readable pressure / density gradient of the
/// magnitude P=ρgh would imply), so what is observable is that the density holds
/// ρ₀ under load, with only a sub-percent residual compression in the correct
/// direction. Returns `(max depth-bin |C|, bottom−top ΔC)` per gravity.
#[test]
fn fluid_stays_incompressible_under_hydrostatic_load() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_statics: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let h = mats.support_radius;
    let m = mats.particle_mass;
    let rho0 = kernels::rest_density(mats.particle_spacing, h, m);
    let cfg = Config::default();

    let probe = |g: f32| -> (f32, f32) {
        let scene = column_scene(g);
        let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        let input = EmissionInput::default();
        for _ in 0..400 {
            solver.step(1.0 / 60.0, &input);
        }
        let pos = solver.read_positions();
        let dens = reconstruct_density(&pos, h, m);
        let prof = depth_profile(&pos, &dens, rho0, 6);
        assert!(
            prof.len() >= 4,
            "g={g}: too few depth bins ({})",
            prof.len()
        );
        let max_abs = prof.iter().map(|c| c.abs()).fold(0.0, f32::max);
        let bottom_top = prof[0] - prof[prof.len() - 1];
        (max_abs, bottom_top)
    };

    let (max_c_1g, dc_1g) = probe(-20.0);
    let (max_c_2g, dc_2g) = probe(-40.0);

    // (1) Incompressible at every depth, at 1g and 2g: ρ stays within ~1% of ρ₀
    //     under load (observed ~0.3% at 1g, ~0.76% at 2g). A compressible fluid
    //     would show order-of-magnitude-larger stratification.
    assert!(
        max_c_1g < 1.5e-2,
        "1g: fluid compressed under load, max depth |C| = {max_c_1g}"
    );
    assert!(
        max_c_2g < 1.5e-2,
        "2g: fluid compressed under load, max depth |C| = {max_c_2g}"
    );

    // (2) The sub-percent residual is genuine hydrostatic compression: denser at
    //     depth (bottom − top > 0), and larger under stronger gravity — confirming
    //     the tiny gradient tracks the load rather than being noise.
    assert!(
        dc_1g > 5e-4,
        "1g: no hydrostatic compression structure (ΔC = {dc_1g})"
    );
    assert!(
        dc_2g > 5e-4,
        "2g: no hydrostatic compression structure (ΔC = {dc_2g})"
    );
    assert!(
        dc_2g > dc_1g,
        "doubling gravity did not increase compression (1g ΔC = {dc_1g}, 2g ΔC = {dc_2g})"
    );
}

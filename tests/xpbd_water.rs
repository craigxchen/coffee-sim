//! PBF water-core gate (GPU-gated; skips without an adapter).
//!
//! Verifies the dam-break is stable, incompressible, clump-free, and settles — the v1
//! water-core acceptance gate from `docs/plans/solver_xpbd.md`.

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::kernels;
use coffee_sim::EmissionInput;

fn max_speed(v: &[[f32; 4]]) -> f32 {
    v.iter()
        .map(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt())
        .fold(0.0, f32::max)
}

fn dist2(a: [f32; 4], b: [f32; 4]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Min nearest-neighbor distance over a strided sample (clumping detector).
fn min_nn_sample(pos: &[[f32; 4]], stride: usize) -> f32 {
    let mut min_nn = f32::INFINITY;
    let mut i = 0;
    while i < pos.len() {
        let mut nn = f32::INFINITY;
        for (j, &p) in pos.iter().enumerate() {
            if j != i {
                nn = nn.min(dist2(pos[i], p));
            }
        }
        min_nn = min_nn.min(nn);
        i += stride;
    }
    min_nn.sqrt()
}

/// (#interior in density band, #interior) over a strided sample. Interior = particles
/// with a full-ish neighborhood (excludes the free surface, where deficit is physical).
fn density_band(pos: &[[f32; 4]], h: f32, m: f32, rho0: f32, stride: usize) -> (usize, usize) {
    let lo = 0.85 * rho0;
    let hi = 1.15 * rho0;
    let mut in_band = 0;
    let mut interior = 0;
    let mut i = 0;
    while i < pos.len() {
        let mut rho = m * kernels::w_poly6(0.0, h);
        let mut neighbors = 0;
        for (j, &p) in pos.iter().enumerate() {
            if j != i {
                let r = dist2(pos[i], p).sqrt();
                if r < h {
                    rho += m * kernels::w_poly6(r, h);
                    neighbors += 1;
                }
            }
        }
        // Interior heuristic: enough neighbors that it's not a surface/edge particle.
        if neighbors >= 24 {
            interior += 1;
            if rho >= lo && rho <= hi {
                in_band += 1;
            }
        }
        i += stride;
    }
    (in_band, interior)
}

fn finite(pts: &[[f32; 4]]) -> bool {
    pts.iter()
        .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite())
}

#[test]
fn dam_break_stable_incompressible_clumpfree_and_settles() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_water: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::dam_break();
    let mats = Materials::default();
    let cfg = Config::default();
    let rho0 = kernels::rest_density(
        mats.particle_spacing,
        mats.support_radius,
        mats.particle_mass,
    );
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let n = solver.particles().particle_count;
    assert!(n > 1000, "expected a few thousand particles, got {n}");

    let input = EmissionInput::default();
    let frames = 360u32;
    let sample_every = 40u32;
    let mut vmax_early = 0.0f32;
    let mut vmax_late = 0.0f32;

    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % sample_every == 0 || f == frames - 1 {
            solver.sample_diagnostics();
            let pos = solver.read_positions();
            let vel = solver.read_velocities();
            assert!(finite(&pos), "non-finite position at frame {f}");
            assert!(finite(&vel), "non-finite velocity at frame {f}");

            let diag = solver.diagnostics();
            assert!(!diag.overflow, "grid bucket overflow at frame {f}");
            assert!(
                diag.max_occupancy < cfg.bucket_capacity,
                "occupancy {} >= K at frame {f}",
                diag.max_occupancy
            );

            // no leak (small margin for the clamp boundary)
            let inside = |v: f32, lo: f32, hi: f32| v >= lo - 0.6 && v <= hi + 0.6;
            for p in &pos {
                assert!(
                    inside(p[0], scene.box_min[0], scene.box_max[0])
                        && inside(p[1], scene.box_min[1], scene.box_max[1])
                        && inside(p[2], scene.box_min[2], scene.box_max[2]),
                    "particle left the box at frame {f}: {p:?}"
                );
            }

            let vmax = max_speed(&vel);
            if f == 0 {
                vmax_early = vmax;
            }
            vmax_late = vmax;
            eprintln!(
                "f{f}: vmax {vmax:.2} occ {} iters {}",
                diag.max_occupancy, diag.effective_iters
            );
        }
    }

    // Settling: late kinetic energy is a small fraction of the violent early phase.
    assert!(
        vmax_late < 0.3 * vmax_early,
        "did not settle: vmax_early {vmax_early:.2} vmax_late {vmax_late:.2}"
    );

    // Final-frame structure checks.
    let pos = solver.read_positions();
    let nn = min_nn_sample(&pos, 17);
    let (in_band, interior) = density_band(&pos, mats.support_radius, mats.particle_mass, rho0, 17);
    let frac = if interior > 0 {
        in_band as f32 / interior as f32
    } else {
        0.0
    };
    eprintln!(
        "final: nn {nn:.3} (spacing {}), density in-band {in_band}/{interior} = {:.2}, rho0 {rho0:.4}",
        mats.particle_spacing, frac
    );

    // No clumping: nearest neighbors keep their spacing.
    assert!(
        nn >= 0.4 * mats.particle_spacing,
        "clumping: min nearest-neighbor {nn:.3} < 0.4·spacing"
    );
    // Incompressibility: most interior particles sit within the density band.
    assert!(
        interior > 50,
        "too few interior samples ({interior}) to judge density"
    );
    assert!(
        frac >= 0.75,
        "incompressibility: only {:.0}% of interior in band",
        frac * 100.0
    );
}

#[test]
fn same_seed_is_reproducible_short_run() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_water: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::dam_break();
    let mats = Materials::default();
    let cfg = Config::default();
    let input = EmissionInput::default();

    let run = |gpu: &GpuContext| {
        let mut s = XpbdSolver::build(&scene, &mats, &cfg, gpu);
        for _ in 0..10 {
            s.step(1.0 / 60.0, &input);
        }
        s.read_positions()
    };
    let a = run(&gpu);
    let b = run(&gpu);
    assert_eq!(a.len(), b.len());
    let mean_diff: f32 = a
        .iter()
        .zip(&b)
        .map(|(p, q)| dist2(*p, *q).sqrt())
        .sum::<f32>()
        / a.len() as f32;
    eprintln!("reproducibility mean per-particle diff after 10 frames: {mean_diff:.5}");
    // Tolerance-level reproducibility (GPU fp reductions may reorder).
    assert!(
        mean_diff < 0.05,
        "not reproducible: mean diff {mean_diff:.5}"
    );
}

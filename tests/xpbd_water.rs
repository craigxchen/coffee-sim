//! PBF water-core invariants (GPU-gated; skips without an adapter).
//!
//! One long dam-break run that asserts the physical invariants that actually matter — not
//! tuned target values. If the solver changes, these should still hold (or reveal a real
//! regression), rather than needing constant re-tuning.

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::kernels;
use coffee_sim::EmissionInput;

fn speed(v: [f32; 4]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dist2(a: [f32; 4], b: [f32; 4]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

fn finite(p: &[[f32; 4]]) -> bool {
    p.iter()
        .all(|q| q[0].is_finite() && q[1].is_finite() && q[2].is_finite())
}

/// Min nearest-neighbor distance over a strided sample (clumping detector).
fn min_nn(pos: &[[f32; 4]], stride: usize) -> f32 {
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

/// (#interior particles within the density band, #interior) over a strided sample.
/// "Interior" = a full-ish neighborhood, so the free surface (physically deficient) is excluded.
fn density_in_band(pos: &[[f32; 4]], h: f32, m: f32, rho0: f32, stride: usize) -> (usize, usize) {
    let (lo, hi) = (0.85 * rho0, 1.15 * rho0);
    let (mut ok, mut interior) = (0, 0);
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
        if neighbors >= 24 {
            interior += 1;
            if rho >= lo && rho <= hi {
                ok += 1;
            }
        }
        i += stride;
    }
    (ok, interior)
}

#[test]
fn dam_break_holds_water_invariants() {
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
    let n = solver.particles().particle_count as usize;
    assert!(n > 1000, "expected a few thousand particles, got {n}");
    let input = EmissionInput::default();

    // Long enough to catch the slow-accumulation global eruption (it detonated ~f1300).
    let frames = 1500u32;
    let mut collapse_peak = 0.0f32;
    let mut settled = 0.0f32;

    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 15 != 0 && f != frames - 1 {
            continue;
        }
        solver.sample_diagnostics();
        let vel = solver.read_velocities();
        let pos = solver.read_positions();

        // Always-true invariants, every sampled frame.
        assert!(
            finite(&pos) && finite(&vel),
            "non-finite state at frame {f}"
        );
        assert!(!solver.diagnostics().overflow, "grid overflow at frame {f}");
        let in_box = |p: &[f32; 4]| {
            (0..3).all(|a| p[a] >= scene.box_min[a] - 0.6 && p[a] <= scene.box_max[a] + 0.6)
        };
        assert!(pos.iter().all(in_box), "particle left the box at frame {f}");

        let vmax = vel.iter().map(|&v| speed(v)).fold(0.0, f32::max);
        if f < 120 {
            collapse_peak = collapse_peak.max(vmax);
        } else {
            // No global eruption: a large fraction of the fluid never goes fast at once.
            let fast = vel.iter().filter(|&&v| speed(v) > 15.0).count();
            assert!(
                fast < n / 20,
                "global eruption at frame {f}: {fast}/{n} fast"
            );
        }
        settled = vmax;
    }

    let pos = solver.read_positions();
    let nn = min_nn(&pos, 17);
    let (ok, interior) = density_in_band(&pos, mats.support_radius, mats.particle_mass, rho0, 17);
    eprintln!(
        "collapse_peak {collapse_peak:.1}, settled {settled:.2}, nn {nn:.2}, density in-band {ok}/{interior}"
    );

    // It fell with energy, then came to rest (not bouncy, not over-damped to a standstill mid-fall).
    assert!(
        collapse_peak > 8.0,
        "dam never gained momentum ({collapse_peak:.1})"
    );
    assert!(
        settled < 0.25 * collapse_peak,
        "did not settle ({settled:.2})"
    );
    // Incompressible: most interior particles sit in the density band.
    assert!(interior > 40, "too few interior samples ({interior})");
    assert!(
        ok * 100 / interior >= 80,
        "incompressibility: {ok}/{interior} interior in band"
    );
    // No clumping: nearest neighbors keep their spacing.
    assert!(nn > 0.4 * mats.particle_spacing, "clumping: nn {nn:.2}");
}

#[test]
fn same_seed_is_reproducible() {
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
    let (a, b) = (run(&gpu), run(&gpu));
    // The cell-order reorder permutes slot assignment non-deterministically (atomic scatter), so a
    // per-index comparison is meaningless. Reproducibility is a SET property: every particle in run
    // A must have a near-coincident counterpart in run B. Mean nearest-neighbor distance — robust to
    // slot permutation. (The residual is the pre-existing float-summation-order divergence, present
    // with or without the reorder; the reorder only adds the slot shuffle this metric sees through.)
    let mean_nn: f32 = a
        .iter()
        .map(|p| {
            b.iter()
                .map(|q| dist2(*p, *q))
                .fold(f32::INFINITY, f32::min)
                .sqrt()
        })
        .sum::<f32>()
        / a.len() as f32;
    eprintln!("reproducibility mean nearest-neighbor diff after 10 frames: {mean_nn:.5}");
    assert!(
        mean_nn < 0.05,
        "not reproducible as a set: mean nn diff {mean_nn:.5}"
    );
}

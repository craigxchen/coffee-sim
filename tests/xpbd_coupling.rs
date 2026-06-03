//! Water/bed coupling invariants (GPU-gated; skips without an adapter).
//!
//! Step 1 (grain exclusion + pore-modulated water density; drag/buoyancy come later): a water
//! column poured onto a grain bed. The invariant is that water threads the **pores** without
//! passing through grain **bodies** (no tunneling), the bed isn't crushed, and nothing blows up.
//! Water percolating to the floor is expected — drag (step 2) is what resists it into a realistic
//! drawdown; here there's no resistance, so water simply drains through the pore network.

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

fn speed(v: [f32; 4]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn finite(p: &[[f32; 4]]) -> bool {
    p.iter()
        .all(|q| q[0].is_finite() && q[1].is_finite() && q[2].is_finite())
}

fn dist(a: [f32; 4], b: [f32; 4]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Closest approach between any sampled water particle and any sampled grain (strided for speed).
fn min_water_grain(pos: &[[f32; 4]], phase: &[u32]) -> f32 {
    let waters: Vec<[f32; 4]> = pos
        .iter()
        .zip(phase)
        .enumerate()
        .filter(|(i, (_, &p))| p == 0 && i % 8 == 0)
        .map(|(_, (p, _))| *p)
        .collect();
    let grains: Vec<[f32; 4]> = pos
        .iter()
        .zip(phase)
        .enumerate()
        .filter(|(i, (_, &p))| p == 1 && i % 4 == 0)
        .map(|(_, (p, _))| *p)
        .collect();
    let mut m = f32::INFINITY;
    for w in &waters {
        for g in &grains {
            m = m.min(dist(*w, *g));
        }
    }
    m
}

#[test]
fn water_threads_the_bed_without_tunneling() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_coupling: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::pour_over();
    let mats = Materials::default();
    let cfg = Config::default();
    let d_wg = mats.grain_diameter;
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    let n = phase.len();
    let n_water = phase.iter().filter(|&&p| p == 0).count();
    let n_grain = n - n_water;
    assert!(
        n_water > 500 && n_grain > 500,
        "expected a mixed scene ({n_water}w/{n_grain}g)"
    );
    let input = EmissionInput::default();

    let frames = 600u32;
    let mut min_wg = f32::INFINITY;
    let mut max_grain_top = 0.0f32;

    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 30 != 0 && f != frames - 1 {
            continue;
        }
        solver.sample_diagnostics();
        let pos = solver.read_positions();
        let vel = solver.read_velocities();

        assert!(
            finite(&pos) && finite(&vel),
            "non-finite state at frame {f}"
        );
        assert!(!solver.diagnostics().overflow, "grid overflow at frame {f}");
        let in_box = |p: &[f32; 4]| {
            (0..3).all(|a| p[a] >= scene.box_min[a] - 0.6 && p[a] <= scene.box_max[a] + 0.6)
        };
        assert!(pos.iter().all(in_box), "particle left the box at frame {f}");

        // Closest water–grain approach (after the impact transient settles).
        if f >= 120 {
            min_wg = min_wg.min(min_water_grain(&pos, &phase));
        }
        // Bed top (highest grain) — the bed must not be crushed flat by the water load.
        let gtop = pos
            .iter()
            .zip(&phase)
            .filter(|(_, &p)| p == 1)
            .map(|(p, _)| p[1])
            .fold(0.0, f32::max);
        max_grain_top = max_grain_top.max(gtop);
    }

    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    let vmax = vel.iter().map(|&v| speed(v)).fold(0.0, f32::max);
    let gtop = pos
        .iter()
        .zip(&phase)
        .filter(|(_, &p)| p == 1)
        .map(|(p, _)| p[1])
        .fold(0.0, f32::max);
    eprintln!(
        "min water–grain {min_wg:.3} (d_wg {d_wg:.2}), bed top {gtop:.1} (peak {max_grain_top:.1}), vmax {vmax:.2}"
    );

    // No tunneling: water never passes through a grain body (stays near the contact distance, not
    // collapsed to ~0). The exclusion is a soft constraint, so allow a modest overlap margin.
    assert!(
        min_wg > 0.55 * d_wg,
        "water tunneled through grains: min water–grain {min_wg:.3} < {:.3}",
        0.55 * d_wg
    );
    // Bed not crushed: the grain column keeps real height (not flattened to a monolayer).
    assert!(
        max_grain_top > 4.0,
        "bed crushed flat (peak grain top {max_grain_top:.1})"
    );
    // No blow-up: the coupled sim stays bounded.
    assert!(vmax < 25.0, "coupled sim unstable (vmax {vmax:.2})");
}

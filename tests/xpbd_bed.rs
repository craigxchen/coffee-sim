//! Dry granular-bed invariants (GPU-gated; skips without an adapter).
//!
//! One long bed-drop run that asserts the *solid* behavior that separates a granular bed from a
//! liquid: it falls, flows, and settles into a stable 3D heap that holds **static with ~zero
//! drift** (a liquid would spread flat). The angle of repose is printed but asserted only in a
//! loose band — spheres-plus-friction give *a* believable repose, not an exact one; the hard gate
//! is stability (no NaN, bounded penetration, a real heap, static hold, freeze-when-static).

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

/// Horizontal centroid (x, z) of the particles — for the no-drift check.
fn centroid_xz(pos: &[[f32; 4]]) -> (f32, f32) {
    let n = pos.len().max(1) as f32;
    let (sx, sz) = pos
        .iter()
        .fold((0.0f32, 0.0f32), |(ax, az), p| (ax + p[0], az + p[2]));
    (sx / n, sz / n)
}

/// Peak height above the floor and base radius about the horizontal centroid.
fn heap_shape(pos: &[[f32; 4]], floor_y: f32) -> (f32, f32) {
    let (cx, cz) = centroid_xz(pos);
    let height = pos.iter().map(|p| p[1] - floor_y).fold(0.0, f32::max);
    let radius = pos
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[2] - cz).powi(2)).sqrt())
        .fold(0.0, f32::max);
    (height, radius)
}

#[test]
fn bed_drop_settles_into_static_heap() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_bed: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::bed_drop();
    let mats = Materials::default();
    let cfg = Config::default();
    let floor_y = scene.box_min[1];
    let s = mats.particle_spacing;
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let n = solver.particles().particle_count as usize;
    assert!(n > 1000, "expected a couple thousand grains, got {n}");
    let input = EmissionInput::default();

    // Settles by ~f420; run well past that to catch any slow creep or late blow-up.
    let frames = 720u32;
    let mut collapse_peak = 0.0f32;
    let mut max_pen = 0.0f32;
    let mut falling_frozen = 0u32; // frozen count mid-fall (should be ~0)
    let mut falling_vmax = 0.0f32; // proves grains are genuinely moving then
    let mut centroid_mid: Option<(f32, f32)> = None; // bulk-settled centroid, for the drift check
    let mut fast_end = 0usize; // grains still moving fast at the end (straggler-tolerant)

    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 15 != 0 && f != frames - 1 {
            continue;
        }
        solver.sample_diagnostics();
        let vel = solver.read_velocities();
        let pos = solver.read_positions();
        let d = solver.diagnostics();

        // Always-true invariants, every sampled frame.
        assert!(
            finite(&pos) && finite(&vel),
            "non-finite state at frame {f}"
        );
        assert!(!d.overflow, "grid overflow at frame {f}");
        let in_box = |p: &[f32; 4]| {
            (0..3).all(|a| p[a] >= scene.box_min[a] - 0.6 && p[a] <= scene.box_max[a] + 0.6)
        };
        assert!(pos.iter().all(in_box), "grain left the box at frame {f}");
        // Bounded contact: the solve never lets grains interpenetrate badly (would blow up a bed).
        assert!(
            d.residual < 0.25,
            "penetration too deep at frame {f}: {:.3}·d",
            d.residual
        );
        max_pen = max_pen.max(d.residual);

        let vmax = vel.iter().map(|&v| speed(v)).fold(0.0, f32::max);
        if f < 90 {
            collapse_peak = collapse_peak.max(vmax);
        }
        // Mid-fall snapshot (pre-impact): grains are moving and nothing has settled to freeze.
        if f == 45 {
            falling_frozen = d.frozen_count;
            falling_vmax = vmax;
        }
        // Once the bulk is frozen, snapshot the centroid for the no-drift comparison vs the end.
        if d.frozen_count > (n as u32) * 9 / 10 && centroid_mid.is_none() {
            centroid_mid = Some(centroid_xz(&pos));
        }
        // Straggler-tolerant rest metric: count grains still genuinely moving (not the single max,
        // which a lone limit-cycling grain dominates). grid_fill's atomic order isn't bit-exact,
        // so a couple of grains can settle a few frames later run-to-run.
        fast_end = vel.iter().filter(|&&v| speed(v) > 0.5).count();
    }

    let pos = solver.read_positions();
    let frozen = solver.diagnostics().frozen_count;
    let (height, radius) = heap_shape(&pos, floor_y);
    let repose = if radius > 1e-3 {
        (height / radius).atan().to_degrees()
    } else {
        90.0
    };
    let (cx_end, cz_end) = centroid_xz(&pos);
    let drift = centroid_mid
        .map(|(cx, cz)| ((cx_end - cx).powi(2) + (cz_end - cz).powi(2)).sqrt())
        .unwrap_or(f32::INFINITY);
    eprintln!(
        "collapse {collapse_peak:.1}, fast@end {fast_end}/{n}, frozen {frozen}/{n}, max-pen {max_pen:.3}, \
         heap h{height:.1} r{radius:.1} repose {repose:.0}°, drift {drift:.3}"
    );

    // It fell with real energy (flowed), then came essentially to rest (a few stragglers allowed).
    assert!(
        collapse_peak > 8.0,
        "grains never gained momentum ({collapse_peak:.1})"
    );
    assert!(
        fast_end < n / 100,
        "bed never went static ({fast_end}/{n} still moving)"
    );

    // Freeze-when-static: mid-fall the grains are moving and unfrozen; ~the whole bed freezes at rest.
    assert!(
        falling_vmax > 5.0,
        "grains weren't moving mid-fall ({falling_vmax:.1})"
    );
    assert!(
        falling_frozen < n as u32 / 20,
        "grains froze while still falling ({falling_frozen})"
    );
    assert!(
        frozen > (n as u32) * 8 / 10,
        "settled bed is not frozen ({frozen}/{n})"
    );

    // A real 3D heap — neither a liquid monolayer (too flat) nor a standing column (too narrow).
    assert!(
        height > 4.0 * s,
        "heap collapsed to a monolayer (h {height:.1})"
    );
    assert!(
        radius > 0.5 * height,
        "did not spread — stood as a column (h {height:.1}, r {radius:.1})"
    );

    // Static rest holds its shape: the settled centroid does not drift (no liquid-like creep).
    assert!(drift < 0.1, "settled heap drifted ({drift:.3})");

    // Angle of repose: loose diagnostic band only (the hard gate is the invariants above).
    assert!(
        (20.0..=55.0).contains(&repose),
        "repose {repose:.0}° outside the believable band"
    );
}

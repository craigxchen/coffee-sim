//! Dry granular-bed invariants (GPU-gated; skips without an adapter).
//!
//! A grain block is poured onto the floor and settles into a heap. This asserts the *solid*
//! behavior that separates a bed from a liquid: it flows, then holds **static with no drift** —
//! a liquid would keep spreading. Static rest is reached by honest contact dissipation + a
//! velocity dead-band (gravity and contacts are still evaluated every frame), NOT by freezing.
//!
//! The angle of repose is printed but asserted only in a loose band: cohesionless spheres roll,
//! so the free repose is shallow (~20°) — a believable steep coffee repose is the job of the
//! deferred rigid-clump follow-up, not friction. The hard gate is stability + static rest.

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
fn bed_settles_into_a_static_heap() {
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

    // The poured slab settles progressively and is reliably static by ~f1050; run past that.
    let frames = 1200u32;
    let mut collapse_peak = 0.0f32;
    let mut max_pen = 0.0f32;
    // Snapshot of the heap once it has settled, to compare against the end (no-drift / no-creep).
    let mut settled_snapshot: Option<(f32, f32, f32)> = None; // (height, cx, cz)

    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f % 15 != 0 && f != frames - 1 {
            continue;
        }
        solver.sample_diagnostics();
        let vel = solver.read_velocities();
        let pos = solver.read_positions();
        let d = solver.diagnostics();

        // Hard invariants, every sampled frame.
        assert!(
            finite(&pos) && finite(&vel),
            "non-finite state at frame {f}"
        );
        assert!(!d.overflow, "grid overflow at frame {f}");
        let in_box = |p: &[f32; 4]| {
            (0..3).all(|a| p[a] >= scene.box_min[a] - 0.6 && p[a] <= scene.box_max[a] + 0.6)
        };
        assert!(pos.iter().all(in_box), "grain left the box at frame {f}");
        // Bounded contact: grains never tunnel through each other (would blow a bed up).
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
        // First snapshot once the bed has mostly come to rest (>=95% of grains static).
        if settled_snapshot.is_none() {
            let moving = vel.iter().filter(|&&v| speed(v) > 0.1).count();
            if moving < n / 20 {
                let (h, _) = heap_shape(&pos, floor_y);
                let (cx, cz) = centroid_xz(&pos);
                settled_snapshot = Some((h, cx, cz));
            }
        }
    }

    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    let (height, radius) = heap_shape(&pos, floor_y);
    let repose = if radius > 1e-3 {
        (height / radius).atan().to_degrees()
    } else {
        90.0
    };
    let moving = vel.iter().filter(|&&v| speed(v) > 0.1).count();
    let (drift, creep) = settled_snapshot
        .map(|(h0, cx, cz)| {
            let (cxe, cze) = centroid_xz(&pos);
            (
                ((cxe - cx).powi(2) + (cze - cz).powi(2)).sqrt(),
                (height - h0).abs(),
            )
        })
        .unwrap_or((f32::INFINITY, f32::INFINITY));
    eprintln!(
        "collapse {collapse_peak:.1}, moving@end {moving}/{n}, max-pen {max_pen:.3}, \
         heap h{height:.1} r{radius:.1} repose {repose:.0}°, drift {drift:.3}, creep {creep:.2}"
    );

    // It flowed into a heap (real momentum), then came to rest.
    assert!(
        collapse_peak > 8.0,
        "grains never gained momentum ({collapse_peak:.1})"
    );
    // Static rest: the bed reached a settled state, and ≥95% of grains are static at the end.
    assert!(
        settled_snapshot.is_some(),
        "bed never reached static rest (>=5% of grains always moving)"
    );
    assert!(
        moving < n / 20,
        "bed not static at end: {moving}/{n} moving"
    );
    // No drift / no creep once settled (a liquid would keep spreading and lowering).
    assert!(drift < 0.3, "settled heap drifted ({drift:.3})");
    assert!(creep < 1.5, "settled heap kept creeping ({creep:.2})");
    // A real 3D heap — not a flat liquid monolayer, not a standing column.
    assert!(
        height > 4.0 * s,
        "heap collapsed to a monolayer (h {height:.1})"
    );
    assert!(
        radius > 0.5 * height,
        "did not spread — stood as a column (h {height:.1}, r {radius:.1})"
    );
    // Angle of repose: loose diagnostic band (cohesionless spheres → shallow; steep coffee repose
    // is the deferred rigid-clump follow-up). The hard gate is stability + static rest above.
    assert!(
        (10.0..=45.0).contains(&repose),
        "repose {repose:.0}° outside the plausible cohesionless-sphere band"
    );
}

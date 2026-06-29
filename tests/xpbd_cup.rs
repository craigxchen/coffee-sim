//! Cup-boundary incompressibility (GPU-gated; skips without an adapter).
//!
//! Guards the boundary-density-compensation fix for the SPH wall deficiency. Near a solid wall the
//! fluid-only density sum under-reads (the wall cuts off part of the kernel sphere), worst at the
//! cup's concave floor/side corner where neighbours are missing on two sides. Before the fix the
//! corner masked genuine over-packing — water crammed ~22% tighter than the bulk while the solve
//! read it as *under*-dense and applied no relieving pressure, storing compression that released as
//! intermittent upward "squeeze-out" eruptions. `compute_boundary` adds the cut-off fraction back
//! (ρ₀·ψ(d)), so corner water reads its true density and relaxes to rest spacing.
//!
//! The regression signature is geometric and resolution-robust: the bottom-corner nearest-neighbour
//! spacing must not collapse below the bulk's (over-packing). On the buggy solver corner NN ≈ 0.69
//! vs interior ≈ 0.91 (ratio ≈ 0.76); fixed, corner NN ≈ 0.96 (ratio ≈ 1.05).

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::sdf::{nearest, SdfPrimitive, SolidKind, MASK_ALL};
use coffee_sim::EmissionInput;
use glam::Vec3;

const FLOOR: f32 = -8.0;
const RADIUS: f32 = 3.0;

fn speed(v: [f32; 4]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Mean nearest-neighbour distance over the given particle indices. O(n²) — fine here.
fn mean_nn(pos: &[[f32; 4]], idx: &[usize]) -> f32 {
    let mut sum = 0.0f64;
    for &i in idx {
        let pi = pos[i];
        let mut nn = f32::INFINITY;
        for (j, &pj) in pos.iter().enumerate() {
            if j == i {
                continue;
            }
            let d = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
            nn = nn.min(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
        sum += nn.sqrt() as f64;
    }
    (sum / idx.len().max(1) as f64) as f32
}

#[test]
fn cup_corner_water_is_not_overpacked() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_cup: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let cfg = Config::default();
    let solids = vec![SdfPrimitive {
        kind: SolidKind::Cylinder {
            center: Vec3::ZERO,
            floor_y: FLOOR,
            rim_y: 12.0, // tall enough that the deep pool stays radially confined as it settles
            radius: RADIUS,
        },
        species_mask: MASK_ALL,
        friction: 0.2,
    }];
    // A multi-layer column dropped (gently) into the cup → a deep, properly-filled pool.
    let scene = Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-4.0, -10.0, -4.0],
        box_max: [4.0, 14.0, 4.0],
        regions: vec![SeedRegion {
            min: [-2.8, -7.5, -2.8],
            max: [2.8, 8.0, 2.8],
            species: Species::Water,
        }],
        solids: solids.clone(),
    };

    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let input = EmissionInput::default();
    let n = solver.particles().particle_count as usize;
    assert!(n > 150, "expected a deep pool, got {n} particles");

    // Settle, then watch the late frames for a sustained eruption (post-settle quiescence).
    let frames = 600u32;
    let mut late_vmax = 0.0f32;
    for f in 0..frames {
        solver.step(1.0 / 60.0, &input);
        if f >= frames - 120 {
            let vel = solver.read_velocities();
            late_vmax = late_vmax.max(vel.iter().map(|&v| speed(v)).fold(0.0, f32::max));
        }
    }

    let pos = solver.read_positions();
    let mut corner = Vec::new();
    let mut interior = Vec::new();
    let mut max_pen = 0.0f32;
    let mut top = f32::NEG_INFINITY;
    for (i, p) in pos.iter().enumerate() {
        let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
        top = top.max(p[1]);
        max_pen = max_pen.max(-nearest(&solids, Vec3::new(p[0], p[1], p[2]), 0).signed);
        if p[1] < FLOOR + 1.2 && r > RADIUS - 0.9 {
            corner.push(i);
        } else if p[1] > FLOOR + 2.0 && r < 1.5 {
            interior.push(i);
        }
    }
    assert!(
        corner.len() >= 12 && interior.len() >= 12,
        "too few samples (corner {}, interior {})",
        corner.len(),
        interior.len()
    );
    let corner_nn = mean_nn(&pos, &corner);
    let interior_nn = mean_nn(&pos, &interior);
    eprintln!(
        "n={n} top={top:.2} corner_nn={corner_nn:.3} interior_nn={interior_nn:.3} \
         ratio={:.2} max_pen={max_pen:.3} late_vmax={late_vmax:.2}",
        corner_nn / interior_nn
    );

    // (1) The corner is not over-packed: its spacing tracks the bulk rather than collapsing.
    //     This deep fill separates cleanly — buggy ratio ≈ 0.82, fixed ≈ 1.08; 0.88 sits between
    //     with margin on both sides (interior settles to ≈ rest spacing in either case).
    assert!(
        corner_nn >= 0.88 * interior_nn,
        "corner over-packed: corner_nn {corner_nn:.3} vs interior_nn {interior_nn:.3} \
         (ratio {:.2} < 0.88) — the hidden-compression squeeze-out reservoir is back",
        corner_nn / interior_nn
    );
    // (2) The wall holds — no water tunnels below/through the cavity surface.
    assert!(max_pen < 0.1, "water penetrated the cup wall: {max_pen:.3}");
    // (3) The pool keeps its volume vertically instead of over-compressing into a shallow puddle.
    //     With this fixed seed the fix settles to top ≈ +2.9 vs the bug's ≈ −1.4 (same count); a
    //     +1.0 floor is a wide, robust guard on "the water held its proper depth".
    assert!(top > 1.0, "pool over-compressed / collapsed: top {top:.2}");
    // (4) Loose sanity that it isn't exploding (a real squeeze-out eruption is tens+; the dedicated
    //     eruption guards live in xpbd_emission). Generous so the deep settling slosh doesn't flake.
    assert!(late_vmax < 5.0, "late eruption: vmax {late_vmax:.2}");
}

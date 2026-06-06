//! Integrator-exactness: a single isolated water particle in free fall must
//! reproduce the solver's exact discrete semi-implicit-Euler recurrence — NOT
//! the continuous `½gt²` form. With no neighbors, every constraint pass is a
//! no-op (density correction sums to zero, XSPH sums to zero); with no solids,
//! `finalize`'s wall branch is skipped (gated on `num_solids > 0`). So the
//! particle evolves purely by predict + velocity-recovery, and its trajectory
//! is the integrator's own closed form. GPU-gated; skips without an adapter.

mod common;

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

use common::{assert_close3, continuous_reference, freefall_reference};

// Round-off budgets (NOT physics bands). The references are exact identities,
// so the only discrepancy is GPU f32 round-off — driven by the velocity-recovery
// cancellation: each substep recovers `v = (pred − pos)/τ`, subtracting two
// same-magnitude f32 positions (error ≈ ulp(coord)/τ) and feeding the result
// into the next substep, so the error accumulates over m = n·substeps substeps
// and is amplified by 1/τ. It is therefore WORST at the finest substeps: with
// substeps=4 and N=120 (480 substeps, coords ≤ ~10) it reaches ~6e-3 on position
// and ~1e-2 on velocity. The budgets below cover that worst config with headroom
// (the plan's rough `~1e-3` estimate undershot the real accumulation). Both stay
// far below any physics signal and ~16× below the ≈0.33 discrete-vs-continuous
// discrimination gap — which is itself checked on POSITION, independent of these.
const POS_ABS_TOL: f64 = 2e-2;
const VEL_ABS_TOL: f64 = 4e-2;
const REL_TOL: f64 = 1e-5;

const GRAVITY: [f32; 3] = [0.0, -20.0, 0.0];
const X0: [f32; 3] = [0.0, 0.0, 0.0];
// Horizontal x-component exercises a no-force axis (g_x = 0 → constant velocity);
// the upward y-component keeps the vertical excursion bounded (~10 units) so the
// particle stays clear of the box walls for the whole run, at coordinate
// magnitudes small enough to keep the round-off budget tight.
const V0: [f32; 3] = [2.0, 20.0, 0.0];
const FRAMES: u32 = 120;
const MARGIN: f32 = 1.0; // wall-clearance the regime guard requires

fn build_scene() -> Scene {
    Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: GRAVITY,
        box_min: [-10.0, -10.0, -10.0],
        box_max: [10.0, 20.0, 10.0],
        regions: vec![SeedRegion {
            min: X0,
            max: X0, // min == max → exactly one lattice point (one particle)
            species: Species::Water,
        }],
        solids: Vec::new(),
    }
}

fn config(substeps: u32) -> Config {
    Config {
        substeps,
        seed_jitter: 0.0,      // exact x₀: no lattice jitter
        velocity_damping: 1.0, // the closed form assumes lossless recovery
        ..Config::default()
    }
}

fn f64v(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn xyz(p: [f32; 4]) -> [f32; 3] {
    [p[0], p[1], p[2]]
}

/// Run one `(dt, substeps)` config: assert the discrete closed form, the
/// unforced-axis behaviour, and the regime guard at sampled frames. Returns the
/// final-frame observed `(position, velocity)` for cross-config checks.
fn run_config(gpu: &GpuContext, dt: f32, substeps: u32) -> ([f32; 3], [f32; 3]) {
    let scene = build_scene();
    let mats = Materials::default();
    let cfg = config(substeps);
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, gpu);
    let input = EmissionInput::default();

    // Exactly one particle, seeded at exactly X0 (validates seed_jitter = 0).
    let seed = solver.read_positions();
    assert_eq!(seed.len(), 1, "expected exactly one seeded particle");
    assert_close3(xyz(seed[0]), f64v(X0), 1e-6, 0.0, "seed position");

    // Seed the initial velocity (SeedRegion sets position only; particles start
    // at rest). The first predict reads this, so the trajectory carries V0.
    solver.write_velocities_for_test(&[[V0[0], V0[1], V0[2], 0.0]]);

    let (x0, v0, g) = (f64v(X0), f64v(V0), f64v(GRAVITY));
    let mut last = ([0.0f32; 3], [0.0f32; 3]);

    for f in 1..=FRAMES {
        solver.step(dt, &input);
        if f % 20 != 0 && f != FRAMES {
            continue; // sample several frames including the last
        }

        let pos = solver.read_positions();
        let vel = solver.read_velocities();
        assert_eq!(pos.len(), 1);
        let p = xyz(pos[0]);
        let v = xyz(vel[0]);

        // Regime guard (R7): no wall clamp or max_speed clamp may fire, else we
        // would be comparing against a corrected trajectory.
        for (a, &pa) in p.iter().enumerate() {
            assert!(
                pa > scene.box_min[a] + MARGIN && pa < scene.box_max[a] - MARGIN,
                "dt={dt} s={substeps} f={f}: particle left interior on axis {a}: {pa}"
            );
        }
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(
            speed < 0.8 * cfg.max_speed,
            "dt={dt} s={substeps} f={f}: speed {speed} approached max_speed clamp"
        );

        // Closed-form assertion (R1, R3, R4): position and velocity match the
        // discrete recurrence for THIS (dt, substeps).
        let (rpos, rvel) = freefall_reference(x0, v0, g, dt as f64, substeps, f);
        assert_close3(
            p,
            rpos,
            POS_ABS_TOL,
            REL_TOL,
            &format!("pos dt={dt} s={substeps} f={f}"),
        );
        assert_close3(
            v,
            rvel,
            VEL_ABS_TOL,
            REL_TOL,
            &format!("vel dt={dt} s={substeps} f={f}"),
        );

        // Unforced axes (R6): x and z have no gravity component, so they must be
        // pure constant velocity — checked independently of the general reference.
        let t = f as f64 * dt as f64;
        let x_const = x0[0] + t * v0[0];
        assert!(
            (p[0] as f64 - x_const).abs() <= POS_ABS_TOL + REL_TOL * x_const.abs(),
            "dt={dt} s={substeps} f={f}: no-force x-axis not constant-velocity: {} vs {x_const}",
            p[0]
        );
        assert!(
            (p[2] as f64 - x0[2]).abs() <= POS_ABS_TOL,
            "dt={dt} s={substeps} f={f}: no-force z-axis drifted: {}",
            p[2]
        );

        if f == FRAMES {
            last = (p, v);
        }
    }
    last
}

/// R1, R3, R4, R5, R6, R7: the bare integrator matches its own discrete closed
/// form across the default config and the dt×substeps sweep; velocity is
/// substep-invariant and position converges toward continuous `½gt²` as
/// substeps increase. Covers AE1, AE3, AE4.
#[test]
fn freefall_matches_discrete_recurrence() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_integrator: no GPU adapter; skipping.");
        return;
    };

    let dts = [1.0f32 / 60.0, 1.0f32 / 120.0];
    let substeps_set = [1u32, 2, 4];

    // (substeps, final_pos, final_vel) at dt = 1/60, for the cross-config checks.
    let mut at_60: Vec<(u32, [f32; 3], [f32; 3])> = Vec::new();

    for &dt in &dts {
        for &s in &substeps_set {
            let (pos, vel) = run_config(&gpu, dt, s);
            if (dt - 1.0 / 60.0).abs() < 1e-9 {
                at_60.push((s, pos, vel));
            }
        }
    }
    at_60.sort_by_key(|&(s, _, _)| s);

    // R5 velocity substep-invariance: v(N) = v₀ + N·dt·g, independent of
    // substeps, so every config's final velocity must agree (within round-off).
    let v_ref = at_60[0].2;
    for &(s, _, v) in &at_60 {
        for a in 0..3 {
            assert!(
                (v[a] - v_ref[a]).abs() as f64 <= 2.0 * VEL_ABS_TOL,
                "substeps={s}: final velocity not substep-invariant on axis {a}: {} vs {}",
                v[a],
                v_ref[a]
            );
        }
    }

    // R5 position convergence: the discrete final y must move monotonically
    // toward the continuous ½gt² value as substeps increase.
    // Use the f32-widened dt the GPU actually ran, not the f64 literal.
    let x_cont = continuous_reference(
        f64v(X0),
        f64v(V0),
        f64v(GRAVITY),
        (1.0f32 / 60.0) as f64,
        FRAMES,
    );
    // Allow position round-off slack; the gap *steps* (~0.08–0.17) dwarf it, so
    // this still asserts a real monotonic decrease toward the continuous limit.
    let mut prev_gap = f64::INFINITY;
    for &(s, pos, _) in &at_60 {
        let gap = (pos[1] as f64 - x_cont[1]).abs();
        assert!(
            gap <= prev_gap + 2.0 * POS_ABS_TOL,
            "substeps={s}: final y not converging toward continuous (gap {gap} > {prev_gap})"
        );
        prev_gap = gap;
    }
    // And substeps=1 must be clearly distinct from continuous — otherwise the
    // sweep is accidentally sitting at the continuous limit and proves nothing.
    let s1_gap = (at_60[0].1[1] as f64 - x_cont[1]).abs();
    assert!(
        s1_gap > 0.1,
        "substeps=1 should differ from continuous ½gt² by ≫ round-off, gap={s1_gap}"
    );
}

/// R2: the observed trajectory must match the DISCRETE recurrence and differ
/// from the continuous `½gt²` form by far more than tolerance — the test fails
/// if the integrator ever produces the textbook continuous trajectory. Covers
/// AE2.
#[test]
fn freefall_rejects_continuous_form() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_integrator: no GPU adapter; skipping.");
        return;
    };

    let dt = 1.0f32 / 60.0;
    let (observed, _) = run_config(&gpu, dt, 1);

    let (r_discrete, _) =
        freefall_reference(f64v(X0), f64v(V0), f64v(GRAVITY), dt as f64, 1, FRAMES);
    let r_continuous = continuous_reference(f64v(X0), f64v(V0), f64v(GRAVITY), dt as f64, FRAMES);

    // Matches the discrete reference (re-affirmed on the final frame).
    assert_close3(
        observed,
        r_discrete,
        POS_ABS_TOL,
        REL_TOL,
        "discrimination: discrete",
    );

    // The gap to the continuous form (≈0.33 sim-units here) must dwarf the
    // round-off budget. 0.1 is ≫ the ~2e-3 budget yet ≪ the true gap, so this
    // both confirms discrimination and guards against a degenerate setup.
    let gap_y = (observed[1] as f64 - r_continuous[1]).abs();
    assert!(
        gap_y > 0.1,
        "continuous ½gt² is only {gap_y} from observed — the test no longer discriminates the integrator"
    );
}

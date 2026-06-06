//! Shared support for the analytic physics-validation tests.
//!
//! Deliberately minimal: it ships only what the analytic tests actually exercise
//! — an exact-compare helper, the closed-form free-fall references, and a CPU
//! SPH density reconstruction (added when the rest-lattice / hydrostatic tests
//! first needed it). A phenomenon-keyed tolerance table is still deferred until a
//! test needs it.
//!
//! The comparator is a float32 ROUND-OFF budget, not a physics tolerance band:
//! the references below are exact algebraic identities, so the only discrepancy
//! from the GPU is f32 round-off (see the per-call tolerances in the test).
//!
//! Each integration-test binary compiles this module fresh and uses a different
//! subset of it, so unused helpers are expected per-binary — allow dead code.
#![allow(dead_code)]

use coffee_sim::utils::kernels::w_poly6;

/// Assert each component of a read-back f32 vector matches its f64 reference
/// within `abs_tol + rel_tol * |expected|`. Panics with the axis, expected,
/// actual, and delta on failure.
pub fn assert_close3(
    actual: [f32; 3],
    expected: [f64; 3],
    abs_tol: f64,
    rel_tol: f64,
    label: &str,
) {
    for k in 0..3 {
        let a = actual[k] as f64;
        let tol = abs_tol + rel_tol * expected[k].abs();
        let d = (a - expected[k]).abs();
        assert!(
            d <= tol,
            "{label}: axis {k}: expected {}, got {a}, |Δ|={d} > tol {tol}",
            expected[k],
        );
    }
}

/// Discrete semi-implicit-Euler free-fall reference — the solver's OWN
/// recurrence, NOT continuous kinematics. Per substep of size `τ = dt/substeps`
/// the solver does `x ← x + τ·v + τ²·g` then recovers `v ← (x_new − x_old)/τ`,
/// i.e. `v ← v + τ·g`. After `n` frames (`m = n·substeps` substeps) this closes
/// to:
///   v(n) = v₀ + n·dt·g                       (independent of substeps)
///   x(n) = x₀ + n·dt·v₀ + τ²·g·m(m+1)/2       (depends on substeps)
pub fn freefall_reference(
    x0: [f64; 3],
    v0: [f64; 3],
    g: [f64; 3],
    dt: f64,
    substeps: u32,
    n: u32,
) -> ([f64; 3], [f64; 3]) {
    let tau = dt / substeps as f64;
    let m = (n as u64 * substeps as u64) as f64;
    let t = n as f64 * dt; // == m * tau
    let mut pos = [0.0; 3];
    let mut vel = [0.0; 3];
    for k in 0..3 {
        vel[k] = v0[k] + t * g[k];
        pos[k] = x0[k] + t * v0[k] + tau * tau * g[k] * (m * (m + 1.0) / 2.0);
    }
    (pos, vel)
}

/// Continuous ballistic reference `x = x₀ + v₀·t + ½·g·t²` — the textbook form
/// the integrator does NOT implement. Used only to prove the test discriminates
/// the discrete recurrence from the continuous approximation, and as the limit
/// the discrete trajectory approaches as `substeps → ∞`.
pub fn continuous_reference(x0: [f64; 3], v0: [f64; 3], g: [f64; 3], dt: f64, n: u32) -> [f64; 3] {
    let t = n as f64 * dt;
    let mut pos = [0.0; 3];
    for k in 0..3 {
        pos[k] = x0[k] + v0[k] * t + 0.5 * g[k] * t * t;
    }
    pos
}

/// Reconstruct per-particle SPH density on the CPU from positions, mirroring the
/// GPU density constraint and `kernels::rest_density`:
///   ρ_i = Σ_j m · W_poly6(|x_i − x_j|, h)   (the j == i self term included).
/// Returns `(density, neighbour_count)` per particle so callers can filter to
/// interior particles (full neighbourhoods) and exclude the physically-deficient
/// free surface. O(N²) — fine for the modest particle counts these tests use.
///
/// `pos` carries the moisture lane in `.w`; only `.xyz` is read here.
pub fn reconstruct_density(pos: &[[f32; 4]], h: f32, mass: f32) -> Vec<(f32, usize)> {
    pos.iter()
        .map(|&pi| {
            let mut rho = mass * w_poly6(0.0, h);
            let mut neighbours = 0usize;
            for &pj in pos {
                let dx = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
                let r = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
                if r > 0.0 && r < h {
                    rho += mass * w_poly6(r, h);
                    neighbours += 1;
                }
            }
            (rho, neighbours)
        })
        .collect()
}

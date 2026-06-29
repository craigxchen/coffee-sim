---
date: 2026-06-05
topic: freefall-integrator-test
---

# Free-Fall Ballistic Recurrence — Integrator Exactness Test

## Summary

Add the first analytic physics-validation test for the XPBD/PBF solver: drop one isolated water particle (no neighbors → every constraint pass is a no-op) and assert that `read_positions`/`read_velocities` reproduce the solver's *own* exact discrete semi-implicit-Euler recurrence — across the default config plus a `dt`×`substeps` sweep, each config checked against its own closed form. This test also introduces a shared `tests/common/` reference harness, of which it is the first consumer.

## Problem Frame

The existing suite (`tests/xpbd_water.rs`, `xpbd_bed.rs`, …) asserts qualitative invariants and bounds — "settles", "density-in-band 80%", "momentum conserved" — but nothing pins the time integrator itself against a closed form. Every settling, hydrostatic, and coupling result rides on the predict/recover step being exact for the unconstrained case; a stray ½ factor, a damping leak, or a gravity-units regression in `predict`/`finalize` would silently bias every downstream test while each of them still passes.

There is a second, subtler trap. The natural "textbook" reference for free fall is the continuous `y = y₀ + v₀t + ½gt²`. The solver does **not** integrate that — `predict` applies the full `dt²·g` per step (`src/solvers/xpbd/common.wgsl:410`), so the exact trajectory is the *discrete* semi-implicit-Euler recurrence, which differs from the continuous form by an O(dt·g·t) term and only converges to it as the substep count grows. A test written against ½gt² would either false-fail or, if loosened to pass, stop discriminating real integrator bugs. The right reference is the solver's own discrete recurrence, derived per timestep.

## Key Decisions

- **Reference is the discrete recurrence, not continuous kinematics.** Per substep of size `τ`: `v ← v + τg`, `x ← x + τv + τ²g`. The test asserts this exact identity and explicitly checks that the continuous ½gt² value is *wrong* by far more than tolerance — so the test provably tests the real integrator.
- **Each `(dt, substeps)` config is checked against its own closed form — position is not substep-invariant.** Velocity after `n` frames is `v₀ + n·dt·g` regardless of substep count, but position is `x₀ + n·dt·v₀ + dt²·g·n(n·s+1)/(2s)` for `s` substeps — it *changes* with `s` and approaches continuous ½gt² as `s→∞`. That convergence direction is itself asserted, rather than a (false) blanket substep-invariance.
- **Tolerance is a float32 round-off budget, not a tuned physics band.** The closed form is an exact algebraic identity; the only error source is float32 round-off accumulating over ≤~480 substeps. The ≈1e-5 relative (velocity) / 1e-4 absolute (position) tolerance is sized to cover that accumulation and documented as such — consistent with the project doctrine of asserting the law, not tuned values.
- **Build the shared harness now, but keep its surface minimal.** A new `tests/common/` module is introduced with free-fall as its first consumer. Its initial surface is limited to what free-fall actually exercises (an exact-compare helper and a parametrized step-runner). Density reconstruction and the phenomenon-keyed tolerance table are deferred until a later analytic test pins what is genuinely reusable, so the harness does not over-fit to one test.
- **Guard the regime, and exercise the no-force axes.** The trajectory is only the bare integrator while the particle touches no wall/SDF and stays under the velocity clamp; the test asserts that guard explicitly. A horizontal `v₀` component is included so the axes with no force are checked as pure constant velocity, catching any spurious force injected on free axes.

## Requirements

**Discrete-recurrence assertion**

- R1. Drive exactly one isolated water particle headless for `N` frames; at sampled frames (including the last), `read_positions`/`read_velocities` match the solver's exact discrete semi-implicit-Euler recurrence. At the default `substeps=1`: `v_n = v₀ + n·dt·g` and `x_n = x₀ + n·dt·v₀ + dt²·g·n(n+1)/2`.
- R2. The test asserts discrimination explicitly: the continuous `½gt²` position differs from the observed position by far more than the tolerance, proving the test distinguishes the real integrator from the textbook approximation.

**Parametrization**

- R3. The default config (`substeps=1`, `dt=1/60`) passes at the round-off tolerance.
- R4. Sweep `dt ∈ {1/60, 1/120}` × `substeps ∈ {1, 2, 4}`; each config is asserted against its own exact discrete closed form computed with `τ = dt/substeps` and `m = n·substeps` substeps total.
- R5. Velocity is asserted substep-invariant (`v₀ + n·dt·g`, independent of `substeps`); position is asserted to move monotonically toward the continuous `½gt²` value as `substeps` increases.

**Isolation and regime guards**

- R6. `v₀` carries a component perpendicular to gravity; the no-force axes are asserted to evolve as pure constant velocity `x₀ + n·dt·v₀`, confirming no spurious force is injected on free axes.
- R7. Throughout the asserted window, the test asserts the particle contacts no box wall or SDF solid (stays within interior margins) and that `‖v‖ < max_speed`. A violation fails the test rather than silently comparing against a clamped or wall-corrected trajectory.
- R8. The scene and initial conditions are sized so the analytic trajectory stays clear of walls and under the clamp for the full `N`: a plain box with no SDF solids, exactly one water particle and no grains, gravity taken from the scene (reduced sim units, default `[0, -20, 0]`).

**Harness and doctrine**

- R9. A shared `tests/common/` module is introduced with free-fall as its first consumer. Its initial surface is limited to what free-fall exercises (exact-compare helper + parametrized step-runner); density-reconstruction and tolerance-table helpers are deferred to later analytic tests.
- R10. The test is GPU-gated: it returns cleanly (skips) when `GpuContext::new_headless()` yields no adapter. It is never `#[ignore]`d.
- R11. The tolerance is documented in-test as a float32 round-off budget (≈1e-5 relative velocity / 1e-4 absolute position over ≤~480 substeps), not a tuned physics band.

## Acceptance Examples

- AE1. **Covers R1.**
  - **Given:** scene gravity `g = [0,-20,0]`, one water particle at `x₀` high in a tall empty box, `v₀ = (3, 0, 0)`, `dt = 1/60`, `substeps = 1`.
  - **When:** stepped `n = 120` frames and read back.
  - **Then:** `v ≈ (3, -40, 0)` and `x ≈ (x₀ₓ + 6, x₀_y − 40.333…, x₀_z)`, each within the round-off tolerance (the y-drop is `dt²·g·n(n+1)/2 = (1/3600)·20·7260`).

- AE2. **Covers R2.**
  - **Given:** the same run as AE1.
  - **When:** the continuous reference `x₀_y + ½·(−20)·(2)² = x₀_y − 40.0` is compared against the observed `x₀_y − 40.333…`.
  - **Then:** the gap (≈0.333) exceeds the position tolerance by orders of magnitude — the test asserts this gap is large, so passing requires the discrete form specifically.

- AE3. **Covers R4, R5.**
  - **Given:** `dt = 1/60`, `n = 120`, comparing `substeps ∈ {1, 2, 4}`.
  - **When:** each config is evolved and its y-position read.
  - **Then:** velocity is `−40` for all three (substep-invariant); y-drop is `−40.333…` (s=1), `−40.166…` (s=2), and closer still (s=4) — strictly decreasing in magnitude toward the continuous `−40.0`, and each value matches its own closed form within tolerance.

- AE4. **Covers R7.**
  - **Given:** any config in the sweep.
  - **When:** the guard is evaluated each sampled frame.
  - **Then:** all positions remain within the box interior margins and `‖v‖ < max_speed` (peak `‖v‖ ≈ 40 < 50`); if either is violated, the test fails instead of asserting against the recorded trajectory.

## Scope Boundaries

- The other seven analytic tests from the ideation (hydrostatic pressure–depth, rest-lattice null, extraction exponential, mixing equilibrium, drag/Kozeny-Carman, volume-conservation, Newton cooling) are out of scope here.
- Pressure/λ readback (`read_lambdas()`), CPU density reconstruction, and the phenomenon-keyed tolerance table are deferred — they belong to the tests that first need them, not this one.
- Multi-particle behavior and any constraint interaction are deliberately excluded; isolation is the whole point.

## Dependencies / Assumptions

- A GPU adapter is required to actually run the assertions; without one the test skips (R10), matching every existing GPU-gated test.
- Assumes the verified integrator facts: `predict` is `pred = pos + dt·v + dt²·g` (`src/solvers/xpbd/common.wgsl:410`) and velocity recovery is `v = (pred − pos)/dt · velocity_damping` with `velocity_damping = 1.0` default (`src/utils/config.rs:146`). If `velocity_damping` is changed from 1.0 by a config the test builds, the closed form must include that factor — the test should construct its own `Config` with `velocity_damping = 1.0` rather than relying on the default silently.
- Assumes a single-cell/point seed yields exactly one water particle with no neighbors within the support radius; the exact seeding mechanism is a planning detail.
- `max_speed = 50.0` default (`src/utils/config.rs:148`) is the clamp the regime guard checks against.

## Sources / Research

- Ideation: `docs/ideation/2026-06-05-physics-validation-test-suite-ideation.md` (survivor #1).
- Integrator: `src/solvers/xpbd/common.wgsl:410` (predict), `:571`+ (finalize / velocity recovery, wall + SDF collision, velocity clamp).
- Observables and harness: `read_positions`/`read_velocities` (`src/solvers/xpbd/mod.rs:549,554`), `GpuContext::new_headless()` (`src/utils/gpu.rs`).
- Existing self-contained test pattern to mirror (and the `density_in_band` reconstruction that a later test will lift into `tests/common/`): `tests/xpbd_water.rs:49`.
- Config defaults: `velocity_damping=1.0`, `max_speed=50.0` (`src/utils/config.rs:146,148`).

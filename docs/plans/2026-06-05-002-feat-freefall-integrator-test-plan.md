---
date: 2026-06-05
type: feat
status: completed
origin: docs/brainstorms/2026-06-05-freefall-integrator-test-requirements.md
---

# feat: Free-Fall Ballistic Recurrence — Integrator-Exactness Test

## Summary

Add the first analytic physics-validation test for the XPBD/PBF solver and the shared `tests/common/` harness it is the first consumer of. The test drops one isolated, neighborless water particle and asserts `read_positions`/`read_velocities` reproduce the solver's exact discrete semi-implicit-Euler recurrence — across the default config plus a `dt`×`substeps` sweep, each config checked against its own closed form, with explicit discrimination against the continuous ½gt² approximation.

## Problem Frame

The existing suite asserts qualitative invariants and bounds but pins nothing against a closed form. Every settling, hydrostatic, and coupling test rides on the bare integrator (predict + velocity recovery) being exact for the unconstrained case; a stray ½ factor, a damping leak, or a gravity-units regression would silently bias all of them while each still passes. This test isolates the integrator so such a regression fails loudly. It also establishes the `tests/common/` harness that the remaining seven analytic tests (see origin) will reuse, which is why "build the harness now" is in scope here rather than deferred.

The reference must be the solver's *discrete* recurrence, not continuous kinematics: `predict` applies the full `dt²·g` per substep (`src/solvers/xpbd/common.wgsl:410`), so the exact trajectory differs from `y₀ + v₀t + ½gt²` by an O(dt·g·t) term and only converges to it as substep count grows. A test written against ½gt² would false-fail or stop discriminating real bugs.

## Requirements Trace

Carried from origin (`docs/brainstorms/2026-06-05-freefall-integrator-test-requirements.md`):

- Discrete-recurrence assertion: R1, R2 → U2
- Parametrization (default + dt×substeps sweep, per-config closed form, velocity invariance, convergence-to-continuous): R3, R4, R5 → U2
- Isolation and regime guards (unforced-axis check, no wall/clamp, sized scene): R6, R7, R8 → U2
- Harness and doctrine (shared `tests/common/`, GPU-gated skip, tolerance-as-round-off-budget): R9 → U1; R10, R11 → U1 + U2
- Acceptance Examples AE1–AE4 → U2 test scenarios

---

## Key Technical Decisions

- **Reference closed form computed in f64; tolerance is a float32 round-off budget.** The harness computes the expected trajectory from the exact algebraic closed form in `f64`, not by replaying the solver's `f32` recurrence — this keeps the test asserting the *law* (and lets it discriminate the ½gt² form per R2) rather than coupling to the implementation. The GPU runs `f32`, so the observed values differ from the f64 reference only by float32 round-off accumulated over the substeps. The tolerance is sized to cover that accumulation (see the round-off-budget decision below), never tuned to a physics value.

- **Isolation via a solid-free box, not a special code path.** A plain AABB scene with `solids: []` and exactly one water particle means: the density solve has no neighbors so its position correction is identically zero; XSPH (neighbor-summed) is zero; and in `finalize` the wall/SDF velocity-response branch is gated on `num_solids > 0` (`src/solvers/xpbd/common.wgsl:592`) so it is skipped entirely. The only mechanisms that can perturb the bare recurrence are the box-AABB clamp inside `apply_dp` and the `max_speed` clamp in `finalize` — both no-ops while the particle stays inside the box and under the clamp. This is why the regime guard (R7) only needs to check box-interior containment and `‖v‖ < max_speed`, not SDF contact.

- **Exact `x₀` requires `seed_jitter = 0.0`.** `seed_block` adds `cfg.seed_jitter * spacing` jitter to every seeded point (`src/solvers/xpbd/mod.rs:421`), and the default is `0.1` (`src/utils/config.rs:149`). A single-point `SeedRegion` (`min == max`) yields exactly one particle, but only at a known position when jitter is disabled. The test builds its `Config` with `seed_jitter = 0.0`.

- **Build `Config` explicitly for the law, not the defaults.** Construct `Config { velocity_damping: 1.0, seed_jitter: 0.0, substeps: <swept>, ..Config::default() }`. `velocity_damping = 1.0` is already the default but is set explicitly because the closed form assumes it; the substep sweep varies `substeps`; `dt` is a `step()` argument, not a `Config` field, so the dt sweep varies the value passed to `step`.

- **Harness surface kept minimal.** `tests/common/` ships only what free-fall exercises: an exact-compare helper (absolute + relative, with a labeled failure message) and a free-fall reference that returns the discrete closed-form `(position, velocity)` at a given frame for a given `(x₀, v₀, g, dt, substeps)`. Density reconstruction and a phenomenon-keyed tolerance table are deferred to the analytic tests that first need them (per origin scope boundary), so the harness does not over-fit to one consumer.

---

## Implementation Units

### U1. Shared `tests/common/` analytic-test harness

**Goal:** Introduce the Cargo integration-test shared module that free-fall and later analytic tests consume, with only the two helpers free-fall needs.

**Requirements:** R9, R10, R11 (helpers + tolerance semantics).

**Dependencies:** none.

**Files:**
- `tests/common/mod.rs` (new) — the shared module. Placing the module at `tests/common/mod.rs` (a subdirectory `mod.rs`, not `tests/common.rs`) is the idiomatic Cargo convention that prevents Cargo from compiling it as its own test binary; consumers declare `mod common;` and `use common::...`.

**Approach:**
- Provide an exact-compare helper for 3-vectors (and scalars) that takes an absolute tolerance and a relative tolerance and a context label, and panics with a message naming the axis, expected, actual, and delta on failure. It is a round-off comparator, not a physics-band comparator — document that intent in a module comment.
- Provide a free-fall reference function: given `x₀`, `v₀`, gravity `g`, frame `dt`, `substeps`, and a frame index `n`, return the discrete closed-form `(pos, vel)`:
  - `τ = dt / substeps`, `m = n * substeps` (substeps elapsed at frame `n`)
  - `vel = v₀ + (n * dt) * g`
  - `pos = x₀ + (n * dt) * v₀ + τ² * g * (m * (m + 1) / 2)`
  - Compute in `f64` and return `f32`/`f64` as the comparator expects.
- Keep the public surface to exactly these two helpers plus any tiny structs they need. Do **not** add density reconstruction, a tolerance table, scene builders, or generic step-loop wrappers — those belong to later tests.

**Patterns to follow:** mirror the self-contained helper style already used inline in `tests/xpbd_water.rs` (e.g., `speed`, `dist2`, `density_in_band`); this unit lifts that style into a shared module rather than inventing a new abstraction. CPU/GPU kernel-twin precedent: `src/utils/kernels.rs`.

**Test scenarios:**
- The harness is exercised by U2; it has no standalone behavioral test. `Test expectation: none -- pure test-support helpers, validated through U2's assertions (a wrong reference formula makes U2 fail).`

**Verification:** `tests/xpbd_integrator.rs` (U2) compiles against the module via `mod common;` and its assertions pass on a GPU-equipped machine; `cargo test` does not emit a "module never used"/extra-binary warning for `tests/common/`.

### U2. Free-fall integrator-exactness test

**Goal:** Assert the bare integrator reproduces the discrete closed form across the default config and the `dt`×`substeps` sweep, with discrimination, unforced-axis, and regime-guard checks.

**Requirements:** R1, R2, R3, R4, R5, R6, R7, R8 (and R10, R11 in setup).

**Dependencies:** U1.

**Files:**
- `tests/xpbd_integrator.rs` (new) — the test, with `mod common;` at the top.

**Approach:**
- **Scene:** a plain AABB box with `solids: []`, gravity at the default reduced sim value `[0, -20, 0]`, and a single-point water `SeedRegion` (`min == max`) at a known interior point. Size the box and pick `N` so the entire discrete trajectory stays a few units clear of every wall and peak `‖v‖` stays well under `max_speed = 50` (e.g., seed near the top of a box taller than the fall, with a small horizontal `v₀` and `N ≈ 120`). Coordinates are kept modest — ideally centering the box near the origin so magnitudes stay single- to low-double-digit — because `ulp(coordinate)` drives both the velocity-recovery and position round-off budgets (see the tolerance note).
- **Config:** `Config { velocity_damping: 1.0, seed_jitter: 0.0, substeps, ..Config::default() }`. Drive the solver with `solver.step(dt, &EmissionInput::default())` (no pour), reading back with `read_positions()` / `read_velocities()` at sampled frames including the last.
- **Initial velocity:** give `v₀` a component perpendicular to gravity (e.g., `+x`) so the no-force axes are exercised; the gravity axis carries the quadratic, the free axes are pure constant velocity.
- **Reference:** for each sampled frame, compare against `common`'s free-fall reference evaluated at that frame for the active `(dt, substeps)`.
- **Discrimination:** also compute the continuous `x₀ + v₀·T + ½·g·T²` value and assert the gap between it and the observed position **exceeds** the position tolerance by a wide margin (orders of magnitude), so passing requires the discrete form specifically.
- **GPU gate:** `let Some(gpu) = GpuContext::new_headless() else { eprintln!(...); return; };` — skip cleanly with no adapter, never `#[ignore]`, mirroring `tests/xpbd_water.rs:77`.

**Technical design (directional, not specification):** per-config expected values, for `substeps = s`, frame `dt = Δ`, `N` frames, `τ = Δ/s`, `m = N·s`:
- `v(N) = v₀ + N·Δ·g` (independent of `s` → the substep-invariance assertion for velocity)
- `x(N) = x₀ + N·Δ·v₀ + τ²·g·m(m+1)/2` (depends on `s` → each config checked against its own value)
- continuous reference `x_cont(N) = x₀ + v₀·(N·Δ) + ½·g·(N·Δ)²`; discrete minus continuous at `s=1` is `Δ²·g·N/2` (e.g. ≈ 0.33 sim-units for `Δ=1/60, g=20, N=120`) — far above any round-off tolerance.

**Patterns to follow:** the GPU-gated headless harness, `Solver` trait usage, and readback pattern in `tests/xpbd_water.rs` and `tests/xpbd_extraction.rs`; `Config`/`Scene`/`Materials` construction as in `src/engine/scene.rs`.

**Test scenarios:**
- Covers AE1. Default config (`substeps = 1`, `dt = 1/60`), `v₀` with a horizontal component, `N ≈ 120`: every sampled frame's position and velocity match the discrete closed form within the round-off budget (velocity y-component reaches `v₀_y + N·dt·g`; position y-drop equals `dt²·g·N(N+1)/2`).
- Covers AE2. Discrimination: the continuous ½gt² position differs from the observed final position by far more than the position tolerance — assert this gap is large (the test fails if observed matches the continuous form).
- Covers AE3 (R4, R5). Sweep `dt ∈ {1/60, 1/120}` × `substeps ∈ {1, 2, 4}`: each `(dt, substeps)` config matches its own per-config closed form; velocity at the final frame is identical across `substeps` for a fixed `dt` (substep-invariant); the position y-drop magnitude decreases monotonically toward the continuous value as `substeps` increases.
- Covers R6. Unforced-axis check: the axis (or axes) with no gravity component evolve as exactly `x₀ + n·dt·v₀` (constant velocity), confirming no spurious force is injected on free axes.
- Covers AE4 (R7). Regime guard: at every sampled frame the particle stays within the box interior margins and `‖v‖ < max_speed`; a violation fails the test rather than asserting against a clamped/wall-corrected trajectory.
- Edge: `active_count()` is exactly 1 after build (the single-point seed yielded one particle and no second particle is within support), so the constraint passes are genuinely no-ops.

**Verification:** on a GPU-equipped machine, `cargo test --test xpbd_integrator` passes all assertions; on a machine with no adapter the test prints the skip line and returns without failing; `cargo clippy` is clean for the new files.

---

## Scope Boundaries

In scope: the free-fall test, the two `tests/common/` helpers, the `dt`×`substeps` sweep, discrimination, unforced-axis, and regime guards.

Outside this work (from origin scope boundaries):
- The other seven analytic tests (hydrostatic, rest-lattice null, extraction exponential, mixing equilibrium, drag/Kozeny-Carman, volume-conservation, Newton cooling).
- `read_lambdas()` / pressure readback, CPU density reconstruction, and a phenomenon-keyed tolerance table — added by the tests that first need them, not here.
- Multi-particle behavior and any constraint interaction — isolation is the point.

### Deferred to Follow-Up Work
- None for this plan.

### Deferred to Implementation
- **Exact tolerance constants.** Tolerance is a float32 round-off budget, not a physics band. There are two distinct f32 error sources, and the velocity one is *not* a smaller version of the position one:
  - **Position accumulation.** The recurrence `p += τ·v + τ²·g` grows monotonically in one direction, so its rounding errors are directionally correlated — the realistic worst case is closer to linear-in-`m` (`m · ulp(magnitude)`, e.g. ~480 · ulp(40) ≈ 1e-3 at `substeps = 4`) than the random-walk `√m · magnitude · ε_f32`. Start the position budget nearer `1e-3` for the `substeps = 4` config rather than `1e-4`.
  - **Velocity-recovery cancellation.** `finalize` recovers `v = (pred − pos)/τ` (`src/solvers/xpbd/common.wgsl:575`), subtracting two same-magnitude f32 positions (both O(coordinate magnitude)) and dividing by `τ = dt/substeps`. The absolute error is ≈ `ulp(coord_magnitude)/τ`, which scales with `1/τ` and is therefore **worst at the finest substeps**, independent of `m`. At coordinate magnitude ~40 and `substeps = 4` this is ~1.5e-5 relative — at or above a naive `1e-5`. Size the velocity budget from this (≈`1e-4` relative is safe and still orders below any physics signal), and/or center the box near the origin so coordinate magnitudes — hence `ulp(coord)` — stay small.
  - Both budgets remain 2–3 orders below the ≈0.33 discrimination gap, so the test stays sharp regardless. Confirm the chosen constants cover the worst swept config (finest `τ`, largest coordinate magnitude) without masking a real bug, and document them in-test as a round-off budget.
- **Exact box dimensions, seed point, `v₀`, and `N`.** Tuned at implementation so the discrete trajectory satisfies the regime guard (R7) with margin; the guard is the contract, the specific numbers are a starting recommendation.

---

## Sources & Research

- Origin requirements: `docs/brainstorms/2026-06-05-freefall-integrator-test-requirements.md`.
- Origin ideation: `docs/ideation/2026-06-05-physics-validation-test-suite-ideation.md` (survivor #1).
- Integrator: predict `np = p + dt·v + dt²·g` (`src/solvers/xpbd/common.wgsl:410`); finalize velocity recovery, wall/SDF branch gated on `num_solids > 0`, and `max_speed` clamp (`src/solvers/xpbd/common.wgsl:571`–630).
- Substep loop: `step()` sets `params.dt = dt / substeps` and loops `for _ in 0..substeps` (`src/solvers/xpbd/mod.rs:1756`, `:1801`); `substeps` is a `Config` field, default 1 (`src/utils/config.rs:14`, `:111`).
- Seeding: `seed_block` lattices `[min, max]` with `seed_jitter * spacing` jitter (`src/solvers/xpbd/mod.rs:402`–453); `seed_jitter` default 0.1 (`src/utils/config.rs:149`); `SeedRegion { min, max, species }` and `Scene` defaults (`src/engine/scene.rs:21`, `:52`).
- Config knobs: `velocity_damping` default 1.0 (`src/utils/config.rs:146`/`:48`), `max_speed` default 50.0 (`:148`/`:50`).
- Test pattern to mirror: GPU-gated headless setup, `Solver` trait, readback (`tests/xpbd_water.rs:76`+; `density_in_band` style at `:49`).

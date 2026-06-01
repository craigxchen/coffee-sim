# coffeesim — `solvers/` Plan (the seam)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/solvers/base.rs` + the registry · **Status:** plan
**Depends on:** `utils/` (GpuContext, spatial hash), `engine/` (Scene), `options/`

## Purpose
Define the contract every complete solver implements and the shared conventions they follow, so `engine`, `vis`, and `profiler` work uniformly across XPBD, MPM, SPH+MPM, and any future method. This module owns *no physics* — only the seam.

## Scope
**In:** the `CoffeeSolver` trait, `SolverDescriptor`, the cross-seam data types (`PourInput`, `BrewStateView`, `BrewMetrics`, `SolverProfile`), the solver registry/factory, and the substep conventions + shared helpers solvers are expected to use.
**Out:** any solver's internal dynamics (those live in `solver_*.md`); the shared physics (`models.md`).

## The contract — `CoffeeSolver`
`step()` advances a **whole frame**; internal substepping, coupling, and extraction are the solver's business. The trait is the version in `ARCHITECTURE.md §5.1`; this plan pins the guarantees:
- `step(dt, input)` must be **deterministic** given identical state + input + seed (reproducible comparisons).
- `step()` must never leave the sim in a non-finite state — each solver is responsible for its own stability guarantees (the KE watchdog helper below is available).
- `state()`/`metrics()`/`profile()` are **read-only** and cheap; they must not trigger a GPU sync that stalls the frame (use the prior frame's resolved query results).

## Cross-seam data types (the universal buffer contract)
For `vis` and `profiler` to be solver-agnostic, every solver exposes its state in a **canonical layout** regardless of internal representation:

- `BrewStateView` — handles to canonical particle buffers: `position`, `velocity`, `phase_tag` (Water | Grain | Fines), `concentration`, `temperature`, `moisture`, plus optional sampled fields (`alpha_s`, `pressure`) for debug overlays. Grid-based solvers (MPM) expose particle views + optional field textures; particle solvers expose buffers directly.
- `PourInput` — kettle position, flow rate, pour angle, plus discrete events (start/stop bloom, reset).
- `BrewMetrics` — `extraction_yield`, `tds`, `drawdown_time`, `evenness`, plus counts (`particle_count`, `iteration_count`).
- `SolverProfile` — per-pass GPU durations (from `timestamp-query`) **and `dispatches_per_frame`** (see `profiler.md`).

## Registry
A factory keyed by a `SolverKind` enum (`Xpbd | Mpm | SphMpm | …`) → `Box<dyn CoffeeSolver>`, built from `(Scene, Materials, SolverOptions, GpuContext)`. Powers the runtime solver-switch (`engine` re-builds on the same `Scene`).

## Shared substep conventions (helpers, not enforcement)
Each solver owns its loop, but the following helpers live here so methods don't re-implement them:
- **Multi-rate substepping** — given a frame `dt`, subcycle CflLimited phases (per `SolverDescriptor.stability`) while holding slower fields constant (CFD-DEM pattern). Unconditional solvers run at frame `dt`.
- **Frozen-bed fast path** — a deformation-rate gate; when the bed is quiescent, skip its solve. Used by XPBD and SPH+MPM beds.
- **KE watchdog** — on a per-particle kinetic-energy spike, halve `dt` for a few steps. Robustness under arbitrary user input.

## Build phases & gates
1. Types + trait + a **no-op solver** (returns empty state). *Gate:* compiles; `engine` steps it; `profiler` reports zero passes.
2. Registry + runtime-switch. *Gate:* switch between two no-op variants at runtime.
3. Shared helpers (multi-rate, frozen-bed, KE watchdog). *Gate:* unit-tested in isolation.

## Open questions
- Should `BrewStateView` mandate buffer formats (e.g. `vec4<f32>` positions) or expose typed accessors? Leaning mandated layout for zero-cost vis interop.
- Coupling-split iteration count for hybrids: a per-solver concern, but the helper for the iteration loop may live here.

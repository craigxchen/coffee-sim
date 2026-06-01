# coffeesim — `engine/` Plan

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/engine/` · **Status:** plan
**Depends on:** `solvers/base.rs` (the seam + registry), `options/`, `geometry/`, `utils/` (GpuContext). Depended on by `vis/`, `examples/`.

## Purpose
Scene-level orchestration: build a brew, drive the **one** active solver through the frame loop, and emit canonical state/metrics/profile for `vis` and `profiler`. Owns no physics. Much of Phase 0 lives here.

## Scope
**In:** `Scene`, `Simulator`, `State`, the registry wiring, the runtime solver-switch, the frame loop. **Out:** solver internals; rendering/UI (`vis.md`).

## Components

### `scene.rs` — `Scene`
User-facing brew description, assembled from `options/`: dripper geometry (`geometry/` SDF: V60/Kalita/flat + filter + cup), coffee dose, brew ratio, water temperature, and the **pour schedule** (kettle path, flow rate, bloom timing). Immutable input to `Solver::build`/`reset`.

### `simulator.rs` — `Simulator`
- Holds **one** `Box<dyn CoffeeSolver>` (the selected method) + the `GpuContext`.
- **Frame loop:** read `PourInput` (from `vis`) → `solver.step(dt, input)` → pull `state()`/`metrics()`/`profile()` → hand to `vis` (render) and `profiler`.
- Passes a **frame `dt`**; the solver does its own internal substepping (per `solvers.md` conventions). The simulator does not micro-manage substeps.
- **Runtime solver-switch:** re-`build` a different `SolverKind` from the registry on the **same `Scene`** (the payoff of the seam) — used by `vis/debug` and `examples/compare_solvers`.

### `state.rs` — `State` / `BrewState`
The canonical snapshot decoupling solvers from rendering: the `BrewStateView` handles (canonical particle buffers + optional fields) plus `BrewMetrics`. `vis` and `profiler` only ever see this, never a solver's internals — which is what makes them solver-agnostic.

## Data flow
`options` → `Scene` → `Simulator` builds active solver (registry) → per frame: `vis` input → `solver.step` → `state`/`metrics`/`profile` → `vis` + `profiler`.

## Build phases & gates
1. `Scene` + `geometry` wiring + `GpuContext` init. *Gate:* a scene builds; dripper SDF loads.
2. `Simulator` frame loop driving the **no-op solver**. *Gate:* steps at 60 fps; profiler sees the loop.
3. Registry + runtime-switch. *Gate:* switch solvers live on one scene.
4. `State` snapshot feeding a trivial `vis`. *Gate:* particles render from `BrewStateView`.

## Open questions
- Fixed `dt` vs real-time-clock `dt` for the frame step (fixed is more reproducible for comparison; real-clock is honest for interactivity) — likely fixed during dev/profiling, real-clock for the shipped app.
- Where the pour-schedule playback clock lives (engine vs `vis`): leaning engine, with `vis` able to scrub it.

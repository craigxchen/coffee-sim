# coffeesim — `profiler/` Plan (local, simple)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/profiler/` · **Status:** plan (intentionally minimal for now)
**Depends on:** `utils/` (GpuContext), `solvers/base.rs` (`SolverProfile`). Used by `engine`, `vis/debug`, `examples/compare_solvers`.

## Purpose
Measure where GPU time goes, locally, so method comparisons are settled by measurement rather than priors. **Local-only for now** — browser profiling is deferred until the sim is in good shape. Local profiling is a sound proxy for kernel cost and method ranking, which is what this phase needs; it is *not* trusted for absolute frame budget or a close XPBD-vs-MPM verdict (those need the eventual in-browser recheck).

## Scope
**In:** the `timestamp-query` wrapper, the dispatches-per-frame counter, the `SolverProfile` aggregation, the comparison harness. **Out:** a custom GUI (that's `vis/debug`); browser instrumentation (deferred, noted below).

## Design (deliberately thin)
- **Per-pass timing:** wrap each compute pass in a `wgpu` `timestamp-query` (when the feature is available on the device); resolve the query set; read GPU-side durations into `SolverProfile`. Read the *previous* frame's resolved results to avoid stalling.
- **Dispatch counter:** increment per `dispatch`/`submit`; report **`dispatches_per_frame`** in `SolverProfile`. This is kept even in the minimal version because it is the single variable that predicts the eventual browser/mobile gap, and it makes the later browser pass a quick check rather than a redo. It is also directly actionable (batch XPBD's iteration passes into fewer dispatches).
- **Instruments** (macOS) for deep single-kernel work (occupancy, bandwidth) — manual, occasional, not wired into the loop.

## `examples/compare_solvers`
Run `Xpbd`, `Mpm`, `SphMpm` on the **same `Scene`** with the **same `models/`** (the fairness contract), report per-pass timings + dispatch counts + `BrewMetrics` side by side. This is the referee for the method debates.

## Deferred (browser phase — do not build now)
When the sim is in good shape: in-browser `timestamp-query` + `performance.now()` frame timing + `chrome://tracing`, and **Firefox as the clean reference** (Firefox's WebGPU *is* `wgpu`, so native-vs-Firefox isolates the browser-process/validation tax with the implementation held constant). Bracket against a Chrome (Dawn) capture. The `dispatches_per_frame` metric carried from now makes this a verification step, not new work.

## Build phases & gates
1. `timestamp-query` wrapper + `SolverProfile`. *Gate:* per-pass μs reported for the no-op solver.
2. Dispatch counter. *Gate:* `dispatches_per_frame` reported.
3. `compare_solvers` harness. *Gate:* two solvers profiled on one scene with identical `models`.

## Open questions
- Timestamp resolution/availability varies by device/driver — confirm on the dev machine; fall back to wall-clock per-pass if needed.
- Aggregation window (per-frame vs rolling average) for stable readouts in `vis/debug`.

# Solver Profiler

This branch makes the headless profiler select a solver at runtime instead of
hard-coding one pressure path in the harness. The profiler entrypoint is still
the solver-neutral ignored Rust test:

```text
cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
```

Runtime selection can come from a native profiler binary, `COFFEE_SIM_PROFILE_ARGS`,
or individual environment variables. `COFFEE_SIM_PROFILE_ARGS` accepts CLI-style
flags or kwargs-style key/value pairs. Cargo's test harness rejects unknown
arguments after `--`, so use the feature-gated binary for real command-line
selection and the environment form for ignored-test runs. Examples:

```text
cargo run -p coffee-sim-wasm --features native-profiler --bin profile_solvers -- \
  --scene center_pour --solvers all --frames 120 --warmup 60 --cal 30
cargo run -p coffee-sim-wasm --features native-profiler --bin profile_solvers -- \
  --scene center_pour --solvers all --frames 120 --warmup 60 --cal 30 --require-gpu
cargo run -p coffee-sim-wasm --features native-profiler --bin profile_solvers -- \
  --scene center_pour --solvers all --frames 120 --warmup 60 --cal 30 --verify-reports
cargo run -p coffee-sim-wasm --features native-profiler --bin profile_solvers -- \
  --scene center_pour --solvers all --frames 120 --warmup 60 --cal 30 --dry-run
cargo run -p coffee-sim-wasm --features native-profiler --bin profile_solvers -- \
  --scene center_pour --solvers all --frames 120 --warmup 60 --cal 30 --dry-run-json
cargo run -p coffee-sim-wasm --features native-profiler --bin profile_solvers -- --list-solvers
COFFEE_SIM_PROFILE_ARGS="scene=center_pour solvers=all frames=120 warmup=60 cal=30" \
  cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
COFFEE_SIM_PROFILE_ARGS="--scene water_block --solver mpm:sparse-cg --pressure-operator collocated" \
  cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
```

Use `--require-gpu`, `require_gpu=true`, or
`COFFEE_SIM_PROFILE_REQUIRE_GPU=true` for measurement gates. Without it, a host
with no native adapter prints a skip message so regular non-GPU development
does not fail. After a measured run, use `--verify-reports`,
`verify_reports=true`, or `COFFEE_SIM_PROFILE_VERIFY_REPORTS=true` with the
same scene, solver, frame, and output arguments to verify that every emitted
JSON report belongs to the requested same-scene solver set.

Runnable solver specs are:

- `mpm:rbgs`
- `mpm:jacobi-cg`
- `mpm:sparse-cg`
- `dfsph` / `dfsph:water`
- `xpbd` / `xpbd:gpu`
- `all`

`xpbd:cpu` is intentionally not registered. The profiler backend for XPBD is
GPU-staged: prediction, hash rebuilds, density solve/apply, bounds solve, and
velocity update run as compute passes.

## Modular Boundary

The profiler has one shared run loop in
[`profiler.rs`](../crates/sim-wasm/src/mpm_3d/profiler.rs): warmup,
calibration, measured frames, JSON metadata, output path fan-out, and summary
printing are solver-neutral. Backends implement the small `ProfileBackend`
boundary for:

- one production/calibration step
- one measured/timestamped step
- timestamp query capacity
- solver metadata
- backend-specific bottleneck notes

The MPM pressure variants are selected by `PressureSolverKind` and share the
production MPM schedule. Backend-level experiments such as DFSPH and XPBD use
their own GPU passes while keeping the same selected scene geometry and output
schema.

## Branch Import Audit

Solver development was spread across several local branches. Current status:

- `codex/mpm-sparse-pressure`: ported into this branch as selectable
  `mpm:sparse-cg`, modular pressure pipelines, shared schedule dispatch ops,
  profiler solver registry, and all-solver output fan-out.
- `warmstart-ref`: ported into the modular CG solver as persistent pressure
  warm-start scratch plus the CG fill cache used by matvec. The fill cache is
  stored in the existing `cg` buffer, so it adds no WebGPU storage binding.
- `codex/perf-60hz-tier1`: selectively ported. The load-bearing sparse-stream
  pressure-domain guards are present: fast freefall is excluded from continuum
  pressure, low-diagonal/unstorable cells are dropped from CG, and the
  classify-before-init invariant for cached fill is documented in WGSL.
- `codex/dfsph-water`: selectively ported as the `dfsph` profiler backend and
  GPU DFSPH water kernels. The tiled grid-pressure path from that branch is
  staged but not wired through the shared MPM bind group because it pushed the
  browser over the common 10-storage-buffer WebGPU limit.
- `xpbd-solver-rewrite`: selectively ported as `xpbd:gpu`; the old branch's app
  scaffold was not merged because it predates the modular profiler and shared
  scene app.
- `codex/rbgs-to-jacobi-cg`: selectively ported as selectable
  `mpm:jacobi-cg` instead of replacing RBGS globally. Solver-port guardrails
  that preserve iteration/scene budget parity are represented by tests in
  `mod.rs` and profiler settings tests.
- `codex/perf-60hz`: not merged wholesale. It contains tuning/test-deletion
  commits that are not appropriate for an honest solver comparison branch.

Additional local experiments were checked but are not registered as solver
backends:

- `codex/genesis-pressure-residual`: not a separate solver. The useful
  pressure-residual concept is represented in the shared MPM schedule as
  `pressure.residual` and in the report metrics.
- `codex/pressure-debug-mode`: browser/debug stepping support, not a solver or
  profiler backend.
- `codex/perf-improvements`: web render-loop overhead work, not a solver
  implementation.
- `codex/incompressible-rewrite`, `dry-bed-dual-grid`,
  `integration/dry-bed-mechanics`, and `integration/physics-mainline`: bed,
  filter, and grid-coupling experiments rather than selectable solver methods.
- `codex/genesis-bed-constraints`, `codex/genesis-inflow-slug`,
  `codex/genesis-pourover-plan`, and `codex/pour-stability-follow-up`: scene,
  inflow, and stability experiments; they should only be pulled after their
  physics changes are promoted independently of solver comparison.

## Verification Status

Focused non-GPU checks prove parser/registry behavior and same-scene settings
selection. The native `--dry-run` and versioned `--dry-run-json` modes resolve
solver fan-out, output paths, scene dimensions, and pressure budgets without
requesting a GPU adapter. Dry-run JSON also includes per-solver `solver_run`
metadata matching real profile reports, so the selected same-scene comparison
set can be checked before any GPU work starts. The
ignored profiler smoke with `solvers=all scene=center_pour` currently compiles
in this environment but skips measurement because no native GPU adapter is
available. Real completion still requires running that profiler on a host with
GPU adapter access and verifying all emitted JSON reports for the same scene
with `--verify-reports`.

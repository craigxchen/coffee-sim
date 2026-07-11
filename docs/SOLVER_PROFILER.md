# Solver Profiler

The existing ignored MPM profiler remains available:

```text
cargo test -p coffee-sim-wasm --lib --release profile_mpm_pipeline -- --ignored --nocapture
```

This refactor also adds the solver-neutral entrypoint:

```text
cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
COFFEE_SIM_PROFILE_ARGS="--solver mpm --dry-run" \
  cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
COFFEE_SIM_PROFILE_ARGS="--solver mpm-v2 --frames 120" \
  cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
COFFEE_SIM_PROFILE_ARGS="--list-solvers" \
  cargo test -p coffee-sim-wasm --lib --release profile_solvers -- --ignored --nocapture
```

Runnable solver specs are currently:

- `mpm`
- `mpm-v2` (`twofield` remains an accepted compatibility alias)
- `all`

Legacy `mpm` reports its instrumented timestamp-query pass breakdown. `mpm-v2`
uses the common `FrameSolver` seam and reports production frame wall time through
GPU completion; it does not fabricate a legacy-MPM per-pass breakdown.

`profile_solvers` supports `--solver`, `--solvers`, `--scene`, `--warmup`,
`--frames`, `--cal`, `--dry-run`, `--dry-run-json`, `--require-gpu`, and
`--list-solvers` through `COFFEE_SIM_PROFILE_ARGS`. The old
`COFFEE_SIM_PROFILE_*` environment variables still work for the MPM run.

This PR intentionally does not port PR #17's DFSPH, XPBD, Jacobi-CG, sparse-CG,
staged pressure, or shader physics changes.

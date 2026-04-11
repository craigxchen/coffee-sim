# Performance Optimization Plan

## Goal

Recover performance regressions without compromising the core solver model.

## Implemented Wins

- cached SDF cell classification texture
- encoder `clear_buffer` path instead of runtime `clear_grid` dispatch
- empty-cell early exits in hot grid passes
- runtime-tunable pressure iteration count

## Deferred Work

- timestamp-query profiling
- aggressive RBGS count reduction without measurement
- adaptive substeps before profiling proves the need

## Validation

Primary checks:

```bash
cargo test -p coffee-sim-wasm --lib
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

Perf changes should only be promoted when they preserve:
- physical behavior expected by the current tests
- browser interaction quality
- stable frame time in the active scenes

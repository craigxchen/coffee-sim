# Performance Optimization Plan

## Goal

Recover the dual-velocity-field regression with low-risk hot-path reductions first.

The original draft mixed implemented changes, rejected revisions, and deferred ideas in one place. This version keeps only the current plan:

- land semantics-preserving runtime wins now
- keep solver-quality tradeoffs explicit and deferred
- avoid extra hot-path pipeline complexity unless the measured win justifies it

## Status

### Implemented

1. **Cached SDF cell classification**
   - `classify_cells` now reads a cached `R8Uint` 3D texture for self-cell and neighbor solid checks instead of repeatedly sampling the live SDF texture.
   - The cache is generated on the CPU once when buffers are created, using the same cell-center convention as the shader.
   - This avoids the extra compute pipeline and storage-texture path from the earlier revision while preserving the runtime benefit.

2. **Encoder clears instead of `clear_grid` dispatch**
   - `grid` and `grid_vel` are zeroed with `encoder.clear_buffer(...)` before the compute pass begins.
   - The old `clear_grid` dispatch is removed from the runtime path.

3. **Empty-cell early exit in `boundary_project`**
   - The pass now checks water/solid grid mass first and skips SDF/box work when both are empty.

4. **Store-free empty-cell early exit in `grid_update`**
   - The pass returns immediately when both water and solid mass are zero.
   - This relies on the pre-pass buffer clears to provide the correct zero state.

5. **Runtime-tunable pressure iterations**
   - `pressure_rbgs_pairs` is now part of `MpmSettings`, defaulting to `20`.
   - No default solver-quality reduction is included in this phase.

### Deferred

1. **Timestamp-query profiling**
   - Useful, but separate from the hot-path fixes above.
   - Not required to realize the immediate dispatch and per-cell savings.

2. **RBGS iteration reduction**
   - Only worth changing under measured A/B testing with divergence metrics.

3. **Adaptive substeps**
   - Deferred until we have cleaner profiling and a clearer CFL policy.

## Implemented Design

### 1. Cached SDF classification

Use a grid-sized `R8Uint` 3D texture as a static cell-solid mask:

- `0`: open cell
- `1`: solid cell

Implementation choice:

- generate on the CPU from the already-generated SDF volume
- upload once with the MPM buffers
- sample in `classify_cells` through `textureLoad`

Why this version:

- no extra compute pipeline
- no storage-texture bind path
- no per-reset GPU precompute step
- same runtime benefit in the pressure-classification hot path

### 2. Buffer clears

Replace the runtime `clear_grid` dispatch with encoder-level clears:

- `encoder.clear_buffer(&grid, ..)`
- `encoder.clear_buffer(&grid_vel, ..)`

Keep `bed_lookup_clear` as a shader dispatch because its sentinel is `-1`, not `0`.

### 3. Early returns

Apply early exits only where they are truly free:

- `boundary_project`: skip entirely if both velocity fields are empty
- `grid_update`: return with no stores if both masses are zero

## Validation

Minimum checks for this phase:

```bash
cargo test -p coffee-sim-wasm --lib mpm_3d::physics_tests::pipelines_fit_within_required_limits -- --exact
cargo test -p coffee-sim-wasm --lib mpm_3d::physics_tests::water_bed_mass_conservation -- --exact
cargo test -p coffee-sim-wasm --lib
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

Added coverage:

- CPU-side SDF class generation matches live cell-center sampling

## Next Perf Work

Only do these after measuring the current build:

1. Add timestamp-query profiling when supported, with CPU wall-clock fallback.
2. Benchmark `pressure_rbgs_pairs = 16` and `12` against the current default of `20`.
3. Revisit adaptive substeps only if profiling shows pressure iterations are no longer the main cost.

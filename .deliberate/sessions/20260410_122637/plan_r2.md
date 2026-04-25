I keep hitting a permission wall on writing PLAN.md. Let me output the revised plan directly instead.

---

# Revised Performance Optimization Plan: Dual Velocity Field Regression

## Summary

After adding dual velocity grids (separate water and solid fields), the simulation dropped below interactive frame rates. The per-frame dispatch structure is **~55 passes x 5 substeps = 275 GPU dispatches/frame** across a 64x80x64 grid (327,680 cells). This plan is split into two phases: **Phase 1** contains safe, semantics-preserving GPU-budget reductions; **Phase 2** contains controlled solver tuning that trades numerical quality for performance under instrumented A/B testing.

## How Each Critique Was Addressed

1. **Substeps 5->3 CFL math was wrong** -- Agreed. `dt` clamps to `1/30`, not `1/60`, so CFL at substeps=3 is ~1.52, above stable. Substep reduction is moved to Phase 2 as an adaptive option only. Substeps stay at 5.

2. **`bed_lookup_clear` cannot use `clear_buffer()`** -- Agreed. Sentinel is `-1`, not `0`. `clear_buffer()` now applies only to `grid` and `grid_vel` buffers. `bed_lookup_clear` shader dispatch remains.

3. **`grid_update` early-exit was effectively a no-op** -- Agreed the absorption path is already guarded. Replaced with a true empty-cell early return (both `mass_w` and `mass_s` <= 1e-6) that skips fixed-point decode, gravity, drag, clamping, and writes.

4. **`classify_cells` SDF hotspot was missed** -- Agreed. Added precomputed `sdf_class` buffer: a one-time pass writes per-cell solid/open classification, then `classify_cells` reads the buffer instead of doing up to 7 trilinear SDF probes per cell per substep. This is the biggest new win in the revised plan.

5. **RBGS reduction should not be bundled with substep reduction** -- Agreed. RBGS iteration count is made runtime-tunable with default kept at 20. A/B testing happens only in Phase 2, after Phase 1 safe wins are measured, so regressions can be attributed to the correct change.

6. **Testing was too coarse** -- Agreed. Testing now uses both `benchmark_free_stream` (no bed) and `benchmark_center_pour` (with bed) presets to separate bed-specific costs from dense-grid costs. Per-pass timing is required before/after each change.

## [REVISED] Files to Modify

### `crates/sim-wasm/src/mpm_3d/shader.rs`

1. **`boundary_project` (line 1097) -- skip empty cells.** Add early-exit before SDF work: read `gv_w` and `gv_s` mass via `grid_vel`, if both <= 1e-6, return immediately. The SDF trilinear interpolation at lines 1110-1111 currently runs unconditionally on all ~328k cells. Move velocity reads before SDF reads (trivial reorder since neither depends on SDF output).

2. **[REVISED] `classify_cells` (line 800) -- precomputed solid-cell mask.** The static SDF is sampled on every cell every substep (line 825), and the divergence stencil does up to 6 additional neighbor SDF probes (lines 896-918). Add a one-time `precompute_sdf_class` pass that writes a per-cell `u32` buffer (`0` = open, `1` = solid). Then `classify_cells` reads this buffer instead of calling `sample_sdf` for its own cell and 6 neighbors. Eliminates **up to 7 trilinear SDF reads per fluid cell per substep**.

3. **[REVISED] `grid_update` (line 672) -- true empty-cell early return.** When both `mass_w <= 1e-6` and `mass_s <= 1e-6`, zero out `scratch_absorbed_idx`, `grid_vel[idx]`, and `grid_vel[grid_vel_solid_idx(idx)]`, then return. Skips fixed-point decode, gravity, drag, clamping on ~80%+ of cells.

### `crates/sim-wasm/src/mpm_3d/mod.rs`

1. **[REVISED] Replace `clear_grid` with `encoder.clear_buffer()`** for `grid` and `grid_vel` only. Keep `bed_lookup_clear` dispatch (sentinel is `-1`). Saves 1 dispatch + 5,120 workgroups/substep.

2. **[REVISED] Keep substeps at 5.** CFL at substeps=3 is ~1.52, above stability. Adaptive substeps deferred to Phase 2.

3. **[REVISED] Make `PRESSURE_RBGS_PAIRS` runtime-tunable** on `MpmSettings`, default 20. No default reduction in this plan.

### [REVISED] `crates/sim-wasm/src/mpm_3d/state.rs`

1. **Add `sdf_class` buffer** -- `u32`-per-cell for precomputed solid/open classification. Add to `MpmBuffers` and bind group.

### `crates/sim-wasm/src/mpm_3d/pipelines.rs`

1. **Add `precompute_sdf_class` pipeline** -- new compute entry point, same bind group as cell-level passes.

## Files to Create

None.

## Dependencies

None.

## [REVISED] Implementation Order

### Phase 1: Safe, Semantics-Preserving Wins

1. **Add `sdf_class` buffer** in `state.rs` and bind group in `pipelines.rs`
2. **Add `precompute_sdf_class` shader entry point** in `shader.rs`
3. **Dispatch `precompute_sdf_class` once** at init (and on reset) in `mod.rs` -- NOT per substep
4. **Rewrite `classify_cells` to use `sdf_class` buffer** -- eliminates up to 7 SDF reads per cell per substep
5. **Add early-exit to `boundary_project`** -- skip SDF on cells with no water or solid mass
6. **Add early-exit to `grid_update`** -- skip work on fully empty cells
7. **Replace `clear_grid` dispatch with `encoder.clear_buffer()`** for `grid` and `grid_vel` only; keep `bed_lookup_clear`
8. **Profile Phase 1** -- both `benchmark_free_stream` and `benchmark_center_pour`

### Phase 2: Controlled Solver Tuning (after Phase 1 measured)

9. **Make `PRESSURE_RBGS_PAIRS` runtime-tunable** -- default 20
10. **A/B test reduced RBGS** (12, 16) with `METRIC_MAX_ABS_DIV_IDX`; adopt lower default only if max divergence stays within 2x of baseline
11. **Consider adaptive substeps** -- reduce to 3-4 only when max particle velocity keeps CFL below 0.9

## [REVISED] Testing Strategy

1. **Per-pass profiling**: Wall-clock timing over 200 frames using both `benchmark_free_stream` (no bed) and `benchmark_center_pour` (with bed) to separate bed-specific vs dense-grid costs.
2. **Mass conservation**: Total mass error < 0.1% after all changes.
3. **Visual regression**: Pour sequence identical -- Phase 1 changes are skip-only on empty cells.
4. **SDF classification correctness**: Verify `sdf_class` matches live `sample_sdf < 0.0` for all cells on init.
5. **Phase 2 isolation**: Each Phase 2 change tested independently against Phase 1 baseline. Monitor `METRIC_MAX_ABS_DIV_IDX` and `METRIC_MASS_OVERFLOW_FIRES_IDX`.

## [REVISED] Risks & Edge Cases

| Risk | Severity | Mitigation |
|------|----------|------------|
| `sdf_class` stale after geometry change | Low | Re-dispatch on reset; geometry is static during simulation |
| `clear_buffer` for `grid` (atomic i32) | None | Zero bits = `0i32` -- correct |
| `clear_buffer` for `grid_vel` (vec4 f32) | None | Zero bits = `0.0f32` -- correct |
| `bed_lookup` cannot use `clear_buffer` | Caught | Sentinel `-1` != `0`; keep shader clear |
| `boundary_project` early-exit skips needed work | None | Inner mass guards already skip these cells |
| `grid_update` early-exit misses side effects | Low | `scratch_absorbed_idx` zeroed; `grid_vel` zeroed explicitly |
| Phase 2 RBGS reduction causes artifacts | Medium | Default stays at 20; reduce only after A/B |
| Phase 2 adaptive substeps instability | Medium | 5 remains default; adaptive requires opt-in + CFL check |

---

Want me to save this to PLAN.md? You'll need to approve the write permission. Or I can proceed directly to implementation of Phase 1.

Permission denied for file write. Here's the complete revised plan — I'll output it directly:

---

# Revised Performance Optimization Plan: Dual Velocity Field Regression

## Summary

After adding dual velocity grids (separate water and solid fields), the simulation dropped below interactive frame rates. The per-frame dispatch structure is **~55 passes x 5 substeps = 275 GPU dispatches/frame** across a 64x80x64 grid (327,680 cells). This plan is split into two phases: **Phase 1** contains safe, semantics-preserving GPU-budget reductions; **Phase 2** contains controlled solver tuning that trades numerical quality for performance under instrumented A/B testing.

## How Each Critique Was Addressed

1. **`sdf_class` storage buffer violates binding limit** — Agreed. The bind group is at exactly 10 storage buffers (`pipelines.rs:27-75`) and `required_limits()` caps at 10 (`mod.rs:38`). A new storage buffer won't fit. **Fix:** use a 3D `r8uint` texture instead. Textures use a separate binding type (`wgpu::BindingType::Texture`) and don't count against `max_storage_buffers_per_shader_stage`. The existing SDF texture at binding 5 proves the pattern works. The new `sdf_class` texture is added at binding 12 as a second texture binding. No buffer limit or test changes needed.

2. **Profiling strategy is not actionable** — Agreed. Wall-clock over 200 frames of a time-evolving pour conflates workload change with optimization effect, and there are no GPU timestamp queries. **Fix:** (a) add `wgpu::Features::TIMESTAMP_QUERY` when the adapter supports it, writing begin/end timestamps around the compute pass; (b) when timestamps are unavailable, use whole-frame `Instant::now()` deltas with pass-toggle isolation (disable one pass, measure delta); (c) run benchmarks after a fixed warm-up window (2s simulated time) and measure over a 1s steady-state window; (d) log active water particle count and fluid-cell count alongside timings so workload is tracked.

3. **`grid_update` early return should be store-free** — Agreed. The proposed early return redundantly wrote `scratch_absorbed_idx`, `grid_vel[idx]`, and `grid_vel[solid_idx]` with zeros, but `clear_grid` already zeroed all of those (`shader.rs:544-553`). **Fix:** load `mass_w` and `mass_s` first; if both `<= 1e-6`, return immediately with *no writes*. The clear step guarantees correct zero state. This also means `encoder.clear_buffer()` for `grid` and `grid_vel` must happen *before* the compute pass begins (since `clear_buffer` is an encoder-level command, not a compute-pass command), which restructures the pass setup in `mod.rs:308-312`.

4. **"Phase 1 is semantics-preserving" claim is too strong for SDF cache** — Agreed. A cached mask is only equivalent if it uses the exact same cell-center convention and the SDF is truly static during simulation. **Fix:** (a) the `precompute_sdf_class` pass uses the identical `cell_center` formula from `classify_cells` (`shader.rs:823-824`); (b) add a debug-mode cell-by-cell equivalence assertion that runs for the first 10 substeps comparing cached vs live SDF; (c) keep the SDF-cache path behind a uniform toggle (`u.use_sdf_cache`) so it can be A/B tested; (d) re-dispatch on reset since geometry is static during simulation.

## Files to Modify

### `crates/sim-wasm/src/mpm_3d/shader.rs`

1. **`boundary_project` (line 1097) — skip empty cells.** Read `grid_vel[idx].w` (water mass) and `grid_vel[grid_vel_solid_idx(idx)].w` (solid mass) *before* the SDF work at lines 1110-1111. If both `<= 1e-6`, return immediately. The existing inner guards at lines 1119 and 1126 already skip velocity projection for zero-mass fields, so this just hoists the check above the SDF sample and gradient computation.

2. **[REVISED] `classify_cells` (line 800) — precomputed SDF class via 3D texture.** Add a new `precompute_sdf_class` entry point that writes a `r8uint` 3D texture with `0u` = open, `1u` = solid, using the same cell-center formula as line 823-824. Then `classify_cells` reads `textureLoad(sdf_class_tex, cell_coords, 0).r` instead of calling `sample_sdf(cell_center)` for the self-cell check. For the 6 neighbor-solid checks in the divergence stencil (lines 895-918), read `textureLoad` at neighbor coords instead of `sample_sdf` at neighbor positions. This eliminates up to 7 trilinear SDF interpolations per fluid cell per substep. Gated behind `u.use_sdf_cache` uniform flag — when `0`, falls through to the existing live-SDF path.

3. **[REVISED] `grid_update` (line 672) — store-free empty-cell early return.** After loading `mass_w` (line 677) and `mass_s` (line 688), if both `<= 1e-6`, return immediately with *no writes*. The `clear_grid` pass (or `encoder.clear_buffer()`) already set `grid_vel[idx]`, `grid_vel[solid_idx]`, and `scratch_absorbed_idx` to zero. This skips fixed-point decode, gravity, drag, the 27-neighbor bed lookup, clamping, and all stores on ~80%+ of cells.

### `crates/sim-wasm/src/mpm_3d/mod.rs`

1. **[REVISED] Replace `clear_grid` dispatch with `encoder.clear_buffer()` for `grid` and `grid_vel`.** These buffers contain `0i32` atomics and `0.0f32` vec4s respectively — zero bits is correct for both. `encoder.clear_buffer()` is an encoder-level command, so it must run *before* `begin_compute_pass`. Restructure the substep loop: call `encoder.clear_buffer(&buffers.grid, ..)` and `encoder.clear_buffer(&buffers.grid_vel, ..)`, *then* begin the compute pass. The `clear_grid` shader dispatch and pipeline are removed. Saves 1 dispatch + 5,120 workgroups per substep.

2. **Keep `bed_lookup_clear` dispatch.** Its sentinel is `-1` (`0xFFFFFFFF`), not `0`. `clear_buffer` writes zeros, so this cannot be replaced.

3. **Keep substeps at 5.** `dt` clamps to `1/30`, not `1/60`, so CFL at substeps=3 is ~1.52, above stable. Adaptive substeps deferred to Phase 2.

4. **[REVISED] Make `PRESSURE_RBGS_PAIRS` runtime-tunable** on `MpmSettings`, default 20. No default reduction in Phase 1.

5. **[REVISED] Add `precompute_sdf_class` dispatch** at init and on reset (not per-substep). Uses a write-only storage texture for the output. Dispatch once with `cell_wg` workgroups.

6. **[REVISED] Add profiling infrastructure.** Request `wgpu::Features::TIMESTAMP_QUERY` when supported. When available, create a `QuerySet` and write timestamps around the compute pass. Expose frame timing + active particle/fluid-cell counts via the existing metrics pathway. When timestamps unavailable, measure whole-frame `Instant::now()` deltas.

### [REVISED] `crates/sim-wasm/src/mpm_3d/state.rs`

1. **Add `sdf_class` 3D texture** — format `R8Uint`, dimensions matching grid dims, usage `TEXTURE_BINDING | STORAGE_BINDING`. Add `sdf_class_texture` and `sdf_class_view` to `MpmBuffers`. No storage buffer limit impact.

2. **Add `use_sdf_cache` flag** to `MpmUniforms` (pack into an existing padding slot or `time_params`).

### `crates/sim-wasm/src/mpm_3d/pipelines.rs`

1. **Add `sdf_class` texture binding at slot 12** — `wgpu::BindingType::Texture` with `sample_type: Uint`, `view_dimension: D3`. Does not affect `max_storage_buffers_per_shader_stage`.

2. **Add `precompute_sdf_class` pipeline** — new compute entry point. Needs the SDF texture (binding 5) as input and `sdf_class` texture as a write-only storage texture output. May use a separate single-use bind group layout to keep the hot-path layout unchanged, or add to the main layout since textures don't affect the storage buffer cap.

3. **Remove `clear_grid` pipeline** — no longer dispatched.

### `crates/sim-wasm/src/mpm_3d/physics_tests.rs`

1. **No changes to `pipelines_fit_within_required_limits`** — storage buffer count stays at 10. The new `sdf_class` is a texture, not a storage buffer.

2. **[REVISED] Add SDF cache equivalence test** — for the first N substeps in debug mode, compare `textureLoad(sdf_class_tex, ...)` against live `sample_sdf(cell_center) < 0.0` for all cells and assert exact match.

## Files to Create

None.

## Dependencies

None.

## [REVISED] Implementation Order

### Phase 1: Safe, Semantics-Preserving Wins

1. **Add `sdf_class` 3D texture** in `state.rs` — `R8Uint`, grid-sized, `TEXTURE_BINDING | STORAGE_BINDING`
2. **Add `sdf_class` texture binding** at slot 12 in `pipelines.rs`
3. **Add `precompute_sdf_class` shader entry point** in `shader.rs` — same cell-center formula as `classify_cells` line 823-824, writes `1u` if `sample_sdf < 0.0`, else `0u`
4. **Add `precompute_sdf_class` pipeline and dispatch once** at init and on reset in `mod.rs`
5. **Add `use_sdf_cache` uniform toggle** in `state.rs`, default enabled
6. **Rewrite `classify_cells` to use `sdf_class` texture when toggle is on** — `textureLoad` replaces `sample_sdf` for self-cell (line 825) and all 6 neighbor checks (lines 895-918)
7. **Add SDF cache equivalence assertion** — debug-mode test comparing cached vs live for first 10 substeps
8. **Add early-exit to `boundary_project`** — read both masses before SDF work, skip when both empty
9. **Add store-free early-exit to `grid_update`** — return immediately after mass loads when both `<= 1e-6`
10. **Replace `clear_grid` dispatch with `encoder.clear_buffer()`** — restructure substep loop to do clears before `begin_compute_pass`; remove `clear_grid` shader/pipeline
11. **Add profiling infrastructure** — timestamp queries when supported, `Instant::now()` fallback, log particle + fluid-cell counts
12. **Profile Phase 1** — run both `benchmark_free_stream` and `benchmark_center_pour` after 2s warm-up, measure 1s steady-state window

### Phase 2: Controlled Solver Tuning (after Phase 1 measured)

13. **Make `PRESSURE_RBGS_PAIRS` runtime-tunable** — default 20, exposed on `MpmSettings`
14. **A/B test reduced RBGS** (12, 16) with `METRIC_MAX_ABS_DIV_IDX`; adopt lower default only if max divergence stays within 2x of baseline
15. **Consider adaptive substeps** — reduce to 3-4 only when max particle velocity keeps CFL below 0.9

## [REVISED] Profiling Strategy

### Infrastructure

- **GPU timestamps (preferred):** Request `wgpu::Features::TIMESTAMP_QUERY` at device creation. When available, use `wgpu::ComputePassTimestampWrites` to bracket the compute pass. Report per-frame GPU time in the metrics/HUD pathway.
- **CPU fallback:** When timestamps are unavailable, use `std::time::Instant` around `queue.submit()` + `device.poll(Wait)` to get synchronous frame time. This captures total GPU work but not per-pass breakdown.
- **Pass-toggle isolation:** To attribute cost to specific passes, add `MpmSettings` booleans that skip individual dispatches (e.g., `skip_pressure_solve`, `skip_boundary_project`). Measure whole-frame delta with one pass disabled vs baseline. This is coarser than per-pass timestamps but works on all hardware.

### Benchmark Protocol

- **Warm-up:** Run simulation for 2s simulated time before starting measurement. This ensures the pour has reached steady state with stable particle counts.
- **Measurement window:** Collect frame times over 1s simulated time (steady-state).
- **Workload logging:** Every measurement frame, log: active water particle count, active bed particle count, fluid-cell count (from metrics), frame time.
- **Presets:** Use `benchmark_free_stream` (no bed, isolates grid/particle cost) and `benchmark_center_pour` (with bed, captures coupling cost).
- **Comparison:** Before/after each Phase 1 change, run both presets. Report mean and p95 frame time with particle/cell counts.

## [REVISED] Risks & Edge Cases

| Risk | Severity | Mitigation |
|------|----------|------------|
| `sdf_class` texture stale after geometry change | Low | Re-dispatch on reset; geometry is static during simulation |
| `sdf_class` cell-center mismatch | Medium | Uses identical formula as `classify_cells:823`; debug equivalence test for first 10 substeps; A/B toggle to disable |
| `clear_buffer` for `grid` (atomic i32) | None | Zero bits = `0i32` — correct |
| `clear_buffer` for `grid_vel` (vec4 f32) | None | Zero bits = `0.0f32` — correct |
| `clear_buffer` requires pass restructuring | Low | Move clears before `begin_compute_pass`; straightforward since clears have no data dependency on prior pass |
| `bed_lookup` cannot use `clear_buffer` | Caught | Sentinel `-1` != `0`; keep shader clear |
| `boundary_project` early-exit skips needed work | None | Inner mass guards at lines 1119/1126 already skip these cells |
| `grid_update` store-free return leaves stale data | None | `clear_grid` / `clear_buffer` already zeroed all slots; no stores needed |
| Phase 2 RBGS reduction causes artifacts | Medium | Default stays at 20; reduce only after A/B with divergence metric |
| Phase 2 adaptive substeps instability | Medium | 5 remains default; adaptive requires opt-in + CFL check |
| Texture binding slot 12 exceeds adapter limits | Very low | `max_bindings_per_bind_group` default is 1000; 13 bindings is well within budget |

---

Want me to save this to `PLAN.md`, or should I proceed directly to implementation of Phase 1?

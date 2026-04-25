Here's the implementation plan:

---

# Performance Optimization Plan: Dual Velocity Field Regression

## Summary

After adding dual velocity grids (separate water and solid fields), the simulation dropped below interactive frame rates. The per-frame dispatch structure is **~55 passes × 5 substeps = 275 GPU dispatches/frame** across a 64×80×64 grid (327,680 cells). The biggest costs are: (1) **40 pressure RBGS dispatches per substep** (the dominant cost, and reducible); (2) `grid_update` runs a 27-neighbor bed-lookup on *every* cell even though ~90%+ have no bed; (3) `boundary_project` samples the SDF on empty cells; (4) `clear_grid` does 10 individual writes per cell where a bulk GPU memset would be faster; (5) **5 substeps** is conservative and can likely go to 3. These target the GPU compute budget without changing physics semantics.

## Files to Modify

### `crates/sim-wasm/src/mpm_3d/shader.rs`

1. **`grid_update` (line 672) — early-exit for cells with no bed presence.** The 27-neighbor `bed_lookup` scan + absorption arrays are initialized and iterated on every cell. The inner gate `mass_w > 1e-6 && mass_s > 1e-6` (line 700, 713) already exists but comes *after* array declaration and init loop at lines 719-726. Move the `mass_s > 1e-6` check to *before* the array initialization so the entire absorption block (lines 714-791) is skipped for cells with no solid mass. This eliminates 27 `bed_lookup_load` calls + deduplication loop on ~90% of cells.

2. **`boundary_project` (line 1097) — skip empty cells.** Add `let gv_w = grid_vel[idx]; let gv_s = grid_vel[grid_vel_solid_idx(idx)]; if gv_w.w <= 1e-6 && gv_s.w <= 1e-6 { return; }` *before* the `sample_sdf` / `sdf_gradient` calls at lines 1110-1111. Currently SDF is sampled for every cell including empty air. This avoids expensive trilinear texture reads on the majority of cells.

3. **`clear_grid` (line 540) — can be removed entirely** once Rust-side `clear_buffer` replaces it (see mod.rs below). Keep the entry point as dead code for debug use.

### `crates/sim-wasm/src/mpm_3d/mod.rs`

1. **Reduce substeps from 5 to 3** (line 113, `substeps: 5`). CFL check: `v_cap × sub_dt / dx = 30 × (1/180) / 0.21875 ≈ 0.76 < 1`. This is the **single highest-leverage change** — cuts total dispatches from ~275 to ~165 per frame (~40% reduction).

2. **Replace `clear_grid` + `bed_lookup_clear` dispatches with `encoder.clear_buffer()`** (lines 318-332). Before `encoder.begin_compute_pass()`, call:
   - `encoder.clear_buffer(&self.buffers.grid, 0, None)` — zeros 8×327k×4 bytes
   - `encoder.clear_buffer(&self.buffers.grid_vel, 0, None)` — zeros 2×327k×16 bytes
   - `encoder.clear_buffer(&self.buffers.bed_lookup, 0, None)`
   
   Then remove the `clear_grid` and `bed_lookup_clear` pipeline dispatches from the compute pass. GPU memset is faster than per-thread atomic stores. This saves 2 dispatches × 5,120 workgroups per substep.

3. **Reduce `PRESSURE_RBGS_PAIRS` from 20 to 12** (line 357). This cuts 16 dispatches per substep (48 fewer per frame at 3 substeps). For a 64×80×64 grid, 12 RBGS pairs gives adequate convergence. Optionally make this a field on `MpmSettings` so it's tunable without recompilation.

### `crates/sim-wasm/src/mpm_3d/state.rs`

No changes needed. Buffer sizes are correct for the dual-field architecture.

## Files to Create

None.

## Dependencies

None.

## Testing Strategy

1. **Performance**: Time `step_frame` over 100 frames with ~20k water + ~12k bed particles. Target ≤16ms/frame at 60 FPS.
2. **Mass conservation**: Total mass error < 0.1% after all changes.
3. **CFL stability**: Max inflow for 10s simulated time at substeps=3 — no NaN, no runaway velocities, check `METRIC_MASS_OVERFLOW_FIRES_IDX`.
4. **Pressure convergence**: With 12 RBGS pairs, confirm max divergence (`METRIC_MAX_ABS_DIV_IDX`) stays within 2× of 20-iteration baseline.
5. **Visual regression**: Pour sequence should look identical; minor pooling differences from fewer pressure iterations acceptable if no visible artifacts.

## Risks & Edge Cases

| Risk | Severity | Mitigation |
|------|----------|------------|
| Substeps=3 instability at high inflow velocity | Medium | Particle-level SDF projection in `g2p` (lines 1250-1254) is a safety net; CFL=0.76 is within stable range; monitor overflow metrics |
| Fewer pressure iterations → visible compression | Low | Make iteration count a tunable setting; bump back to 16-20 if artifacts appear |
| `clear_buffer` semantics | None | Zero bits = `0i32` for atomics, `0.0f32` for vec4s — both correct |
| Early exit in `grid_update` changes absorption | None | Gate on `mass_s <= 1e-6` is semantically identical to existing inner check; no code between current location and the check modifies `mass_s` |

## Implementation Order

1. **Reduce substeps 5→3** in `mod.rs:113` — one-line change, ~40% fewer dispatches
2. **Replace `clear_grid`/`bed_lookup_clear` with `encoder.clear_buffer()`** in `mod.rs:318-335` — removes 2 dispatches/substep
3. **Add early-exit to `grid_update`** in `shader.rs:714` — skip 27-neighbor loop on ~90% of cells
4. **Add early-exit to `boundary_project`** in `shader.rs:1097` — skip SDF sampling on empty cells
5. **Reduce `PRESSURE_RBGS_PAIRS` 20→12** in `mod.rs:357`
6. **Profile and validate** — run benchmarks + conservation checks

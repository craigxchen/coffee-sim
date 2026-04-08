# Codebase Audit — Open Issues

Audit scope: last 5 commits on `codex/incompressible-rewrite` (566b3cf..b38e2fa) — pressure projection rewrite, momentum preservation refinements, MPM polish, legacy stack removal.

Date: 2026-04-07

## Summary

- 21 total findings: 1 CRITICAL, 6 HIGH, 8 MEDIUM, 6 LOW
- 5 auto-fixed during audit
- 16 open issues remain (listed below)
- Workspace builds clean; 23 tests passing

## Open Issues

### HIGH

#### H4. Bed-coupled projection sink and particle absorption are inconsistent
**File:** `crates/sim-wasm/src/mpm_3d/shader.rs:136-151, 515-517, 860-917`

`classify_cells` now subtracts a predicted bed sink from divergence for `CELL_BED_COUPLED`, but `bed_coupling` still performs the old direct particle-side absorption and retirement path independently. That means the solver is effectively accounting for bed intake twice conceptually: once in the pressure RHS and again by physically removing particle mass later in the same substep. The two paths also use different local information (`cell_mass` in the projection pass versus particle mass/speed thresholds in `bed_coupling`), so the sink seen by projection is not the sink actually applied to the particles.

**Visible symptom:** `Free Stream` can form cup height while the default bed scene still collapses into a thin base layer with no lasting perched water above the bed.

**Fix:** Make one path authoritative. Either derive actual absorption from the projection sink for bed-coupled cells, or sharply reduce/directly gate particle-side absorption while projection is enabled until Milestone 2 rewrites bed coupling around shared grid-side exchange.

---

#### H5. Fixed-cost projection/substep budget causes late-stage slowdown even after emission stops
**File:** `crates/sim-wasm/src/mpm_3d/mod.rs:192-199, 269-283`

The solver continues to run a fixed `substeps = 5` and `8` RBGS red/black pairs per substep even when the scene is nearly static. That means settled late-stage states still pay for `clear_grid`, `bed_coupling`, `p2g`, `grid_update`, `boundary_project`, `classify_cells`, 16 projection passes, `g2p`, `bed_dynamics`, and `prepare_render` every frame. Once particles pack into a dense cup-floor layer, the workload also becomes worse for atomics contention in `p2g`, so FPS can degrade over time even when particle count is flat.

**Fix:** Add a low-motion fast path: adaptive projection iterations and/or substeps, plus a sleeping path for settled cup particles. Instrument max occupied cell mass / occupied fluid cells to confirm dense-state contention is the dominant cost before adding more heuristics.

---

#### H1. Silent pressure/divergence clamp in WGSL
**File:** `crates/sim-wasm/src/mpm_3d/shader.rs:117-129`

`pressure_store` and `divergence_store` clamp to ±2048 before encoding to `i32` via `FP_SCALE = 1048576`. With `MAX_VELOCITY=30` and the smallest `dt = 1/(30*5) ≈ 6.7 ms`, an interior cell with full divergence near the velocity cap produces `rhs = div/dt ≈ 30/(6.7e-3 * dx)`, which can exceed 2048 by an order of magnitude under impact conditions. The pressure solve quietly under-corrects and divergence persists with no error signal.

**Fix:** Instrument with a debug counter that increments when a clamp fires; raise the cap (or scale rhs by `dx²`) so the encode range is derived from `MAX_VELOCITY`, `dx`, and the worst-case substep, not a magic number.

---

#### H2. V60 obstacle constants hardcoded in WGSL
**File:** `crates/sim-wasm/src/mpm_3d/shader.rs:217-243`

`resolve_scene_obstacles` hardcodes `cone_top_y=3.0`, `cone_bot_y=-3.0`, `mix(0.8, 4.5, t)`, `cup_radius=3.0`, `floor_y=-8.0`. These duplicate `MpmSettings::default_v60()` constants. Any benchmark scene that swaps obstacles still runs against the V60 numbers in the shader, so collision and SDF disagree.

**Mitigation already in place:** Rust unit test `default_v60_shader_constants_in_sync` pins the Rust-side defaults so a future `Obstacle` tweak fails the test.

**Architectural fix needed:** Push the obstacle parameters through uniforms instead of hardcoding them in WGSL.

---

#### H3. Pressure RBGS has no convergence check
**File:** `crates/sim-wasm/src/mpm_3d/mod.rs:269-284`

Pressure projection runs a hardcoded 8 RBGS sweeps each substep with no convergence check, no residual report, and no termination heuristic. Combined with the silent ±2048 clamp (H1), the divergence-free constraint can drift quietly.

**Fix:** Add a residual probe (one more reduction kernel) and either a configurable iteration count, or early-out when residual drops below a threshold; expose the value to the JS panel.

---

### MEDIUM

#### M7. Bed deformation is still driven by impact heuristics, not pressure/wetness
**File:** `crates/sim-wasm/src/mpm_3d/shader.rs:951-1010`

`bed_dynamics` still derives motion mainly from sampled grid velocity/mass and an `impact_v * bed_impact()` kick, then restores toward rest with a spring. There is no direct use of projected pore pressure, overburden, or local sink/retained-water load. As a result, the bed can crater or rebound in the wrong places: the visually wettest region is not necessarily the region that deforms the most.

**Fix:** Tie deformation to bed-coupled pressure / retained-water state instead of impact velocity alone. At minimum, reduce spring-back in saturated regions and incorporate local bed-held water or pressure load before Milestone 7’s effective-stress model lands.

---

#### M8. Bed lookup is static and will drift as the bed deforms
**File:** `crates/sim-wasm/src/mpm_3d/bed.rs:149-222`

`build_cell_lookup` is only run at init/reset, but bed particles are then moved in `bed_dynamics`. As deformation grows, water/bed coupling continues to use stale nearest-bed indices, so coupling can occur at the wrong cells and miss the actual wettest/deformed regions. This is already called out in the plan, but it is not tracked as an open code issue here.

**Fix:** Rebuild the lookup every substep or every N substeps, using the existing control-volume logic before attempting a more elaborate hashed structure.

---

#### M1. Magic constants in sparse-ballistic G2P override
**File:** `crates/sim-wasm/src/mpm_3d/shader.rs:765-797`

The sparse-ballistic G2P override stacks two corrections (support_ratio path then density_ratio path) with magic constants `1.15`, `0.95`, `0.72`, `0.88`, `0.65`, `0.75` and no derivation comment. Both branches mutate the same `new_v` and `new_C*` and could over-correct. There is no test that pins the behavior.

**Fix:** Document each constant, simplify to one preserving function, and add a unit test that locks the velocity-blend output for representative `support_ratio`/`density_ratio` pairs.

---

#### M2. Dead `scratch_residual_idx` write
**File:** `crates/sim-wasm/src/mpm_3d/shader.rs:411`

`scratch_residual_idx` (`grid_mom_y`) is written to 0 in `classify_cells` but never read elsewhere. Dead state; the alias clutters the buffer layout for no benefit.

**Fix:** Drop the write and the helper, or actually use it for the convergence probe in H3.

---

#### M3. `InflowState::emit_particles` takes 9 arguments
**File:** `crates/sim-wasm/src/mpm_3d/inflow.rs:108`

`InflowState::emit_particles` takes 9 arguments, including `&Queue`, `&MpmBuffers`, particle counters, and limits. clippy `too_many_arguments` flags this; more importantly, the call site in `mod.rs:199-208` is already a struct-level operation.

**Fix:** Introduce an `EmitContext { queue, buffers, counts, max_particles }` so future call sites cannot accidentally pass the wrong count.

---

#### M4. Undocumented `volume_to_ml` scale factor
**File:** `crates/sim-wasm/src/mpm_3d/inflow.rs:14-22`

`SpoutSettings::volume_to_ml = 5.4` is a hidden coordinate-to-mL scale factor with no comment. The function `flow_rate_from_speed` converts `(π·r²·v·Cd) * volume_to_ml` to mL/s only because the world is assumed to be ~5.4 mL per simulation unit cubed.

**Fix:** Add a comment that ties this constant to the chosen world scale, and add a regression test that exercises a known head/area combination against an analytical mL/s value.

---

#### M5. `bed_delta` buffer oversized
**File:** `crates/sim-wasm/src/mpm_3d/state.rs:94-99`

`bed_delta` buffer is sized to `max_particles * 4 bytes` (~880 KB at 220k slots), but only the first `num_bed` entries (~12k) are touched by any pass.

**Fix:** Size the buffer to `num_bed`, or to a tighter bound after the bed is built.

---

#### M6. Stale state on `reset()`
**File:** `crates/sim-wasm/src/mpm_3d/mod.rs:319`

On `reset()`, `init_bed` only writes the new `num_bed` entries, leaving stale data in `bed_extract` and `affine` for indices `[new_num_bed, max_particles)`. Currently safe because shaders only read `< num_bed`, but if `num_bed` ever grows after a reset, stale state leaks in.

**Fix:** Zero-fill the affected ranges or write a single zero buffer over the whole bed_extract region in reset.

---

### LOW

#### L1. `build_cell_lookup` takes 8 arguments
**File:** `crates/sim-wasm/src/mpm_3d/bed.rs:149`

Same architectural smell as M3, same fix shape (group geometry into a struct).

---

#### L2. Frame-time accumulator drift
**File:** `crates/sim-wasm/src/mpm_3d/mod.rs:191`

`step_frame` clamps `dt` to `1/30` and divides by substeps. Under sustained low FPS the residual time is silently dropped, so simulation time runs slower than wall time.

**Fix:** Track an accumulator of dropped time, or expose a `simulated_seconds` debug counter to spot drift.

---

#### L3. Unguarded WebGPU init
**File:** `crates/sim-wasm/www-3d/main.js:46-47`

`await init()` and `WasmSim3D.create(canvas)` are called at module top level with no try/catch. On a browser without WebGPU the page silently freezes and the user sees no error.

**Fix:** Wrap in try/catch and render a "WebGPU not available" message into the canvas container.

---

#### L4. `FP_SCALE` worst-case bound not asserted
**File:** `crates/sim-wasm/src/mpm_3d/state.rs:8-11`

`FP_SCALE = 1048576` derivation comment claims max ~50 particles per cell. The actual worst case under heavy pours can exceed this; nothing in code asserts the bound.

**Fix:** Add a debug-only counter for `mass_contrib * fp_scale > i32::MAX/2` saturations and surface it in the UI.

---

## Test Coverage Gaps

Filled in this audit (9 new unit tests):
- `crates/sim-wasm/src/mpm_3d/bed.rs` — 5 tests covering empty-height collapse, particle count truncation, phase-marker correctness, `bed_extract` initialization, `cell_lookup` index validity
- `crates/sim-wasm/src/mpm_3d/mod.rs` — `dispatch_size_zero_count_returns_zero`, `dispatch_size_rounds_up`, `default_v60_shader_constants_in_sync`, `default_v60_grid_uses_uniform_dx`

Still missing:
- No test exercises the WGSL pressure projection math (only `shader_parses_with_naga`). A CPU-side reference implementation of the 6-point Laplacian RBGS would catch H1 and the mixed Dirichlet/Neumann bookkeeping.
- No test for the sparse-ballistic G2P blend constants (M1) — locking them to known outputs would catch accidental tuning regressions.
- No CPU-side mass conservation test for the inflow accumulator under fractional emission rates (the `accumulator -= count as f32` is exercised only via integration; a unit test would catch the integer-cast loss).
- No test for `BedConfig` extreme cases (zero particles, extreme aspect ratios) — `init_bed_particles` should not panic.

## Architectural Follow-Ups

1. **Push V60 obstacle parameters through uniforms** instead of hardcoding them in WGSL `resolve_scene_obstacles` (addresses H2).
2. **Add residual probe and configurable iteration count** to the pressure RBGS solver, with UI surfacing (addresses H3 + reuses M2's dead buffer).
3. **Derive fixed-point clamp bounds** from `MAX_VELOCITY`/`dx`/substep instead of magic ±2048; add overflow telemetry (addresses H1 + L4).
4. **Group function arguments into context structs** (`EmitContext`, `BedGeometry`) — addresses M3 and L1.
5. **Add CPU-side reference solver** for the 6-point Laplacian RBGS to lock pressure-projection math against regressions.

## Auto-Fixed in This Audit

- Deleted orphan `crates/sim-core/tests/test_integration.rs` referencing types removed in b38e2fa (was breaking `cargo test --workspace`)
- Removed dangling `<div class="control-panel">` in `crates/sim-wasm/www-3d/index.html:133`
- Removed unused `latestFrameSeconds` variable in `crates/sim-wasm/www-3d/main.js:42`
- Replaced manual `(count + threads - 1) / threads` with `.div_ceil()` in `crates/sim-wasm/src/mpm_3d/mod.rs:490`
- Added 9 unit tests covering bed initialization, dispatch sizing, and V60 constant synchronization

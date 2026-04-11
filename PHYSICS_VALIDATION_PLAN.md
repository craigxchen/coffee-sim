# Physics Validation Plan — Grid-of-Simulations View

## Goal

Turn the browser app into a validation tool. The user opens `www-3d/index.html` and sees a grid of small simulation canvases. Each cell runs one deterministic scenario that exercises a specific solver property. Each cell has a visible pass/fail border and a one-line metric readout. The grid gives an at-a-glance correctness read on the MPM solver.

Phase 1 is water-only, no bed, no extraction. Four scenes total.

---

## 1. Architecture — Multi-Canvas Mounting

### Current state

`WasmSim3D::create(canvas)` at `crates/sim-wasm/src/lib.rs:31` takes a single `HtmlCanvasElement`, constructs its own `Renderer` (which creates a `wgpu::Instance`, `Adapter`, `Device`, `Queue`), builds one `MpmSim3D`, and owns one `OrbitCamera`. `index.html` has a single `<canvas id="sim-canvas">` inside `<section class="viewer">`. `main.js` creates exactly one `WasmSim3D` instance and drives it from `requestAnimationFrame(animate)`.

### Phase 1 decision — one wgpu device per canvas

Create N canvases in `index.html` and call `WasmSim3D.create()` N times, once per canvas. Accept the overhead of one `wgpu::Instance` / `Adapter` / `Device` / `Queue` per scene for Phase 1. With N=4 scenes the cost is four small devices — acceptable on a dev box, not acceptable long-term.

Rationale: sharing a device across instances would require refactoring `WasmSim3D::create` at `lib.rs:31` to accept an externally-owned `wgpu::Device` + `wgpu::Queue`, plus teaching `Renderer::new` to borrow rather than own. That is the right long-term fix but it is out of scope for Phase 1. Flagged as a follow-up.

### HTML/JS changes

- Replace the single `<canvas id="sim-canvas">` inside `<section class="viewer">` with a grid container `<section class="validation-grid">` holding one `<div class="scene-tile">` per scene. Each tile wraps a `<canvas class="scene-canvas" data-scene="<id>">`, a `<div class="scene-title">`, and a `<div class="scene-status">`.
- `main.js`: replace the single-instance bootstrap (currently `lines 67-74`) with a loop that reads the scene registry, calls `WasmSim3D.create(canvas)` per tile, and pushes the instances into an array driven by a single `requestAnimationFrame` loop. Each tick advances every scene by the same fixed dt and renders each one.
- Keep `animate` as the single RAF driver. It walks the scene list, calls `stepFrame(fixedDt)`, `render()`, then after a configurable interval re-evaluates pass/fail and updates the status display.

Phase 1 target: N = 4 scenes in a 2x2 grid. The existing sidebar becomes a lean header with a "Run All", "Reset All", and step-count label. All the per-run tuning sliders (kettle angle, spout xyz, exit speed) are removed from the header because the scenes own those settings.

---

## 2. Scene Definition — Rust-Side

### Where it lives

New file `crates/sim-wasm/src/mpm_3d/validation.rs`. Exposed through `lib.rs` via a thin wasm-bindgen wrapper.

### Rust data model

```rust
pub(crate) struct ValidationScene {
    pub id: &'static str,
    pub name: &'static str,
    pub settings: MpmSettings,
    pub setup: fn(&mut MpmSim3D, &wgpu::Device, &wgpu::Queue),
    pub duration_seconds: f32,
    pub checkpoints: &'static [f32],
    pub check: fn(&SceneContext) -> PassFail,
    pub metric_label: &'static str,
}

pub(crate) struct SceneContext<'a> {
    pub sim: &'a MpmSim3D,
    pub elapsed: f32,
    pub baseline: Option<&'a BaselineSnapshot>,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
}

pub(crate) enum PassFail {
    Pending,
    Pass { metric: f32 },
    Fail { metric: f32, reason: &'static str },
}
```

`BaselineSnapshot` stores whatever the scene recorded at the first checkpoint (e.g. initial mass, initial particle count, initial centroid), so later checkpoints can compute drift.

### JS exposure

Add to `WasmSim3D` (in `lib.rs`):

- `validation_scene_id() -> String` — which scene this instance is running
- `validation_load(scene_id: &str)` — rebuild with that scene's settings + run setup
- `validation_elapsed() -> f32`
- `validation_status() -> JsValue` — serialized `{ state: "pending" | "pass" | "fail", metric: f32, label: String, reason: String }`
- `validation_checkpoint()` — snapshot baseline at `elapsed == 0`; subsequent calls re-evaluate

`setup` runs in Rust because several scenes need to plant particles directly into the GPU buffers with `queue.write_buffer`, and because inflow determinism requires the state machine (`InflowState`) to be reset in the same process that drives stepping. `check` runs in Rust because three of the four Phase 1 scenes need a CPU-side mass readback that is already Rust-only (`readback_mass_snapshot` at `mpm_3d/mod.rs:720`).

### Test gating

Drop the `#[cfg(target_arch = "wasm32")]` gates on `refresh_metrics` and on the validation module so the same scene harness runs under `cargo test` against a native GPU adapter. This mirrors the pattern already used by the ignored test at `mpm_3d/mod.rs:945` (`water_mass_stable_after_pour_off`).

---

## 3. Phase 1 Scenes

Four scenes, no more. Each is water-only, uses a bed-less `MpmSettings`, and runs at fixed dt = 1/60 s driven by the harness.

### Scene A — Hydrostatic Rest

- **Goal:** Closed container, no inflow, no initial motion. Water should not jitter and its volume should not drift.
- **Settings:** `MpmSettings::benchmark_free_stream()` as a base, then `settings.bed = None`, `settings.spout` unused (inflow disabled via `set_kettle_angle(0.0)`).
- **Setup:** Inject a packed slab of particles directly via `queue.write_buffer` into `buffers.particles`: a cuboid of roughly 6000 particles filling the lower half of the box below the V60, with `v = 0`, `C = 0`, `J = 1`, `mass = MASS_UNITS_PER_ML / PARTICLES_PER_ML`. Kettle angle = 0.
- **Duration:** 5 s.
- **Checkpoints:** at t = 0.5 s (baseline snapshot after transients), then t = 5 s.
- **Metric:** `(m(5 s) - m(0.5 s)) / m(0.5 s)` and peak per-particle speed in the last 60 frames.
- **Pass criterion:** mass drift < 0.1 % AND max speed < 0.05 sim units/s.
- **Failure looks like:** water volume visibly shrinking or expanding, particles jittering after settling.
- **Readback:** reuses `readback_mass_snapshot` at `mpm_3d/mod.rs:720` for mass. Needs a new small helper `readback_velocity_extrema` that walks the particle buffer the same way. No GPU metrics readback required.

### Scene B — Ballistic Arc

- **Goal:** Verify gravity and particle advection against a closed-form trajectory. No inflow, no obstacles in the active region, no pressure projection.
- **Settings:** Start from `benchmark_free_stream()`. Disable pressure projection via `set_pressure_projection_enabled(false)`. Disable sparse-ballistic heuristic via `set_temp_sparse_ballistic_enabled(false)`. Zero the bed (`settings.bed = None`) and keep the obstacles list empty by also clearing `settings.obstacles`.
- **Setup:** Write a thin cluster of ~500 particles at position `(-5, 5, 0)` with velocity `(v_x0, 0, 0)` where `v_x0 = 3.0`. All other state zeroed. Kettle angle = 0.
- **Duration:** 1.0 s (before the cluster leaves the domain).
- **Checkpoints:** t = 0.05 s (baseline centroid), t = 1.0 s (final centroid).
- **Metric:** distance between the final centroid and the analytic target `(x0 + v_x0 * 0.95, y0 - 0.5 * 10 * 0.95^2, 0)` where `gravity = settings.gravity = -10`. Report in units of `dx` (from `settings.bounds_size.x / settings.grid_dims[0]`).
- **Pass criterion:** centroid within 1 dx of analytic target.
- **Failure looks like:** particles curving sideways, decelerating midair, vanishing into walls.
- **Readback:** a new `readback_centroid_snapshot` helper, same pattern as `readback_mass_snapshot`. No GPU metrics readback required.

### Scene C — V60 Pool Rest

- **Goal:** Pour a fixed volume into the V60, stop the pour, verify the pool settles and does not leak.
- **Settings:** `MpmSettings::benchmark_center_pour()` unchanged (this is the scene that existing pour-off regression tests target).
- **Setup:** Default init. Kettle angle is driven by the harness.
- **Duration:** 5.5 s total — 2.5 s pour (angle = 36) + 0.5 s spin-down (angle = 0) + 2.5 s rest.
- **Checkpoints:** t = 3.0 s (baseline, right after pour cutoff + 0.5 s drain), t = 5.5 s.
- **Metric:** `(m(5.5) - m(3.0)) / m(3.0)`, reported in percent.
- **Pass criterion:** mass drift < 1 %. This is the threshold the existing `water_mass_stable_after_pour_off` test at `mpm_3d/mod.rs:945` uses (2 %, loosened because it runs against the free-stream geometry); pool rest in the V60 should be tighter at 1 %.
- **Failure looks like:** pool slowly draining through the cone, particles blinking at the cup corner.
- **Readback:** `readback_mass_snapshot`. No GPU metrics readback required.

### Scene D — Free Stream Continuity

- **Goal:** Continuous inflow, let the system reach a quasi-steady state, verify that emitted mass equals mass currently in the domain plus mass that has exited through the lower boundary. Phase 1 domain has no carafe outflow so "exited" should be zero — this catches the solver silently deleting particles under load.
- **Settings:** `MpmSettings::benchmark_free_stream()`. `settings.bed = None`.
- **Setup:** Default. Kettle angle 36 for the entire run.
- **Duration:** 3 s.
- **Checkpoints:** sample every 0.5 s, baseline at t = 1.0 s (after stream establishes), final at t = 3.0 s.
- **Metric:** `(total_emitted_mass - active_particle_mass - total_dropped_mass) / total_emitted_mass`. `total_emitted_mass` is available via `sim.total_emitted_mass()`. `total_dropped_mass` can be computed from `total_dropped_particles() * nominal_mass`. `active_particle_mass` comes from `readback_mass_snapshot`.
- **Pass criterion:** leakage ratio < 2 %.
- **Failure looks like:** stream visibly shrinking, or HUD `Total Emit` growing faster than the pool.
- **Readback:** `readback_mass_snapshot` + `total_emitted_mass()` + `total_dropped_particles()`. No GPU metrics readback required.

### Out of the Phase 1 set

Skip `Closed Box No-Growth`, `Wall Corner Stability`, and anything requiring bed coupling or extraction. Add them in Phase 2 only after the four above are green.

---

## 4. Pass/Fail Surfacing

### Visual grammar

- Each tile has a 4 px border: `#2a3e3f` for pending, `#5a9f3a` for pass, `#b5411f` for fail.
- Tile footer shows two rows:
  - row 1: scene name + elapsed/total time
  - row 2: metric label + numeric value + target range in muted text
- When a scene finishes it freezes on the last rendered frame and holds until the user clicks `Reset All` or `Run All`.

### CSS

Extend `crates/sim-wasm/www-3d/styles.css`:

- `.validation-grid` uses `display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;`
- `.scene-tile` with the pending border color, `.scene-tile.pass`, `.scene-tile.fail` modifier classes
- `.scene-canvas` sized to the tile with `width: 100%; aspect-ratio: 1 / 1;`
- `.scene-status` with a muted row 1 and colored row 2

Reuse `--panel`, `--muted`, and `--accent` from the existing root variables. Do not introduce a CSS framework.

### JS wiring

`main.js` drives status updates from the `animate` loop. After each `stepFrame` call on a scene instance, if `elapsed >= duration` it calls `validation_status()` once and then stops stepping that instance, but keeps rendering so the final frame stays visible. The status string is written into the tile's footer and the tile class is toggled to `pass` / `fail`.

---

## 5. Metrics Dependency Matrix

`refresh_metrics` at `crates/sim-wasm/src/mpm_3d/mod.rs:573` is currently a stub that returns `Ok(())` — every GPU metrics accessor on `WasmSim3D` (`maxAbsDivergence`, `fluidCellCount`, `divClampFires`, `pressureClampFires`, `massOverflowFires`) returns the default 0. Phase 1 scenes must work around this.

| Scene | Needs GPU metrics readback? | Needs CPU particle readback? | Dependency status |
|---|---|---|---|
| A. Hydrostatic Rest | No | Yes (mass + velocity extrema) | Ships on existing infra |
| B. Ballistic Arc | No | Yes (centroid) | Ships on existing infra |
| C. V60 Pool Rest | No | Yes (mass) | Ships on existing infra |
| D. Free Stream Continuity | No | Yes (mass) | Ships on existing infra |

All four Phase 1 scenes route through the CPU-side readback pattern already used by the ignored GPU test at `mpm_3d/mod.rs:908` (`manual_mass_readback_harness_runs`) and `mpm_3d/mod.rs:945` (`water_mass_stable_after_pour_off`). That code path has been proven to work under `cargo test`. The only new Rust work is:

1. A `readback_velocity_extrema` helper next to `readback_mass_snapshot` at `mpm_3d/mod.rs:720`.
2. A `readback_centroid_snapshot` helper, same pattern.
3. Dropping the `#[cfg(test)]` gate on these helpers so `validation.rs` can call them from non-test code, or moving the helpers into `mpm_3d/state.rs` as `pub(crate)` utilities.

Fixing `refresh_metrics` is a follow-up and is NOT a blocker for Phase 1. It becomes a dependency for Phase 2 scenes that want to assert on `max_abs_divergence` or overflow counters.

---

## 6. Determinism

Every Phase 1 scene must be reproducible frame-for-frame across runs on the same GPU adapter.

- Harness drives every instance with `stepFrame(1.0 / 60.0)`. The existing benchmark scenes already force `fixedStepSeconds = 1/60` in `main.js:107` — the validation grid does this universally.
- Pour angle changes are bound to checkpoint times, not wall clock. Scene C schedules `{ at: 0.0, angle: 36 }` and `{ at: 2.5, angle: 0 }`.
- `MpmSim3D::step_frame` already caps `dt` at `1/30` (`mpm_3d/mod.rs:258`); the harness dt of 1/60 is below that cap so no clamping happens.
- `InflowState` emission in `inflow.rs` uses a deterministic state machine (no JS `Math.random`). Verify there is no `rand::thread_rng` on the hot path before Phase 1a ships; if one exists, replace with a seeded `StdRng` stored on `InflowState`.
- No bed in Phase 1, so `bed::init_bed_particles` RNG does not apply. If a future scene uses a bed, it must take a seed through `MpmSettings`.

---

## 7. Phased Rollout

### Phase 1a — Plumbing

Landing gate: the browser shows a 2x2 grid of four empty canvases, each running the existing `default_v60` scene, stepping at fixed 1/60, sharing a single RAF loop. No pass/fail logic yet.

1. Create `crates/sim-wasm/src/mpm_3d/validation.rs` with the types from Section 2 and a skeleton scene list holding a single placeholder entry.
2. Add `pub mod validation;` to `mpm_3d/mod.rs`.
3. Add `validation_load`, `validation_elapsed`, `validation_status`, `validation_checkpoint`, `validation_scene_id` wasm-bindgen methods on `WasmSim3D` in `lib.rs`. They can stub pass/fail initially.
4. Lift `readback_mass_snapshot` out of the `#[cfg(test)]` module or create `pub(crate)` twins in `mpm_3d/state.rs`. Add `readback_velocity_extrema` and `readback_centroid_snapshot` as siblings.
5. Rewrite `www-3d/index.html` to host the 2x2 grid layout. Rewrite `www-3d/main.js` to create N instances in parallel and drive them from one RAF loop. Strip the per-run sliders.
6. Extend `www-3d/styles.css` with `.validation-grid`, `.scene-tile`, `.scene-tile.pass`, `.scene-tile.fail`, `.scene-canvas`, `.scene-status`.
7. Run `cargo test -p coffee-sim-wasm --lib` and `wasm-pack build --target web --release` to make sure the plumbing compiles on both targets.

### Phase 1b — Scenes

Implement the four scenes one at a time, in order A → B → C → D. For each:

1. Add the scene entry to the registry in `validation.rs`.
2. Implement `setup` and `check`.
3. Add a corresponding `#[ignore]`-gated Rust test that runs the scene end-to-end on a native GPU adapter, asserts pass, and serves as the regression test. Pattern: copy `water_mass_stable_after_pour_off` at `mpm_3d/mod.rs:945`.
4. Verify the scene in the browser (pass border appears within the expected runtime).

Do not start scene B until A is green. Do not start C until B is green.

### Phase 1c — Polish

1. Pass/fail rendering: border colors, footer metrics, held final frame.
2. `Reset All` button: rebuild every instance via `WasmSim3D::loadXxx` + `validation_load`.
3. `Run All` button: step through all scenes in parallel, which is what the RAF loop already does, but this button also resets timers so the user can re-run without a full page reload.
4. Headless mode: a `cargo test --lib -- --ignored --test-threads=1` target that runs all four scenes through their Rust tests and prints a summary. This becomes the CI hook once Phase 1 is green.

---

## 8. Out of Scope for Phase 1

- Bed particles, bed coupling, infiltration, drainage, bed compaction
- Extraction scalars, TDS, EY, pour uniformity, cup metrics
- Temperature fields, viscosity coupling, CO2
- Scenes requiring `refresh_metrics` GPU readback (divergence residual, clamp fire counters)
- Performance benchmarks, per-pass timings, frame budget regressions
- Cross-browser testing
- Device / adapter sharing between `WasmSim3D` instances
- HTML polish beyond the grid layout + borders + footer
- Convergence monitoring on RBGS iterations (ISSUES H3)
- Bed-coupling double-accounting fixes (ISSUES H4)
- Adaptive substepping
- Any scene requiring water to flow through the coffee bed

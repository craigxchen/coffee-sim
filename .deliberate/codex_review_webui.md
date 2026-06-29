REVISE

The direction is viable, but the plan is not implementation-ready. The largest issues are API ownership, web-safe metrics, and some overclaimed build/render work.

**Key Findings**

- `src/lib.rs` already exists and exports the crate API ([src/lib.rs](/Users/cxc/Github/coffee-sim/src/lib.rs:1)). The plan should say “extend it with cfg-gated wasm exports,” not “confirm/create placeholder” as if this were absent.

- `ui::Renderer` is genuinely surface-agnostic: it takes a `TextureView` and `ParticleBuffers` ([src/ui/render.rs](/Users/cxc/Github/coffee-sim/src/ui/render.rs:226)). But the plan misplaces `get_current_texture()` work in `src/ui/render.rs`; acquisition/configuration lives in the shell, currently [examples/water_app.rs](/Users/cxc/Github/coffee-sim/examples/water_app.rs:352). The existing native path already uses `wgpu::CurrentSurfaceTexture`, not the old `Result` form.

- The proposed `CoffeeSimApp { sim: Simulator, renderer, surface, config }` is not sufficient as written. `Simulator` privately owns `GpuContext` and only exposes `step()` and `switch_solver()` ([src/engine/simulator.rs](/Users/cxc/Github/coffee-sim/src/engine/simulator.rs:15)). It does not expose reset, scene replacement, device access for surface reconfigure, or solver diagnostics. Either the web app should mirror `water_app` with `GpuContext + XpbdSolver` directly, or `Simulator` needs explicit public APIs for reset, rebuild-on-new-scene, particles/state access, and GPU access.

- The scorecard claim is underspecified and currently wrong for web. `metrics()` only returns cached values ([src/solvers/xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2495)); yield/TDS/evenness/drawdown are refreshed by `sample_diagnostics()`, which performs blocking readbacks via `std::sync::mpsc`, `map_async`, `device.poll(Wait)`, and `recv()` ([src/solvers/xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:704)). Calling that on the browser main thread is not acceptable and may hang. The plan needs a web-safe async diagnostics path, or it must explicitly ship stale/limited metrics.

- The velocity-to-flow conversion is directionally sound but must be stated exactly. v1 used exit speed in sim units, then converted to capped mL/s: `flow_ml_s = area * discharge_coeff * exit_speed_sim * ML_PER_SIM_UNIT3` (v1 `inflow.rs`, lines 266-278). The rewrite expects `EmissionInput.flow_rate` in sim-volume/s and internally derives exit speed as `flow / A_eff` ([src/solvers/xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:957)). Therefore:
  `flow_rate_sim = PI * nozzle_radius^2 * discharge_coeff * (speed_m_s * SIM_UNITS_PER_METER)`.
  `ML_PER_SIM_UNIT3` is only needed for display or if intentionally applying a v1-style mL/s cap. Do not multiply/divide by it incorrectly.

- The build plan overstates current blockers. `cargo check --target wasm32-unknown-unknown` and `cargo check --examples --target wasm32-unknown-unknown` both pass now. `crate-type = ["cdylib", "rlib"]` and wasm-bindgen/web-sys deps are still needed for wasm-pack exports, but `getrandom` is not currently a proven blocker in this repo. If `winit`/`pollster` are moved to native-only deps, wasm example targets that import `winit` will need explicit gating or should be excluded from wasm checks.

- The wgpu 29 details are mostly right: `get_current_texture()` is `CurrentSurfaceTexture`; `InstanceDescriptor::new_without_display_handle()` with `Backends::BROWSER_WEBGPU` is valid; `Depth32Float` is baseline enough for this use. But the repo currently resolves `wgpu` to 29.0.3 under the `29.0.1` requirement, so avoid claiming exact 29.0.1 behavior unless pinned.

- Frontend porting “near-verbatim” is optimistic. v1 `main.js` is deeply coupled to exact metrics readback, debug panels, evaluation hooks, and the bad `REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 10` preflight (v1 `main.js`, lines 57-64 and 481-489). The plan says to strip these, which is right, but that means it is a structural rewrite of `main.js`, not near-verbatim.

- The CSS cross-section frame without a cross-section render pass is acceptable only as an explicit partial-parity caveat. It will not actually “look the same” in that region because the v1 overlay contained rendered cross-section content.

**REVISE Items**

- Add a concrete `CoffeeSimApp` ownership design: either use `GpuContext + XpbdSolver` directly like `water_app`, or extend `Simulator` with reset, load-scene/rebuild, GPU/device access, and current-state access.

- Add a web-safe async diagnostics/metrics unit, or reduce U5 scorecard scope to metrics that are already cheap and non-blocking.

- Correct the velocity conversion formula and tests: m/s → sim units/s via `SIM_UNITS_PER_METER`; then `A_eff * speed_sim` gives sim-units³/s. Use `ML_PER_SIM_UNIT3` only for display/capping.

- Move `get_current_texture()`/surface-status work out of `ui::Renderer` planning and into the wasm shell; note the native shell already uses the enum form.

- Decide how wasm example targets are handled after native-only `winit`/`pollster` dependency gating.

- Reword build risks: `getrandom wasm_js` may be a future guard, but it is not currently required by observed wasm compilation.

- Treat `main.js` as an adapted rewrite, not near-verbatim, and explicitly strip the 10-storage-buffer preflight plus all exact-metrics/debug/timeseries/evaluation couplings.
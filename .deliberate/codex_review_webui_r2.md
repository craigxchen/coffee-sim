**APPROVE**

The round-1 items are folded well enough to implement.

KTD-5 holds. `CoffeeSimApp` owning `GpuContext + XpbdSolver` matches the native shell pattern, and `XpbdSolver::build` clones `wgpu::Device`/`Queue` handles from `GpuContext`, so it does not borrow `GpuContext` by lifetime ([xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:1091)). The handle can call `build`, `reset`, `step`, `particles`, and `metrics` through the `Solver` trait ([base.rs](/Users/cxc/Github/coffee-sim/src/solvers/base.rs:36)); implementation just needs the trait in scope. Drop-and-rebuild is sound: GPU resources are RAII handles, no explicit teardown path is needed.

KTD-6/U4 is feasible, with one implementation watch: there is no single existing “metrics buffer” containing yield/TDS/evenness/drawdown. `sample_diagnostics()` blocks on status/timestamps, then computes brew metrics CPU-side from `chem`, `pos`, `vel`, and `phase` readbacks ([xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:704), [xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:777), [xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:823)). That is not a plan blocker because U4 already calls for a distinct async readback path and value-equivalence tests, but implement it as async snapshots of those buffers plus a factored CPU computation helper, not as mapping only `status_readback`. Use cloned `Device`/`Queue`/buffer handles and copied metadata so the returned Promise does not hold `&mut CoffeeSimApp` across await. Sampling every N frames is fine; solver correctness does not depend on metrics freshness.

KTD-7 is correct. `KEEP.md` defines `SIM_UNITS_PER_METER ~= 27.7` and `ML_PER_SIM_UNIT3 ~= 5.20` separately ([KEEP.md](/Users/cxc/Github/coffee-sim/KEEP.md:26)). The emitter uses exactly `A_eff = pi * r^2 * discharge_coeff` and derives `exit_speed = flow / A_eff` ([xpbd/mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:962)). So `flow_rate_sim = pi*r^2*C_d*(speed_m_s*SIM_UNITS_PER_METER)` round-trips to the requested exit speed. Do not involve `ML_PER_SIM_UNIT3` except display/capping.

KTD-3/U2 is now placed correctly. Native surface acquire is in `examples/water_app.rs`, already matching `CurrentSurfaceTexture` statuses ([water_app.rs](/Users/cxc/Github/coffee-sim/examples/water_app.rs:352)); `ui::Renderer` does not need reshaping. The async `GpuContext` split is the right seam: current `pollster::block_on` usage is concentrated in `src/utils/gpu.rs` ([gpu.rs](/Users/cxc/Github/coffee-sim/src/utils/gpu.rs:28), [gpu.rs](/Users/cxc/Github/coffee-sim/src/utils/gpu.rs:68)).

U1 is achievable as described. `wasm-pack build` targets the lib, not examples. If someone later insists on `cargo check --examples --target wasm32-unknown-unknown`, the examples need real `required-features`; otherwise documenting examples as native-only is enough for this plan. In non-example lib code, the native-only issue found by grep is `pollster`/`winit` in `gpu.rs`; `std::time::Instant` is only in examples.

No new sequencing contradiction found. The DAG `U1 -> U2 -> U3 -> U4 -> U5` holds, the dropped native-reshape unit is gone, and `main.js` is correctly described as an adapted rewrite.

Watch during implementation:
- Factor XPBD metric math so blocking and async paths share one CPU calculation.
- Avoid overlapping async readbacks on one reusable readback buffer; allocate/snapshot per request or gate in-flight sampling.
- Keep `sample_diagnostics()` out of wasm-facing paths.
- Keep the web device at `Limits::default()` and strip the v1 10-buffer preflight.
- Make scene rebuild also rebuild camera framing/control defaults where needed.
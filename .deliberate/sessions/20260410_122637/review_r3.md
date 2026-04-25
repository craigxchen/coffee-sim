REVISE

1. The `sdf_class` cache is not ready as specified. With `wgpu = 29.0.1` in [Cargo.toml](/Users/cxc/Github/coffee-sim/crates/sim-wasm/Cargo.toml#L12), treating `R8Uint` as a writeable 3D storage texture is not a safe assumption on the current target stack. If you want this optimization, the lower-risk architecture is to generate the mask on CPU alongside the existing SDF upload in [state.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/state.rs#L189) and bind it as a sampled integer texture in [pipelines.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/pipelines.rs#L27). If you keep GPU precompute, first switch to a proven storage-capable format and validate it on both native and wasm.

2. The plan misses the cheapest safe win: move `prepare_render` out of the substep loop. It runs every substep in [mod.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/mod.rs#L390), but [shader.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/shader.rs#L1328) only mirrors final particle state into `render_data`, and no later solver pass reads that buffer. That should be Phase 1 before adding new textures, pipelines, and toggles.

3. The plan also misses that `bed_lookup_clear` is wasted when there is no bed. In `benchmark_free_stream`, `num_bed == 0`, but [mod.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/mod.rs#L331) still clears the entire lookup grid every substep. Gate both `bed_lookup_clear` and `bed_lookup_scatter` on `self.num_bed > 0`; [shader.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/shader.rs#L739) only consumes `bed_lookup` when solid mass exists.

4. The profiling section is not implementable as written on the shipped wasm path. The browser renderer currently requests no optional GPU features in [renderer.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/renderer.rs#L255), and metrics readback is intentionally stubbed in [mod.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/mod.rs#L537). Split this into two tracks: browser telemetry with `performance.now()` over fixed workloads, and a native benchmark harness that uses timestamp queries when supported.

5. The SDF-cache test plan is too vague. Because [classify_cells](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/shader.rs#L800) changes pressure boundary conditions, “debug assertion for the first 10 substeps” is not enough. Add a deterministic mask-comparison test or readback harness at init/reset, then rerun the existing stability and mass tests in [physics_tests.rs](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/physics_tests.rs#L259) against both cache modes.

The rest is mostly solid. Replacing `clear_grid` with encoder-side clears is valid for [grid](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/state.rs#L104) and [grid_vel](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/state.rs#L111), and the early-outs in [grid_update](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/shader.rs#L672) and [boundary_project](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/shader.rs#L1097) are good Phase 1 work. Keeping RBGS tuning separate from semantics-preserving changes is also the right call.

I would revise the plan around this order:

1. Move `prepare_render` to once per frame.
2. Skip bed lookup passes when `num_bed == 0`.
3. Replace `clear_grid` with `encoder.clear_buffer()`.
4. Add the two empty-cell early-outs.
5. Profile again.
6. Only then decide whether an SDF cache is still worth the extra architecture cost.
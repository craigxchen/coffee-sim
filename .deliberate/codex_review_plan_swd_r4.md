**APPROVE**

All five Round-3 blockers are adequately addressed in the plan.

The plan now gives an implementation-ready `flip` ABI: `(c_surface, density_gate, div_scale, water_splash_cap)`, moves `approach_scale`, affine `k`, and density width `W` to shader constants, and correctly defines `div_scale <= 0` as disabling the whole merge discriminator. That maps cleanly onto the current 256-byte Rust/WGSL `Params` layout and 272-byte target in [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:315) and [common.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/common.wgsl:22).

The core curve is concrete enough to implement in [transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:153): `merge`, `dens_c`, final `c`, velocity mix, and affine damping are all specified. The density-only control is also now meaningful: with `merge = 0`, a sparse drip remains preserved and should reproduce the main-style stir failure.

The `grad_m` fix is addressed: the plan requires an explicit quadratic B-spline derivative, `1/h` scaled from `fx`, and explicitly rejects reusing APIC `B`’s `d` offset. It also uses `grad_m / max(length(grad_m), eps)`, avoiding the biased `normalize(grad_m + eps)` form.

Telemetry is correctly scoped as CPU-twin rederivation from readback state, with no diagnostic GPU buffer or ABI lane. That is compatible with the existing readback surfaces, especially `read_grid_velocities()` after a frame in [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:1204).

Holistically, the plan is implementable U1→U4 without inventing missing physics. It correctly leaves pressure/incompressibility out of scope, covers all three water clamp sites including `drag_fold` at [coupling.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/coupling.wgsl:292), and preserves the solid clamps in [plasticity.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/plasticity.wgsl:420). Minor implementation note: make the `water_splash_cap` off/default mapping explicit in code, either `flip.w = max_speed` or `flip.w <= 0 -> max_speed`, to preserve byte-identical default behavior. Not a plan blocker.

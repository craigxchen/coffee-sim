REVISE

Core transfer mechanism is sound, and the round-1 fixes are mostly addressed: affine damping is now correctly `C *= 1 - k*(1-c)`, divergence is `dinv*(b0.x+b1.y+b2.z)` with the post-`project` caveat, R5 now has telemetry plus a density-only negative control, splash claims are mostly scoped to transfer-side gain, and the default-off path explicitly removes the current `FLIP_PROTO` block in [transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:148).

The plan still has implementability holes.

The biggest issue is U3’s cap split. `drag_fold` clamps the water grid velocity with `params.max_speed` at [coupling.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/coupling.wgsl:292), and the pipeline runs `drag_fold -> ... -> project -> g2p_water` before G2P at [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:2519). If the new water cap is applied only at p2g/g2p clamp sites, as U3 says, the grid path is still clipped by the old web cap of 12 from [web.rs](/Users/cxc/Github/coffee-sim/src/web.rs:452). That defeats much of the crown/impact reason for the cap. The fix should be: add a water cap used by water `p2g_water`, water-grid `drag_fold`, and `g2p_water`; keep solid `solid_update`/`g2p_solid` on the existing global cap.

Related: the Params packing is not complete. The new `flip: [f32;4]` handles `(c_surface, density_gate, div_scale, affine_damp_k)` and fixes the round-1 divergence-scale lane issue, but U3 also needs a shader-visible water cap. There is no lane left and no updated size beyond 272. Add an explicit home for `water_splash_cap` and update the Rust/WGSL layout assertion accordingly.

The drip-stir guard is better conceptually, but “approach into dense neighborhood” is still underspecified. The plan says `v_own` points toward higher density, but `g2p_water` currently gathers only `v_grid`, `m_local`, and `B` from `grid_vel` [transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:182). A scalar `m_local` plus `|v_own-v_grid|` cannot distinguish a drip entering a pool from a crown particle leaving a dense surface. Specify a cheap directional signal, e.g. gather a mass first moment/gradient proxy from `grid_vel.w` and gate on `max(0, dot(v_own - v_grid, grad_m_hat))`, or bind/use `cell_meta` deliberately. If `near_air` comes from `cell_meta.w`, U2 must also update the `g2p_water` bind group; it currently binds only params/pos/vel/cmat/grid_vel/solids at [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:1907).

The R5 telemetry/control is stated, but not wired. The plan needs an explicit way to record/assert per-particle `c` without breaking byte-identical disabled behavior, and an explicit test hook to disable the merge discriminator while keeping the same tuned curve. If approach weights are compile-time constants, the “merge disabled” negative control is not actually available without a second shader variant.

Also tighten remaining over-claim wording: lines like “restores splash on free/impacting water” and U2 verification “shows a crown/scatter” should be phrased as transfer-side liveliness/surface velocity diversity improvement versus disabled pure-PIC baseline. Full pool-impact crown remains pressure-follow-up scope.

Specific REVISE items:

- Add `water_splash_cap` to Params explicitly; update Rust/WGSL layout size, docs, Config wiring, and tests.
- Use the water cap in `p2g_water`, `drag_fold`’s water-grid clamp, and `g2p_water`; leave solid clamps on global `max_speed`.
- Define `approach_into_density` formula concretely, including the density-gradient/source data and sign convention.
- If using `near_air`, specify the existing buffer/binding path, likely `cell_meta`, and update `g2p_water` bind group/storage-budget comments.
- Specify telemetry storage/readback for per-particle `c` that preserves disabled byte-identity.
- Specify the runtime/test hook for the density-only negative control.
- Downgrade the few remaining “crown/splash restored” statements to transfer-side improvement language.

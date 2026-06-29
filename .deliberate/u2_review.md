REVISE

**Findings**

- [src/solvers/xpbd/mod.rs:847](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:847): capacity clamp is not quite correct. If `active_count == capacity`, the loop decrements `axial`, sets `clamped`, then `emit_n == 0` returns before the warning at line 892. That silently consumes arclength for a layer that was not emitted and violates the “overflow warns+clamps” requirement. Check capacity before spending axial, or emit the clamp warning before the zero-emission return.

- [src/solvers/xpbd/mod.rs:856](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:856): partial layers are center-biased. `i` restarts at zero for every layer, while `cursor` only changes angle, so when `this_layer < n_layer` the outer annuli are repeatedly skipped. For non-integer `A_eff * spacing / V_w`, many layers are partial, so the stream is not actually sampled over `r_eff` at rest-density area. Use a deterministic per-layer slot order or carry a radial slot cursor so partial layers distribute over the full disc.

- [src/solvers/xpbd/mod.rs:805](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:805): `EmissionInput.event == Reset` is ignored. `seed()` resets inflow state, but the plan says the accumulator resets on Reset/seed. If Reset events are part of the emitter contract, this leaves stale accumulator/axial/cursor state.

The core volume algebra is otherwise mostly right: `A_eff`, `exit_speed`, `accumulator += flow / V_w * dt`, `N_layer = ceil(A_eff * spacing / V_w)`, and subtracting only `emit_n` prevent over-delivery and preserve dose under normal capacity. `depth = axial after decrement` is the right placement for emitted layers: within a frame layers are exactly `spacing` apart, and across frames the retained residual gives the next layer one spacing behind the previous layer’s advected position.

GPU writes and step ordering look correct: writes use `active_count * 16` for vec4 buffers and `active_count * 4` for phase, `active_count` grows after writes, params upload happens after emission, and `predict` initializes `pred` for newly active particles. With default `flow_rate = 0` and no backlog, `emit()` returns before mutating state, so non-pour scenes are not perturbed.

I ran the narrow rate test: `cargo test emission_rate_matches_flow_no_starvation --test xpbd_emission` passed.
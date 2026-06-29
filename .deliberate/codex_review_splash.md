REVISE

The document’s main diagnosis is mostly right, but it overstates separability and underspecifies the hard parts.

Code checks:
- `pic_blend` semantics are stated correctly. `PIC_BLEND_DEFAULT = 0.05`, and G2P scales APIC `C` by `apic_keep = 1 - params.pic_blend`; `pic_blend = 1` is pure PIC. See [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:151), [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:326), [transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:191).
- There is no FLIP `Δv` path in `g2p_water`; velocity is gathered from `grid_vel`, and APIC state is only `C`. The “high-frequency modes are filtered” diagnosis is physically plausible.
- The doc undercounts clamps. It correctly cites P2G affine clamp and G2P clamp in [transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:93), [transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:203), but `drag_fold` also caps grid velocity after gravity/drag/BC in [coupling.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/coupling.wgsl:292).
- `nozzle_radius` is a real lever, but the doc misframes it as cheap. The web setup deliberately widens the twofield jet to `0.55` and lowers `max_speed` to `12` because a sub-2-cell jet was known to whip/scatter/plunge and trigger air-pocket collapse. See [web.rs](/Users/cxc/Github/coffee-sim/src/web.rs:383) and [web.rs](/Users/cxc/Github/coffee-sim/src/web.rs:412). “Sharpen the jet” must preserve a grid-resolvable lower bound or pair with resolution changes.

The trilemma section needs revision. `tests/twofield_settled.rs` supports the two-source story: pure PIC kills the carrier; relief-off kills much of the source. See [twofield_settled.rs](/Users/cxc/Github/coffee-sim/tests/twofield_settled.rs:11). A state-aware transfer switch is plausible, because settled/deep/slow water can remain damped while fast/surface/impact water retains energy. But R8 is too strong: with production density relief on, settled liveliness is already partly sourced by relief through `cell_meta.x` pressure RHS in [pressure.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/pressure.wgsl:482) and under-density suction in [surface.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/surface.wgsl:175). The requirement can say “do not modify or depend on the parked `dbg.w` path”; it cannot honestly say “does not source liveliness energy from density relief.”

Constraint feasibility is plausible but not as written. A true FLIP/ASFLIP arm likely needs access to pre-update grid velocity. That might be possible without a new buffer by decoding old P2G velocity from `grid_fp`, but then `g2p_water` gains extra reads/bindings and cost. If it needs a separate `grid_vel_prev`, R7’s “no new storage buffer” blocks the core approach. Also, relaxing `max_speed` is fixed-point-sensitive: FP headroom is explicitly coupled to the cap in [common.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/common.wgsl:117), with a probe in [twofield_water.rs](/Users/cxc/Github/coffee-sim/tests/twofield_water.rs:335). R7 should require cap-hit and FP-saturation proxies, not just “keep overflow backstop.”

Requirement quality:
- R10 is directionally good but gameable. Velocity variance/upward velocity/spread can be satisfied by noisy non-water motion or cap artifacts. Require mass-weighted, localized post-impact upward/outward free-surface momentum or rebound height/spread, with low cap-hit rate and conservation/no-popcorn checks.
- R11 is a principled re-point only if “pathological agitation” is made quantitative: tail KE mean/peak, max speed, non-growth, and maybe circulation/angular-momentum bounds. Existing settled gates already provide a good shape in [twofield_settled.rs](/Users/cxc/Github/coffee-sim/tests/twofield_settled.rs:291).
- The grid-only scope is acceptable as a first bet, but the doc should stop claiming A+B+C are sufficient. The repo itself documents the grid-resolution floor around jet width. Add an explicit go/no-go: if transfer/clamp/nozzle changes fail without saturation or cap artifacts, escalate to resolution/hybrid rather than tuning indefinitely.

Specific REVISE items:
- R8: replace “does not source energy from density relief” with isolation requirements across relief/default/`dbg.w` modes.
- R7: distinguish hard WebGPU cap 16 from current internal 7-buffer budget; allow a budgeted FLIP history strategy or constrain A to pure APIC gating.
- Add `drag_fold` as a third `max_speed` clamp site.
- Add fixed-point/cap-hit/saturation gates for clamp relaxation.
- Reframe jet sharpening with a grid-resolvable minimum jet diameter and clarify fixed-flow vs fixed-exit-speed behavior.
- Tighten R10 and R11 into non-gameable quantitative gates.

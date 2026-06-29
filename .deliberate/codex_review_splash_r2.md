APPROVE

The round-1 blockers are adequately addressed. R8 is now honest: the doc explicitly names relief/suction as an existing settled-energy source and reframes the gate as non-dependence plus isolation across relief modes, matching [pressure.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/pressure.wgsl:482) and [surface.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/surface.wgsl:175). R7 now distinguishes the 16-buffer policy ceiling from the internal 7-buffer budget and gives valid FLIP-history options or a pure-APIC fallback. The third clamp in `drag_fold` is included at [coupling.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/coupling.wgsl:292), and R12 now correctly ties cap relaxation to cap-hit and fixed-point saturation probes from [common.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/common.wgsl:117) / [twofield_water.rs](/Users/cxc/Github/coffee-sim/tests/twofield_water.rs:335).

The jet framing is also fixed. The doc now treats `nozzle_radius = 0.55`, `max_speed = 12` as deliberate anti-artifact choices from [web.rs](/Users/cxc/Github/coffee-sim/src/web.rs:390) and [web.rs](/Users/cxc/Github/coffee-sim/src/web.rs:420), not cheap knobs. The `~2-cell diameter` floor is consistent with `h = 2 * particle_spacing` in [mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:304), and the fixed-flow vs fixed-exit-speed caveat is now visible.

The revised R10/R11/R12 gates are no longer trivially gameable. R10 requires localized, mass-weighted post-impact momentum/rebound/spread, low cap-hit rate, and conservation/no-popcorn. R11 adds tail mean/peak KE, max-speed, non-growth, and swirl/circulation bounds, which is the right shape for the existing settled-tail test family around [twofield_settled.rs](/Users/cxc/Github/coffee-sim/tests/twofield_settled.rs:291). Numeric thresholds are still deferred, but for a brainstorm requirements doc that is acceptable because the observable forms are now pinned.

On the four focus areas:

- The diagnosis is now balanced enough: transfer dissipation is a plausible primary mechanism, but the grid-resolution floor and nozzle/pocket artifacts are explicitly acknowledged. The go/no-go escalation prevents endless tuning if A+B+C fail cleanly.
- The two-source/trilemma risk is handled honestly. R8 remains useful because it tests code-path separability and result robustness instead of claiming relief contributes no energy.
- The state-aware bet is feasible but tight. A FLIP arm plus a free-surface state signal can plausibly fit the 7-buffer internal budget if it reuses `grid_fp` or adds one budgeted history buffer; if not, the pure-APIC gating fallback is a valid scoped core.
- The falsifiability is now acceptable. R10/R11/R12 still need calibration, but they are measurable requirements rather than vague visual aspirations.

Non-blocking issues to carry into planning:

- The requirements say “hard cap 16,” but this checkout still requests 9 storage buffers in [gpu.rs](/Users/cxc/Github/coffee-sim/src/utils/gpu.rs:108) and documents 9 in [common.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/common.wgsl:9). That is not a blocker because the proposed paths should fit under 7, but planning should not assume 16 is already requested.
- Soften the absolute wording “solvable inside the grid solver” in Key Decisions. The rest of the doc correctly says “first bet” with escalation, so that phrase is the only remaining over-claim.
- When pinning R10, normalize or scenario-lock the pour conditions so jet sharpening cannot pass by silently changing total flow instead of improving impact dynamics.

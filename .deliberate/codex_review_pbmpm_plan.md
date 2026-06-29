REVISE

**Findings**

1. **KD8 is not enforceable as written.** The requirements say the go/no-go bar is the *projected assembled* frame clearing `≤33 ms@200k`, not the single-phase number alone ([requirements](</Users/cxc/Github/coffee-sim/docs/brainstorms/2026-06-19-pbmpm-rearchitecture-requirements.md:36>)). But U6 leaves the projection method open until measurement and says the arithmetic is “written down, not asserted as a pass/fail” ([plan](</Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:197>), [plan](</Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:205>)). That can allow a false GO from a fast single-phase pass. The projection must be pre-registered, conservative, and decision-bearing.

2. **Candidate D is not quite the claimed candidate.** The plan calls D “compliant density projection” ([plan](</Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:39>)), but the repo hook is `dbg.w` uncapped/two-sided density target. The shader explicitly says the compliance regularization was “tried and dropped” ([pressure.wgsl](</Users/cxc/Github/coffee-sim/src/solvers/twofield/pressure.wgsl:497>)). Using it is a useful DensU proxy, but the plan must say that, pin `set_density_target_mode_for_test(true)` plus `set_density_rate_k_for_test(k)`, and avoid claiming it measures the full compliant Candidate D unless that variant is actually built.

3. **The “same scene” comparison is under-specified.** `src/web.rs::setup_for` applies solver-specific water-only settings: twofield gets a wider nozzle and lower speed cap ([web.rs](</Users/cxc/Github/coffee-sim/src/web.rs:465>), [web.rs](</Users/cxc/Github/coffee-sim/src/web.rs:482>)). A bounce metric comparing PB-MPM, XPBD, twofield, and D must pin the physical emitter, particle spacing, cap, geometry, and impact window outside per-solver web defaults, or the D/PBMPM delta can be apples-to-oranges.

4. **The PB-MPM state/buffer model is internally inconsistent.** The design says the grid carries only `[mass, mom.xyz]` atomics ([plan](</Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:75>)), but U5 gates fixed-point overflow for “deformation_displacement/position-correction lanes” ([plan](</Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:176>)). It also never defines the storage layout, lifetime, units, initialization, or update path for `D`. KD10 specifically requires this headroom surface to be decided up front ([requirements](</Users/cxc/Github/coffee-sim/docs/brainstorms/2026-06-19-pbmpm-rearchitecture-requirements.md:37>)).

5. **The transfer/perf assumptions need correction.** The plan says “quadratic-B-spline 4×4×4 scatter” ([plan](</Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:134>)), but the twofield quadratic template is 3×3×3 ([transfers.wgsl](</Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:118>)). If PB-MPM intentionally uses 4 taps, that is a 64-node scatter/gather budget, not the twofield 27-node pattern. Also, the claimed reusable cell-order reorder is XPBD neighbor-gather machinery, while P2G atomic contention is only an open issue for grid scatter ([hash.rs](</Users/cxc/Github/coffee-sim/src/utils/hash.rs:3>), [issue 004](</Users/cxc/Github/coffee-sim/docs/issues/004-reduce-p2g-atomic-contention.md:15>)).

6. **A live high-velocity jet needs emission, but no unit owns it.** The pipeline diagram includes `emit pour particles` ([plan](</Users/cxc/Github/coffee-sim/docs/plans/2026-06-20-001-feat-pbmpm-prototype-plan.md:62>)), and R1 depends on `high-velocity-jet-impact`, which declares a pour ([scene.rs](</Users/cxc/Github/coffee-sim/src/engine/scene.rs:409>)). U1-U3 only describe scaffold, transfers, and constraint; no unit ports the water-pool/dormant-slot/emission activation path before the visual oracle.

7. **Visual-first is acceptable, but U6 must be hardened.** Deferring heavy Phase A tests is sound for this exploratory bet. The backstop is only trustworthy if U6 pre-registers the rebound/spread threshold, D delta threshold, cap-hit/no-popcorn/conservation bounds, exact scenes/configs, and makes both bounce-vs-D and projected assembled cost actual pass/fail decision inputs.

**Specific REVISE Items**

- Make U6’s assembled-cost projection a hard gate: fixed method before measurement, includes second field, drag, lambda ping-pong, bed/plasticity/wetting costs or conservative upper bounds, substep multiplier, U7 if triggered, and buffer ledger against 9 and 16.
- Reword Candidate D as the existing DensU proxy, or implement the true compliant Candidate D. Pin the D mode/rate and identical physical test setup.
- Define PB-MPM `D`/correction storage precisely and reconcile it with R7 fixed-point overflow math.
- Choose 3×3×3 vs 4×4×4 transfers explicitly and update perf/atomic-contention assumptions.
- Add an emission/water-pool unit before U3 visual validation.
- Treat XPBD reorder reuse as a ported subsystem with measured cost, not a free existing P2G fix.
- Pin the bounce metric thresholds and comparison configs before reading PB-MPM/D results.

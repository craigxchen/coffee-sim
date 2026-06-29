**REVISE**

The revision fixes the original overwrite issue in structure: a separate `freeze -> impact_grain -> impact_water -> apply_drag_pred` block after buoyancy will compose with prior drag/buoyancy velocity deltas, because `vel_frozen` is re-snapshotted after those passes. In the current code, that would slot after [mod.rs:2363](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2363)-[2403](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2403). It should not clobber drag/buoyancy.

But I still would not land this plan as written.

1. The proposed `Δv` cap can break momentum conservation.

The per-pair formula is conserving only before the cap:

```text
Δv_g = +s * m_w/M * n
Δv_w = -s * m_g/M * n
```

However U2 says to cap each particle’s accumulated `|Δv| <= k * coupling_h / dt`. If that cap is applied independently in `impact_grain` and `impact_water`, one side of a pair or one species’ accumulated gather can clamp while the other does not. That reintroduces net momentum.

Fix: cap the shared scalar `s_pair` inside the canonical helper before the mass split. For normal reversal, use the fact that the split changes relative normal approach by exactly `s_pair`, so:

```text
s_pair = min(s_pair_raw, approach)
```

For CFL, cap the same shared scalar, e.g.:

```text
dv_cap = k * coupling_h / dt
s_pair <= dv_cap / max(m_w/M, m_g/M)
```

That bounds either side for that pair while preserving equal-and-opposite impulse. If the plan truly needs an accumulated per-particle cap across all neighbors, that cannot be made exactly conserving with two independent gather passes unless you add another symmetric limiter/reduction design. Do not use a post-accumulation per-particle clamp and still claim exact conservation.

2. The impact velocity sampling is probably misordered relative to the stated mechanism.

The plan says this recovers water momentum “before it’s dissipated,” but the proposed block runs after the water PBF loop, after drag subiters, and after buoyancy. The current solver does water PBF at [mod.rs:2202](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2202)-[2321](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2321), drag at [2324](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2324)-[2361](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2361), then buoyancy at [2363](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2363)-[2403](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:2403). Re-snapshotting there measures residual post-drag/post-buoyancy velocity, not the incoming jet velocity.

That does not double-count in the overwrite sense, but it makes the impact term order-dependent and potentially too late to solve the diagnosed “PBF cushion already ate the jet” problem. The plan needs to choose explicitly:

- If impact is meant to use incoming jet velocity, move it earlier or snapshot the relevant velocity before drag/buoyancy/PBF dissipation.
- If it intentionally uses residual post-coupling approach, update the rationale and calibration expectations; the threshold window may be much tighter.

3. U4/U5 sequencing is circular.

U4 says web calibration depends on the U5 percolation/ponding A/B gate passing first. U5 says it depends on U4’s calibrated value. Split this into either:

- U5a invariant/pass-level tests, U4 calibration, U5b calibrated scenario gates; or
- U4 exploratory calibration with temporary harnesses, then U5 codifies the chosen values.

As written, it is not a clean incremental sequence.

4. Params layout needs a concrete decision.

Current layout has `_pad_coupling0` and `_pad_chem1` visible in both WGSL and Rust ([common.wgsl:57](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/common.wgsl:57)-[84](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/common.wgsl:84), [mod.rs:79](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:79)-[106](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:106)). Three new floats require either using `fines.w`, growing `Params` and updating the `336` assert at [mod.rs:112](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:112), or packing/configuring differently. The plan allows extension, but KTD-6 still leans on “size stays 336.” Make the chosen route explicit.

5. Add two small implementation guards.

The canonical helper should specify an `r` guard before `normalize(x_grain - x_water)`; coincident or near-coincident water/grain particles can otherwise produce NaNs. Also specify the same water effective-mass floor on both sides of the helper, or an absorbed/near-empty water particle can create divergent behavior.

Verdict: the plan is close, but the cap issue is a real conservation bug, and the velocity-sampling/order claim still does not match the stated physics. Fix those before approval.
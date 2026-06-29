ROUND 6 (final confirmation) of an SDF-boundary plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator (V60 cone + grains-only filter + cup). Rounds 1–5 folded in.

Round 5 left ONE paper-correctness issue (and stated everything else resolved + remaining items empirical/test-settled). It is now fixed: the cone gradient no longer returns zero on the smooth side wall. The pseudo-code (KTD-2) now branches:
- off-surface (`|dir| > EPS`): normal = safe_normalize2 of the p→closest direction;
- exact smooth side-wall (`cp` on the segment INTERIOR): normal = `inward_normal(seg)` from the segment tangent, rotated toward the smaller-r cavity side (finite, unit, nonzero);
- exact tip/axis corner (non-differentiable): zero is acceptable ONLY here.
U1 now tests that the exact on-smooth-wall gradient is finite/unit/nonzero with `sample(p+ε·grad) > sample(p)`, and the tip/axis is finite (no NaN). U5 now tests that a particle placed exactly on the smooth wall with `contact_offset > 0` projects to `sample ≥ contact_offset − tol`.

Confirm this resolves the round-5 issue and that no paper-correctness issue remains. First line: **APPROVE**, **REVISE**, or **REJECT**. If the only remaining items are empirical/numerical points settled by the plan's executable test assertions (not resolvable on paper), treat that as APPROVE and say so explicitly. Cite section/unit IDs. Do not re-open already-resolved items.

Hard constraints (verified): ≤8 storage buffers/stage; no float atomics; Params exactly 256 B byte-matched w/ assert; boundary folds into apply_dp (friction + post-friction recheck, →6/8) + apply_drag_pred (push-out+gate only, →5/8); preserve `.w`; per-pipeline auto-layout bind groups; existing suites/scenes green; perf deferred. User is a physics reviewer who rejects hand-waving.

## Revised Plan to Review


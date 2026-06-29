You are reviewing ROUND 3 of an SDF-boundary implementation plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator (V60 cone + grains-only filter + cup). Rounds 1–2 are folded in.

Round 2 returned REVISE with 2 remaining blockers + 1 cleanup; all are addressed here:
1. **Closed-tip on-axis leak** — the cone pseudo-code (KTD-2) now has an explicit closed-tip branch: for `apex_open=false`, any `y < apex_y` returns `(-distance(p, tip_point), grad toward tip)` (negative/forbidden, gradient up), correctly handling the exact `(0, apex_y−ε, 0)` case where `r=inner=0` previously misclassified as inside. KTD-5 + U1 test (the exact on-axis point) updated.
2. **Rejection seeding** — now mandatory everywhere: R-3 states it with no AABB fallback, and the Open Question that said "AABB first, escalate" is reframed (rejection is settled, not optional). U3 already required it.
3. **Cleanup** — "disjoint" replaced with "nested, non-crossing cavity constraints" (the filter sits inside the support cone; both bind grains via the min-signed/intersection union). HTD union line, KTD-3, and U1 tests updated; a nested-pair test added.

Verify these are genuinely resolved and surface any remaining correctness issue. First line: **APPROVE**, **REVISE**, or **REJECT**, then explain citing section/unit IDs. Prior fully-resolved blockers (round-1 #1 gradient axis/sign, #4 per-solid friction, #5 ordering/recheck, #6 momentum framing, #7 volume tests) do not need re-litigating unless this revision broke them.

Hard constraints (verified): ≤8 storage buffers/stage; no portable WGSL float atomics; Params exactly 256 B byte-matched with a compile assert; boundary handling folds into apply_dp/apply_drag_pred reusing the box-clamp normal-from-delta + friction; preserve the `.w` moisture lane; per-pipeline auto-layout bind groups; existing suites/scenes stay green; perf deferred. The user is a physics reviewer who rejects hand-waving.

## Revised Plan to Review


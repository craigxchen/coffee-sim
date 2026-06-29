ROUND 5 review of an SDF-boundary plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator (V60 cone + grains-only filter + cup). Rounds 1–4 folded in.

Round 4 left 3 issues + 1 cleanup; all addressed:
1. **2D normalize NaN** — the cone gradient now `safe_normalize2(dir)` on the 2D surface direction (zero on the wall/tip) *before* the 3D safe-normalize; KTD-2 now lists three mandatory guards (axis r̂, 2D dir, final 3D); U1 tests the exact on-wall point for NaN.
2. **`apply_drag_pred` bindings** — corrected: `apply_dp` gains only `solids` → 6/8 (already has pos+phase, does friction + post-friction recheck); `apply_drag_pred` gains `solids`+`phase` → 5/8 and does push-out + species gate ONLY (no friction, so no `pos`). Storage table + KTD-3 + U5 + Files list updated.
3. **Radius contract** — pinned to ONE contract: `apex_r`/`top_r` are OUTER wall radii; the cavity surface is `inner_r(y)=max(lerp(apex_r,top_r,t)−thickness, hole_radius)`, used identically by the distance segment AND the inside test. KTD-4 Primitive record, KTD-2 pseudo-code, U1, U2 all state outer-radius + this surface.
4. **R-2 wording** — "non-intersecting" → "nested / non-crossing cavity solids fully inside the domain box".

Verify resolution and surface any remaining correctness issue. First line: **APPROVE**, **REVISE**, or **REJECT**, then explain citing section/unit IDs. Do not re-litigate already-resolved items unless this revision broke one. If the only remaining items are fine numerical/empirical points that can only be settled by running code (not on paper), say so explicitly and APPROVE — those are encoded as the plan's executable test assertions.

Hard constraints (verified): ≤8 storage buffers/stage; no float atomics; Params exactly 256 B byte-matched w/ compile assert; boundary folds into apply_dp (friction) + apply_drag_pred (push-out+gate); preserve `.w` lane; per-pipeline auto-layout bind groups; existing suites/scenes green; perf deferred. The user is a physics reviewer who rejects hand-waving.

## Revised Plan to Review


ROUND 4 review of an SDF-boundary plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator (V60 cone + grains-only filter + cup). Rounds 1–3 folded in.

Round 3 left 1 blocker + 2 cleanups; all addressed here:
1. **Closed-tip Euclidean distance (blocker)** — the KTD-2 pseudo-code no longer early-returns the tip for `y < apex_y`. It now uses an **endpoint-clamped point-to-segment** distance for all `y ≤ top_y` (clamps to the tip endpoint on-axis, to the slanted surface off-axis → true Euclidean everywhere), with the sign forced negative below the apex via an explicit `inside = (y ≥ apex_y) && (r ≤ inner)` test, and the gradient flipped to always point into the cavity. KTD-5 + U1 updated; U1 now tests both the exact on-axis tip case AND an off-axis below-apex point whose closest point is on the slant.
2. **U5 scope wording** — changed from "non-intersecting solids" to "nested / non-crossing cavity solids fully inside the domain box" (aligns with KTD-3; the V60's filter-inside-support pair is allowed).
3. **U2 wording** — "a grain just inside the filter surface is forbidden" → "just outside / through the filter wall (outside the filter cavity but inside the support cone)", consistent with the positive-inside convention.

Verify resolution and surface any remaining correctness issue. First line: **APPROVE**, **REVISE**, or **REJECT**, then explain citing section/unit IDs. Do not re-litigate the already-resolved round-1/2 blockers unless this revision broke one.

Hard constraints (verified): ≤8 storage buffers/stage; no float atomics; Params exactly 256 B byte-matched w/ compile assert; boundary folds into apply_dp/apply_drag_pred reusing box-clamp normal+friction; preserve `.w` lane; per-pipeline auto-layout bind groups; existing suites/scenes green; perf deferred. The user is a physics reviewer who rejects hand-waving.

## Revised Plan to Review


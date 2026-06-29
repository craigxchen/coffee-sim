ROUND 5 (final confirmation) of the Phase 1.5 (extraction + thermal) plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator. Rounds 1–4 folded in.

Round 4 left ONE edit: the HTD "Two-pool dissolution" directional pseudo-code (and the KTD-2 intro headroom line) still used `V_w` for water volume. Now fixed — `water_headroom = max(c_sat−c,0)·(f_w·V_w)`, `c_w += Σtake/(f_w·V_w)`, with the `f_w==0` guard, consistent with KTD-2 precondition (d), R8/KTD-7/U5/U7 invariants, and the yield/TDS denominators. Verified no bare `c·V_w` or `·V_w` water-volume term remains.

Round 4 said: "After that edit, the only remaining open items are empirical/calibration points covered by U7's executable gates, and I would approve."

Confirm consistency and surface any remaining paper-correctness issue. First line: **APPROVE**, **REVISE**, or **REJECT**. If the only remaining items are empirical/calibration points settled by the plan's executable tests (U7), treat that as APPROVE and say so explicitly. Cite section/unit IDs; do not re-open resolved items.

## Revised Plan to Review


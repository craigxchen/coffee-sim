ROUND 4 (final confirmation) of the Phase 1.5 (extraction + thermal) plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator. Rounds 1–3 folded in.

Round 3 confirmed the chem_frozen snapshot, per-grain flux, and symmetric thermal clamp are resolved, and left ONE consistency fix: the conservation invariant statements still said `Σ(active water c·V_w)` but the transfer math uses `c·f_w·V_w`. That is now fixed everywhere — R8, KTD-7, the U5 per-step conservation gate, the U7 full-brew conservation gate, and the yield Open Question all now read `Σ(grain s_f+s_s) + Σ(active water c·f_w·V_w)` (consistent with the water update `c += take/(f_w·V_w)` and the `f_w·V_w` headroom). Verified no bare `c·V_w` remains.

Round 3 said: "After that, the remaining open points are empirical calibration items covered by U7's executable gates, and I would approve."

Confirm the invariant is now consistent and surface any remaining paper-correctness issue. First line: **APPROVE**, **REVISE**, or **REJECT**. If the only remaining items are empirical/calibration points settled by the plan's executable tests, treat that as APPROVE and say so. Cite section/unit IDs. Do not re-open already-resolved items.

Hard constraints (verified): ≤8 storage buffers/stage; no float atomics; Params byte-matched (~320B); counting-sort grid doesn't reorder particle arrays; existing suites stay green (opt-in); user rejects silent conservation leaks.

## Revised Plan to Review


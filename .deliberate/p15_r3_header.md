ROUND 3 review of the Phase 1.5 (extraction + thermal) plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator. Rounds 1–2 folded in. Round 2 had 1 blocker + 3 second-order issues; all addressed:

1. **Frozen-snapshot blocker fixed:** added `chem_frozen` (binding 22). The models stage copies `chem → chem_frozen` before dissolution and again before thermal; all transfer/thermal passes **read `chem_frozen` and write disjoint slots of the live `chem`** (grain pass → grain slots, water pass → water slots) — so no intra-dispatch read/write race, conservation + determinism hold. KTD-2, HTD budget, U4, U5, U6 updated.
2. **`f_w` coupling fixed:** water volume is `f_w·V_w` everywhere — headroom `max(c_sat−c,0)·(f_w·V_w)`, water update `c += Σtake/(f_w·V_w)` with an `f_w=0` guard, TDS denominator `Σ(cup water f_w·V_w·ρ)`, thermal `C_water` scales with `f_w·mass` (wet grain mass includes `ρ·V_abs`). KTD-2, R4, U5, U6, U7.
3. **Symmetric pair-`q` clamp:** stability clamps the pair heat `|q_ij| ≤ ε·min(C_i,C_j)·|ΔT|` (antisymmetric), NOT a per-particle ΔT clamp. R2, U2, U6.
4. **Per-grain aggregate flux:** `diss_count` computes `(N_w, flux_g)` per grain (aggregate `|u_rel|` from finalized `vel`) into the `vec2` `diss_neighbors` (b21); both transfer passes read the stored `flux_g` → recompute the identical per-grain `release`/`take`. Budget: diss_count 6/8, dissolve_grain/water 7/8, thermal 6/8 (the six 8/8 kernels untouched). KTD-3, HTD budget, U5.

Round-1 resolved (don't re-litigate): KTD-1 passive-scalar framing, KTD-4 free Lagrangian advection, the N=0/release_total=0/proportional-depletion/saturated-water/CPU-GPU-parity gates, no-drained accounting.

Verify each round-2 item is genuinely resolved and surface any remaining correctness issue. First line: **APPROVE**, **REVISE**, or **REJECT**, then explain citing section/unit IDs. If the only remaining items are empirical/calibration points settled by the plan's executable tests (not on paper), say so and APPROVE.

Hard constraints (verified): ≤8 storage buffers/stage; no float atomics; Params byte-matched (grows ~320B); counting-sort grid does NOT reorder particle arrays; Phase-1.4 wetting is the proven atomic-free two-sided transfer template (reads pred.w, writes pos.w); existing suites stay green (opt-in, extract_rate=0); user is a physics reviewer who rejects silent conservation leaks.

## Revised Plan to Review


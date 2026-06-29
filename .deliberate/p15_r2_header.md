ROUND 2 review of the Phase 1.5 (extraction + thermal) plan for a Rust + wgpu + WGSL two-species XPBD coffee simulator. Round 1 returned REVISE with 8 items; all are folded in:

1. **Dissolution cap tightened** (KTD-2/U5): identical eligibility predicate on all three passes (diss_count + both transfer passes), explicit `N=0` guard, and **proportional pool depletion** `s_p −= Σtake·release_p/release_total` with a `release_total=0` guard.
2. **KTD-1 reframed**: conserves an *extractable-solute scalar inventory* (and thermal-energy inventory), NOT mechanical mass — the dilute-solution density/inertia neglect is stated as a deliberate, bounded (~1.3% TDS, <1.5% density) simplification.
3. **KTD-4** (free Lagrangian advection) — confirmed correct in round 1; unchanged.
4. **Thermal energy conservation fixed** (R2/U2/U6/HTD): now **capacity-weighted** — `ΔT_i = (κ·dt/C_i)·Σ(T_j−T_i)` with per-species `C=mass·specific_heat`, which conserves `Σ C_i·T_i` for UNEQUAL capacities (each pair contributes ±q antisymmetrically). Added per-species specific-heat constants (U3). Symmetric ΔT explicitly called out as wrong unless C_i=C_j. U2/U6 gates now use an unequal-capacity pair.
5. **Flux source pinned** (KTD-3/U5): the dissolution pass recomputes |u_rel| from the **finalized `vel` (b3)** with the same range h, used identically by all three passes (no drift); a real velocity, not the solve residual.
6. **Yield/TDS accounting fixed** (R6/R8/U7/KTD-7): water is never removed, so the conserved quantity is `Σ(grain s_f+s_s) + Σ(active water c·V_w)` — no separate "drained" term (that double-counted). The drained Open Question is reduced to a reporting nicety.
7. **Exact bind layouts** (HTD budget table): chem=b20, diss_neighbors=b21 (dedicated); diss_count 6/8, dissolve_grain/water 7/8 (vel b3 included for the flux), thermal_exchange 5/8; the six 8/8 kernels untouched.
8. **Tests added** (U1/U5/U6): Arrhenius-clamp + u_half=0 guards; saturated-adjacent-to-unsaturated water; release_total=0 guard; CPU/GPU parity on a tiny hand-computable pair graph; unequal-capacity thermal energy gate.

Verify each is genuinely resolved (not hand-waved) and surface any NEW or second-order correctness issue. First line: **APPROVE**, **REVISE**, or **REJECT**, then explain citing section/unit IDs.

Project hard constraints (verified): ≤8 storage buffers/stage; no float atomics; Params byte-matched Rust↔WGSL (grows to ~320B for the new constants); counting-sort grid does NOT reorder particle arrays (so chem[i] rides particle i); Phase-1.4 wetting is the proven atomic-free two-sided transfer template; existing suites must stay green (opt-in, extract_rate=0 default); the user is a physics reviewer who rejects silent conservation leaks.

## Revised Plan to Review


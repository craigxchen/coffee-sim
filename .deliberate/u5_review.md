APPROVE

No blocking correctness findings for Unit 5.

The per-pair `take` is structurally identical in both transfer passes: same eligibility, same frozen `chem_frozen`, same frozen `diss_neighbors`, same `release/N_w` and `headroom/N_g` formula in [extraction.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/extraction.wgsl:115) and [extraction.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/extraction.wgsl:163). Conservation is not bit-exact because f32 pool splitting and concentration reconstruction round separately, but I do not see a structural divergence path.

Race-freedom looks correct: `chem` is copied to `chem_frozen` before counting/transfer, then grain and water passes read frozen chem and write disjoint live `chem` slots in command order. The saturated-water cap is applied on both sides via the same headroom term, so if input water starts `<= c_sat`, the pass cannot drive it past `c_sat` except normal f32 tolerance. A pre-existing oversaturated water stays oversaturated but accepts no more solute.

The requested guards are present: `N_w`, `N_g`, `f_w`, and `rel.z` are checked before division. `pos.xyz` against a `pred`-built grid is correct here because extraction runs after `finalize`, which assigns `pos.xyz = pred.xyz`; wetting only changes `pos.w`. Transfer bind groups use uniform params plus 7 storage buffers, so they stay within the WebGPU 8-storage-buffer stage limit.
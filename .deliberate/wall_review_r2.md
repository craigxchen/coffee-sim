**REVISE**

**Findings**

1. **U2 no-op binding step is likely not valid as sequenced.**  
   The plan says to add `boundary_grad` to the `compute_boundary`, `compute_lambda`, and `compute_dp` bind groups before the shaders read/write it. With wgpu’s auto-derived layouts, unused shader bindings generally are not present in the entry-point layout, so adding an extra bind-group entry can fail validation. Revise U2 to either only allocate/request the limit there, then bind when U3/U4 statically use the buffer, or make U2 include harmless static shader uses for each entry point.

2. **Corner test is necessary but still too weak as written.**  
   “Correction magnitude near the seam stays bounded” will catch explosions, but not systematic under-compensation from the one-sided nearest-surface normal. At a floor-wall seam, the true occluded kernel is two-plane-ish, while the SDF union chooses one face. Add assertions that seam particles remain density/incompressibility-in-band and do not form a persistent seam sheet or leak/climb along the unselected face. Ideally sample cases where `d_side ≈ d_floor` and require both vertical and radial behavior to stay plausible over time.

3. **The AABB byte-unchanged claim is logically plausible but the verification wording is not airtight.**  
   If `compute_lambda`/`compute_dp` strictly guard all `boundary_grad` reads behind `params.num_solids > 0u`, then `num_solids == 0` scenes should be numerically unchanged. But “water-core suite stays green” is not a byte-equality check. If byte-unchanged is a requirement, add an explicit deterministic before/after state-byte comparison for a solid-free scene. Also phrase the guarantee as “no read and no arithmetic when `num_solids == 0`”; the buffer does not need to be zeroed for AABB scenes.

**Sanity Checks**

The five Round-1 revisions are now substantively correct: the `g_b` sign wording is fixed, the 16-buffer grant is a hard request-device failure, the sign diagnostic is the right kind of deterministic guard, the no-climb metric is splash-robust, and the floor-wall corner test is the right class of regression test.

Adding `g_b` to `sum_g` is physically correct and not inherently destabilizing. It may change λ near walls, including reducing the self-gradient term when the analytic wall gradient cancels the missing-neighbor fluid gradient, but that is the expected Jacobian for a fixed boundary. The mitigation is the planned near-wall λ/eruption/no-climb regression sweep, not removing the denominator term.

Applying the boundary correction every solver iteration is correct for PBF. It is not a separate force being accumulated once per frame; it is part of each iterative constraint projection, recomputed from the current predicted positions.

So the core physics is acceptable, but the U2 sequencing and corner/AABB verification details need tightening before approval.
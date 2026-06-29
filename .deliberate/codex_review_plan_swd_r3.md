REVISE

The plan is much closer, and the previous cap/pressure-scope/wording issues are mostly resolved. The remaining blockers are implementation-readiness gaps in U1/U2, not the deferred pressure work.

Findings:

1. U2’s load-bearing approach term has no ABI home. KTD4 defines `flip = (c_surface, density_gate, div_scale, water_splash_cap)` ([plan](/Users/cxc/Github/coffee-sim/docs/plans/2026-06-18-001-feat-twofield-surface-weighted-dissipation-plan.md:54)), but U2 uses `approach_scale` in `merge = max(-div·div_scale, approach·approach_scale)` ([plan](/Users/cxc/Github/coffee-sim/docs/plans/2026-06-18-001-feat-twofield-surface-weighted-dissipation-plan.md:102)). Also, `div_scale <= 0` would not disable the approach term unless the plan explicitly ties that sentinel to the whole merge discriminator. As written, the R5 density-only control is not actually guaranteed.

2. The inert per-particle `c` diagnostic is still underspecified. The plan says it is written under a diag flag with no new production buffer ([plan](/Users/cxc/Github/coffee-sim/docs/plans/2026-06-18-001-feat-twofield-surface-weighted-dissipation-plan.md:86)), but all current `dbg` lanes are occupied ([common.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/common.wgsl:52)) and the proposed `flip` vec4 is full. Current `g2p_water` binds `pos`, `vel`, `cmat`, `grid_vel`, and `solids` plus params ([mod.rs](/Users/cxc/Github/coffee-sim/src/solvers/twofield/mod.rs:1907)); adding a diagnostic buffer or writing an existing lane like `vel.w` needs to be explicitly designed and gated to preserve byte identity when off.

3. `approach_into_density` is physically plausible, but the implementation detail is not quite concrete. `g2p_water` currently computes B from `w` and `d`, not explicit `∇w` ([transfers.wgsl](/Users/cxc/Github/coffee-sim/src/solvers/twofield/transfers.wgsl:187)). The plan should specify the quadratic B-spline derivative formula, `1/h` scaling, and stable normalization such as `grad_m / max(length(grad_m), eps)`. `normalize(grad_m + ε)` is not precise enough and can bias the direction.

4. The `density_gate` lane and `merge_to_c()` curve are still not concrete enough for the R5 control. The plan says density-only keying must fail, but the formula shown does not define how density enters `c`. That leaves implementers inventing the actual smoothing curve.

Resolved from round 2: water cap coverage now correctly includes `p2g_water`, `drag_fold`, and `g2p_water` while leaving solid clamps on `max_speed`; Params resize to 272 is called out; pressure/crown wording is appropriately scoped to transfer-side liveliness.

Specific REVISE items:
- Define the full `flip` ABI, including `approach_scale` or a shared `merge_scale`, and make `div_scale <= 0` disable the entire merge discriminator.
- Specify the exact `c(density, merge)` curve, including how `density_gate` works and how the density-only negative control is produced.
- Give the `c` diagnostic a concrete flag and storage/readback path that preserves off-path byte identity.
- Add the exact mass-gradient derivative and normalization formula for WGSL.
- Fix the U1 comment note saying `160 bytes -> 256`; after `flip`, the Params layout is 272 bytes.

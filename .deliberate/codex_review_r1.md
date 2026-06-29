REVISE

The plan has the right high-level invariant, but U5/KTD-3 is not yet conservative by construction. The main blocker is multi-grain competition for one water particle: two independent gathers cannot guarantee that grain gains and water losses agree without a shared per-pair allocation rule that caps each water particle’s total committed volume.

**Required changes**

1. **KTD-3/U5: fix over-subscription before implementation.**  
   Current gather-not-scatter can double-spend water. If grains independently compute `take_g` from neighboring water availability, multiple grains may each claim the same `f_w * V_w`. Then the water-side pass must either remove more than available or disagree with grain gains.  
   Change U5 to a two-stage allocation:
   - grain pass computes demand/weight only;
   - water pass computes `D_w = Σ_g demand_share(g,w)` over neighboring grains and allocates `take_wg = avail_w * demand_share(g,w) / max(D_w, avail_w_rate_basis)` with a hard cap `Σ_g take_wg ≤ f_w * V_w`;
   - grain pass then gathers the same deterministic `take_wg` formula.  
   This is still atomic-free, but conservation comes from water-normalized allocation, not symmetric intention.

2. **KTD-3/R-2: choose continuous shrink, but mirror `f_w` to `pred.w` and weight PBF.**  
   Binary water is simpler but creates quantized volume loss, noisy wet fronts, and bad coupling to absorption rate. Continuous shrink is the right choice for this simulator, but only if `compute_lambda`/`compute_dp` see it.  
   Change the plan to require:
   - `predict` copies `pos.w -> pred.w`;
   - all kernels that use predicted water state preserve/read `pred.w`;
   - PBF density contribution from water is weighted by `f_w`;
   - lambda denominator/gradient scaling is consistent with the same effective particle volume/mass.  
   “Accept binary water for density while shrinking only affects bookkeeping” should be removed; it is physically inconsistent and will over-count density near wet fronts.

3. **KTD-1/U5/R5: add absorbed mass, not just absorbed volume.**  
   Volume conservation is handled by `V_eff = V_dry + V_abs`, but momentum conservation requires mass accounting. Absorbed water mass is `m_abs = rho_w * V_abs`; grain effective mass should be `m_dry + m_abs`, and water effective mass should scale with `f_w`. If the solver keeps using constant `params.grain_mass` and full water mass, the momentum test can pass only superficially.  
   Change U5/U6/U9 to test and use effective masses wherever velocities, momentum transfer, drag, and XPBD weighting depend on particle mass.

4. **KTD-5/R-5: do not raise packing clamp by `0.74 * (1 + r_max)`.**  
   Growing `V_eff` already raises local solid fraction. Raising the clamp as well permits impossible over-packing. The mitigation says “pick one primary mechanism”; make that decision now: use swelling through `V_eff`, keep the physical packing limit near random/close packing bounds, and clamp `alpha_s` to a plausible maximum. Do not allow `0.74 * 2.5 = 1.85`.

5. **U7: use a conservative cohesion combiner.**  
   Mean cohesion can create attraction between a wet/cohesive grain and a dry or saturated/noncohesive grain. Prefer `min(c_i, c_j)` or `sqrt(c_i*c_j)` for pair cohesion. `min` is safer and prevents a single wet particle from gluing noncohesive neighbors unrealistically.

6. **KTD-2/U4: `pos.w` is acceptable but needs stricter invariants.**  
   Phase-dependent aliasing is fragile but probably necessary under the bind limit. Strengthen the plan:
   - every position write preserves `.w`;
   - `pred.w` mirrors `pos.w`;
   - inactive water sentinel cannot be confused with phase or moisture;
   - tests cover `predict`, `apply_dp`, `finalize`, sorting, deactivation, and render/readback paths.  
   Do not initialize `pos.w` inside `predict`; seed at particle creation and preserve thereafter.

7. **U9 tests are not sufficient yet. Add gates for the actual failure modes.**  
   Add explicit tests for:
   - one water particle surrounded by multiple grains where total grain gain must not exceed water availability;
   - deactivation exactly conserving final residual volume or accounting for discarded epsilon;
   - continuous `f_w` affecting PBF density, not just absorption bookkeeping;
   - conservation using effective mass and volume after several substeps;
   - finest-grind/high-`r_max` swelling with packing clamp bounded below 1.0;
   - saturated grains in contact with water causing zero further absorption.

8. **Sequencing needs adjustment.**  
   U9’s core conservation and competition tests must be written before U5, not after U8. U6 depends on a finalized mass/volume convention from U5. U7 should come after saturation and capacity math are stable, but before diagnostics. Recommended order: U1/U2, U3, U4, U9 skeleton, U5, U6, U7, U8, full U9.

9. **Rate law needs bounded per-step transfer.**  
   `demand = k_abs * dt * deficit` can overshoot for large `dt`. Specify `demand = deficit * (1 - exp(-k_abs * dt))` or clamp to deficit. Also cap per-substep swelling/contact-radius growth for stability.

Verdict: revise, not reject. The volume-state idea is sound, but the current no-atomic transfer scheme is under-specified and can violate the hard conservation constraint in exactly the multi-grain case the plan flags as open.
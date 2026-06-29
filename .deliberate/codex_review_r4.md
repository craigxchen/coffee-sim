REVISE

One remaining correctness ambiguity blocks approval: the plan says near-empty active water `f_w <= eps` is skipped by PBF but “still absorbable later,” while the absorption eligibility/count contract repeatedly defines eligible waters as `f_w > eps`.

That creates a stranded-residual case: if a capped transfer leaves water with `0 < f_w <= eps`, it remains on the conservation books, but `wet_count`/`wet_water`/`wet_grain` will no longer include it, so it can never be absorbed later unless `eps` is only a PBF threshold and absorption uses a smaller `roundoff` threshold.

Fix needed: separate thresholds explicitly.

- PBF skip threshold: `f_w <= pbf_eps`
- Absorption eligibility: active and `f_w > absorb_roundoff`
- Deactivation: only when `new_f_w <= absorb_roundoff`

With that clarification, the round-3 fixes are otherwise captured correctly: no force-dump, `wet_count` reads frozen `pred.w`, merge mass uses pre-absorption `V_abs_snapshot`, PBF denominator is guarded, and grain-grain contact uses effective inverse masses. The two-sided cap remains conservation-sound for volume, and the exact merge is momentum-sound as written.
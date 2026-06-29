**REVISE**

Two issues still need edits before I’d approve.

1. **KTD-9 overstates velocity sampling.**  
   `apply_dp` does only write `pred`, and `finalize` is after coupling, so current PBF corrections do not overwrite `vel` before impact. But `predict` also does **not** write `vel = v_prev + g·dt`; it only writes `pred`. The coupling blocks freeze/read the stored `vel`, not the predicted velocity. So revise KTD-9 to say impact reads the pre-finalize stored velocity, unaffected by the current density solve, but not the actual `(pred-pos)/dt` predicted velocity and not current-step gravity unless already present in `vel`.

2. **The CFL cap bounds one pair, not the accumulated particle delta.**  
   For one pair, the formula is momentum-conserving and the CFL term is correct:  
   `max(|Δv_g|, |Δv_w|) = s_pair * max(m_w/M, m_g/M) <= k·h/dt`.  
   But each pass sums many pair contributions, so this does **not** mathematically bound a particle’s total `|Δv|` or pre-finalize `pred` displacement. The plan still claims a bounded per-substep displacement gate. Either revise that claim to “tested empirically under strong pour,” or add a conserving aggregate-safe cap/normalization, e.g. divide the pair CFL by a shared neighbor/weight count in the same spirit as the drag cap.

Everything else checks out:

- The shared `s_pair` cap before mass split is exactly conserving per pair.
- `min(s_raw, approach, cfl)` is dimensionally fine **as written**, because `s_pair` is a velocity impulse magnitude in this formula, not a dimensionless drag coefficient multiplying `v_rel`.
- U4/U5 are now acyclic: U4 probes/calibrates, U5 codifies.
- Reusing `_pad_coupling0` for one `impact_scale` float keeps `Params` size/layout stable; thresholds/CFL as WGSL consts are sound.
- The `r` floor and shared water effective-mass floor are sufficient guards against NaN/divergence.
- Once per substep is a reasonable incremental choice; under-resolution is a calibration/test outcome, not a plan blocker.
- The two-sided crater test plus sign-flip, conservation, no-creep, percolation/ponding, and bounded-displacement gates are the right verification shape.
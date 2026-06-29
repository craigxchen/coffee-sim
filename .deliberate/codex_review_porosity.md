REVISE

1. Yes, if the pair uses one shared scalar `s_ij`.

   Let `M = m_i + m_j`, `u = v_j - v_i`, and:

   `s_ij = f(sym(rate_i, rate_j), dt) * min(c_i, c_j)`

   Then:

   `dv_i = s_ij * (m_j / M) * u`

   `dv_j = s_ij * (m_i / M) * (-u)`

   So:

   `m_i dv_i + m_j dv_j = m_i s_ij m_j/M u - m_j s_ij m_i/M u = 0`

   Conservation depends on both particles using the same `s_ij`; symmetry ensures either kernel/order computes the same scalar.

2. The per-particle rate does not break conservation if the final pair scalar is symmetric, e.g. `f(sym(rate_i, rate_j)) * min(c_i, c_j)`. The cap interaction is fine if both cap and rate are combined symmetrically. Frozen Jacobi snapshots also preserve pairwise impulse symmetry within that iteration; they may affect convergence/physical lag, not momentum conservation.

3. Conservation is unaffected. `min` vs harmonic mean only changes drag magnitude. For Kozeny-Carman across porosity discontinuities, harmonic mean is usually preferable for permeability/resistance-like blends across layered media because the lower-permeability side dominates without hard-clipping as aggressively as `min`. `min(rate)` is suspicious if `rate ~ 1/k`: it selects the weaker drag/larger permeability side. If blending rates, harmonic mean may also underweight the high-resistance side; consider harmonic mean of `k`, then convert to rate, or use a symmetric resistance blend.

4. Packing both into one `f32` is the main correctness risk. Lossy folding can distort monotonicity, caps, or rates, and NaN/Inf/denormal behavior can silently corrupt drag. It will not break momentum conservation if both sides decode identically, but it can break physical bounds. Prefer a real spare lane, recompute one value, or use explicit quantization with tested ranges and clamps.

5. Yes, `alpha_s` will lag to wherever it was last produced in the step/substep. That is usually acceptable for an explicit/Jacobi-style coupling if the rate is clamped and smoothly updated. It becomes a stability risk if Kozeny-Carman sends `rate` sharply upward near low `phi_f`; clamp `phi_f`, cap `rate*dt`, and avoid discontinuous jumps.
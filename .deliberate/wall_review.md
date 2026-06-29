REVISE

The core physics approach is sound, but the plan has a sign-language error that is dangerous enough to revise before implementation.

**Required Findings**

1. **Fix the sign explanation for `g_b`.**
   The formula is correct:
   `g_b = rho0 * f_i * psi'(d) * n`, with `n = grad(SDF)` pointing into the fluid and `psi'(d) < 0`.

   But `g_b` itself points **toward the wall**, not away from it. The wall-repulsive correction is `lambda_i * g_b`, because over-density gives `lambda_i < 0`, so `lambda_i * g_b` points **into the fluid**. The plan’s wording that `g_b` is the inward wall-force direction is wrong and could cause an implementer to flip the sign.

2. **Derivative is correct.**
   For
   `psi(t) = 0.5 - (315/256) * poly(t)`,
   `poly'(t) = 1 - 4t^2 + 6t^4 - 4t^6 + t^8 = (1 - t^2)^4`.

   Therefore:
   `dpsi/dd = -(315/256) * (1 - t^2)^4 / h`.

   Clamping `t` to `[-1, 1]` is consistent with the clamped `boundary_psi`; derivative becomes zero at `|d| >= h`.

3. **`sum_g` yes, `sum_g2` no is correct.**
   The boundary density depends on `p_i`, so it contributes to the self Jacobian:
   `∇_{p_i} C_i`.

   The wall is not a DOF, so there is no boundary particle gradient term in the denominator. Do not add a fake `|∇C_b|^2` to `sum_g2`.

4. **`compute_dp` using only `lambda_i * g_b` is correct.**
   No `lambda_b`: the wall has no constraint multiplier.
   No `s_corr`: that term is a fluid-fluid artificial pressure/anti-clustering correction, not a wall pressure term.

5. **Route B is reasonable.**
   Staging `boundary_grad` from `compute_boundary` is the right performance choice here. It avoids re-running `solid_union` in both hot kernels. Moving from 8 to 9 storage buffers is fine under the project’s stated 16-buffer target, but the plan should keep the “grant check” as a hard failure, not a soft assumption.

6. **Add an explicit sign/physics test or diagnostic.**
   Integration gates are necessary but not sufficient for this kind of sign-sensitive core change. Add a small deterministic probe, even if host-side, that verifies for a particle near a cylinder wall:
   `dot(boundary_grad, hit.grad) < 0`
   and, for `lambda_i < 0`,
   `dot(lambda_i * boundary_grad, hit.grad) > 0`.

7. **Tighten the no-climb gate.**
   A raw `wall_max_y` can be noisy under an active pour because one splash particle can fail the test. Prefer a sustained/quantile metric: e.g. wall-region 95th/99th percentile height over a steady window, plus a minimum occupancy threshold for detecting a persistent sheet.

8. **Corner stability remains the main risk.**
   The SDF union gives a nearest-surface normal, so at floor-wall concave corners the density/gradient is only a one-sided approximation, not the true additive occlusion from both planes. The proposed sequencing is sound, but U6 should explicitly include a floor-wall corner fill test and inspect bounded correction magnitudes near the seam.

With those revisions, the plan’s physics is defensible: it completes the existing boundary density term with the matching constraint gradient and applies the correct PBF wall correction.
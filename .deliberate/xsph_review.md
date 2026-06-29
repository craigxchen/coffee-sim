APPROVE

No blocking physics issue. The normalization is the standard XSPH term: `Σ (m_j / ρ_j) (v_j - v_i) W_ij`. For equal-mass, full-volume water particles, `params.particle_mass / params.rest_density` is the right `V_w`. The sign is correct: it moves `v_i` toward neighbor velocities. No missing `dt` factor for this PBF-style XSPH form.

At default spacing this is not a concern. With the repo’s `rest_density(spacing=1, h=2, m=1)`, `V_w` is about `0.99`, so the effective viscosity barely changes. At spacing `0.15`, `V_w` is about `0.00334`, which cancels the `1/h^3` kernel-sum growth and brings the neighbor sum back to order one.

The post-clamp analysis is also right. `finalize` clamps `vel`, then `xsph` writes `vel_smoothed`, and the buffer copy overwrites `vel` with no second clamp. So an unnormalized XSPH overshoot can escape `max_speed`.

Remaining caveat, not a blocker for this focused fix: `xsph` skips grains as the target particle, but water still sums over all neighboring particles, including grains, because there is no `phase[j] == PHASE_WATER` guard in the neighbor loop. That is pre-existing, and mixed suites passing makes it acceptable here, but strict fluid-only XSPH would eventually guard water neighbors and possibly weight partial water neighbors by `pred[j].w`.

The regression test is adequate for this bug: it exercises fine spacing through the full solve and checks the observable failure mode, post-XSPH `vmax` exceeding the clamp. A GPU-skipped test is consistent with the surrounding suite, though it means non-GPU CI will not catch this specific regression.
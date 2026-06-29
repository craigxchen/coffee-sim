APPROVE

The round-4 algebra now closes:

- `A_eff = pi*r^2*discharge_coeff` and `exit_speed = flow_rate / A_eff`.
- `N_layer = ceil(A_eff * spacing / V_w)` gives layer throughput  
  `(exit_speed / spacing) * N_layer >= flow_rate / V_w`.
- The accumulator remains the source of truth, so `ceil` only gives per-layer capacity headroom; it does not over-emit.
- Using `r_eff = sqrt(discharge_coeff) * nozzle_radius` keeps the geometric spawn disc consistent with the same effective flux area for `discharge_coeff < 1`.

No remaining blocker for a competent implementer.

Minor non-blocking notes:
- The “packed to rest density” wording is algebraically exact only up to integer quantization from `ceil`; the U2 full-solve inlet sweep is the right empirical guard.
- At pour shutdown, make sure the implementation/test treats any integer accumulator backlog consistently; the plan already requires drawdown backlog drainage and ±1 rate fidelity, so this should be caught.
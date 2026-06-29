REVISE

**Findings**

1. **KTD-3 throughput algebra is still wrong with `discharge_coeff`.**  
   The plan states:

   `flow_rate = nozzle_area * exit_speed * discharge_coeff`

   but then claims:

   `(exit_speed / spacing) * N_layer = flow_rate / V_w`

   with `N_layer = round(nozzle_area * spacing / V_w)`.

   Ignoring rounding, that throughput is:

   `exit_speed * nozzle_area / V_w = flow_rate / (discharge_coeff * V_w)`

   not `flow_rate / V_w`, unless `discharge_coeff == 1`.

2. **`round(...)` can still reintroduce permanent backlog.**  
   Even apart from `discharge_coeff`, `round(A * spacing / V_w)` can round down. The invariant only holds if:

   `N_layer >= discharge_coeff * nozzle_area * spacing / V_w`

   under the current orifice relation. `round` does not guarantee that. Use `ceil(...)`, or maintain a fractional layer quota so integer layer sizes alternate without long-term drift.

3. **The rest-density inlet claim is inconsistent when `discharge_coeff < 1`.**  
   If particles are spread over the full geometric nozzle area but the physical flow is reduced by `discharge_coeff`, then the emitted layer is under-filled, not packed to rest density. To preserve the rest-density argument, either fold `discharge_coeff` into an effective area, e.g. `A_eff = discharge_coeff * nozzle_area` and spawn over `sqrt(discharge_coeff) * nozzle_radius`, or define `exit_speed = flow_rate / nozzle_area` for the particle stream and treat discharge separately.

**Required Fix**

Make the layer model use one consistent effective flux area:

```text
A_eff = nozzle_area * discharge_coeff
exit_speed = flow_rate / A_eff
N_layer_capacity = ceil(A_eff * particle_spacing / V_w)
```

Then the layer capacity is guaranteed `>= flow_rate / V_w`, and the accumulator can keep exact average count without permanent backlog.

Minor note: the Risks section still mentions a “per-frame burst cap”; tighten that wording so it does not revive the stale cap concept that R3 was supposed to remove.
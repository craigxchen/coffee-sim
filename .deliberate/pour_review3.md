REVISE

Round-2 fixes are mostly present, but the arclength-credit emitter is still not internally consistent enough to approve.

The blocker is the interaction between `flow_rate`, `V_w`, `N_layer`, `nozzle_radius`, and `exit_speed`.

The plan says the volume accumulator is the master count budget:

```text
accumulator += flow_rate / V_w * dt
want = floor(accumulator)
```

Then it meters that budget through axial layers:

```text
axial += exit_speed * dt
while axial >= particle_spacing:
    emit up to N_layer
```

But `N_layer = round(pi * nozzle_radius^2 / particle_spacing^2)` only matches the volume budget if you implicitly assume `V_w == particle_spacing^3` and `exit_speed` is physically coupled to `flow_rate / nozzle_area`. The plan does not state either invariant.

For a physically consistent stream, each axial layer of thickness `particle_spacing` represents volume:

```text
nozzle_area * particle_spacing
```

So the count capacity per layer should be tied to:

```text
N_layer ≈ nozzle_area * particle_spacing / V_w
```

or the plan must explicitly derive `exit_speed` from:

```text
flow_rate = exit_speed * nozzle_area
```

and prove the chosen `N_layer`/spacing approximation preserves the requested particle count within tolerance. As written, the layer rule is a density throttle that may under-deliver and grow backlog forever whenever:

```text
flow_rate / V_w > (exit_speed / particle_spacing) * N_layer
```

That violates R1/R3/R9 unless the plan pins the inequality and tests it as a configuration invariant.

There is also stale contradictory language in KTD-3. It still describes the rejected per-frame cap:

```text
requested ≤ exit_speed·dt / particle_spacing
```

and a per-frame stagger by `i/requested`, while U2 describes axial-credit layers. Those are different algorithms. A competent implementer should not have to decide which one is authoritative.

Round-2 fold verification:

- Arclength-credit emitter: partially folded, but not yet correct because `N_layer` is not reconciled with `V_w` and `flow_rate`.
- Backlog drain after flow stops: folded conceptually with `flow_rate > 0 || backlog pending`.
- Density bound: improved, but still underspecified because radial layer count uses spacing area rather than particle volume unless an invariant is added.
- `read_temperature` phase mirror: folded.
- `total_emitted_water_mass` and KEEP bed-gain gate: folded.
- V60 particle budget and `5.20 ml/sim-unit^3` units: folded.

Required revision: replace the stale KTD-3 text with the U2 arclength-credit algorithm, then define the flow consistency invariant explicitly. Either compute `N_layer` from `nozzle_area * particle_spacing / V_w`, or derive/validate `exit_speed`, `nozzle_radius`, `particle_spacing`, and `V_w` so the layer capacity can deliver `flow_rate / V_w` without permanent backlog. Then add a test that fails if requested flow exceeds layer throughput.
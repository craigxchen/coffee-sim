REVISE

**Blocking Findings**
1. **KTD-3/U2 still does not define a valid density-safe emitter.**  
   `max_axial = floor(exit_speed * dt / particle_spacing)` can be `0` for normal pours. From [KEEP.md](/Users/cxc/Github/coffee-sim/KEEP.md:24), pour speeds are ~3.3 to 12.5 scene-units/s; at 60 Hz with `particle_spacing = 1`, this gives 0.055 to 0.208 spacing/frame, so `floor(...) == 0`. The emitter can starve forever. If clamped to 1, it then violates cross-frame spacing by spawning a new particle at the nozzle before the previous one has moved one spacing away. The plan needs an axial/arclength credit carried across frames, not just a per-frame cap.

2. **Backlog conservation conflicts with `flow_rate > 0` gating.**  
   U2 says `emit()` is wired into `step()` gated on `flow_rate > 0`, but the same unit relies on backlog metering. If a recipe window ends while backlog remains, drawdown has `flow_rate = 0` and the backlog never drains. That is still dose loss, just delayed. Call the emitter while `flow_rate > 0 || backlog/accumulator can emit`, or explicitly model backlog as a delayed source tail and test final-pour backlog drain.

3. **The promised support-volume/grid-cell density bound is still underspecified.**  
   R3 promises a cap per support volume / grid cell and a disc radius tied to spacing, but U2 only gives axial cap + burst cap. “Disc radius scales with spacing and count” leaves the implementer inventing the actual criterion. Define a quantitative packing rule, for example a minimum 3D emitter-space separation or a max count per axial layer and radial disc derived from `particle_spacing`, `cell_size`, and `support_radius`.

**Fold Audit**
- R1 active-count audit: mostly real. It names the current copy spans, pass-selection issue, phase mirror, and reset path. Add that all host readback helpers using `initial_phases` must switch too, notably `read_temperature`, not only `sample_extraction`.
- R3 source conservation: conceptually fixed, but add an explicit `total_emitted_water_mass` / emitted-particle counter so U4 conservation is not reconstructed ad hoc.
- R4 capacity/perf: direction is acceptable, but the V60 particle budget should be pinned to a number, not just “stated later.”
- R5 clamp-before-decrement: correct, but undermined by the backlog gating issue above.
- R6 units: formula is good; pin the actual scene-volume scale, likely `5.20 ml / scene-unit^3` from KEEP, and document `flow_rate` as scene-volume/s, not “scene units/s.”

The architecture is close and implementable after these revisions, but KTD-3 is still load-bearing and currently leaves enough ambiguity to produce either no pour or an over-dense inlet.
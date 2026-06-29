REVISE

The plan is directionally sound, but not implementable safely as written. It has several physics and solver-contract gaps that a competent implementer would still have to invent.

**Main blockers**

1. **Active-count refactor is incomplete.**
   The plan catches most per-particle buffers, `sorted_indices`, and readbacks, and correctly leaves `cell_start = num_cells + 1`. But it misses several active-vs-capacity consequences:
   - copy spans in `step()` currently use `self.particle_count * 16` for `vel_frozen`, `vel`, `chem_frozen`; those must become active-count spans, while buffers allocate to capacity.
   - `sample_extraction()` indexes `self.initial_phases[i]`; emitted particles would exceed that vector unless a live phase mirror or GPU phase readback is used.
   - `seed()`/`reset()` must restore `active_count = seed_count` and only write/zero seed-active ranges, not assume `particle_count == seed_count`.
   - scene species flags are currently derived from seeded regions. A V60 pour scene that seeds only grains will set `has_water = false`, so water, coupling, wetting, and extraction passes will never run after emission. The plan must add “potential emitted water” to pass selection.

2. **KTD-3 is not sufficient as a PBF-safe inlet guarantee.**
   Sub-frame axis staggering helps, but it does not bound local density. Failure cases remain:
   - high flow or low frame rate can place many particles inside one `exit_speed * dt` stream segment;
   - if `exit_speed * dt < particle_spacing`, the stagger is still over-dense;
   - the golden-angle disc can still be too narrow relative to support radius;
   - particles emitted outside/near the domain clamp into boundary grid cells;
   - all particles are activated before the frame/substeps, so a burst still enters the solve simultaneously.
   
   The plan needs an explicit inlet density criterion: max emitted particles per support volume/cell, minimum axial spacing, nozzle radius tied to particle spacing and flow, burst backlog behavior, and tests that sweep flow, dt, exit speed, and nozzle radius.

3. **Source conservation is mostly right but underspecified.**
   Emitting water with `c = 0`, `f_w = 1`, phase water does not add solute, so dissolution should still conserve solute in isolation. But the plan must explicitly update the inventory/readout paths for emitted particles. Current yield/TDS readout uses the initial phase list, so it will miss or panic on emitted particles. Also, wetting absorption of already-extracted water remains a solute sink unless modeled as absorbed-solute transfer; the accepted 1-2% drift should be stated as applying after source emission too.

4. **Capacity = full dose is physically clean but not performance-safe by assertion.**
   It avoids drain banking/readback, but active count monotonically rises to the whole recipe dose. That changes dispatch cost, grid scatter size, neighbor gather cost, and cup-pool density. “Cup water is quiescent” does not make it cheap: kernels still dispatch over it, sort it, and may do dense cup-neighbor loops. This can be acceptable for a bounded V60 scene, but the plan needs a capacity/perf budget in particles, not just “250 ml”, plus a gate on max occupancy and dispatch time.

5. **R4 is internally inconsistent.**
   “No silent drop” is achieved by warn + clamp, but “no dropped water” is not. If capacity is full dose, overflow should be treated as a scenario/build error in the pour example/tests, not just a warning. Also, if `requested` is subtracted from the accumulator before clamping, overflow permanently loses dose; that is fine only if counted as explicitly dropped and surfaced as failure for recipe scenes.

6. **Units are not pinned down.**
   `EmissionInput.flow_rate` currently says scene units/s, while the recipe layer uses mL/s. The plan says `particles_per_volume` and `Scene.water_ml` conversion but does not define the conversion from mL to scene volume/particle count. Old code had `PARTICLES_PER_ML`; the new solver has `particle_mass / rest_density` in simulation units. This must be specified before implementation.

**Bug Audit**

Mostly plausible, but overstated in two places:
- “Pancake emission is fatal in PBF” is a reasonable risk, but the proposed stagger is not a proof-level fix.
- “Drain = pool live” fixes slot reclamation by avoiding deactivation, but it does not fix capacity pressure; it shifts it into active-count growth and dense cup cost.

**Verdict**

Revise before implementation. The high-level architecture is right: fixed GPU capacity, active count, CPU-side activation, no float atomics, live cup readout. But the plan needs explicit active/capacity invariants, dynamic water pass selection, unit conversion, inlet density bounds, overflow semantics, and conservation/readout updates for emitted particles.
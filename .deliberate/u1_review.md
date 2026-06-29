APPROVE

I do not see a missed `particle_count` site that would pull dormant slots into the live solve.

The mapping is correct: allocation-sized buffers use `capacity`; dispatch groups, `params.particle_count`, GPU copy spans, readbacks, `ParticleBuffers`, and `Metrics` use `active_count`. `sorted_indices` is capacity-sized, which is necessary because future active growth can fill the whole pool. WGSL continues to guard every particle-indexed kernel on `params.particle_count`, and host code refreshes that from `active_count` before dispatch.

The seed buffer rewrite is safe. `Self::storage` keeps `STORAGE | COPY_DST` and the passed `COPY_SRC`, so `pos`, `phase`, and `chem` retain the prior copy/readback capabilities. The write range is exactly the seed slice at offset 0. Dormant slots are not dispatched, not scattered into the grid, and not read back, so their contents are inert.

`has_water = seeded_water || scene.declares_pour()` is correct for the next unit. Existing non-pour scenes remain unchanged because `pour_water_ml` defaults to `0.0`, and the explicitly updated literals also set `0.0`.

The `phase_mirror` replacement is also correct for host-side readouts. `initial_phases` remains the reset baseline, while `phase_mirror` represents the live active set. `seed()` restores `active_count` to the seed count and resets `phase_mirror` from `initial_phases`, so reset cannot expose stale dormant phase tags.

`seed()`/`reset()` writing only the seed range is correct for this refactor: after reset, `active_count` drops to seed, `params.particle_count` is refreshed on the next `step()`, and dormant data is outside all live spans.

The capacity formula is sane for the stated model: `seed_count + ceil(pour_water_ml / 5.20 / V_w)`, with `V_w = particle_mass / rest_density`. It allocates one particle per water-particle volume after converting mL to sim volume.

The only non-blocking nit is the comment mentioning `v60_pour()` before such a constructor exists, but that is not a physics or correctness issue for Unit 1. I did not run the full suite because this environment is read-only, but the code-level audit supports the byte-unchanged guarantee for non-pour scenes.
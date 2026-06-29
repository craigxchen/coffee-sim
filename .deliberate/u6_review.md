APPROVE

No blocking correctness issues found for Unit 6.

Energy exchange is conservative for active particles: `th_pair_heat` negates both `raw` and `cap` when `i/j` are swapped, so the clamp interval is mirrored and `q_ji = -q_ij` with frozen `chem_frozen` temperatures. Thermal reads from `chem_frozen` and writes only `chem[i]`, so there is no read/write race.

The zero-capacity handling is effectively symmetric: neighbors with `c_j <= 1e-6` are skipped, and self particles with `c_i <= 1e-6` do not apply `q_sum / c_i`. Exact zero capacity also makes the pair cap zero. Sub-threshold particles may still compute a discarded `q_sum`, but no pair energy is applied on either side. Ambient can still change their stored temperature, but that is external heat exchange and contributes no enthalpy for zero capacity.

Lane handling is correct: `th_temp` reads water temperature from `chem.y` and grain temperature from `chem.z`; final writes update only `.y` for water or `.z` for grain while preserving post-dissolution solute/pool lanes from the second snapshot.

Snapshot ordering is correct: after `dissolve_grain` and `dissolve_water`, `chem` is copied again into `chem_frozen` before `thermal_exchange`, so thermal preserves the updated dissolution state. The pass writes only its own particle slot.

Capacity matches the stated model: water uses `particle_mass * f_w * cp_water`; grain uses `(grain_mass + rest_density * V_abs) * cp_grain`, reading post-wetting `pos.w`.

Storage-buffer budget is within WebGPU baseline: `thermal_exchange` uses 6 storage buffers plus the uniform params buffer, below the 8 storage-buffer stage limit. I did not run tests in this read-only sandbox.
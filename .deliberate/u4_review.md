APPROVE

No blocking issues found.

`Scene::v60_pour()` is structurally correct: it reuses the V60 solids and bed region, seeds only grains, sets `water_ml = 0.0`, and `pour_water_ml = 250.0` makes `declares_pour()` true so the solver allocates water headroom and enables water/mixed passes for a grain-only seed.

The driver mapping is sane: recipe `(x,z)` normalized coordinates are scaled by bed radius into world-space kettle position, and mL/s is converted to sim-volume/s with `/ 5.20`, matching the solver’s pool-sizing constant.

The integration gate checks the right mechanisms. `tds > 0` is not trivially true here because there is no seeded water column and TDS is computed only from cup-region water; nonzero TDS implies extracted solute reached cup water. Yield rising is mechanism-level, not absolute calibration. The volume balance is also the right conservation check: emitted water volume should equal remaining water-particle volume plus grain absorbed volume. Extraction changes solute inventory/concentration, not water moisture volume, so it should not enter that balance. The 2% tolerance is loose but acceptable for a full GPU integration gate; it should catch real volume loss while avoiding numeric/ordering noise.

The determinism gate is justified. Exact `active_count` equality effectively asserts deterministic CPU-side emission count because seed count is fixed, while the loose yield band acknowledges non-bit-exact GPU scatter/solve ordering. That does not mask the emission determinism bug this gate is meant to catch; stricter GPU aggregate equality would be brittle.

Existing non-pour scenes look unaffected: capacity remains seed-sized unless `pour_water_ml > 0`, and prior tests cover that path. Minor nit only: in the integration test `flow = 8.0` is sim-volume/s, while the example’s `FLOW` is mL/s before conversion, so naming/comment clarity could prevent future confusion. I did not run the GPU tests in this read-only review environment.
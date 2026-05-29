# Roadmap

This is the current planning surface for `main`. It replaces the older
branch-audit and single-purpose planning notes that predated the physics
mainline merge.

## Product Goal

Build a browser-native pour-over simulator where visible behavior comes from a
coherent particle-first physical model:

- coherent kettle inflow
- near-incompressible free water
- deformable porous coffee bed
- filter-mediated drainage
- extraction layered on top of the hydraulic state

## Current Physics Direction

The active solver model is:

- water remains explicit particles on a shared MPM grid
- pressure projection enforces near-incompressibility
- coffee grounds are a separate particle material coupled through the same grid
- bed porosity, permeability, saturation, and compaction feed the water solve
- rendering stays downstream of particle state

Current mainline already has finite pore capacity, Darcy/Brinkman resistance,
bed compaction feedback, suspended coffee support, a GPU cross-section view,
and particle-carried dissolved-solids extraction.

## Near-Term Physics Work

Priority work:

- improve coffee-particle advection so wet grounds are swept into flow instead
  of mostly moving aside from water
- add grind-size distribution across coffee particles
- let fines suspend and migrate more readily than coarse particles
- couple local grind distribution and compaction to permeability
- calibrate extraction rates against measured pour-over or espresso curves
- add an outlet/cup accumulator for final beverage TDS and extraction yield

Known solver gaps:

- free-stream spout translation can still inject excessive lateral momentum
- high-velocity free jets still show small side-to-side wobble from sparse
  particle sampling and grid-transfer noise; revisit with a more coherent
  airborne jet model rather than treating it as air resistance
- high-viscosity pooled-water kinetic-energy regression needs follow-up
- pressure projection has no residual/convergence readback in the browser
- V60/filter geometry is still partly duplicated between Rust setup and WGSL

## Validation Direction

`crates/sim-wasm/src/mpm_3d/physics_tests.rs` is the authoritative regression
surface.

Validation should lock:

- water mass conservation
- free-stream continuity
- hydrostatic and cup-floor stability
- center-pour pool buildup and drawdown
- dry-bed settle stability
- bounded wet-bed deformation and filter containment

Validation should not lock temporary tuning constants that are expected to be
replaced by better material models.

## Performance Direction

Implemented wins:

- cached SDF cell classification texture
- `clear_buffer` path for hot buffer resets
- empty-cell early exits in hot grid passes
- runtime-tunable pressure iteration count

Deferred work:

- timestamp-query profiling
- adaptive substeps or pressure iterations based on measured residuals
- settled-particle sleeping only after profiling identifies the bottleneck

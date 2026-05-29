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

- add pressure residual/readback metrics so projection work is driven by
  measured convergence rather than fixed iteration counts
- make kettle inflow a coherent geometric slug across the nozzle aperture and
  emission length, not independent per-particle jitter
- improve coffee-particle advection so wet grounds are swept into flow instead
  of mostly moving aside from water
- add grind-size distribution across coffee particles
- let fines suspend and migrate more readily than coarse particles
- add local bed/fines support constraints inspired by XPBD, scoped to coffee-bed
  mechanics rather than a generic soft-body solver
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

## Genesis-Informed Scope

The Genesis engine is a broad robotics simulator. Its useful lessons for this
project are the pieces that make a pour-over more physically coherent without
turning `coffee-sim` into a general multi-solver engine:

- use residual-driven fluid observability from DFSPH, but keep the WebGPU grid
  pressure projection as the primary water solver
- borrow geometric emitter structure for kettle streams, but keep flow-rate and
  dose accounting in `inflow.rs`
- borrow XPBD-style local constraints for wet grounds and fines, but scope them
  to coffee-bed support, suspension, redeposition, and compaction
- borrow explicit coupler boundaries for water-bed-filter exchange, but keep
  simulation truth in the existing particle/grid buffers
- borrow contact material parameters for filter/dripper behavior, but avoid a
  generic rigid-body contact stack

Out of scope unless a later profile or validation result proves otherwise:

- a full SPH, PBD, FEM, SAP, or IPC solver port
- robotics-style articulation coupling
- broad force-field abstractions in production physics
- differentiable checkpointing machinery

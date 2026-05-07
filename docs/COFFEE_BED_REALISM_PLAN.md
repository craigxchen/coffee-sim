# Coffee Bed Realism Plan

## Goal

Make the coffee bed behave like a porous granular material coupled to the
pressure-projected water solver. The bed should not be tuned as a scene-specific
damper or an elastic object that springs back to its original shape. It should
store solid packing state, pore water, and permeability, then exchange mass and
momentum with water through local conservation laws.

## Design Principles

- Keep free water governed by the water solver. Bed logic should feed porosity,
  permeability, pore storage, and drag into that solver rather than override
  water velocity in hand-picked regions.
- Treat grounds as a damped, frictional, plastic porous solid. Compression can
  persist; rebound should be a consequence of later hydraulic or granular state,
  not a spring to the initial bed.
- Preserve mass explicitly. Free-water loss, pore-water gain, bed-held water,
  and drained water must be auditable in tests.
- Prefer local material state over scene coordinates. Coupling should depend on
  saturation, porosity, permeability, pressure gradient, and particle/grid
  overlap.

## Current State

- Bed particles carry solid position, velocity, a mutable rest position, initial
  packed rest height, pore-water state, porosity, permeability, saturation, and a
  compaction diagnostic.
- Water uses the MLS-MPM/APIC particle-grid path plus pressure projection.
- Bed-overlapped cells that contain water now participate in the same pressure
  solve as free water. This is only the first porous-compatible step; pore
  storage and Darcy resistance are still handled by the older particle-side
  coupling.
- Initial implementation slice: bed dynamics are overdamped and plastic rather
  than spring-restoring. Water impact can compact the bed; the bed no longer
  automatically rebounds toward the original packed height.

## Stage 1: Plastic Bed Dynamics

Status: in progress.

- Remove elastic shape-memory rebound from the bed update.
- Keep per-particle rest height as a mutable plastic packing reference.
- Lower rest height when water impact compresses a particle beyond a threshold.
- Clamp upward dilation tightly so impact/contact cannot create visible popping.
- Keep dry-bed and first-impact regression tests green.

Validation:

- Dry bed does not creep after initial settle.
- First water impact does not collapse the bed.
- After pour-off, bed mean height should not rebound materially above its wet
  compacted state.

## Stage 2: Porosity and Permeability Fields

Status: planned.

- Deposit bed solid fraction to grid cells.
- Derive porosity from solid fraction and particle compaction.
- Compute permeability from porosity and grind scale, initially with a
  Kozeny-Carman-style relationship.
- Store pore capacity per bed/grid cell so free water has a physical place to
  enter before exiting the bed.

Validation:

- Packed-bed permeability changes monotonically with porosity.
- Pore capacity is bounded and finite.
- Free-water mass loss matches pore-water gain during active wetting.

## Stage 3: Porous Pressure Coupling

Status: partially started.

- Keep bed-coupled occupied cells in the pressure solve.
- Let pressure projection include pore-volume capacity and porosity-scaled
  divergence.
- Add Darcy or Darcy-Forchheimer resistance proportional to pressure gradient
  and inverse permeability.
- Keep the same pressure system authoritative for free water and bed-overlapped
  water.

Validation:

- Vertical column drawdown rate follows permeability.
- Bed residence time increases without destroying total water mass.
- Post-bed stream velocity is lower because of hydraulic resistance, not a
  downstream damping region.

## Stage 4: Granular Yield

Status: planned.

- Add a frictional plastic yield model for grounds, starting with a simple
  Drucker-Prager-like cap/yield rule.
- Let wetness and compaction alter mobility.
- Add grind-size parameters only after the core porosity/permeability path is
  stable.

Validation:

- Dry bed remains stable.
- Wet bed compacts under flow and remains mostly compacted after pour-off.
- Lateral slumping is bounded by frictional yield, not hard scene clamps.

## Stage 5: Extraction and Fines

Status: deferred.

- Couple extraction to pore-water residence time and saturation.
- Add optional fines migration or local permeability reduction after the porous
  flow model is stable.
- Avoid promoting grind-size redistribution until pressure/free-surface and bed
  hydraulics are trustworthy.

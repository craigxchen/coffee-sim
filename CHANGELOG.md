# Changelog

## Unreleased

**Date:** 2026-04-10

Granular bed mechanics overhaul focused on making the dry coffee bed behave
more like a packed frictional skeleton instead of a damped gel.

What changed:
- Added grind-size-driven bed hydraulics with non-uniform per-particle
  porosity, permeability, pore capacity, and rendered particle size
- Switched bed startup to a dropped-grounds workflow with manual pour start in
  the web demo
- Added bed deformation-gradient state and fixed-corotated dry elasticity
- Replaced the earlier ad hoc dry-plastic proxy with a stress-based granular
  plasticity pass that tracks both shear hardening and irreversible
  compaction
- Added rigid-support validation mode so dry-bed mechanics can be tested
  independently of the deformable filter support, and exposed that preset in
  the browser demo
- Expanded dry-bed regression coverage with rigid-support settle and long-run
  creep tests
- Fixed the hydrostatic branch of the granular return map so pure dilation
  remains elastic while capped pure compaction still accumulates irreversible
  compaction state

Known limitations:
- Filter contact is still a separate unilateral/frictional approximation rather
  than a full contact solve
- The free water jet can still fragment unrealistically in mid-air; that is a
  water-phase cohesion / transfer issue, not a bed-mechanics issue
- Wet weakening is still a temporary closure, not a true pore-pressure /
  effective-stress formulation
- Bed drawdown and redistribution are still local-storage approximations, not a
  full intra-bed drainage model

## Demo v0 — Baseline MPM Pour-Over Prototype

**Date:** 2026-04-06
**Commit:** 6946339

Browser-native MPM pour-over simulation with interactive pouring. This is the
baseline state before implementing the Moroney-inspired double-porosity bed model.

What works:
- WebGPU MLS-MPM water particles with P2G/G2P pipeline
- V60 dripper and carafe SDF geometry
- Kettle-angle-controlled inflow with adjustable pour rate
- Bed particles with basic drag-based absorption on contact
- Interactive spout positioning and demo autoplay loop
- Particle rendering with phase-based coloring

Known limitations:
- Water disappears instantly on bed contact (no surface pooling)
- No intergranular/intragranular saturation model
- No head-driven drainage — outflow is purely drag-based
- No compaction or bed memory effects
- No extraction model

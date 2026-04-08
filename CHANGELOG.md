# Changelog

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

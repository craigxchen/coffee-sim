# Changelog

## Unreleased

Mainline physics and documentation cleanup after the physics-mainline merge.

What changed:
- rewrote `README.md` around the current MPM browser app
- added `docs/ARCHITECTURE.md` as the as-built implementation map
- consolidated active planning into `docs/ROADMAP.md`
- removed stale branch-audit and historical issue files that no longer described `main`
- merged the porous-bed physics mainline into `main`

Known issues still open:
- free water jet can still fragment unrealistically in mid-air
- coffee-bed advection, grind-size distribution, and extraction color remain future work
- filter contact is still an approximation rather than a full contact solve

## Demo v0 — Baseline MPM Pour-Over Prototype

**Date:** 2026-04-06
**Commit:** `6946339`

Browser-native MPM pour-over simulation with interactive pouring.

What worked at that milestone:
- WebGPU MLS-MPM water particles with P2G/G2P pipeline
- V60 dripper and carafe SDF geometry
- kettle-angle-controlled inflow with adjustable pour rate
- bed particles with basic drag-based absorption on contact
- interactive spout positioning and demo autoplay loop
- particle rendering with phase-based coloring

Known limitations at that milestone:
- water disappeared too quickly on bed contact
- no intergranular/intragranular saturation model
- no head-driven drainage
- no compaction or bed memory effects
- no extraction model

//! Shared physics imported by **every** solver (permeability, extraction kinetics,
//! cohesion, wetting, thermal) + material params, so cross-solver comparison is fair:
//! same physics, different numerics. This is the layer that replaces a generic coupler.
//!
//! Phase 0: the `Materials` placeholder (so the `Solver::build` signature is stable).
//! The physics functions and calibration tables land in the models phase; their target
//! values from v1 are recorded in `KEEP.md` §1.

/// Coffee + water material parameters and presets. Empty placeholder for Phase 0; fields
/// (grind diameter, porosity, kinetics constants, …) arrive with the models phase.
#[derive(Clone, Debug, Default)]
pub struct Materials {}

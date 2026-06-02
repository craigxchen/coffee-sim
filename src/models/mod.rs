//! Shared physics imported by **every** solver (permeability, extraction kinetics,
//! cohesion, wetting, thermal) + material params, so cross-solver comparison is fair:
//! same physics, different numerics. This is the layer that replaces a generic coupler.
//!
//! Phase 1.1: `Materials` carries the water params the PBF water core needs. The physics
//! functions and calibration tables land in later steps; their target values from v1 are
//! recorded in `KEEP.md`.

/// Coffee + water material parameters and presets.
///
/// For the water core only the water discretization params are used. `rest_density` ρ₀ is
/// **derived** at solver build time from the rest lattice (`Σ_j W_poly6` at `particle_spacing`),
/// so it is not stored here — only the inputs that determine it.
#[derive(Clone, Debug)]
pub struct Materials {
    /// Particle spacing (the lattice pitch the water block is seeded at).
    pub particle_spacing: f32,
    /// SPH support radius `h` (≈ 2·spacing for a healthy neighbor count).
    pub support_radius: f32,
    /// Per-particle mass (1.0; ρ₀ is computed from the lattice to match).
    pub particle_mass: f32,
}

impl Default for Materials {
    fn default() -> Self {
        Self {
            particle_spacing: 1.0,
            support_radius: 2.0,
            particle_mass: 1.0,
        }
    }
}

//! Shared physics imported by **every** solver (permeability, extraction kinetics,
//! cohesion, wetting, thermal) + material params, so cross-solver comparison is fair:
//! same physics, different numerics. This is the layer that replaces a generic coupler.
//!
//! Phase 1.1: `Materials` carries the water params the PBF water core needs. Phase 1.2 adds the
//! dry-bed grain params (contact, friction, cohesion). The physics functions and calibration
//! tables land in later steps; their target values from v1 are recorded in `KEEP.md`.

pub mod cohesion;
pub mod permeability;

/// Coffee + water material parameters and presets.
///
/// For the water core only the water discretization params are used. `rest_density` ρ₀ is
/// **derived** at solver build time from the rest lattice (`Σ_j W_poly6` at `particle_spacing`),
/// so it is not stored here — only the inputs that determine it.
///
/// The grain params drive the granular bed (grain phase). They're folded in here rather than a
/// separate `materials/` module: the single solver reads one material spec.
#[derive(Clone, Debug)]
pub struct Materials {
    /// Particle spacing (the lattice pitch the seed block is laid out at).
    pub particle_spacing: f32,
    /// SPH support radius `h` (≈ 2·spacing for a healthy neighbor count).
    pub support_radius: f32,
    /// Per-particle mass (1.0; ρ₀ is computed from the lattice to match).
    pub particle_mass: f32,

    // --- grain (dry bed) ---
    /// Contact diameter `d`: grains within this distance push apart (≈ particle spacing).
    pub grain_diameter: f32,
    /// Water↔grain exclusion contact distance. Defaults to `grain_diameter` (water rests on the
    /// bed). Set **below** the grain spacing to let water thread the pores of a packed grain wall
    /// (porous through-flow) while grain–grain contact still holds the wall together.
    pub water_grain_distance: f32,
    /// Grain–grain Coulomb friction coefficient (the slope-holding yield stress).
    pub friction_mu: f32,
    /// Grain–boundary (floor/wall) Coulomb friction — stops the pile sliding flat.
    pub floor_mu: f32,
    /// Dry inter-grain cohesion strength (weak; 0 disables it). See [`cohesion::dry`].
    pub dry_cohesion: f32,
    /// Per-frame grain velocity retention (rolling-resistance proxy; <1 bleeds energy).
    pub rolling_damping: f32,

    // --- coupling (water ↔ bed) ---
    /// Per-grain particle mass (grains are denser than water; sets the interphase mass weighting).
    pub grain_mass: f32,
    /// Bed porosity φ (pore/fluid volume fraction of a packed bed) — the Kozeny–Carman input.
    pub porosity: f32,
}

impl Default for Materials {
    fn default() -> Self {
        Self {
            particle_spacing: 1.0,
            support_radius: 2.0,
            particle_mass: 1.0,
            // Grain defaults (coffee-grounds-ish): grains touch at the lattice pitch, steep
            // friction, light cohesion + rolling damping so the sphere pile isn't too shallow.
            // Calibrated against the standing-heap invariant, not an exact repose angle.
            grain_diameter: 1.0,
            water_grain_distance: 1.0, // = grain_diameter: water rests on the bed (override for porous flow)
            friction_mu: 0.8,
            floor_mu: 0.8,
            dry_cohesion: cohesion::dry(),
            rolling_damping: 0.9,
            // Coupling: grains a bit denser than water (a settled bed resists being lifted); 40%
            // bed porosity (KEEP.md §1).
            grain_mass: 1.5,
            porosity: 0.40,
        }
    }
}

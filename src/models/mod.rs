//! Shared physics imported by **every** solver (permeability, extraction kinetics,
//! cohesion, wetting, thermal) + material params, so cross-solver comparison is fair:
//! same physics, different numerics. This is the layer that replaces a generic coupler.
//!
//! Phase 1.1: `Materials` carries the water params the PBF water core needs. Phase 1.2 adds the
//! dry-bed grain params (contact, friction, cohesion). The physics functions and calibration
//! tables land in later steps; their target values from v1 are recorded in `KEEP.md`.

pub mod cohesion;
pub mod extraction;
pub mod fines;
pub mod permeability;
pub mod thermal;
pub mod wetting;

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
    /// Water↔grain coupling/porosity support radius `h_c`.
    ///
    /// `0` means derive it at solver build time: keep legacy single-resolution scenes on the water
    /// PBF radius, and widen only when grains are coarser than water.
    pub coupling_radius: f32,
    /// Bed porosity φ (pore/fluid volume fraction of a packed bed) — the Kozeny–Carman input.
    pub porosity: f32,

    // --- wetting / cohesion (Phase 1.4) ---
    /// Moisture ratio at saturation `r_max` (mass water / mass dry grain). Coffee ≈ 1.5 — grounds
    /// absorb ~1.5× their dry mass, which is why bloom/drawdown-slowing are pronounced.
    pub r_max: f32,
    /// Grain/water density ratio `ρ_s/ρ_w`. Converts absorbed water mass into the grain's swelling
    /// volume so absorption conserves volume (fluid lost = solid gained).
    pub rho_ratio: f32,
    /// Saturation `s ∈ [0,1]` at the cohesion-curve peak (~0.4; capillary bridges are strongest at
    /// partial saturation, collapse at full saturation). See [`cohesion::for_saturation`].
    pub s_peak: f32,
    /// Peak wet cohesion strength `c_max` (curve scale). 0 until calibrated to a wet-dome repose.
    pub c_max: f32,

    // --- extraction / thermal (Phase 1.5) ---
    /// Base fast/slow-pool dissolution rate constants (1/s). Fast = surfaces/fines, slow = interiors.
    pub k0_fast: f32,
    pub k0_slow: f32,
    /// Arrhenius `Ea/R` (temperature sensitivity of extraction) and reference temperature
    /// (`k_T(t_ref) = 1`). Temperatures are a normalized scale (pour ≈ `t_ref`, ambient below).
    pub ea_over_r: f32,
    pub t_ref: f32,
    /// Max solute concentration `c_sat` (8% by mass; the `(1−c/c_sat)` driving force → 0 here).
    pub c_sat: f32,
    /// Fraction of the soluble dose in the fast pool, and the soluble fraction of the dry dose
    /// (≈0.28 — only this much of a grain is extractable). Used to seed the grain pools.
    pub fast_fraction: f32,
    pub soluble_fraction: f32,
    /// Reference grind diameter for the surface-area factor (`area ∝ d_ref/d_p`).
    pub d_ref: f32,
    /// Flux half-saturation for the flow→extraction bridge `u/(u+u_half)`.
    pub u_half: f32,
    /// Moisture-gate onset saturation (a grain below this is too dry to extract).
    pub s_on: f32,
    /// Thermal pair conductance `κ`, per-species specific heats (`C = mass·cp`), ambient heat-loss
    /// rate, ambient temperature, and the initial brew-water (pour) temperature. Normalized scale.
    pub kappa: f32,
    pub cp_water: f32,
    pub cp_grain: f32,
    pub h_amb: f32,
    pub t_amb: f32,
    pub pour_t: f32,

    // --- fines migration (Phase 6) ---
    /// Fraction of a grain's volume that is detachable fines (the seeded per-grain inventory =
    /// `fines_fraction · grain_volume`). 0 = no fines (default; the feature is off until a brew
    /// scene sets this **and** `Config::fines_rate > 0`). See [`fines::fines_seed`].
    pub fines_fraction: f32,
    /// Critical Darcy flux (reduced sim units) where erosion and deposition balance — the
    /// zero-crossing of [`fines::net_rate`]. Above it fines scour loose; below it they settle.
    pub fines_crit_flux: f32,
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
            coupling_radius: 0.0,
            porosity: 0.40,
            // Wetting: coffee retains ~1.5× its dry mass; grain density ~1.3× water; cohesion peaks
            // near 40% saturation. c_max starts at 0 (no wet cohesion until calibrated).
            r_max: 1.5,
            rho_ratio: 1.3,
            s_peak: 0.4,
            c_max: 0.0,
            // Extraction/thermal (Phase 1.5) — KEEP.md §1 kinetics + a normalized thermal scale.
            // These are reference values; calibration to the yield/TDS band is U7. Extraction is
            // OFF until a scene sets Config.extract_rate > 0, so these defaults don't perturb suites.
            k0_fast: 0.18,
            k0_slow: 0.018,
            ea_over_r: 6.0,
            t_ref: 1.0,
            c_sat: 0.08,
            fast_fraction: 0.30,
            soluble_fraction: 0.28,
            d_ref: 1.0,
            u_half: 1.0,
            s_on: 0.1,
            kappa: 2.0,
            cp_water: 1.0,
            cp_grain: 1.0,
            h_amb: 0.02,
            t_amb: 0.85,
            pour_t: 1.0,
            // Fines OFF by default (like absorb_rate/extract_rate): a brew scene opts in by setting
            // fines_fraction > 0 alongside Config.fines_rate > 0.
            fines_fraction: 0.0,
            fines_crit_flux: fines::CRIT_FLUX_DEFAULT,
        }
    }
}

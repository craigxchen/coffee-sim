use super::brew_config::DEFAULT_BREW;

pub(crate) const MASS_UNITS_PER_ML: f32 = DEFAULT_BREW.water_mass_units_per_ml;
pub(crate) const PARTICLES_PER_ML: f32 = DEFAULT_BREW.water_particles_per_ml;

pub(crate) fn particle_mass_units() -> f32 {
    MASS_UNITS_PER_ML / PARTICLES_PER_ML
}

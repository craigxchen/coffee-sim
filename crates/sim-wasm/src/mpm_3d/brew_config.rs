/// Compile-time brew recipe and coarse-graining parameters for the V60 scene.
///
/// These values are intentionally grouped here before exposing them in UI. The
/// MPM particles are material samples, not literal grains or droplets, so this
/// config records both physical recipe parameters and the current sampling
/// choices used to map that recipe into the solver.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct BrewConfig {
    pub coffee_dose_g: f32,
    pub brew_water_ml: f32,
    pub grind_diameter_um: f32,
    pub bed_porosity: f32,
    pub bed_particle_samples: u32,
    pub water_particles_per_ml: f32,
    pub water_mass_units_per_ml: f32,
    pub water_sample_radius_dx: f32,
    pub bed_sample_radius_dx: f32,
    pub max_flow_rate_ml_s: f32,
    pub gentle_pour_exit_speed_m_s: f32,
    pub initial_kettle_angle_deg: f32,
    pub water_viscosity: f32,
    pub initial_bed_permeability: f32,
    pub bed_drag_coeff: f32,
    pub bed_absorption_rate: f32,
    pub target_bed_retention_ml: f32,
}

pub(crate) const DEFAULT_BREW: BrewConfig = BrewConfig {
    // A modest single-cup V60 recipe. The water dose is not yet used as an
    // automatic pour stop; for now it documents the intended physical scale.
    coffee_dose_g: 15.0,
    brew_water_ml: 250.0,
    // Medium-fine pourover grind. The current solver still uses the
    // permeability scalar below directly; this diameter is the planned input
    // for the Darcy / Kozeny-Carman permeability mapping.
    grind_diameter_um: 650.0,
    bed_porosity: 0.40,
    bed_particle_samples: 12_000,
    water_particles_per_ml: 320.0,
    water_mass_units_per_ml: 80.0,
    water_sample_radius_dx: 0.18,
    bed_sample_radius_dx: 0.62,
    max_flow_rate_ml_s: 4.0,
    gentle_pour_exit_speed_m_s: 0.12,
    initial_kettle_angle_deg: 9.0,
    water_viscosity: 1.2,
    initial_bed_permeability: 0.32,
    bed_drag_coeff: 90.0,
    bed_absorption_rate: 1.6,
    target_bed_retention_ml: 42.0,
};

#[allow(dead_code)]
impl BrewConfig {
    pub(crate) const fn water_particle_mass_units(self) -> f32 {
        self.water_mass_units_per_ml / self.water_particles_per_ml
    }

    pub(crate) const fn bed_sample_mass_g(self) -> f32 {
        self.coffee_dose_g / self.bed_particle_samples as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_brew_parameters_are_plausible_for_v60() {
        assert!((10.0..=25.0).contains(&DEFAULT_BREW.coffee_dose_g));
        assert!((150.0..=400.0).contains(&DEFAULT_BREW.brew_water_ml));
        assert!((300.0..=1_200.0).contains(&DEFAULT_BREW.grind_diameter_um));
        assert!((0.30..=0.55).contains(&DEFAULT_BREW.bed_porosity));
        assert!(DEFAULT_BREW.bed_particle_samples > 1_000);
        assert!(DEFAULT_BREW.bed_sample_radius_dx > DEFAULT_BREW.water_sample_radius_dx);
    }

    #[test]
    fn default_particle_sampling_has_positive_masses() {
        assert!(DEFAULT_BREW.water_particle_mass_units() > 0.0);
        assert!(DEFAULT_BREW.bed_sample_mass_g() > 0.0);
    }
}

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
    pub high_pour_exit_speed_m_s: f32,
    pub initial_kettle_angle_deg: f32,
    pub water_viscosity: f32,
    pub water_kinematic_viscosity_m2_s: f32,
    pub min_bed_permeability_m2: f32,
    pub bed_absorption_rate: f32,
    pub target_bed_retention_ml: f32,
    pub bed_pore_capacity_scale: f32,
    pub bed_pore_overfill_alpha: f32,
    pub bed_surface_void_scale: f32,
    pub filter_pore_capacity_scale: f32,
    pub dripper_outlet_radius: f32,
    pub filter_absorption_rate_s: f32,
    pub bed_compaction_rate: f32,
    pub bed_impact_rate: f32,
}

pub(crate) const DEFAULT_BREW: BrewConfig = BrewConfig {
    // A modest single-cup V60 recipe. The water dose is not yet used as an
    // automatic pour stop; for now it documents the intended physical scale.
    coffee_dose_g: 15.0,
    brew_water_ml: 250.0,
    // Medium-fine pourover grind. This feeds the bed permeability through a
    // Kozeny-Carman estimate, so finer grind lowers flow roughly with d^2.
    grind_diameter_um: 450.0,
    bed_porosity: 0.40,
    bed_particle_samples: 12_000,
    water_particles_per_ml: 320.0,
    water_mass_units_per_ml: 80.0,
    water_sample_radius_dx: 0.18,
    bed_sample_radius_dx: 0.62,
    max_flow_rate_ml_s: 12.0,
    gentle_pour_exit_speed_m_s: 0.12,
    high_pour_exit_speed_m_s: 0.45,
    initial_kettle_angle_deg: 9.0,
    water_viscosity: 1.2,
    water_kinematic_viscosity_m2_s: 1.0e-6,
    min_bed_permeability_m2: 1.0e-12,
    bed_absorption_rate: 1.6,
    target_bed_retention_ml: 42.0,
    bed_pore_capacity_scale: 1.0,
    bed_pore_overfill_alpha: 18.0,
    bed_surface_void_scale: 1.0,
    filter_pore_capacity_scale: 1.0,
    dripper_outlet_radius: 0.42,
    filter_absorption_rate_s: 1.2,
    bed_compaction_rate: 5.5,
    bed_impact_rate: 8.0,
};

#[allow(dead_code)]
impl BrewConfig {
    pub(crate) const fn water_particle_mass_units(self) -> f32 {
        self.water_mass_units_per_ml / self.water_particles_per_ml
    }

    pub(crate) const fn bed_sample_mass_g(self) -> f32 {
        self.coffee_dose_g / self.bed_particle_samples as f32
    }

    pub(crate) const fn bed_permeability_m2(self) -> f32 {
        kozeny_carman_permeability_m2(self.grind_diameter_um, self.bed_porosity)
    }

    pub(crate) const fn darcy_resistance_rate_s(self) -> f32 {
        self.water_kinematic_viscosity_m2_s / self.bed_permeability_m2()
    }
}

pub(crate) const fn kozeny_carman_permeability_m2(grind_diameter_um: f32, porosity: f32) -> f32 {
    let d_m = grind_diameter_um * 1.0e-6;
    let solid_fraction = 1.0 - porosity;
    d_m * d_m * porosity * porosity * porosity / (180.0 * solid_fraction * solid_fraction)
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
        assert!(DEFAULT_BREW.bed_permeability_m2() > DEFAULT_BREW.min_bed_permeability_m2);
        assert!(DEFAULT_BREW.darcy_resistance_rate_s() > 0.0);
    }

    #[test]
    fn default_particle_sampling_has_positive_masses() {
        assert!(DEFAULT_BREW.water_particle_mass_units() > 0.0);
        assert!(DEFAULT_BREW.bed_sample_mass_g() > 0.0);
    }

    #[test]
    fn kozeny_carman_permeability_tracks_grind_size_squared() {
        let fine = kozeny_carman_permeability_m2(400.0, DEFAULT_BREW.bed_porosity);
        let coarse = kozeny_carman_permeability_m2(800.0, DEFAULT_BREW.bed_porosity);

        assert!(fine > 0.0);
        assert!(coarse > fine);
        assert!((coarse / fine - 4.0).abs() < 0.01);
    }
}

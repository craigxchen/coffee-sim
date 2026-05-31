/// Recipe and coarse-graining defaults shared by simulation backends.
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
    pub initial_water_speed_m_s: f32,
    pub water_viscosity: f32,
    pub water_kinematic_viscosity_m2_s: f32,
    pub min_bed_permeability_m2: f32,
    pub bed_absorption_rate: f32,
    pub target_bed_retention_ml: f32,
    pub extractable_yield_fraction: f32,
    pub fast_extractable_fraction: f32,
    pub fast_extraction_rate_s: f32,
    pub slow_extraction_rate_s: f32,
    pub max_solute_concentration: f32,
    pub pore_to_water_mass_transfer_rate_s: f32,
}

pub(crate) const DEFAULT_BREW: BrewConfig = BrewConfig {
    coffee_dose_g: 15.0,
    brew_water_ml: 250.0,
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
    initial_water_speed_m_s: 0.12,
    water_viscosity: 1.2,
    water_kinematic_viscosity_m2_s: 1.0e-6,
    min_bed_permeability_m2: 1.0e-12,
    bed_absorption_rate: 1.6,
    target_bed_retention_ml: 42.0,
    extractable_yield_fraction: 0.28,
    fast_extractable_fraction: 0.30,
    fast_extraction_rate_s: 0.18,
    slow_extraction_rate_s: 0.018,
    max_solute_concentration: 0.08,
    pore_to_water_mass_transfer_rate_s: 4.0,
};

impl BrewConfig {
    pub(crate) const fn water_particle_mass_units(self) -> f32 {
        self.water_mass_units_per_ml / self.water_particles_per_ml
    }

    pub(crate) const fn bed_sample_mass_g(self) -> f32 {
        self.coffee_dose_g / self.bed_particle_samples as f32
    }

    pub(crate) const fn bed_sample_extractable_mass_units(self) -> f32 {
        self.bed_sample_mass_g() * self.extractable_yield_fraction * self.water_mass_units_per_ml
    }

    pub(crate) const fn bed_permeability_m2(self) -> f32 {
        super::coffee::kozeny_carman_permeability_m2(self.grind_diameter_um, self.bed_porosity)
    }
}

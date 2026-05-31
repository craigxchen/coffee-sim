use coffee_sim_core::Vec3;

use crate::boundaries::filter::FilterConfig;
use crate::materials::DEFAULT_BREW;

pub(crate) const REFERENCE_V60_FILTER_HEIGHT_M: f32 = 0.10;
pub(crate) const MODEL_FILTER_HEIGHT_UNITS: f32 = 2.75 - (-3.02);
pub(crate) const SIM_UNITS_PER_METER: f32 =
    MODEL_FILTER_HEIGHT_UNITS / REFERENCE_V60_FILTER_HEIGHT_M;
pub(crate) const METERS_PER_SIM_UNIT: f32 = 1.0 / SIM_UNITS_PER_METER;
pub(crate) const ML_PER_SIM_UNIT_CUBED: f32 =
    METERS_PER_SIM_UNIT * METERS_PER_SIM_UNIT * METERS_PER_SIM_UNIT * 1_000_000.0;
pub(crate) const STANDARD_GRAVITY_M_S2: f32 = 9.806_65;
pub(crate) const EARTH_GRAVITY_SIM_UNITS: f32 = -STANDARD_GRAVITY_M_S2 * SIM_UNITS_PER_METER;
pub(crate) const MAX_WATER_SPEED_M_S: f32 = 1.5;
pub(crate) const MAX_WATER_SPEED_SIM_UNITS: f32 = MAX_WATER_SPEED_M_S * SIM_UNITS_PER_METER;

pub(crate) fn sim_speed_to_meters_per_second(speed: f32) -> f32 {
    speed * METERS_PER_SIM_UNIT
}

pub(crate) fn sim_speed_from_meters_per_second(speed: f32) -> f32 {
    speed * SIM_UNITS_PER_METER
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SpoutSettings {
    pub origin: Vec3,
    pub direction: Vec3,
    pub nozzle_radius: f32,
    pub stem_radius: f32,
    pub discharge_coeff: f32,
    pub volume_to_ml: f32,
    pub max_flow_rate_ml_s: f32,
    pub max_exit_speed: f32,
    pub stem_length: f32,
}

impl Default for SpoutSettings {
    fn default() -> Self {
        Self {
            origin: Vec3::new(0.0, 7.3, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
            nozzle_radius: 0.18,
            stem_radius: 0.24,
            discharge_coeff: 0.92,
            volume_to_ml: ML_PER_SIM_UNIT_CUBED,
            max_flow_rate_ml_s: DEFAULT_BREW.max_flow_rate_ml_s,
            max_exit_speed: sim_speed_from_meters_per_second(DEFAULT_BREW.high_pour_exit_speed_m_s),
            stem_length: 1.9,
        }
    }
}

impl SpoutSettings {
    pub(crate) fn translate_origin_to(&mut self, origin: Vec3) {
        self.origin = origin;
        self.direction = Vec3::new(0.0, -1.0, 0.0);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BedSpec {
    pub enabled: bool,
    pub center: Vec3,
    pub top_y: f32,
    pub bot_y: f32,
    pub top_radius: f32,
    pub bot_radius: f32,
    pub num_particles: u32,
    pub initial_porosity: f32,
    pub grind_diameter_um: f32,
    pub initially_saturated: bool,
}

impl BedSpec {
    pub(crate) fn seated_in_filter(filter: &FilterConfig) -> Self {
        Self {
            enabled: true,
            center: filter.center,
            top_y: 0.45,
            bot_y: -2.70,
            top_radius: 2.65,
            bot_radius: 0.18,
            num_particles: DEFAULT_BREW.bed_particle_samples,
            initial_porosity: DEFAULT_BREW.bed_porosity,
            grind_diameter_um: DEFAULT_BREW.grind_diameter_um,
            initially_saturated: false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SimSettings {
    pub bounds_size: Vec3,
    pub max_particles: u32,
    pub substeps: u32,
    pub xpbd_iterations: u32,
    pub xpbd_max_iterations: u32,
    pub density_residual_target: f32,
    pub gravity: f32,
    pub render_radius: f32,
    pub spout: SpoutSettings,
    pub initial_water_speed_m_s: f32,
    pub filter: Option<FilterConfig>,
    pub bed: Option<BedSpec>,
}

impl SimSettings {
    pub(crate) fn default_v60() -> Self {
        let filter = FilterConfig::default();
        let dx = 14.0 / 80.0;
        Self {
            bounds_size: Vec3::new(14.0, 20.0, 14.0),
            max_particles: 220_000,
            substeps: 4,
            xpbd_iterations: 6,
            xpbd_max_iterations: 10,
            density_residual_target: 0.0,
            gravity: EARTH_GRAVITY_SIM_UNITS,
            render_radius: dx * 0.7,
            spout: SpoutSettings::default(),
            initial_water_speed_m_s: DEFAULT_BREW.initial_water_speed_m_s,
            filter: Some(filter.clone()),
            bed: Some(BedSpec::seated_in_filter(&filter)),
        }
    }
}

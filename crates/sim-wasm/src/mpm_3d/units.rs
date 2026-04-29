/// Unit map for the V60 MPM scene.
///
/// The solver uses abstract scene units for positions and velocities, but time
/// is seconds. Calibrate the length scale from the current filter height: a
/// typical V60 paper is roughly 10 cm tall, and the modeled filter spans 5.77
/// scene units from apex to rim. That keeps the existing volume scale
/// (`~5.2 mL / u^3`) close to the value the inflow had been using by hand.
pub(crate) const REFERENCE_V60_FILTER_HEIGHT_M: f32 = 0.10;
#[allow(dead_code)]
pub(crate) const REFERENCE_V60_FILTER_DIAMETER_M: f32 = 0.10;
pub(crate) const MODEL_FILTER_HEIGHT_UNITS: f32 = 2.75 - (-3.02);
#[allow(dead_code)]
pub(crate) const MODEL_FILTER_TOP_DIAMETER_UNITS: f32 = 2.0 * 4.10;

pub(crate) const SIM_UNITS_PER_METER: f32 =
    MODEL_FILTER_HEIGHT_UNITS / REFERENCE_V60_FILTER_HEIGHT_M;
pub(crate) const METERS_PER_SIM_UNIT: f32 = 1.0 / SIM_UNITS_PER_METER;
pub(crate) const ML_PER_SIM_UNIT_CUBED: f32 =
    METERS_PER_SIM_UNIT * METERS_PER_SIM_UNIT * METERS_PER_SIM_UNIT * 1_000_000.0;

pub(crate) const STANDARD_GRAVITY_M_S2: f32 = 9.806_65;
pub(crate) const EARTH_GRAVITY_SIM_UNITS: f32 = -STANDARD_GRAVITY_M_S2 * SIM_UNITS_PER_METER;

pub(crate) const MAX_WATER_SPEED_M_S: f32 = 1.5;
pub(crate) const MAX_WATER_SPEED_SIM_UNITS: f32 = MAX_WATER_SPEED_M_S * SIM_UNITS_PER_METER;

pub(crate) const GENTLE_POUR_EXIT_SPEED_M_S: f32 = 0.12;
pub(crate) const GENTLE_POUR_EXIT_SPEED_SIM_UNITS: f32 =
    GENTLE_POUR_EXIT_SPEED_M_S * SIM_UNITS_PER_METER;

pub(crate) fn sim_speed_to_meters_per_second(speed: f32) -> f32 {
    speed * METERS_PER_SIM_UNIT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v60_height_sets_scene_length_scale() {
        let modeled_height_m = MODEL_FILTER_HEIGHT_UNITS * METERS_PER_SIM_UNIT;
        assert!((modeled_height_m - REFERENCE_V60_FILTER_HEIGHT_M).abs() < 1e-6);
    }

    #[test]
    fn volume_scale_matches_length_scale() {
        assert!((ML_PER_SIM_UNIT_CUBED - 5.20).abs() < 0.05);
    }

    #[test]
    fn gentle_pour_speed_matches_kettle_scale() {
        assert!((GENTLE_POUR_EXIT_SPEED_M_S - 0.12).abs() < 1e-6);
        assert!((GENTLE_POUR_EXIT_SPEED_SIM_UNITS * METERS_PER_SIM_UNIT - 0.12).abs() < 1e-6);
    }

    #[test]
    fn current_filter_opening_is_larger_than_reference_paper() {
        let modeled_diameter_m = MODEL_FILTER_TOP_DIAMETER_UNITS * METERS_PER_SIM_UNIT;
        assert!(modeled_diameter_m > REFERENCE_V60_FILTER_DIAMETER_M);
    }
}

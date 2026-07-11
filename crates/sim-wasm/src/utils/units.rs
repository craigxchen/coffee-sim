//! Shared conversion between SI units and the simulation's spatial units.
//!
//! Time remains physical seconds. The reference spatial scale is derived from volume so length,
//! velocity, gravity, volume, and permeability cannot silently use contradictory calibrations.

use std::fmt;

pub const ML_PER_CUBIC_METER: f64 = 1_000_000.0;
pub const REFERENCE_ML_PER_SIM_UNIT_CUBED: f64 = 5.20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiUnitScale {
    ml_per_sim_unit_cubed: f64,
    sim_units_per_meter: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitScaleError {
    NonFiniteVolumeScale,
    NonPositiveVolumeScale,
}

impl fmt::Display for UnitScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteVolumeScale => f.write_str("volume scale must be finite"),
            Self::NonPositiveVolumeScale => f.write_str("volume scale must be greater than zero"),
        }
    }
}

impl std::error::Error for UnitScaleError {}

impl SiUnitScale {
    pub fn from_ml_per_sim_unit_cubed(value: f64) -> Result<Self, UnitScaleError> {
        if !value.is_finite() {
            return Err(UnitScaleError::NonFiniteVolumeScale);
        }
        if value <= 0.0 {
            return Err(UnitScaleError::NonPositiveVolumeScale);
        }
        let sim_units_per_meter = (ML_PER_CUBIC_METER / value).cbrt();
        Ok(Self {
            ml_per_sim_unit_cubed: value,
            sim_units_per_meter,
        })
    }

    pub fn reference() -> Self {
        Self::from_ml_per_sim_unit_cubed(REFERENCE_ML_PER_SIM_UNIT_CUBED)
            .expect("the compile-time reference volume scale is positive and finite")
    }

    pub fn ml_per_sim_unit_cubed(self) -> f64 {
        self.ml_per_sim_unit_cubed
    }

    pub fn sim_units_per_meter(self) -> f64 {
        self.sim_units_per_meter
    }

    pub fn meters_per_sim_unit(self) -> f64 {
        self.sim_units_per_meter.recip()
    }

    pub fn sim_length_from_meters(self, meters: f64) -> f64 {
        meters * self.sim_units_per_meter
    }

    pub fn meters_from_sim_length(self, sim_units: f64) -> f64 {
        sim_units * self.meters_per_sim_unit()
    }

    pub fn sim_speed_from_meters_per_second(self, meters_per_second: f64) -> f64 {
        self.sim_length_from_meters(meters_per_second)
    }

    pub fn meters_per_second_from_sim_speed(self, sim_units_per_second: f64) -> f64 {
        self.meters_from_sim_length(sim_units_per_second)
    }

    pub fn sim_acceleration_from_meters_per_second_squared(self, value: f64) -> f64 {
        self.sim_length_from_meters(value)
    }

    pub fn sim_volume_from_ml(self, ml: f64) -> f64 {
        ml / self.ml_per_sim_unit_cubed
    }

    pub fn ml_from_sim_volume(self, sim_units_cubed: f64) -> f64 {
        sim_units_cubed * self.ml_per_sim_unit_cubed
    }

    pub fn sim_permeability_from_m2(self, square_meters: f64) -> f64 {
        square_meters * self.sim_units_per_meter * self.sim_units_per_meter
    }

    pub fn m2_from_sim_permeability(self, sim_units_squared: f64) -> f64 {
        sim_units_squared / (self.sim_units_per_meter * self.sim_units_per_meter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1.0e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn reference_length_scale_is_derived_from_volume() {
        let scale = SiUnitScale::reference();
        assert!((57.0..58.0).contains(&scale.sim_units_per_meter()));
        let reconstructed_ml = ML_PER_CUBIC_METER / scale.sim_units_per_meter().powi(3);
        assert!(close(reconstructed_ml, scale.ml_per_sim_unit_cubed()));
    }

    #[test]
    fn invalid_volume_scales_are_rejected() {
        for value in [0.0, -1.0] {
            assert_eq!(
                SiUnitScale::from_ml_per_sim_unit_cubed(value),
                Err(UnitScaleError::NonPositiveVolumeScale)
            );
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                SiUnitScale::from_ml_per_sim_unit_cubed(value),
                Err(UnitScaleError::NonFiniteVolumeScale)
            );
        }
    }

    #[test]
    fn conversions_round_trip_and_preserve_order() {
        let scale = SiUnitScale::reference();
        for meters in [0.001, 0.1, 2.0] {
            assert!(close(
                scale.meters_from_sim_length(scale.sim_length_from_meters(meters)),
                meters
            ));
        }
        for ml in [1.0, 42.0, 250.0] {
            assert!(close(
                scale.ml_from_sim_volume(scale.sim_volume_from_ml(ml)),
                ml
            ));
        }
        for permeability in [1.0e-14, 1.0e-11, 1.0e-9] {
            assert!(close(
                scale.m2_from_sim_permeability(scale.sim_permeability_from_m2(permeability),),
                permeability
            ));
        }
        assert!(scale.sim_length_from_meters(2.0) > scale.sim_length_from_meters(1.0));
        assert!(scale.sim_volume_from_ml(250.0) > scale.sim_volume_from_ml(125.0));
    }
}

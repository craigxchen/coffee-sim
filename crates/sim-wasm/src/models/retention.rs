//! Capillary-retention and porous-mobility reference equations shared by solvers.
//!
//! All volumes passed to this module must use the same unit. The helpers deliberately contain no
//! coffee-specific targets or curve constants: recipe calibration supplies the inventories and
//! constitutive shape parameter, while these functions enforce bounds and bookkeeping.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetentionError {
    NonFinite { parameter: &'static str },
    Negative { parameter: &'static str },
    NonPositive { parameter: &'static str },
}

impl fmt::Display for RetentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { parameter } => write!(f, "{parameter} must be finite"),
            Self::Negative { parameter } => write!(f, "{parameter} must not be negative"),
            Self::NonPositive { parameter } => {
                write!(f, "{parameter} must be greater than zero")
            }
        }
    }
}

impl std::error::Error for RetentionError {}

/// Whether the requested capillary inventory fits in the available pore volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetAttainment {
    Attainable,
    Unmet { volume: f32 },
}

/// Bounded residual pore saturation and the volume it can actually retain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualSaturationResult {
    pub residual_saturation: f32,
    pub retained_volume: f32,
    pub attainment: TargetAttainment,
}

/// Capillary and mobile water as fractions of the local pore volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoreWaterFractions {
    pub capillary: f32,
    pub mobile: f32,
}

/// Capillary volume required after bound-water and filter-paper inventories are accounted for.
///
/// When those inventories already meet or exceed the whole-bed target, the capillary target is
/// zero rather than a negative volume.
pub fn capillary_target_volume(
    whole_bed_retention_volume: f32,
    bound_water_volume: f32,
    filter_water_volume: f32,
) -> Result<f32, RetentionError> {
    nonnegative(whole_bed_retention_volume, "whole_bed_retention_volume")?;
    nonnegative(bound_water_volume, "bound_water_volume")?;
    nonnegative(filter_water_volume, "filter_water_volume")?;
    Ok((whole_bed_retention_volume - bound_water_volume - filter_water_volume).max(0.0))
}

/// Convert a capillary target volume into a residual pore saturation.
///
/// A target larger than the available pore volume saturates the available pore space and reports
/// the exact unmet remainder. A positive target with zero pore volume is therefore explicitly
/// unattainable; the all-zero case is attainable and dry.
pub fn residual_pore_saturation(
    target_capillary_volume: f32,
    available_pore_volume: f32,
) -> Result<ResidualSaturationResult, RetentionError> {
    nonnegative(target_capillary_volume, "target_capillary_volume")?;
    nonnegative(available_pore_volume, "available_pore_volume")?;

    if target_capillary_volume == 0.0 {
        return Ok(ResidualSaturationResult {
            residual_saturation: 0.0,
            retained_volume: 0.0,
            attainment: TargetAttainment::Attainable,
        });
    }
    if available_pore_volume == 0.0 {
        return Ok(ResidualSaturationResult {
            residual_saturation: 0.0,
            retained_volume: 0.0,
            attainment: TargetAttainment::Unmet {
                volume: target_capillary_volume,
            },
        });
    }

    let retained_volume = target_capillary_volume.min(available_pore_volume);
    let residual_saturation = (retained_volume / available_pore_volume).clamp(0.0, 1.0);
    let unmet_volume = target_capillary_volume - retained_volume;
    let attainment = if unmet_volume > 0.0 {
        TargetAttainment::Unmet {
            volume: unmet_volume,
        }
    } else {
        TargetAttainment::Attainable
    };
    Ok(ResidualSaturationResult {
        residual_saturation,
        retained_volume,
        attainment,
    })
}

/// Split a bounded local pore saturation into capillary and mobile pore-volume fractions.
///
/// Water up to the residual saturation is capillary-held; only the excess is mobile. Thus
/// `capillary + mobile` is exactly the bounded local saturation.
pub fn capillary_mobile_fractions(
    local_saturation: f32,
    residual_saturation: f32,
) -> Result<PoreWaterFractions, RetentionError> {
    finite(local_saturation, "local_saturation")?;
    finite(residual_saturation, "residual_saturation")?;
    let local = local_saturation.clamp(0.0, 1.0);
    let residual = residual_saturation.clamp(0.0, 1.0);
    let capillary = local.min(residual);
    Ok(PoreWaterFractions {
        capillary,
        mobile: local - capillary,
    })
}

/// Wetting-phase Corey relative permeability on effective mobile saturation.
///
/// `S_e = clamp((S - S_r) / (1 - S_r), 0, 1)` and `k_r = S_e^n`. The positive Corey exponent
/// `n` is supplied explicitly by the material calibration; this function does not embed a
/// coffee-tuned curve shape. A residual saturation of one leaves no mobile pore range and returns
/// zero permeability.
pub fn corey_relative_permeability(
    local_saturation: f32,
    residual_saturation: f32,
    shape_exponent: f32,
) -> Result<f32, RetentionError> {
    finite(local_saturation, "local_saturation")?;
    finite(residual_saturation, "residual_saturation")?;
    positive(shape_exponent, "shape_exponent")?;

    let local = local_saturation.clamp(0.0, 1.0);
    let residual = residual_saturation.clamp(0.0, 1.0);
    if residual >= 1.0 || local <= residual {
        return Ok(0.0);
    }
    if local >= 1.0 {
        return Ok(1.0);
    }
    let effective = ((local - residual) / (1.0 - residual)).clamp(0.0, 1.0);
    Ok(effective.powf(shape_exponent).clamp(0.0, 1.0))
}

/// Exact normal-velocity relaxation of pore water toward the porous skeleton.
///
/// Solves `d(v_f - v_s)/dt = -rate * (v_f - v_s)` over one time step. The exponential form is
/// independent of substep partitioning and cannot overshoot the skeleton velocity for a
/// non-negative rate and time step.
pub fn relax_porous_normal_velocity(
    fluid_normal_velocity: f32,
    skeleton_normal_velocity: f32,
    relaxation_rate_per_s: f32,
    dt_s: f32,
) -> Result<f32, RetentionError> {
    finite(fluid_normal_velocity, "fluid_normal_velocity")?;
    finite(skeleton_normal_velocity, "skeleton_normal_velocity")?;
    nonnegative(relaxation_rate_per_s, "relaxation_rate_per_s")?;
    nonnegative(dt_s, "dt_s")?;

    let decay = (-relaxation_rate_per_s * dt_s).exp();
    // This convex form is algebraically identical to `v_s + (v_f - v_s) * decay` but avoids an
    // overflowing intermediate when the two finite velocities have opposite extreme signs.
    let relaxed = fluid_normal_velocity * decay + skeleton_normal_velocity * (1.0 - decay);
    if relaxed.is_finite() {
        Ok(relaxed)
    } else {
        Err(RetentionError::NonFinite {
            parameter: "relaxed_normal_velocity",
        })
    }
}

fn finite(value: f32, parameter: &'static str) -> Result<(), RetentionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RetentionError::NonFinite { parameter })
    }
}

fn nonnegative(value: f32, parameter: &'static str) -> Result<(), RetentionError> {
    finite(value, parameter)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(RetentionError::Negative { parameter })
    }
}

fn positive(value: f32, parameter: &'static str) -> Result<(), RetentionError> {
    finite(value, parameter)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(RetentionError::NonPositive { parameter })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 2.0e-6 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn capillary_target_closes_the_inventory_ledger_and_scales_with_volume_units() {
        let whole = 13.0;
        let bound = 3.0;
        let filter = 2.0;
        let capillary = capillary_target_volume(whole, bound, filter).unwrap();
        assert!(close(capillary + bound + filter, whole));

        let unit_scale = 7.0;
        let scaled =
            capillary_target_volume(whole * unit_scale, bound * unit_scale, filter * unit_scale)
                .unwrap();
        assert!(close(scaled, capillary * unit_scale));
    }

    #[test]
    fn capillary_target_is_bounded_and_monotone_in_each_inventory() {
        let base = capillary_target_volume(10.0, 2.0, 1.0).unwrap();
        assert!(capillary_target_volume(12.0, 2.0, 1.0).unwrap() > base);
        assert!(capillary_target_volume(10.0, 3.0, 1.0).unwrap() < base);
        assert!(capillary_target_volume(10.0, 2.0, 2.0).unwrap() < base);
        assert_eq!(capillary_target_volume(1.0, 2.0, 3.0).unwrap(), 0.0);
    }

    #[test]
    fn residual_saturation_reports_attainable_and_unmet_targets_conservatively() {
        let attainable = residual_pore_saturation(3.0, 12.0).unwrap();
        assert!(close(attainable.residual_saturation, 0.25));
        assert!(close(attainable.retained_volume, 3.0));
        assert_eq!(attainable.attainment, TargetAttainment::Attainable);

        let unmet = residual_pore_saturation(15.0, 12.0).unwrap();
        assert_eq!(unmet.residual_saturation, 1.0);
        assert!(close(unmet.retained_volume, 12.0));
        let TargetAttainment::Unmet { volume } = unmet.attainment else {
            panic!("target above pore capacity must be reported as unmet");
        };
        assert!(close(volume, 3.0));
        assert!(close(unmet.retained_volume + volume, 15.0));
    }

    #[test]
    fn residual_saturation_is_bounded_and_monotone() {
        let low = residual_pore_saturation(2.0, 10.0)
            .unwrap()
            .residual_saturation;
        let high = residual_pore_saturation(6.0, 10.0)
            .unwrap()
            .residual_saturation;
        let roomier = residual_pore_saturation(6.0, 20.0)
            .unwrap()
            .residual_saturation;
        assert!((0.0..=1.0).contains(&low));
        assert!(high > low);
        assert!(roomier < high);
    }

    #[test]
    fn capillary_and_mobile_fractions_partition_local_saturation() {
        for local in [-0.5, 0.0, 0.2, 0.7, 1.0, 1.5] {
            let fractions = capillary_mobile_fractions(local, 0.3).unwrap();
            let bounded_local = local.clamp(0.0, 1.0);
            assert!((0.0..=1.0).contains(&fractions.capillary));
            assert!((0.0..=1.0).contains(&fractions.mobile));
            assert!(close(fractions.capillary + fractions.mobile, bounded_local));
        }
        assert_eq!(
            capillary_mobile_fractions(0.4, 0.0).unwrap(),
            PoreWaterFractions {
                capillary: 0.0,
                mobile: 0.4,
            }
        );
        assert_eq!(
            capillary_mobile_fractions(0.4, 1.0).unwrap(),
            PoreWaterFractions {
                capillary: 0.4,
                mobile: 0.0,
            }
        );
    }

    #[test]
    fn corey_curve_is_bounded_monotone_and_shape_parameterized() {
        let residual = 0.2;
        let exponent = 3.0;
        let saturations = [0.0, residual, 0.4, 0.7, 1.0];
        let mut previous = 0.0;
        for saturation in saturations {
            let value = corey_relative_permeability(saturation, residual, exponent).unwrap();
            assert!((0.0..=1.0).contains(&value));
            assert!(value >= previous);
            previous = value;
        }
        assert_eq!(previous, 1.0);

        let mid = 0.6;
        let shallow = corey_relative_permeability(mid, residual, 1.5).unwrap();
        let steep = corey_relative_permeability(mid, residual, 4.0).unwrap();
        assert!(steep < shallow);
    }

    #[test]
    fn exact_velocity_relaxation_composes_across_substeps_and_preserves_equilibrium() {
        let fluid = -2.0;
        let skeleton = 0.5;
        let rate = 3.0;
        let dt_a = 0.07;
        let dt_b = 0.11;
        let once = relax_porous_normal_velocity(fluid, skeleton, rate, dt_a + dt_b).unwrap();
        let first = relax_porous_normal_velocity(fluid, skeleton, rate, dt_a).unwrap();
        let split = relax_porous_normal_velocity(first, skeleton, rate, dt_b).unwrap();
        assert!(close(once, split));
        assert!(once > fluid && once < skeleton);

        let rescaled_units =
            relax_porous_normal_velocity(fluid, skeleton, rate * 5.0, (dt_a + dt_b) / 5.0).unwrap();
        assert!(close(once, rescaled_units));
        assert_eq!(
            relax_porous_normal_velocity(skeleton, skeleton, rate, 10.0).unwrap(),
            skeleton
        );
    }

    #[test]
    fn zero_and_degenerate_inputs_are_finite_and_explicit() {
        assert_eq!(capillary_target_volume(0.0, 0.0, 0.0).unwrap(), 0.0);

        let dry = residual_pore_saturation(0.0, 0.0).unwrap();
        assert_eq!(dry.residual_saturation, 0.0);
        assert_eq!(dry.attainment, TargetAttainment::Attainable);

        let blocked = residual_pore_saturation(4.0, 0.0).unwrap();
        assert_eq!(blocked.residual_saturation, 0.0);
        assert_eq!(blocked.attainment, TargetAttainment::Unmet { volume: 4.0 });
        assert_eq!(corey_relative_permeability(1.0, 1.0, 2.0).unwrap(), 0.0);
        assert_eq!(
            relax_porous_normal_velocity(2.0, -1.0, 0.0, 10.0).unwrap(),
            2.0
        );
        assert_eq!(
            relax_porous_normal_velocity(2.0, -1.0, 5.0, 0.0).unwrap(),
            2.0
        );
        assert!(
            relax_porous_normal_velocity(f32::MAX, -f32::MAX, f32::MAX, f32::MAX)
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn invalid_physical_parameters_are_rejected_instead_of_producing_nan() {
        assert!(matches!(
            capillary_target_volume(f32::NAN, 0.0, 0.0),
            Err(RetentionError::NonFinite { .. })
        ));
        assert!(matches!(
            residual_pore_saturation(-1.0, 1.0),
            Err(RetentionError::Negative { .. })
        ));
        assert!(matches!(
            corey_relative_permeability(0.5, 0.1, 0.0),
            Err(RetentionError::NonPositive { .. })
        ));
        assert!(matches!(
            relax_porous_normal_velocity(0.0, 0.0, 1.0, -0.1),
            Err(RetentionError::Negative { .. })
        ));
    }
}

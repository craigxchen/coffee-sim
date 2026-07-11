//! Filter-paper geometry and finite water-capacity bookkeeping shared by solvers.
//!
//! This module owns only paper area, pore-volume, and reservoir-capacity equations. Darcy
//! permeability conversion remains in [`crate::models::scaling`], while exact porous normal-
//! velocity relaxation remains in [`crate::models::retention`].

use std::f64::consts::PI;
use std::fmt;

use crate::models::scaling::GRAMS_PER_KILOGRAM;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilterError {
    NonFinite { parameter: &'static str },
    Negative { parameter: &'static str },
    NonPositive { parameter: &'static str },
    InvalidPorosity,
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { parameter } => write!(f, "{parameter} must be finite"),
            Self::Negative { parameter } => write!(f, "{parameter} must not be negative"),
            Self::NonPositive { parameter } => {
                write!(f, "{parameter} must be greater than zero")
            }
            Self::InvalidPorosity => f.write_str("paper_porosity must be strictly between 0 and 1"),
        }
    }
}

impl std::error::Error for FilterError {}

/// Effective saturated paper geometry derived from its liquid pore capacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaturatedPaperGeometry {
    pub effective_thickness_m: f64,
    pub pore_volume_m3: f64,
}

/// Finite reservoir result. Retained and overflow volumes always sum to the offered volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterSaturationResult {
    pub saturation: f64,
    pub retained_volume: f64,
    pub overflow_volume: f64,
}

/// Lateral paper area of a right conical frustum.
///
/// `A = π (r_top + r_bottom) sqrt(h² + (r_top - r_bottom)²)`. All lengths must
/// use the same unit; the returned area uses that unit squared.
pub fn conical_frustum_paper_area(
    top_radius: f64,
    bottom_radius: f64,
    axial_height: f64,
) -> Result<f64, FilterError> {
    nonnegative(top_radius, "top_radius")?;
    nonnegative(bottom_radius, "bottom_radius")?;
    nonnegative(axial_height, "axial_height")?;

    let slant_height = axial_height.hypot(top_radius - bottom_radius);
    let area = PI * (top_radius + bottom_radius) * slant_height;
    finite_result(area, "conical_frustum_paper_area")
}

/// Convert a water-mass capacity in grams to liquid volume in cubic metres.
///
/// Density is explicit so the helper does not silently assume a temperature-dependent water
/// density. The identity is `V = (mass_g / 1000) / density_kg_m3`.
pub fn water_mass_capacity_to_volume_m3(
    water_mass_capacity_g: f64,
    liquid_density_kg_m3: f64,
) -> Result<f64, FilterError> {
    nonnegative(water_mass_capacity_g, "water_mass_capacity_g")?;
    positive(liquid_density_kg_m3, "liquid_density_kg_m3")?;
    finite_result(
        water_mass_capacity_g / GRAMS_PER_KILOGRAM / liquid_density_kg_m3,
        "water_capacity_volume_m3",
    )
}

/// Infer effective saturated paper thickness from area, porosity, and liquid pore capacity.
///
/// At saturation, liquid capacity is the pore volume and
/// `V_pore = area * effective_thickness * porosity`. Zero capacity produces zero pore volume and
/// thickness, including for a degenerate zero-area surface. Positive capacity requires positive
/// area. Porosity is a physical volume fraction strictly between zero and one.
pub fn saturated_paper_geometry(
    paper_area_m2: f64,
    paper_porosity: f64,
    saturated_liquid_capacity_m3: f64,
) -> Result<SaturatedPaperGeometry, FilterError> {
    nonnegative(paper_area_m2, "paper_area_m2")?;
    valid_porosity(paper_porosity)?;
    nonnegative(saturated_liquid_capacity_m3, "saturated_liquid_capacity_m3")?;

    if saturated_liquid_capacity_m3 == 0.0 {
        return Ok(SaturatedPaperGeometry {
            effective_thickness_m: 0.0,
            pore_volume_m3: 0.0,
        });
    }
    positive(paper_area_m2, "paper_area_m2")?;
    let effective_thickness_m = saturated_liquid_capacity_m3 / (paper_area_m2 * paper_porosity);
    finite_result(effective_thickness_m, "effective_thickness_m")?;
    Ok(SaturatedPaperGeometry {
        effective_thickness_m,
        pore_volume_m3: saturated_liquid_capacity_m3,
    })
}

/// Fill a finite filter-paper reservoir without deleting excess water.
///
/// Both volumes must use the same unit. A zero-capacity reservoir retains nothing, reports all
/// offered water as overflow, and returns zero saturation because no storage phase exists.
pub fn finite_capacity_saturation(
    offered_water_volume: f64,
    capacity_volume: f64,
) -> Result<FilterSaturationResult, FilterError> {
    nonnegative(offered_water_volume, "offered_water_volume")?;
    nonnegative(capacity_volume, "capacity_volume")?;

    if capacity_volume == 0.0 {
        return Ok(FilterSaturationResult {
            saturation: 0.0,
            retained_volume: 0.0,
            overflow_volume: offered_water_volume,
        });
    }
    let retained_volume = offered_water_volume.min(capacity_volume);
    let overflow_volume = offered_water_volume - retained_volume;
    Ok(FilterSaturationResult {
        saturation: (retained_volume / capacity_volume).clamp(0.0, 1.0),
        retained_volume,
        overflow_volume,
    })
}

fn finite(value: f64, parameter: &'static str) -> Result<(), FilterError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FilterError::NonFinite { parameter })
    }
}

fn nonnegative(value: f64, parameter: &'static str) -> Result<(), FilterError> {
    finite(value, parameter)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(FilterError::Negative { parameter })
    }
}

fn positive(value: f64, parameter: &'static str) -> Result<(), FilterError> {
    finite(value, parameter)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(FilterError::NonPositive { parameter })
    }
}

fn valid_porosity(value: f64) -> Result<(), FilterError> {
    finite(value, "paper_porosity")?;
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(FilterError::InvalidPorosity)
    }
}

fn finite_result(value: f64, parameter: &'static str) -> Result<f64, FilterError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FilterError::NonFinite { parameter })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1.0e-11 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn frustum_area_reduces_to_cylinder_and_cone_identities() {
        let radius = 0.7;
        let height = 2.3;
        let cylinder = conical_frustum_paper_area(radius, radius, height).unwrap();
        assert!(close(cylinder, 2.0 * PI * radius * height));

        let cone = conical_frustum_paper_area(radius, 0.0, height).unwrap();
        assert!(close(
            cone,
            PI * radius * (height * height + radius * radius).sqrt()
        ));
        assert_eq!(
            conical_frustum_paper_area(radius, 0.2, height).unwrap(),
            conical_frustum_paper_area(0.2, radius, height).unwrap()
        );
    }

    #[test]
    fn frustum_area_scales_quadratically_with_length_units() {
        let area = conical_frustum_paper_area(0.8, 0.15, 1.9).unwrap();
        let scale = 6.0;
        let scaled = conical_frustum_paper_area(0.8 * scale, 0.15 * scale, 1.9 * scale).unwrap();
        assert!(close(scaled, area * scale * scale));
    }

    #[test]
    fn water_mass_capacity_conversion_preserves_mass_and_is_monotone() {
        let mass_g = 320.0;
        let density = 870.0;
        let volume = water_mass_capacity_to_volume_m3(mass_g, density).unwrap();
        assert!(close(volume * density, mass_g / GRAMS_PER_KILOGRAM));
        assert!(water_mass_capacity_to_volume_m3(mass_g * 2.0, density).unwrap() > volume);
        assert!(water_mass_capacity_to_volume_m3(mass_g, density * 2.0).unwrap() < volume);
    }

    #[test]
    fn saturated_geometry_closes_pore_volume_identity_and_is_monotone() {
        let area = 0.42;
        let porosity = 0.63;
        let capacity = 0.0024;
        let geometry = saturated_paper_geometry(area, porosity, capacity).unwrap();
        assert!(close(
            area * geometry.effective_thickness_m * porosity,
            geometry.pore_volume_m3
        ));
        assert!(close(geometry.pore_volume_m3, capacity));

        let larger_area = saturated_paper_geometry(area * 2.0, porosity, capacity).unwrap();
        assert!(larger_area.effective_thickness_m < geometry.effective_thickness_m);
        let larger_porosity = saturated_paper_geometry(area, porosity + 0.2, capacity).unwrap();
        assert!(larger_porosity.effective_thickness_m < geometry.effective_thickness_m);
        let larger_capacity = saturated_paper_geometry(area, porosity, capacity * 2.0).unwrap();
        assert!(larger_capacity.effective_thickness_m > geometry.effective_thickness_m);
    }

    #[test]
    fn finite_capacity_saturation_conserves_overflow_and_is_bounded() {
        let capacity = 8.0;
        let mut previous_saturation = 0.0;
        for offered in [0.0, 2.0, 8.0, 11.0, 20.0] {
            let result = finite_capacity_saturation(offered, capacity).unwrap();
            assert!((0.0..=1.0).contains(&result.saturation));
            assert!(result.saturation >= previous_saturation);
            assert!(result.retained_volume <= capacity);
            assert!(close(
                result.retained_volume + result.overflow_volume,
                offered
            ));
            previous_saturation = result.saturation;
        }

        let overflow = finite_capacity_saturation(11.0, capacity).unwrap();
        assert!(close(overflow.retained_volume, capacity));
        assert!(close(overflow.overflow_volume, 3.0));
        assert_eq!(overflow.saturation, 1.0);

        let smaller_capacity = finite_capacity_saturation(11.0, 4.0).unwrap();
        let larger_capacity = finite_capacity_saturation(11.0, 10.0).unwrap();
        assert!(larger_capacity.retained_volume > smaller_capacity.retained_volume);
        assert!(larger_capacity.overflow_volume < smaller_capacity.overflow_volume);
    }

    #[test]
    fn zero_capacity_and_zero_geometry_are_safe_and_conservative() {
        assert_eq!(water_mass_capacity_to_volume_m3(0.0, 900.0).unwrap(), 0.0);
        assert_eq!(
            saturated_paper_geometry(0.0, 0.5, 0.0).unwrap(),
            SaturatedPaperGeometry {
                effective_thickness_m: 0.0,
                pore_volume_m3: 0.0,
            }
        );
        assert_eq!(
            finite_capacity_saturation(4.0, 0.0).unwrap(),
            FilterSaturationResult {
                saturation: 0.0,
                retained_volume: 0.0,
                overflow_volume: 4.0,
            }
        );
    }

    #[test]
    fn invalid_inputs_are_rejected_instead_of_producing_nonfinite_outputs() {
        assert!(matches!(
            conical_frustum_paper_area(-1.0, 0.0, 1.0),
            Err(FilterError::Negative { .. })
        ));
        assert!(matches!(
            water_mass_capacity_to_volume_m3(1.0, 0.0),
            Err(FilterError::NonPositive { .. })
        ));
        for porosity in [0.0, -0.1, 1.0, 1.1] {
            assert_eq!(
                saturated_paper_geometry(1.0, porosity, 1.0),
                Err(FilterError::InvalidPorosity)
            );
        }
        assert!(matches!(
            saturated_paper_geometry(0.0, 0.5, 1.0),
            Err(FilterError::NonPositive { .. })
        ));
        assert!(matches!(
            finite_capacity_saturation(f64::NAN, 1.0),
            Err(FilterError::NonFinite { .. })
        ));
        assert!(matches!(
            finite_capacity_saturation(1.0, -1.0),
            Err(FilterError::Negative { .. })
        ));
    }
}

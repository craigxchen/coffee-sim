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

/// Physical paper geometry. Reservoir capacity is deliberately independent: cellulose sorption
/// and folds can retain water beyond the open pore volume, while zero storage capacity must not
/// remove the paper's hydraulic resistance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaperGeometry {
    pub physical_thickness_m: f64,
    pub pore_volume_m3: f64,
}

/// Finite reservoir result. Retained and overflow volumes always sum to the offered volume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterSaturationResult {
    pub saturation: f64,
    pub retained_volume: f64,
    pub overflow_volume: f64,
}

/// Exact response of the one-cell thin-sheet Darcy regularization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaperRelaxation {
    pub coverage: f64,
    pub alpha: f64,
    pub beta_per_s: f64,
    pub decay: f64,
    pub pressure_mobility: f64,
}

/// Reference equation mirrored by the WGSL filter-paper response.
pub fn thin_sheet_relaxation(
    signed_distance: f64,
    numerical_band_thickness: f64,
    physical_paper_thickness: f64,
    kinematic_viscosity: f64,
    permeability: f64,
    dt: f64,
) -> Result<PaperRelaxation, FilterError> {
    finite(signed_distance, "signed_distance")?;
    positive(numerical_band_thickness, "numerical_band_thickness")?;
    nonnegative(physical_paper_thickness, "physical_paper_thickness")?;
    positive(kinematic_viscosity, "kinematic_viscosity")?;
    positive(permeability, "permeability")?;
    nonnegative(dt, "dt")?;

    let coverage = (1.0 - signed_distance.abs() / numerical_band_thickness).clamp(0.0, 1.0);
    if coverage == 0.0 || physical_paper_thickness == 0.0 {
        return Ok(PaperRelaxation {
            coverage,
            alpha: 0.0,
            beta_per_s: 0.0,
            decay: 1.0,
            pressure_mobility: 1.0,
        });
    }
    let alpha = physical_paper_thickness / numerical_band_thickness * coverage;
    let beta_per_s = kinematic_viscosity / permeability * alpha;
    let x = beta_per_s * dt;
    let (decay, pressure_mobility) = if x < 1.0e-3 {
        (1.0 - x + 0.5 * x * x, 1.0 - 0.5 * x + x * x / 6.0)
    } else {
        let decay = (-x).exp();
        (decay, (1.0 - decay) / x)
    };
    for (value, operation) in [
        (alpha, "thin-sheet alpha"),
        (beta_per_s, "thin-sheet beta"),
        (decay, "thin-sheet decay"),
        (pressure_mobility, "thin-sheet pressure mobility"),
    ] {
        finite_result(value, operation)?;
    }
    Ok(PaperRelaxation {
        coverage,
        alpha,
        beta_per_s,
        decay: decay.clamp(0.0, 1.0),
        pressure_mobility: pressure_mobility.clamp(0.0, 1.0),
    })
}

/// Compose a symmetric hard-wall projector with the paper's allowed normal mobility.
///
/// `B = P - (1-sigma) n_a n_a^T`, with `n_a = normalize(P n)`. The helper is a CPU reference
/// for operator tests; the GPU evaluates the same rank-one update in `pressure.wgsl`.
pub fn compose_porous_projector(
    hard_projector: [[f64; 3]; 3],
    paper_normal: [f64; 3],
    pressure_mobility: f64,
) -> Result<[[f64; 3]; 3], FilterError> {
    for row in hard_projector {
        for value in row {
            finite(value, "hard_projector")?;
        }
    }
    for value in paper_normal {
        finite(value, "paper_normal")?;
    }
    finite(pressure_mobility, "pressure_mobility")?;
    let sigma = pressure_mobility.clamp(0.0, 1.0);
    let mut allowed = [0.0; 3];
    for i in 0..3 {
        allowed[i] = hard_projector[i][0] * paper_normal[0]
            + hard_projector[i][1] * paper_normal[1]
            + hard_projector[i][2] * paper_normal[2];
    }
    let len = allowed
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if len <= 1.0e-12 {
        return Ok(hard_projector);
    }
    for value in &mut allowed {
        *value /= len;
    }
    let mut result = hard_projector;
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] -= (1.0 - sigma) * allowed[i] * allowed[j];
        }
    }
    Ok(result)
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

/// Compute paper pore volume from independently calibrated physical geometry.
///
/// `V_pore = area * physical_thickness * porosity`. Capacity is intentionally absent: changing
/// the finite reservoir must not silently alter Darcy resistance or pore transit distance.
pub fn paper_geometry(
    paper_area_m2: f64,
    physical_thickness_m: f64,
    paper_porosity: f64,
) -> Result<PaperGeometry, FilterError> {
    nonnegative(paper_area_m2, "paper_area_m2")?;
    nonnegative(physical_thickness_m, "physical_thickness_m")?;
    valid_porosity(paper_porosity)?;
    let pore_volume_m3 = paper_area_m2 * physical_thickness_m * paper_porosity;
    finite_result(pore_volume_m3, "paper_pore_volume_m3")?;
    Ok(PaperGeometry {
        physical_thickness_m,
        pore_volume_m3,
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
    fn physical_geometry_closes_pore_volume_identity_and_is_monotone() {
        let area = 0.42;
        let thickness = 0.000_2;
        let porosity = 0.63;
        let geometry = paper_geometry(area, thickness, porosity).unwrap();
        assert!(close(
            area * geometry.physical_thickness_m * porosity,
            geometry.pore_volume_m3
        ));
        assert!(close(geometry.physical_thickness_m, thickness));

        let larger_area = paper_geometry(area * 2.0, thickness, porosity).unwrap();
        assert!(larger_area.pore_volume_m3 > geometry.pore_volume_m3);
        let larger_porosity = paper_geometry(area, thickness, porosity + 0.2).unwrap();
        assert!(larger_porosity.pore_volume_m3 > geometry.pore_volume_m3);
        let thicker = paper_geometry(area, thickness * 2.0, porosity).unwrap();
        assert!(thicker.pore_volume_m3 > geometry.pore_volume_m3);
    }

    #[test]
    fn thin_sheet_resistance_and_darcy_limits_are_grid_band_invariant() {
        let (thickness, nu, permeability, dt) = (0.000_2, 3.2e-7, 1.0e-13, 1.0 / 120.0);
        let (gravity, pressure_jump, density) = (9.806_65, 2_400.0, 970.0);
        let mut reference_gravity_velocity = None;
        let mut reference_pressure_velocity = None;
        for band in [0.000_5, 0.002, 0.02] {
            let response =
                thin_sheet_relaxation(0.0, band, thickness, nu, permeability, dt).unwrap();
            assert!(close(
                response.beta_per_s * band,
                nu * thickness / permeability
            ));

            let gravity_velocity = response.alpha * gravity / response.beta_per_s;
            let pressure_acceleration = pressure_jump / (density * band);
            let pressure_velocity = pressure_acceleration / response.beta_per_s;
            let expected_gravity = permeability * gravity / nu;
            let expected_pressure = permeability * pressure_jump / (density * nu * thickness);
            assert!(close(gravity_velocity, expected_gravity));
            assert!(close(pressure_velocity, expected_pressure));
            if let Some(reference) = reference_gravity_velocity {
                assert!(close(gravity_velocity, reference));
            }
            if let Some(reference) = reference_pressure_velocity {
                assert!(close(pressure_velocity, reference));
            }
            reference_gravity_velocity = Some(gravity_velocity);
            reference_pressure_velocity = Some(pressure_velocity);
        }
    }

    #[test]
    fn exact_paper_relaxation_is_bounded_and_composes_with_timestep_partitioning() {
        let args = (0.0, 0.003, 0.000_2, 4.0e-7, 2.0e-12);
        let whole = thin_sheet_relaxation(args.0, args.1, args.2, args.3, args.4, 0.02).unwrap();
        let half = thin_sheet_relaxation(args.0, args.1, args.2, args.3, args.4, 0.01).unwrap();
        assert!((0.0..=1.0).contains(&whole.decay));
        assert!((0.0..=1.0).contains(&whole.pressure_mobility));
        assert!(close(whole.decay, half.decay * half.decay));

        let outside = thin_sheet_relaxation(0.004, 0.003, args.2, args.3, args.4, 0.02).unwrap();
        assert_eq!(outside.coverage, 0.0);
        assert_eq!(outside.decay, 1.0);
        assert_eq!(outside.pressure_mobility, 1.0);
    }

    #[test]
    fn porous_projector_is_symmetric_psd_and_has_sigma_normal_eigenvalue() {
        let hard = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let sigma = 0.23;
        let paper_normal = [1.0, 2.0, 3.0];
        let tensor = compose_porous_projector(hard, paper_normal, sigma).unwrap();
        for (i, row) in tensor.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!(close(*value, tensor[j][i]));
            }
        }
        let allowed_len = 5.0_f64.sqrt();
        let allowed = [1.0 / allowed_len, 2.0 / allowed_len, 0.0];
        let tangent = [-allowed[1], allowed[0], 0.0];
        let apply = |vector: [f64; 3]| {
            [
                tensor[0][0] * vector[0] + tensor[0][1] * vector[1] + tensor[0][2] * vector[2],
                tensor[1][0] * vector[0] + tensor[1][1] * vector[1] + tensor[1][2] * vector[2],
                tensor[2][0] * vector[0] + tensor[2][1] * vector[1] + tensor[2][2] * vector[2],
            ]
        };
        let normal_result = apply(allowed);
        let tangent_result = apply(tangent);
        for axis in 0..3 {
            assert!(close(normal_result[axis], sigma * allowed[axis]));
            assert!(close(tangent_result[axis], tangent[axis]));
        }
        for vector in [[1.0, -2.0, 0.4], [-0.3, 0.8, 2.0], [0.0, 0.0, 1.0]] {
            let result = apply(vector);
            let energy = vector
                .iter()
                .zip(result)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f64>();
            assert!(energy >= -1.0e-12);
        }
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
            paper_geometry(0.0, 0.0, 0.5).unwrap(),
            PaperGeometry {
                physical_thickness_m: 0.0,
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
                paper_geometry(1.0, 0.001, porosity),
                Err(FilterError::InvalidPorosity)
            );
        }
        assert!(matches!(
            paper_geometry(1.0, -0.001, 0.5),
            Err(FilterError::Negative { .. })
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

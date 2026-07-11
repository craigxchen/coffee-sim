//! Physical-to-reduced scaling helpers shared by every solver.
//!
//! Numerical particles are material samples, not literal coffee grains or water droplets. This
//! module keeps that distinction explicit: spacing determines a water sample's physical volume,
//! recipe dose determines the total represented solid mass, and dimensional quantities are
//! converted using declared length, mass, and time scales.

use std::fmt;

use crate::utils::units::SiUnitScale;

pub const GRAMS_PER_KILOGRAM: f64 = 1_000.0;
pub const STANDARD_GRAVITY_M_S2: f64 = 9.806_65;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSamplePhysical {
    pub edge_length_m: f64,
    pub volume_m3: f64,
    pub volume_ml: f64,
    pub mass_kg: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolidSampleMass {
    pub physical_mass_kg: f64,
    pub reduced_mass: f64,
}

/// Base units used by a reduced solver.
///
/// If `L = length_m * L'`, `M = mass_kg * M'`, and `t = time_s * t'`, then one
/// reduced stress unit is `mass_kg / (length_m * time_s²)` pascals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducedUnitScales {
    length_m_per_unit: f64,
    mass_kg_per_unit: f64,
    time_s_per_unit: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingError {
    NonFinite { parameter: &'static str },
    NonPositive { parameter: &'static str },
    Negative { parameter: &'static str },
    InvalidPorosity { parameter: &'static str },
    ZeroSolidSamples,
    NonFiniteResult { operation: &'static str },
}

impl fmt::Display for ScalingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { parameter } => write!(f, "{parameter} must be finite"),
            Self::NonPositive { parameter } => {
                write!(f, "{parameter} must be greater than zero")
            }
            Self::Negative { parameter } => write!(f, "{parameter} must not be negative"),
            Self::InvalidPorosity { parameter } => {
                write!(f, "{parameter} must be strictly between zero and one")
            }
            Self::ZeroSolidSamples => f.write_str("seeded solid sample count must be nonzero"),
            Self::NonFiniteResult { operation } => {
                write!(f, "{operation} produced a non-finite result")
            }
        }
    }
}

impl std::error::Error for ScalingError {}

impl ReducedUnitScales {
    pub fn new(
        length_m_per_unit: f64,
        mass_kg_per_unit: f64,
        time_s_per_unit: f64,
    ) -> Result<Self, ScalingError> {
        positive(length_m_per_unit, "length_m_per_unit")?;
        positive(mass_kg_per_unit, "mass_kg_per_unit")?;
        positive(time_s_per_unit, "time_s_per_unit")?;
        Ok(Self {
            length_m_per_unit,
            mass_kg_per_unit,
            time_s_per_unit,
        })
    }

    pub fn from_spatial_scale(
        spatial: SiUnitScale,
        mass_kg_per_unit: f64,
        time_s_per_unit: f64,
    ) -> Result<Self, ScalingError> {
        Self::new(
            spatial.meters_per_sim_unit(),
            mass_kg_per_unit,
            time_s_per_unit,
        )
    }

    pub fn length_m_per_unit(self) -> f64 {
        self.length_m_per_unit
    }

    pub fn mass_kg_per_unit(self) -> f64 {
        self.mass_kg_per_unit
    }

    pub fn time_s_per_unit(self) -> f64 {
        self.time_s_per_unit
    }

    pub fn pressure_pa_per_reduced_stress(self) -> f64 {
        self.mass_kg_per_unit / (self.length_m_per_unit * self.time_s_per_unit.powi(2))
    }

    pub fn reduced_stress_from_pa(self, pressure_pa: f64) -> Result<f64, ScalingError> {
        finite(pressure_pa, "pressure_pa")?;
        finite_result(
            pressure_pa / self.pressure_pa_per_reduced_stress(),
            "SI pressure to reduced stress conversion",
        )
    }

    pub fn pa_from_reduced_stress(self, reduced_stress: f64) -> Result<f64, ScalingError> {
        finite(reduced_stress, "reduced_stress")?;
        finite_result(
            reduced_stress * self.pressure_pa_per_reduced_stress(),
            "reduced stress to SI pressure conversion",
        )
    }

    pub fn reduced_acceleration_from_m_s2(
        self,
        acceleration_m_s2: f64,
    ) -> Result<f64, ScalingError> {
        finite(acceleration_m_s2, "acceleration_m_s2")?;
        finite_result(
            acceleration_m_s2 * self.time_s_per_unit.powi(2) / self.length_m_per_unit,
            "SI acceleration to reduced acceleration conversion",
        )
    }

    pub fn reduced_rate_from_per_second(self, rate_per_s: f64) -> Result<f64, ScalingError> {
        nonnegative(rate_per_s, "rate_per_s")?;
        finite_result(
            rate_per_s * self.time_s_per_unit,
            "SI rate to reduced rate conversion",
        )
    }

    pub fn reduced_kinematic_viscosity_from_m2_s(
        self,
        viscosity_m2_s: f64,
    ) -> Result<f64, ScalingError> {
        nonnegative(viscosity_m2_s, "viscosity_m2_s")?;
        finite_result(
            viscosity_m2_s * self.time_s_per_unit / self.length_m_per_unit.powi(2),
            "SI kinematic viscosity to reduced viscosity conversion",
        )
    }
}

/// Material volume and mass represented by one water sample on a cubic rest lattice.
pub fn water_sample_from_spacing(
    particle_spacing_sim_units: f64,
    water_density_kg_m3: f64,
    spatial: SiUnitScale,
) -> Result<WaterSamplePhysical, ScalingError> {
    positive(particle_spacing_sim_units, "particle_spacing_sim_units")?;
    positive(water_density_kg_m3, "water_density_kg_m3")?;

    let edge_length_m = spatial.meters_from_sim_length(particle_spacing_sim_units);
    let volume_m3 = edge_length_m.powi(3);
    let volume_ml = volume_m3 * crate::utils::units::ML_PER_CUBIC_METER;
    let mass_kg = water_density_kg_m3 * volume_m3;
    for (value, operation) in [
        (edge_length_m, "water sample edge length"),
        (volume_m3, "water sample volume"),
        (volume_ml, "water sample milliliter volume"),
        (mass_kg, "water sample mass"),
    ] {
        positive_result(value, operation)?;
    }

    Ok(WaterSamplePhysical {
        edge_length_m,
        volume_m3,
        volume_ml,
        mass_kg,
    })
}

/// Physical and reduced mass carried by each coarse solid sample.
///
/// `mass_kg_per_reduced_unit` is normally the water-sample mass, making the returned reduced
/// mass the solid parcel's mass relative to one water parcel.
pub fn solid_sample_mass_from_recipe(
    coffee_dose_g: f64,
    seeded_solid_samples: u32,
    mass_kg_per_reduced_unit: f64,
) -> Result<SolidSampleMass, ScalingError> {
    nonnegative(coffee_dose_g, "coffee_dose_g")?;
    if seeded_solid_samples == 0 {
        return Err(ScalingError::ZeroSolidSamples);
    }
    positive(mass_kg_per_reduced_unit, "mass_kg_per_reduced_unit")?;

    let physical_mass_kg = coffee_dose_g / GRAMS_PER_KILOGRAM / f64::from(seeded_solid_samples);
    let reduced_mass = physical_mass_kg / mass_kg_per_reduced_unit;
    finite_result(physical_mass_kg, "solid sample physical mass")?;
    finite_result(reduced_mass, "solid sample reduced mass")?;
    Ok(SolidSampleMass {
        physical_mass_kg,
        reduced_mass,
    })
}

/// Kozeny-Carman porosity term relative to a declared reference porosity.
///
/// This changes permeability without changing the separately calibrated reference `K`:
/// `K(phi) = K_ref * relative_permeability(phi, phi_ref)`.
pub fn porosity_relative_permeability(
    porosity: f64,
    reference_porosity: f64,
) -> Result<f64, ScalingError> {
    valid_porosity(porosity, "porosity")?;
    valid_porosity(reference_porosity, "reference_porosity")?;
    let value = porosity_term(porosity) / porosity_term(reference_porosity);
    positive_result(value, "porosity-relative permeability")
}

/// Darcy relaxation rate `beta = nu / K_effective` in inverse seconds.
pub fn darcy_beta_per_s(
    kinematic_viscosity_m2_s: f64,
    reference_permeability_m2: f64,
    relative_permeability: f64,
) -> Result<f64, ScalingError> {
    positive(kinematic_viscosity_m2_s, "kinematic_viscosity_m2_s")?;
    positive(reference_permeability_m2, "reference_permeability_m2")?;
    positive(relative_permeability, "relative_permeability")?;
    positive_result(
        kinematic_viscosity_m2_s / (reference_permeability_m2 * relative_permeability),
        "Darcy relaxation rate",
    )
}

pub fn darcy_beta_for_porosity_per_s(
    kinematic_viscosity_m2_s: f64,
    reference_permeability_m2: f64,
    porosity: f64,
    reference_porosity: f64,
) -> Result<f64, ScalingError> {
    let relative = porosity_relative_permeability(porosity, reference_porosity)?;
    darcy_beta_per_s(
        kinematic_viscosity_m2_s,
        reference_permeability_m2,
        relative,
    )
}

/// Acceleration produced by a capillary pressure head across a wetting-front thickness.
///
/// Since `delta_p = rho * g * head`, dividing the pressure gradient by `rho` gives
/// `a = g * head / front_thickness`; density cancels exactly.
pub fn capillary_suction_acceleration_m_s2(
    pressure_head_m: f64,
    front_thickness_m: f64,
    gravity_m_s2: f64,
) -> Result<f64, ScalingError> {
    nonnegative(pressure_head_m, "pressure_head_m")?;
    positive(front_thickness_m, "front_thickness_m")?;
    positive(gravity_m_s2, "gravity_m_s2")?;
    finite_result(
        gravity_m_s2 * pressure_head_m / front_thickness_m,
        "capillary suction acceleration",
    )
}

fn porosity_term(porosity: f64) -> f64 {
    porosity.powi(3) / (1.0 - porosity).powi(2)
}

fn finite(value: f64, parameter: &'static str) -> Result<(), ScalingError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScalingError::NonFinite { parameter })
    }
}

fn positive(value: f64, parameter: &'static str) -> Result<(), ScalingError> {
    finite(value, parameter)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScalingError::NonPositive { parameter })
    }
}

fn nonnegative(value: f64, parameter: &'static str) -> Result<(), ScalingError> {
    finite(value, parameter)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ScalingError::Negative { parameter })
    }
}

fn valid_porosity(value: f64, parameter: &'static str) -> Result<(), ScalingError> {
    finite(value, parameter)?;
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(ScalingError::InvalidPorosity { parameter })
    }
}

fn finite_result(value: f64, operation: &'static str) -> Result<f64, ScalingError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ScalingError::NonFiniteResult { operation })
    }
}

fn positive_result(value: f64, operation: &'static str) -> Result<f64, ScalingError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(ScalingError::NonFiniteResult { operation })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1.0e-10 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn water_sample_volume_and_mass_follow_spacing_cubed() {
        let spatial = SiUnitScale::reference();
        let small = water_sample_from_spacing(0.5, 1_000.0, spatial).unwrap();
        let large = water_sample_from_spacing(1.0, 1_000.0, spatial).unwrap();
        assert!(close(large.volume_m3 / small.volume_m3, 8.0));
        assert!(close(large.mass_kg / small.mass_kg, 8.0));
        assert!(close(large.mass_kg, 1_000.0 * large.volume_m3));
        assert!(close(
            large.volume_ml,
            large.volume_m3 * crate::utils::units::ML_PER_CUBIC_METER
        ));
    }

    #[test]
    fn solid_sample_mass_reconstructs_recipe_dose_and_refines_monotonically() {
        let dose_g = 17.3;
        let mass_scale = water_sample_from_spacing(1.0, 1_000.0, SiUnitScale::reference())
            .unwrap()
            .mass_kg;
        let coarse = solid_sample_mass_from_recipe(dose_g, 1_000, mass_scale).unwrap();
        let fine = solid_sample_mass_from_recipe(dose_g, 2_000, mass_scale).unwrap();
        assert!(close(
            coarse.physical_mass_kg * 1_000.0,
            dose_g / GRAMS_PER_KILOGRAM
        ));
        assert!(close(coarse.physical_mass_kg / fine.physical_mass_kg, 2.0));
        assert!(close(coarse.reduced_mass / fine.reduced_mass, 2.0));
        assert!(close(
            coarse.reduced_mass * mass_scale,
            coarse.physical_mass_kg
        ));
    }

    #[test]
    fn porosity_relative_factor_is_normalized_and_darcy_rate_is_monotonic() {
        let reference = 0.4;
        assert!(close(
            porosity_relative_permeability(reference, reference).unwrap(),
            1.0
        ));
        let tight = porosity_relative_permeability(0.3, reference).unwrap();
        let open = porosity_relative_permeability(0.5, reference).unwrap();
        assert!(tight < 1.0);
        assert!(open > 1.0);

        let nu = 1.0e-6;
        let permeability = 1.0e-11;
        let tight_beta = darcy_beta_per_s(nu, permeability, tight).unwrap();
        let open_beta = darcy_beta_per_s(nu, permeability, open).unwrap();
        assert!(tight_beta > open_beta);
        assert!(
            darcy_beta_per_s(nu * 2.0, permeability, 1.0).unwrap()
                > darcy_beta_per_s(nu, permeability, 1.0).unwrap()
        );
        assert!(
            darcy_beta_per_s(nu, permeability * 2.0, 1.0).unwrap()
                < darcy_beta_per_s(nu, permeability, 1.0).unwrap()
        );
        assert!(close(
            darcy_beta_per_s(nu, permeability, open).unwrap() * permeability * open,
            nu
        ));
    }

    #[test]
    fn capillary_acceleration_grows_with_head_and_shrinks_with_front_thickness() {
        let base = capillary_suction_acceleration_m_s2(0.05, 0.02, STANDARD_GRAVITY_M_S2).unwrap();
        let more_head =
            capillary_suction_acceleration_m_s2(0.10, 0.02, STANDARD_GRAVITY_M_S2).unwrap();
        let wider_front =
            capillary_suction_acceleration_m_s2(0.05, 0.04, STANDARD_GRAVITY_M_S2).unwrap();
        assert!(more_head > base);
        assert!(wider_front < base);
        assert_eq!(
            capillary_suction_acceleration_m_s2(0.0, 0.02, STANDARD_GRAVITY_M_S2).unwrap(),
            0.0
        );
    }

    #[test]
    fn explicit_reduced_scales_round_trip_stress_and_obey_dimensions() {
        let scales = ReducedUnitScales::new(0.02, 0.003, 0.5).unwrap();
        let pressure = 1_234.5;
        let reduced = scales.reduced_stress_from_pa(pressure).unwrap();
        assert!(close(
            scales.pa_from_reduced_stress(reduced).unwrap(),
            pressure
        ));

        let heavier_mass_scale = ReducedUnitScales::new(0.02, 0.006, 0.5).unwrap();
        assert!(heavier_mass_scale.reduced_stress_from_pa(pressure).unwrap() < reduced);
        let longer_time_scale = ReducedUnitScales::new(0.02, 0.003, 1.0).unwrap();
        assert!(longer_time_scale.reduced_stress_from_pa(pressure).unwrap() > reduced);

        let acceleration = 3.0;
        assert!(
            longer_time_scale
                .reduced_acceleration_from_m_s2(acceleration)
                .unwrap()
                > scales.reduced_acceleration_from_m_s2(acceleration).unwrap()
        );
        assert!(close(
            scales.reduced_rate_from_per_second(2.0).unwrap(),
            2.0 * scales.time_s_per_unit()
        ));
    }

    #[test]
    fn guarded_apis_reject_invalid_inputs() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                water_sample_from_spacing(bad, 1_000.0, SiUnitScale::reference()),
                Err(ScalingError::NonFinite { .. })
            ));
            assert!(matches!(
                darcy_beta_per_s(bad, 1.0e-11, 1.0),
                Err(ScalingError::NonFinite { .. })
            ));
        }
        assert_eq!(
            solid_sample_mass_from_recipe(15.0, 0, 1.0),
            Err(ScalingError::ZeroSolidSamples)
        );
        for porosity in [-0.1, 0.0, 1.0, 1.1] {
            assert!(matches!(
                porosity_relative_permeability(porosity, 0.4),
                Err(ScalingError::InvalidPorosity { .. })
            ));
        }
        assert!(matches!(
            capillary_suction_acceleration_m_s2(0.1, 0.0, STANDARD_GRAVITY_M_S2),
            Err(ScalingError::NonPositive { .. })
        ));
        assert!(matches!(
            ReducedUnitScales::new(0.0, 1.0, 1.0),
            Err(ScalingError::NonPositive { .. })
        ));
    }
}

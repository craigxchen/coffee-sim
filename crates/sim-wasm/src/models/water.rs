//! Atmospheric-pressure liquid-water properties used at the SI/reduced-units boundary.
//!
//! The viscosity correlation is a compact 0–100 °C fit used only to choose the Darcy
//! relaxation rate. It reproduces the IAPWS/NIST atmospheric values closely enough for the
//! calibration slider; it is not a replacement for the full pressure-dependent IAPWS release.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaterPropertyError {
    NonFiniteTemperature,
    TemperatureOutsideLiquidRange,
}

impl fmt::Display for WaterPropertyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteTemperature => f.write_str("water temperature must be finite"),
            Self::TemperatureOutsideLiquidRange => {
                f.write_str("water property fit is valid from 0 through 100 degrees Celsius")
            }
        }
    }
}

impl std::error::Error for WaterPropertyError {}

fn validate_temperature(temperature_c: f64) -> Result<(), WaterPropertyError> {
    if !temperature_c.is_finite() {
        return Err(WaterPropertyError::NonFiniteTemperature);
    }
    if !(0.0..=100.0).contains(&temperature_c) {
        return Err(WaterPropertyError::TemperatureOutsideLiquidRange);
    }
    Ok(())
}

/// Dynamic viscosity in Pa·s for liquid water near atmospheric pressure.
///
/// `mu = 2.414e-5 * 10^(247.8 / (T_K - 140))` is the standard compact interpolation over the
/// liquid brewing range. Reference formulation/data: IAPWS R12-08 and NIST JPCRD water tables.
pub fn dynamic_viscosity_pa_s(temperature_c: f64) -> Result<f64, WaterPropertyError> {
    validate_temperature(temperature_c)?;
    let temperature_k = temperature_c + 273.15;
    Ok(2.414e-5 * 10.0_f64.powf(247.8 / (temperature_k - 140.0)))
}

/// Liquid-water density in kg/m³ near atmospheric pressure.
///
/// This Kell-style interpolation is smooth over 0–100 °C and keeps the viscosity-to-kinematic
/// conversion honest without introducing a full equation-of-state dependency.
pub fn density_kg_m3(temperature_c: f64) -> Result<f64, WaterPropertyError> {
    validate_temperature(temperature_c)?;
    let t = temperature_c;
    let correction = ((t + 288.9414) / (508_929.2 * (t + 68.129_63))) * (t - 3.9863).powi(2);
    Ok(1_000.0 * (1.0 - correction))
}

/// Kinematic viscosity `nu = mu/rho` in m²/s.
pub fn kinematic_viscosity_m2_s(temperature_c: f64) -> Result<f64, WaterPropertyError> {
    Ok(dynamic_viscosity_pa_s(temperature_c)? / density_kg_m3(temperature_c)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotter_brewing_water_is_less_viscous() {
        let cool = kinematic_viscosity_m2_s(20.0).unwrap();
        let brew = kinematic_viscosity_m2_s(93.0).unwrap();
        assert!(brew < cool);
        assert!(cool.is_finite() && brew.is_finite());
    }

    #[test]
    fn atmospheric_reference_values_are_in_physical_bands() {
        let mu_20 = dynamic_viscosity_pa_s(20.0).unwrap();
        let mu_90 = dynamic_viscosity_pa_s(90.0).unwrap();
        assert!((0.000_95..0.001_05).contains(&mu_20));
        assert!((0.000_29..0.000_34).contains(&mu_90));
        assert!((950.0..1_005.0).contains(&density_kg_m3(90.0).unwrap()));
    }

    #[test]
    fn invalid_temperatures_are_rejected() {
        for temperature in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                dynamic_viscosity_pa_s(temperature),
                Err(WaterPropertyError::NonFiniteTemperature)
            );
        }
        for temperature in [-0.1, 100.1] {
            assert_eq!(
                density_kg_m3(temperature),
                Err(WaterPropertyError::TemperatureOutsideLiquidRange)
            );
        }
    }
}

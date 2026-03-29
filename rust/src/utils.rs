//! Common math helpers: interpolation, physical constants, RNG utilities.

// Water properties at standard brewing conditions
pub const WATER_DENSITY: f64 = 971.8; // kg/m³ at 93°C
pub const WATER_VISCOSITY_93C: f64 = 0.000306; // Pa·s at 93°C
pub const WATER_THERMAL_DIFFUSIVITY: f64 = 1.67e-7; // m²/s
pub const GRAVITY: f64 = 9.81; // m/s²

// Coffee extraction parameters
pub const C_SAT_93C: f64 = 0.25; // g/mL saturation concentration at 93°C
pub const SOLUBLES_FAST_FRACTION: f64 = 0.21; // fraction of dry mass easily extractable
pub const SOLUBLES_SLOW_FRACTION: f64 = 0.09; // fraction of dry mass hard to extract
pub const K_FAST_REF: f64 = 5e-5; // m/s fast extraction rate at T_ref
pub const K_SLOW_REF: f64 = 2e-7; // m/s slow extraction rate at T_ref
pub const E_ACTIVATION: f64 = 65000.0; // J/mol activation energy
pub const DIFFUSIVITY_SOLUBLES: f64 = 5e-10; // m²/s effective diffusivity
pub const R_GAS: f64 = 8.314; // J/(mol·K)
pub const T_REF: f64 = 366.15; // K (93°C)

// CO₂ (bloom)
pub const CO2_CONTENT_FRESH: f64 = 0.01; // kg CO₂ / kg coffee (fresh roast)
pub const CO2_RELEASE_RATE: f64 = 0.05; // 1/s first-order release constant

// Bed properties
pub const BED_TORTUOSITY: f64 = 1.5;
pub const COFFEE_PARTICLE_DENSITY: f64 = 1100.0; // kg/m³
pub const COFFEE_SPECIFIC_HEAT: f64 = 1670.0; // J/(kg·K)

// Viscosity model constants (Arrhenius fit)
// μ(T) = A · exp(B / T)  where T in Kelvin
pub const VISCOSITY_A: f64 = 2.414e-5; // Pa·s
pub const VISCOSITY_B: f64 = 247.8; // K — used as μ = A * exp(B / (T - 140))

/// Water dynamic viscosity as a function of temperature (Kelvin).
/// Uses the standard fit: μ = A * 10^(B / (T - C)) with C = 140 K.
pub fn water_viscosity(t_kelvin: f64) -> f64 {
    VISCOSITY_A * 10.0_f64.powf(VISCOSITY_B / (t_kelvin - 140.0))
}

/// Arrhenius rate scaling factor relative to T_ref.
pub fn arrhenius_factor(t_kelvin: f64) -> f64 {
    (-E_ACTIVATION / R_GAS * (1.0 / t_kelvin - 1.0 / T_REF)).exp()
}

/// Saturation concentration of coffee solubles (g/mL), temperature-dependent.
/// Simple linear approximation around 93°C.
pub fn saturation_concentration(t_kelvin: f64) -> f64 {
    // Approximate: C_sat increases ~0.5% per degree above ref
    let dt = t_kelvin - T_REF;
    (C_SAT_93C * (1.0 + 0.005 * dt)).max(0.01)
}

/// Kozeny-Carman permeability: k = (ε³ · d_p²) / (180 · (1 - ε)²)
pub fn kozeny_carman(porosity: f64, d_p: f64) -> f64 {
    if porosity <= 0.0 || porosity >= 1.0 {
        return 0.0;
    }
    let eps3 = porosity * porosity * porosity;
    let one_minus_eps = 1.0 - porosity;
    (eps3 * d_p * d_p) / (180.0 * one_minus_eps * one_minus_eps)
}

/// Specific surface area for spherical particles: a_s = 6(1-ε) / d_p
pub fn specific_surface_area(porosity: f64, d_p: f64) -> f64 {
    if d_p <= 0.0 {
        return 0.0;
    }
    6.0 * (1.0 - porosity) / d_p
}

// Ergun equation constants
pub const ERGUN_A: f64 = 150.0; // viscous constant
pub const ERGUN_B: f64 = 1.75; // inertial constant

/// Linear interpolation
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Clamp value to range
pub fn clamp(val: f64, min: f64, max: f64) -> f64 {
    val.max(min).min(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kozeny_carman() {
        // For typical pourover: ε≈0.4, d_p=700μm=7e-4m
        let k = kozeny_carman(0.4, 7e-4);
        // Expected: (0.064 * 4.9e-7) / (180 * 0.36) = ~4.8e-10 m²
        assert!(k > 1e-11 && k < 1e-8, "k = {k}");
    }

    #[test]
    fn test_water_viscosity() {
        let mu = water_viscosity(T_REF);
        // Should be close to 0.000306 Pa·s at 93°C
        assert!((mu - WATER_VISCOSITY_93C).abs() < 1e-4, "mu = {mu}");
    }

    #[test]
    fn test_arrhenius_factor_at_ref() {
        let f = arrhenius_factor(T_REF);
        assert!((f - 1.0).abs() < 1e-10);
    }
}

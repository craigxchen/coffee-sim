// Water
pub const WATER_DENSITY: f64 = 971.8;          // kg/m³ at 93°C
pub const GRAVITY: f64 = 9.81;                 // m/s²
pub const VISC_A: f64 = 2.414e-5;              // Pa·s
pub const VISC_B: f64 = 247.8;                 // K
pub const VISC_C: f64 = 140.0;                 // K

// Thermodynamics
pub const R_GAS: f64 = 8.314;                  // J/(mol·K)
pub const T_REF: f64 = 366.15;                 // K (93°C)
pub const T_AMBIENT: f64 = 295.0;              // K (22°C)
pub const THERMAL_DIFFUSIVITY: f64 = 1.5e-7;   // m²/s
pub const H_AIR: f64 = 10.0;                   // W/(m²·K)
pub const H_WALL: f64 = 5.0;                   // W/(m²·K)
pub const WATER_CP: f64 = 4186.0;              // J/(kg·K)

// Extraction
pub const C_SAT_REF: f64 = 250.0;              // kg/m³ (= 0.25 g/mL) at T_REF
pub const SOLUBLES_FAST_FRAC: f64 = 0.21;
pub const SOLUBLES_SLOW_FRAC: f64 = 0.09;
pub const K_FAST_REF: f64 = 5e-5;              // m/s at T_REF
pub const K_SLOW_REF: f64 = 2e-7;              // m/s at T_REF
pub const E_ACTIVATION: f64 = 65_000.0;        // J/mol

// Transport
pub const DIFFUSIVITY_SOLUBLES: f64 = 5e-10;   // m²/s
pub const TORTUOSITY: f64 = 1.5;

// CO₂
pub const CO2_RELEASE_RATE: f64 = 0.05;        // 1/s
pub const CO2_MOLAR_MASS: f64 = 0.044;         // kg/mol

// Bed
pub const COFFEE_DENSITY: f64 = 1100.0;        // kg/m³ particle density
pub const BASE_POROSITY: f64 = 0.40;
pub const POROSITY_VARIATION: f64 = 0.05;
pub const SATURATION_THRESHOLD: f64 = 0.2;
pub const SATURATION_RESIDUAL: f64 = 0.15;     // irreducible saturation (Brooks-Corey)
pub const CAPILLARY_DIFFUSIVITY: f64 = 1e-6;   // m²/s (saturation spreading)

// Dripper wall thermal mass
pub const WALL_MASS: f64 = 0.3;                // kg (ceramic V60)
pub const WALL_CP: f64 = 800.0;                // J/(kg·K)
pub const WALL_H_CONTACT: f64 = 50.0;          // W/(m²·K)

/// Dynamic viscosity of water as function of temperature (Kelvin).
pub fn viscosity(t_k: f64) -> f64 {
    VISC_A * 10.0_f64.powf(VISC_B / (t_k - VISC_C))
}

/// Arrhenius rate scaling from reference temperature.
pub fn arrhenius(k_ref: f64, t_k: f64) -> f64 {
    k_ref * ((E_ACTIVATION / R_GAS) * (1.0 / T_REF - 1.0 / t_k)).exp()
}

/// Saturation concentration (kg/m³) as function of temperature.
pub fn c_saturation(t_k: f64) -> f64 {
    C_SAT_REF * (1.0 + 0.01 * (t_k - T_REF))
}

/// Brooks-Corey relative permeability.
pub fn relative_permeability(s: f64) -> f64 {
    if s <= SATURATION_RESIDUAL {
        0.0
    } else {
        let s_eff = (s - SATURATION_RESIDUAL) / (1.0 - SATURATION_RESIDUAL);
        s_eff * s_eff * s_eff
    }
}

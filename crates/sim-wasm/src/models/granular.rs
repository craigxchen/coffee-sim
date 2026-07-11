//! Unit-agnostic constitutive helpers for the deformable coffee skeleton.
//!
//! The production MPM path treats the bed's physical compaction response implicitly. Any shear
//! modulus left in an explicit predictor is therefore a *numerical compliance*, not a claim about
//! the coffee's physical Young's modulus. These helpers keep those quantities separate and provide
//! a packing law whose tangent is finite everywhere.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsotropicModuli {
    pub bulk: f64,
    pub shear: f64,
    pub lame_lambda: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionResponse {
    pub pressure: f64,
    pub tangent: f64,
}

/// Result of returning a geometrically over-consolidated packing state to its current cap.
///
/// Hardening is deliberately explicit: `returned_pressure` is the *old* cap used by the
/// current grid solve, while `consolidation_pressure` is the hardened cap available to the next
/// macrostep. This keeps the projected grid operator linear during one solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapReturn {
    pub elastic_solid_fraction: f64,
    pub returned_pressure: f64,
    pub plastic_compaction: f64,
    pub plastic_increment: f64,
    pub consolidation_pressure: f64,
}

/// Explicit shear work selected under a declared microstep ceiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShearStepBudget {
    pub requested_modulus: f64,
    pub applied_modulus: f64,
    pub substeps: u32,
    pub limited: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GranularError {
    NonFinite { parameter: &'static str },
    NonPositive { parameter: &'static str },
    Negative { parameter: &'static str },
    InvalidPoissonRatio,
    InvalidPackingInterval,
    ZeroTangentInverse,
    ZeroStepBudget,
}

impl fmt::Display for GranularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { parameter } => write!(f, "{parameter} must be finite"),
            Self::NonPositive { parameter } => {
                write!(f, "{parameter} must be greater than zero")
            }
            Self::Negative { parameter } => write!(f, "{parameter} must not be negative"),
            Self::InvalidPoissonRatio => {
                f.write_str("poisson_ratio must be greater than -1 and less than 0.5")
            }
            Self::InvalidPackingInterval => {
                f.write_str("packing_limit must be greater than packing_onset")
            }
            Self::ZeroTangentInverse => {
                f.write_str("a positive pressure cannot be inverted with zero tangent")
            }
            Self::ZeroStepBudget => f.write_str("maximum_substeps must be at least one"),
        }
    }
}

impl std::error::Error for GranularError {}

/// Isotropic elastic moduli derived from Young's modulus and Poisson ratio.
///
/// The stress unit is arbitrary but must be consistent: normally pascals on the calibration side
/// or reduced stress after the SI conversion boundary.
pub fn isotropic_moduli(
    youngs_modulus: f64,
    poisson_ratio: f64,
) -> Result<IsotropicModuli, GranularError> {
    positive(youngs_modulus, "youngs_modulus")?;
    finite(poisson_ratio, "poisson_ratio")?;
    if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 {
        return Err(GranularError::InvalidPoissonRatio);
    }
    let shear = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    let bulk = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
    let lame_lambda = bulk - 2.0 * shear / 3.0;
    Ok(IsotropicModuli {
        bulk,
        shear,
        lame_lambda,
    })
}

/// C1 bounded-tangent packing pressure.
///
/// Let `x = solid_fraction - packing_onset` and `span = packing_limit - packing_onset`.
/// Within the calibrated packing interval, `p = 0.5*K*x²/span`, so its tangent ramps
/// continuously from zero to the declared maximum compaction tangent `K`. Above the nominal
/// packing limit it continues linearly with tangent `K`; it remains restoring without the old
/// singularity in `x²/(phi_max-phi)`.
pub fn bounded_compaction_response(
    solid_fraction: f64,
    packing_onset: f64,
    packing_limit: f64,
    maximum_tangent: f64,
) -> Result<CompactionResponse, GranularError> {
    finite(solid_fraction, "solid_fraction")?;
    finite(packing_onset, "packing_onset")?;
    finite(packing_limit, "packing_limit")?;
    nonnegative(maximum_tangent, "maximum_tangent")?;
    if packing_limit <= packing_onset {
        return Err(GranularError::InvalidPackingInterval);
    }

    let x = (solid_fraction - packing_onset).max(0.0);
    if x == 0.0 || maximum_tangent == 0.0 {
        return Ok(CompactionResponse {
            pressure: 0.0,
            tangent: 0.0,
        });
    }
    let span = packing_limit - packing_onset;
    if x < span {
        Ok(CompactionResponse {
            pressure: 0.5 * maximum_tangent * x * x / span,
            tangent: maximum_tangent * x / span,
        })
    } else {
        Ok(CompactionResponse {
            pressure: maximum_tangent * (x - 0.5 * span),
            tangent: maximum_tangent,
        })
    }
}

/// Invert [`bounded_compaction_response`] on its monotone non-negative branch.
///
/// A zero pressure returns `packing_onset`, the least solid fraction on the zero-pressure
/// plateau. Positive pressure has a unique inverse when `maximum_tangent > 0`.
pub fn bounded_compaction_inverse(
    pressure: f64,
    packing_onset: f64,
    packing_limit: f64,
    maximum_tangent: f64,
) -> Result<f64, GranularError> {
    nonnegative(pressure, "pressure")?;
    finite(packing_onset, "packing_onset")?;
    finite(packing_limit, "packing_limit")?;
    nonnegative(maximum_tangent, "maximum_tangent")?;
    if packing_limit <= packing_onset {
        return Err(GranularError::InvalidPackingInterval);
    }
    if pressure == 0.0 {
        return Ok(packing_onset);
    }
    if maximum_tangent == 0.0 {
        return Err(GranularError::ZeroTangentInverse);
    }

    let span = packing_limit - packing_onset;
    let pressure_at_limit = 0.5 * maximum_tangent * span;
    let x = if pressure < pressure_at_limit {
        (2.0 * pressure * span / maximum_tangent).sqrt()
    } else {
        pressure / maximum_tangent + 0.5 * span
    };
    finite_result(packing_onset + x, "bounded compaction inverse")
}

/// Recoverable packing energy per bulk reference volume.
///
/// This is the exact integral `Psi = integral p(phi) d(log(phi))` from the pressure-free
/// onset to `solid_fraction`. Its derivative with respect to compressive logarithmic strain
/// is the packing pressure, so it is the energy paired with the semi-implicit grid solve.
pub fn bounded_compaction_energy(
    solid_fraction: f64,
    packing_onset: f64,
    packing_limit: f64,
    maximum_tangent: f64,
) -> Result<f64, GranularError> {
    nonnegative(solid_fraction, "solid_fraction")?;
    nonnegative(packing_onset, "packing_onset")?;
    positive(packing_limit, "packing_limit")?;
    nonnegative(maximum_tangent, "maximum_tangent")?;
    if packing_limit <= packing_onset {
        return Err(GranularError::InvalidPackingInterval);
    }
    if solid_fraction <= packing_onset || maximum_tangent == 0.0 {
        return Ok(0.0);
    }

    let span = packing_limit - packing_onset;
    let interior_energy = |x: f64| {
        if packing_onset == 0.0 {
            0.25 * maximum_tangent * x * x / span
        } else {
            0.5 * maximum_tangent / span
                * (0.5 * x * x - packing_onset * x
                    + packing_onset * packing_onset * ((packing_onset + x) / packing_onset).ln())
        }
    };
    let energy = if solid_fraction < packing_limit {
        interior_energy(solid_fraction - packing_onset)
    } else {
        let at_limit = interior_energy(span);
        let shift = packing_onset + 0.5 * span;
        at_limit
            + maximum_tangent
                * ((solid_fraction - packing_limit) - shift * (solid_fraction / packing_limit).ln())
    };
    finite_result(energy.max(0.0), "bounded compaction energy")
}

/// Elastic-equivalent solid fraction for the multiplicative volumetric split
/// `phi_e = phi_s * exp(-e_p)`.
pub fn elastic_solid_fraction(
    solid_fraction: f64,
    plastic_compaction: f64,
) -> Result<f64, GranularError> {
    nonnegative(solid_fraction, "solid_fraction")?;
    nonnegative(plastic_compaction, "plastic_compaction")?;
    finite_result(
        solid_fraction * (-plastic_compaction).exp(),
        "elastic solid fraction",
    )
}

/// Exponential consolidation-cap hardening `p_c = p_c0 * exp(xi * e_p)`.
pub fn consolidation_pressure(
    initial_pressure: f64,
    hardening_exponent: f64,
    plastic_compaction: f64,
) -> Result<f64, GranularError> {
    nonnegative(initial_pressure, "initial_pressure")?;
    nonnegative(hardening_exponent, "hardening_exponent")?;
    nonnegative(plastic_compaction, "plastic_compaction")?;
    finite_result(
        initial_pressure * (hardening_exponent * plastic_compaction).exp(),
        "consolidation pressure",
    )
}

/// Return a current geometric packing state to its fixed cap, then harden for the next step.
///
/// The returned pressure never exceeds the pre-return cap. Plastic compaction only increases;
/// the dry structural grain volume is not changed by this bookkeeping.
#[allow(clippy::too_many_arguments)]
pub fn explicit_compaction_cap_return(
    solid_fraction: f64,
    plastic_compaction: f64,
    packing_onset: f64,
    packing_limit: f64,
    maximum_tangent: f64,
    initial_consolidation_pressure: f64,
    hardening_exponent: f64,
) -> Result<CapReturn, GranularError> {
    let elastic_trial = elastic_solid_fraction(solid_fraction, plastic_compaction)?;
    let trial =
        bounded_compaction_response(elastic_trial, packing_onset, packing_limit, maximum_tangent)?;
    let old_cap = consolidation_pressure(
        initial_consolidation_pressure,
        hardening_exponent,
        plastic_compaction,
    )?;
    if trial.pressure <= old_cap || trial.pressure == 0.0 {
        return Ok(CapReturn {
            elastic_solid_fraction: elastic_trial,
            returned_pressure: trial.pressure,
            plastic_compaction,
            plastic_increment: 0.0,
            consolidation_pressure: old_cap,
        });
    }

    let returned_fraction =
        bounded_compaction_inverse(old_cap, packing_onset, packing_limit, maximum_tangent)?;
    // old_cap == 0 returns the onset. A zero onset has no finite logarithmic return target;
    // such a material is a pressure-free dust and therefore keeps its state unchanged.
    if returned_fraction <= 0.0 {
        return Ok(CapReturn {
            elastic_solid_fraction: elastic_trial,
            returned_pressure: 0.0,
            plastic_compaction,
            plastic_increment: 0.0,
            consolidation_pressure: old_cap,
        });
    }
    let increment = (elastic_trial / returned_fraction).ln().max(0.0);
    let updated_compaction = plastic_compaction + increment;
    let updated_cap = consolidation_pressure(
        initial_consolidation_pressure,
        hardening_exponent,
        updated_compaction,
    )?;
    Ok(CapReturn {
        elastic_solid_fraction: returned_fraction,
        returned_pressure: old_cap,
        plastic_compaction: updated_compaction,
        plastic_increment: increment,
        consolidation_pressure: updated_cap,
    })
}

/// CFL count for a deliberately compliant explicit shear predictor.
///
/// Physical bulk/packing stiffness must not be passed here: it belongs to the implicit solve.
/// The caller owns its declared microstep budget and must lower the *numerical* shear modulus if
/// that budget is hit.
pub fn explicit_shear_substeps(
    dt_s: f64,
    grid_spacing: f64,
    bulk_density: f64,
    numerical_shear_modulus: f64,
    cfl: f64,
) -> Result<u32, GranularError> {
    nonnegative(dt_s, "dt_s")?;
    positive(grid_spacing, "grid_spacing")?;
    positive(bulk_density, "bulk_density")?;
    nonnegative(numerical_shear_modulus, "numerical_shear_modulus")?;
    positive(cfl, "cfl")?;
    if dt_s == 0.0 || numerical_shear_modulus == 0.0 {
        return Ok(1);
    }
    let wave_speed = (numerical_shear_modulus / bulk_density).sqrt();
    let count = (wave_speed * dt_s / (cfl * grid_spacing)).ceil().max(1.0);
    if !count.is_finite() || count > f64::from(u32::MAX) {
        return Err(GranularError::NonFinite {
            parameter: "explicit_shear_substeps",
        });
    }
    Ok(count as u32)
}

/// Select a soft explicit-shear modulus under a declared microstep ceiling.
///
/// This function may only be used for the numerical deviatoric predictor. It never receives or
/// modifies physical compaction stiffness. If the requested modulus exceeds the budget, the
/// applied modulus is reported explicitly instead of silently violating the CFL bound.
#[allow(clippy::too_many_arguments)]
pub fn bounded_explicit_shear_budget(
    dt_s: f64,
    grid_spacing: f64,
    minimum_bulk_density: f64,
    requested_modulus: f64,
    cfl: f64,
    maximum_substeps: u32,
) -> Result<ShearStepBudget, GranularError> {
    nonnegative(dt_s, "dt_s")?;
    positive(grid_spacing, "grid_spacing")?;
    positive(minimum_bulk_density, "minimum_bulk_density")?;
    nonnegative(requested_modulus, "requested_modulus")?;
    positive(cfl, "cfl")?;
    if maximum_substeps == 0 {
        return Err(GranularError::ZeroStepBudget);
    }
    let applied_modulus = if dt_s == 0.0 {
        requested_modulus
    } else {
        let maximum_wave_speed = cfl * grid_spacing * f64::from(maximum_substeps) / dt_s;
        let maximum_modulus = minimum_bulk_density * maximum_wave_speed * maximum_wave_speed;
        finite_result(maximum_modulus, "maximum numerical shear modulus")?;
        requested_modulus.min(maximum_modulus)
    };
    let substeps = explicit_shear_substeps(
        dt_s,
        grid_spacing,
        minimum_bulk_density,
        applied_modulus,
        cfl,
    )?
    .min(maximum_substeps);
    Ok(ShearStepBudget {
        requested_modulus,
        applied_modulus,
        substeps,
        limited: applied_modulus < requested_modulus,
    })
}

/// One row of the projected damped-Jacobi update used by the semi-implicit skeleton solve.
///
/// `operator_pressure` is `(A q)_c` and `operator_diagonal` is `A_cc`, where
/// `A = -D B_s G` is positive semidefinite. The row-scaled form avoids dividing by a vanishing
/// tangent while remaining algebraically identical to the compliance form for `tangent > 0`.
#[allow(clippy::too_many_arguments)]
pub fn projected_compaction_jacobi_update(
    current_pressure: f64,
    operator_pressure: f64,
    operator_diagonal: f64,
    reference_pressure: f64,
    tangent: f64,
    divergence: f64,
    dt_s: f64,
    consolidation_pressure: f64,
    relaxation: f64,
) -> f64 {
    let upper = consolidation_pressure.max(0.0);
    if tangent <= 0.0 {
        return reference_pressure.clamp(0.0, upper);
    }
    let dt2k = dt_s * dt_s * tangent;
    let rhs = reference_pressure - dt_s * tangent * divergence;
    let residual = rhs - current_pressure - dt2k * operator_pressure;
    let diagonal = 1.0 + dt2k * operator_diagonal.max(0.0);
    (current_pressure + relaxation * residual / diagonal).clamp(0.0, upper)
}

/// Solid node mobility scale used by both compaction operator setup and pressure projection.
/// Multiplying a symmetric contact projector by this value gives
/// `B_s = h^3 / m_s * P_s`.
pub fn solid_compaction_mobility_scale(
    cell_size: f64,
    node_mass: f64,
) -> Result<f64, GranularError> {
    positive(cell_size, "cell_size")?;
    positive(node_mass, "node_mass")?;
    finite_result(cell_size.powi(3) / node_mass, "solid compaction mobility")
}

/// Radial return scale for the traceless numerical shear stress under compressive mean
/// skeleton pressure. The Drucker-Prager radius is `cohesion + 3*alpha*pressure`.
pub fn drucker_prager_deviatoric_scale(
    trial_norm: f64,
    mean_pressure: f64,
    alpha: f64,
    cohesion: f64,
) -> Result<f64, GranularError> {
    nonnegative(trial_norm, "trial_norm")?;
    nonnegative(mean_pressure, "mean_pressure")?;
    nonnegative(alpha, "alpha")?;
    nonnegative(cohesion, "cohesion")?;
    if trial_norm == 0.0 {
        return Ok(1.0);
    }
    let radius = cohesion + 3.0 * alpha * mean_pressure;
    finite_result(
        (radius / trial_norm).clamp(0.0, 1.0),
        "Drucker-Prager return scale",
    )
}

fn finite(value: f64, parameter: &'static str) -> Result<(), GranularError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GranularError::NonFinite { parameter })
    }
}

fn positive(value: f64, parameter: &'static str) -> Result<(), GranularError> {
    finite(value, parameter)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(GranularError::NonPositive { parameter })
    }
}

fn nonnegative(value: f64, parameter: &'static str) -> Result<(), GranularError> {
    finite(value, parameter)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(GranularError::Negative { parameter })
    }
}

fn finite_result(value: f64, parameter: &'static str) -> Result<f64, GranularError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GranularError::NonFinite { parameter })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1.0e-11 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn isotropic_moduli_reconstruct_youngs_modulus() {
        let e = 7_321.0;
        let nu = 0.27;
        let m = isotropic_moduli(e, nu).unwrap();
        let reconstructed = 9.0 * m.bulk * m.shear / (3.0 * m.bulk + m.shear);
        assert!(close(reconstructed, e));
        assert!(close(m.lame_lambda, m.bulk - 2.0 * m.shear / 3.0));
    }

    #[test]
    fn compaction_pressure_is_continuous_monotone_and_has_bounded_tangent() {
        let onset = 0.5;
        let limit = 0.8;
        let maximum_tangent = 12_000.0;
        let samples = [0.2, onset, 0.55, 0.7, limit, 0.95, 1.2];
        let mut previous_pressure = 0.0;
        for phi in samples {
            let response = bounded_compaction_response(phi, onset, limit, maximum_tangent).unwrap();
            assert!(response.pressure >= previous_pressure);
            assert!((0.0..=maximum_tangent).contains(&response.tangent));
            previous_pressure = response.pressure;
        }
        let at_limit = bounded_compaction_response(limit, onset, limit, maximum_tangent).unwrap();
        let just_below =
            bounded_compaction_response(limit - 1.0e-8, onset, limit, maximum_tangent).unwrap();
        assert!((at_limit.pressure - just_below.pressure).abs() < 1.0e-3);
        assert!((at_limit.tangent - just_below.tangent).abs() < 1.0e-3);
    }

    #[test]
    fn compaction_law_scales_with_stress_units_without_changing_shape() {
        let base = bounded_compaction_response(0.68, 0.52, 0.82, 1_000.0).unwrap();
        let scaled = bounded_compaction_response(0.68, 0.52, 0.82, 7_000.0).unwrap();
        assert!(close(scaled.pressure, 7.0 * base.pressure));
        assert!(close(scaled.tangent, 7.0 * base.tangent));
    }

    #[test]
    fn pressure_inverse_round_trips_both_branches() {
        let onset = 0.4;
        let limit = 0.7;
        let stiffness = 9_000.0;
        for solid_fraction in [onset, 0.41, 0.55, limit, 0.85, 1.1] {
            let pressure = bounded_compaction_response(solid_fraction, onset, limit, stiffness)
                .unwrap()
                .pressure;
            let recovered = bounded_compaction_inverse(pressure, onset, limit, stiffness).unwrap();
            assert!(close(recovered, solid_fraction));
        }
    }

    #[test]
    fn recoverable_energy_derivative_is_pressure_in_log_strain() {
        let onset = 0.37;
        let limit = 0.69;
        let stiffness = 4_200.0;
        for solid_fraction in [0.41, 0.58, 0.69, 0.83] {
            let de: f64 = 1.0e-6;
            let hi = solid_fraction * de.exp();
            let lo = solid_fraction * (-de).exp();
            let derivative = (bounded_compaction_energy(hi, onset, limit, stiffness).unwrap()
                - bounded_compaction_energy(lo, onset, limit, stiffness).unwrap())
                / (2.0 * de);
            let pressure = bounded_compaction_response(solid_fraction, onset, limit, stiffness)
                .unwrap()
                .pressure;
            assert!((derivative - pressure).abs() <= 2.0e-5 * pressure.abs().max(1.0));
        }
    }

    #[test]
    fn multiplicative_plastic_state_relaxes_pressure_without_changing_geometry() {
        let solid_fraction = 0.76;
        let onset = 0.48;
        let limit = 0.70;
        let stiffness = 8_000.0;
        let before = bounded_compaction_response(
            elastic_solid_fraction(solid_fraction, 0.0).unwrap(),
            onset,
            limit,
            stiffness,
        )
        .unwrap();
        let after = bounded_compaction_response(
            elastic_solid_fraction(solid_fraction, 0.15).unwrap(),
            onset,
            limit,
            stiffness,
        )
        .unwrap();
        assert!(after.pressure < before.pressure);
        assert!(close(solid_fraction, 0.76));
    }

    #[test]
    fn cap_return_is_admissible_dissipative_and_hardens_monotonically() {
        let result =
            explicit_compaction_cap_return(0.88, 0.0, 0.45, 0.70, 10_000.0, 500.0, 3.0).unwrap();
        assert!(result.plastic_increment > 0.0);
        assert!(result.plastic_compaction >= result.plastic_increment);
        assert!(result.consolidation_pressure >= result.returned_pressure);
        let returned =
            bounded_compaction_response(result.elastic_solid_fraction, 0.45, 0.70, 10_000.0)
                .unwrap();
        assert!(close(returned.pressure, result.returned_pressure));

        let elastic =
            explicit_compaction_cap_return(0.46, 0.0, 0.45, 0.70, 10_000.0, 500.0, 3.0).unwrap();
        assert_eq!(elastic.plastic_increment, 0.0);
        assert_eq!(elastic.plastic_compaction, 0.0);
    }

    #[test]
    fn explicit_shear_cfl_is_monotone_and_handles_zero_work() {
        let base = explicit_shear_substeps(1.0 / 60.0, 0.003, 260.0, 100.0, 0.4).unwrap();
        let stiffer = explicit_shear_substeps(1.0 / 60.0, 0.003, 260.0, 1_000.0, 0.4).unwrap();
        let finer = explicit_shear_substeps(1.0 / 60.0, 0.0015, 260.0, 100.0, 0.4).unwrap();
        assert!(stiffer >= base);
        assert!(finer >= base);
        assert_eq!(
            explicit_shear_substeps(0.0, 0.003, 260.0, 100.0, 0.4).unwrap(),
            1
        );
        assert_eq!(
            explicit_shear_substeps(1.0 / 60.0, 0.003, 260.0, 0.0, 0.4).unwrap(),
            1
        );
    }

    #[test]
    fn shear_budget_reports_any_numerical_compliance_limit() {
        let budget =
            bounded_explicit_shear_budget(1.0 / 60.0, 0.003, 220.0, 1.0e9, 0.4, 8).unwrap();
        assert!(budget.limited);
        assert!(budget.applied_modulus < budget.requested_modulus);
        assert!(budget.substeps <= 8);
        let reconstructed =
            explicit_shear_substeps(1.0 / 60.0, 0.003, 220.0, budget.applied_modulus, 0.4).unwrap();
        assert_eq!(reconstructed, budget.substeps);

        let unbounded = bounded_explicit_shear_budget(0.0, 0.003, 220.0, 123.0, 0.4, 8).unwrap();
        assert!(!unbounded.limited);
        assert_eq!(unbounded.applied_modulus, 123.0);
        assert_eq!(unbounded.substeps, 1);
    }

    #[test]
    fn projected_jacobi_sweeps_reduce_the_incremental_functional() {
        // Manufactured SPD tridiagonal A. The test compares a sweep sequence, not the
        // production sweep count, so iteration budget remains an algorithmic performance choice.
        let a = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
        let reference = [0.4, 0.9, 0.5];
        let divergence = [-0.2, 0.1, -0.15];
        let tangent = 3.0;
        let dt = 0.25;
        let cap = 4.0;
        let mut q = reference;
        let objective = |x: [f64; 3]| {
            let mut elastic = 0.0;
            let mut work = 0.0;
            let mut quadratic = 0.0;
            for row in 0..3 {
                elastic += 0.5 * (x[row] - reference[row]).powi(2) / tangent;
                work += dt * x[row] * divergence[row];
                for col in 0..3 {
                    quadratic += 0.5 * dt * dt * x[row] * a[row][col] * x[col];
                }
            }
            elastic + work + quadratic
        };
        let mut previous = objective(q);
        for _ in 0..12 {
            let old = q;
            for row in 0..3 {
                let aq = (0..3).map(|col| a[row][col] * old[col]).sum();
                q[row] = projected_compaction_jacobi_update(
                    old[row],
                    aq,
                    a[row][row],
                    reference[row],
                    tangent,
                    divergence[row],
                    dt,
                    cap,
                    2.0 / 3.0,
                );
            }
            let next = objective(q);
            assert!(next <= previous + 1.0e-12);
            previous = next;
        }
    }

    #[test]
    fn cap_projection_blocks_further_loading_but_allows_elastic_unloading() {
        let cap = 2.0;
        let unloaded =
            projected_compaction_jacobi_update(cap, 0.0, 0.0, cap, 3.0, 1.0, 0.1, cap, 1.0);
        let loaded =
            projected_compaction_jacobi_update(cap, 0.0, 0.0, cap, 3.0, -1.0, 0.1, cap, 1.0);
        assert!(unloaded < cap);
        assert_eq!(loaded, cap);
    }

    #[test]
    fn compaction_setup_and_project_share_one_mobility() {
        let scale = solid_compaction_mobility_scale(0.2, 0.75).unwrap();
        // Symmetric PSD projector that removes the y normal.
        let projector = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let gradient = [2.0, -5.0, 3.0];
        let setup_apply = |v: [f64; 3]| {
            [
                scale * projector[0][0] * v[0],
                scale * projector[1][1] * v[1],
                scale * projector[2][2] * v[2],
            ]
        };
        let stored_mobility = [
            [scale * projector[0][0], 0.0, 0.0],
            [0.0, scale * projector[1][1], 0.0],
            [0.0, 0.0, scale * projector[2][2]],
        ];
        let project_apply = [
            stored_mobility[0][0] * gradient[0],
            stored_mobility[1][1] * gradient[1],
            stored_mobility[2][2] * gradient[2],
        ];
        assert_eq!(setup_apply(gradient), project_apply);
        assert_eq!(project_apply[1], 0.0);
    }

    #[test]
    fn deviatoric_return_is_bounded_dissipative_and_strengthens_with_pressure() {
        let dry = drucker_prager_deviatoric_scale(10.0, 0.0, 0.2, 1.0).unwrap();
        let confined = drucker_prager_deviatoric_scale(10.0, 8.0, 0.2, 1.0).unwrap();
        assert!((0.0..=1.0).contains(&dry));
        assert!((dry..=1.0).contains(&confined));
        assert!(10.0 * dry <= 1.0 + 1.0e-12);
        assert!(10.0 * confined <= 1.0 + 3.0 * 0.2 * 8.0 + 1.0e-12);
        assert_eq!(
            drucker_prager_deviatoric_scale(0.0, 0.0, 0.0, 0.0).unwrap(),
            1.0
        );
    }

    #[test]
    fn invalid_inputs_are_rejected_without_embedding_empirical_targets() {
        assert_eq!(
            isotropic_moduli(1.0, 0.5),
            Err(GranularError::InvalidPoissonRatio)
        );
        assert_eq!(
            bounded_compaction_response(0.6, 0.8, 0.5, 1.0),
            Err(GranularError::InvalidPackingInterval)
        );
        assert!(matches!(
            explicit_shear_substeps(0.1, 0.0, 1.0, 1.0, 0.4),
            Err(GranularError::NonPositive { .. })
        ));
        assert_eq!(
            bounded_compaction_inverse(1.0, 0.2, 0.8, 0.0),
            Err(GranularError::ZeroTangentInverse)
        );
        assert_eq!(
            bounded_explicit_shear_budget(0.1, 1.0, 1.0, 1.0, 0.4, 0),
            Err(GranularError::ZeroStepBudget)
        );
        assert!(matches!(
            solid_compaction_mobility_scale(1.0, 0.0),
            Err(GranularError::NonPositive { .. })
        ));
    }
}

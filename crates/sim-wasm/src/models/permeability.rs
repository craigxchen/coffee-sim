//! Packed-bed permeability models shared by solvers.

/// Viscous Kozeny-Carman / Ergun coefficient used for packed-bed Darcy drag.
pub const DARCY_VISCOUS_COEFF: f32 = 150.0;

/// Kozeny-Carman permeability for a packed bed of roughly spherical grains.
///
/// `d` is grain diameter and `phi` is bed porosity. Smaller grains lower permeability and
/// therefore increase drag. The porosity term is clamped away from singular endpoints so callers
/// can safely expose it as a material knob.
pub fn kozeny_carman(d: f32, phi: f32) -> f32 {
    let phi = phi.clamp(1.0e-4, 0.999);
    let solid = (1.0 - phi).max(1.0e-4);
    d * d * phi.powi(3) / (DARCY_VISCOUS_COEFF * solid.powi(2))
}

/// Resolve the solver drag rate from permeability.
pub fn drag_rate(k: f32, scale: f32) -> f32 {
    scale / k.max(1.0e-9)
}

/// Local intrinsic-velocity Darcy drag rate in reduced solver units.
///
/// The solver evolves pore velocity `v_f`, while Darcy flux is superficial
/// `q = phi * (v_f - v_s)`. With `beta = phi * nu / K`, steady forcing `F` therefore gives
/// `q = phi * F / beta = K * F / nu` independently of which porosity is used to represent the
/// same intrinsic permeability.
pub fn darcy_drag_rate(d: f32, phi: f32, scale: f32) -> f32 {
    darcy_mobile_drag_rate(d, phi, phi, scale)
}

/// Intrinsic mobile-velocity drag rate for a partially occupied structural pore.
///
/// `K` is fixed by the structural porosity, while superficial mobile flux is
/// `q = phi_mobile * (v_mobile-v_composite)`. Consequently beta uses `phi_mobile` in its
/// numerator. Relative permeability is applied by the caller as `beta/kr`.
pub fn darcy_mobile_drag_rate(d: f32, structural_phi: f32, mobile_phi: f32, scale: f32) -> f32 {
    mobile_phi.clamp(0.0, 1.0) * drag_rate(kozeny_carman(d, structural_phi), scale)
}

#[cfg(test)]
mod tests {
    use super::{
        darcy_drag_rate, darcy_mobile_drag_rate, drag_rate, kozeny_carman, DARCY_VISCOUS_COEFF,
    };

    #[test]
    fn finer_grind_has_lower_permeability_and_higher_drag() {
        let phi = 0.40;
        let fine = kozeny_carman(0.6, phi);
        let coarse = kozeny_carman(1.2, phi);
        assert!(fine < coarse, "fine {fine} coarse {coarse}");

        let gamma = 1.0;
        assert!(drag_rate(fine, gamma) > drag_rate(coarse, gamma));
    }

    /// Quantitative Kozeny–Carman law: permeability matches the closed form and
    /// scales as `d²` at fixed porosity, so the solver's resolved drag rate
    /// (`β = phi*drag_scale / k`) scales as `d⁻²`.
    /// This pins the grind→flow exponent the ordinal test above only orders.
    #[test]
    fn permeability_obeys_kozeny_carman_scaling() {
        let phi: f32 = 0.40;
        // k(d) = d²·φ³ / (150·(1−φ)²), exact to float precision.
        let closed = |d: f32| d * d * phi.powi(3) / (DARCY_VISCOUS_COEFF * (1.0 - phi).powi(2));
        for &d in &[0.4f32, 0.6, 1.0, 1.5, 2.0] {
            let (k, e) = (kozeny_carman(d, phi), closed(d));
            assert!((k - e).abs() <= 1e-6 * e, "k({d}) = {k}, closed form {e}");
        }

        // d² scaling at fixed porosity: doubling the grind quadruples permeability.
        let k_ratio = kozeny_carman(1.2, phi) / kozeny_carman(0.6, phi);
        assert!(
            (k_ratio - 4.0).abs() < 1e-4,
            "k(2d)/k(d) = {k_ratio}, expected 4"
        );

        // The solver wires β = drag_scale / k, so resolved drag ∝ d⁻²: halving the
        // grain diameter quadruples the drag rate (the grind→flow knob).
        let gamma = 0.02;
        let drag_ratio =
            drag_rate(kozeny_carman(0.6, phi), gamma) / drag_rate(kozeny_carman(1.2, phi), gamma);
        assert!(
            (drag_ratio - 4.0).abs() < 1e-4,
            "γ(d/2)/γ(d) = {drag_ratio}, expected 4"
        );

        // Porosity dependence φ³/(1−φ)² at fixed diameter.
        let phi_ratio = kozeny_carman(1.0, 0.45) / kozeny_carman(1.0, 0.35);
        let closed_phi = (0.45f32.powi(3) / (1.0f32 - 0.45).powi(2))
            / (0.35f32.powi(3) / (1.0f32 - 0.35).powi(2));
        assert!(
            (phi_ratio - closed_phi).abs() <= 1e-5 * closed_phi,
            "φ-scaling {phi_ratio} vs {closed_phi}"
        );

        let beta = darcy_drag_rate(1.0, phi, gamma);
        let closed_beta = gamma * DARCY_VISCOUS_COEFF * (1.0 - phi).powi(2) / phi.powi(2);
        assert!(
            (beta - closed_beta).abs() <= 1.0e-5 * closed_beta,
            "Darcy beta {beta} vs closed form {closed_beta}"
        );
    }

    #[test]
    fn intrinsic_drag_recovers_superficial_darcy_flux_across_porosity() {
        let forcing = 2.7;
        let viscosity_scale = 0.031;
        for phi in [0.25_f32, 0.4, 0.65, 0.85] {
            let permeability = kozeny_carman(0.8, phi);
            let beta = darcy_drag_rate(0.8, phi, viscosity_scale);
            let intrinsic_relative_velocity = forcing / beta;
            let superficial_flux = phi * intrinsic_relative_velocity;
            let darcy_flux = permeability * forcing / viscosity_scale;
            assert!(
                (superficial_flux - darcy_flux).abs()
                    <= 2.0e-6 * superficial_flux.abs().max(darcy_flux.abs()).max(1.0),
                "phi={phi}: q={superficial_flux}, K F/nu={darcy_flux}"
            );
        }
    }

    #[test]
    fn partial_pore_drag_recovers_corey_scaled_darcy_flux_independent_of_theta() {
        let (diameter, structural_phi, forcing, viscosity_scale, relative_permeability) =
            (0.8_f32, 0.42, 2.7, 0.031, 0.23);
        let permeability = kozeny_carman(diameter, structural_phi);
        let expected = permeability * relative_permeability * forcing / viscosity_scale;
        for (mobile_phi, theta) in [(0.08_f32, 0.15_f32), (0.21, 0.5), (0.41, 0.95)] {
            let beta =
                darcy_mobile_drag_rate(diameter, structural_phi, mobile_phi, viscosity_scale)
                    / relative_permeability;
            let intrinsic_mobile_velocity = forcing / beta;
            let superficial_flux = mobile_phi * intrinsic_mobile_velocity;
            assert!((superficial_flux - expected).abs() <= 2.0e-6 * expected.abs().max(1.0));

            // theta changes only how v_mobile is stored in v_bar; it cannot enter intrinsic beta.
            let homogenized_relative_velocity = theta * intrinsic_mobile_velocity;
            assert!(
                (homogenized_relative_velocity / theta - intrinsic_mobile_velocity).abs()
                    <= 2.0e-6 * intrinsic_mobile_velocity.abs().max(1.0)
            );
        }
    }
}

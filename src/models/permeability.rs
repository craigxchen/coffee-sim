//! Packed-bed permeability models shared by solvers.

/// Kozeny-Carman permeability for a packed bed of roughly spherical grains.
///
/// `d` is grain diameter and `phi` is bed porosity. Smaller grains lower permeability and
/// therefore increase drag. The porosity term is clamped away from singular endpoints so callers
/// can safely expose it as a material knob.
pub fn kozeny_carman(d: f32, phi: f32) -> f32 {
    let phi = phi.clamp(1.0e-4, 0.999);
    let solid = (1.0 - phi).max(1.0e-4);
    d * d * phi.powi(3) / (180.0 * solid.powi(2))
}

/// Resolve the solver drag rate from permeability.
pub fn drag_rate(k: f32, gamma: f32) -> f32 {
    gamma / k.max(1.0e-9)
}

#[cfg(test)]
mod tests {
    use super::{drag_rate, kozeny_carman};

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
    /// (`γ = drag_gamma / k`, wired in `solvers::xpbd::build`) scales as `d⁻²`.
    /// This pins the grind→flow exponent the ordinal test above only orders.
    #[test]
    fn permeability_obeys_kozeny_carman_scaling() {
        let phi: f32 = 0.40;
        // k(d) = d²·φ³ / (180·(1−φ)²), exact to float precision.
        let closed = |d: f32| d * d * phi.powi(3) / (180.0 * (1.0 - phi).powi(2));
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

        // The solver wires γ = drag_gamma / k, so resolved drag ∝ d⁻²: halving the
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
    }
}

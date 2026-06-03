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
}

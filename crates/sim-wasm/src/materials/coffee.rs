pub(crate) const fn kozeny_carman_permeability_m2(grind_diameter_um: f32, porosity: f32) -> f32 {
    let d_m = grind_diameter_um * 1.0e-6;
    let phi = porosity;
    let solid = 1.0 - phi;
    d_m * d_m * phi * phi * phi / (180.0 * solid * solid)
}

pub(crate) fn clamped_porosity(phi: f32) -> f32 {
    phi.clamp(0.25, 0.55)
}

pub(crate) fn permeability_from_grind_and_porosity(grind_diameter_um: f32, porosity: f32) -> f32 {
    kozeny_carman_permeability_m2(grind_diameter_um, clamped_porosity(porosity))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finer_grind_has_lower_permeability() {
        let fine = permeability_from_grind_and_porosity(320.0, 0.40);
        let coarse = permeability_from_grind_and_porosity(800.0, 0.40);
        assert!(fine < coarse);
    }
}

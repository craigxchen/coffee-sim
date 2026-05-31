#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstraintConfig {
    pub water_compliance: f32,
    pub coffee_contact_compliance: f32,
    pub coffee_rest_shape_compliance_dry: f32,
    pub coffee_rest_shape_compliance_wet: f32,
    pub xsph_viscosity: f32,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            water_compliance: 1.0e-6,
            coffee_contact_compliance: 5.0e-5,
            coffee_rest_shape_compliance_dry: 2.0e-4,
            coffee_rest_shape_compliance_wet: 8.0e-5,
            xsph_viscosity: 0.02,
        }
    }
}

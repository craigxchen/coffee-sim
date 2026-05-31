use coffee_sim_core::Vec3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct CupConfig {
    pub center: Vec3,
    pub radius: f32,
    pub top_y: f32,
    pub bot_y: f32,
}

impl Default for CupConfig {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            radius: 3.0,
            top_y: -3.5,
            bot_y: -8.0,
        }
    }
}

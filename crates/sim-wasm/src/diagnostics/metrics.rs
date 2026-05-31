#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MetricsSnapshot {
    pub max_abs_div: f32,
    pub fluid_cells: u32,
    pub div_clamp_fires: u32,
    pub pressure_clamp_fires: u32,
    pub mass_overflow_fires: u32,
    pub projection_residual_max_abs_div: f32,
    pub projection_residual_mean_abs_div: f32,
    pub projection_residual_cells: u32,
    pub mean_tds: f32,
    pub cup_tds: f32,
    pub extraction_yield: f32,
}

pub(crate) const METRICS_SLOT_COUNT: usize = 12;

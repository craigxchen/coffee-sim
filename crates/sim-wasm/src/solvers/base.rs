use crate::emission::EmissionInput;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::ui::RenderView;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;
use coffee_sim_core::Vec3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SolverId {
    Mpm,
    Xpbd,
    Twofield,
}

impl SolverId {
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Mpm => "mpm",
            Self::Xpbd => "xpbd",
            Self::Twofield => "twofield",
        }
    }

    pub(crate) fn all() -> &'static [Self] {
        &[Self::Mpm, Self::Xpbd, Self::Twofield]
    }

    pub(crate) fn from_id(id: &str) -> Option<Self> {
        match id {
            "mpm" => Some(Self::Mpm),
            "xpbd" => Some(Self::Xpbd),
            "twofield" => Some(Self::Twofield),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Paradigm {
    ForceBased,
    PositionBased,
    Hybrid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Stability {
    CflLimited { c: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SolverInfo {
    pub(crate) id: SolverId,
    pub(crate) name: &'static str,
    pub(crate) paradigm: Paradigm,
    pub(crate) owns_grid: bool,
    pub(crate) stability: Stability,
    pub(crate) experimental: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SceneSpec {
    CenterPour,
    FreeStream,
    Debug { id: String },
}

impl SceneSpec {
    pub(crate) fn is_debug(&self) -> bool {
        matches!(self, Self::Debug { .. })
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct ControlState {
    pub(crate) water_velocity_m_s: f32,
    pub(crate) spout_position: Vec3,
}

#[derive(Clone, Copy)]
pub(crate) struct FrameContext<'a> {
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
    pub(crate) dt: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct ResetContext<'a> {
    pub(crate) device: &'a wgpu::Device,
    pub(crate) queue: &'a wgpu::Queue,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CommonMetrics {
    pub(crate) particle_count: usize,
    pub(crate) water_slots_used: u32,
    pub(crate) bed_particle_count: u32,
    pub(crate) max_particles: u32,
    pub(crate) sim_time_s: f32,
    pub(crate) frame_emitted_mass: f32,
    pub(crate) frame_emitted_ml: f32,
    pub(crate) total_emitted_mass: f32,
    pub(crate) total_emitted_ml: f32,
    pub(crate) frame_dropped_particles: u32,
    pub(crate) total_dropped_particles: u32,
    pub(crate) flow_rate_ml_s: f32,
    pub(crate) exit_speed: f32,
    pub(crate) exit_speed_m_s: f32,
    pub(crate) spout_position: Vec3,
    pub(crate) has_bed: bool,
    pub(crate) last_pressure_pairs: u32,
    pub(crate) max_abs_divergence: f32,
    pub(crate) fluid_cell_count: u32,
    pub(crate) div_clamp_fires: u32,
    pub(crate) pressure_clamp_fires: u32,
    pub(crate) mass_overflow_fires: u32,
    pub(crate) projection_residual_max_abs_divergence: f32,
    pub(crate) projection_residual_mean_abs_divergence: f32,
    pub(crate) projection_residual_cell_count: u32,
    pub(crate) mean_tds: f32,
    pub(crate) cup_tds: f32,
    pub(crate) extraction_yield: f32,
    pub(crate) estimated_cup_tds: f32,
    pub(crate) estimated_extraction_yield: f32,
}

pub(crate) struct FrameSnapshot<'a> {
    pub(crate) render: RenderView<'a>,
    pub(crate) metrics: CommonMetrics,
}

pub(crate) trait FrameSolver {
    fn reset(&mut self, ctx: ResetContext<'_>) -> CommonMetrics;
    fn reset_scene(
        &mut self,
        ctx: ResetContext<'_>,
        scene: &SceneSpec,
    ) -> Result<CommonMetrics, String>;
    fn step_frame(&mut self, ctx: FrameContext<'_>) -> CommonMetrics;
    fn snapshot(&self) -> FrameSnapshot<'_>;
    fn set_water_velocity_m_s(&mut self, speed_m_s: f32);
    fn set_spout_position(&mut self, x: f32, y: f32, z: f32);
    fn set_pressure_residual_adaptation(&mut self, _target: f32, _max_pairs: u32) {}
    #[cfg(target_arch = "wasm32")]
    fn metrics_buffer(&self) -> Option<wgpu::Buffer> {
        None
    }
}

/// Compatibility seam used by the imported `rewrite` XPBD core.
///
/// Keep this trait's shape aligned with `rewrite:src/solvers/base.rs` so
/// `solvers::xpbd` can remain a direct import. The production app goes through
/// [`FrameSolver`]; the XPBD adapter bridges between the two traits.
pub(crate) trait Solver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self
    where
        Self: Sized;

    fn reset(&mut self, scene: &Scene);
    fn step(&mut self, dt: f32, input: &EmissionInput);
    fn particles(&self) -> ParticleBuffers;
    fn metrics(&self) -> Metrics;
    fn profile(&self) -> Profile;
}

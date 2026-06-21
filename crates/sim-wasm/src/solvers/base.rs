use std::any::Any;

use crate::ui::RenderView;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SolverId {
    Mpm,
}

impl SolverId {
    #[allow(dead_code)]
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Mpm => "mpm",
        }
    }

    #[allow(dead_code)]
    pub(crate) fn all() -> &'static [Self] {
        &[Self::Mpm]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Paradigm {
    ForceBased,
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
}

pub(crate) trait Solver: Any {
    #[allow(dead_code)]
    fn id(&self) -> SolverId;
    #[allow(dead_code)]
    fn info(&self) -> SolverInfo;
    fn reset(&mut self, queue: &wgpu::Queue, device: &wgpu::Device);
    fn step_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32);
    fn render_view(&self) -> RenderView<'_>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

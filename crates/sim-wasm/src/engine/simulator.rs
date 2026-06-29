use crate::solvers::base::{
    CommonMetrics, FrameContext, FrameSnapshot, FrameSolver, ResetContext, SceneSpec, SolverId,
};
use crate::solvers::registry::build_solver;

pub(crate) struct Simulator {
    active_id: SolverId,
    scene: SceneSpec,
    metrics: CommonMetrics,
    solver: Box<dyn FrameSolver>,
}

impl Simulator {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: SolverId,
        scene: SceneSpec,
    ) -> Result<Self, String> {
        let solver = build_solver(id, device, queue, &scene)?;
        let metrics = solver.snapshot().metrics;
        Ok(Self {
            active_id: id,
            scene,
            metrics,
            solver,
        })
    }

    pub(crate) fn active_id(&self) -> SolverId {
        self.active_id
    }

    pub(crate) fn switch_solver(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: SolverId,
    ) -> Result<(), String> {
        let scene = if self.scene.is_debug() && id != SolverId::Mpm {
            SceneSpec::CenterPour
        } else {
            self.scene.clone()
        };
        let solver = build_solver(id, device, queue, &scene)?;
        let metrics = solver.snapshot().metrics;
        self.active_id = id;
        self.scene = scene;
        self.metrics = metrics;
        self.solver = solver;
        Ok(())
    }

    pub(crate) fn reset(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        let ctx = ResetContext { device, queue };
        self.metrics = self.solver.reset(ctx);
    }

    pub(crate) fn load_scene(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        scene: SceneSpec,
    ) -> Result<(), String> {
        let scene = if scene.is_debug() && self.active_id != SolverId::Mpm {
            SceneSpec::CenterPour
        } else {
            scene
        };
        let ctx = ResetContext { device, queue };
        self.metrics = self.solver.reset_scene(ctx, &scene)?;
        self.scene = scene;
        Ok(())
    }

    pub(crate) fn step_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        self.metrics = self.solver.step_frame(FrameContext { device, queue, dt });
    }

    pub(crate) fn snapshot(&self) -> FrameSnapshot<'_> {
        self.solver.snapshot()
    }

    pub(crate) fn common_metrics(&self) -> CommonMetrics {
        self.metrics
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn metrics_buffer(&self) -> Option<wgpu::Buffer> {
        self.solver.metrics_buffer()
    }

    pub(crate) fn set_exit_speed_m_s(&mut self, speed_m_s: f32) {
        self.solver.set_water_velocity_m_s(speed_m_s);
        self.metrics = self.solver.snapshot().metrics;
    }

    pub(crate) fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.solver.set_spout_position(x, y, z);
        self.metrics = self.solver.snapshot().metrics;
    }

    pub(crate) fn set_pressure_residual_adaptation(&mut self, target: f32, max_pairs: u32) {
        self.solver
            .set_pressure_residual_adaptation(target, max_pairs);
        self.metrics = self.solver.snapshot().metrics;
    }
}

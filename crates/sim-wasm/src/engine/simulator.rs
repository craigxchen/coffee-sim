use crate::solvers::base::{Solver, SolverId, SolverInfo};
#[cfg(target_arch = "wasm32")]
use crate::solvers::mpm::WaterDiagnostics;
use crate::solvers::mpm::{DebugScene, MetricsSnapshot, MpmSettings, MpmSim3D};
use crate::solvers::registry::{build_solver, info_for, SolverBuildConfig};
use crate::ui::RenderView;

pub(crate) struct Simulator {
    active_id: SolverId,
    active_info: SolverInfo,
    solver: Box<dyn Solver>,
}

impl Simulator {
    pub(crate) fn new_mpm(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: MpmSettings,
    ) -> Self {
        let active_id = SolverId::Mpm;
        let solver = build_solver(active_id, device, queue, SolverBuildConfig::Mpm(settings));
        Self {
            active_id,
            active_info: info_for(active_id),
            solver,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn active_id(&self) -> SolverId {
        self.active_id
    }

    #[allow(dead_code)]
    pub(crate) fn active_info(&self) -> SolverInfo {
        self.active_info
    }

    pub(crate) fn rebuild_mpm(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: MpmSettings,
    ) {
        self.solver = build_solver(
            SolverId::Mpm,
            device,
            queue,
            SolverBuildConfig::Mpm(settings),
        );
        self.active_id = SolverId::Mpm;
        self.active_info = info_for(SolverId::Mpm);
    }

    pub(crate) fn reset(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        self.solver.reset(queue, device);
    }

    pub(crate) fn step_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        self.solver.step_frame(device, queue, dt);
    }

    pub(crate) fn render_view(&self) -> RenderView<'_> {
        self.solver.render_view()
    }

    pub(crate) fn mpm(&self) -> &MpmSim3D {
        self.solver
            .as_any()
            .downcast_ref::<MpmSim3D>()
            .expect("active solver is MPM")
    }

    fn mpm_mut(&mut self) -> &mut MpmSim3D {
        self.solver
            .as_any_mut()
            .downcast_mut::<MpmSim3D>()
            .expect("active solver is MPM")
    }

    pub(crate) fn seed_filter_water_block(&mut self, queue: &wgpu::Queue) {
        self.mpm_mut().seed_filter_water_block(queue);
    }

    pub(crate) fn seed_debug_scene(&mut self, scene: DebugScene, queue: &wgpu::Queue) {
        scene.seed(self.mpm_mut(), queue);
    }

    pub(crate) fn set_exit_speed_m_s(&mut self, speed_m_s: f32) {
        self.mpm_mut().set_exit_speed_m_s(speed_m_s);
    }

    pub(crate) fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.mpm_mut().set_spout_position(x, y, z);
    }

    pub(crate) fn spout_position(&self) -> coffee_sim_core::Vec3 {
        self.mpm().spout_position()
    }

    pub(crate) fn flow_rate_ml_s(&self) -> f32 {
        self.mpm().flow_rate_ml_s()
    }

    pub(crate) fn exit_speed(&self) -> f32 {
        self.mpm().exit_speed()
    }

    pub(crate) fn exit_speed_m_s(&self) -> f32 {
        self.mpm().exit_speed_m_s()
    }

    pub(crate) fn particle_count(&self) -> usize {
        self.mpm().particle_count()
    }

    pub(crate) fn water_slots_used(&self) -> u32 {
        self.mpm().water_slots_used()
    }

    pub(crate) fn bed_particle_count(&self) -> u32 {
        self.mpm().bed_particle_count()
    }

    pub(crate) fn max_particles(&self) -> u32 {
        self.mpm().max_particles()
    }

    pub(crate) fn total_time(&self) -> f32 {
        self.mpm().total_time()
    }

    pub(crate) fn frame_emitted_mass(&self) -> f32 {
        self.mpm().frame_emitted_mass()
    }

    pub(crate) fn frame_emitted_ml(&self) -> f32 {
        self.mpm().frame_emitted_ml()
    }

    pub(crate) fn frame_dropped_particles(&self) -> u32 {
        self.mpm().frame_dropped_particles()
    }

    pub(crate) fn total_emitted_mass(&self) -> f32 {
        self.mpm().total_emitted_mass()
    }

    pub(crate) fn total_emitted_ml(&self) -> f32 {
        self.mpm().total_emitted_ml()
    }

    pub(crate) fn total_dropped_particles(&self) -> u32 {
        self.mpm().total_dropped_particles()
    }

    pub(crate) fn settings(&self) -> &MpmSettings {
        self.mpm().settings()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn metrics_buffer(&self) -> wgpu::Buffer {
        self.mpm().metrics_buffer()
    }

    pub(crate) fn latest_metrics(&self) -> MetricsSnapshot {
        self.mpm().latest_metrics()
    }

    pub(crate) fn last_pressure_rbgs_pairs(&self) -> u32 {
        self.mpm().last_pressure_rbgs_pairs()
    }

    pub(crate) fn set_pressure_residual_adaptation(&mut self, target: f32, max_pairs: u32) {
        self.mpm_mut()
            .set_pressure_residual_adaptation(target, max_pairs);
    }

    pub(crate) fn estimated_cup_tds(&self) -> f32 {
        self.mpm().estimated_cup_tds()
    }

    pub(crate) fn estimated_extraction_yield(&self) -> f32 {
        self.mpm().estimated_extraction_yield()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn refresh_metrics(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), wasm_bindgen::JsValue> {
        self.mpm_mut().refresh_metrics(device, queue).await
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn water_diagnostics(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<WaterDiagnostics, wasm_bindgen::JsValue> {
        self.mpm().water_diagnostics(device, queue).await
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) async fn sample_metrics_after_delay(
        device: wgpu::Device,
        queue: wgpu::Queue,
        metrics: wgpu::Buffer,
        has_bed: bool,
        delay_frames: u32,
    ) -> Result<MetricsSnapshot, wasm_bindgen::JsValue> {
        MpmSim3D::sample_metrics_after_delay(device, queue, metrics, has_bed, delay_frames).await
    }
}

use coffee_sim_core::Vec3;

use crate::boundaries::filter::FilterMesh;
use crate::diagnostics::{MetricsSnapshot, WaterDiagnostics};
use crate::engine::xpbd::XpbdEngine;
use crate::engine::SimulationEngine;
use crate::scene::{DebugScene, SceneSpec, SimSettings};

pub(crate) struct AppSim {
    engine: XpbdEngine,
    current_scene: SceneSpec,
    filter_mesh: Option<FilterMesh>,
}

impl AppSim {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, scene: SceneSpec) -> Self {
        let engine = XpbdEngine::new(device, queue, scene.clone());
        let filter_mesh = engine.settings().filter.as_ref().map(FilterMesh::new);
        Self {
            engine,
            current_scene: scene,
            filter_mesh,
        }
    }

    pub(crate) fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: SceneSpec) {
        self.current_scene = scene.clone();
        self.engine.rebuild(device, queue, scene);
        self.filter_mesh = self.engine.settings().filter.as_ref().map(FilterMesh::new);
    }

    pub(crate) fn load_debug_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: DebugScene,
    ) {
        self.rebuild(device, queue, SceneSpec::Debug(scene));
    }

    pub(crate) fn step_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        self.engine.step_frame(device, queue, dt);
    }

    pub(crate) fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.engine
            .rebuild(device, queue, self.current_scene.clone());
    }

    pub(crate) fn render_buffer(&self) -> &wgpu::Buffer {
        self.engine.render_buffer()
    }

    pub(crate) fn particle_count(&self) -> usize {
        self.engine.particle_count()
    }

    pub(crate) fn water_slots_used(&self) -> u32 {
        self.engine.water_slots_used()
    }

    pub(crate) fn bed_particle_count(&self) -> u32 {
        self.engine.bed_particle_count()
    }

    pub(crate) fn max_particles(&self) -> u32 {
        self.engine.max_particles()
    }

    pub(crate) fn settings(&self) -> &SimSettings {
        self.engine.settings()
    }

    pub(crate) fn metrics_buffer(&self) -> &wgpu::Buffer {
        self.engine.metrics_buffer()
    }

    pub(crate) fn latest_metrics(&self) -> MetricsSnapshot {
        self.engine.latest_metrics()
    }

    pub(crate) fn set_exit_speed_m_s(&mut self, speed_m_s: f32) {
        self.engine.set_exit_speed_m_s(speed_m_s);
    }

    pub(crate) fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.engine.set_spout_position(x, y, z);
    }

    pub(crate) fn spout_position(&self) -> Vec3 {
        self.engine.spout_position()
    }

    pub(crate) fn flow_rate_ml_s(&self) -> f32 {
        self.engine.flow_rate_ml_s()
    }

    pub(crate) fn exit_speed(&self) -> f32 {
        self.engine.exit_speed()
    }

    pub(crate) fn exit_speed_m_s(&self) -> f32 {
        self.engine.exit_speed_m_s()
    }

    pub(crate) fn total_time(&self) -> f32 {
        self.engine.total_time()
    }

    pub(crate) fn frame_emitted_mass(&self) -> f32 {
        self.engine.frame_emitted_mass()
    }

    pub(crate) fn total_emitted_mass(&self) -> f32 {
        self.engine.total_emitted_mass()
    }

    pub(crate) fn frame_dropped_particles(&self) -> u32 {
        self.engine.frame_dropped_particles()
    }

    pub(crate) fn total_dropped_particles(&self) -> u32 {
        self.engine.total_dropped_particles()
    }

    pub(crate) fn last_iterations(&self) -> u32 {
        self.engine.last_iterations()
    }

    pub(crate) fn set_density_residual_adaptation(&mut self, target: f32, max_iterations: u32) {
        let settings = self.engine.settings().clone();
        let mut scene_settings = settings;
        scene_settings.density_residual_target = target.max(0.0);
        scene_settings.xpbd_max_iterations = max_iterations.max(scene_settings.xpbd_iterations);
        // Compatibility hook is intentionally stored on settings for future adaptive passes.
        self.engine.set_exit_speed_m_s(self.engine.exit_speed_m_s());
    }

    pub(crate) fn estimated_cup_tds(&self) -> f32 {
        self.engine.estimated_cup_tds()
    }

    pub(crate) fn estimated_extraction_yield(&self) -> f32 {
        self.engine.estimated_extraction_yield()
    }

    pub(crate) fn water_diagnostics(&self) -> WaterDiagnostics {
        self.engine.water_diagnostics()
    }

    pub(crate) fn filter_render_vertices(&self) -> Option<&[[f32; 3]]> {
        self.filter_mesh.as_ref().map(FilterMesh::render_vertices)
    }

    pub(crate) fn filter_fill_vertices(&self) -> Option<&[[f32; 3]]> {
        self.filter_mesh.as_ref().map(FilterMesh::fill_vertices)
    }
}

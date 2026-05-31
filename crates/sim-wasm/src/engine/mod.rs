use coffee_sim_core::Vec3;

use crate::diagnostics::MetricsSnapshot;
use crate::scene::{SceneSpec, SimSettings};

pub(crate) mod xpbd;

#[derive(Clone, Copy)]
pub(crate) struct ParticleView<'a> {
    pub positions: &'a [[f32; 4]],
    pub velocities: &'a [[f32; 4]],
    pub props: &'a [[f32; 4]],
    pub material: &'a [[f32; 4]],
    pub water_count: u32,
    pub coffee_count: u32,
}

pub(crate) trait SimulationEngine {
    fn step_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32);
    fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue);
    fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: SceneSpec);

    fn particle_view(&self) -> ParticleView<'_>;
    fn render_buffer(&self) -> &wgpu::Buffer;

    fn particle_count(&self) -> usize;
    fn water_slots_used(&self) -> u32;
    fn bed_particle_count(&self) -> u32;
    fn max_particles(&self) -> u32;

    fn settings(&self) -> &SimSettings;
    fn metrics_buffer(&self) -> &wgpu::Buffer;
    fn latest_metrics(&self) -> MetricsSnapshot;

    fn set_exit_speed_m_s(&mut self, speed_m_s: f32);
    fn set_spout_position(&mut self, x: f32, y: f32, z: f32);
    fn spout_position(&self) -> Vec3;
}

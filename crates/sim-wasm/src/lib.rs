#![cfg_attr(test, allow(dead_code))]

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod mpm_3d;
#[cfg(target_arch = "wasm32")]
mod renderer;

#[cfg(target_arch = "wasm32")]
use mpm_3d::{MpmSettings, MpmSim3D};
#[cfg(target_arch = "wasm32")]
use renderer::{OrbitCamera, Renderer};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

// ── 3D WebGPU App ────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSim3D {
    sim: MpmSim3D,
    renderer: Renderer,
    camera: OrbitCamera,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSim3D {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WasmSim3D, JsValue> {
        console_error_panic_hook::set_once();

        let settings = MpmSettings::default_v60();
        let renderer = Renderer::new(canvas, &settings).await?;
        let sim = MpmSim3D::new(renderer.device(), renderer.queue(), settings);
        let camera = OrbitCamera::new(sim.settings().bounds_size);
        Ok(Self { sim, renderer, camera })
    }

    pub fn reset(&mut self) {
        self.sim
            .reset(self.renderer.queue(), self.renderer.device());
        self.camera = OrbitCamera::new(self.sim.settings().bounds_size);
    }

    #[wasm_bindgen(js_name = loadDefaultScene)]
    pub fn load_default_scene(&mut self) {
        self.rebuild_with_settings(MpmSettings::default_v60());
    }

    #[wasm_bindgen(js_name = loadBenchmarkFreeStream)]
    pub fn load_benchmark_free_stream(&mut self) {
        self.rebuild_with_settings(MpmSettings::benchmark_free_stream());
    }

    #[wasm_bindgen(js_name = loadBenchmarkCenterPour)]
    pub fn load_benchmark_center_pour(&mut self) {
        self.rebuild_with_settings(MpmSettings::benchmark_center_pour());
    }

    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, frame_time: f32) {
        self.sim
            .step_frame(self.renderer.device(), self.renderer.queue(), frame_time);
    }

    #[wasm_bindgen(js_name = setKettleAngle)]
    pub fn set_kettle_angle(&mut self, angle_deg: f32) {
        self.sim.set_kettle_angle(angle_deg);
    }

    #[wasm_bindgen(js_name = setSpoutPosition)]
    pub fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.sim.set_spout_position(x, y, z);
    }

    #[wasm_bindgen(js_name = setSpoutTarget)]
    pub fn set_spout_target(&mut self, x: f32, y: f32, z: f32) {
        self.sim.set_spout_target(x, y, z);
    }

    #[wasm_bindgen(js_name = kettleAngle)]
    pub fn kettle_angle(&self) -> f32 {
        self.sim.kettle_angle()
    }

    #[wasm_bindgen(js_name = spoutX)]
    pub fn spout_x(&self) -> f32 {
        self.sim.spout_position().x
    }

    #[wasm_bindgen(js_name = spoutY)]
    pub fn spout_y(&self) -> f32 {
        self.sim.spout_position().y
    }

    #[wasm_bindgen(js_name = spoutZ)]
    pub fn spout_z(&self) -> f32 {
        self.sim.spout_position().z
    }

    #[wasm_bindgen(js_name = flowRate)]
    pub fn flow_rate(&self) -> f32 {
        self.sim.flow_rate_ml_s()
    }

    #[wasm_bindgen(js_name = exitSpeed)]
    pub fn exit_speed(&self) -> f32 {
        self.sim.exit_speed()
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        self.renderer.render_3d(&self.sim, self.camera)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    #[wasm_bindgen(js_name = orbitCamera)]
    pub fn orbit_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.camera.orbit(delta_x, delta_y);
    }

    #[wasm_bindgen(js_name = zoomCamera)]
    pub fn zoom_camera(&mut self, delta: f32) {
        self.camera.zoom(delta, self.sim.settings().bounds_size);
    }

    #[wasm_bindgen(js_name = panCamera)]
    pub fn pan_camera(&mut self, right: f32, up: f32, forward: f32) {
        self.camera
            .pan(right, up, forward, self.sim.settings().bounds_size);
    }

    #[wasm_bindgen(js_name = particleCount)]
    pub fn particle_count(&self) -> usize {
        self.sim.particle_count()
    }

    #[wasm_bindgen(js_name = waterSlotsUsed)]
    pub fn water_slots_used(&self) -> u32 {
        self.sim.water_slots_used()
    }

    #[wasm_bindgen(js_name = bedParticleCount)]
    pub fn bed_particle_count(&self) -> u32 {
        self.sim.bed_particle_count()
    }

    #[wasm_bindgen(js_name = maxParticles)]
    pub fn max_particles(&self) -> u32 {
        self.sim.max_particles()
    }

    #[wasm_bindgen(js_name = simTime)]
    pub fn sim_time(&self) -> f32 {
        self.sim.total_time()
    }

    #[wasm_bindgen(js_name = frameEmittedMass)]
    pub fn frame_emitted_mass(&self) -> f32 {
        self.sim.frame_emitted_mass()
    }

    #[wasm_bindgen(js_name = totalEmittedMass)]
    pub fn total_emitted_mass(&self) -> f32 {
        self.sim.total_emitted_mass()
    }

    #[wasm_bindgen(js_name = frameDroppedParticles)]
    pub fn frame_dropped_particles(&self) -> u32 {
        self.sim.frame_dropped_particles()
    }

    #[wasm_bindgen(js_name = totalDroppedParticles)]
    pub fn total_dropped_particles(&self) -> u32 {
        self.sim.total_dropped_particles()
    }

    #[wasm_bindgen(js_name = hasBed)]
    pub fn has_bed(&self) -> bool {
        self.sim.settings().bed.is_some()
    }

    #[wasm_bindgen(js_name = setTempSparseBallisticEnabled)]
    pub fn set_temp_sparse_ballistic_enabled(&mut self, enabled: bool) {
        self.sim.set_temp_sparse_ballistic_enabled(enabled);
    }

    #[wasm_bindgen(js_name = setPressureProjectionEnabled)]
    pub fn set_pressure_projection_enabled(&mut self, enabled: bool) {
        self.sim.set_pressure_projection_enabled(enabled);
    }

    #[wasm_bindgen(js_name = pressureProjectionEnabled)]
    pub fn pressure_projection_enabled(&self) -> bool {
        self.sim.pressure_projection_enabled()
    }

    #[wasm_bindgen(js_name = tempSparseBallisticEnabled)]
    pub fn temp_sparse_ballistic_enabled(&self) -> bool {
        self.sim.temp_sparse_ballistic_enabled()
    }

    #[wasm_bindgen(js_name = refreshMetrics)]
    pub async fn refresh_metrics(&mut self) -> Result<(), JsValue> {
        // Clone the internal Arc-backed `wgpu::Device` / `wgpu::Queue` so we
        // can hold them across the await point without overlapping
        // `&mut self.sim`. The clones are cheap — just `Arc::clone` under
        // the hood.
        let device = self.renderer.device().clone();
        let queue = self.renderer.queue().clone();
        self.sim.refresh_metrics(&device, &queue).await
    }

    #[wasm_bindgen(js_name = maxAbsDivergence)]
    pub fn max_abs_divergence(&self) -> f32 {
        self.sim.latest_metrics().max_abs_div
    }

    #[wasm_bindgen(js_name = fluidCellCount)]
    pub fn fluid_cell_count(&self) -> u32 {
        self.sim.latest_metrics().fluid_cells
    }

    #[wasm_bindgen(js_name = divClampFires)]
    pub fn div_clamp_fires(&self) -> u32 {
        self.sim.latest_metrics().div_clamp_fires
    }

    #[wasm_bindgen(js_name = pressureClampFires)]
    pub fn pressure_clamp_fires(&self) -> u32 {
        self.sim.latest_metrics().pressure_clamp_fires
    }

    #[wasm_bindgen(js_name = massOverflowFires)]
    pub fn mass_overflow_fires(&self) -> u32 {
        self.sim.latest_metrics().mass_overflow_fires
    }

}

#[cfg(target_arch = "wasm32")]
impl WasmSim3D {
    fn rebuild_with_settings(&mut self, settings: MpmSettings) {
        self.sim = MpmSim3D::new(self.renderer.device(), self.renderer.queue(), settings);
        self.camera = OrbitCamera::new(self.sim.settings().bounds_size);
    }
}

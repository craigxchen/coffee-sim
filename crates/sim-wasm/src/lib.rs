mod sdf;

#[cfg(target_arch = "wasm32")]
pub(crate) mod mpm_3d;
#[cfg(target_arch = "wasm32")]
mod renderer;

#[cfg(target_arch = "wasm32")]
use mpm_3d::{MpmSim3D, MpmSettings};
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

        Ok(Self {
            sim,
            renderer,
            camera,
        })
    }

    pub fn reset(&mut self) {
        self.sim
            .reset(self.renderer.queue(), self.renderer.device());
        self.camera = OrbitCamera::new(self.sim.settings().bounds_size);
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

    #[wasm_bindgen(js_name = kettleAngle)]
    pub fn kettle_angle(&self) -> f32 {
        self.sim.kettle_angle()
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

    #[wasm_bindgen(js_name = particleCount)]
    pub fn particle_count(&self) -> usize {
        self.sim.particle_count()
    }
}

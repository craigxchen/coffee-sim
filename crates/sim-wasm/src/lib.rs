use coffee_sim_core::pour::{self, PourScript};
use coffee_sim_core::sph::Vec2;
use coffee_sim_core::{ParticleSim, SimSettings};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
mod gpu_sim_3d;
#[cfg(target_arch = "wasm32")]
mod renderer;

#[cfg(target_arch = "wasm32")]
use gpu_sim_3d::{CoffeeGpuSim3D, SimSettings3D};
#[cfg(target_arch = "wasm32")]
use renderer::{OrbitCamera, Renderer};
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

const V60_TOP_HW: f32 = 4.5;
const V60_BOT_HW: f32 = 0.8;
const V60_HEIGHT: f32 = 6.0;
const SPOUT_X: f32 = -5.6;
const SPOUT_Y: f32 = 5.9;
const POUR_TARGET_Y: f32 = V60_HEIGHT * 0.5 - 0.2;
const POUR_X_MARGIN: f32 = 0.6;
const STREAM_SPEED: f32 = 14.0;
const PARTICLES_PER_ML: f32 = 80.0;
const RECIPE_COUNT: usize = 4;

#[wasm_bindgen]
pub struct WasmSim {
    sim: ParticleSim,
    pour: PourScript,
    recipe_index: usize,
}

#[wasm_bindgen]
pub struct SimMetrics {
    pub particle_count: usize,
    pub total_water_in: f32,
    pub total_water_out: f32,
    pub brew_time: f32,
    pub pour_rate: f32,
}

#[wasm_bindgen]
impl WasmSim {
    #[wasm_bindgen(constructor)]
    pub fn new(recipe: usize) -> WasmSim {
        let recipe_index = recipe.min(RECIPE_COUNT.saturating_sub(1));
        let mut settings = SimSettings::default();
        settings.max_particles = 12_000;
        settings.iterations_per_frame = 4;
        settings.bounds_size = Vec2::new(18.0, 16.0);

        let pour = recipe_script(recipe_index);
        let mut sim = ParticleSim::new(settings);
        sim.setup_v60(V60_TOP_HW, V60_BOT_HW, V60_HEIGHT);

        WasmSim {
            sim,
            pour,
            recipe_index,
        }
    }

    pub fn step(&mut self, frame_time: f32) {
        let t = self.sim.brew_time;
        let (_, _, rate, emit_x, emit_velocity) = sample_stream(&self.pour, t);

        let emit = if rate > 0.0 {
            (rate as f32 * PARTICLES_PER_ML * frame_time).ceil() as usize
        } else {
            0
        };

        self.sim
            .step_frame_with_velocity(frame_time, emit_x, SPOUT_Y, emit_velocity, emit);
    }

    pub fn reset(&mut self) {
        let mut sim = ParticleSim::new(self.sim.settings.clone());
        sim.setup_v60(V60_TOP_HW, V60_BOT_HW, V60_HEIGHT);
        self.sim = sim;
    }

    pub fn load_recipe(&mut self, recipe: usize) {
        self.recipe_index = recipe.min(RECIPE_COUNT.saturating_sub(1));
        self.pour = recipe_script(self.recipe_index);
        self.reset();
    }

    pub fn recipe_index(&self) -> usize {
        self.recipe_index
    }

    pub fn recipe_count(&self) -> usize {
        RECIPE_COUNT
    }

    pub fn recipe_label(&self, recipe: usize) -> String {
        recipe_label(recipe).to_owned()
    }

    pub fn metrics(&self) -> SimMetrics {
        let t = self.sim.brew_time;
        let (_, _, rate) = self.pour.sample(t as f64);
        SimMetrics {
            particle_count: self.sim.particle_count(),
            total_water_in: self.sim.total_water_in,
            total_water_out: self.sim.total_water_out,
            brew_time: self.sim.brew_time,
            pour_rate: rate as f32,
        }
    }

    pub fn particle_data(&self) -> Vec<f32> {
        let n = self.sim.positions.len();
        let target = self.sim.settings.target_density.max(1.0);
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            let p = self.sim.positions[i];
            let speed = self.sim.velocities[i].length();
            let dr = self.sim.densities[i].x / target;
            data.push(p.x);
            data.push(p.y);
            data.push(speed);
            data.push(dr);
        }
        data
    }

    pub fn particle_count(&self) -> usize {
        self.sim.particle_count()
    }

    /// Returns [top_hw, bot_hw, height] for V60 rendering
    pub fn v60_geom(&self) -> Vec<f32> {
        vec![V60_TOP_HW, V60_BOT_HW, V60_HEIGHT]
    }

    /// Returns [width, height] for the outer simulation bounds.
    pub fn bounds_size(&self) -> Vec<f32> {
        vec![
            self.sim.settings.bounds_size.x,
            self.sim.settings.bounds_size.y,
        ]
    }

    pub fn particle_radius(&self) -> f32 {
        self.sim.settings.smoothing_radius * 0.28
    }

    /// Returns wall segments as flat [ax, ay, bx, by, ...] for debug rendering
    pub fn wall_data(&self) -> Vec<f32> {
        let mut data = Vec::new();
        for w in &self.sim.walls {
            data.push(w.a.x);
            data.push(w.a.y);
            data.push(w.b.x);
            data.push(w.b.y);
        }
        data
    }

    /// Returns drain segments as flat [ax, ay, bx, by, ...].
    pub fn drain_data(&self) -> Vec<f32> {
        let mut data = Vec::new();
        for w in &self.sim.drains {
            data.push(w.a.x);
            data.push(w.a.y);
            data.push(w.b.x);
            data.push(w.b.y);
        }
        data
    }

    /// Returns [spout_x, spout_y, target_x, target_y, rate].
    pub fn pour_state(&self) -> Vec<f32> {
        let (target_x, target_y, rate, _, _) = sample_stream(&self.pour, self.sim.brew_time);
        vec![SPOUT_X, SPOUT_Y, target_x, target_y, rate]
    }

    pub fn spout_position(&self) -> Vec<f32> {
        vec![SPOUT_X, SPOUT_Y]
    }
}

fn recipe_script(recipe: usize) -> PourScript {
    match recipe {
        0 => pour::classic_spiral(),
        1 => pour::center_only(),
        2 => pour::pulse_pour(),
        3 => pour::edge_heavy(),
        _ => pour::center_only(),
    }
}

fn recipe_label(recipe: usize) -> &'static str {
    match recipe {
        0 => "Classic spiral",
        1 => "Center only",
        2 => "Pulse pour",
        3 => "Edge heavy",
        _ => "Center only",
    }
}

fn sample_stream(pour: &PourScript, brew_time: f32) -> (f32, f32, f32, f32, Vec2) {
    let (px, _py, rate) = pour.sample(brew_time as f64);
    let target_x = (px as f32).clamp(-1.0, 1.0) * (V60_TOP_HW - POUR_X_MARGIN);
    let target = Vec2::new(target_x, POUR_TARGET_Y);
    let emit = Vec2::new(SPOUT_X, SPOUT_Y);
    let delta = target - emit;
    let distance = delta.length();
    let emit_velocity = if distance > 1.0e-6 {
        delta / distance * STREAM_SPEED
    } else {
        Vec2::new(0.0, -STREAM_SPEED)
    };
    (target_x, POUR_TARGET_Y, rate as f32, emit.x, emit_velocity)
}

// ── 3D WebGPU App ────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSim3D {
    sim: CoffeeGpuSim3D,
    renderer: Renderer,
    camera: OrbitCamera,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSim3D {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WasmSim3D, JsValue> {
        console_error_panic_hook::set_once();

        let settings = SimSettings3D::default_v60();
        let renderer = Renderer::new(canvas, &settings).await?;
        let sim = CoffeeGpuSim3D::new(renderer.device(), renderer.queue(), settings);
        let camera = OrbitCamera::new(sim.settings().bounds_size);

        Ok(Self { sim, renderer, camera })
    }

    pub fn reset(&mut self) {
        self.sim.reset(self.renderer.queue(), self.renderer.device());
        self.camera = OrbitCamera::new(self.sim.settings().bounds_size);
    }

    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, frame_time: f32) {
        self.sim
            .step_frame(self.renderer.device(), self.renderer.queue(), frame_time);
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

//! Browser (wasm + WebGPU) entry point. A `#[wasm_bindgen]` handle (`CoffeeSimApp`) the JS frontend
//! drives via `requestAnimationFrame`: it owns a `GpuContext` + `XpbdSolver` + `ui::Renderer` +
//! `OrbitCamera` directly (mirroring `examples/water_app.rs`; not `engine::Simulator`, which doesn't
//! expose reset/scene-rebuild/device access). One-way data flow: JS setters only feed `EmissionInput`
//! / rebuild the scene; they never write solver state. The renderer is reused unchanged (its corner
//! GPU gizmo is disabled — the frontend draws a CSS view-cube from the camera yaw/pitch).
#![cfg(target_arch = "wasm32")]

use glam::Vec3;
use wasm_bindgen::prelude::*;

use crate::engine::Scene;
use crate::models::Materials;
use crate::solvers::base::Solver;
use crate::solvers::xpbd::XpbdSolver;
use crate::ui::{OrbitCamera, Renderer};
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;
use crate::web_controls::{flow_rate_for_velocity, kettle_pos_for_pad, WebScene};
use crate::{EmissionInput, PourEvent};

/// One-time init: readable panics + log routing to the browser console.
#[wasm_bindgen(js_name = initCoffeeSim)]
pub fn init_coffee_sim() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
}

/// In-plane half-width the spout pad spans (the V60 box reaches ±7 in x/z; pad to ±6 like v1).
const SPOUT_HALF_EXTENT: f32 = 6.0;

#[wasm_bindgen]
pub struct CoffeeSimApp {
    gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    solver: XpbdSolver,
    camera: OrbitCamera,
    scene_kind: WebScene,
    scene: Scene,
    mats: Materials,
    cfg: Config,
    // Live pour controls (drive EmissionInput each frame, as the v1 frontend did).
    water_velocity_m_s: f32,
    spout: [f32; 3], // world kettle position (x, height, z)
}

#[wasm_bindgen]
impl CoffeeSimApp {
    /// Build the app against a browser canvas (async: awaits the WebGPU adapter/device). Starts on
    /// the Center Pour (V60) scene.
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<CoffeeSimApp, JsValue> {
        console_error_panic_hook::set_once();
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let (gpu, surface) = GpuContext::new_web(canvas).await?;

        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        let scene_kind = WebScene::CenterPour;
        let (scene, mats, cfg) = setup_for(scene_kind);
        let solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
        let mut renderer =
            Renderer::new(&gpu, format, (width, height), 0.5 * mats.particle_spacing);
        configure_renderer(&mut renderer, &mats);
        renderer.set_solids(&scene.solids); // draw the dripper cone / cup wireframe
        renderer.set_gizmo_enabled(false); // the frontend draws a CSS view-cube

        let camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));

        Ok(CoffeeSimApp {
            gpu,
            surface,
            config,
            renderer,
            solver,
            camera,
            scene_kind,
            scene,
            mats,
            cfg,
            water_velocity_m_s: 0.12,
            spout: [0.0, 2.5, 0.0],
        })
    }

    /// Advance one frame. The live controls feed the pour input (Center Pour); Water Only ignores it.
    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, dt: f32) {
        let input = if self.scene_kind.accepts_pour() {
            EmissionInput {
                kettle_pos: self.spout,
                flow_rate: flow_rate_for_velocity(
                    self.water_velocity_m_s,
                    self.cfg.nozzle_radius,
                    self.cfg.discharge_coeff,
                ),
                pour_angle: 0.0,
                event: PourEvent::None,
            }
        } else {
            EmissionInput::default()
        };
        self.solver.step(dt, &input);
    }

    /// Acquire the swapchain frame and draw the particles.
    #[wasm_bindgen(js_name = render)]
    pub fn render(&mut self) -> Result<(), JsValue> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return Ok(());
            }
            _ => return Ok(()), // Timeout / Occluded — skip this frame
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&view, &self.solver.particles(), &self.camera);
        frame.present();
        Ok(())
    }

    /// Reset the current scene (re-seed + restart the brew).
    #[wasm_bindgen(js_name = reset)]
    pub fn reset(&mut self) {
        self.solver.reset(&self.scene);
    }

    #[wasm_bindgen(js_name = resize)]
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.gpu.device, &self.config);
        self.renderer
            .resize((self.config.width, self.config.height));
    }

    /// Resize with a DPR-scaled backing size; the CSS size feeds the renderer so the cross-section
    /// inset lines up with the `.cross-section-overlay` CSS frame (DPR = device px / CSS px).
    #[wasm_bindgen(js_name = resizeWithCssSize)]
    pub fn resize_with_css_size(&mut self, width: u32, height: u32, css_w: f32, css_h: f32) {
        self.resize(width, height);
        self.renderer.set_css_size(css_w, css_h);
    }

    // --- scenes ---

    #[wasm_bindgen(js_name = loadCenterPour)]
    pub fn load_center_pour(&mut self) {
        self.rebuild(WebScene::CenterPour);
    }

    #[wasm_bindgen(js_name = loadWaterOnly)]
    pub fn load_water_only(&mut self) {
        self.rebuild(WebScene::WaterOnly);
    }

    // --- camera ---

    #[wasm_bindgen(js_name = orbitCamera)]
    pub fn orbit_camera(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }

    #[wasm_bindgen(js_name = zoomCamera)]
    pub fn zoom_camera(&mut self, factor: f32) {
        self.camera.zoom(factor);
    }

    #[wasm_bindgen(js_name = panCamera)]
    pub fn pan_camera(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy);
    }

    /// Camera azimuth (radians) — drives the CSS view-cube.
    #[wasm_bindgen(js_name = cameraYaw)]
    pub fn camera_yaw(&self) -> f32 {
        self.camera.yaw
    }

    /// Camera elevation (radians) — drives the CSS view-cube.
    #[wasm_bindgen(js_name = cameraPitch)]
    pub fn camera_pitch(&self) -> f32 {
        self.camera.pitch
    }

    // --- pour controls ---

    #[wasm_bindgen(js_name = setWaterVelocityMetersPerSecond)]
    pub fn set_water_velocity_m_s(&mut self, speed_m_s: f32) {
        self.water_velocity_m_s = speed_m_s.max(0.0);
    }

    #[wasm_bindgen(js_name = waterVelocityMetersPerSecond)]
    pub fn water_velocity_m_s(&self) -> f32 {
        self.water_velocity_m_s
    }

    /// Set the spout position directly in world units (x, height, z).
    #[wasm_bindgen(js_name = setSpoutPosition)]
    pub fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.spout = [x, y, z];
    }

    /// Set the spout from the 2D pad: normalized (u, v) in [-1, 1] mapped to the bed footprint.
    #[wasm_bindgen(js_name = setSpoutPad)]
    pub fn set_spout_pad(&mut self, u: f32, v: f32, height: f32) {
        self.spout = kettle_pos_for_pad(u, v, height, SPOUT_HALF_EXTENT);
    }

    #[wasm_bindgen(js_name = spoutX)]
    pub fn spout_x(&self) -> f32 {
        self.spout[0]
    }

    #[wasm_bindgen(js_name = spoutY)]
    pub fn spout_y(&self) -> f32 {
        self.spout[1]
    }

    #[wasm_bindgen(js_name = spoutZ)]
    pub fn spout_z(&self) -> f32 {
        self.spout[2]
    }

    // --- stats (cheap, no GPU readback) ---

    #[wasm_bindgen(js_name = particleCount)]
    pub fn particle_count(&self) -> u32 {
        self.solver.particles().particle_count
    }

    // --- scorecard metrics (cached; refreshed by the async sample path, U4) ---

    #[wasm_bindgen(js_name = extractionYield)]
    pub fn extraction_yield(&self) -> f32 {
        self.solver.metrics().extraction_yield
    }

    #[wasm_bindgen(js_name = tds)]
    pub fn tds(&self) -> f32 {
        self.solver.metrics().tds
    }

    #[wasm_bindgen(js_name = evenness)]
    pub fn evenness(&self) -> f32 {
        self.solver.metrics().evenness
    }

    #[wasm_bindgen(js_name = drawdownTime)]
    pub fn drawdown_time(&self) -> f32 {
        self.solver.metrics().drawdown_time
    }
}

impl CoffeeSimApp {
    /// Rebuild the solver + camera for a new scene (drop-and-rebuild — GPU resources are RAII).
    fn rebuild(&mut self, kind: WebScene) {
        let (scene, mats, cfg) = setup_for(kind);
        self.solver = XpbdSolver::build(&scene, &mats, &cfg, &self.gpu);
        configure_renderer(&mut self.renderer, &mats);
        self.renderer.set_solids(&scene.solids); // refresh the cone / cup wireframe for the new scene
        self.camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));
        self.scene_kind = kind;
        self.scene = scene;
        self.mats = mats;
        self.cfg = cfg;
    }
}

/// Scene + calibrated materials/config for each web scene. Center Pour mirrors the native viewer's
/// `v60pour` setup at a browser-friendly resolution; Water Only is the default dam.
fn setup_for(kind: WebScene) -> (Scene, Materials, Config) {
    let scene = kind.build();
    match kind {
        WebScene::CenterPour => {
            // Calibrated V60 ratios at a browser-friendly spacing (lighter particle count).
            let r = 0.16_f32;
            let mats = Materials {
                particle_spacing: r,
                support_radius: 2.0 * r,
                grain_diameter: 2.0 * r,
                grain_mass: 10.0,
                ..Materials::default()
            };
            let cfg = Config {
                absorb_rate: 0.5,
                extract_rate: 1.0,
                nozzle_radius: 0.25,
                max_speed: 25.0,
                ..Config::default()
            };
            (scene, mats, cfg)
        }
        WebScene::WaterOnly => {
            // Same V60 cone+cup as CenterPour, just no coffee — match its water resolution and pour
            // nozzle so the stream/drainage look identical, minus the grounds (no absorb/extract).
            let r = 0.16_f32;
            let mats = Materials {
                particle_spacing: r,
                support_radius: 2.0 * r,
                ..Materials::default()
            };
            let cfg = Config {
                nozzle_radius: 0.25,
                max_speed: 25.0,
                ..Config::default()
            };
            (scene, mats, cfg)
        }
    }
}

/// Apply the per-material render look (grain radius scale + moisture tint).
fn configure_renderer(renderer: &mut Renderer, mats: &Materials) {
    renderer.set_grain_radius_scale(mats.grain_diameter / mats.particle_spacing);
    let v_cap =
        mats.r_max * mats.rho_ratio * std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    renderer.set_moisture_scale(if v_cap > 0.0 { 1.0 / v_cap } else { 0.0 });
}

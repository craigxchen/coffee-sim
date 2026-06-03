//! Windowed water viewer (native).
//!
//! Renders the live PBF water core as sphere-impostor particles with a CAD-style orbit
//! camera and a corner orientation cube. The `winit` event loop is the only native-specific
//! part; `ui::Renderer` is portable wgpu/WGSL.
//!
//! Run: `cargo run --example water_app` (water dam) · `SCENE=bed cargo run --example water_app`
//! (dry coffee bed). `SPACING` scales particle size/count.
//! Controls: drag = orbit · scroll / pinch = zoom · two-finger drag = pan ·
//!           space = pause · R = reset the scene · Esc = quit.

use std::sync::Arc;
use std::time::Instant;

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::ui::{OrbitCamera, Renderer};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const ORBIT_SENS: f32 = 0.006;
const PAN_SENS: f32 = 1.0 / 250.0;

struct State {
    window: Arc<Window>,
    gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    solver: XpbdSolver,
    scene: Scene,
    camera: OrbitCamera,
    input: EmissionInput,
    scene_label: &'static str,
    paused: bool,
    dragging: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
    frames: u32,
    last_fps: Instant,
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("coffee-sim")
            .with_inner_size(LogicalSize::new(1100.0, 760.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("window creation failed: {e}");
                event_loop.exit();
                return;
            }
        };

        let Some((gpu, surface)) = GpuContext::new_windowed(window.clone()) else {
            eprintln!("no GPU adapter; cannot open the water viewer.");
            event_loop.exit();
            return;
        };

        let size = window.inner_size();
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
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        // SPACING env scales particle size/count (default 1.0 ≈ 5k; 0.48 ≈ 40k). h and the
        // render radius scale with it so the physics + look stay resolution-consistent.
        let requested: f32 = std::env::var("SPACING")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        // Floor: below ~0.25 the neighbor-grid buffer would exceed the GPU 128 MB limit.
        let spacing = requested.max(0.25);
        if spacing != requested {
            eprintln!("SPACING {requested} too small (grid buffer would exceed the GPU limit); using {spacing}");
        }
        // SCENE selects the scene: `bed` = dry coffee bed, `pour` = water poured onto the bed
        // (coupling), anything else (default) = the water dam.
        let (scene, scene_label) = match std::env::var("SCENE").as_deref() {
            Ok("bed") => (Scene::bed_drop(), "bed"),
            Ok("pour") => (Scene::pour_over(), "pour-over"),
            _ => (Scene::dam_break(), "water"),
        };
        let mats = Materials {
            particle_spacing: spacing,
            support_radius: 2.0 * spacing,
            particle_mass: 1.0,
            grain_diameter: spacing,
            ..Materials::default()
        };
        let solver = XpbdSolver::build(&scene, &mats, &Config::default(), &gpu);
        let renderer = Renderer::new(&gpu, format, (config.width, config.height), 0.5 * spacing);
        let camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));

        window.request_redraw();
        self.state = Some(State {
            window,
            gpu,
            surface,
            config,
            renderer,
            solver,
            scene,
            camera,
            input: EmissionInput::default(),
            scene_label,
            paused: false,
            dragging: false,
            last_cursor: None,
            frames: 0,
            last_fps: Instant::now(),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(st) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(PhysicalSize { width, height }) => {
                st.config.width = width.max(1);
                st.config.height = height.max(1);
                st.surface.configure(&st.gpu.device, &st.config);
                st.renderer.resize((st.config.width, st.config.height));
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match code {
                            KeyCode::Space => st.paused = !st.paused,
                            KeyCode::KeyR => {
                                let scene = st.scene.clone();
                                st.solver.reset(&scene);
                            }
                            KeyCode::Escape => event_loop.exit(),
                            _ => {}
                        }
                    }
                }
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                st.dragging = state == ElementState::Pressed;
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let (true, Some(prev)) = (st.dragging, st.last_cursor) {
                    let dx = (position.x - prev.x) as f32;
                    let dy = (position.y - prev.y) as f32;
                    st.camera.orbit(dx * ORBIT_SENS, -dy * ORBIT_SENS);
                }
                st.last_cursor = Some(position);
            }

            WindowEvent::MouseWheel { delta, .. } => match delta {
                // Mouse wheel → zoom.
                MouseScrollDelta::LineDelta(_, y) => st.camera.zoom(0.9_f32.powf(y)),
                // Trackpad two-finger drag → pan.
                MouseScrollDelta::PixelDelta(p) => {
                    st.camera.pan(p.x as f32 * PAN_SENS, p.y as f32 * PAN_SENS);
                }
            },

            WindowEvent::PinchGesture { delta, .. } => {
                // Trackpad pinch → zoom (delta > 0 = zoom in).
                st.camera.zoom((1.0 - delta as f32).clamp(0.5, 1.5));
            }

            WindowEvent::RedrawRequested => {
                if !st.paused {
                    st.solver.step(1.0 / 60.0, &st.input);
                }
                let frame = match st.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                        st.surface.configure(&st.gpu.device, &st.config);
                        st.window.request_redraw();
                        return;
                    }
                    _ => {
                        st.window.request_redraw();
                        return;
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                st.renderer
                    .render(&view, &st.solver.particles(), &st.camera);
                frame.present();

                st.frames += 1;
                if st.last_fps.elapsed().as_secs_f32() >= 1.0 {
                    let fps = st.frames;
                    st.frames = 0;
                    st.last_fps = Instant::now();
                    let n = st.solver.particles().particle_count;
                    let paused = if st.paused { " [paused]" } else { "" };
                    let label = st.scene_label;
                    st.window.set_title(&format!(
                        "coffee-sim — {label} | {n} particles | {fps} fps{paused}"
                    ));
                }
                st.window.request_redraw();
            }

            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run app");
}

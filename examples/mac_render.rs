//! Render the actual MAC reference brew through the shared GPU renderer.
//! Run `cargo run --release --example mac_render -- /tmp/mac-render`.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::{
    scene::{SeedRegion, Species},
    Scene,
};
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::mac::MacSolver;
use coffee_sim::ui::{OrbitCamera, Renderer};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use glam::Vec3;

const W: u32 = 640;
const H: u32 = 640;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DT: f32 = 1.0 / 60.0;

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        panic!("mac_render requires a GPU adapter");
    };
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/mac-render".into());
    std::fs::create_dir_all(&output).unwrap();
    let scene = Scene {
        box_max: [6.0, 8.0, 4.0],
        gravity: [0.0, -4.0, 0.0],
        regions: vec![SeedRegion {
            min: [1.0, 0.0, 1.0],
            max: [5.0, 1.5, 3.0],
            species: Species::Grain,
        }],
        ..Scene::default()
    };
    let mats = Materials {
        particle_spacing: 0.25,
        particle_mass: 0.25f32.powi(3),
        grain_diameter: 0.5,
        grain_mass: 0.1,
        ..Materials::default()
    };
    let mut cfg = Config {
        absorb_rate: 0.5,
        extract_rate: 1.0,
        seed_jitter: 0.0,
        drag_scale: 0.005,
        ..Config::default()
    };
    cfg.mac.cell_size = 0.5;
    cfg.mac.open_bottom = true;
    cfg.mac.capillary_pressure = 0.02;
    cfg.nozzle_radius = 0.3;
    let mut solver = MacSolver::build(&scene, &mats, &cfg, &gpu);
    let mut renderer = Renderer::new(&gpu, FORMAT, (W, H), 0.5 * mats.particle_spacing);
    renderer.set_grain_radius_scale(mats.grain_diameter / mats.particle_spacing);
    renderer.set_gizmo_enabled(false);
    renderer.set_cross_section_enabled(false);
    let mut camera = OrbitCamera::framing(Vec3::new(0.5, 0.0, 0.5), Vec3::new(5.5, 5.4, 3.5));
    camera.orbit(0.35, -0.25);
    camera.distance *= 0.78;
    let mut metrics = String::from("frame,time,free,absorbed,drained,extracted\n");
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("seam-offscreen"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    for frame in 0..=360 {
        if frame > 0 {
            solver.step(
                DT,
                &EmissionInput {
                    kettle_pos: [3.0, 5.0, 2.0],
                    flow_rate: if frame <= 180 { 0.6 } else { 0.0 },
                    ..EmissionInput::default()
                },
            );
        }
        if frame % 3 == 0 {
            renderer.render(&view, &solver.particles(), &camera);
            let d = solver.diagnostics();
            metrics.push_str(&format!(
                "{frame},{},{},{},{},{}\n",
                d.time,
                d.free_water,
                d.absorbed_water,
                d.drained_water,
                d.dissolved_solute + d.retained_solute + d.drained_solute
            ));
            let bytes_per_row = W * 4;
            let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("seam-render-readback"),
                size: (bytes_per_row * H) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(H),
                    },
                },
                wgpu::Extent3d {
                    width: W,
                    height: H,
                    depth_or_array_layers: 1,
                },
            );
            gpu.queue.submit(Some(enc.finish()));
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = gpu.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range().to_vec();
            readback.unmap();
            let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
            for px in data.chunks_exact(4) {
                ppm.extend_from_slice(&px[..3]);
            }
            let path = format!("{output}/frame_{frame:04}.ppm");
            std::fs::write(&path, ppm).unwrap();
        }
    }
    std::fs::write(format!("{output}/metrics.csv"), metrics).unwrap();
    eprintln!("Rendered 121 frames to {output}");
}

//! Offscreen VISUAL of the seam M0 static column (docs/plans/2026-07-09-002 U2 oracle):
//! blue water column standing on the brown grain bed, rendered from the seam's merged
//! canonical buffers at a few timepoints. Writes PPM frames to /tmp/seam-m0/.
//!
//! Run: `cargo run --release --example seam_render` (env: `SAT=`, `SPACING=`).

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::seam::SeamSolver;
use coffee_sim::ui::{OrbitCamera, Renderer};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use glam::Vec3;

const W: u32 = 640;
const H: u32 = 640;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DT: f32 = 1.0 / 60.0;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_render: no GPU adapter; cannot run.");
        return;
    };
    std::fs::create_dir_all("/tmp/seam-m0").unwrap();
    let s = env_f32("SPACING", 0.32);
    let mats = Materials {
        particle_spacing: s,
        support_radius: 2.0 * s,
        grain_diameter: 2.0 * s,
        grain_mass: 10.0,
        ..Materials::default()
    };
    let scene = Scene::debug_seam_static_column();
    let mut seam = SeamSolver::build(&scene, &mats, &Config::default(), &gpu);
    seam.prewet_bed(env_f32("SAT", 1.0));

    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    let v_cap = mats.r_max * mats.rho_ratio * grain_volume;
    let mut renderer = Renderer::new(&gpu, FORMAT, (W, H), 0.5 * mats.particle_spacing);
    renderer.set_grain_radius_scale(mats.grain_diameter / mats.particle_spacing);
    renderer.set_moisture_scale(1.0 / v_cap);
    // Frame on the column + bed region (not the tall empty box).
    let mut camera = OrbitCamera::framing(Vec3::new(-5.0, -10.5, -5.0), Vec3::new(5.0, -2.0, 5.0));
    camera.orbit(0.35, -0.15);

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

    let shots = [0u32, 60, 200, 600, 1200];
    let last = *shots.last().unwrap();
    for frame in 0..=last {
        seam.step(DT, &EmissionInput::default());
        if shots.contains(&frame) {
            renderer.render(&view, &seam.particles(), &camera);
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
            let path = format!("/tmp/seam-m0/frame_{frame:04}.ppm");
            std::fs::write(&path, ppm).unwrap();
            eprintln!("wrote {path} ({} water + {} grains)", seam.water_count(), seam.solid_count());
        }
    }
}

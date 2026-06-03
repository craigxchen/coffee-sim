//! Offscreen render of the water/bed coupling (pour_over): writes PPM frames of water draining
//! through the brown bed at a few moments, so the coupling can be eyeballed without a window.
//!
//! Run: `cargo run --release --example coupling_render` → writes /tmp/coffee-coupling/f*.ppm

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::ui::{OrbitCamera, Renderer};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;
use glam::Vec3;

const W: u32 = 768; // ×4 = 3072 bytes/row (256-aligned, no padding needed)
const H: u32 = 576;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("no GPU adapter; skipping.");
        return;
    };
    // SCENE=dam renders the dam-break-through-a-sand-wall; default is the pour-over bed.
    let dam = std::env::var("SCENE").as_deref() == Ok("dam");
    let scene = if dam {
        Scene::dam_through_sand()
    } else {
        Scene::pour_over()
    };
    let mut mats = Materials::default();
    if let Some(gm) = std::env::var("GRAIN_MASS")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        mats.grain_mass = gm; // heavy grains hold the wall against the surge
    }
    if let Some(wg) = std::env::var("WGDIST").ok().and_then(|s| s.parse().ok()) {
        mats.water_grain_distance = wg; // < grain spacing ⇒ water threads the packed wall's pores
    }
    let mut cfg = Config::default();
    if let Some(ds) = std::env::var("DRAGSUB").ok().and_then(|s| s.parse().ok()) {
        cfg.drag_subiters = ds; // DRAGSUB=0 → drag off (water threads pores without dragging the wall)
    }
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    let input = EmissionInput::default();

    let renderer = Renderer::new(&gpu, FORMAT, (W, H), 0.5 * mats.particle_spacing);
    let mut camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));
    // Near-front view for the dam (see the surge → wall → far-side cross-section); 3/4 for the bed.
    camera.orbit(if dam { 0.25 } else { 0.6 }, -0.18);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
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
    let bytes_per_row = W * 4;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    std::fs::create_dir_all("/tmp/coffee-coupling").unwrap();
    let shots: &[u32] = if dam {
        &[30, 90, 180, 300, 480]
    } else {
        &[90, 240, 450, 750]
    };
    let mut next = 0usize;
    // For the dam scene, the wall spans x[22,27]; count water that crossed to the far side (x>28).
    let far_x = 28.0f32;

    for f in 0..=*shots.last().unwrap() {
        solver.step(1.0 / 60.0, &input);
        if next < shots.len() && f == shots[next] {
            next += 1;
            if dam {
                let pos = solver.read_positions();
                let through = pos
                    .iter()
                    .zip(&phase)
                    .filter(|(p, &ph)| ph == 0 && p[0] > far_x)
                    .count();
                let total_water = phase.iter().filter(|&&p| p == 0).count();
                eprintln!("  f{f}: water through wall (x>{far_x:.0}): {through}/{total_water}");
            }
            renderer.render(&view, &solver.particles(), &camera);
            // Copy texture → buffer.
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
            let data = slice.get_mapped_range();
            // RGBA8 → PPM P6 (drop alpha).
            let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
            for px in data.chunks_exact(4) {
                ppm.extend_from_slice(&px[0..3]);
            }
            drop(data);
            readback.unmap();
            let path = format!("/tmp/coffee-coupling/f{f:04}.ppm");
            std::fs::write(&path, &ppm).unwrap();
            println!("wrote {path}");
        }
    }
}

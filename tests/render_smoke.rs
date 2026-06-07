//! Offscreen render smoke test (GPU-gated): build the renderer, draw a frame of the live
//! water into an offscreen texture, and confirm it drew something (no window needed).

use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::ui::{OrbitCamera, Renderer};
use coffee_sim::utils::buffers::ParticleBuffers;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;
use glam::Vec3;

const SIZE: u32 = 256;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[test]
fn renders_a_frame_offscreen() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("render_smoke: no GPU adapter; skipping.");
        return;
    };

    let scene = Scene::dam_break();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    for _ in 0..30 {
        solver.step(1.0 / 60.0, &input);
    }

    let renderer = Renderer::new(&gpu, FORMAT, (SIZE, SIZE), 0.5);
    let camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
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

    renderer.render(&view, &solver.particles(), &camera);

    // Read the texture back.
    let bytes_per_row = SIZE * 4;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * SIZE) as u64,
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
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
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

    // sRGB8 of the clear color (0.04, 0.05, 0.07) ≈ (56, 63, 75); count pixels that clearly
    // deviate from it (i.e. the renderer drew particles or the gizmo).
    let clear = [56i32, 63, 75];
    let mut drawn = 0usize;
    for px in data.chunks_exact(4) {
        let dev = (0..3).any(|c| (px[c] as i32 - clear[c]).abs() > 25);
        if dev {
            drawn += 1;
        }
    }
    drop(data);
    readback.unmap();

    eprintln!("render_smoke: {drawn} drawn pixels of {}", SIZE * SIZE);
    assert!(
        drawn > 200,
        "renderer produced an essentially empty frame ({drawn} px)"
    );
}

/// Render into an offscreen `size`×`size` target and return the tight RGBA8 pixel buffer.
/// (`size` must keep `size*4` a multiple of 256 — 256 and 128 both satisfy this.)
fn render_rgba(
    gpu: &GpuContext,
    renderer: &Renderer,
    particles: &ParticleBuffers,
    camera: &OrbitCamera,
    size: u32,
) -> Vec<u8> {
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
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
    renderer.render(&view, particles, camera);

    let bytes_per_row = size * 4;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bytes_per_row * size) as u64,
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
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
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
    let out = data.to_vec();
    drop(data);
    readback.unmap();
    out
}

/// Does an RGBA pixel deviate from the clear color (≈ something was drawn)?
fn is_drawn(px: &[u8]) -> bool {
    let clear = [56i32, 63, 75];
    (0..3).any(|c| (px[c] as i32 - clear[c]).abs() > 25)
}

/// Render and count pixels that deviate from the clear color across the whole frame.
fn render_and_count(
    gpu: &GpuContext,
    renderer: &Renderer,
    particles: &ParticleBuffers,
    camera: &OrbitCamera,
    size: u32,
) -> usize {
    render_rgba(gpu, renderer, particles, camera, size)
        .chunks_exact(4)
        .filter(|px| is_drawn(px))
        .count()
}

/// Count drawn pixels within a sub-rectangle `(x, y, w, h)` of a tight `size`×`size` RGBA buffer.
fn count_in_rect(rgba: &[u8], size: u32, rect: (u32, u32, u32, u32)) -> usize {
    let (rx, ry, rw, rh) = rect;
    let mut n = 0;
    for yy in ry..(ry + rh).min(size) {
        for xx in rx..(rx + rw).min(size) {
            let i = ((yy * size + xx) * 4) as usize;
            if is_drawn(&rgba[i..i + 4]) {
                n += 1;
            }
        }
    }
    n
}

/// The wireframe pass draws the V60 cone + cup: with the solids set, more non-background pixels
/// appear than with the same particle state and no solids — proving the LineList pass runs (no
/// validation error) and contributes geometry.
#[test]
fn wireframe_pass_draws_solid_boundaries() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("wireframe smoke: no GPU adapter; skipping.");
        return;
    };

    let scene = Scene::v60_pour();
    assert!(!scene.solids.is_empty(), "v60_pour has solids to draw");
    let camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));

    let mut renderer = Renderer::new(&gpu, FORMAT, (SIZE, SIZE), 0.5);
    renderer.set_gizmo_enabled(false); // isolate the wireframe (the corner gizmo fills the frame)

    // Render over an EMPTY particle set so the wireframe lines are the only thing that can draw —
    // isolates the wireframe pass from particle overlap/occlusion.
    let empty = ParticleBuffers {
        particle_count: 0,
        ..Default::default()
    };

    renderer.set_solids(&[]);
    let blank = render_and_count(&gpu, &renderer, &empty, &camera, SIZE);

    renderer.set_solids(&scene.solids);
    let lines = render_and_count(&gpu, &renderer, &empty, &camera, SIZE);

    eprintln!("wireframe smoke: {blank} px blank, {lines} px with wireframe");
    assert!(
        lines > 200,
        "wireframe pass drew the cone/cup lines ({lines} px)"
    );
    assert!(
        lines > blank,
        "wireframe adds pixels over a blank frame ({lines} > {blank})"
    );
}

/// `resize()` after `set_solids` rebuilds the depth target at the new size and still renders the
/// wireframe without a depth-size mismatch.
#[test]
fn resize_after_set_solids_renders() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("wireframe resize smoke: no GPU adapter; skipping.");
        return;
    };

    let scene = Scene::v60_pour();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    for _ in 0..10 {
        solver.step(1.0 / 60.0, &input);
    }
    let particles = solver.particles();
    let camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));

    let mut renderer = Renderer::new(&gpu, FORMAT, (SIZE, SIZE), 0.5);
    renderer.set_solids(&scene.solids);
    renderer.resize((SIZE / 2, SIZE / 2));
    let drawn = render_and_count(&gpu, &renderer, &particles, &camera, SIZE / 2);
    assert!(drawn > 50, "renders after resize ({drawn} px)");
}

/// The cross-section inset renders a 2D center slice into the top-right corner: enabling it adds
/// pixels in the top-right quadrant over the same scene with it disabled, and it is a safe no-op
/// when there are no particles.
#[test]
fn cross_section_inset_renders_a_slice() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("cross-section smoke: no GPU adapter; skipping.");
        return;
    };

    let scene = Scene::v60_pour();
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    for _ in 0..30 {
        solver.step(1.0 / 60.0, &input);
    }
    let particles = solver.particles();
    let camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));

    let mut renderer = Renderer::new(&gpu, FORMAT, (SIZE, SIZE), 0.5);
    renderer.set_gizmo_enabled(false); // isolate the inset (the gizmo would fill the corner)

    // The inset sits in the top-right; that quadrant should gain pixels when the slice is on.
    let quadrant = (SIZE / 2, 0, SIZE / 2, SIZE / 2);

    renderer.set_cross_section_enabled(false);
    let off = render_rgba(&gpu, &renderer, &particles, &camera, SIZE);
    let off_n = count_in_rect(&off, SIZE, quadrant);

    renderer.set_cross_section_enabled(true);
    let on = render_rgba(&gpu, &renderer, &particles, &camera, SIZE);
    let on_n = count_in_rect(&on, SIZE, quadrant);

    eprintln!("cross-section smoke: {off_n} px off, {on_n} px on (top-right quadrant)");
    assert!(
        on_n > off_n && on_n - off_n > 50,
        "the inset slice adds substantial pixels in the top-right ({on_n} vs {off_n})"
    );

    // No particles → the inset pass is a safe no-op (no panic, nothing drawn).
    let empty = ParticleBuffers {
        particle_count: 0,
        ..Default::default()
    };
    let blank = render_and_count(&gpu, &renderer, &empty, &camera, SIZE);
    assert_eq!(
        blank, 0,
        "no particles + gizmo off → blank frame even with the cross-section enabled"
    );
}

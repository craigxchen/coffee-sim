//! Offscreen demo of U5 (SDF solid-boundary collision): renders particles conforming to invisible
//! analytic solids. The renderer draws only particles, so the proof is the shape they settle into —
//! a grain funnel held up by a cone wall (vs a flat pile with no solid), and a flat-topped water
//! pool held by a cup. Writes PPM frames to /tmp/coffee-sdf/.
//!
//! Run: `cargo run --release --example sdf_demo`

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::ui::{OrbitCamera, Renderer};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::sdf::{SdfPrimitive, SolidKind, MASK_ALL};
use coffee_sim::EmissionInput;
use glam::Vec3;

const W: u32 = 640;
const H: u32 = 640;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn closed_cone() -> SdfPrimitive {
    SdfPrimitive {
        kind: SolidKind::Cone {
            center: Vec3::ZERO,
            apex_y: 0.0,
            top_y: 8.0,
            apex_r: 0.6,
            top_r: 4.0,
            thickness: 0.1,
            hole_radius: 0.6,
            apex_open: false,
        },
        species_mask: MASK_ALL,
        friction: 0.6,
    }
}

fn cup() -> SdfPrimitive {
    SdfPrimitive {
        kind: SolidKind::Cylinder {
            center: Vec3::ZERO,
            floor_y: 0.0,
            rim_y: 8.0,
            radius: 3.0,
        },
        species_mask: MASK_ALL,
        friction: 0.3,
    }
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("no GPU adapter; skipping.");
        return;
    };
    std::fs::create_dir_all("/tmp/coffee-sdf").unwrap();
    let mats = Materials::default();
    let cfg = Config::default();

    // Grains poured into a cone: they settle into a funnel held up by the cone wall.
    let cone_scene = Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-6.0, -2.0, -6.0],
        box_max: [6.0, 12.0, 6.0],
        regions: vec![SeedRegion {
            min: [-3.0, 1.0, -3.0],
            max: [3.0, 6.0, 3.0],
            species: Species::Grain,
        }],
        solids: vec![closed_cone()],
    };
    // Control: the SAME grains with no solid — they fall to the box floor in a flat pile.
    let mut control = cone_scene.clone();
    control.solids.clear();

    // Water poured into a cup: it pools into a flat-topped cylinder.
    let cup_scene = Scene {
        dose_g: 0.0,
        water_ml: 0.0,
        pour_water_ml: 0.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [-5.0, -2.0, -5.0],
        box_max: [5.0, 12.0, 5.0],
        regions: vec![SeedRegion {
            min: [-2.0, 2.0, -2.0],
            max: [2.0, 6.0, 2.0],
            species: Species::Water,
        }],
        solids: vec![cup()],
    };

    render(
        &gpu,
        &cone_scene,
        &mats,
        &cfg,
        600,
        "/tmp/coffee-sdf/cone_grains.ppm",
    );
    render(
        &gpu,
        &control,
        &mats,
        &cfg,
        600,
        "/tmp/coffee-sdf/control_nosolid.ppm",
    );
    render(
        &gpu,
        &cup_scene,
        &mats,
        &cfg,
        600,
        "/tmp/coffee-sdf/cup_water.ppm",
    );

    // The full V60 dripper draining (calibrated permeable bed): fine water threads the coarser bed,
    // drains through the cone apex into the cup; grains stay trapped in the cone. Three timepoints.
    let v60_mats = Materials {
        particle_spacing: 0.5,
        support_radius: 1.0,
        grain_diameter: 1.0,
        min_pore_fraction: 0.35,
        grain_mass: 10.0,
        ..Materials::default()
    };
    let v60 = Scene::v60();
    render(&gpu, &v60, &v60_mats, &cfg, 60, "/tmp/coffee-sdf/v60_a.ppm");
    render(
        &gpu,
        &v60,
        &v60_mats,
        &cfg,
        250,
        "/tmp/coffee-sdf/v60_b.ppm",
    );
    render(
        &gpu,
        &v60,
        &v60_mats,
        &cfg,
        500,
        "/tmp/coffee-sdf/v60_c.ppm",
    );
    eprintln!(
        "wrote /tmp/coffee-sdf/{{cone_grains,control_nosolid,cup_water,v60_a,v60_b,v60_c}}.ppm"
    );
}

fn render(gpu: &GpuContext, scene: &Scene, mats: &Materials, cfg: &Config, steps: u32, path: &str) {
    let mut solver = XpbdSolver::build(scene, mats, cfg, gpu);
    let input = EmissionInput::default();
    for _ in 0..steps {
        solver.step(1.0 / 60.0, &input);
    }
    let mut renderer = Renderer::new(gpu, FORMAT, (W, H), 0.5 * mats.particle_spacing);
    renderer.set_grain_radius_scale(mats.grain_diameter / mats.particle_spacing);
    let mut camera = OrbitCamera::framing(Vec3::from(scene.box_min), Vec3::from(scene.box_max));
    // Near-side, slightly-above view so the funnel / pool cross-section reads clearly.
    camera.orbit(0.35, -0.22);

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sdf-demo"),
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

    renderer.render(&view, &solver.particles(), &camera);
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
    let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
    for px in data.chunks_exact(4) {
        ppm.extend_from_slice(&px[0..3]);
    }
    drop(data);
    readback.unmap();
    std::fs::write(path, &ppm).unwrap();
}

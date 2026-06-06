//! Offscreen VISUAL of the V60 continuous pour: water is injected at the kettle over the brew,
//! threads the (invisible) grain bed in the cone, drains through the apex, and pools in the cup.
//! The renderer draws only particles — water (blue) + grains (brown, darkening as they wet) — so the
//! invisible cone/cup geometry reads from the shapes the particles take. Writes PPM frames to
//! /tmp/coffee-pour/ at several timepoints across the brew.
//!
//! Run: `cargo run --release --example pour_render`  (env: `FLOW=` mL/s).

use coffee_sim::emission::pour::{PourCommand, PourPattern, PourScript};
use coffee_sim::emission::{EmissionInput, PourEvent};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::ui::{OrbitCamera, Renderer};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use glam::Vec3;

const W: u32 = 768; // multiple of 64 so bytes_per_row (W·4) respects the 256-byte copy alignment
const H: u32 = 768;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const ML_PER_SIM_UNIT3: f32 = 5.20;

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn pour_input(script: &PourScript, t: f32, bed_radius: f32, kettle_y: f32) -> EmissionInput {
    let (nx, nz, flow_ml) = script.sample(t);
    EmissionInput {
        kettle_pos: [nx * bed_radius, kettle_y, nz * bed_radius],
        flow_rate: flow_ml / ML_PER_SIM_UNIT3,
        pour_angle: 0.0,
        event: PourEvent::None,
    }
}

fn main() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pour_render: no GPU adapter; cannot run.");
        return;
    };
    std::fs::create_dir_all("/tmp/coffee-pour").unwrap();

    // Resolution: SPACING scales particle size; count scales ~1/spacing³. The V60 length ratios
    // (grain_diameter = 2·spacing, h = 2·spacing) are preserved so the physics is
    // resolution-consistent; grain_mass is fixed (the grain/water density contrast is
    // scale-invariant). Default 0.12 ≈ ~16k particles (realistic); 0.18 ≈ ~5k (faster); 0.1 ≈ ~50k.
    let s = env_f32("SPACING", 0.12).max(0.06);
    let mats = Materials {
        particle_spacing: s,
        support_radius: 2.0 * s,
        grain_diameter: 2.0 * s,
        // water↔grain contact = 1.2·spacing: grains render at radius 1.0·spacing, so water must
        // rest at ≥ that to sit ON the bed (visibly interacting) instead of threading INSIDE the
        // grain spheres — 0.7·spacing let water centers cross into the grains and read as "passing
        // through". Still porous enough to drain (water reaches the cup).
        water_grain_distance: 1.2 * s,
        grain_mass: 10.0,
        ..Materials::default()
    };
    let cfg = Config {
        absorb_rate: 0.5,
        extract_rate: 1.0,
        // Nozzle radius 0.25 (default 0.5): each emitted layer is a disc of this radius, so 0.5 gave a
        // 1.0-wide descending CURTAIN. 0.25 makes it a tight straight-down column ("funnel down").
        nozzle_radius: 0.25,
        // Velocity backstop 25 (default 50): the canonical wall contact-response in finalize is the
        // real anti-eruption fix; this is the safety net — 25 is just above the deepest legitimate
        // fall (kettle→cup ≈ 20) and below the ~28 domain-crossing speed, so it never clips real flow
        // but caps any residual transient pressure burst before it can fling a stray across the box.
        max_speed: 25.0,
        xsph_viscosity_c: env_f32("VISC", 0.05),
        s_corr_n: env_f32("SCN", 16.0),
        ..Config::default()
    };
    // 5 mL/s — a realistic active pour. The bed only accepts water as fast as it percolates; at the
    // old firehose 12 mL/s water backs up at the bed and the density solve EXPELS the excess upward
    // (a position-correction velocity injection, worse at smaller dt), which looks like spray
    // "bouncing off the walls". A sustainable rate lets it pool and drain instead.
    let flow = env_f32("FLOW", 5.0);
    // Center pour: a fixed straight-down stream at the bed center (not a wandering spiral).
    let script = PourScript {
        commands: vec![PourCommand {
            t_start: 0.0,
            t_end: 12.0,
            flow_rate: flow,
            pattern: PourPattern::Center,
        }],
    };

    let scene = Scene::v60_pour();
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);

    // Renderer: grains sized to the contact diameter, tinted by saturation (dry → wet darkens).
    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    let v_cap = mats.r_max * mats.rho_ratio * grain_volume;
    let mut renderer = Renderer::new(&gpu, FORMAT, (W, H), 0.5 * mats.particle_spacing);
    renderer.set_grain_radius_scale(mats.grain_diameter / mats.particle_spacing);
    renderer.set_moisture_scale(1.0 / v_cap);

    // Frame TIGHTLY on the dripper+cup region (not the whole domain) so the small bed, the incoming
    // stream, and the cup pool fill the frame. Near-side, slightly-above view for the cross-section.
    let mut camera = OrbitCamera::framing(Vec3::new(-3.5, -8.5, -3.5), Vec3::new(3.5, 3.5, 3.5));
    camera.orbit(0.30, -0.18);

    // Timepoints (frames): initial bed, bloom landing, pouring through, draining, cup filling.
    let shots: [(u32, &str); 6] = [
        (0, "0_bed"),
        (90, "1_bloom"),
        (240, "2_pourthrough"),
        (450, "3_draining"),
        (720, "4_cup"),
        (1020, "5_late"),
    ];
    let _ = &shots;
    let dt = 1.0 / 60.0;
    let last = env_f32("STEPS", 1020.0) as u32;
    let mut max_bubble = 0.0f32;
    for step in 0..=last {
        let t = step as f32 * dt;
        solver.step(dt, &pour_input(&script, t % 12.0, 2.0, 2.5)); // LOOP the pour (fill the cup deep)
        if step % 60 == 0 {
            let vel = solver.read_velocities();
            let pos = solver.read_positions();
            let ph = solver.read_phases();
            let mut bub = 0u32;
            let mut bmax = 0.0f32;
            for ((v, q), &p) in vel.iter().zip(&pos).zip(&ph) {
                if p != 0 {
                    continue;
                }
                if q[1] < -3.5 && v[1] > 3.0 {
                    bub += 1;
                    bmax = bmax.max(v[1]);
                }
            }
            max_bubble = max_bubble.max(bmax);
            if step % 300 == 0 || bub > 5 {
                eprintln!(
                    "  t={:>5.1}s active={:>5} cup-bubble(vy>3) count={bub:>3} max={bmax:.1}",
                    t,
                    solver.active_count(),
                );
            }
        }
    }
    dump_frame(
        &gpu,
        &mut renderer,
        &solver,
        &camera,
        "/tmp/coffee-pour/deep.ppm",
    );
    eprintln!("MAX CUP BUBBLE vy = {max_bubble:.1}");
}

fn dump_frame(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    solver: &XpbdSolver,
    camera: &OrbitCamera,
    path: &str,
) {
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pour-render"),
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

    renderer.render(&view, &solver.particles(), camera);
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

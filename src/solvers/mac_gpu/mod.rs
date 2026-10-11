//! WebGPU MAC porous-mixture solver. CPU work is initialization, command encoding,
//! and asynchronous diagnostics only; all evolving physical state stays on the GPU.
use crate::{
    emission::EmissionInput,
    engine::{Metrics, Scene},
    models::Materials,
    profiling::Profile,
    solvers::{base::Solver, mac::MacSimulation},
    utils::{buffers::ParticleBuffers, config::Config, gpu::GpuContext, sdf::SolidKind},
};
use bytemuck::{Pod, Zeroable};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 4],
    pub velocity: [f32; 4],
    pub inventory: [f32; 4],
    pub material: [f32; 4],
    pub extra: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dims: [u32; 4],
    origin: [f32; 4],
    high: [f32; 4],
    gravity: [f32; 4],
    physics: [f32; 4],
    rates: [f32; 4],
    thermal: [f32; 4],
    kinetics: [f32; 4],
    flags: [u32; 4],
    misc: [f32; 4],
    inlet: [f32; 4],
    jet: [f32; 4],
    control: [f32; 4],
    counts: [u32; 4],
}
/// A delayed, nonblocking snapshot of actual GPU inventories. Includes drained water.
#[derive(Clone, Debug, Default)]
pub struct GpuDiagnostics {
    pub time: f32,
    pub pending_time: f32,
    pub free_water: f32,
    pub absorbed_water: f32,
    pub drained_water: f32,
    pub emitted_water: f32,
    pub pending_water: f32,
    pub dissolved_solute: f32,
    pub retained_solute: f32,
    pub remaining_solute: f32,
    pub drained_solute: f32,
    pub water_error: f32,
    pub solute_error: f32,
    pub heat: f32,
    pub momentum: [f32; 3],
    pub pressure_residual: f32,
    pub evenness: f32,
    pub drawdown_time: f32,
    pub particle_count: u32,
}
pub struct MacGpuSolver {
    gpu: GpuContext,
    mats: Materials,
    device: wgpu::Device,
    queue: wgpu::Queue,
    scene: Scene,
    cfg: Config,
    params: Params,
    uniform: wgpu::Buffer,
    storage: Vec<Arc<wgpu::Buffer>>,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline, [wgpu::BindGroup; 2])>,
    parity: usize,
    buffers: ParticleBuffers,
    initial: Vec<GpuParticle>,
    initial_water: f32,
    initial_solute: f32,
    dose: f32,
    readback: wgpu::Buffer,
    pending: Arc<AtomicBool>,
    snapshot: Arc<Mutex<GpuDiagnostics>>,
    dispatches: u32,
}
const ENTRIES: &[&str] = &[
    "clear",
    "bin",
    "inventory",
    "absorb",
    "dissolution",
    "dissolve",
    "heat",
    "scatter_faces",
    "assemble",
    "red",
    "black",
    "project",
    "move_particles",
    "contacts",
    "begin_frame",
    "timestep",
    "export_particles",
    "reduce",
];
impl MacGpuSolver {
    pub fn diagnostics(&self) -> GpuDiagnostics {
        self.snapshot.lock().unwrap().clone()
    }
    /// GPU-owned AoS state, exposed for validation readback (never used by the frame loop).
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.storage[self.parity]
    }
    pub fn capacity(&self) -> u32 {
        self.params.dims[3]
    }
    fn run(&mut self, encoder: &mut wgpu::CommandEncoder, name: &str, n: u32) {
        let pipeline = self.pipelines.iter().find(|p| p.0 == name).unwrap();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.1);
        pass.set_bind_group(0, &pipeline.2[self.parity], &[]);
        pass.dispatch_workgroups(n.div_ceil(64).max(1), 1, 1);
        self.dispatches += 1;
    }
    fn flip(&mut self) {
        self.parity = 1 - self.parity;
    }
    fn cell_count(&self) -> u32 {
        self.params.dims[..3].iter().product()
    }
    fn face_count(&self) -> u32 {
        let d = self.params.dims;
        (d[0] + 1) * d[1] * d[2] + d[0] * (d[1] + 1) * d[2] + d[0] * d[1] * (d[2] + 1)
    }
    fn export(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.run(encoder, "export_particles", self.capacity());
        self.run(encoder, "reduce", 1);
    }
    fn sample(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if self.pending.swap(true, Ordering::AcqRel) {
            return false;
        }
        encoder.copy_buffer_to_buffer(&self.storage[13], 0, &self.readback, 0, 112);
        true
    }
    fn initialize(&mut self) {
        self.pending = Arc::new(AtomicBool::new(false));
        self.snapshot = Arc::new(Mutex::new(GpuDiagnostics::default()));
        self.readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MAC reset diagnostics"),
            size: 112,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.parity = 0;
        let mut data = vec![GpuParticle::zeroed(); self.capacity() as usize];
        for p in &mut data {
            p.position[3] = 3.0;
        }
        data[..self.initial.len()].copy_from_slice(&self.initial);
        self.queue
            .write_buffer(&self.storage[0], 0, bytemuck::cast_slice(&data));
        self.queue
            .write_buffer(&self.storage[1], 0, bytemuck::cast_slice(&data));
        self.queue.write_buffer(&self.storage[13], 0, &[0; 112]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.export(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        *self.snapshot.lock().unwrap() = GpuDiagnostics {
            free_water: self.initial_water,
            remaining_solute: self.initial_solute,
            particle_count: self.initial.len() as u32,
            ..Default::default()
        };
    }
    fn map_sample(&self) {
        let pending = self.pending.clone();
        let snapshot = self.snapshot.clone();
        let buffer = self.readback.clone();
        let initial_water = self.initial_water;
        let initial_solute = self.initial_solute;
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let bytes = buffer.slice(..).get_mapped_range();
                    let v: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
                    *snapshot.lock().unwrap() = GpuDiagnostics {
                        time: v[0][1],
                        pending_time: v[0][2],
                        free_water: v[1][0],
                        absorbed_water: v[1][1],
                        drained_water: v[1][2],
                        particle_count: v[1][3] as u32,
                        dissolved_solute: v[2][0],
                        retained_solute: v[2][1],
                        remaining_solute: v[2][2],
                        drained_solute: v[2][3],
                        water_error: v[1][0] + v[1][1] + v[1][2] - initial_water - v[5][2],
                        solute_error: v[2].iter().sum::<f32>() - initial_solute,
                        momentum: [v[3][0], v[3][1], v[3][2]],
                        heat: v[3][3],
                        pressure_residual: v[5][0],
                        evenness: v[5][1],
                        drawdown_time: v[6][0],
                        emitted_water: v[5][2],
                        pending_water: v[5][3],
                    };
                    drop(bytes);
                    buffer.unmap();
                }
                pending.store(false, Ordering::Release);
            });
    }
}
impl Solver for MacGpuSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        assert!(cfg.mac.gpu_pressure_iterations > 0);
        // Reuse the deterministic reference initializer, never its advance/physics path.
        let seed = MacSimulation::new(scene, mats, cfg);
        let mut initial = Vec::new();
        for g in seed.grains() {
            initial.push(GpuParticle {
                position: [
                    g.position.x as f32,
                    g.position.y as f32,
                    g.position.z as f32,
                    1.0,
                ],
                velocity: [0.0; 4],
                inventory: [
                    g.absorbed as f32,
                    g.fast as f32,
                    g.heat as f32,
                    g.slow as f32,
                ],
                material: [
                    g.dry_volume as f32,
                    g.initial_mass as f32,
                    g.capacity as f32,
                    g.insoluble as f32,
                ],
                extra: [g.retained_solute as f32, 0.0, 0.0, 0.0],
            });
        }
        let grains = initial.len() as u32;
        for p in seed.water() {
            initial.push(GpuParticle {
                position: [
                    p.position.x as f32,
                    p.position.y as f32,
                    p.position.z as f32,
                    0.0,
                ],
                velocity: [0.0; 4],
                inventory: [p.volume as f32, p.solute as f32, p.heat as f32, 0.0],
                material: [0.0; 4],
                extra: [0.0; 4],
            });
        }
        let capacity = cfg
            .mac
            .gpu_particle_capacity
            .max(initial.len() as u32)
            .max(1);
        let h = if cfg.mac.cell_size > 0.0 {
            cfg.mac.cell_size as f32
        } else {
            2.0 * mats.particle_spacing
        };
        let dims: Vec<u32> = (0..3)
            .map(|a| ((scene.box_max[a] - scene.box_min[a]) / h).ceil() as u32)
            .collect();
        let params = Params {
            dims: [dims[0], dims[1], dims[2], capacity],
            origin: [scene.box_min[0], scene.box_min[1], scene.box_min[2], h],
            high: [scene.box_max[0], scene.box_max[1], scene.box_max[2], 0.0],
            gravity: [
                scene.gravity[0],
                scene.gravity[1],
                scene.gravity[2],
                mats.particle_mass / mats.particle_spacing.powi(3),
            ],
            physics: [
                mats.grain_diameter,
                cfg.drag_scale,
                cfg.mac.swelling_ratio as f32,
                cfg.mac.capillary_pressure as f32,
            ],
            rates: [
                cfg.absorb_rate,
                cfg.extract_rate,
                mats.k0_fast,
                mats.k0_slow,
            ],
            thermal: [
                mats.cp_water,
                mats.cp_grain,
                mats.kappa,
                u32::from(cfg.mac.thermal_exchange) as f32,
            ],
            kinetics: [mats.ea_over_r, mats.t_ref, mats.d_ref, mats.u_half],
            flags: [
                cfg.mac.open_bottom as u32,
                cfg.mac.frozen_bed as u32,
                scene.solids.len() as u32,
                cfg.seed as u32,
            ],
            misc: [
                mats.c_sat,
                mats.friction_mu,
                mats.floor_mu,
                cfg.mac.max_dt as f32,
            ],
            inlet: [0.0; 4],
            jet: [0.0; 4],
            control: [
                0.0,
                mats.particle_spacing.powi(3),
                mats.pour_t,
                cfg.mac.cfl as f32,
            ],
            counts: [grains, 0, 0, 0],
        };
        let mut solids = Vec::<[[f32; 4]; 4]>::new();
        for s in &scene.solids {
            let mut data = [[0.0; 4]; 4];
            data[0][3] = s.species_mask as f32;
            match s.kind {
                SolidKind::Cone {
                    center,
                    apex_y,
                    top_y,
                    apex_r,
                    top_r,
                    thickness,
                    hole_radius,
                    apex_open,
                } => {
                    data[0][0] = center.x;
                    data[0][1] = center.z;
                    data[1] = [apex_y, top_y, apex_r, top_r];
                    data[2] = [thickness, hole_radius, 0.0, apex_open as u32 as f32];
                }
                SolidKind::Cylinder {
                    center,
                    floor_y,
                    rim_y,
                    radius,
                } => {
                    data[0][0] = center.x;
                    data[0][1] = center.z;
                    data[0][2] = 1.0;
                    data[1] = [floor_y, rim_y, radius, 0.0];
                }
                SolidKind::PolyCup {
                    center,
                    floor_y,
                    rim_y,
                    apothem,
                    sides,
                } => {
                    data[0][0] = center.x;
                    data[0][1] = center.z;
                    data[0][2] = 2.0;
                    data[1] = [floor_y, rim_y, apothem, 0.0];
                    data[2][2] = sides as f32;
                }
            }
            data[3][0] = s.friction;
            solids.push(data);
        }
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MAC parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let nc = dims.iter().product::<u32>();
        let nf = (dims[0] + 1) * dims[1] * dims[2]
            + dims[0] * (dims[1] + 1) * dims[2]
            + dims[0] * dims[1] * (dims[2] + 1);
        let sizes = [
            capacity as u64 * 80,
            capacity as u64 * 80,
            (nc + nf) as u64 * 4,
            capacity as u64 * 32 * 16,
            nc as u64 * 96,
            nf as u64 * 48,
            nc as u64 * 4,
            (solids.len().max(1) * 64) as u64,
            capacity as u64 * 16,
            capacity as u64 * 16,
            capacity as u64 * 4,
            capacity as u64 * 16,
            capacity as u64 * 4,
            112,
        ];
        let storage: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                assert!(
                    size <= device.limits().max_storage_buffer_binding_size,
                    "MAC buffer {i} exceeds device limit"
                );
                Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("MAC storage"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }))
            })
            .collect();
        if !solids.is_empty() {
            queue.write_buffer(&storage[7], 0, bytemuck::cast_slice(&solids));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MAC physics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });
        // Auto layouts contain only bindings actually used by an entry point. The
        // largest pass uses seven storage buffers, below the WebGPU baseline of eight.
        let pipelines = ENTRIES
            .iter()
            .map(|&name| {
                let bindings: &[u32] = match name {
                    "clear" => &[0, 3, 7],
                    "bin" => &[0, 1, 3, 4],
                    "inventory" | "dissolution" => &[0, 1, 3, 4, 5, 14],
                    "absorb" | "dissolve" | "heat" => &[0, 1, 2, 5, 14],
                    "scatter_faces" => &[0, 1, 3, 4, 5, 6, 8, 14],
                    "assemble" => &[0, 5, 6, 14],
                    "red" | "black" => &[0, 5, 6, 7],
                    "project" => &[0, 6, 7, 14],
                    "move_particles" => &[0, 1, 2, 6, 8, 14],
                    "contacts" => &[0, 1, 2, 8],
                    "begin_frame" => &[0, 14],
                    "timestep" => &[0, 1, 8, 14],
                    "export_particles" => &[0, 1, 9, 10, 11, 12, 13],
                    "reduce" => &[0, 1, 5, 6, 7, 8, 14],
                    _ => unreachable!(),
                };
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &shader,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let layout = pipeline.get_bind_group_layout(0);
                let groups = std::array::from_fn(|parity| {
                    let entries: Vec<_> = bindings
                        .iter()
                        .map(|&binding| {
                            let resource = if binding == 0 {
                                uniform.as_entire_binding()
                            } else {
                                let i = binding as usize - 1;
                                let ix = if i < 2 { i ^ parity } else { i };
                                storage[ix].as_entire_binding()
                            };
                            wgpu::BindGroupEntry { binding, resource }
                        })
                        .collect();
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(name),
                        layout: &layout,
                        entries: &entries,
                    })
                });
                (name, pipeline, groups)
            })
            .collect();
        let buffers = ParticleBuffers {
            particle_count: capacity,
            position: Some(storage[8].clone()),
            velocity: Some(storage[9].clone()),
            phase_tag: Some(storage[10].clone()),
            concentration: Some(storage[11].clone()),
            temperature: Some(storage[11].clone()),
            moisture: Some(storage[12].clone()),
            ..Default::default()
        };
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MAC async diagnostics"),
            size: 112,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let d = seed.diagnostics();
        let dose = seed.grains().iter().map(|g| g.initial_mass as f32).sum();
        let mut this = Self {
            gpu: gpu.clone(),
            mats: mats.clone(),
            device,
            queue,
            scene: scene.clone(),
            cfg: cfg.clone(),
            params,
            uniform,
            storage,
            pipelines,
            parity: 0,
            buffers,
            initial,
            initial_water: (d.free_water + d.absorbed_water) as f32,
            initial_solute: d.remaining_solute as f32,
            dose,
            readback,
            pending: Arc::new(AtomicBool::new(false)),
            snapshot: Arc::new(Mutex::new(GpuDiagnostics::default())),
            dispatches: 0,
        };
        this.initialize();
        this
    }
    fn reset(&mut self, scene: &Scene) {
        *self = Self::build(scene, &self.mats, &self.cfg, &self.gpu);
    }
    fn step(&mut self, dt: f32, input: &EmissionInput) {
        if input.event == crate::emission::PourEvent::Reset {
            self.reset(&self.scene.clone());
            return;
        }
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Poll);
        let flow = if input.flow_rate.is_finite()
            && input.flow_rate > 0.0
            && input.kettle_pos.iter().all(|v| v.is_finite())
            && input.pour_angle.is_finite()
        {
            input.flow_rate
        } else {
            0.0
        };
        let speed = flow
            / (std::f32::consts::PI * self.cfg.nozzle_radius.powi(2) * self.cfg.discharge_coeff)
                .max(1e-12);
        self.params.inlet = [
            input.kettle_pos[0],
            input.kettle_pos[1],
            input.kettle_pos[2],
            flow * dt,
        ];
        let angle = if input.pour_angle.is_finite() {
            input.pour_angle
        } else {
            0.0
        };
        self.params.jet = [
            speed * angle.sin(),
            -speed * angle.cos(),
            0.0,
            self.cfg.nozzle_radius,
        ];
        self.params.control[0] = dt;
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.params));
        self.dispatches = 0;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.run(&mut encoder, "begin_frame", 1);
        let n = self.capacity();
        let nc = self.cell_count();
        let nf = self.face_count();
        // CFL is evaluated from GPU velocities. Unconsumed time stays in stats[0].z,
        // making overload observable instead of silently losing time or clamping speeds.
        let steps = (dt / self.params.misc[3]).ceil().max(1.0) as u32;
        for _ in 0..steps {
            self.run(&mut encoder, "timestep", 1);
            self.run(&mut encoder, "clear", nc + nf);
            self.run(&mut encoder, "bin", n);
            self.run(&mut encoder, "inventory", nc);
            self.run(&mut encoder, "absorb", n);
            self.flip();
            self.run(&mut encoder, "inventory", nc);
            self.run(&mut encoder, "dissolution", nc);
            self.run(&mut encoder, "dissolve", n);
            self.flip();
            self.run(&mut encoder, "inventory", nc);
            self.run(&mut encoder, "heat", n);
            self.flip();
            self.run(&mut encoder, "inventory", nc);
            self.run(&mut encoder, "scatter_faces", nf);
            self.run(&mut encoder, "assemble", nc);
            for _ in 0..self.cfg.mac.gpu_pressure_iterations {
                self.run(&mut encoder, "red", nc);
                self.run(&mut encoder, "black", nc);
            }
            self.run(&mut encoder, "project", nf);
            self.run(&mut encoder, "move_particles", n);
            self.flip();
            for _ in 0..self.cfg.mac.contact_iterations {
                self.run(&mut encoder, "contacts", n);
                self.flip();
            }
        }
        self.export(&mut encoder);
        let sample = self.sample(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        if sample {
            self.map_sample();
        }
    }
    fn particles(&self) -> ParticleBuffers {
        self.buffers.clone()
    }
    fn metrics(&self) -> Metrics {
        let d = self.diagnostics();
        let extracted = d.dissolved_solute + d.retained_solute + d.drained_solute;
        let brewed = d.dissolved_solute + d.drained_solute;
        let water = (d.free_water + d.drained_water) * self.params.gravity[3];
        Metrics {
            extraction_yield: extracted / self.dose.max(1e-20),
            tds: brewed / (water + brewed).max(1e-20),
            particle_count: d.particle_count,
            iteration_count: self.cfg.mac.gpu_pressure_iterations,
            evenness: d.evenness,
            drawdown_time: d.drawdown_time,
        }
    }
    fn simulation_progress(&self) -> Option<(f32, f32)> {
        let d = self.diagnostics();
        Some((d.time, d.pending_time))
    }
    fn profile(&self) -> Profile {
        Profile {
            dispatches_per_frame: self.dispatches,
            ..Default::default()
        }
    }
}

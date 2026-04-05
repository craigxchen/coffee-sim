use coffee_sim_core::sph::Vec3;

mod state;
mod pipelines;
mod shader;
pub(crate) mod inflow;
pub(crate) mod bed;

use state::{MpmBuffers, MpmUniforms, FP_SCALE, MAX_VELOCITY, NUM_THREADS, SDF_RES};
use pipelines::MpmPipelines;
use inflow::{InflowState, SpoutSettings, MASS_UNITS_PER_ML};
use bed::{BedConfig, BedInit};

const TARGET_BED_RETENTION_ML: f32 = 42.0;

#[derive(Clone, Debug)]
pub(crate) enum Obstacle {
    TruncatedCone {
        center: Vec3,
        top_radius: f32,
        bot_radius: f32,
        top_y: f32,
        bot_y: f32,
    },
    Cylinder {
        center: Vec3,
        radius: f32,
        top_y: f32,
        bot_y: f32,
    },
}

#[derive(Clone)]
pub(crate) struct MpmSettings {
    pub bounds_size: Vec3,
    pub grid_dims: [u32; 3],
    pub max_particles: u32,
    pub substeps: u32,
    pub gravity: f32,
    pub bulk_modulus: f32,
    pub viscosity: f32,
    pub render_radius: f32,
    pub obstacles: Vec<Obstacle>,
    pub spout: SpoutSettings,
    pub initial_kettle_angle_deg: f32,
    pub bed: Option<BedConfig>,
}

impl MpmSettings {
    pub fn default_v60() -> Self {
        let bounds_size = Vec3::new(14.0, 20.0, 14.0);
        // Ensure uniform cell spacing: derive gy from dx = bounds_x / gx
        let gx = 80u32;
        let dx = bounds_size.x / gx as f32;
        let gy = (bounds_size.y / dx).ceil() as u32;
        let gz = 80u32;
        let grid_dims = [gx, gy, gz];

        Self {
            bounds_size,
            grid_dims,
            max_particles: 220_000,
            substeps: 3,
            gravity: -10.0,
            bulk_modulus: 420.0,
            viscosity: 0.05,
            render_radius: dx * 0.7,
            obstacles: vec![
                Obstacle::TruncatedCone {
                    center: Vec3::ZERO,
                    top_radius: 4.5,
                    bot_radius: 0.8,
                    top_y: 3.0,
                    bot_y: -3.0,
                },
                Obstacle::Cylinder {
                    center: Vec3::ZERO,
                    radius: 3.0,
                    top_y: -3.5,
                    bot_y: -8.0,
                },
            ],
            spout: SpoutSettings::default(),
            initial_kettle_angle_deg: 36.0,
            bed: Some(BedConfig::default()),
        }
    }
}

pub(crate) struct MpmSim3D {
    settings: MpmSettings,
    buffers: MpmBuffers,
    pipelines: MpmPipelines,
    inflow: InflowState,
    num_water: u32,
    num_bed: u32,
    total_time: f32,
}

impl MpmSim3D {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: MpmSettings,
    ) -> Self {
        let buffers = MpmBuffers::new(device, queue, &settings);
        let pipelines = MpmPipelines::new(device, &buffers);
        let inflow = InflowState::new(settings.initial_kettle_angle_deg);

        let mut sim = Self {
            settings,
            buffers,
            pipelines,
            inflow,
            num_water: 0,
            num_bed: 0,
            total_time: 0.0,
        };

        sim.init_bed(queue);
        sim
    }

    fn init_bed(&mut self, queue: &wgpu::Queue) {
        let config = match &self.settings.bed {
            Some(c) => c,
            None => return,
        };

        let BedInit {
            particles,
            affines,
            bed_extracts,
            cell_lookup,
        } = bed::init_bed_particles(config, self.settings.grid_dims, self.settings.bounds_size);
        let count = particles.len() as u32;
        if count == 0 {
            return;
        }

        self.num_bed = count;

        // Active particle layout is contiguous with bed particles first and water
        // particles appended after them.
        queue.write_buffer(&self.buffers.particles, 0, bytemuck::cast_slice(&particles));
        queue.write_buffer(&self.buffers.affine, 0, bytemuck::cast_slice(&affines));
        queue.write_buffer(&self.buffers.bed_extract, 0, bytemuck::cast_slice(&bed_extracts));
        queue.write_buffer(&self.buffers.bed_lookup, 0, bytemuck::cast_slice(&cell_lookup));
        let zero_delta = vec![0_i32; self.settings.max_particles as usize];
        queue.write_buffer(&self.buffers.bed_delta, 0, bytemuck::cast_slice(&zero_delta));
    }

    pub fn step_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
    ) {
        let dt = dt.min(1.0 / 30.0);
        let substeps = self.settings.substeps.max(1);
        let sub_dt = dt / substeps as f32;

        for _ in 0..substeps {
            // Emit new particles
            let emitted = self.inflow.emit_particles(
                queue,
                &self.buffers,
                &self.settings.spout,
                sub_dt,
                MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML,
                self.num_water,
                self.num_bed,
                self.settings.max_particles,
            );
            self.num_water += emitted;

            // Update uniforms
            self.write_uniforms(queue, sub_dt);

            // Dispatch compute passes
            let total_cells = self.settings.grid_dims[0]
                * self.settings.grid_dims[1]
                * self.settings.grid_dims[2];
            let num_particles = self.num_water + self.num_bed;
            let cell_wg = dispatch_size(total_cells, NUM_THREADS);
            let particle_wg = dispatch_size(num_particles, NUM_THREADS);
            let bed_wg = dispatch_size(self.num_bed, NUM_THREADS);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mpm step"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mpm compute"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.pipelines.bind_group, &[]);

                // 1. clear_grid
                pass.set_pipeline(&self.pipelines.clear_grid);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // 2. bed_coupling
                if particle_wg > 0 {
                    pass.set_pipeline(&self.pipelines.bed_coupling);
                    pass.dispatch_workgroups(particle_wg, 1, 1);
                }

                // 3. extraction_advect
                if bed_wg > 0 {
                    pass.set_pipeline(&self.pipelines.extraction_advect);
                    pass.dispatch_workgroups(bed_wg, 1, 1);
                }

                // 4. p2g
                if particle_wg > 0 {
                    pass.set_pipeline(&self.pipelines.p2g);
                    pass.dispatch_workgroups(particle_wg, 1, 1);
                }

                // 5. grid_update
                pass.set_pipeline(&self.pipelines.grid_update);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // 6. boundary_project
                pass.set_pipeline(&self.pipelines.boundary_project);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // 7. g2p
                if particle_wg > 0 {
                    pass.set_pipeline(&self.pipelines.g2p);
                    pass.dispatch_workgroups(particle_wg, 1, 1);
                }

                // 8. bed_dynamics
                if bed_wg > 0 {
                    pass.set_pipeline(&self.pipelines.bed_dynamics);
                    pass.dispatch_workgroups(bed_wg, 1, 1);
                }

                // 9. prepare_render
                if particle_wg > 0 {
                    pass.set_pipeline(&self.pipelines.prepare_render);
                    pass.dispatch_workgroups(particle_wg, 1, 1);
                }
            }
            queue.submit(Some(encoder.finish()));

            self.total_time += sub_dt;
        }
    }

    pub fn reset(&mut self, queue: &wgpu::Queue, _device: &wgpu::Device) {
        self.num_water = 0;
        self.num_bed = 0;
        self.total_time = 0.0;
        self.inflow = InflowState::new(self.settings.initial_kettle_angle_deg);
        self.init_bed(queue);
    }

    pub fn set_kettle_angle(&mut self, angle_deg: f32) {
        self.inflow.set_angle(angle_deg);
    }

    pub fn kettle_angle(&self) -> f32 {
        self.inflow.angle()
    }

    pub fn flow_rate_ml_s(&self) -> f32 {
        self.inflow.flow_rate()
    }

    pub fn exit_speed(&self) -> f32 {
        self.inflow.exit_speed()
    }

    pub fn particle_count(&self) -> usize {
        (self.num_water + self.num_bed) as usize
    }

    pub fn render_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.render_data
    }

    pub fn settings(&self) -> &MpmSettings {
        &self.settings
    }

    fn write_uniforms(&self, queue: &wgpu::Queue, dt: f32) {
        let [gx, gy, gz] = self.settings.grid_dims;
        let total_cells = gx * gy * gz;
        let bs = self.settings.bounds_size;
        let dx = bs.x / gx as f32;
        let inv_dx = 1.0 / dx;
        let particle_vol = dx * dx * dx * 0.25;
        let bed_capacity_per_particle = if self.num_bed > 0 {
            TARGET_BED_RETENTION_ML * MASS_UNITS_PER_ML / self.num_bed as f32
        } else {
            0.7
        };

        let uniforms = MpmUniforms {
            grid_dims: [gx, gy, gz, total_cells],
            counts: [
                self.num_water,
                self.num_bed,
                self.settings.max_particles,
                self.settings.substeps,
            ],
            sim_params: [dt, self.settings.gravity, dx, inv_dx],
            grid_origin: [-bs.x * 0.5, -bs.y * 0.5, -bs.z * 0.5, 0.0],
            bounds_max: [bs.x * 0.5, bs.y * 0.5, bs.z * 0.5, 0.0],
            fluid_params: [self.settings.bulk_modulus, self.settings.viscosity, 1.0, particle_vol],
            fp_params: [FP_SCALE, 1.0 / FP_SCALE, MAX_VELOCITY, 0.0],
            inflow_origin: [
                self.settings.spout.origin.x,
                self.settings.spout.origin.y,
                self.settings.spout.origin.z,
                0.0,
            ],
            inflow_dir: [
                self.settings.spout.direction.x,
                self.settings.spout.direction.y,
                self.settings.spout.direction.z,
                self.inflow.exit_speed(),
            ],
            inflow_params: [self.settings.spout.nozzle_radius, 0.0, 0.0, 0.0],
            sdf_params: [SDF_RES as f32, 0.3, 0.0, 0.05],
            // Tie bed retention to an overall retained-water target so the bed
            // wets realistically without swallowing most of the brew.
            bed_params: [34.0, 8.0, bed_capacity_per_particle, 0.0],
            extraction_params: [0.01, 18.0, 7.0, 14.0],
            time_params: [self.total_time, dt, 0.0, 0.0],
        };

        queue.write_buffer(
            &self.buffers.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }
}

fn dispatch_size(count: u32, threads: u32) -> u32 {
    (count + threads - 1) / threads
}

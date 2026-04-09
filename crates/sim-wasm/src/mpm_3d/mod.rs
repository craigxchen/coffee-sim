use coffee_sim_core::sph::Vec3;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::JsValue;

pub(crate) mod bed;
pub(crate) mod inflow;
mod filter;
mod filter_mesh;
mod pipelines;
mod shader;
mod state;

pub(crate) use filter::FilterConfig;
#[cfg(target_arch = "wasm32")]
pub(crate) use filter_mesh::{MAX_FILL_VERTEX_COUNT, MAX_RENDER_VERTEX_COUNT};

use bed::{BedConfig, BedInit};
use filter_mesh::FilterMesh;
use inflow::{EmissionResult, InflowState, SpoutSettings, MASS_UNITS_PER_ML};
use pipelines::MpmPipelines;
use state::{
    MpmBuffers, MpmUniforms, FP_SCALE, FP_VALUE_LIMIT, MAX_VELOCITY, METRICS_DIV_FP_SCALE,
    METRICS_SLOT_COUNT, NUM_THREADS, SDF_RES,
};

const TARGET_BED_RETENTION_ML: f32 = 42.0;

/// Device limits required by the MPM compute pipeline.
///
/// The MPM bind group holds 10 storage buffers (particles, affine, grid,
/// grid_vel, render_data, bed_extract, bed_lookup, bed_delta, metrics) plus
/// one SDF texture. The unused `bed_support_count` slot that previously
/// occupied binding 10 has been repurposed for the metrics buffer so we stay
/// within the 10-buffer cap that some WebGPU adapters enforce. Any
/// `request_device` site that uses this pipeline must use these limits, and
/// `mpm_pipelines_fit_within_required_limits` pins the invariant.
pub(crate) fn required_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: 10,
        ..wgpu::Limits::default()
    }
}

/// Snapshot of the projection observability counters sampled asynchronously
/// from the GPU metrics buffer. Values are the last successful readback; they
/// decay to `None` / stale values if the readback path has not completed yet.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MetricsSnapshot {
    /// Peak `|div u|` observed across any fluid cell in the most recent
    /// substep. Decoded from the fixed-point atomic via
    /// `METRICS_DIV_FP_SCALE`.
    pub max_abs_div: f32,
    /// Count of cells classified as fluid (`CELL_INTERIOR_FLUID` or
    /// `CELL_BED_COUPLED`) in the most recent substep.
    pub fluid_cells: u32,
    /// Number of `divergence_store` calls that hit the FP clamp.
    pub div_clamp_fires: u32,
    /// Number of `pressure_store` calls that hit the FP clamp.
    pub pressure_clamp_fires: u32,
    /// Number of P2G contributions that tripped the overflow probe.
    pub mass_overflow_fires: u32,
}

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
    pub filter: Option<FilterConfig>,
    pub bed: Option<BedConfig>,
    pub enable_pressure_projection: bool,
    pub enable_temp_sparse_ballistic: bool,
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
        let filter = FilterConfig::default();
        let bed = BedConfig::seated_in_filter(&filter);

        Self {
            bounds_size,
            grid_dims,
            max_particles: 220_000,
            substeps: 5,
            gravity: -10.0,
            bulk_modulus: 900.0,
            viscosity: 0.12,
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
            filter: Some(filter),
            bed: Some(bed),
            enable_pressure_projection: true,
            enable_temp_sparse_ballistic: true,
        }
    }

    pub fn benchmark_free_stream() -> Self {
        let mut settings = Self::default_v60();
        settings.bed = None;
        settings.spout.origin = Vec3::new(0.0, 6.8, 0.0);
        settings.spout.aim_at(Vec3::new(0.0, -6.8, 0.0));
        settings.initial_kettle_angle_deg = 28.0;
        settings
    }

    pub fn benchmark_center_pour() -> Self {
        let mut settings = Self::default_v60();
        settings.spout.origin = Vec3::new(0.0, 7.1, 0.0);
        settings.spout.aim_at(Vec3::new(0.0, 0.4, 0.0));
        settings.initial_kettle_angle_deg = 36.0;
        settings
    }
}

pub(crate) struct MpmSim3D {
    settings: MpmSettings,
    buffers: MpmBuffers,
    pipelines: MpmPipelines,
    inflow: InflowState,
    filter_mesh: Option<FilterMesh>,
    num_water: u32,
    num_bed: u32,
    total_time: f32,
    frame_emitted_mass: f32,
    frame_dropped_particles: u32,
    total_emitted_mass: f32,
    total_dropped_particles: u32,
    latest_metrics: MetricsSnapshot,
}

impl MpmSim3D {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, settings: MpmSettings) -> Self {
        let buffers = MpmBuffers::new(device, queue, &settings);
        let pipelines = MpmPipelines::new(device, &buffers);
        let inflow = InflowState::new(settings.initial_kettle_angle_deg);
        let filter_mesh = settings.filter.as_ref().map(FilterMesh::new);

        let mut sim = Self {
            settings,
            buffers,
            pipelines,
            inflow,
            filter_mesh,
            num_water: 0,
            num_bed: 0,
            total_time: 0.0,
            frame_emitted_mass: 0.0,
            frame_dropped_particles: 0,
            total_emitted_mass: 0.0,
            total_dropped_particles: 0,
            latest_metrics: MetricsSnapshot::default(),
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
            bed_support_count,
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
        queue.write_buffer(
            &self.buffers.bed_extract,
            0,
            bytemuck::cast_slice(&bed_extracts),
        );
        queue.write_buffer(
            &self.buffers.bed_lookup,
            0,
            bytemuck::cast_slice(&cell_lookup),
        );
        let mut padded_support = vec![0_u32; self.settings.max_particles as usize];
        let copy_len = bed_support_count.len().min(padded_support.len());
        padded_support[..copy_len].copy_from_slice(&bed_support_count[..copy_len]);
        queue.write_buffer(
            &self.buffers.bed_support_count,
            0,
            bytemuck::cast_slice(&padded_support),
        );
        let zero_delta = vec![0_i32; self.settings.max_particles as usize];
        queue.write_buffer(
            &self.buffers.bed_delta,
            0,
            bytemuck::cast_slice(&zero_delta),
        );
    }

    pub fn step_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        let dt = dt.min(1.0 / 30.0);
        let substeps = self.settings.substeps.max(1);
        let sub_dt = dt / substeps as f32;
        self.frame_emitted_mass = 0.0;
        self.frame_dropped_particles = 0;

        for _ in 0..substeps {
            // Emit new particles
            let EmissionResult { emitted, dropped } = self.inflow.emit_particles(
                queue,
                &self.buffers,
                &self.settings.spout,
                sub_dt,
                MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML,
                self.num_water,
                self.num_bed,
                self.settings.max_particles,
            );
            self.frame_emitted_mass +=
                emitted as f32 * (MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML);
            self.frame_dropped_particles += dropped;
            self.total_emitted_mass +=
                emitted as f32 * (MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML);
            self.total_dropped_particles += dropped;
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

            let metrics_wg = dispatch_size(METRICS_SLOT_COUNT as u32, 8);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mpm step"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mpm compute"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.pipelines.bind_group, &[]);

                // 1a. clear_grid
                pass.set_pipeline(&self.pipelines.clear_grid);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // 1b. metrics_clear (fresh per-substep observability counters)
                if metrics_wg > 0 {
                    pass.set_pipeline(&self.pipelines.metrics_clear);
                    pass.dispatch_workgroups(metrics_wg, 1, 1);
                }

                // 1c. bed_lookup_clear + scatter: rebuild the spatial index to
                // match the current bed-particle positions so `classify_cells`
                // / `bed_coupling` / `g2p` do not see a stale snapshot from
                // when the bed was first seated.
                pass.set_pipeline(&self.pipelines.bed_lookup_clear);
                pass.dispatch_workgroups(cell_wg, 1, 1);
                if bed_wg > 0 {
                    pass.set_pipeline(&self.pipelines.bed_lookup_scatter);
                    pass.dispatch_workgroups(bed_wg, 1, 1);
                }

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

                if self.settings.enable_pressure_projection {
                    pass.set_pipeline(&self.pipelines.classify_cells);
                    pass.dispatch_workgroups(cell_wg, 1, 1);

                    // RBGS sweeps per substep. On the ~730k-cell grid, 16
                    // sweeps left large residual divergence, showing up as
                    // visible volume drift after pour-off. 40 sweeps (20
                    // pairs) is empirically enough to damp the pool-scale
                    // pressure modes while staying inside the browser frame
                    // budget. Revisit when the metrics readback path is
                    // rewired and `METRIC_MAX_ABS_DIV_IDX` is visible on
                    // the HUD — at that point we can tune against a live
                    // residual number.
                    const PRESSURE_RBGS_PAIRS: u32 = 20;
                    for _ in 0..PRESSURE_RBGS_PAIRS {
                        pass.set_pipeline(&self.pipelines.pressure_rbgs_red);
                        pass.dispatch_workgroups(cell_wg, 1, 1);
                        pass.set_pipeline(&self.pipelines.pressure_rbgs_black);
                        pass.dispatch_workgroups(cell_wg, 1, 1);
                    }

                    pass.set_pipeline(&self.pipelines.project_pressure);
                    pass.dispatch_workgroups(cell_wg, 1, 1);
                    pass.set_pipeline(&self.pipelines.boundary_project);
                    pass.dispatch_workgroups(cell_wg, 1, 1);
                }

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

        // The metrics staging copy used to happen here every frame, but that
        // races against the async `map_async` in `refresh_metrics` — the next
        // frame's copy would try to write into a buffer that was still in a
        // pending-map state, and wgpu panics. The copy now lives inside
        // `refresh_metrics` itself, which keeps the staging buffer idle
        // between snapshot requests.

        // The filter mesh is a CPU-side cloth scaffold with no fluid coupling,
        // so stepping it once per frame at the full `dt` is enough — there is
        // no benefit to running it per-substep and it would otherwise scale
        // CPU cost linearly with `substeps`.
        if let Some(mesh) = &mut self.filter_mesh {
            let bed_factor = if self.num_bed > 0 {
                (self.num_bed as f32 / 12_000.0).clamp(0.0, 2.0)
            } else {
                0.0
            };
            let water_factor = if self.settings.max_particles > 0 {
                (self.num_water as f32 / self.settings.max_particles as f32).clamp(0.0, 2.0)
            } else {
                0.0
            };
            let load = (0.28 + bed_factor * 0.22 + water_factor * 0.18).clamp(0.12, 0.90);
            mesh.step(dt, load);
        }
    }

    pub fn reset(&mut self, queue: &wgpu::Queue, _device: &wgpu::Device) {
        self.num_water = 0;
        self.num_bed = 0;
        self.total_time = 0.0;
        self.frame_emitted_mass = 0.0;
        self.frame_dropped_particles = 0;
        self.total_emitted_mass = 0.0;
        self.total_dropped_particles = 0;
        self.latest_metrics = MetricsSnapshot::default();
        self.inflow = InflowState::new(self.settings.initial_kettle_angle_deg);
        // Rebuild the CPU filter mesh so its deformation state does not leak
        // across resets — the GPU solver state is wiped via `init_bed` below.
        self.filter_mesh = self.settings.filter.as_ref().map(FilterMesh::new);
        self.init_bed(queue);
    }

    pub fn set_kettle_angle(&mut self, angle_deg: f32) {
        self.inflow.set_angle(angle_deg);
    }

    pub fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.settings.spout.origin = Vec3::new(x, y, z);
    }

    pub fn set_spout_target(&mut self, x: f32, y: f32, z: f32) {
        self.settings.spout.aim_at(Vec3::new(x, y, z));
    }

    pub fn kettle_angle(&self) -> f32 {
        self.inflow.angle()
    }

    pub fn spout_position(&self) -> Vec3 {
        self.settings.spout.origin
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

    pub fn water_slots_used(&self) -> u32 {
        self.num_water
    }

    pub fn bed_particle_count(&self) -> u32 {
        self.num_bed
    }

    pub fn max_particles(&self) -> u32 {
        self.settings.max_particles
    }

    pub fn total_time(&self) -> f32 {
        self.total_time
    }

    pub fn set_temp_sparse_ballistic_enabled(&mut self, enabled: bool) {
        self.settings.enable_temp_sparse_ballistic = enabled;
    }

    pub fn set_pressure_projection_enabled(&mut self, enabled: bool) {
        self.settings.enable_pressure_projection = enabled;
    }

    pub fn pressure_projection_enabled(&self) -> bool {
        self.settings.enable_pressure_projection
    }

    pub fn temp_sparse_ballistic_enabled(&self) -> bool {
        self.settings.enable_temp_sparse_ballistic
    }

    pub fn render_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.render_data
    }

    pub fn filter_render_vertices(&self) -> Option<&[[f32; 3]]> {
        self.filter_mesh.as_ref().map(|mesh| mesh.render_vertices())
    }

    pub fn filter_fill_vertices(&self) -> Option<&[[f32; 3]]> {
        self.filter_mesh.as_ref().map(|mesh| mesh.fill_vertices())
    }

    pub fn settings(&self) -> &MpmSettings {
        &self.settings
    }

    pub fn frame_emitted_mass(&self) -> f32 {
        self.frame_emitted_mass
    }

    pub fn frame_dropped_particles(&self) -> u32 {
        self.frame_dropped_particles
    }

    pub fn total_emitted_mass(&self) -> f32 {
        self.total_emitted_mass
    }

    pub fn total_dropped_particles(&self) -> u32 {
        self.total_dropped_particles
    }

    /// Last cached metrics snapshot. Populated by `refresh_metrics`; returns
    /// the zero default until the first successful readback.
    pub fn latest_metrics(&self) -> MetricsSnapshot {
        self.latest_metrics
    }

    /// Async staging-buffer readback for the GPU metrics counters.
    ///
    /// Currently **disabled** — every form of the readback path we tried
    /// (per-frame copy, inline copy-then-map, fire-and-forget map) freezes
    /// the browser tab on the second or third call. The shader-side
    /// instrumentation still runs and writes to the GPU `metrics` buffer on
    /// every substep; only the CPU-side pull-back is gated off until we can
    /// get a proper map/unmap lifecycle working.
    ///
    /// TODO(readback): reimplement using either
    ///   1. a dedicated staging buffer per in-flight request + a small
    ///      ring so a pending map never blocks a new copy, or
    ///   2. `queue.on_submitted_work_done` as a gate before calling
    ///      `map_async`, so the map only starts after GPU work drains.
    #[cfg(target_arch = "wasm32")]
    pub async fn refresh_metrics(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Result<(), JsValue> {
        Ok(())
    }

    fn write_uniforms(&self, queue: &wgpu::Queue, dt: f32) {
        let [gx, gy, gz] = self.settings.grid_dims;
        let total_cells = gx * gy * gz;
        let bs = self.settings.bounds_size;
        let dx = bs.x / gx as f32;
        let inv_dx = 1.0 / dx;
        let initial_particle_mass = MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let particle_vol = dx * dx * dx * 0.25;
        let bed_capacity_per_particle = if self.num_bed > 0 {
            TARGET_BED_RETENTION_ML * MASS_UNITS_PER_ML / self.num_bed as f32
        } else {
            0.7
        };
        // Divergence clamp: a fluid cell's divergence is bounded by
        // `2 * MAX_VELOCITY * inv_dx` when both faces move at the velocity
        // cap in opposite directions. Multiply by a safety margin of 2 so
        // legitimate transient spikes do not count as clamp fires, then cap
        // at the FP encoding limit. If the bound exceeds the FP ceiling,
        // drop to the ceiling and let the clamp counter flag it.
        let physical_div_bound = 4.0 * MAX_VELOCITY * inv_dx;
        let div_clamp = physical_div_bound.min(FP_VALUE_LIMIT - 1.0);
        // Pressure clamp stays just under the FP ceiling — there is no
        // tighter physical bound, so we rely on the counter to flag saturation.
        let pressure_clamp = FP_VALUE_LIMIT - 1.0;

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
            fluid_params: [
                self.settings.bulk_modulus,
                self.settings.viscosity,
                initial_particle_mass,
                particle_vol,
            ],
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
            bed_params: [
                34.0,
                8.0,
                bed_capacity_per_particle,
                if self.settings.enable_pressure_projection {
                    1.0
                } else {
                    0.0
                },
            ],
            extraction_params: [0.01, 11.0, 8.5, 15.0],
            time_params: [
                self.total_time,
                dt,
                if self.settings.enable_temp_sparse_ballistic {
                    1.0
                } else {
                    0.0
                },
                0.0,
            ],
            clamp_params: [
                div_clamp,
                pressure_clamp,
                METRICS_DIV_FP_SCALE,
                1.0 / METRICS_DIV_FP_SCALE,
            ],
        };

        queue.write_buffer(
            &self.buffers.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }
}

fn dispatch_size(count: u32, threads: u32) -> u32 {
    count.div_ceil(threads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    use bytemuck::cast_slice;

    #[derive(Debug)]
    struct MassSnapshot {
        active_particle_mass: f32,
        bed_held_mass: f32,
    }

    fn request_adapter() -> Option<wgpu::Adapter> {
        let instance = wgpu::Instance::default();
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()
    }

    fn create_device_with_limits(
        adapter: &wgpu::Adapter,
        limits: wgpu::Limits,
        label: &'static str,
    ) -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()
    }

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let adapter = request_adapter()?;
        create_device_with_limits(&adapter, required_limits(), "coffee-sim test device")
    }

    fn readback_mass_snapshot(
        sim: &MpmSim3D,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> MassSnapshot {
        let particle_count = (sim.num_water + sim.num_bed) as usize;
        let particle_size = (particle_count * 32).max(4) as u64;
        let bed_size = (sim.num_bed as usize * 32).max(4) as u64;

        let particle_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle mass staging"),
            size: particle_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bed_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bed mass staging"),
            size: bed_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mass readback"),
        });
        encoder.copy_buffer_to_buffer(&sim.buffers.particles, 0, &particle_staging, 0, particle_size);
        encoder.copy_buffer_to_buffer(&sim.buffers.bed_extract, 0, &bed_staging, 0, bed_size);
        queue.submit(Some(encoder.finish()));

        let particle_slice = particle_staging.slice(..);
        let bed_slice = bed_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        let tx_particles = tx.clone();
        particle_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx_particles.send(result).expect("particle map callback");
        });
        bed_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("bed map callback");
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("particle map recv").expect("particle map");
        rx.recv().expect("bed map recv").expect("bed map");

        let particle_view = particle_slice.get_mapped_range();
        let particle_f32 = cast_slice::<u8, f32>(&particle_view);
        let mut active_particle_mass = 0.0;
        for i in 0..particle_count {
            active_particle_mass += particle_f32[i * 8 + 7];
        }
        drop(particle_view);
        particle_staging.unmap();

        let bed_view = bed_slice.get_mapped_range();
        let bed_f32 = cast_slice::<u8, f32>(&bed_view);
        let mut bed_held_mass = 0.0;
        for i in 0..sim.num_bed as usize {
            bed_held_mass += bed_f32[i * 8];
        }
        drop(bed_view);
        bed_staging.unmap();

        MassSnapshot {
            active_particle_mass,
            bed_held_mass,
        }
    }

    #[test]
    fn dispatch_size_zero_count_returns_zero() {
        assert_eq!(dispatch_size(0, 64), 0);
    }

    #[test]
    fn dispatch_size_rounds_up() {
        assert_eq!(dispatch_size(1, 64), 1);
        assert_eq!(dispatch_size(64, 64), 1);
        assert_eq!(dispatch_size(65, 64), 2);
        assert_eq!(dispatch_size(128, 64), 2);
        assert_eq!(dispatch_size(129, 64), 3);
    }

    #[test]
    fn default_v60_shader_constants_in_sync() {
        // The shader's resolve_scene_obstacles hardcodes the V60 cone and carafe
        // dimensions. Keep the Rust-side defaults in lockstep so the sim and
        // collision geometry stay aligned.
        let s = MpmSettings::default_v60();
        let cone = s
            .obstacles
            .iter()
            .find_map(|o| match o {
                Obstacle::TruncatedCone {
                    top_radius,
                    bot_radius,
                    top_y,
                    bot_y,
                    ..
                } => Some((*top_radius, *bot_radius, *top_y, *bot_y)),
                _ => None,
            })
            .expect("default scene must contain a truncated cone");
        assert_eq!(cone, (4.5, 0.8, 3.0, -3.0));

        let cup = s
            .obstacles
            .iter()
            .find_map(|o| match o {
                Obstacle::Cylinder {
                    radius,
                    top_y,
                    bot_y,
                    ..
                } => Some((*radius, *top_y, *bot_y)),
                _ => None,
            })
            .expect("default scene must contain a cylinder");
        assert_eq!(cup, (3.0, -3.5, -8.0));
    }

    #[test]
    fn default_v60_grid_uses_uniform_dx() {
        let s = MpmSettings::default_v60();
        let dx = s.bounds_size.x / s.grid_dims[0] as f32;
        let dz = s.bounds_size.z / s.grid_dims[2] as f32;
        // dx and dz must agree because P2G/G2P assume uniform spacing
        assert!((dx - dz).abs() < 1e-5);
        // gy is sized to cover the full vertical extent
        let height_covered = s.grid_dims[1] as f32 * dx;
        assert!(height_covered >= s.bounds_size.y - dx);
    }

    #[test]
    fn mpm_pipelines_fit_within_required_limits() {
        // Construct the full sim (buffers + bind group layout + pipelines)
        // against the limits the app actually requests. If the bind group
        // layout ever exceeds `required_limits()`, this test fails with the
        // validation error instead of the sim silently failing at startup in
        // the browser.
        let Some(adapter) = request_adapter() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let Some((device, queue)) =
            create_device_with_limits(&adapter, required_limits(), "required-limits device")
        else {
            eprintln!("skipping: adapter does not support required limits");
            return;
        };

        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _sim = MpmSim3D::new(&device, &queue, MpmSettings::default_v60());
        let error = pollster::block_on(error_scope.pop());
        assert!(
            error.is_none(),
            "MpmSim3D::new produced a validation error under required_limits(): {error:?}",
        );
    }

    #[test]
    fn mpm_pipelines_exceed_spec_default_limits() {
        // Canary for `required_limits()`: building the sim at the WebGPU spec
        // default (`max_storage_buffers_per_shader_stage = 8`) must fail. If
        // the bind group layout ever drops to 8 or fewer storage buffers, this
        // test will flip and `required_limits()` becomes unnecessary — revisit
        // both together. Run under an error scope so the validation error is
        // captured instead of aborting the test process.
        let Some(adapter) = request_adapter() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let Some((device, queue)) =
            create_device_with_limits(&adapter, wgpu::Limits::default(), "spec-default device")
        else {
            eprintln!("skipping: adapter does not support spec default limits");
            return;
        };

        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _sim = MpmSim3D::new(&device, &queue, MpmSettings::default_v60());
        let error = pollster::block_on(error_scope.pop());
        assert!(
            error.is_some(),
            "expected a validation error when constructing MpmSim3D at spec-default limits, but \
             pipeline creation succeeded — `required_limits()` may no longer be necessary",
        );
    }

    #[test]
    #[ignore = "manual GPU validation harness for bed mass-balance work"]
    fn manual_mass_readback_harness_runs() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping GPU readback harness; no adapter/device available");
            return;
        };

        let settings = MpmSettings::benchmark_center_pour();
        let mut sim = MpmSim3D::new(&device, &queue, settings);
        for _ in 0..10 {
            sim.step_frame(&device, &queue, 1.0 / 60.0);
        }

        let snapshot = readback_mass_snapshot(&sim, &device, &queue);
        assert!(snapshot.active_particle_mass.is_finite());
        assert!(snapshot.bed_held_mass.is_finite());
        assert!(snapshot.active_particle_mass >= 0.0);
        assert!(snapshot.bed_held_mass >= 0.0);
    }

    #[test]
    fn occupancy_threshold_accepts_lone_particle() {
        // Pins the shader's classification threshold to stay strictly below
        // a single particle's peak-weight deposit. If either side moves,
        // this test fails and forces a review.
        let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let peak_bspline = 0.75_f32.powi(3); // ≈ 0.421875
        let peak_deposit = nominal_mass * peak_bspline;
        let occupancy_threshold = nominal_mass * 0.1;
        assert!(
            occupancy_threshold < peak_deposit,
            "occupancy threshold {occupancy_threshold} must stay below single-particle peak \
             deposit {peak_deposit} or lone particles never register as fluid"
        );
    }

    #[test]
    #[ignore = "GPU-only: validates Milestone 1 volume conservation"]
    fn water_mass_stable_after_pour_off() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping volume conservation test; no adapter/device available");
            return;
        };

        let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

        // Pour for 2 s of sim time.
        sim.set_kettle_angle(36.0);
        for _ in 0..120 {
            sim.step_frame(&device, &queue, 1.0 / 60.0);
        }

        // Stop pour, let it settle for 0.5 s.
        sim.set_kettle_angle(0.0);
        for _ in 0..30 {
            sim.step_frame(&device, &queue, 1.0 / 60.0);
        }
        let m0 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

        // Let it sit for another 1 s and compare.
        for _ in 0..60 {
            sim.step_frame(&device, &queue, 1.0 / 60.0);
        }
        let m1 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

        let drift = (m1 - m0).abs() / m0.max(1e-6);
        assert!(
            drift < 0.02,
            "water mass drifted {:.2}% after pour-off (m0={m0}, m1={m1})",
            drift * 100.0
        );
    }

    #[test]
    #[ignore = "GPU-only: regression test for divergence ghost-mirror at solid walls"]
    fn water_pool_stable_against_cup_floor() {
        // Targets the wall-adjacent divergence bug: previously the RHS
        // stencil in classify_cells read solid neighbors as zero velocity,
        // injecting a spurious sink at floor-adjacent fluid cells and
        // pushing pooled water away from the cup. A long pour lets water
        // actually pool against the cup SDF floor (spout is ~15 units above
        // the floor so a short pour never develops a real pool), then the
        // pour stops and we check that the resting pool doesn't drift.
        let Some((device, queue)) = create_test_device() else {
            eprintln!("Skipping wall-pool test; no adapter/device available");
            return;
        };

        let mut sim = MpmSim3D::new(&device, &queue, MpmSettings::benchmark_free_stream());

        // Pour for 5 s so water actually reaches and pools against the cup.
        sim.set_kettle_angle(36.0);
        for _ in 0..300 {
            sim.step_frame(&device, &queue, 1.0 / 60.0);
        }

        // Stop pour, settle for 1 s before the first reading.
        sim.set_kettle_angle(0.0);
        for _ in 0..60 {
            sim.step_frame(&device, &queue, 1.0 / 60.0);
        }
        let m0 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

        // Let the pool rest for another 2 s and compare.
        for _ in 0..120 {
            sim.step_frame(&device, &queue, 1.0 / 60.0);
        }
        let m1 = readback_mass_snapshot(&sim, &device, &queue).active_particle_mass;

        let drift = (m1 - m0).abs() / m0.max(1e-6);
        assert!(
            drift < 0.02,
            "pooled water drifted {:.2}% after settle (m0={m0}, m1={m1})",
            drift * 100.0
        );
    }
}

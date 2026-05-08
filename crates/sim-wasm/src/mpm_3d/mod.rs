use coffee_sim_core::Vec3;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::JsValue;

pub(crate) mod bed;
mod brew_config;
mod filter;
mod filter_mesh;
pub(crate) mod inflow;
#[cfg(test)]
mod physics_tests;
mod pipelines;
mod shader;
mod state;
mod units;

pub(crate) use filter::FilterConfig;
#[cfg(target_arch = "wasm32")]
pub(crate) use filter_mesh::{MAX_FILL_VERTEX_COUNT, MAX_RENDER_VERTEX_COUNT};

use bed::{BedConfig, BedInit};
use brew_config::DEFAULT_BREW;
use filter_mesh::FilterMesh;
use inflow::{EmissionResult, InflowState, SpoutSettings, MASS_UNITS_PER_ML};
use pipelines::MpmPipelines;
use state::{
    MpmBuffers, MpmUniforms, FP_SCALE, FP_VALUE_LIMIT, MAX_VELOCITY, METRICS_DIV_FP_SCALE,
    METRICS_SLOT_COUNT, NUM_THREADS, SDF_RES,
};

const TARGET_BED_RETENTION_ML: f32 = DEFAULT_BREW.target_bed_retention_ml;
pub(crate) const OBSTACLE_WALL_THICKNESS: f32 = 0.4;

/// Device limits required by the MPM compute pipeline.
///
/// The MPM bind group holds 9 storage buffers (particles, affine, grid,
/// grid_vel, render_data, bed_extract, bed_lookup, bed_delta, metrics) plus
/// one SDF texture. This stays within the 10-buffer cap that some WebGPU
/// adapters enforce. Any
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
    pub pressure_rbgs_pairs: u32,
    pub use_sdf_cache: bool,
    pub obstacles: Vec<Obstacle>,
    pub spout: SpoutSettings,
    pub initial_kettle_angle_deg: f32,
    pub filter: Option<FilterConfig>,
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
        let filter = FilterConfig::default();
        let bed = BedConfig::seated_in_filter(&filter);

        Self {
            bounds_size,
            grid_dims,
            max_particles: 220_000,
            substeps: 10,
            gravity: units::EARTH_GRAVITY_SIM_UNITS,
            bulk_modulus: 900.0,
            viscosity: DEFAULT_BREW.water_viscosity,
            render_radius: dx * 0.7,
            pressure_rbgs_pairs: 40,
            use_sdf_cache: true,
            obstacles: vec![
                v60_support_cone(&filter),
                Obstacle::Cylinder {
                    center: Vec3::ZERO,
                    radius: 3.0,
                    top_y: -3.5,
                    bot_y: -8.0,
                },
            ],
            spout: SpoutSettings::default(),
            initial_kettle_angle_deg: DEFAULT_BREW.initial_kettle_angle_deg,
            filter: Some(filter),
            bed: Some(bed),
        }
    }

    pub fn benchmark_free_stream() -> Self {
        let mut settings = Self::default_v60();
        settings.bed = None;
        settings.spout.origin = Vec3::new(0.0, 6.8, 0.0);
        settings.spout.aim_at(Vec3::new(0.0, -6.8, 0.0));
        settings.initial_kettle_angle_deg = DEFAULT_BREW.initial_kettle_angle_deg;
        settings
    }

    pub fn benchmark_center_pour() -> Self {
        let mut settings = Self::default_v60();
        settings.spout.origin = Vec3::new(0.0, 7.1, 0.0);
        settings.spout.aim_at(Vec3::new(0.0, 0.4, 0.0));
        settings.initial_kettle_angle_deg = DEFAULT_BREW.initial_kettle_angle_deg;
        settings
    }
}

fn v60_support_cone(filter: &FilterConfig) -> Obstacle {
    let top_y = 3.0;
    let bot_y = -3.0;
    let filter_height = (filter.top_y - filter.bot_y).max(1e-6);
    let filter_slope = (filter.top_radius - filter.bot_radius) / filter_height;
    let bot_radius = DEFAULT_BREW.dripper_outlet_radius;

    Obstacle::TruncatedCone {
        center: Vec3::ZERO,
        top_radius: bot_radius + filter_slope * (top_y - bot_y),
        bot_radius,
        top_y,
        bot_y,
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
        // Match the shader's four-lane bed_delta layout: water plus impulse xyz.
        let zero_delta = vec![0_i32; self.settings.max_particles as usize * 4];
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
            encoder.clear_buffer(&self.buffers.grid, 0, None);
            encoder.clear_buffer(&self.buffers.grid_vel, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mpm compute"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.pipelines.bind_group, &[]);

                // 1a. metrics_clear (fresh per-substep observability counters)
                if metrics_wg > 0 {
                    pass.set_pipeline(&self.pipelines.metrics_clear);
                    pass.dispatch_workgroups(metrics_wg, 1, 1);
                }

                // 1b. bed_lookup_clear + scatter: rebuild the spatial index
                // so classify_cells / bed_coupling / g2p see current
                // bed-particle positions.
                pass.set_pipeline(&self.pipelines.bed_lookup_clear);
                pass.dispatch_workgroups(cell_wg, 1, 1);
                if bed_wg > 0 {
                    pass.set_pipeline(&self.pipelines.bed_lookup_scatter);
                    pass.dispatch_workgroups(bed_wg, 1, 1);
                }

                // 2. p2g
                if particle_wg > 0 {
                    pass.set_pipeline(&self.pipelines.p2g);
                    pass.dispatch_workgroups(particle_wg, 1, 1);
                }

                // 3. grid_update
                pass.set_pipeline(&self.pipelines.grid_update);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // 4. boundary_project
                pass.set_pipeline(&self.pipelines.boundary_project);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // Pressure projection: classify cells, RBGS pressure
                // solve, velocity correction, then re-project boundaries.
                pass.set_pipeline(&self.pipelines.classify_cells);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                for _ in 0..self.settings.pressure_rbgs_pairs {
                    pass.set_pipeline(&self.pipelines.pressure_rbgs_red);
                    pass.dispatch_workgroups(cell_wg, 1, 1);
                    pass.set_pipeline(&self.pipelines.pressure_rbgs_black);
                    pass.dispatch_workgroups(cell_wg, 1, 1);
                }

                pass.set_pipeline(&self.pipelines.project_pressure);
                pass.dispatch_workgroups(cell_wg, 1, 1);
                pass.set_pipeline(&self.pipelines.boundary_project);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // Packing pressure reuses the projection scratch lanes before
                // viscosity overwrites the grid momentum lanes with temporary
                // FP-encoded velocity scratch.
                pass.set_pipeline(&self.pipelines.packing_prepare);
                pass.dispatch_workgroups(cell_wg, 1, 1);
                pass.set_pipeline(&self.pipelines.packing_apply);
                pass.dispatch_workgroups(cell_wg, 1, 1);
                pass.set_pipeline(&self.pipelines.boundary_project);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // Viscosity is split after the pressure and packing
                // projections so neither correction can immediately
                // reintroduce the high-frequency pool velocities that
                // diffusion just removed.
                pass.set_pipeline(&self.pipelines.viscosity_prepare);
                pass.dispatch_workgroups(cell_wg, 1, 1);
                pass.set_pipeline(&self.pipelines.viscosity_apply);
                pass.dispatch_workgroups(cell_wg, 1, 1);
                pass.set_pipeline(&self.pipelines.boundary_project);
                pass.dispatch_workgroups(cell_wg, 1, 1);

                // 6. g2p
                if particle_wg > 0 {
                    pass.set_pipeline(&self.pipelines.g2p);
                    pass.dispatch_workgroups(particle_wg, 1, 1);
                }

                // 7. bed_coupling (after g2p so absorption uses projected
                //    velocities and remains the sole bed storage transfer)
                if particle_wg > 0 {
                    pass.set_pipeline(&self.pipelines.bed_coupling);
                    pass.dispatch_workgroups(particle_wg, 1, 1);
                }

                // 8. extraction_advect (consumes bed water delta from bed_coupling)
                if bed_wg > 0 {
                    pass.set_pipeline(&self.pipelines.extraction_advect);
                    pass.dispatch_workgroups(bed_wg, 1, 1);
                }

                // 9. bed_dynamics
                if bed_wg > 0 {
                    pass.set_pipeline(&self.pipelines.bed_dynamics);
                    pass.dispatch_workgroups(bed_wg, 1, 1);
                }

                // 10. prepare_render
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

        // The filter mesh is static CPU render geometry, not solver state, so
        // there is no per-frame mesh work here.
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
        // Rebuild the CPU filter mesh so reset/scene changes keep render
        // geometry aligned with the active filter config.
        self.filter_mesh = self.settings.filter.as_ref().map(FilterMesh::new);
        self.init_bed(queue);
    }

    pub fn set_kettle_angle(&mut self, angle_deg: f32) {
        self.inflow.set_angle(angle_deg);
    }

    pub fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.settings.spout.translate_origin_to(Vec3::new(x, y, z));
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

    pub fn exit_speed_m_s(&self) -> f32 {
        units::sim_speed_to_meters_per_second(self.inflow.exit_speed())
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

    pub fn frame_emitted_ml(&self) -> f32 {
        self.frame_emitted_mass / MASS_UNITS_PER_ML
    }

    pub fn frame_dropped_particles(&self) -> u32 {
        self.frame_dropped_particles
    }

    pub fn total_emitted_mass(&self) -> f32 {
        self.total_emitted_mass
    }

    pub fn total_emitted_ml(&self) -> f32 {
        self.total_emitted_mass / MASS_UNITS_PER_ML
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
                u32::from(self.settings.use_sdf_cache),
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
            fp_params: [
                FP_SCALE,
                1.0 / FP_SCALE,
                MAX_VELOCITY,
                DEFAULT_BREW.dripper_outlet_radius,
            ],
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
            inflow_params: [
                self.settings.spout.nozzle_radius,
                DEFAULT_BREW.water_sample_radius_dx,
                DEFAULT_BREW.bed_sample_radius_dx,
                DEFAULT_BREW.filter_absorption_rate_s,
            ],
            sdf_params: [SDF_RES as f32, 0.3, 0.0, 0.05],
            // Tie bed retention to an overall retained-water target so the bed
            // wets realistically without swallowing most of the brew.
            bed_params: [
                DEFAULT_BREW.water_kinematic_viscosity_m2_s,
                DEFAULT_BREW.bed_absorption_rate,
                bed_capacity_per_particle,
                DEFAULT_BREW.min_bed_permeability_m2,
            ],
            extraction_params: [
                0.01,
                DEFAULT_BREW.bed_compaction_rate,
                8.5,
                DEFAULT_BREW.bed_impact_rate,
            ],
            time_params: [
                self.total_time,
                dt,
                DEFAULT_BREW.bed_pore_capacity_scale,
                DEFAULT_BREW.bed_pore_overfill_alpha,
            ],
            clamp_params: [
                div_clamp,
                pressure_clamp,
                METRICS_DIV_FP_SCALE,
                1.0 / METRICS_DIV_FP_SCALE,
            ],
            projection_params: [32.0, 2.0, 1.20, DEFAULT_BREW.bed_surface_void_scale],
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
        let filter = FilterConfig::default();
        let filter_slope = (filter.top_radius - filter.bot_radius) / (filter.top_y - filter.bot_y);
        let expected_top_radius =
            DEFAULT_BREW.dripper_outlet_radius + filter_slope * (cone.2 - cone.3);
        assert!((cone.0 - expected_top_radius).abs() < 0.01);
        assert!((cone.1 - DEFAULT_BREW.dripper_outlet_radius).abs() < 1e-6);
        assert_eq!((cone.2, cone.3), (3.0, -3.0));
        let cone_slope = (cone.0 - cone.1) / (cone.2 - cone.3);
        assert!((cone_slope - filter_slope).abs() < 1e-5);

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
        assert!(shader::MPM_COMPUTE_SHADER.contains("const OBSTACLE_WALL_THICKNESS: f32 = 0.4;"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn dripper_top_radius()"));
        assert!(shader::MPM_COMPUTE_SHADER
            .contains("mix(dripper_outlet_radius(), dripper_top_radius(), t)"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn viscosity_prepare("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn viscosity_apply("));
    }

    #[test]
    fn default_v60_grid_uses_uniform_dx() {
        let s = MpmSettings::default_v60();
        let dx = s.bounds_size.x / s.grid_dims[0] as f32;
        let dz = s.bounds_size.z / s.grid_dims[2] as f32;
        assert!((dx - dz).abs() < 1e-5);
        let height_covered = s.grid_dims[1] as f32 * dx;
        assert!(height_covered >= s.bounds_size.y - dx);
    }

    #[test]
    fn default_v60_uses_slow_gooseneck_angle() {
        let s = MpmSettings::default_v60();
        assert!(s.initial_kettle_angle_deg <= 10.0);
        assert!(s.spout.max_flow_rate_ml_s <= 12.0);

        let mut inflow = InflowState::new(s.initial_kettle_angle_deg);
        inflow.update(&s.spout);
        assert!(inflow.exit_speed() * units::METERS_PER_SIM_UNIT <= 0.13);
        assert!(s.spout.max_exit_speed * units::METERS_PER_SIM_UNIT <= 0.50);
    }

    #[test]
    fn expanded_grid_atomic_lanes_fit_storage_binding_limit() {
        let s = MpmSettings::default_v60();
        let total_cells = s.grid_dims[0] as u64 * s.grid_dims[1] as u64 * s.grid_dims[2] as u64;
        // `grid` is bound in WGSL as `array<atomic<i32>>`, not as a vecN array.
        // Extra lanes are scalar structure-of-arrays slices addressed as
        // `lane * total_cells + cell`, so there is no vec4 stride to preserve.
        assert_eq!(std::mem::size_of::<i32>(), 4);
        let proposed_grid_lanes = 6_u64;
        let proposed_grid_bytes =
            proposed_grid_lanes * total_cells * std::mem::size_of::<i32>() as u64;
        let limit = required_limits().max_storage_buffer_binding_size as u64;

        assert!(
            proposed_grid_bytes <= limit,
            "proposed {proposed_grid_lanes}-lane grid atomics buffer is {proposed_grid_bytes} \
             bytes, exceeding max_storage_buffer_binding_size={limit}"
        );
    }

    #[test]
    fn occupancy_threshold_accepts_lone_particle() {
        let nominal_mass = inflow::MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let peak_bspline = 0.75_f32.powi(3);
        let peak_deposit = nominal_mass * peak_bspline;
        let occupancy_threshold = nominal_mass * 0.1;
        assert!(
            occupancy_threshold < peak_deposit,
            "occupancy threshold {occupancy_threshold} must stay below single-particle peak \
             deposit {peak_deposit} or lone particles never register as fluid"
        );
    }
}

use coffee_sim_core::Vec3;

use crate::boundaries::{cup::CupConfig, sdf};
use crate::diagnostics::{MetricsSnapshot, PassTimings, WaterDiagnostics};
use crate::engine::{ParticleView, SimulationEngine};
use crate::extraction::{heat, kinetics};
use crate::materials::{water, DEFAULT_BREW};
use crate::render_pack;
use crate::scene::seeds::{self, ParticleKind, SeedParticle};
use crate::scene::{sim_speed_to_meters_per_second, DebugScene, SceneSpec, SimSettings};

mod constraints;
mod hash;
mod inflow;
mod passes;
mod pipelines;
mod shader;
mod state;
#[cfg(test)]
mod tests;

use constraints::ConstraintConfig;
use hash::NeighborHash;
use inflow::InflowState;
use pipelines::XpbdPipelines;
use state::{XpbdState, TYPE_COFFEE, TYPE_WATER};

#[cfg(target_arch = "wasm32")]
fn profiler_now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
fn profiler_now_ms() -> f64 {
    0.0
}

pub(crate) struct XpbdEngine {
    state: XpbdState,
    inflow: InflowState,
    constraints: ConstraintConfig,
    hash: NeighborHash,
    _pipelines: XpbdPipelines,
}

impl XpbdEngine {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, scene: SceneSpec) -> Self {
        let settings = scene.settings();
        let mut engine = Self {
            inflow: InflowState::new(settings.initial_water_speed_m_s),
            constraints: ConstraintConfig::default(),
            hash: NeighborHash::new(settings.render_radius * 2.4),
            _pipelines: XpbdPipelines::new(device),
            state: XpbdState::new(device, settings),
        };
        engine.seed_scene(queue, scene);
        engine
    }

    pub(crate) fn flow_rate_ml_s(&self) -> f32 {
        self.inflow.flow_rate()
    }

    pub(crate) fn exit_speed(&self) -> f32 {
        self.inflow.exit_speed()
    }

    pub(crate) fn exit_speed_m_s(&self) -> f32 {
        sim_speed_to_meters_per_second(self.inflow.exit_speed())
    }

    pub(crate) fn total_time(&self) -> f32 {
        self.state.total_time
    }

    pub(crate) fn frame_emitted_mass(&self) -> f32 {
        self.state.frame_emitted_mass
    }

    pub(crate) fn total_emitted_mass(&self) -> f32 {
        self.state.total_emitted_mass
    }

    pub(crate) fn frame_dropped_particles(&self) -> u32 {
        self.state.frame_dropped_particles
    }

    pub(crate) fn total_dropped_particles(&self) -> u32 {
        self.state.total_dropped_particles
    }

    pub(crate) fn last_iterations(&self) -> u32 {
        self.state.last_iterations
    }

    pub(crate) fn estimated_cup_tds(&self) -> f32 {
        self.state.latest_metrics.cup_tds
    }

    pub(crate) fn estimated_extraction_yield(&self) -> f32 {
        self.state.latest_metrics.extraction_yield
    }

    pub(crate) fn water_diagnostics(&self) -> WaterDiagnostics {
        let mut diag = WaterDiagnostics {
            sim_time_s: self.state.total_time,
            emitted_ml: self.state.total_emitted_mass / water::MASS_UNITS_PER_ML,
            cup_water_mass: self.state.cup_water_mass,
            cup_solute_mass: self.state.cup_solute_mass,
            cup_tds: self.state.latest_metrics.cup_tds,
            extraction_yield: self.state.latest_metrics.extraction_yield,
            ..WaterDiagnostics::default()
        };
        let mut count = 0u32;
        let mut mass = 0.0;
        let mut ke = 0.0;
        let mut momentum = Vec3::ZERO;
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut centroid = Vec3::ZERO;
        let mut solute = 0.0;
        for ((p, v), m) in self
            .state
            .pos_type
            .iter()
            .zip(&self.state.vel_mass)
            .zip(&self.state.material)
        {
            if (p[3] - TYPE_WATER).abs() > 0.1 {
                continue;
            }
            let pos = Vec3::new(p[0], p[1], p[2]);
            let vel = Vec3::new(v[0], v[1], v[2]);
            let pm = v[3];
            count += 1;
            mass += pm;
            ke += 0.5 * pm * vel.length_squared();
            momentum = momentum + vel * pm;
            centroid = centroid + pos;
            min = Vec3::new(min.x.min(pos.x), min.y.min(pos.y), min.z.min(pos.z));
            max = Vec3::new(max.x.max(pos.x), max.y.max(pos.y), max.z.max(pos.z));
            solute += m[0];
            diag.all_finite &= p.iter().chain(v.iter()).all(|x| x.is_finite());
            diag.max_speed = diag.max_speed.max(vel.length());
        }
        if count > 0 {
            centroid = centroid / count as f32;
        } else {
            min = Vec3::ZERO;
            max = Vec3::ZERO;
        }
        diag.active_count = count;
        diag.pool_count = count;
        diag.active_mass = mass;
        diag.active_mass_ml = mass / water::MASS_UNITS_PER_ML;
        diag.rest_volume_ml = diag.active_mass_ml;
        diag.current_volume_ml = diag.active_mass_ml;
        diag.kinetic_energy = ke;
        diag.rms_speed = (2.0 * ke / mass.max(1e-6)).sqrt();
        diag.momentum = momentum;
        diag.momentum_magnitude = momentum.length();
        diag.centroid = centroid;
        diag.min = min;
        diag.max = max;
        diag.extent = max - min;
        diag.dissolved_solute_mass = solute;
        diag.mean_tds = solute / mass.max(1e-6);
        diag
    }

    fn seed_scene(&mut self, queue: &wgpu::Queue, scene: SceneSpec) {
        self.clear_particles();
        for p in seeds::seed_bed(&self.state.settings) {
            self.push_seed(p);
        }
        if let SceneSpec::Debug(debug) = scene {
            for p in seeds::seed_debug_water(debug, &self.state.settings) {
                self.push_seed(p);
            }
            if matches!(debug, DebugScene::HighVelocityJetImpact) {
                self.set_exit_speed_m_s(0.48);
            }
        }
        self.pack_render(queue);
        self.update_metrics(queue, 0.0, 0.0);
    }

    fn clear_particles(&mut self) {
        self.state.pos_type.clear();
        self.state.vel_mass.clear();
        self.state.props.clear();
        self.state.material.clear();
        self.state.num_water = 0;
        self.state.num_coffee = 0;
        self.state.total_time = 0.0;
        self.state.frame_emitted_mass = 0.0;
        self.state.total_emitted_mass = 0.0;
        self.state.frame_dropped_particles = 0;
        self.state.total_dropped_particles = 0;
        self.state.cup_water_mass = 0.0;
        self.state.cup_solute_mass = 0.0;
    }

    fn push_seed(&mut self, p: SeedParticle) {
        if self.state.pos_type.len() >= self.state.settings.max_particles as usize {
            return;
        }
        let kind = match p.kind {
            ParticleKind::Water => {
                self.state.num_water += 1;
                TYPE_WATER
            }
            ParticleKind::Coffee => {
                self.state.num_coffee += 1;
                TYPE_COFFEE
            }
        };
        self.state.pos_type.push([p.pos.x, p.pos.y, p.pos.z, kind]);
        self.state
            .vel_mass
            .push([p.vel.x, p.vel.y, p.vel.z, p.mass]);
        self.state.props.push([p.radius, p.temperature_c, 1.0, 0.0]);
        self.state
            .material
            .push([p.solute.max(p.wetness), p.fast_solute, p.slow_solute, 0.0]);
    }

    fn integrate_substep(&mut self, dt: f32) {
        let bounds = self.state.settings.bounds_size;
        let filter = self.state.settings.filter.clone();
        let cup = CupConfig::default();
        let gravity = self.state.settings.gravity;
        let damping = 1.0 - self.constraints.xsph_viscosity * dt;
        for i in 0..self.state.pos_type.len() {
            let ty = self.state.pos_type[i][3];
            let radius = self.state.props[i][0];
            let mut pos = Vec3::new(
                self.state.pos_type[i][0],
                self.state.pos_type[i][1],
                self.state.pos_type[i][2],
            );
            let mut vel = Vec3::new(
                self.state.vel_mass[i][0],
                self.state.vel_mass[i][1],
                self.state.vel_mass[i][2],
            );
            if ty == TYPE_WATER {
                vel.y += gravity * dt;
                vel = vel * damping;
            } else if ty == TYPE_COFFEE {
                vel.y += gravity
                    * dt
                    * (0.08 + 0.18 * (1.0 - self.state.material[i][0].clamp(0.0, 1.0)));
                vel = vel * (1.0 - (2.0 + self.state.material[i][0] * 6.0) * dt).clamp(0.0, 1.0);
            }
            pos = pos + vel * dt;
            pos = sdf::project_bounds(pos, radius, bounds);
            if let Some(filter) = &filter {
                pos = sdf::project_filter_inside(pos, radius, filter);
            }
            pos = sdf::project_cup_inside(pos, radius, cup);
            self.state.vel_mass[i][0] = (pos.x - self.state.pos_type[i][0]) / dt.max(1e-6);
            self.state.vel_mass[i][1] = (pos.y - self.state.pos_type[i][1]) / dt.max(1e-6);
            self.state.vel_mass[i][2] = (pos.z - self.state.pos_type[i][2]) / dt.max(1e-6);
            self.state.pos_type[i][0] = pos.x;
            self.state.pos_type[i][1] = pos.y;
            self.state.pos_type[i][2] = pos.z;
        }
        self.apply_local_coupling(dt);
        self.drain_cup();
    }

    fn apply_local_coupling(&mut self, dt: f32) {
        let len = self.state.pos_type.len();
        for wi in 0..len {
            if self.state.pos_type[wi][3] != TYPE_WATER {
                continue;
            }
            let wp = Vec3::new(
                self.state.pos_type[wi][0],
                self.state.pos_type[wi][1],
                self.state.pos_type[wi][2],
            );
            for ci in 0..len {
                if self.state.pos_type[ci][3] != TYPE_COFFEE {
                    continue;
                }
                let cp = Vec3::new(
                    self.state.pos_type[ci][0],
                    self.state.pos_type[ci][1],
                    self.state.pos_type[ci][2],
                );
                let delta = wp - cp;
                let dist2 = delta.length_squared();
                let influence = (self.state.props[wi][0] + self.state.props[ci][0]) * 2.0;
                if dist2 > influence * influence || dist2 <= 1e-8 {
                    continue;
                }
                let wet = &mut self.state.material[ci][0];
                let uptake =
                    (DEFAULT_BREW.bed_absorption_rate * dt * (1.0 - *wet)).clamp(0.0, 0.08);
                *wet = (*wet + uptake).clamp(0.0, 1.0);
                let rel = Vec3::new(
                    self.state.vel_mass[wi][0] - self.state.vel_mass[ci][0],
                    self.state.vel_mass[wi][1] - self.state.vel_mass[ci][1],
                    self.state.vel_mass[wi][2] - self.state.vel_mass[ci][2],
                );
                let drag = (0.18 * dt * (1.0 + *wet)).clamp(0.0, 0.35);
                self.state.vel_mass[wi][0] -= rel.x * drag;
                self.state.vel_mass[wi][1] -= rel.y * drag;
                self.state.vel_mass[wi][2] -= rel.z * drag;
                self.state.vel_mass[ci][0] += rel.x * drag * 0.025;
                self.state.vel_mass[ci][1] += rel.y * drag * 0.025;
                self.state.vel_mass[ci][2] += rel.z * drag * 0.025;

                let water_conc = self.state.material[wi][0] / self.state.vel_mass[wi][3].max(1e-6);
                let (fast, slow) = kinetics::two_pool_transfer(
                    self.state.material[ci][1],
                    self.state.material[ci][2],
                    water_conc,
                    self.state.props[wi][1],
                    rel.length(),
                    dt,
                );
                self.state.material[ci][1] -= fast;
                self.state.material[ci][2] -= slow;
                self.state.material[wi][0] += fast + slow;
                let (tw, tc) = heat::exchange_temperatures(
                    self.state.props[wi][1],
                    self.state.props[ci][1],
                    dt,
                );
                self.state.props[wi][1] = tw;
                self.state.props[ci][1] = tc;
                break;
            }
        }
    }

    fn drain_cup(&mut self) {
        for i in 0..self.state.pos_type.len() {
            if self.state.pos_type[i][3] != TYPE_WATER {
                continue;
            }
            if self.state.pos_type[i][1] <= -7.95 {
                self.state.cup_water_mass += self.state.vel_mass[i][3];
                self.state.cup_solute_mass += self.state.material[i][0];
                self.state.pos_type[i][3] = 2.0;
            }
        }
    }

    fn emit(&mut self, dt: f32) {
        let emission = self.inflow.emit(
            &self.state.settings.spout,
            dt,
            self.state.settings.render_radius * 0.75,
        );
        let available = self
            .state
            .settings
            .max_particles
            .saturating_sub(self.state.pos_type.len() as u32);
        let count = emission.particles.len().min(available as usize);
        for (pos, vel, props, mat) in emission.particles.into_iter().take(count) {
            self.state.pos_type.push(pos);
            self.state.vel_mass.push(vel);
            self.state.props.push(props);
            self.state.material.push(mat);
            self.state.num_water += 1;
            self.state.frame_emitted_mass += vel[3];
            self.state.total_emitted_mass += vel[3];
        }
        let dropped = emission.requested.saturating_sub(count as u32);
        self.state.frame_dropped_particles += dropped;
        self.state.total_dropped_particles += dropped;
    }

    fn pack_render(&mut self, queue: &wgpu::Queue) {
        render_pack::pack_instances(
            &self.state.pos_type,
            &self.state.props,
            &self.state.material,
            &mut self.state.render_instances,
        );
        if !self.state.render_instances.is_empty() {
            queue.write_buffer(
                &self.state.buffers.render_data,
                0,
                bytemuck::cast_slice(&self.state.render_instances),
            );
        }
    }

    fn update_metrics(&mut self, queue: &wgpu::Queue, max_residual: f32, mean_residual: f32) {
        let active_water_mass: f32 = self
            .state
            .pos_type
            .iter()
            .zip(&self.state.vel_mass)
            .filter(|(p, _)| p[3] == TYPE_WATER)
            .map(|(_, v)| v[3])
            .sum();
        let active_solute: f32 = self
            .state
            .pos_type
            .iter()
            .zip(&self.state.material)
            .filter(|(p, _)| p[3] == TYPE_WATER)
            .map(|(_, m)| m[0])
            .sum();
        let cup_tds = self.state.cup_solute_mass / self.state.cup_water_mass.max(1e-6);
        let extraction_yield = (active_solute + self.state.cup_solute_mass)
            / (DEFAULT_BREW.coffee_dose_g * water::MASS_UNITS_PER_ML);
        self.state.latest_metrics = MetricsSnapshot {
            max_abs_div: max_residual,
            fluid_cells: self.hash.estimate_active_cells(&self.state.pos_type),
            div_clamp_fires: 0,
            pressure_clamp_fires: 0,
            mass_overflow_fires: self.state.total_dropped_particles,
            projection_residual_max_abs_div: max_residual,
            projection_residual_mean_abs_div: mean_residual,
            projection_residual_cells: self.state.num_water,
            mean_tds: active_solute / active_water_mass.max(1e-6),
            cup_tds,
            extraction_yield,
        };
        let raw = [
            (max_residual * 1024.0) as u32,
            self.state.latest_metrics.fluid_cells,
            0,
            0,
            self.state.total_dropped_particles,
            (active_water_mass * 1024.0) as u32,
            (active_solute * 65536.0) as u32,
            (self.state.cup_water_mass * 1024.0) as u32,
            (self.state.cup_solute_mass * 65536.0) as u32,
            (max_residual * 1024.0) as u32,
            (mean_residual * 16.0) as u32,
            self.state.num_water,
        ];
        queue.write_buffer(&self.state.buffers.metrics, 0, bytemuck::cast_slice(&raw));
    }
}

impl SimulationEngine for XpbdEngine {
    fn step_frame(&mut self, _device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        let start_ms = profiler_now_ms();
        self.state.frame_emitted_mass = 0.0;
        self.state.frame_dropped_particles = 0;
        let dt = dt.min(1.0 / 30.0);
        let substeps = self.state.settings.substeps.max(1);
        let sub_dt = dt / substeps as f32;
        for _ in 0..substeps {
            self.emit(sub_dt);
            for _ in 0..self.state.settings.xpbd_iterations {
                self.integrate_substep(sub_dt / self.state.settings.xpbd_iterations.max(1) as f32);
            }
            self.state.total_time += sub_dt;
        }
        self.state.last_iterations = self.state.settings.xpbd_iterations;
        self.pack_render(queue);
        self.update_metrics(queue, 0.0, 0.0);
        self.state.profiler.record_frame(
            (profiler_now_ms() - start_ms).max(0.0) as f32,
            PassTimings {
                emit_predict_ms: 0.0,
                hash_ms: 0.0,
                constraints_ms: 0.0,
                boundaries_ms: 0.0,
                extraction_ms: 0.0,
                render_pack_ms: 0.0,
                metrics_ms: 0.0,
            },
        );
    }

    fn reset(&mut self, _device: &wgpu::Device, queue: &wgpu::Queue) {
        self.seed_scene(queue, SceneSpec::DefaultV60);
    }

    fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: SceneSpec) {
        *self = Self::new(device, queue, scene);
    }

    fn particle_view(&self) -> ParticleView<'_> {
        ParticleView {
            positions: &self.state.pos_type,
            velocities: &self.state.vel_mass,
            props: &self.state.props,
            material: &self.state.material,
            water_count: self.state.num_water,
            coffee_count: self.state.num_coffee,
        }
    }

    fn render_buffer(&self) -> &wgpu::Buffer {
        &self.state.buffers.render_data
    }

    fn particle_count(&self) -> usize {
        self.state.render_instances.len()
    }

    fn water_slots_used(&self) -> u32 {
        self.state.num_water
    }

    fn bed_particle_count(&self) -> u32 {
        self.state.num_coffee
    }

    fn max_particles(&self) -> u32 {
        self.state.settings.max_particles
    }

    fn settings(&self) -> &SimSettings {
        &self.state.settings
    }

    fn metrics_buffer(&self) -> &wgpu::Buffer {
        &self.state.buffers.metrics
    }

    fn latest_metrics(&self) -> MetricsSnapshot {
        self.state.latest_metrics
    }

    fn set_exit_speed_m_s(&mut self, speed_m_s: f32) {
        self.inflow
            .set_exit_speed_m_s(speed_m_s, &self.state.settings.spout);
    }

    fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.state
            .settings
            .spout
            .translate_origin_to(Vec3::new(x, y, z));
        self.inflow.update(&self.state.settings.spout);
    }

    fn spout_position(&self) -> Vec3 {
        self.state.settings.spout.origin
    }
}

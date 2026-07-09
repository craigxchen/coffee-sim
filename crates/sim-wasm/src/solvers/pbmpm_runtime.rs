use crate::emission::{EmissionInput, PourEvent};
use crate::engine::Scene;
use crate::models::Materials;
use crate::solvers::base::{
    CommonMetrics, FrameContext, FrameSnapshot, FrameSolver, ResetContext, SceneSpec,
    Solver as RewriteSolver,
};
use crate::solvers::pbmpm::PbmpmSolver;
use crate::ui::{RenderMaterial, RenderView, STANDARD_WATER_RENDER_RADIUS};
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;
use coffee_sim_core::Vec3;

const ML_PER_SIM_UNIT3: f32 = 5.20;
const SIM_UNITS_PER_METER: f32 = 27.7;
const DEFAULT_WATER_VELOCITY_M_S: f32 = 0.12;
const DEFAULT_SPOUT_UI: [f32; 3] = [0.0, 2.5, 0.0];

pub(crate) struct PbmpmFrameSolver {
    core: PbmpmSolver,
    scene_spec: SceneSpec,
    scene: Scene,
    materials: Materials,
    config: Config,
    particles: ParticleBuffers,
    water_velocity_m_s: f32,
    spout_ui: [f32; 3],
    sim_time_s: f32,
    total_emitted_mass: f32,
    frame_emitted_mass: f32,
}

impl PbmpmFrameSolver {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_spec: &SceneSpec,
    ) -> Result<Self, String> {
        let (scene, materials, config) = setup_for_scene_spec(scene_spec)?;
        let gpu = GpuContext::from_device_queue(device, queue);
        let core = <PbmpmSolver as RewriteSolver>::build(&scene, &materials, &config, &gpu);
        let particles = RewriteSolver::particles(&core);
        Ok(Self {
            core,
            scene_spec: scene_spec.clone(),
            scene,
            materials,
            config,
            particles,
            water_velocity_m_s: DEFAULT_WATER_VELOCITY_M_S,
            spout_ui: DEFAULT_SPOUT_UI,
            sim_time_s: 0.0,
            total_emitted_mass: 0.0,
            frame_emitted_mass: 0.0,
        })
    }

    fn refresh_particles(&mut self) {
        self.particles = RewriteSolver::particles(&self.core);
    }

    fn emission_input(&self) -> EmissionInput {
        if !pbmpm_scene_accepts_pour(&self.scene_spec) {
            return EmissionInput::default();
        }
        EmissionInput {
            kettle_pos: self.spout_ui,
            flow_rate: flow_rate_for_velocity(
                self.water_velocity_m_s,
                self.config.nozzle_radius,
                self.config.discharge_coeff,
            ),
            pour_angle: 0.0,
            event: PourEvent::None,
        }
    }

    fn common_metrics(&self) -> CommonMetrics {
        let rewrite_metrics = RewriteSolver::metrics(&self.core);
        let profile = RewriteSolver::profile(&self.core);
        let flow_rate = if pbmpm_scene_accepts_pour(&self.scene_spec) {
            flow_rate_for_velocity(
                self.water_velocity_m_s,
                self.config.nozzle_radius,
                self.config.discharge_coeff,
            )
        } else {
            0.0
        };
        CommonMetrics {
            particle_count: rewrite_metrics.particle_count as usize,
            water_slots_used: self.core.active_count(),
            bed_particle_count: 0,
            max_particles: self.core.capacity(),
            sim_time_s: self.sim_time_s,
            frame_emitted_mass: self.frame_emitted_mass,
            frame_emitted_ml: self.frame_emitted_mass * ML_PER_SIM_UNIT3,
            total_emitted_mass: self.total_emitted_mass,
            total_emitted_ml: self.total_emitted_mass * ML_PER_SIM_UNIT3,
            flow_rate_ml_s: flow_rate * ML_PER_SIM_UNIT3,
            exit_speed: self.water_velocity_m_s * SIM_UNITS_PER_METER,
            exit_speed_m_s: self.water_velocity_m_s,
            spout_position: Vec3::new(self.spout_ui[0], self.spout_ui[1], self.spout_ui[2]),
            has_bed: false,
            last_pressure_pairs: profile.dispatches_per_frame,
            ..CommonMetrics::default()
        }
    }

    fn render_view(&self) -> RenderView<'_> {
        let bounds = Vec3::new(
            self.scene.box_max[0] - self.scene.box_min[0],
            self.scene.box_max[1] - self.scene.box_min[1],
            self.scene.box_max[2] - self.scene.box_min[2],
        );
        let positions = self
            .particles
            .position
            .as_deref()
            .expect("PB-MPM exposes positions");
        let velocities = self
            .particles
            .velocity
            .as_deref()
            .expect("PB-MPM exposes velocities");
        let phases = self
            .particles
            .phase_tag
            .as_deref()
            .expect("PB-MPM exposes phase tags");
        RenderView::new_canonical(
            bounds,
            STANDARD_WATER_RENDER_RADIUS,
            self.particles.particle_count as usize,
            positions,
            velocities,
            phases,
            RenderMaterial::default(),
        )
    }
}

impl FrameSolver for PbmpmFrameSolver {
    fn reset(&mut self, _ctx: ResetContext<'_>) -> CommonMetrics {
        RewriteSolver::reset(&mut self.core, &self.scene);
        self.sim_time_s = 0.0;
        self.total_emitted_mass = 0.0;
        self.frame_emitted_mass = 0.0;
        self.refresh_particles();
        self.common_metrics()
    }

    fn reset_scene(
        &mut self,
        ctx: ResetContext<'_>,
        scene: &SceneSpec,
    ) -> Result<CommonMetrics, String> {
        *self = Self::new(ctx.device, ctx.queue, scene)?;
        Ok(self.common_metrics())
    }

    fn step_frame(&mut self, ctx: FrameContext<'_>) -> CommonMetrics {
        let before = self.core.active_count();
        let input = self.emission_input();
        RewriteSolver::step(&mut self.core, ctx.dt, &input);
        let after = self.core.active_count();
        self.frame_emitted_mass =
            after.saturating_sub(before) as f32 * self.materials.particle_mass;
        self.total_emitted_mass += self.frame_emitted_mass;
        self.sim_time_s += ctx.dt;
        self.refresh_particles();
        self.common_metrics()
    }

    fn snapshot(&self) -> FrameSnapshot<'_> {
        FrameSnapshot {
            render: self.render_view(),
            metrics: self.common_metrics(),
        }
    }

    fn set_water_velocity_m_s(&mut self, speed_m_s: f32) {
        self.water_velocity_m_s = speed_m_s.max(0.0);
    }

    fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.spout_ui = [x, y, z];
    }
}

fn setup_for_scene_spec(scene_spec: &SceneSpec) -> Result<(Scene, Materials, Config), String> {
    match scene_spec {
        SceneSpec::CenterPour => {
            let r = 0.16_f32;
            let scene = Scene::v60_pour_water_only();
            let materials = Materials {
                particle_spacing: r,
                support_radius: 2.0 * r,
                ..Materials::default()
            };
            let config = Config {
                nozzle_radius: 0.55,
                max_speed: 25.0,
                ..Config::default()
            };
            Ok((scene, materials, config))
        }
        SceneSpec::FreeStream => {
            let r = 0.16_f32;
            let scene = Scene::v60_pour_water_only();
            let materials = Materials {
                particle_spacing: r,
                support_radius: 2.0 * r,
                ..Materials::default()
            };
            let config = Config {
                nozzle_radius: 0.55,
                max_speed: 25.0,
                ..Config::default()
            };
            Ok((scene, materials, config))
        }
        SceneSpec::Debug { id } => Err(format!("PB-MPM does not support MPM debug scene: {id}")),
    }
}

fn pbmpm_scene_accepts_pour(scene_spec: &SceneSpec) -> bool {
    matches!(scene_spec, SceneSpec::CenterPour | SceneSpec::FreeStream)
}

fn flow_rate_for_velocity(speed_m_s: f32, nozzle_radius: f32, discharge_coeff: f32) -> f32 {
    let a_eff = std::f32::consts::PI * nozzle_radius * nozzle_radius * discharge_coeff;
    a_eff * (speed_m_s.max(0.0) * SIM_UNITS_PER_METER)
}

#[cfg(test)]
mod tests {
    #[test]
    fn shader_parses_with_naga() {
        let source = format!(
            "{}\n{}\n{}",
            include_str!("pbmpm/common.wgsl"),
            include_str!("pbmpm/transfers.wgsl"),
            include_str!("pbmpm/constraint.wgsl"),
        );
        naga::front::wgsl::parse_str(&source).expect("PB-MPM shader parses");
    }
}

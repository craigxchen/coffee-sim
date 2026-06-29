use crate::emission::{EmissionInput, PourEvent};
use crate::engine::scene::Species;
use crate::engine::Scene;
use crate::models::Materials;
use crate::solvers::base::{
    CommonMetrics, FrameContext, FrameSnapshot, FrameSolver, ResetContext, SceneSpec,
    Solver as RewriteSolver,
};
use crate::solvers::xpbd::XpbdSolver;
use crate::ui::{RenderMaterial, RenderView, STANDARD_WATER_RENDER_RADIUS};
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;
use coffee_sim_core::Vec3;

const ML_PER_SIM_UNIT3: f32 = 5.20;
const SIM_UNITS_PER_METER: f32 = 27.7;
const DEFAULT_WATER_VELOCITY_M_S: f32 = 0.12;
const DEFAULT_SPOUT_UI: [f32; 3] = [0.0, 2.5, 0.0];

pub(crate) struct XpbdFrameSolver {
    core: XpbdSolver,
    scene_spec: SceneSpec,
    scene: Scene,
    materials: Materials,
    config: Config,
    particles: ParticleBuffers,
    water_velocity_m_s: f32,
    spout_ui: [f32; 3],
    sim_time_s: f32,
}

impl XpbdFrameSolver {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_spec: &SceneSpec,
    ) -> Result<Self, String> {
        let (scene, materials, config) = setup_for_scene_spec(scene_spec)?;
        let gpu = GpuContext::from_device_queue(device, queue);
        let core = <XpbdSolver as RewriteSolver>::build(&scene, &materials, &config, &gpu);
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
        })
    }

    fn refresh_particles(&mut self) {
        self.particles = RewriteSolver::particles(&self.core);
    }

    fn emission_input(&self) -> EmissionInput {
        if !xpbd_scene_accepts_pour(&self.scene_spec) {
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
        let diagnostics = self.core.diagnostics();
        let has_bed = self
            .scene
            .regions
            .iter()
            .any(|region| region.species == Species::Grain);
        let flow_rate = if xpbd_scene_accepts_pour(&self.scene_spec) {
            flow_rate_for_velocity(
                self.water_velocity_m_s,
                self.config.nozzle_radius,
                self.config.discharge_coeff,
            )
        } else {
            0.0
        };
        let emitted_mass = self.core.total_emitted_water_mass();
        let residual = diagnostics.residual.abs();
        CommonMetrics {
            particle_count: rewrite_metrics.particle_count as usize,
            water_slots_used: rewrite_metrics.particle_count,
            bed_particle_count: if has_bed {
                rewrite_metrics.particle_count
            } else {
                0
            },
            max_particles: self.core.pool_capacity(),
            sim_time_s: self.sim_time_s,
            total_emitted_mass: emitted_mass,
            total_emitted_ml: emitted_mass * ML_PER_SIM_UNIT3,
            flow_rate_ml_s: flow_rate * ML_PER_SIM_UNIT3,
            exit_speed: self.water_velocity_m_s * SIM_UNITS_PER_METER,
            exit_speed_m_s: self.water_velocity_m_s,
            spout_position: Vec3::new(self.spout_ui[0], self.spout_ui[1], self.spout_ui[2]),
            has_bed,
            last_pressure_pairs: diagnostics.effective_iters,
            max_abs_divergence: residual,
            projection_residual_max_abs_divergence: residual,
            projection_residual_mean_abs_divergence: residual,
            projection_residual_cell_count: rewrite_metrics.particle_count,
            mean_tds: rewrite_metrics.tds,
            cup_tds: rewrite_metrics.tds,
            extraction_yield: rewrite_metrics.extraction_yield,
            estimated_cup_tds: rewrite_metrics.tds,
            estimated_extraction_yield: rewrite_metrics.extraction_yield,
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
            .expect("rewrite XPBD exposes positions");
        let velocities = self
            .particles
            .velocity
            .as_deref()
            .expect("rewrite XPBD exposes velocities");
        let phases = self
            .particles
            .phase_tag
            .as_deref()
            .expect("rewrite XPBD exposes phase tags");
        let v_cap = self.materials.r_max
            * self.materials.rho_ratio
            * std::f32::consts::FRAC_PI_6
            * self.materials.grain_diameter.powi(3);
        let moisture_inv_cap = if v_cap > 0.0 { 1.0 / v_cap } else { 0.0 };
        RenderView::new_canonical(
            bounds,
            STANDARD_WATER_RENDER_RADIUS,
            self.particles.particle_count as usize,
            positions,
            velocities,
            phases,
            RenderMaterial::standard_coffee_particles(moisture_inv_cap),
        )
    }
}

impl FrameSolver for XpbdFrameSolver {
    fn reset(&mut self, _ctx: ResetContext<'_>) -> CommonMetrics {
        RewriteSolver::reset(&mut self.core, &self.scene);
        self.sim_time_s = 0.0;
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
        let input = self.emission_input();
        RewriteSolver::step(&mut self.core, ctx.dt, &input);
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
            let scene = Scene::v60_pour();
            let materials = Materials {
                particle_spacing: r,
                support_radius: 2.0 * r,
                grain_diameter: 2.0 * r,
                grain_mass: 10.0,
                ..Materials::default()
            };
            let config = Config {
                absorb_rate: 0.5,
                extract_rate: 1.0,
                nozzle_radius: 0.25,
                max_speed: 25.0,
                substeps: 2,
                drag_beta_max: 0.92,
                drag_subiters: 6,
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
                nozzle_radius: 0.25,
                max_speed: 25.0,
                xsph_viscosity_c: 0.02,
                ..Config::default()
            };
            Ok((scene, materials, config))
        }
        SceneSpec::Debug { id } => Err(format!("XPBD does not support MPM debug scene: {id}")),
    }
}

fn xpbd_scene_accepts_pour(scene_spec: &SceneSpec) -> bool {
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
        let shader_src = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("xpbd/common.wgsl"),
            include_str!("xpbd/water.wgsl"),
            include_str!("xpbd/bed.wgsl"),
            include_str!("xpbd/coupling.wgsl"),
            include_str!("xpbd/wetting.wgsl"),
            include_str!("xpbd/extraction.wgsl"),
            include_str!("xpbd/fines.wgsl"),
        );
        naga::front::wgsl::parse_str(&shader_src).expect("XPBD WGSL should parse");
    }
}

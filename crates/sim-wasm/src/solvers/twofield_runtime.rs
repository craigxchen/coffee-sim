use crate::emission::{EmissionInput, PourEvent};
use crate::engine::scene::Species;
use crate::engine::Scene;
use crate::models::calibration::{CalibrationControl, CalibrationProfile};
use crate::models::scaling::{
    capillary_suction_acceleration_m_s2, darcy_beta_per_s, water_sample_from_spacing,
    ReducedUnitScales,
};
use crate::models::water::{density_kg_m3, kinematic_viscosity_m2_s};
use crate::models::Materials;
use crate::solvers::base::{
    CalibrationApplyResult, CalibrationSupport, CommonMetrics, FrameContext, FrameSnapshot,
    FrameSolver, ResetContext, SceneSpec, Solver as RewriteSolver,
};
use crate::solvers::twofield::TwofieldSolver;
use crate::ui::{RenderMaterial, RenderView, STANDARD_WATER_RENDER_RADIUS};
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;
use crate::utils::units::SiUnitScale;
use coffee_sim_core::Vec3;

const DEFAULT_WATER_VELOCITY_M_S: f32 = 0.50;
const DEFAULT_MAX_WATER_SPEED_M_S: f32 = 2.5;
const DEFAULT_NOZZLE_RADIUS_M: f64 = 0.002;
const TARGET_RECIPE_WATER_SAMPLES: u32 = 78_000;
const SOLID_SAMPLE_SPACING: f32 = 0.100;
const PRODUCTION_SOLID_SAMPLE_BUDGET: u32 = 12_000;
const EXTRACTION_EA_OVER_R_K: f32 = 3_000.0;
const GRAIN_TO_WATER_HEAT_CAPACITY_RATIO: f32 = 0.36;
const EXTRACTION_FLOW_HALF_SPEED_M_S: f64 = 0.01;
const DEFAULT_SPOUT_UI: [f32; 3] = [0.0, 2.5, 0.0];

pub(crate) const MPM_V2_LATE_BREW_BENCHMARK_SCENE_ID: &str = "mpm-v2-late-brew-benchmark";
pub(crate) const MPM_V2_CALIBRATION_DRAINAGE_SCENE_ID: &str = "mpm-v2-uniform-bed-saturation";

fn unit_scale() -> SiUnitScale {
    SiUnitScale::reference()
}

fn sim_units_per_meter() -> f32 {
    unit_scale().sim_units_per_meter() as f32
}

fn emitted_water_volume_ml(emitted_mass: f32, materials: &Materials) -> f32 {
    if materials.particle_mass <= 0.0 {
        return 0.0;
    }
    let emitted_particles = emitted_mass / materials.particle_mass;
    let volume_sim_units = emitted_particles * materials.particle_spacing.powi(3);
    (unit_scale().ml_from_sim_volume(volume_sim_units as f64)) as f32
}

fn water_mass_g_from_volume_ml(volume_ml: f32, density_kg_m3: f32) -> f32 {
    if !volume_ml.is_finite() || !density_kg_m3.is_finite() || density_kg_m3 <= 0.0 {
        return 0.0;
    }
    // kg/m³ == g/L; one millilitre is 1/1000 litre.
    volume_ml.max(0.0) * density_kg_m3 / 1_000.0
}

fn water_spacing_for_recipe(volume_ml: f32, samples: u32) -> Result<f32, String> {
    if !volume_ml.is_finite() || volume_ml <= 0.0 {
        return Err("recipe water volume must be positive and finite".to_string());
    }
    if samples == 0 {
        return Err("target water sample count must be nonzero".to_string());
    }
    let sample_volume = unit_scale().sim_volume_from_ml(volume_ml as f64) / f64::from(samples);
    let spacing = sample_volume.cbrt();
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err("recipe water spacing conversion produced an invalid value".to_string());
    }
    Ok(spacing as f32)
}

fn calibrated_drag_scale(
    materials: &Materials,
    profile: &CalibrationProfile,
) -> Result<f32, String> {
    let viscosity =
        crate::models::water::kinematic_viscosity_m2_s(profile.inlet_temperature_c() as f64)
            .map_err(|error| error.to_string())?;
    let beta_reference = darcy_beta_per_s(viscosity, profile.permeability_m2() as f64, 1.0)
        .map_err(|error| error.to_string())?
        * f64::from(profile.drag_multiplier());
    // The WGSL evaluates beta = phi_f*drag_scale/K_KC(d,phi). Choosing the scale below makes the
    // intrinsic-velocity equation beta=phi_f*nu/K_ref; therefore superficial flux is exactly
    // q=phi_f*(vf-vs)=K_ref*F/nu. The numerical sample pitch cancels from the porosity-relative
    // response and is not misreported as a physical grind size.
    let numerical_reference_k =
        crate::models::permeability::kozeny_carman(materials.grain_diameter, materials.porosity);
    let scale = beta_reference * f64::from(numerical_reference_k);
    if !scale.is_finite() || scale <= 0.0 || scale > f64::from(f32::MAX) {
        return Err("calibrated Darcy drag scale is outside the solver range".to_string());
    }
    Ok(scale as f32)
}

fn calibrated_suction_accel(
    materials: &Materials,
    profile: &CalibrationProfile,
) -> Result<f32, String> {
    let front_thickness_m = unit_scale().meters_from_sim_length(f64::from(
        crate::solvers::twofield::CELL_SIZE_FACTOR * materials.particle_spacing,
    ));
    let acceleration_m_s2 = capillary_suction_acceleration_m_s2(
        profile.suction_head_m() as f64,
        front_thickness_m,
        crate::models::scaling::STANDARD_GRAVITY_M_S2,
    )
    .map_err(|error| error.to_string())?;
    let reduced = unit_scale().sim_acceleration_from_meters_per_second_squared(acceleration_m_s2);
    if !reduced.is_finite() || reduced < 0.0 || reduced > f64::from(f32::MAX) {
        return Err("capillary suction conversion is outside the solver range".to_string());
    }
    Ok(reduced as f32)
}

fn reduced_unit_scales(
    materials: &Materials,
    water_density_kg_m3: f32,
) -> Result<ReducedUnitScales, String> {
    let water_sample = water_sample_from_spacing(
        materials.particle_spacing as f64,
        f64::from(water_density_kg_m3),
        unit_scale(),
    )
    .map_err(|error| error.to_string())?;
    ReducedUnitScales::from_spatial_scale(unit_scale(), water_sample.mass_kg, 1.0)
        .map_err(|error| error.to_string())
}

fn reduced_stress_from_pa(
    materials: &Materials,
    water_density_kg_m3: f32,
    pressure_pa: f32,
) -> Result<f32, String> {
    let reduced = reduced_unit_scales(materials, water_density_kg_m3)?
        .reduced_stress_from_pa(pressure_pa as f64)
        .map_err(|error| error.to_string())?;
    if !reduced.is_finite() || reduced < 0.0 || reduced > f64::from(f32::MAX) {
        return Err("calibrated stress is outside the solver range".to_string());
    }
    Ok(reduced as f32)
}

fn bound_water_capacity_per_solid(
    scene: &Scene,
    profile: &CalibrationProfile,
    water_density_kg_m3: f32,
    solid_count: u32,
) -> f32 {
    if solid_count == 0 || scene.dose_g <= 0.0 {
        return 0.0;
    }
    let capacity_g = profile.bound_water_capacity_g_per_g() * scene.dose_g;
    let capacity_m3 = crate::models::filter::water_mass_capacity_to_volume_m3(
        f64::from(capacity_g),
        f64::from(water_density_kg_m3),
    )
    .unwrap_or(0.0);
    let capacity_sim = capacity_m3 / unit_scale().meters_per_sim_unit().powi(3);
    (capacity_sim / f64::from(solid_count)) as f32
}

pub(crate) struct MpmV2FrameSolver {
    core: TwofieldSolver,
    scene_spec: SceneSpec,
    scene: Scene,
    materials: Materials,
    config: Config,
    particles: ParticleBuffers,
    water_velocity_m_s: f32,
    spout_ui: [f32; 3],
    sim_time_s: f32,
    calibration: CalibrationProfile,
}

impl MpmV2FrameSolver {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_spec: &SceneSpec,
    ) -> Result<Self, String> {
        Self::new_with_calibration(device, queue, scene_spec, CalibrationProfile::default())
    }

    pub(crate) fn new_with_calibration(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene_spec: &SceneSpec,
        calibration: CalibrationProfile,
    ) -> Result<Self, String> {
        calibration.validate().map_err(|error| error.to_string())?;
        let (scene, materials, config) = setup_for_scene_spec(scene_spec, &calibration)?;
        let gpu = GpuContext::from_device_queue(device, queue);
        let core = <TwofieldSolver as RewriteSolver>::build(&scene, &materials, &config, &gpu);
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
            calibration,
        })
    }

    fn rebuild_with_calibration(
        &mut self,
        ctx: ResetContext<'_>,
        calibration: CalibrationProfile,
    ) -> Result<(), String> {
        let water_velocity_m_s = self.water_velocity_m_s;
        let spout_ui = self.spout_ui;
        let mut rebuilt =
            Self::new_with_calibration(ctx.device, ctx.queue, &self.scene_spec, calibration)?;
        rebuilt.water_velocity_m_s = water_velocity_m_s;
        rebuilt.spout_ui = spout_ui;
        *self = rebuilt;
        Ok(())
    }

    fn refresh_particles(&mut self) {
        self.particles = RewriteSolver::particles(&self.core);
    }

    fn emission_input(&self) -> EmissionInput {
        if !twofield_scene_accepts_pour(&self.scene_spec) {
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
        let (water_count, solid_count) = self.core.phase_counts();
        let has_bed = self
            .scene
            .regions
            .iter()
            .any(|region| region.species == Species::Grain);
        let flow_rate = if twofield_scene_accepts_pour(&self.scene_spec) {
            flow_rate_for_velocity(
                self.water_velocity_m_s,
                self.config.nozzle_radius,
                self.config.discharge_coeff,
            )
        } else {
            0.0
        };
        let emitted_mass = self.core.total_emitted_water_mass();
        let emitted_ml = emitted_water_volume_ml(emitted_mass, &self.materials);
        let emitted_g = water_mass_g_from_volume_ml(emitted_ml, self.config.tf_water_density_kg_m3);
        let grams_per_reduced_mass =
            reduced_unit_scales(&self.materials, self.config.tf_water_density_kg_m3)
                .map(|scales| (1_000.0 * scales.mass_kg_per_unit()) as f32)
                .unwrap_or(0.0);
        let mass_g = |mass: f32| mass * grams_per_reduced_mass;
        let carrier_mass_g_from_volume = |volume: f32| {
            mass_g(
                volume * self.materials.particle_mass
                    / self.materials.particle_spacing.powi(3).max(1.0e-9),
            )
        };
        let carrier_ml = |mass: f32| {
            if self.materials.particle_mass <= 0.0 {
                0.0
            } else {
                let volume =
                    mass / self.materials.particle_mass * self.materials.particle_spacing.powi(3);
                unit_scale().ml_from_sim_volume(f64::from(volume)) as f32
            }
        };
        let volume_ml = |volume: f32| unit_scale().ml_from_sim_volume(f64::from(volume)) as f32;
        let profile = RewriteSolver::profile(&self.core);
        CommonMetrics {
            particle_count: rewrite_metrics.particle_count as usize,
            water_slots_used: water_count,
            bed_particle_count: solid_count,
            max_particles: self.core.water_pool_capacity() + solid_count,
            sim_time_s: self.sim_time_s,
            // Emission accounting keeps volume and mass distinct. The core reports reduced
            // particle mass; the shared inlet-temperature density closes the physical conversion.
            total_emitted_mass: emitted_g,
            total_emitted_ml: emitted_ml,
            flow_rate_ml_s: (unit_scale().ml_from_sim_volume(flow_rate as f64)) as f32,
            exit_speed: self.water_velocity_m_s * sim_units_per_meter(),
            exit_speed_m_s: self.water_velocity_m_s,
            spout_position: Vec3::new(self.spout_ui[0], self.spout_ui[1], self.spout_ui[2]),
            has_bed,
            last_pressure_pairs: rewrite_metrics.iteration_count,
            fluid_cell_count: diagnostics.free_water_above_bed,
            mean_tds: rewrite_metrics.tds,
            cup_tds: rewrite_metrics.tds,
            extraction_yield: rewrite_metrics.extraction_yield,
            estimated_cup_tds: rewrite_metrics.tds,
            estimated_extraction_yield: rewrite_metrics.extraction_yield,
            diagnostics_supported: true,
            diagnostics_valid: diagnostics.sample_valid,
            diagnostics_all_finite: diagnostics.sample_valid && !diagnostics.has_nonfinite_state,
            diagnostics_pending: self.core.diagnostics_pending(),
            diagnostics_sample_age_s: if diagnostics.sample_valid {
                (self.sim_time_s - diagnostics.sample_time_s).max(0.0)
            } else {
                0.0
            },
            dispatches_per_frame: profile.dispatches_per_frame,
            nonfinite_particle_count: diagnostics.nonfinite_particle_count,
            nonfinite_grid_node_count: diagnostics.nonfinite_grid_node_count,
            mean_water_temperature_k: diagnostics.mean_water_temperature,
            mean_grain_temperature_k: diagnostics.mean_grain_temperature,
            pond_carrier_mass_g: mass_g(diagnostics.pond_carrier_mass),
            pond_carrier_volume_ml: carrier_ml(diagnostics.pond_carrier_mass),
            mean_bound_saturation: diagnostics.mean_bound_saturation,
            mean_diameter_ratio: diagnostics.mean_diameter_ratio,
            mobile_carrier_mass_g: mass_g(diagnostics.mobile_carrier_mass),
            paper_carrier_mass_g: mass_g(diagnostics.paper_carrier_mass),
            bound_carrier_mass_g: mass_g(diagnostics.bound_carrier_mass),
            total_carrier_mass_g: mass_g(diagnostics.total_carrier_mass),
            mobile_carrier_volume_ml: carrier_ml(diagnostics.mobile_carrier_mass),
            paper_carrier_volume_ml: carrier_ml(diagnostics.paper_carrier_mass),
            bound_carrier_volume_ml: carrier_ml(diagnostics.bound_carrier_mass),
            total_carrier_volume_ml: carrier_ml(diagnostics.total_carrier_mass),
            mobile_solution_mass_g: mass_g(diagnostics.mobile_solution_mass),
            paper_solution_mass_g: mass_g(diagnostics.paper_solution_mass),
            grain_mass_g: mass_g(diagnostics.grain_mass),
            mobile_solute_mass_g: mass_g(diagnostics.mobile_solute_mass),
            paper_solute_mass_g: mass_g(diagnostics.paper_solute_mass),
            remaining_grain_solute_mass_g: mass_g(diagnostics.remaining_grain_solute_mass),
            dissolved_solute_mass_g: mass_g(diagnostics.dissolved_solute_mass),
            cup_carrier_mass_g: mass_g(diagnostics.cup_carrier_mass),
            cup_solute_mass_g: mass_g(diagnostics.cup_solute_mass),
            cup_solution_mass_g: mass_g(diagnostics.cup_solution_mass),
            total_solute_mass_g: mass_g(diagnostics.total_solute),
            expected_carrier_mass_g: mass_g(diagnostics.expected_carrier_mass),
            expected_solute_mass_g: mass_g(diagnostics.expected_solute_mass),
            carrier_mass_residual_g: mass_g(diagnostics.carrier_mass_residual),
            solute_mass_residual_g: mass_g(diagnostics.solute_mass_residual),
            mechanical_mass_residual_g: mass_g(diagnostics.mechanical_mass_residual),
            retention_target_carrier_mass_g: carrier_mass_g_from_volume(
                diagnostics.retention_target_volume,
            ),
            retention_bound_capacity_mass_g: carrier_mass_g_from_volume(
                diagnostics.retention_bound_capacity_volume,
            ),
            retention_capillary_target_mass_g: carrier_mass_g_from_volume(
                diagnostics.retention_capillary_target_volume,
            ),
            retention_available_pore_volume_ml: volume_ml(
                diagnostics.retention_available_pore_volume,
            ),
            retention_capillary_carrier_mass_g: carrier_mass_g_from_volume(
                diagnostics.retention_capillary_volume,
            ),
            retention_unmet_carrier_mass_g: carrier_mass_g_from_volume(
                diagnostics.retention_unmet_volume,
            ),
            residual_pore_saturation: diagnostics.residual_pore_saturation,
            retention_target_attainable: diagnostics.retention_target_attainable,
            filter_occupied_ml: volume_ml(diagnostics.filter_occupied_volume),
            filter_capacity_ml: volume_ml(diagnostics.filter_capacity_volume),
            filter_saturation: diagnostics.filter_saturation,
            filter_volume_residual_ml: volume_ml(diagnostics.filter_volume_residual),
            filter_count_residual: diagnostics.filter_count_residual,
            filter_held_parcels: diagnostics.filter_held_parcels,
            filter_admitted: diagnostics.filter_admitted,
            filter_released: diagnostics.filter_released,
            filter_overflow: diagnostics.filter_overflow,
            filter_errors: diagnostics.filter_errors,
            filter_beta_max_per_s: diagnostics.filter_beta_max_per_s,
            filter_pressure_mobility_min: diagnostics.filter_pressure_mobility_min,
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
            .expect("rewrite two-field exposes positions");
        let velocities = self
            .particles
            .velocity
            .as_deref()
            .expect("rewrite two-field exposes velocities");
        let phases = self
            .particles
            .phase_tag
            .as_deref()
            .expect("rewrite two-field exposes phase tags");
        let (water_count, solid_count) = self.core.phase_counts();
        let solid_start = self.core.water_pool_capacity();
        let solid_end = solid_start
            .checked_add(solid_count)
            .expect("two-field particle range exceeds u32");
        let v_cap = bound_water_capacity_per_solid(
            &self.scene,
            &self.calibration,
            self.config.tf_water_density_kg_m3,
            solid_count,
        );
        let moisture_inv_cap = if v_cap > 0.0 { 1.0 / v_cap } else { 0.0 };
        RenderView::new_canonical_with_ranges(
            bounds,
            STANDARD_WATER_RENDER_RADIUS,
            [0..water_count, solid_start..solid_end],
            positions,
            velocities,
            phases,
            RenderMaterial::standard_coffee_particles(moisture_inv_cap)
                .with_max_diameter_swelling(self.calibration.max_diameter_swelling_fraction()),
        )
    }
}

impl FrameSolver for MpmV2FrameSolver {
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
        let calibration = self.calibration.clone();
        *self = Self::new_with_calibration(ctx.device, ctx.queue, scene, calibration)?;
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

    fn calibration_profile(&self) -> Option<&CalibrationProfile> {
        Some(&self.calibration)
    }

    fn calibration_support(&self, control: CalibrationControl) -> CalibrationSupport {
        calibration_support(control)
    }

    fn set_calibration_value(
        &mut self,
        ctx: ResetContext<'_>,
        id: &str,
        value: f32,
    ) -> Result<CalibrationApplyResult, String> {
        let mut calibration = self.calibration.clone();
        let update = calibration
            .set_by_id(id, value)
            .map_err(|error| error.to_string())?;
        let support = calibration_support(update.control);
        let simulation_reset = update.requires_rebuild || support == CalibrationSupport::Rebuild;
        if simulation_reset {
            self.rebuild_with_calibration(ctx, calibration)?;
        } else {
            self.calibration = calibration;
        }
        Ok(CalibrationApplyResult {
            update,
            support,
            simulation_reset,
        })
    }

    fn reset_calibration_defaults(&mut self, ctx: ResetContext<'_>) -> Result<bool, String> {
        self.rebuild_with_calibration(ctx, CalibrationProfile::default())?;
        Ok(true)
    }
}

fn calibration_support(control: CalibrationControl) -> CalibrationSupport {
    match control {
        CalibrationControl::BoundWaterCapacity
        | CalibrationControl::MaxDiameterSwelling
        | CalibrationControl::Log10Permeability
        | CalibrationControl::DragMultiplier
        | CalibrationControl::AbsorptionRate
        | CalibrationControl::SuctionHead
        | CalibrationControl::BloomDelay
        | CalibrationControl::DpFrictionAngle
        | CalibrationControl::DpCohesion
        | CalibrationControl::BedCompactionTangent
        | CalibrationControl::InitialConsolidationPressure
        | CalibrationControl::NumericalShearModulus
        | CalibrationControl::FullCompactionSolidFraction
        | CalibrationControl::ExtractionFastRate
        | CalibrationControl::ExtractionSlowRate
        | CalibrationControl::InletTemperature
        | CalibrationControl::AmbientTemperature
        | CalibrationControl::ThermalExchangeRate
        | CalibrationControl::ReferenceBedPorosity
        | CalibrationControl::BedRetention
        | CalibrationControl::CapillaryRelativePermeabilityExponent
        | CalibrationControl::CapillaryMobilityFloor => CalibrationSupport::Rebuild,
        CalibrationControl::Log10FilterPaperPermeability
        | CalibrationControl::FilterPaperThickness
        | CalibrationControl::FilterPaperPorosity
        | CalibrationControl::FilterWaterCapacity => CalibrationSupport::Rebuild,
    }
}

fn apply_calibration_profile(
    materials: &mut Materials,
    config: &mut Config,
    profile: &CalibrationProfile,
) -> Result<(), String> {
    materials.r_max = profile.bound_water_capacity_g_per_g();
    materials.porosity = profile.reference_bed_porosity_fraction();
    materials.friction_mu = profile.dp_friction_mu();
    materials.k0_fast = profile.extraction_fast_rate_per_s();
    materials.k0_slow = profile.extraction_slow_rate_per_s();
    // Chemistry stores kelvin. Specific heats are normalized by water's heat capacity because
    // only their ratio enters the conservative enthalpy exchange; this avoids large fixed-point
    // enthalpy lanes while preserving the physical capacity weighting.
    materials.ea_over_r = EXTRACTION_EA_OVER_R_K;
    materials.t_ref = profile.inlet_temperature_k();
    materials.pour_t = profile.inlet_temperature_k();
    materials.t_amb = profile.ambient_temperature_k();
    materials.kappa = profile.thermal_exchange_rate_per_s();
    materials.cp_water = 1.0;
    materials.cp_grain = GRAIN_TO_WATER_HEAT_CAPACITY_RATIO;
    materials.u_half =
        unit_scale().sim_speed_from_meters_per_second(EXTRACTION_FLOW_HALF_SPEED_M_S) as f32;
    config.drag_scale = calibrated_drag_scale(materials, profile)?;
    config.tf_absorb_rate = profile.absorption_rate_per_s();
    config.tf_suction_accel = calibrated_suction_accel(materials, profile)?;
    config.tf_bloom_delay = profile.bloom_delay_s();
    config.tf_max_diameter_swelling = profile.max_diameter_swelling_fraction();
    config.tf_retention_target_g_per_g = profile.bed_retention_g_per_g();
    config.tf_capillary_exponent = profile.capillary_relative_permeability_exponent();
    config.tf_capillary_mobility_floor = profile.capillary_mobility_floor_fraction();
    config.tf_filter_permeability_m2 = profile.filter_paper_permeability_m2();
    config.tf_filter_kinematic_viscosity_m2_s =
        kinematic_viscosity_m2_s(f64::from(profile.inlet_temperature_c()))
            .map_err(|error| format!("filter-paper viscosity: {error}"))? as f32;
    config.tf_water_density_kg_m3 = density_kg_m3(f64::from(profile.inlet_temperature_c()))
        .map_err(|error| format!("inlet-water density: {error}"))?
        as f32;
    config.tf_filter_capacity_g = profile.filter_water_capacity_g();
    config.tf_filter_thickness_m = profile.filter_paper_thickness_m();
    config.tf_filter_porosity = profile.filter_paper_porosity_fraction();
    config.tf_wet_cohesion = reduced_stress_from_pa(
        materials,
        config.tf_water_density_kg_m3,
        profile.dp_cohesion_pa(),
    )?;
    config.tf_compaction_tangent = reduced_stress_from_pa(
        materials,
        config.tf_water_density_kg_m3,
        profile.bed_compaction_tangent_pa(),
    )?;
    config.tf_initial_consolidation_pressure = reduced_stress_from_pa(
        materials,
        config.tf_water_density_kg_m3,
        profile.initial_consolidation_pressure_pa(),
    )?;
    config.tf_numerical_shear_modulus = reduced_stress_from_pa(
        materials,
        config.tf_water_density_kg_m3,
        profile.numerical_shear_modulus_pa(),
    )?;
    config.tf_full_compaction_solid_fraction = profile.full_compaction_solid_fraction();
    Ok(())
}

fn setup_production_bed_scene(
    mut scene: Scene,
    recipe_water_ml: f32,
    calibration: &CalibrationProfile,
) -> Result<(Scene, Materials, Config), String> {
    let water_spacing = water_spacing_for_recipe(recipe_water_ml, TARGET_RECIPE_WATER_SAMPLES)?;
    scene.gravity = [
        0.0,
        unit_scale().sim_acceleration_from_meters_per_second_squared(-9.806_65) as f32,
        0.0,
    ];
    let mut materials = Materials {
        particle_spacing: water_spacing,
        support_radius: 2.0 * water_spacing,
        // Numerical solid material points are coarser than water points. This is a sample pitch,
        // not the physical grind diameter used to infer permeability.
        grain_diameter: SOLID_SAMPLE_SPACING,
        grain_mass: 1.0,
        ..Materials::default()
    };
    let mut config = Config {
        absorb_rate: 0.0,
        extract_rate: 1.0,
        thermal_rate: 1.0,
        nozzle_radius: unit_scale().sim_length_from_meters(DEFAULT_NOZZLE_RADIUS_M) as f32,
        max_speed: DEFAULT_MAX_WATER_SPEED_M_S * sim_units_per_meter(),
        tf_solid_sample_budget: PRODUCTION_SOLID_SAMPLE_BUDGET,
        tf_all_gas_atmospheric: true,
        substeps: 2,
        drag_beta_max: 0.92,
        drag_subiters: 6,
        solid_dynamics: true,
        tf_absorb_rate: 0.15,
        tf_wet_cohesion: 0.0,
        tf_filter_floor: false,
        tf_flip_c_surface: 0.1,
        tf_flip_density_gate: 0.5,
        tf_flip_div_scale: 1.0,
        tf_flip_water_splash_cap: DEFAULT_MAX_WATER_SPEED_M_S * sim_units_per_meter(),
        ..Config::default()
    };
    apply_calibration_profile(&mut materials, &mut config, calibration)?;
    Ok((scene, materials, config))
}

fn seeded_bed_pore_volume_sim(
    scene: &Scene,
    materials: &Materials,
    config: &Config,
) -> Result<f64, String> {
    let mut solid_scene = scene.clone();
    solid_scene.water_ml = 0.0;
    solid_scene.pour_water_ml = 0.0;
    solid_scene
        .regions
        .retain(|region| region.species == Species::Grain);
    let (positions, _, water_count, quadrature_weight) =
        crate::solvers::twofield::seed_ranges(&solid_scene, materials, config);
    let solid_count = positions.len().saturating_sub(water_count as usize);
    if solid_count == 0 {
        return Err("uniform drainage fixture has no accepted solid samples".to_string());
    }
    let bulk_volume = solid_count as f64
        * f64::from(materials.grain_diameter).powi(3)
        * f64::from(quadrature_weight);
    let pore_volume = bulk_volume * f64::from(materials.porosity);
    if !pore_volume.is_finite() || pore_volume <= 0.0 {
        return Err("uniform drainage fixture produced an invalid pore volume".to_string());
    }
    Ok(pore_volume)
}

fn setup_uniform_bed_drainage_scene(
    scene: Scene,
    calibration: &CalibrationProfile,
) -> Result<(Scene, Materials, Config), String> {
    let provisional_volume_ml = scene.water_ml;
    let (mut scene, mut materials, mut config) =
        setup_production_bed_scene(scene, provisional_volume_ml, calibration)?;
    let pore_volume_sim = seeded_bed_pore_volume_sim(&scene, &materials, &config)?;
    scene.water_ml = unit_scale().ml_from_sim_volume(pore_volume_sim) as f32;
    materials.particle_spacing =
        water_spacing_for_recipe(scene.water_ml, TARGET_RECIPE_WATER_SAMPLES)?;
    materials.support_radius = 2.0 * materials.particle_spacing;
    // Suction-front thickness and all reduced stress/mass scales depend on water spacing, so the
    // same calibration profile must be re-applied after the pore-volume-derived parcel size.
    apply_calibration_profile(&mut materials, &mut config, calibration)?;
    Ok((scene, materials, config))
}

fn setup_for_scene_spec(
    scene_spec: &SceneSpec,
    calibration: &CalibrationProfile,
) -> Result<(Scene, Materials, Config), String> {
    match scene_spec {
        SceneSpec::CenterPour => {
            let scene = Scene::v60_pour();
            let recipe_water_ml = scene.pour_water_ml;
            setup_production_bed_scene(scene, recipe_water_ml, calibration)
        }
        SceneSpec::FreeStream => {
            let mut scene = Scene::v60_pour_water_only();
            let water_spacing =
                water_spacing_for_recipe(scene.pour_water_ml, TARGET_RECIPE_WATER_SAMPLES)?;
            scene.gravity = [
                0.0,
                unit_scale().sim_acceleration_from_meters_per_second_squared(-9.806_65) as f32,
                0.0,
            ];
            let mut materials = Materials {
                particle_spacing: water_spacing,
                support_radius: 2.0 * water_spacing,
                ..Materials::default()
            };
            let mut config = Config {
                thermal_rate: 1.0,
                nozzle_radius: unit_scale().sim_length_from_meters(DEFAULT_NOZZLE_RADIUS_M) as f32,
                max_speed: DEFAULT_MAX_WATER_SPEED_M_S * sim_units_per_meter(),
                xsph_viscosity_c: 0.02,
                tf_all_gas_atmospheric: true,
                tf_flip_c_surface: 0.1,
                tf_flip_density_gate: 0.5,
                tf_flip_div_scale: 1.0,
                tf_flip_water_splash_cap: DEFAULT_MAX_WATER_SPEED_M_S * sim_units_per_meter(),
                ..Config::default()
            };
            apply_calibration_profile(&mut materials, &mut config, calibration)?;
            Ok((scene, materials, config))
        }
        SceneSpec::Debug { id } if id == MPM_V2_LATE_BREW_BENCHMARK_SCENE_ID => {
            let scene = Scene::v60_late_brew_benchmark();
            let recipe_water_ml = scene.water_ml;
            setup_production_bed_scene(scene, recipe_water_ml, calibration)
        }
        SceneSpec::Debug { id } if id == MPM_V2_CALIBRATION_DRAINAGE_SCENE_ID => {
            let scene = Scene::debug_uniform_bed_saturation();
            setup_uniform_bed_drainage_scene(scene, calibration)
        }
        SceneSpec::Debug { id } => Err(format!("MPM v2 does not support MPM debug scene: {id}")),
    }
}

fn twofield_scene_accepts_pour(scene_spec: &SceneSpec) -> bool {
    matches!(scene_spec, SceneSpec::CenterPour | SceneSpec::FreeStream)
}

fn flow_rate_for_velocity(speed_m_s: f32, nozzle_radius: f32, discharge_coeff: f32) -> f32 {
    let a_eff = std::f32::consts::PI * nozzle_radius * nozzle_radius * discharge_coeff;
    a_eff * (speed_m_s.max(0.0) * sim_units_per_meter())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_parses_with_naga() {
        let shader_src = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("twofield/common.wgsl"),
            include_str!("twofield/transfers.wgsl"),
            include_str!("twofield/pressure.wgsl"),
            include_str!("twofield/active_tiles.wgsl"),
            include_str!("twofield/surface.wgsl"),
            include_str!("twofield/coupling.wgsl"),
            include_str!("twofield/plasticity.wgsl"),
            include_str!("twofield/compaction.wgsl"),
        );
        let module =
            naga::front::wgsl::parse_str(&shader_src).expect("two-field WGSL should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("two-field WGSL should validate for baseline WebGPU");
    }

    #[test]
    fn runtime_units_are_derived_from_the_shared_volume_scale() {
        let scale = unit_scale();
        let gravity = scale.sim_acceleration_from_meters_per_second_squared(9.806_65);
        assert!(gravity > 0.0 && gravity.is_finite());
        assert!(
            (scale.ml_from_sim_volume(1.0) - scale.ml_per_sim_unit_cubed()).abs() < f64::EPSILON
        );
        let speed = 0.42;
        let reduced = speed * sim_units_per_meter();
        assert!(
            (scale.meters_per_second_from_sim_speed(reduced as f64) - speed as f64).abs() < 1.0e-6
        );
    }

    #[test]
    fn emitted_volume_and_mass_round_trip_through_temperature_density() {
        let materials = Materials {
            particle_spacing: 0.2,
            particle_mass: 2.0,
            ..Materials::default()
        };
        let two_particles_reduced_mass = 2.0 * materials.particle_mass;
        let expected = unit_scale().ml_from_sim_volume(2.0 * 0.2_f64.powi(3)) as f32;
        let actual = emitted_water_volume_ml(two_particles_reduced_mass, &materials);
        assert!((actual - expected).abs() < 1.0e-6);
        for temperature_c in [20.0_f32, 60.0, 93.0] {
            let density = density_kg_m3(f64::from(temperature_c)).unwrap() as f32;
            let mass_g = water_mass_g_from_volume_ml(actual, density);
            let reconstructed_ml = mass_g * 1_000.0 / density;
            assert!((reconstructed_ml - actual).abs() <= 2.0e-6 * actual.max(1.0));
        }
    }

    #[test]
    fn recipe_spacing_reconstructs_the_declared_water_volume() {
        let volume_ml = 250.0;
        let spacing = water_spacing_for_recipe(volume_ml, TARGET_RECIPE_WATER_SAMPLES).unwrap();
        let reconstructed = unit_scale().ml_from_sim_volume(
            f64::from(spacing).powi(3) * f64::from(TARGET_RECIPE_WATER_SAMPLES),
        );
        assert!((reconstructed - f64::from(volume_ml)).abs() < 1.0e-3);
    }

    #[test]
    fn late_brew_benchmark_routes_production_setup_without_emission_at_declared_scale() {
        let spec = SceneSpec::Debug {
            id: MPM_V2_LATE_BREW_BENCHMARK_SCENE_ID.to_string(),
        };
        let (scene, materials, config) =
            setup_for_scene_spec(&spec, &CalibrationProfile::default()).unwrap();
        let expected_spacing =
            water_spacing_for_recipe(250.0, TARGET_RECIPE_WATER_SAMPLES).unwrap();
        assert_eq!(materials.particle_spacing, expected_spacing);
        assert_eq!(materials.grain_diameter, SOLID_SAMPLE_SPACING);
        assert_eq!(
            config.tf_solid_sample_budget,
            PRODUCTION_SOLID_SAMPLE_BUDGET
        );
        assert!(config.solid_dynamics);
        assert!(!twofield_scene_accepts_pour(&spec));
        assert_eq!(scene.pour_water_ml, 0.0);

        let (positions, _, water, _) =
            crate::solvers::twofield::seed_ranges(&scene, &materials, &config);
        let solids = positions.len() as u32 - water;
        let parcel_volume = materials.particle_spacing.powi(3);
        let represented_volume = f64::from(water) * f64::from(parcel_volume);
        let declared_volume = unit_scale().sim_volume_from_ml(f64::from(scene.water_ml));
        assert!(
            (represented_volume - declared_volume).abs() <= 0.5 * f64::from(parcel_volume),
            "seeded inventory must close to within half a parcel: represented={represented_volume}, declared={declared_volume}"
        );
        assert!(
            solids == PRODUCTION_SOLID_SAMPLE_BUDGET,
            "late-brew bed workload must exactly match its declared solid budget; got {solids}"
        );
    }

    #[test]
    fn calibration_drainage_scene_routes_a_seeded_bed_without_enabling_the_emitter() {
        let spec = SceneSpec::Debug {
            id: MPM_V2_CALIBRATION_DRAINAGE_SCENE_ID.to_string(),
        };
        let (scene, materials, config) =
            setup_for_scene_spec(&spec, &CalibrationProfile::default()).unwrap();
        assert!(!twofield_scene_accepts_pour(&spec));
        assert_eq!(scene.pour_water_ml, 0.0);
        assert!(scene
            .regions
            .iter()
            .any(|region| region.species == Species::Water));
        assert!(scene
            .regions
            .iter()
            .any(|region| region.species == Species::Grain));
        let declared_volume = unit_scale().sim_volume_from_ml(f64::from(scene.water_ml));
        let pore_volume = seeded_bed_pore_volume_sim(&scene, &materials, &config).unwrap();
        assert!(
            (declared_volume - pore_volume).abs()
                <= 2.0e-6 * declared_volume.abs().max(pore_volume.abs()).max(1.0),
            "declared drainage water must equal accepted bed pore volume"
        );
        let (positions, _, water_count, _) =
            crate::solvers::twofield::seed_ranges(&scene, &materials, &config);
        assert!(
            positions.len() > water_count as usize,
            "fixture must retain its bed"
        );
        let parcel_volume = f64::from(materials.particle_spacing).powi(3);
        let represented_water = f64::from(water_count) * parcel_volume;
        assert!(
            (represented_water - pore_volume).abs() <= 0.5 * parcel_volume,
            "seeded water must close pore volume within half a parcel"
        );
    }

    #[test]
    fn physical_permeability_maps_monotonically_to_numerical_drag() {
        let (_, materials, _) =
            setup_for_scene_spec(&SceneSpec::CenterPour, &CalibrationProfile::default()).unwrap();
        let mut tight = CalibrationProfile::default();
        tight.set_by_id("log10_permeability_m2", -13.0).unwrap();
        let mut open = CalibrationProfile::default();
        open.set_by_id("log10_permeability_m2", -10.0).unwrap();
        assert!(
            calibrated_drag_scale(&materials, &tight).unwrap()
                > calibrated_drag_scale(&materials, &open).unwrap()
        );
    }

    #[test]
    fn calibration_maps_only_dimensionally_unambiguous_build_fields() {
        let mut profile = CalibrationProfile::default();
        profile
            .set_by_id("bound_water_capacity_g_per_g", 0.4)
            .unwrap();
        profile.set_by_id("drag_multiplier", 2.0).unwrap();
        profile.set_by_id("absorption_rate_per_s", 0.25).unwrap();
        profile.set_by_id("bloom_delay_s", 3.0).unwrap();
        profile
            .set_by_id("max_diameter_swelling_fraction", 0.1)
            .unwrap();
        profile.set_by_id("dp_friction_angle_deg", 30.0).unwrap();
        profile
            .set_by_id("reference_bed_porosity_fraction", 0.45)
            .unwrap();
        profile
            .set_by_id("extraction_fast_rate_per_s", 0.2)
            .unwrap();
        profile
            .set_by_id("extraction_slow_rate_per_s", 0.02)
            .unwrap();
        profile
            .set_by_id("bed_compaction_tangent_pa", 15_000.0)
            .unwrap();
        profile
            .set_by_id("initial_consolidation_pressure_pa", 800.0)
            .unwrap();
        profile
            .set_by_id("numerical_shear_modulus_pa", 40.0)
            .unwrap();
        profile
            .set_by_id("log10_filter_paper_permeability_m2", -12.0)
            .unwrap();
        profile
            .set_by_id("filter_paper_thickness_mm", 0.35)
            .unwrap();
        profile
            .set_by_id("filter_paper_porosity_fraction", 0.62)
            .unwrap();
        profile.set_by_id("filter_water_capacity_g", 2.7).unwrap();

        let (_, materials, config) =
            setup_for_scene_spec(&SceneSpec::CenterPour, &profile).unwrap();
        assert_eq!(materials.r_max, profile.bound_water_capacity_g_per_g());
        assert_eq!(
            materials.porosity,
            profile.reference_bed_porosity_fraction()
        );
        assert_eq!(materials.friction_mu, profile.dp_friction_mu());
        assert_eq!(materials.k0_fast, profile.extraction_fast_rate_per_s());
        assert_eq!(materials.k0_slow, profile.extraction_slow_rate_per_s());
        assert_eq!(config.tf_absorb_rate, profile.absorption_rate_per_s());
        assert_eq!(config.tf_bloom_delay, profile.bloom_delay_s());
        assert_eq!(
            config.tf_max_diameter_swelling,
            profile.max_diameter_swelling_fraction()
        );
        assert_eq!(
            config.drag_scale,
            calibrated_drag_scale(&materials, &profile).unwrap()
        );
        assert_eq!(
            config.tf_compaction_tangent,
            reduced_stress_from_pa(
                &materials,
                config.tf_water_density_kg_m3,
                profile.bed_compaction_tangent_pa(),
            )
            .unwrap()
        );
        assert_eq!(
            config.tf_initial_consolidation_pressure,
            reduced_stress_from_pa(
                &materials,
                config.tf_water_density_kg_m3,
                profile.initial_consolidation_pressure_pa()
            )
            .unwrap()
        );
        assert_eq!(
            config.tf_numerical_shear_modulus,
            reduced_stress_from_pa(
                &materials,
                config.tf_water_density_kg_m3,
                profile.numerical_shear_modulus_pa(),
            )
            .unwrap()
        );
        assert_eq!(
            config.tf_filter_permeability_m2,
            profile.filter_paper_permeability_m2()
        );
        assert_eq!(
            config.tf_filter_thickness_m,
            profile.filter_paper_thickness_m()
        );
        assert_eq!(
            config.tf_filter_porosity,
            profile.filter_paper_porosity_fraction()
        );
        assert_eq!(
            config.tf_filter_capacity_g,
            profile.filter_water_capacity_g()
        );
        assert!(config.tf_filter_kinematic_viscosity_m2_s.is_finite());
        assert!(config.tf_filter_kinematic_viscosity_m2_s > 0.0);
        assert!(config.tf_water_density_kg_m3.is_finite());
        assert!(config.tf_water_density_kg_m3 > 0.0);
    }

    #[test]
    fn calibration_support_does_not_claim_unwired_physics() {
        assert_eq!(
            calibration_support(CalibrationControl::DragMultiplier),
            CalibrationSupport::Rebuild
        );
        assert_eq!(
            calibration_support(CalibrationControl::MaxDiameterSwelling),
            CalibrationSupport::Rebuild
        );
        assert_eq!(
            calibration_support(CalibrationControl::Log10Permeability),
            CalibrationSupport::Rebuild
        );
        assert_eq!(
            calibration_support(CalibrationControl::SuctionHead),
            CalibrationSupport::Rebuild
        );
        assert_eq!(
            calibration_support(CalibrationControl::ReferenceBedPorosity),
            CalibrationSupport::Rebuild
        );
        for control in [
            CalibrationControl::DpCohesion,
            CalibrationControl::BedCompactionTangent,
            CalibrationControl::InitialConsolidationPressure,
            CalibrationControl::NumericalShearModulus,
            CalibrationControl::FullCompactionSolidFraction,
            CalibrationControl::ExtractionFastRate,
            CalibrationControl::ExtractionSlowRate,
            CalibrationControl::InletTemperature,
            CalibrationControl::AmbientTemperature,
            CalibrationControl::ThermalExchangeRate,
            CalibrationControl::BedRetention,
            CalibrationControl::CapillaryRelativePermeabilityExponent,
            CalibrationControl::CapillaryMobilityFloor,
            CalibrationControl::Log10FilterPaperPermeability,
            CalibrationControl::FilterWaterCapacity,
            CalibrationControl::FilterPaperThickness,
            CalibrationControl::FilterPaperPorosity,
        ] {
            assert_eq!(calibration_support(control), CalibrationSupport::Rebuild);
        }
    }
}

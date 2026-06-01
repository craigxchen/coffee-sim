use coffee_sim_core::Vec3;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{closure::Closure, JsCast};

pub(crate) mod bed;
mod brew_config;
mod dfsph;
mod filter;
mod filter_mesh;
pub(crate) mod inflow;
#[cfg(test)]
mod physics_tests;
mod pipelines;
mod pressure;
#[cfg(test)]
mod profiler;
mod shader;
mod state;
pub(crate) mod units;
#[cfg(test)]
mod xpbd;

pub(crate) use filter::FilterConfig;
#[cfg(target_arch = "wasm32")]
pub(crate) use filter_mesh::{MAX_FILL_VERTEX_COUNT, MAX_RENDER_VERTEX_COUNT};

use bed::{BedConfig, BedInit};
use brew_config::{kozeny_carman_permeability_m2, DEFAULT_BREW};
use filter_mesh::FilterMesh;
use inflow::{EmissionResult, InflowState, SpoutSettings, MASS_UNITS_PER_ML};
use pipelines::MpmPipelines;
use pressure::{PressureContext, PressureOperatorKind, PressureSolverKind};
use state::{
    MpmBuffers, MpmUniforms, FP_SCALE, FP_VALUE_LIMIT, MAX_VELOCITY, METRICS_DIV_FP_SCALE,
    METRICS_SLOT_COUNT, NUM_THREADS, SDF_RES,
};

const TARGET_BED_RETENTION_ML: f32 = DEFAULT_BREW.target_bed_retention_ml;
pub(crate) const CONTACT_OFFSET: f32 = 0.05;
pub(crate) const OBSTACLE_WALL_THICKNESS: f32 = 0.4;

#[cfg(test)]
const COMMON_TIMED_SCOPES_PER_SUBSTEP: u32 = 17;
#[cfg(test)]
const MIN_TIMESTAMP_QUERY_CAPACITY: u32 = 64;

#[derive(Clone, Copy)]
pub(crate) struct MpmDispatchSizes {
    pub cell_wg: u32,
    pub particle_wg: u32,
    pub bed_wg: u32,
    pub metrics_wg: u32,
}

#[derive(Clone, Copy)]
pub(crate) enum MpmDispatch<'a> {
    Direct(u32),
    Indirect {
        buffer: &'a wgpu::Buffer,
        offset: u64,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum MpmScheduleOp<'a> {
    Pipeline {
        label: MpmPassLabel,
        pipeline: &'a wgpu::ComputePipeline,
        dispatch: MpmDispatch<'a>,
    },
    PressureSolve {
        label: MpmPassLabel,
        pressure: &'a pressure::PressurePipelines,
        ctx: PressureContext,
    },
    CopyBufferToBuffer {
        src: &'a wgpu::Buffer,
        src_offset: u64,
        dst: &'a wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum MpmPassLabel {
    MetricsClear,
    BedLookupClear,
    BedLookupScatter,
    P2G,
    GridUpdate,
    BoundaryProject,
    PressureClassify,
    PressureSolve,
    PressureProject,
    PressureResidual,
    PackingPrepare,
    PackingApply,
    ViscosityPrepare,
    ViscosityApply,
    G2P,
    BedCoupling,
    ExtractionAdvect,
    BedDynamics,
    PrepareRender,
}

#[cfg(test)]
impl MpmPassLabel {
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::MetricsClear => "metrics.clear",
            Self::BedLookupClear => "bed.lookup_clear",
            Self::BedLookupScatter => "bed.lookup_scatter",
            Self::P2G => "transfer.p2g",
            Self::GridUpdate => "grid.update",
            Self::BoundaryProject => "boundary.project",
            Self::PressureClassify => "pressure.classify",
            Self::PressureSolve => "pressure.solve",
            Self::PressureProject => "pressure.project",
            Self::PressureResidual => "pressure.residual",
            Self::PackingPrepare => "packing.prepare",
            Self::PackingApply => "packing.apply",
            Self::ViscosityPrepare => "viscosity.prepare",
            Self::ViscosityApply => "viscosity.apply",
            Self::G2P => "transfer.g2p",
            Self::BedCoupling => "bed.coupling",
            Self::ExtractionAdvect => "bed.extraction_advect",
            Self::BedDynamics => "bed.dynamics",
            Self::PrepareRender => "render.prepare",
        }
    }

    pub(crate) fn display(self) -> &'static str {
        match self {
            Self::MetricsClear => "metrics_clear",
            Self::BedLookupClear => "bed_lookup_clear",
            Self::BedLookupScatter => "bed_lookup_scatter",
            Self::P2G => "p2g",
            Self::GridUpdate => "grid_update",
            Self::BoundaryProject => "boundary_project",
            Self::PressureClassify => "classify_cells",
            Self::PressureSolve => "pressure_solve",
            Self::PressureProject => "project_pressure",
            Self::PressureResidual => "pressure_residual",
            Self::PackingPrepare => "packing_prepare",
            Self::PackingApply => "packing_apply",
            Self::ViscosityPrepare => "viscosity_prepare",
            Self::ViscosityApply => "viscosity_apply",
            Self::G2P => "g2p",
            Self::BedCoupling => "bed_coupling",
            Self::ExtractionAdvect => "extraction_advect",
            Self::BedDynamics => "bed_dynamics",
            Self::PrepareRender => "prepare_render",
        }
    }

    pub(crate) fn category(self) -> &'static str {
        match self {
            Self::MetricsClear => "metrics",
            Self::BedLookupClear
            | Self::BedLookupScatter
            | Self::BedCoupling
            | Self::ExtractionAdvect
            | Self::BedDynamics => "bed",
            Self::P2G | Self::G2P => "transfer",
            Self::GridUpdate | Self::BoundaryProject => "grid",
            Self::PressureClassify
            | Self::PressureSolve
            | Self::PressureProject
            | Self::PressureResidual => "pressure",
            Self::PackingPrepare | Self::PackingApply => "packing",
            Self::ViscosityPrepare | Self::ViscosityApply => "viscosity",
            Self::PrepareRender => "render",
        }
    }
}

/// Device limits required by the MPM compute pipeline.
///
/// The MPM bind group holds 10 storage buffers (particles, affine, grid,
/// pressure CG scratch, grid_vel, render_data, bed_extract, bed_lookup, bed_delta, metrics) plus
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
    /// Peak absolute divergence residual after pressure projection and boundary
    /// projection in the most recent sampled substep.
    pub projection_residual_max_abs_div: f32,
    /// Mean absolute divergence residual over the cells sampled by
    /// `pressure_residual`.
    pub projection_residual_mean_abs_div: f32,
    pub projection_residual_cells: u32,
    pub pressure_active_cells: u32,
    pub pressure_cg_rz: f32,
    pub pressure_cg_pap: f32,
    pub pressure_cg_new_rz: f32,
    pub pressure_cg_initial_rz: f32,
    pub pressure_cg_final_rz: f32,
    pub mean_tds: f32,
    pub cup_tds: f32,
    pub extraction_yield: f32,
}

/// One-shot water-state readback used by the browser realism evaluator.
///
/// These are deliberately physical-ish quantities rather than shader internals:
/// mass/volume conservation, kinetic energy, net momentum, and a coarse
/// top-surface roughness estimate over the cup pool. The visual evaluator can
/// pair this with canvas snapshots so we stop relying on vibes alone.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WaterDiagnostics {
    pub all_finite: bool,
    pub sim_time_s: f32,
    pub active_count: u32,
    pub pool_count: u32,
    pub active_mass: f32,
    pub active_mass_ml: f32,
    pub emitted_ml: f32,
    pub rest_volume_ml: f32,
    pub current_volume_ml: f32,
    pub mean_j: f32,
    pub kinetic_energy: f32,
    pub rms_speed: f32,
    pub vertical_rms_speed: f32,
    pub mean_vertical_speed: f32,
    pub lateral_rms_speed: f32,
    pub max_speed: f32,
    pub max_upward_speed: f32,
    pub max_downward_speed: f32,
    pub upward_momentum: f32,
    pub downward_momentum: f32,
    pub vertical_dipole: Vec3,
    pub vertical_dipole_magnitude: f32,
    pub momentum: Vec3,
    pub momentum_magnitude: f32,
    pub centroid: Vec3,
    pub min: Vec3,
    pub max: Vec3,
    pub extent: Vec3,
    pub surface_bin_count: u32,
    pub surface_possible_bins: u32,
    pub surface_mean_y: f32,
    pub surface_rms_y: f32,
    pub surface_min_y: f32,
    pub surface_max_y: f32,
    pub surface_peak_to_peak_y: f32,
    pub surface_tilt: Vec3,
    pub surface_tilt_magnitude: f32,
    pub surface_tilt_height_y: f32,
    pub surface_residual_rms_y: f32,
    pub surface_residual_peak_to_peak_y: f32,
    pub hydrostatic_sample_count: u32,
    pub hydrostatic_depth_m: f32,
    pub hydrostatic_top_pressure_pa: f32,
    pub hydrostatic_bottom_pressure_pa: f32,
    pub hydrostatic_delta_pressure_pa: f32,
    pub hydrostatic_gradient_pa_per_m: f32,
    pub hydrostatic_bottom_higher: bool,
    pub dissolved_solute_mass: f32,
    pub mean_tds: f32,
    pub cup_water_mass: f32,
    pub cup_solute_mass: f32,
    pub cup_tds: f32,
    pub extraction_yield: f32,
}

impl Default for WaterDiagnostics {
    fn default() -> Self {
        Self {
            all_finite: true,
            sim_time_s: 0.0,
            active_count: 0,
            pool_count: 0,
            active_mass: 0.0,
            active_mass_ml: 0.0,
            emitted_ml: 0.0,
            rest_volume_ml: 0.0,
            current_volume_ml: 0.0,
            mean_j: 0.0,
            kinetic_energy: 0.0,
            rms_speed: 0.0,
            vertical_rms_speed: 0.0,
            mean_vertical_speed: 0.0,
            lateral_rms_speed: 0.0,
            max_speed: 0.0,
            max_upward_speed: 0.0,
            max_downward_speed: 0.0,
            upward_momentum: 0.0,
            downward_momentum: 0.0,
            vertical_dipole: Vec3::ZERO,
            vertical_dipole_magnitude: 0.0,
            momentum: Vec3::ZERO,
            momentum_magnitude: 0.0,
            centroid: Vec3::ZERO,
            min: Vec3::ZERO,
            max: Vec3::ZERO,
            extent: Vec3::ZERO,
            surface_bin_count: 0,
            surface_possible_bins: 0,
            surface_mean_y: 0.0,
            surface_rms_y: 0.0,
            surface_min_y: 0.0,
            surface_max_y: 0.0,
            surface_peak_to_peak_y: 0.0,
            surface_tilt: Vec3::ZERO,
            surface_tilt_magnitude: 0.0,
            surface_tilt_height_y: 0.0,
            surface_residual_rms_y: 0.0,
            surface_residual_peak_to_peak_y: 0.0,
            hydrostatic_sample_count: 0,
            hydrostatic_depth_m: 0.0,
            hydrostatic_top_pressure_pa: 0.0,
            hydrostatic_bottom_pressure_pa: 0.0,
            hydrostatic_delta_pressure_pa: 0.0,
            hydrostatic_gradient_pa_per_m: 0.0,
            hydrostatic_bottom_higher: false,
            dissolved_solute_mass: 0.0,
            mean_tds: 0.0,
            cup_water_mass: 0.0,
            cup_solute_mass: 0.0,
            cup_tds: 0.0,
            extraction_yield: 0.0,
        }
    }
}

fn metrics_snapshot_from_raw(raw: &[u32], has_bed: bool) -> Option<MetricsSnapshot> {
    if raw.len() < METRICS_SLOT_COUNT {
        return None;
    }

    let active_water_mass =
        raw[state::METRIC_ACTIVE_WATER_MASS_IDX] as f32 / state::METRICS_MASS_FP_SCALE;
    let active_solute_mass =
        raw[state::METRIC_ACTIVE_SOLUTE_MASS_IDX] as f32 / state::METRICS_SOLUTE_FP_SCALE;
    let cup_water_mass =
        raw[state::METRIC_CUP_WATER_MASS_IDX] as f32 / state::METRICS_MASS_FP_SCALE;
    let cup_solute_mass =
        raw[state::METRIC_CUP_SOLUTE_MASS_IDX] as f32 / state::METRICS_SOLUTE_FP_SCALE;
    let projection_residual_cells = raw[state::METRIC_PROJECTION_RESIDUAL_CELLS_IDX];
    let projection_residual_sum = raw[state::METRIC_PROJECTION_RESIDUAL_SUM_IDX] as f32
        / state::METRICS_RESIDUAL_SUM_FP_SCALE;
    let pressure_cg_initial_rz =
        raw[state::METRIC_PRESSURE_INITIAL_RZ_IDX] as f32 / state::METRICS_CG_DOT_FP_SCALE;
    let pressure_cg_final_rz =
        raw[state::METRIC_PRESSURE_FINAL_RZ_IDX] as f32 / state::METRICS_CG_DOT_FP_SCALE;
    let pressure_cg_rz = raw[state::METRIC_CG_RZ_IDX] as f32 / state::METRICS_CG_DOT_FP_SCALE;
    let pressure_cg_pap = raw[state::METRIC_CG_PAP_IDX] as f32 / state::METRICS_CG_DOT_FP_SCALE;
    let pressure_cg_new_rz =
        raw[state::METRIC_CG_NEW_RZ_IDX] as f32 / state::METRICS_CG_DOT_FP_SCALE;

    Some(MetricsSnapshot {
        max_abs_div: raw[state::METRIC_MAX_ABS_DIV_IDX] as f32 / METRICS_DIV_FP_SCALE,
        fluid_cells: raw[state::METRIC_FLUID_CELLS_IDX],
        div_clamp_fires: raw[state::METRIC_DIV_CLAMP_FIRES_IDX],
        pressure_clamp_fires: raw[state::METRIC_PRESSURE_CLAMP_FIRES_IDX],
        mass_overflow_fires: raw[state::METRIC_MASS_OVERFLOW_FIRES_IDX],
        projection_residual_max_abs_div: raw[state::METRIC_PROJECTION_RESIDUAL_MAX_IDX] as f32
            / METRICS_DIV_FP_SCALE,
        projection_residual_mean_abs_div: projection_residual_sum
            / (projection_residual_cells as f32).max(1.0),
        projection_residual_cells,
        pressure_active_cells: raw[state::METRIC_PRESSURE_ACTIVE_COUNT_IDX],
        pressure_cg_rz,
        pressure_cg_pap,
        pressure_cg_new_rz,
        pressure_cg_initial_rz,
        pressure_cg_final_rz,
        mean_tds: active_solute_mass / active_water_mass.max(1e-6),
        cup_tds: cup_solute_mass / cup_water_mass.max(1e-6),
        extraction_yield: if has_bed {
            active_solute_mass / (DEFAULT_BREW.coffee_dose_g * MASS_UNITS_PER_ML)
        } else {
            0.0
        },
    })
}

#[cfg(target_arch = "wasm32")]
async fn wait_animation_frames(frames: u32) -> Result<(), JsValue> {
    for _ in 0..frames {
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let Some(window) = web_sys::window() else {
                let _ = resolve.call0(&JsValue::NULL);
                return;
            };
            let callback = Closure::once_into_js(move |_timestamp: f64| {
                let _ = resolve.call0(&JsValue::NULL);
            });
            if let Err(err) = window.request_animation_frame(callback.unchecked_ref()) {
                let _ = reject.call1(&JsValue::NULL, &err);
            }
        });
        wasm_bindgen_futures::JsFuture::from(promise).await?;
    }
    Ok(())
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
    pub pressure_solver: PressureSolverKind,
    pub pressure_operator: PressureOperatorKind,
    pub pressure_rbgs_pairs: u32,
    pub pressure_cg_iterations: u32,
    /// Optional residual target for browser-driven adaptive pressure solves.
    /// `<= 0` keeps the fixed `pressure_rbgs_pairs` behavior.
    pub pressure_residual_target: f32,
    pub pressure_rbgs_max_pairs: u32,
    pub use_sdf_cache: bool,
    pub obstacles: Vec<Obstacle>,
    pub spout: SpoutSettings,
    pub initial_water_speed_m_s: f32,
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
            pressure_solver: PressureSolverKind::Rbgs,
            pressure_operator: PressureOperatorKind::Collocated,
            pressure_rbgs_pairs: 40,
            pressure_cg_iterations: 40,
            pressure_residual_target: 0.0,
            pressure_rbgs_max_pairs: 40,
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
            initial_water_speed_m_s: DEFAULT_BREW.initial_water_speed_m_s,
            filter: Some(filter),
            bed: Some(bed),
        }
    }

    pub fn benchmark_free_stream() -> Self {
        let mut settings = Self::default_v60();
        settings.bed = None;
        settings.spout.origin = Vec3::new(0.0, 6.8, 0.0);
        // Free-water pools have no porous bed to dissipate pressure-solve
        // residuals, so the water-only scene needs tighter projection
        // convergence to keep the cup surface from settling into a lopsided
        // heap.
        settings.pressure_rbgs_pairs = 80;
        settings.initial_water_speed_m_s = DEFAULT_BREW.initial_water_speed_m_s;
        settings
    }

    pub fn benchmark_center_pour() -> Self {
        let mut settings = Self::default_v60();
        settings.spout.origin = Vec3::new(0.0, 7.1, 0.0);
        settings.initial_water_speed_m_s = DEFAULT_BREW.initial_water_speed_m_s;
        settings
    }

    pub fn benchmark_filter_water_block() -> Self {
        let mut settings = Self::default_v60();
        settings.spout.origin = Vec3::new(0.0, 7.1, 0.0);
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 80;
        settings
    }

    pub fn debug_off_center_filter_wall_pour() -> Self {
        let mut settings = Self::default_v60();
        if let Some(filter) = settings.filter.as_ref() {
            let local_y = 1.35;
            let wall_radius = filter.inner_radius_at_y(local_y);
            settings.spout.origin = Vec3::new((wall_radius - 0.34).max(0.0), 7.1, 0.0);
        }
        settings.initial_water_speed_m_s = 0.28;
        settings.spout.nozzle_radius = 0.15;
        settings.spout.max_flow_rate_ml_s = 10.0;
        settings.spout.max_exit_speed = units::sim_speed_from_meters_per_second(0.6);
        settings.pressure_rbgs_pairs = 80;
        settings
    }

    pub fn debug_seeded_paper_wall_sheet() -> Self {
        let mut settings = Self::default_v60();
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 80;
        settings
    }

    pub fn debug_filter_apex_drain() -> Self {
        let mut settings = Self::default_v60();
        settings.bed = None;
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 80;
        settings
    }

    pub fn debug_cup_wall_floor_corner_contact() -> Self {
        let mut settings = Self::cup_only_water_scene();
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 90;
        settings
    }

    pub fn debug_asymmetric_cup_mound() -> Self {
        let mut settings = Self::cup_only_water_scene();
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 90;
        settings
    }

    pub fn debug_hydrostatic_column() -> Self {
        let mut settings = Self::cup_only_water_scene();
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 100;
        settings
    }

    pub fn debug_dam_break_slosh() -> Self {
        let mut settings = Self::cup_only_water_scene();
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 90;
        settings
    }

    pub fn debug_sparse_free_jet() -> Self {
        let mut settings = Self::cup_only_water_scene();
        settings.spout.origin = Vec3::new(0.0, 7.4, 0.0);
        settings.spout.nozzle_radius = 0.07;
        settings.spout.max_flow_rate_ml_s = 2.0;
        settings.initial_water_speed_m_s = 0.18;
        settings
    }

    pub fn debug_high_velocity_jet_impact() -> Self {
        let mut settings = Self::cup_only_water_scene();
        settings.spout.origin = Vec3::new(0.0, 6.9, 0.0);
        settings.spout.nozzle_radius = 0.14;
        settings.spout.max_flow_rate_ml_s = 14.0;
        settings.spout.max_exit_speed = units::sim_speed_from_meters_per_second(0.65);
        settings.initial_water_speed_m_s = 0.48;
        settings.pressure_rbgs_pairs = 100;
        settings
    }

    pub fn debug_uniform_bed_saturation() -> Self {
        let mut settings = Self::default_v60();
        settings.initial_water_speed_m_s = 0.0;
        settings.pressure_rbgs_pairs = 80;
        settings
    }

    pub fn debug_permeability_comparison() -> Self {
        let mut settings = Self::default_v60();
        if let Some(bed) = settings.bed.as_mut() {
            bed.initial_permeability = kozeny_carman_permeability_m2(320.0, bed.initial_porosity);
        }
        settings.spout.origin = Vec3::new(0.0, 7.1, 0.0);
        settings.initial_water_speed_m_s = 0.18;
        settings.pressure_rbgs_pairs = 80;
        settings
    }

    pub fn debug_particle_capacity_stress() -> Self {
        let mut settings = Self::default_v60();
        settings.max_particles = 32_000;
        settings.spout.origin = Vec3::new(0.0, 7.1, 0.0);
        settings.spout.nozzle_radius = 0.20;
        settings.spout.max_flow_rate_ml_s = 18.0;
        settings.spout.max_exit_speed = units::sim_speed_from_meters_per_second(0.65);
        settings.initial_water_speed_m_s = 0.42;
        settings.pressure_rbgs_pairs = 80;
        settings
    }

    fn cup_only_water_scene() -> Self {
        let mut settings = Self::benchmark_free_stream();
        settings.filter = None;
        settings.bed = None;
        settings
            .obstacles
            .retain(|obstacle| matches!(obstacle, Obstacle::Cylinder { .. }));
        settings
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DebugScene {
    FilterWaterBlock,
    OffCenterFilterWallPour,
    SeededPaperWallSheet,
    FilterApexDrain,
    CupWallFloorCornerContact,
    AsymmetricCupMoundSettle,
    HydrostaticColumn,
    DamBreakSlosh,
    SparseFreeJet,
    HighVelocityJetImpact,
    UniformBedSaturation,
    PermeabilityComparison,
    ParticleCapacityStress,
}

impl DebugScene {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 13] = [
        Self::FilterWaterBlock,
        Self::OffCenterFilterWallPour,
        Self::SeededPaperWallSheet,
        Self::FilterApexDrain,
        Self::CupWallFloorCornerContact,
        Self::AsymmetricCupMoundSettle,
        Self::HydrostaticColumn,
        Self::DamBreakSlosh,
        Self::SparseFreeJet,
        Self::HighVelocityJetImpact,
        Self::UniformBedSaturation,
        Self::PermeabilityComparison,
        Self::ParticleCapacityStress,
    ];

    pub(crate) fn from_id(id: &str) -> Option<Self> {
        match id {
            "filter-water-block" => Some(Self::FilterWaterBlock),
            "off-center-filter-wall-pour" => Some(Self::OffCenterFilterWallPour),
            "seeded-paper-wall-sheet" => Some(Self::SeededPaperWallSheet),
            "filter-apex-drain" => Some(Self::FilterApexDrain),
            "cup-wall-floor-corner-contact" => Some(Self::CupWallFloorCornerContact),
            "asymmetric-cup-mound-settle" => Some(Self::AsymmetricCupMoundSettle),
            "hydrostatic-column" => Some(Self::HydrostaticColumn),
            "dam-break-slosh" => Some(Self::DamBreakSlosh),
            "sparse-free-jet" => Some(Self::SparseFreeJet),
            "high-velocity-jet-impact" => Some(Self::HighVelocityJetImpact),
            "uniform-bed-saturation" => Some(Self::UniformBedSaturation),
            "permeability-comparison" => Some(Self::PermeabilityComparison),
            "particle-capacity-stress" => Some(Self::ParticleCapacityStress),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::FilterWaterBlock => "filter-water-block",
            Self::OffCenterFilterWallPour => "off-center-filter-wall-pour",
            Self::SeededPaperWallSheet => "seeded-paper-wall-sheet",
            Self::FilterApexDrain => "filter-apex-drain",
            Self::CupWallFloorCornerContact => "cup-wall-floor-corner-contact",
            Self::AsymmetricCupMoundSettle => "asymmetric-cup-mound-settle",
            Self::HydrostaticColumn => "hydrostatic-column",
            Self::DamBreakSlosh => "dam-break-slosh",
            Self::SparseFreeJet => "sparse-free-jet",
            Self::HighVelocityJetImpact => "high-velocity-jet-impact",
            Self::UniformBedSaturation => "uniform-bed-saturation",
            Self::PermeabilityComparison => "permeability-comparison",
            Self::ParticleCapacityStress => "particle-capacity-stress",
        }
    }

    pub(crate) fn settings(self) -> MpmSettings {
        match self {
            Self::FilterWaterBlock => MpmSettings::benchmark_filter_water_block(),
            Self::OffCenterFilterWallPour => MpmSettings::debug_off_center_filter_wall_pour(),
            Self::SeededPaperWallSheet => MpmSettings::debug_seeded_paper_wall_sheet(),
            Self::FilterApexDrain => MpmSettings::debug_filter_apex_drain(),
            Self::CupWallFloorCornerContact => MpmSettings::debug_cup_wall_floor_corner_contact(),
            Self::AsymmetricCupMoundSettle => MpmSettings::debug_asymmetric_cup_mound(),
            Self::HydrostaticColumn => MpmSettings::debug_hydrostatic_column(),
            Self::DamBreakSlosh => MpmSettings::debug_dam_break_slosh(),
            Self::SparseFreeJet => MpmSettings::debug_sparse_free_jet(),
            Self::HighVelocityJetImpact => MpmSettings::debug_high_velocity_jet_impact(),
            Self::UniformBedSaturation => MpmSettings::debug_uniform_bed_saturation(),
            Self::PermeabilityComparison => MpmSettings::debug_permeability_comparison(),
            Self::ParticleCapacityStress => MpmSettings::debug_particle_capacity_stress(),
        }
    }

    pub(crate) fn seed(self, sim: &mut MpmSim3D, queue: &wgpu::Queue) {
        match self {
            Self::FilterWaterBlock => sim.seed_filter_water_block(queue),
            Self::SeededPaperWallSheet => sim.seed_paper_wall_sheet(queue),
            Self::FilterApexDrain => sim.seed_filter_apex_drain(queue),
            Self::CupWallFloorCornerContact => sim.seed_cup_wall_floor_corner_contact(queue),
            Self::AsymmetricCupMoundSettle => sim.seed_asymmetric_cup_mound(queue),
            Self::HydrostaticColumn => sim.seed_hydrostatic_column(queue),
            Self::DamBreakSlosh => sim.seed_dam_break_slosh(queue),
            Self::HighVelocityJetImpact => sim.seed_high_velocity_jet_impact_pool(queue),
            Self::UniformBedSaturation => sim.seed_uniform_bed_saturation(queue),
            Self::OffCenterFilterWallPour
            | Self::SparseFreeJet
            | Self::PermeabilityComparison
            | Self::ParticleCapacityStress => {}
        }
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

fn cup_region(settings: &MpmSettings) -> Option<(f32, f32, f32)> {
    settings
        .obstacles
        .iter()
        .find_map(|obstacle| match obstacle {
            Obstacle::Cylinder {
                radius,
                top_y,
                bot_y,
                ..
            } => Some((*radius, *top_y, *bot_y)),
            _ => None,
        })
}

fn cup_region_full(settings: &MpmSettings) -> Option<(Vec3, f32, f32, f32)> {
    settings
        .obstacles
        .iter()
        .find_map(|obstacle| match obstacle {
            Obstacle::Cylinder {
                center,
                radius,
                top_y,
                bot_y,
            } => Some((*center, *radius, *top_y, *bot_y)),
            _ => None,
        })
}

fn deterministic_unit_float(mut value: u32) -> f32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}

fn filter_config_key(filter: &FilterConfig) -> u64 {
    let mut key = 0xcbf2_9ce4_8422_2325u64;
    for bits in [
        filter.center.x.to_bits(),
        filter.center.y.to_bits(),
        filter.center.z.to_bits(),
        filter.top_y.to_bits(),
        filter.bot_y.to_bits(),
        filter.top_radius.to_bits(),
        filter.bot_radius.to_bits(),
        filter.thickness.to_bits(),
        filter.hole_radius.to_bits(),
    ] {
        key ^= u64::from(bits);
        key = key.wrapping_mul(0x0000_0100_0000_01b3);
    }
    key
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
    last_pressure_rbgs_pairs: u32,
    latest_metrics: MetricsSnapshot,
}

fn encode_mpm_substep_schedule<'a, RunOp>(
    pipelines: &'a MpmPipelines,
    buffers: &'a MpmBuffers,
    dispatch: MpmDispatchSizes,
    pressure_ctx: PressureContext,
    mut run_op: RunOp,
) where
    RunOp: FnMut(MpmScheduleOp<'a>),
{
    let common = &pipelines.common;
    macro_rules! run_pipeline {
        ($label:expr, $pipeline:expr, $workgroups:expr $(,)?) => {
            run_op(MpmScheduleOp::Pipeline {
                label: $label,
                pipeline: $pipeline,
                dispatch: MpmDispatch::Direct($workgroups),
            });
        };
    }

    if dispatch.metrics_wg > 0 {
        run_pipeline!(
            MpmPassLabel::MetricsClear,
            &common.metrics_clear,
            dispatch.metrics_wg,
        );
    }
    run_pipeline!(
        MpmPassLabel::BedLookupClear,
        &common.bed_lookup_clear,
        dispatch.cell_wg,
    );
    if dispatch.bed_wg > 0 {
        run_pipeline!(
            MpmPassLabel::BedLookupScatter,
            &common.bed_lookup_scatter,
            dispatch.bed_wg,
        );
    }
    if dispatch.particle_wg > 0 {
        run_pipeline!(MpmPassLabel::P2G, &common.p2g, dispatch.particle_wg);
    }
    run_pipeline!(
        MpmPassLabel::GridUpdate,
        &common.grid_update,
        dispatch.cell_wg
    );
    run_pipeline!(
        MpmPassLabel::BoundaryProject,
        &common.boundary_project,
        dispatch.cell_wg,
    );

    run_pipeline!(
        MpmPassLabel::PressureClassify,
        pipelines.pressure.classify_pipeline_for(pressure_ctx.kind),
        dispatch.cell_wg,
    );
    pipelines
        .pressure
        .encode_solve_ops(buffers, pressure_ctx, &mut run_op);
    run_pipeline!(
        MpmPassLabel::PressureProject,
        pipelines.pressure.project_pipeline_for(pressure_ctx.kind),
        dispatch.cell_wg,
    );
    run_pipeline!(
        MpmPassLabel::BoundaryProject,
        &common.boundary_project,
        dispatch.cell_wg,
    );
    run_pipeline!(
        MpmPassLabel::PressureResidual,
        pipelines.pressure.residual_pipeline_for(pressure_ctx.kind),
        dispatch.cell_wg,
    );

    run_pipeline!(
        MpmPassLabel::PackingPrepare,
        &common.packing_prepare,
        dispatch.cell_wg,
    );
    run_pipeline!(
        MpmPassLabel::PackingApply,
        &common.packing_apply,
        dispatch.cell_wg,
    );
    run_pipeline!(
        MpmPassLabel::BoundaryProject,
        &common.boundary_project,
        dispatch.cell_wg,
    );

    run_pipeline!(
        MpmPassLabel::ViscosityPrepare,
        &common.viscosity_prepare,
        dispatch.cell_wg,
    );
    run_pipeline!(
        MpmPassLabel::ViscosityApply,
        &common.viscosity_apply,
        dispatch.cell_wg,
    );
    run_pipeline!(
        MpmPassLabel::BoundaryProject,
        &common.boundary_project,
        dispatch.cell_wg,
    );

    if dispatch.particle_wg > 0 {
        run_pipeline!(MpmPassLabel::G2P, &common.g2p, dispatch.particle_wg);
        run_pipeline!(
            MpmPassLabel::BedCoupling,
            &common.bed_coupling,
            dispatch.particle_wg,
        );
    }
    if dispatch.bed_wg > 0 {
        run_pipeline!(
            MpmPassLabel::ExtractionAdvect,
            &common.extraction_advect,
            dispatch.bed_wg,
        );
        run_pipeline!(
            MpmPassLabel::BedDynamics,
            &common.bed_dynamics,
            dispatch.bed_wg,
        );
    }
    if dispatch.particle_wg > 0 {
        run_pipeline!(
            MpmPassLabel::PrepareRender,
            &common.prepare_render,
            dispatch.particle_wg,
        );
    }
}

fn encode_mpm_compute_ops(pass: &mut wgpu::ComputePass<'_>, ops: &[MpmScheduleOp<'_>]) {
    for op in ops {
        match *op {
            MpmScheduleOp::Pipeline {
                label,
                pipeline,
                dispatch,
            } => {
                let _ = label;
                pass.set_pipeline(pipeline);
                match dispatch {
                    MpmDispatch::Direct(workgroups) => {
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                    MpmDispatch::Indirect { buffer, offset } => {
                        pass.dispatch_workgroups_indirect(buffer, offset);
                    }
                }
            }
            MpmScheduleOp::PressureSolve {
                label,
                pressure,
                ctx,
            } => {
                let _ = label;
                pressure.encode_solve(pass, ctx);
            }
            MpmScheduleOp::CopyBufferToBuffer { .. } => {}
        }
    }
}

fn encode_mpm_schedule_production(
    encoder: &mut wgpu::CommandEncoder,
    bind_group: &wgpu::BindGroup,
    ops: &[MpmScheduleOp<'_>],
) {
    let mut batch_start = 0usize;
    for (i, op) in ops.iter().enumerate() {
        if let MpmScheduleOp::CopyBufferToBuffer {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        } = *op
        {
            if batch_start < i {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mpm compute"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, bind_group, &[]);
                encode_mpm_compute_ops(&mut pass, &ops[batch_start..i]);
            }
            encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
            batch_start = i + 1;
        }
    }

    if batch_start < ops.len() {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mpm compute"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, bind_group, &[]);
        encode_mpm_compute_ops(&mut pass, &ops[batch_start..]);
    }
}

impl MpmSim3D {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, settings: MpmSettings) -> Self {
        let buffers = MpmBuffers::new(device, queue, &settings);
        let pipelines = MpmPipelines::new(device, &buffers);
        let mut inflow = InflowState::new(units::sim_speed_from_meters_per_second(
            settings.initial_water_speed_m_s,
        ));
        inflow.update(&settings.spout);
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
            last_pressure_rbgs_pairs: 0,
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
        // Match the shader's six-lane bed_delta layout: water, impulse xyz,
        // mobile solute availability, and solute deposited by absorbed water.
        let zero_delta = vec![0_i32; self.settings.max_particles as usize * 6];
        queue.write_buffer(
            &self.buffers.bed_delta,
            0,
            bytemuck::cast_slice(&zero_delta),
        );
    }

    pub fn seed_filter_water_block(&mut self, queue: &wgpu::Queue) {
        let Some(filter) = self.settings.filter.as_ref() else {
            return;
        };

        let [gx, _, _] = self.settings.grid_dims;
        let dx = self.settings.bounds_size.x / gx as f32;
        let spacing = dx * 0.80;
        let particle_mass = MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let available = self.settings.max_particles.saturating_sub(self.num_bed);
        if available == 0 {
            return;
        }

        let bed_top_y = self
            .settings
            .bed
            .as_ref()
            .map(|bed| bed.center.y + bed.top_y)
            .unwrap_or(filter.center.y + filter.bot_y + 0.8);
        let y_min = bed_top_y + spacing * 1.5;
        let y_max = (filter.center.y + filter.top_y - spacing * 3.0).min(y_min + 1.65);
        let wall_margin = dx * 3.0;
        let golden_angle = 2.399_963_1_f32;

        let mut particle_data: Vec<[f32; 8]> = Vec::new();

        let mut layer = 0_u32;
        let mut y = y_min;
        while y <= y_max && particle_data.len() < available as usize {
            let local_y = y - filter.center.y;
            let inner_radius = (filter.inner_radius_at_y(local_y) - wall_margin).max(0.0);
            let layer_area = std::f32::consts::PI * inner_radius * inner_radius;
            let sample_area = spacing * spacing * 0.92;
            let samples = (layer_area / sample_area).ceil().max(1.0) as u32;
            let layer_rotation = layer as f32 * 0.618_034;
            for i in 0..samples {
                if particle_data.len() >= available as usize {
                    break;
                }
                let seed = layer.wrapping_mul(1_664_525).wrapping_add(i);
                let radius_jitter =
                    (deterministic_unit_float(seed ^ 0x9e37_79b9) - 0.5) * spacing * 0.30;
                let angle_jitter = (deterministic_unit_float(seed ^ 0x85eb_ca6b) - 0.5) * 0.22;
                let t = (i as f32 + 0.5) / samples as f32;
                let radius_limit = (inner_radius - spacing * 0.35).max(0.0);
                let radius = (inner_radius * t.sqrt() + radius_jitter).clamp(0.0, radius_limit);
                let angle = i as f32 * golden_angle + layer_rotation + angle_jitter;
                let x = filter.center.x + radius * angle.cos();
                let z = filter.center.z + radius * angle.sin();
                particle_data.push([x, y, z, 1.0, 0.0, 0.0, 0.0, particle_mass]);
            }
            layer += 1;
            y += spacing;
        }

        self.write_seeded_water(queue, particle_data, particle_mass, 0.0);
    }

    fn write_seeded_water(
        &mut self,
        queue: &wgpu::Queue,
        mut particle_data: Vec<[f32; 8]>,
        particle_mass: f32,
        exit_speed_m_s: f32,
    ) {
        let available = self.settings.max_particles.saturating_sub(self.num_bed) as usize;
        particle_data.truncate(available);
        if particle_data.is_empty() {
            self.num_water = 0;
            self.total_time = 0.0;
            self.frame_emitted_mass = 0.0;
            self.total_emitted_mass = 0.0;
            self.frame_dropped_particles = 0;
            self.total_dropped_particles = 0;
            self.last_pressure_rbgs_pairs = 0;
            self.latest_metrics = MetricsSnapshot::default();
            self.set_exit_speed_m_s(exit_speed_m_s);
            return;
        }

        let affine_data = vec![[0.0; 12]; particle_data.len()];
        let particle_offset = (self.num_bed as u64) * 32;
        let affine_offset = (self.num_bed as u64) * 48;
        queue.write_buffer(
            &self.buffers.particles,
            particle_offset,
            bytemuck::cast_slice(&particle_data),
        );
        queue.write_buffer(
            &self.buffers.affine,
            affine_offset,
            bytemuck::cast_slice(&affine_data),
        );

        self.num_water = particle_data.len() as u32;
        self.total_time = 0.0;
        self.frame_emitted_mass = particle_mass * self.num_water as f32;
        self.total_emitted_mass = self.frame_emitted_mass;
        self.frame_dropped_particles = 0;
        self.total_dropped_particles = 0;
        self.last_pressure_rbgs_pairs = 0;
        self.latest_metrics = MetricsSnapshot::default();
        self.set_exit_speed_m_s(exit_speed_m_s);
    }

    fn water_seed_spacing(&self, scale: f32) -> f32 {
        let [gx, _, _] = self.settings.grid_dims;
        self.settings.bounds_size.x / gx as f32 * scale
    }

    fn seed_filter_disc_layers<F>(
        &mut self,
        queue: &wgpu::Queue,
        y_min: f32,
        y_max: f32,
        mut radius_at_y: F,
        velocity: Vec3,
        exit_speed_m_s: f32,
    ) where
        F: FnMut(f32) -> f32,
    {
        let particle_mass = MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let available = self.settings.max_particles.saturating_sub(self.num_bed) as usize;
        let spacing = self.water_seed_spacing(0.86);
        let golden_angle = 2.399_963_1_f32;
        let mut particle_data = Vec::new();
        let mut layer = 0_u32;
        let mut y = y_min;

        while y <= y_max && particle_data.len() < available {
            let radius = radius_at_y(y).max(0.0);
            let layer_area = std::f32::consts::PI * radius * radius;
            let sample_area = spacing * spacing * 0.92;
            let samples = (layer_area / sample_area).ceil().max(1.0) as u32;
            let layer_rotation = layer as f32 * 0.618_034;
            for i in 0..samples {
                if particle_data.len() >= available {
                    break;
                }
                let seed = layer.wrapping_mul(1_103_515_245).wrapping_add(i);
                let radius_jitter =
                    (deterministic_unit_float(seed ^ 0x9e37_79b9) - 0.5) * spacing * 0.24;
                let angle_jitter = (deterministic_unit_float(seed ^ 0x85eb_ca6b) - 0.5) * 0.18;
                let t = (i as f32 + 0.5) / samples as f32;
                let radius_limit = (radius - spacing * 0.35).max(0.0);
                let r = (radius * t.sqrt() + radius_jitter).clamp(0.0, radius_limit);
                let angle = i as f32 * golden_angle + layer_rotation + angle_jitter;
                particle_data.push([
                    r * angle.cos(),
                    y,
                    r * angle.sin(),
                    1.0,
                    velocity.x,
                    velocity.y,
                    velocity.z,
                    particle_mass,
                ]);
            }
            layer += 1;
            y += spacing;
        }

        self.write_seeded_water(queue, particle_data, particle_mass, exit_speed_m_s);
    }

    fn seed_cup_volume<F>(
        &mut self,
        queue: &wgpu::Queue,
        y_min: f32,
        y_max: f32,
        mut include: F,
        velocity: Vec3,
        exit_speed_m_s: f32,
    ) where
        F: FnMut(Vec3) -> bool,
    {
        let Some((center, cup_radius, cup_top_y, cup_bot_y)) = cup_region_full(&self.settings)
        else {
            return;
        };

        let particle_mass = MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let available = self.settings.max_particles.saturating_sub(self.num_bed) as usize;
        let spacing = self.water_seed_spacing(0.88);
        let radius = (cup_radius - spacing * 1.5).max(0.0);
        let y_min = y_min.clamp(cup_bot_y + spacing, cup_top_y - spacing);
        let y_max = y_max.clamp(y_min, cup_top_y - spacing);
        let nx = ((radius * 2.0) / spacing).ceil() as i32;
        let ny = ((y_max - y_min) / spacing).ceil().max(1.0) as i32;

        let mut particle_data = Vec::new();
        for iy in 0..ny {
            if particle_data.len() >= available {
                break;
            }
            let y = y_min + (iy as f32 + 0.5) * spacing;
            for ix in 0..nx {
                if particle_data.len() >= available {
                    break;
                }
                let x = center.x - radius + (ix as f32 + 0.5) * spacing;
                for iz in 0..nx {
                    if particle_data.len() >= available {
                        break;
                    }
                    let z = center.z - radius + (iz as f32 + 0.5) * spacing;
                    let pos = Vec3::new(x, y, z);
                    let radial_sq = (pos.x - center.x).powi(2) + (pos.z - center.z).powi(2);
                    if radial_sq > radius * radius || !include(pos) {
                        continue;
                    }
                    particle_data.push([
                        pos.x,
                        pos.y,
                        pos.z,
                        1.0,
                        velocity.x,
                        velocity.y,
                        velocity.z,
                        particle_mass,
                    ]);
                }
            }
        }

        self.write_seeded_water(queue, particle_data, particle_mass, exit_speed_m_s);
    }

    pub fn seed_paper_wall_sheet(&mut self, queue: &wgpu::Queue) {
        let Some(filter) = self.settings.filter.as_ref() else {
            return;
        };

        let particle_mass = MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let available = self.settings.max_particles.saturating_sub(self.num_bed) as usize;
        let spacing = self.water_seed_spacing(0.86);
        let wall_margin = spacing * 1.6;
        let y_min = filter.center.y + 0.15;
        let y_max = (filter.center.y + filter.top_y - spacing * 3.0).min(y_min + 2.25);
        let arc_half_angle = 1.20_f32;
        let mut particle_data = Vec::new();
        let mut y = y_min;
        let mut layer = 0_u32;

        while y <= y_max && particle_data.len() < available {
            let local_y = y - filter.center.y;
            let wall_radius = filter.inner_radius_at_y(local_y);
            let r_inner = (wall_radius - wall_margin).max(0.0);
            let r_outer = (wall_radius - spacing * 0.55).max(r_inner);
            let arc_len = (r_outer * arc_half_angle * 2.0).max(spacing);
            let angle_samples = (arc_len / spacing).ceil().max(1.0) as u32;
            let radial_samples = ((r_outer - r_inner) / spacing).ceil().max(1.0) as u32;

            for ia in 0..angle_samples {
                for ir in 0..radial_samples {
                    if particle_data.len() >= available {
                        break;
                    }
                    let seed = layer
                        .wrapping_mul(1_664_525)
                        .wrapping_add(ia * 31)
                        .wrapping_add(ir);
                    let angle_jitter = (deterministic_unit_float(seed ^ 0x85eb_ca6b) - 0.5) * 0.16;
                    let radius_jitter =
                        (deterministic_unit_float(seed ^ 0xc2b2_ae35) - 0.5) * spacing * 0.18;
                    let angle = -arc_half_angle
                        + (ia as f32 + 0.5) / angle_samples as f32 * arc_half_angle * 2.0
                        + angle_jitter;
                    let r = (r_inner
                        + (ir as f32 + 0.5) / radial_samples as f32 * (r_outer - r_inner)
                        + radius_jitter)
                        .clamp(r_inner, r_outer);
                    particle_data.push([
                        filter.center.x + r * angle.cos(),
                        y,
                        filter.center.z + r * angle.sin(),
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        particle_mass,
                    ]);
                }
            }
            layer += 1;
            y += spacing;
        }

        self.write_seeded_water(queue, particle_data, particle_mass, 0.0);
    }

    pub fn seed_filter_apex_drain(&mut self, queue: &wgpu::Queue) {
        let Some(filter) = self.settings.filter.clone() else {
            return;
        };
        let spacing = self.water_seed_spacing(0.86);
        let y_min = filter.center.y + filter.bot_y + spacing * 3.0;
        let y_max = (y_min + 1.9).min(filter.center.y + filter.top_y - spacing * 4.0);
        self.seed_filter_disc_layers(
            queue,
            y_min,
            y_max,
            |y| {
                let local_y = y - filter.center.y;
                (filter.inner_radius_at_y(local_y) - spacing * 1.25)
                    .max(0.0)
                    .min(1.15)
            },
            Vec3::ZERO,
            0.0,
        );
    }

    pub fn seed_cup_wall_floor_corner_contact(&mut self, queue: &wgpu::Queue) {
        let Some((center, radius, _, bot_y)) = cup_region_full(&self.settings) else {
            return;
        };
        self.seed_cup_volume(
            queue,
            bot_y + 0.20,
            bot_y + 1.20,
            |pos| {
                let dx = pos.x - center.x;
                let dz = pos.z - center.z;
                let radial = (dx * dx + dz * dz).sqrt();
                let angle = dz.atan2(dx);
                radial >= radius * 0.58 && angle.abs() <= 0.72
            },
            Vec3::ZERO,
            0.0,
        );
    }

    pub fn seed_asymmetric_cup_mound(&mut self, queue: &wgpu::Queue) {
        let Some((_center, _radius, _, bot_y)) = cup_region_full(&self.settings) else {
            return;
        };
        let mound_center = Vec3::new(0.95, bot_y + 1.10, -0.55);
        self.seed_cup_volume(
            queue,
            bot_y + 0.25,
            bot_y + 2.35,
            |pos| {
                let dx = (pos.x - mound_center.x) / 1.55;
                let dy = (pos.y - mound_center.y) / 1.20;
                let dz = (pos.z - mound_center.z) / 1.35;
                dx * dx + dy * dy + dz * dz <= 1.0
            },
            Vec3::ZERO,
            0.0,
        );
    }

    pub fn seed_hydrostatic_column(&mut self, queue: &wgpu::Queue) {
        let Some((center, _, _, bot_y)) = cup_region_full(&self.settings) else {
            return;
        };
        self.seed_cup_volume(
            queue,
            bot_y + 0.25,
            bot_y + 3.85,
            |pos| {
                let dx = pos.x - center.x;
                let dz = pos.z - center.z;
                dx * dx + dz * dz <= 1.05 * 1.05
            },
            Vec3::ZERO,
            0.0,
        );
    }

    pub fn seed_dam_break_slosh(&mut self, queue: &wgpu::Queue) {
        let Some((center, _, _, bot_y)) = cup_region_full(&self.settings) else {
            return;
        };
        self.seed_cup_volume(
            queue,
            bot_y + 0.25,
            bot_y + 2.55,
            |pos| pos.x < center.x - 0.20,
            Vec3::ZERO,
            0.0,
        );
    }

    pub fn seed_high_velocity_jet_impact_pool(&mut self, queue: &wgpu::Queue) {
        let Some((center, _, _, bot_y)) = cup_region_full(&self.settings) else {
            return;
        };
        self.seed_cup_volume(
            queue,
            bot_y + 0.25,
            bot_y + 1.25,
            |pos| {
                let dx = pos.x - center.x;
                let dz = pos.z - center.z;
                dx * dx + dz * dz <= 2.35 * 2.35
            },
            Vec3::ZERO,
            self.settings.initial_water_speed_m_s,
        );
    }

    pub fn seed_uniform_bed_saturation(&mut self, queue: &wgpu::Queue) {
        let Some(bed) = self.settings.bed.clone() else {
            return;
        };
        let spacing = self.water_seed_spacing(0.92);
        let y_min = bed.center.y + bed.bot_y + spacing * 1.5;
        let y_max = bed.center.y + bed.top_y + spacing * 1.5;
        self.seed_filter_disc_layers(
            queue,
            y_min,
            y_max,
            |y| {
                let height = (bed.top_y - bed.bot_y).max(1e-6);
                let t = ((y - bed.center.y - bed.bot_y) / height).clamp(0.0, 1.0);
                let bed_radius = bed.bot_radius + (bed.top_radius - bed.bot_radius) * t;
                (bed_radius - spacing * 1.5).max(0.0)
            },
            Vec3::ZERO,
            0.0,
        );
    }

    fn water_diagnostics_from_particle_data(
        &self,
        data: &[f32],
        affine_data: &[f32],
    ) -> WaterDiagnostics {
        const SURFACE_BINS: usize = 16;

        let [gx, _, _] = self.settings.grid_dims;
        let dx = self.settings.bounds_size.x / gx as f32;
        let particle_vol = dx * dx * dx * 0.25;
        let nominal_mass = MASS_UNITS_PER_ML / inflow::PARTICLES_PER_ML;
        let inactive_thresh = nominal_mass * 0.1;
        let mut diagnostics = WaterDiagnostics {
            sim_time_s: self.total_time,
            emitted_ml: self.total_emitted_ml(),
            ..WaterDiagnostics::default()
        };

        let start = self.num_bed as usize;
        let end = start + self.num_water as usize;
        if data.len() < end * 8 || affine_data.len() < end * 12 {
            diagnostics.all_finite = false;
            return diagnostics;
        }

        let cup = cup_region(&self.settings);
        let mut surface_y = [f32::NEG_INFINITY; SURFACE_BINS * SURFACE_BINS];
        let mut surface_possible_bins = 0u32;
        if let Some((cup_radius, _, _)) = cup {
            for bx in 0..SURFACE_BINS {
                for bz in 0..SURFACE_BINS {
                    let x = (((bx as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
                    let z = (((bz as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
                    if x * x + z * z <= cup_radius * cup_radius {
                        surface_possible_bins += 1;
                    }
                }
            }
        }

        let mut centroid_sum = Vec3::ZERO;
        let mut mass_weighted_speed_sq = 0.0_f32;
        let mut mass_weighted_vertical_speed_sq = 0.0_f32;
        let mut mass_weighted_lateral_speed_sq = 0.0_f32;
        let mut vertical_velocity_sum = 0.0_f32;
        let mut vertical_dipole_sum = Vec3::ZERO;
        let mut j_sum = 0.0_f32;
        let mut min_pos = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_pos = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for i in start..end {
            let base = i * 8;
            let x = data[base];
            let y = data[base + 1];
            let z = data[base + 2];
            let j = data[base + 3];
            let vx = data[base + 4];
            let vy = data[base + 5];
            let vz = data[base + 6];
            let mass = data[base + 7];
            let solute_mass = affine_data[i * 12 + 7].max(0.0);

            diagnostics.all_finite &= x.is_finite()
                && y.is_finite()
                && z.is_finite()
                && j.is_finite()
                && vx.is_finite()
                && vy.is_finite()
                && vz.is_finite()
                && mass.is_finite()
                && solute_mass.is_finite();

            if mass <= inactive_thresh {
                continue;
            }

            let pos = Vec3::new(x, y, z);
            let vel = Vec3::new(vx, vy, vz);
            let speed_sq = vel.length_squared();
            let vertical_speed_sq = vy * vy;
            let lateral_speed_sq = vx * vx + vz * vz;
            let speed = speed_sq.sqrt();
            let mass_scale = mass / nominal_mass.max(1e-6);
            let rest_volume = mass_scale * particle_vol;
            let current_volume = rest_volume * j.clamp(0.40, 2.00);

            diagnostics.active_count += 1;
            diagnostics.active_mass += mass;
            diagnostics.dissolved_solute_mass += solute_mass;
            diagnostics.rest_volume_ml += rest_volume * units::ML_PER_SIM_UNIT_CUBED;
            diagnostics.current_volume_ml += current_volume * units::ML_PER_SIM_UNIT_CUBED;
            diagnostics.kinetic_energy += 0.5 * mass * speed_sq;
            diagnostics.momentum = diagnostics.momentum + vel * mass;
            diagnostics.max_speed = diagnostics.max_speed.max(speed);
            diagnostics.max_upward_speed = diagnostics.max_upward_speed.max(vy.max(0.0));
            diagnostics.max_downward_speed = diagnostics.max_downward_speed.max((-vy).max(0.0));
            diagnostics.upward_momentum += mass * vy.max(0.0);
            diagnostics.downward_momentum += mass * (-vy).max(0.0);
            centroid_sum = centroid_sum + pos * mass;
            mass_weighted_speed_sq += mass * speed_sq;
            mass_weighted_vertical_speed_sq += mass * vertical_speed_sq;
            mass_weighted_lateral_speed_sq += mass * lateral_speed_sq;
            vertical_velocity_sum += mass * vy;
            vertical_dipole_sum = vertical_dipole_sum + Vec3::new(x, 0.0, z) * (mass * vy);
            j_sum += j;

            min_pos.x = min_pos.x.min(x);
            min_pos.y = min_pos.y.min(y);
            min_pos.z = min_pos.z.min(z);
            max_pos.x = max_pos.x.max(x);
            max_pos.y = max_pos.y.max(y);
            max_pos.z = max_pos.z.max(z);

            if let Some((cup_radius, cup_top_y, cup_bot_y)) = cup {
                let r_sq = x * x + z * z;
                if r_sq <= (cup_radius + dx) * (cup_radius + dx)
                    && y <= cup_top_y + dx
                    && y >= cup_bot_y - dx
                {
                    diagnostics.pool_count += 1;
                    diagnostics.cup_water_mass += mass;
                    diagnostics.cup_solute_mass += solute_mass;
                    if r_sq <= cup_radius * cup_radius {
                        let bx =
                            (((x / cup_radius + 1.0) * 0.5) * SURFACE_BINS as f32).floor() as i32;
                        let bz =
                            (((z / cup_radius + 1.0) * 0.5) * SURFACE_BINS as f32).floor() as i32;
                        if bx >= 0
                            && bz >= 0
                            && (bx as usize) < SURFACE_BINS
                            && (bz as usize) < SURFACE_BINS
                        {
                            let bin = bx as usize + bz as usize * SURFACE_BINS;
                            surface_y[bin] = surface_y[bin].max(y);
                        }
                    }
                }
            }
        }

        diagnostics.active_mass_ml = diagnostics.active_mass / MASS_UNITS_PER_ML;
        if diagnostics.active_count == 0 || diagnostics.active_mass <= 0.0 {
            return diagnostics;
        }

        diagnostics.mean_tds =
            diagnostics.dissolved_solute_mass / diagnostics.active_mass.max(1e-6);
        diagnostics.cup_tds = diagnostics.cup_solute_mass / diagnostics.cup_water_mass.max(1e-6);
        diagnostics.extraction_yield = self
            .settings
            .bed
            .as_ref()
            .map(|_| {
                diagnostics.dissolved_solute_mass
                    / (DEFAULT_BREW.coffee_dose_g * MASS_UNITS_PER_ML).max(1e-6)
            })
            .unwrap_or(0.0);

        diagnostics.centroid = centroid_sum / diagnostics.active_mass;
        diagnostics.min = min_pos;
        diagnostics.max = max_pos;
        diagnostics.extent = max_pos - min_pos;
        diagnostics.mean_j = j_sum / diagnostics.active_count as f32;
        diagnostics.rms_speed = (mass_weighted_speed_sq / diagnostics.active_mass.max(1e-6)).sqrt();
        diagnostics.vertical_rms_speed =
            (mass_weighted_vertical_speed_sq / diagnostics.active_mass.max(1e-6)).sqrt();
        diagnostics.mean_vertical_speed = vertical_velocity_sum / diagnostics.active_mass.max(1e-6);
        diagnostics.lateral_rms_speed =
            (mass_weighted_lateral_speed_sq / diagnostics.active_mass.max(1e-6)).sqrt();
        if let Some((cup_radius, _, _)) = cup {
            diagnostics.vertical_dipole =
                vertical_dipole_sum / (diagnostics.active_mass.max(1e-6) * cup_radius.max(1e-6));
            diagnostics.vertical_dipole_magnitude = diagnostics.vertical_dipole.length();
        }
        diagnostics.momentum_magnitude = diagnostics.momentum.length();

        let mut surface_sum = 0.0_f32;
        let mut surface_x_sum = 0.0_f32;
        let mut surface_z_sum = 0.0_f32;
        let mut surface_min = f32::MAX;
        let mut surface_max = f32::MIN;
        for (bin, y) in surface_y.iter().copied().enumerate() {
            if !y.is_finite() {
                continue;
            }
            let bx = bin % SURFACE_BINS;
            let bz = bin / SURFACE_BINS;
            let Some((cup_radius, _, _)) = cup else {
                continue;
            };
            let x = (((bx as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
            let z = (((bz as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
            diagnostics.surface_bin_count += 1;
            surface_sum += y;
            surface_x_sum += x;
            surface_z_sum += z;
            surface_min = surface_min.min(y);
            surface_max = surface_max.max(y);
        }
        diagnostics.surface_possible_bins = surface_possible_bins;
        if diagnostics.surface_bin_count > 0 {
            diagnostics.surface_mean_y = surface_sum / diagnostics.surface_bin_count as f32;
            let surface_mean_x = surface_x_sum / diagnostics.surface_bin_count as f32;
            let surface_mean_z = surface_z_sum / diagnostics.surface_bin_count as f32;
            let mut x_variance = 0.0_f32;
            let mut z_variance = 0.0_f32;
            let mut xy_covariance = 0.0_f32;
            let mut zy_covariance = 0.0_f32;
            let mut variance_sum = 0.0_f32;
            for (bin, y) in surface_y.iter().copied().enumerate() {
                if !y.is_finite() {
                    continue;
                }
                let bx = bin % SURFACE_BINS;
                let bz = bin / SURFACE_BINS;
                let Some((cup_radius, _, _)) = cup else {
                    continue;
                };
                let x = (((bx as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
                let z = (((bz as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
                let dx = x - surface_mean_x;
                let dz = z - surface_mean_z;
                let dy = y - diagnostics.surface_mean_y;
                x_variance += dx * dx;
                z_variance += dz * dz;
                xy_covariance += dx * dy;
                zy_covariance += dz * dy;
                variance_sum += dy * dy;
            }
            diagnostics.surface_rms_y =
                (variance_sum / diagnostics.surface_bin_count as f32).sqrt();
            diagnostics.surface_min_y = surface_min;
            diagnostics.surface_max_y = surface_max;
            diagnostics.surface_peak_to_peak_y = surface_max - surface_min;
            diagnostics.surface_tilt = Vec3::new(
                xy_covariance / x_variance.max(1e-6),
                0.0,
                zy_covariance / z_variance.max(1e-6),
            );
            diagnostics.surface_tilt_magnitude = diagnostics.surface_tilt.length();
            if let Some((cup_radius, _, _)) = cup {
                diagnostics.surface_tilt_height_y =
                    diagnostics.surface_tilt_magnitude * cup_radius * 2.0;
            }
            let mut residual_sum = 0.0_f32;
            let mut residual_min = f32::MAX;
            let mut residual_max = f32::MIN;
            for (bin, y) in surface_y.iter().copied().enumerate() {
                if !y.is_finite() {
                    continue;
                }
                let bx = bin % SURFACE_BINS;
                let bz = bin / SURFACE_BINS;
                let Some((cup_radius, _, _)) = cup else {
                    continue;
                };
                let x = (((bx as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
                let z = (((bz as f32) + 0.5) / SURFACE_BINS as f32 * 2.0 - 1.0) * cup_radius;
                let predicted_y = diagnostics.surface_mean_y
                    + diagnostics.surface_tilt.x * (x - surface_mean_x)
                    + diagnostics.surface_tilt.z * (z - surface_mean_z);
                let residual = y - predicted_y;
                residual_sum += residual * residual;
                residual_min = residual_min.min(residual);
                residual_max = residual_max.max(residual);
            }
            diagnostics.surface_residual_rms_y =
                (residual_sum / diagnostics.surface_bin_count as f32).sqrt();
            diagnostics.surface_residual_peak_to_peak_y = residual_max - residual_min;
        }

        let surface_y_for_pressure = if diagnostics.surface_bin_count > 0 {
            diagnostics.surface_mean_y
        } else {
            diagnostics.max.y
        };
        self.update_hydrostatic_diagnostics(
            data,
            start,
            end,
            inactive_thresh,
            surface_y_for_pressure,
            &mut diagnostics,
        );

        diagnostics
    }

    fn update_hydrostatic_diagnostics(
        &self,
        data: &[f32],
        start: usize,
        end: usize,
        inactive_thresh: f32,
        surface_y: f32,
        diagnostics: &mut WaterDiagnostics,
    ) {
        const PRESSURE_BANDS: usize = 6;
        const WATER_DENSITY_KG_M3: f32 = 1_000.0;

        if diagnostics.active_count < 2 || !surface_y.is_finite() {
            return;
        }

        let height = diagnostics.extent.y;
        if height <= 1e-4 {
            return;
        }

        let mut band_counts = [0_u32; PRESSURE_BANDS];
        let mut band_y_sums = [0.0_f32; PRESSURE_BANDS];
        for i in start..end {
            let base = i * 8;
            let y = data[base + 1];
            let mass = data[base + 7];
            if mass <= inactive_thresh || !y.is_finite() {
                continue;
            }

            let t = ((y - diagnostics.min.y) / height).clamp(0.0, 0.999_999);
            let band = (t * PRESSURE_BANDS as f32) as usize;
            band_counts[band] += 1;
            band_y_sums[band] += y;
        }

        let bottom_band = band_counts.iter().position(|count| *count > 0);
        let top_band = band_counts.iter().rposition(|count| *count > 0);
        let (Some(bottom_band), Some(top_band)) = (bottom_band, top_band) else {
            return;
        };
        if bottom_band == top_band {
            return;
        }

        let bottom_y = band_y_sums[bottom_band] / band_counts[bottom_band] as f32;
        let top_y = band_y_sums[top_band] / band_counts[top_band] as f32;
        let pressure_surface_y = surface_y.max(top_y);
        let pressure_from_head = |y: f32| {
            WATER_DENSITY_KG_M3
                * units::STANDARD_GRAVITY_M_S2
                * ((pressure_surface_y - y).max(0.0) * units::METERS_PER_SIM_UNIT)
        };
        let top_pressure = pressure_from_head(top_y);
        let bottom_pressure = pressure_from_head(bottom_y);
        let depth_m = ((top_y - bottom_y).max(0.0)) * units::METERS_PER_SIM_UNIT;
        let delta = bottom_pressure - top_pressure;

        diagnostics.hydrostatic_sample_count = band_counts[bottom_band] + band_counts[top_band];
        diagnostics.hydrostatic_depth_m = depth_m;
        diagnostics.hydrostatic_top_pressure_pa = top_pressure;
        diagnostics.hydrostatic_bottom_pressure_pa = bottom_pressure;
        diagnostics.hydrostatic_delta_pressure_pa = delta;
        diagnostics.hydrostatic_gradient_pa_per_m =
            if depth_m > 1e-6 { delta / depth_m } else { 0.0 };
        diagnostics.hydrostatic_bottom_higher = delta > 1.0;
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
            let pressure_pairs = self.pressure_rbgs_pairs_for_substep();
            self.last_pressure_rbgs_pairs = pressure_pairs;

            // Dispatch compute passes
            let total_cells = self.settings.grid_dims[0]
                * self.settings.grid_dims[1]
                * self.settings.grid_dims[2];
            let num_particles = self.num_water + self.num_bed;
            let dispatch = MpmDispatchSizes {
                cell_wg: dispatch_size(total_cells, NUM_THREADS),
                particle_wg: dispatch_size(num_particles, NUM_THREADS),
                bed_wg: dispatch_size(self.num_bed, NUM_THREADS),
                metrics_wg: dispatch_size(METRICS_SLOT_COUNT as u32, 8),
            };
            let pressure_ctx = PressureContext {
                kind: self.settings.pressure_solver,
                cell_wg: dispatch.cell_wg,
                rbgs_pairs: pressure_pairs,
                cg_iterations: self.settings.pressure_cg_iterations,
            };

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mpm step"),
            });
            encoder.clear_buffer(&self.buffers.grid, 0, None);
            encoder.clear_buffer(&self.buffers.grid_vel, 0, None);
            let mut ops = Vec::new();
            encode_mpm_substep_schedule(
                &self.pipelines,
                &self.buffers,
                dispatch,
                pressure_ctx,
                |op| {
                    ops.push(op);
                },
            );
            encode_mpm_schedule_production(&mut encoder, &self.pipelines.bind_group, &ops);
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
        self.last_pressure_rbgs_pairs = 0;
        self.latest_metrics = MetricsSnapshot::default();
        self.inflow = InflowState::new(units::sim_speed_from_meters_per_second(
            self.settings.initial_water_speed_m_s,
        ));
        self.inflow.update(&self.settings.spout);
        // Rebuild the CPU filter mesh so reset/scene changes keep render
        // geometry aligned with the active filter config.
        self.filter_mesh = self.settings.filter.as_ref().map(FilterMesh::new);
        queue.write_buffer(
            &self.buffers.cg,
            0,
            &vec![0u8; self.buffers.cg.size() as usize],
        );
        self.init_bed(queue);
    }

    pub fn set_exit_speed_m_s(&mut self, speed_m_s: f32) {
        self.inflow
            .set_exit_speed(units::sim_speed_from_meters_per_second(speed_m_s));
        self.inflow.update(&self.settings.spout);
    }

    pub fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.settings.spout.translate_origin_to(Vec3::new(x, y, z));
        self.inflow.update(&self.settings.spout);
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

    pub fn static_filter_mesh_key(&self) -> Option<u64> {
        self.settings.filter.as_ref().map(filter_config_key)
    }

    pub fn settings(&self) -> &MpmSettings {
        &self.settings
    }

    pub fn set_pressure_residual_adaptation(&mut self, target: f32, max_pairs: u32) {
        self.settings.pressure_residual_target = target.max(0.0);
        self.settings.pressure_rbgs_max_pairs = max_pairs.max(self.settings.pressure_rbgs_pairs);
    }

    fn pressure_rbgs_pairs_for_substep(&self) -> u32 {
        let base = self.settings.pressure_rbgs_pairs.max(1);
        let target = self.settings.pressure_residual_target;
        if target <= 0.0 || self.latest_metrics.projection_residual_cells == 0 {
            return base;
        }

        let max_pairs = self.settings.pressure_rbgs_max_pairs.max(base);
        if max_pairs <= base || self.latest_metrics.projection_residual_mean_abs_div <= target {
            return base;
        }

        let overshoot =
            (self.latest_metrics.projection_residual_mean_abs_div / target).clamp(1.0, 4.0);
        let t = (overshoot - 1.0) / 3.0;
        base + ((max_pairs - base) as f32 * t).ceil() as u32
    }

    pub fn last_pressure_rbgs_pairs(&self) -> u32 {
        self.last_pressure_rbgs_pairs
    }

    #[cfg(test)]
    pub(crate) fn pressure_solver_kind(&self) -> PressureSolverKind {
        self.settings.pressure_solver
    }

    #[cfg(test)]
    pub(crate) fn pressure_operator_kind(&self) -> PressureOperatorKind {
        self.settings.pressure_operator
    }

    #[cfg(test)]
    pub(crate) fn pressure_solver_iterations_per_substep(&self) -> u32 {
        self.pipelines
            .pressure
            .iterations_per_substep(PressureContext {
                kind: self.settings.pressure_solver,
                cell_wg: 0,
                rbgs_pairs: self
                    .last_pressure_rbgs_pairs
                    .max(self.settings.pressure_rbgs_pairs),
                cg_iterations: self.settings.pressure_cg_iterations,
            })
    }

    #[cfg(test)]
    pub(crate) fn profiler_timestamp_query_capacity(&self) -> u32 {
        let pressure_ctx = PressureContext {
            kind: self.settings.pressure_solver,
            cell_wg: 0,
            rbgs_pairs: self
                .last_pressure_rbgs_pairs
                .max(self.settings.pressure_rbgs_pairs),
            cg_iterations: self.settings.pressure_cg_iterations,
        };
        let scopes = COMMON_TIMED_SCOPES_PER_SUBSTEP
            + self
                .pipelines
                .pressure
                .estimated_timestamp_scopes(pressure_ctx);
        ((scopes * 2) + 8).max(MIN_TIMESTAMP_QUERY_CAPACITY)
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

    pub fn estimated_extraction_yield(&self) -> f32 {
        if self.settings.bed.is_none() {
            return 0.0;
        }
        let wetting = (self.total_emitted_ml() / DEFAULT_BREW.target_bed_retention_ml.max(1e-6))
            .clamp(0.0, 1.0);
        let wet_contact = wetting * wetting * (3.0 - 2.0 * wetting);
        let effective_time = self.total_time * wet_contact;
        let fast = DEFAULT_BREW.fast_extractable_fraction
            * (1.0 - (-DEFAULT_BREW.fast_extraction_rate_s * effective_time).exp());
        let slow = (1.0 - DEFAULT_BREW.fast_extractable_fraction)
            * (1.0 - (-DEFAULT_BREW.slow_extraction_rate_s * effective_time).exp());
        DEFAULT_BREW.extractable_yield_fraction * (fast + slow)
    }

    pub fn estimated_cup_tds(&self) -> f32 {
        if self.settings.bed.is_none() || self.total_emitted_mass <= 1e-6 {
            return 0.0;
        }
        let extracted_mass =
            self.estimated_extraction_yield() * DEFAULT_BREW.coffee_dose_g * MASS_UNITS_PER_ML;
        extracted_mass / self.total_emitted_mass.max(1e-6)
    }

    /// Last cached metrics snapshot. Populated by `refresh_metrics`; returns
    /// the zero default until the first successful readback.
    pub fn latest_metrics(&self) -> MetricsSnapshot {
        self.latest_metrics
    }

    #[cfg(target_arch = "wasm32")]
    pub fn metrics_buffer(&self) -> wgpu::Buffer {
        self.buffers.metrics.clone()
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn sample_metrics_after_delay(
        device: wgpu::Device,
        queue: wgpu::Queue,
        metrics: wgpu::Buffer,
        has_bed: bool,
        delay_frames: u32,
    ) -> Result<MetricsSnapshot, JsValue> {
        let metrics_size = (METRICS_SLOT_COUNT * std::mem::size_of::<u32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm metrics delayed staging"),
            size: metrics_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm metrics delayed readback"),
        });
        encoder.copy_buffer_to_buffer(&metrics, 0, &staging, 0, metrics_size);
        queue.submit(Some(encoder.finish()));

        wait_animation_frames(delay_frames).await?;

        let slice = staging.slice(..);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            slice.map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => {
                    let _ = resolve.call0(&JsValue::NULL);
                }
                Err(err) => {
                    let _ = reject.call1(
                        &JsValue::NULL,
                        &JsValue::from_str(&format!("metrics map failed: {err:?}")),
                    );
                }
            });
        });
        wasm_bindgen_futures::JsFuture::from(promise).await?;

        let view = slice.get_mapped_range();
        let raw = bytemuck::cast_slice::<u8, u32>(&view);
        let snapshot = metrics_snapshot_from_raw(raw, has_bed)
            .ok_or_else(|| JsValue::from_str("metrics readback returned too few slots"))?;
        drop(view);
        staging.unmap();
        Ok(snapshot)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn water_diagnostics(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<WaterDiagnostics, JsValue> {
        let particle_count = (self.num_water + self.num_bed) as usize;
        let particle_size = (particle_count * 32).max(4) as u64;
        let affine_size = (particle_count * 48).max(4) as u64;
        let particle_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water diagnostics particle staging"),
            size: particle_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let affine_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water diagnostics affine staging"),
            size: affine_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("water diagnostics readback"),
        });
        encoder.copy_buffer_to_buffer(
            &self.buffers.particles,
            0,
            &particle_staging,
            0,
            particle_size,
        );
        encoder.copy_buffer_to_buffer(&self.buffers.affine, 0, &affine_staging, 0, affine_size);
        queue.submit(Some(encoder.finish()));

        let queue_done = js_sys::Promise::new(&mut |resolve, _reject| {
            queue.on_submitted_work_done(move || {
                let _ = resolve.call0(&JsValue::NULL);
            });
        });
        wasm_bindgen_futures::JsFuture::from(queue_done).await?;

        let particle_slice = particle_staging.slice(..);
        let affine_slice = affine_staging.slice(..);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            particle_slice.map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => {
                    let _ = resolve.call0(&JsValue::NULL);
                }
                Err(err) => {
                    let _ = reject.call1(
                        &JsValue::NULL,
                        &JsValue::from_str(&format!("water diagnostics map failed: {err:?}")),
                    );
                }
            });
        });
        wasm_bindgen_futures::JsFuture::from(promise).await?;
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            affine_slice.map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => {
                    let _ = resolve.call0(&JsValue::NULL);
                }
                Err(err) => {
                    let _ = reject.call1(
                        &JsValue::NULL,
                        &JsValue::from_str(&format!(
                            "water diagnostics affine map failed: {err:?}"
                        )),
                    );
                }
            });
        });
        wasm_bindgen_futures::JsFuture::from(promise).await?;

        let particle_view = particle_slice.get_mapped_range();
        let particle_data = bytemuck::cast_slice::<u8, f32>(&particle_view);
        let affine_view = affine_slice.get_mapped_range();
        let affine_data = bytemuck::cast_slice::<u8, f32>(&affine_view);
        let diagnostics = self.water_diagnostics_from_particle_data(particle_data, affine_data);
        drop(affine_view);
        affine_staging.unmap();
        drop(particle_view);
        particle_staging.unmap();
        Ok(diagnostics)
    }

    /// Async staging-buffer readback for the GPU metrics counters.
    ///
    #[cfg(target_arch = "wasm32")]
    pub async fn refresh_metrics(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), JsValue> {
        let metrics_size = (METRICS_SLOT_COUNT * std::mem::size_of::<u32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm metrics one-shot staging"),
            size: metrics_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mpm metrics readback"),
        });
        encoder.copy_buffer_to_buffer(&self.buffers.metrics, 0, &staging, 0, metrics_size);
        queue.submit(Some(encoder.finish()));

        let queue_done = js_sys::Promise::new(&mut |resolve, _reject| {
            queue.on_submitted_work_done(move || {
                let _ = resolve.call0(&JsValue::NULL);
            });
        });
        wasm_bindgen_futures::JsFuture::from(queue_done).await?;

        let slice = staging.slice(..);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            slice.map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => {
                    let _ = resolve.call0(&JsValue::NULL);
                }
                Err(err) => {
                    let _ = reject.call1(
                        &JsValue::NULL,
                        &JsValue::from_str(&format!("metrics map failed: {err:?}")),
                    );
                }
            });
        });
        wasm_bindgen_futures::JsFuture::from(promise).await?;

        let view = slice.get_mapped_range();
        let raw = bytemuck::cast_slice::<u8, u32>(&view);
        if let Some(snapshot) = metrics_snapshot_from_raw(raw, self.settings.bed.is_some()) {
            self.latest_metrics = snapshot;
        }
        drop(view);
        staging.unmap();
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
            sdf_params: [SDF_RES as f32, 0.3, 0.0, CONTACT_OFFSET],
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
            solute_params: [
                DEFAULT_BREW.fast_extraction_rate_s,
                DEFAULT_BREW.slow_extraction_rate_s,
                DEFAULT_BREW.max_solute_concentration,
                DEFAULT_BREW.pore_to_water_mass_transfer_rate_s,
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
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn contact_offset() -> f32"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("const OBSTACLE_WALL_THICKNESS: f32 = 0.4;"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn dripper_top_radius()"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn resolve_conical_barrier("));
        assert!(shader::MPM_COMPUTE_SHADER
            .contains("Use analytic obstacles for live contact and pressure classification"));
        assert!(shader::MPM_COMPUTE_SHADER
            .contains("return sample_sdf(cell_center_from_cell(cell)) < 0.0;"));
        assert!(shader::MPM_COMPUTE_SHADER
            .contains("(dripper_top_radius() - dripper_outlet_radius()) / cone_height"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn viscosity_prepare("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn viscosity_apply("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("|| kind == CELL_SURFACE_FLUID"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("@group(0) @binding(12)"));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn pressure_cg_init("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn pressure_cg_warmstart("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn pressure_cg_matvec("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn pressure_cg_apply_alpha("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn pressure_cg_update_dir("));
        assert!(shader::MPM_COMPUTE_SHADER.contains("fn pressure_cg_finish_iteration("));
    }

    #[test]
    fn mpm_compute_shader_parses() {
        let module = naga::front::wgsl::parse_str(shader::MPM_COMPUTE_SHADER)
            .expect("MPM WGSL should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("MPM WGSL should validate");
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
    fn default_v60_uses_gentle_vertical_water_speed() {
        let s = MpmSettings::default_v60();
        assert!(s.initial_water_speed_m_s <= 0.13);
        assert!(s.spout.max_flow_rate_ml_s <= 12.0);
        assert!(s.spout.direction.x.abs() < 1e-6);
        assert!((s.spout.direction.y + 1.0).abs() < 1e-6);
        assert!(s.spout.direction.z.abs() < 1e-6);

        let mut inflow = InflowState::new(units::sim_speed_from_meters_per_second(
            s.initial_water_speed_m_s,
        ));
        inflow.update(&s.spout);
        assert!(inflow.exit_speed() * units::METERS_PER_SIM_UNIT <= 0.13);
        assert!(s.spout.max_exit_speed * units::METERS_PER_SIM_UNIT <= 0.50);
    }

    #[test]
    fn filter_water_block_scene_keeps_filter_bed_and_disables_inflow() {
        let s = MpmSettings::benchmark_filter_water_block();
        assert!(s.filter.is_some());
        assert!(s.bed.is_some());
        assert_eq!(s.initial_water_speed_m_s, 0.0);
        assert!(s.pressure_rbgs_pairs >= MpmSettings::default_v60().pressure_rbgs_pairs);
    }

    #[test]
    fn debug_scene_catalog_ids_round_trip() {
        let mut ids = Vec::new();
        for scene in DebugScene::ALL {
            let id = scene.id();
            assert_eq!(DebugScene::from_id(id), Some(scene));
            assert!(!ids.contains(&id), "duplicate debug scene id: {id}");
            ids.push(id);
        }
        assert_eq!(ids.len(), 13);
    }

    #[test]
    fn debug_scene_catalog_stays_in_sync_with_browser_ui() {
        let html = include_str!("../../www-3d/index.html");
        let js = include_str!("../../www-3d/main.js");

        for scene in DebugScene::ALL {
            let id = scene.id();
            assert!(
                html.contains(&format!("data-debug-scene=\"{id}\"")),
                "missing debug scene button for {id}",
            );
            assert!(
                js.contains(&format!("[\"{id}\",")),
                "missing DEBUG_SCENE_LABELS entry for {id}",
            );
        }

        assert_eq!(
            html.matches("data-debug-scene=").count(),
            DebugScene::ALL.len(),
            "HTML should expose exactly the Rust debug-scene catalog",
        );
    }

    #[test]
    fn debug_scene_settings_match_expected_test_geometry() {
        for scene in DebugScene::ALL {
            let settings = scene.settings();
            assert!(settings.grid_dims.iter().all(|dim| *dim > 0));
            assert!(settings.max_particles > 0);

            let bed_budget = settings
                .bed
                .as_ref()
                .map(|bed| bed.num_particles)
                .unwrap_or(0);
            assert!(
                settings.max_particles > bed_budget + 2_000,
                "{} leaves too little water capacity",
                scene.id(),
            );

            match scene {
                DebugScene::FilterWaterBlock
                | DebugScene::OffCenterFilterWallPour
                | DebugScene::SeededPaperWallSheet
                | DebugScene::UniformBedSaturation
                | DebugScene::PermeabilityComparison
                | DebugScene::ParticleCapacityStress => {
                    assert!(
                        settings.filter.is_some(),
                        "{} needs filter geometry",
                        scene.id()
                    );
                    assert!(settings.bed.is_some(), "{} needs bed geometry", scene.id());
                }
                DebugScene::FilterApexDrain => {
                    assert!(settings.filter.is_some());
                    assert!(settings.bed.is_none());
                }
                DebugScene::CupWallFloorCornerContact
                | DebugScene::AsymmetricCupMoundSettle
                | DebugScene::HydrostaticColumn
                | DebugScene::DamBreakSlosh
                | DebugScene::SparseFreeJet
                | DebugScene::HighVelocityJetImpact => {
                    assert!(
                        settings.filter.is_none(),
                        "{} should isolate cup water",
                        scene.id()
                    );
                    assert!(
                        settings.bed.is_none(),
                        "{} should isolate cup water",
                        scene.id()
                    );
                    assert!(
                        cup_region(&settings).is_some(),
                        "{} needs a cup",
                        scene.id()
                    );
                }
            }
        }
    }

    #[test]
    fn debug_scene_inflow_defaults_match_scene_intent() {
        for scene in [
            DebugScene::FilterWaterBlock,
            DebugScene::SeededPaperWallSheet,
            DebugScene::FilterApexDrain,
            DebugScene::CupWallFloorCornerContact,
            DebugScene::AsymmetricCupMoundSettle,
            DebugScene::HydrostaticColumn,
            DebugScene::DamBreakSlosh,
            DebugScene::UniformBedSaturation,
        ] {
            assert_eq!(
                scene.settings().initial_water_speed_m_s,
                0.0,
                "{}",
                scene.id()
            );
        }

        assert!(
            DebugScene::OffCenterFilterWallPour
                .settings()
                .initial_water_speed_m_s
                > MpmSettings::default_v60().initial_water_speed_m_s
        );
        assert!(DebugScene::SparseFreeJet.settings().initial_water_speed_m_s > 0.0);
        assert!(
            DebugScene::HighVelocityJetImpact
                .settings()
                .initial_water_speed_m_s
                > MpmSettings::default_v60().initial_water_speed_m_s
        );
        assert_eq!(
            DebugScene::ParticleCapacityStress.settings().max_particles,
            32_000
        );

        let default_perm = MpmSettings::default_v60()
            .bed
            .as_ref()
            .expect("default bed")
            .initial_permeability;
        let comparison_perm = DebugScene::PermeabilityComparison
            .settings()
            .bed
            .as_ref()
            .expect("comparison bed")
            .initial_permeability;
        assert!(comparison_perm < default_perm);
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

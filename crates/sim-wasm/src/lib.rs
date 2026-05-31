#![cfg_attr(test, allow(dead_code))]
#![allow(dead_code, unused_imports, unused_variables, unused_assignments)]

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod app_sim;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod boundaries;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod diagnostics;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod engine;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod extraction;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod materials;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod render_contract;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod render_pack;
#[cfg(target_arch = "wasm32")]
mod renderer;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod scene;

#[cfg(target_arch = "wasm32")]
use app_sim::AppSim;
#[cfg(target_arch = "wasm32")]
use diagnostics::{MetricsSnapshot, WaterDiagnostics};
#[cfg(target_arch = "wasm32")]
use renderer::{OrbitCamera, Renderer};
#[cfg(target_arch = "wasm32")]
use scene::{DebugScene, SceneSpec, METERS_PER_SIM_UNIT};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

#[cfg(target_arch = "wasm32")]
fn set_number(obj: &js_sys::Object, key: &str, value: f32) -> Result<(), JsValue> {
    js_sys::Reflect::set(
        obj,
        &JsValue::from_str(key),
        &JsValue::from_f64(value as f64),
    )
    .map(|_| ())
}

#[cfg(target_arch = "wasm32")]
fn set_u32(obj: &js_sys::Object, key: &str, value: u32) -> Result<(), JsValue> {
    js_sys::Reflect::set(
        obj,
        &JsValue::from_str(key),
        &JsValue::from_f64(value as f64),
    )
    .map(|_| ())
}

#[cfg(target_arch = "wasm32")]
fn set_bool(obj: &js_sys::Object, key: &str, value: bool) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_bool(value)).map(|_| ())
}

#[cfg(target_arch = "wasm32")]
fn set_object(obj: &js_sys::Object, key: &str, value: &js_sys::Object) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), value).map(|_| ())
}

#[cfg(target_arch = "wasm32")]
fn vec3_object(v: coffee_sim_core::Vec3) -> Result<js_sys::Object, JsValue> {
    let obj = js_sys::Object::new();
    set_number(&obj, "x", v.x)?;
    set_number(&obj, "y", v.y)?;
    set_number(&obj, "z", v.z)?;
    set_number(&obj, "xMeters", v.x * METERS_PER_SIM_UNIT)?;
    set_number(&obj, "yMeters", v.y * METERS_PER_SIM_UNIT)?;
    set_number(&obj, "zMeters", v.z * METERS_PER_SIM_UNIT)?;
    Ok(obj)
}

#[cfg(target_arch = "wasm32")]
fn water_diagnostics_object(d: WaterDiagnostics) -> Result<js_sys::Object, JsValue> {
    let obj = js_sys::Object::new();
    set_bool(&obj, "allFinite", d.all_finite)?;
    set_number(&obj, "simTimeSeconds", d.sim_time_s)?;
    set_u32(&obj, "activeCount", d.active_count)?;
    set_u32(&obj, "poolCount", d.pool_count)?;
    set_number(&obj, "activeMass", d.active_mass)?;
    set_number(&obj, "activeMassMl", d.active_mass_ml)?;
    set_number(&obj, "emittedMl", d.emitted_ml)?;
    set_number(&obj, "restVolumeMl", d.rest_volume_ml)?;
    set_number(&obj, "currentVolumeMl", d.current_volume_ml)?;
    set_number(&obj, "meanJ", d.mean_j)?;
    set_number(&obj, "kineticEnergy", d.kinetic_energy)?;
    set_number(
        &obj,
        "rmsSpeedMetersPerSecond",
        d.rms_speed * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "verticalRmsSpeedMetersPerSecond",
        d.vertical_rms_speed * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "meanVerticalSpeedMetersPerSecond",
        d.mean_vertical_speed * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "lateralRmsSpeedMetersPerSecond",
        d.lateral_rms_speed * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "maxSpeedMetersPerSecond",
        d.max_speed * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "maxUpwardSpeedMetersPerSecond",
        d.max_upward_speed * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "maxDownwardSpeedMetersPerSecond",
        d.max_downward_speed * METERS_PER_SIM_UNIT,
    )?;
    set_number(&obj, "upwardMomentum", d.upward_momentum)?;
    set_number(&obj, "downwardMomentum", d.downward_momentum)?;
    set_number(
        &obj,
        "verticalDipoleMetersPerSecond",
        d.vertical_dipole_magnitude * METERS_PER_SIM_UNIT,
    )?;
    set_object(&obj, "verticalDipole", &vec3_object(d.vertical_dipole)?)?;
    set_number(&obj, "momentumMagnitude", d.momentum_magnitude)?;
    set_object(&obj, "momentum", &vec3_object(d.momentum)?)?;
    set_object(&obj, "centroid", &vec3_object(d.centroid)?)?;
    set_object(&obj, "min", &vec3_object(d.min)?)?;
    set_object(&obj, "max", &vec3_object(d.max)?)?;
    set_object(&obj, "extent", &vec3_object(d.extent)?)?;

    let surface = js_sys::Object::new();
    set_u32(&surface, "binCount", d.surface_bin_count)?;
    set_u32(&surface, "possibleBins", d.surface_possible_bins)?;
    set_number(&surface, "meanY", d.surface_mean_y)?;
    set_number(&surface, "rmsY", d.surface_rms_y)?;
    set_number(&surface, "minY", d.surface_min_y)?;
    set_number(&surface, "maxY", d.surface_max_y)?;
    set_number(&surface, "peakToPeakY", d.surface_peak_to_peak_y)?;
    set_number(&surface, "rmsMeters", d.surface_rms_y * METERS_PER_SIM_UNIT)?;
    set_number(
        &surface,
        "peakToPeakMeters",
        d.surface_peak_to_peak_y * METERS_PER_SIM_UNIT,
    )?;
    set_number(&surface, "tiltX", d.surface_tilt.x)?;
    set_number(&surface, "tiltZ", d.surface_tilt.z)?;
    set_number(&surface, "tilt", d.surface_tilt_magnitude)?;
    set_number(
        &surface,
        "tiltHeightMeters",
        d.surface_tilt_height_y * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &surface,
        "residualRmsMeters",
        d.surface_residual_rms_y * METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &surface,
        "residualPeakToPeakMeters",
        d.surface_residual_peak_to_peak_y * METERS_PER_SIM_UNIT,
    )?;
    let coverage = if d.surface_possible_bins > 0 {
        d.surface_bin_count as f32 / d.surface_possible_bins as f32
    } else {
        0.0
    };
    set_number(&surface, "coverage", coverage)?;
    set_object(&obj, "surface", &surface)?;

    let pressure = js_sys::Object::new();
    set_u32(&pressure, "sampleCount", d.hydrostatic_sample_count)?;
    set_number(&pressure, "depthMeters", d.hydrostatic_depth_m)?;
    set_number(&pressure, "topPressurePa", d.hydrostatic_top_pressure_pa)?;
    set_number(
        &pressure,
        "bottomPressurePa",
        d.hydrostatic_bottom_pressure_pa,
    )?;
    set_number(
        &pressure,
        "deltaPressurePa",
        d.hydrostatic_delta_pressure_pa,
    )?;
    set_number(
        &pressure,
        "gradientPaPerMeter",
        d.hydrostatic_gradient_pa_per_m,
    )?;
    set_bool(&pressure, "bottomHigher", d.hydrostatic_bottom_higher)?;
    set_object(&obj, "hydrostaticPressure", &pressure)?;
    set_number(&obj, "dissolvedSoluteMass", d.dissolved_solute_mass)?;
    set_number(&obj, "meanTds", d.mean_tds)?;
    set_number(&obj, "cupWaterMass", d.cup_water_mass)?;
    set_number(&obj, "cupSoluteMass", d.cup_solute_mass)?;
    set_number(&obj, "cupTds", d.cup_tds)?;
    set_number(&obj, "extractionYield", d.extraction_yield)?;
    Ok(obj)
}

#[cfg(target_arch = "wasm32")]
fn metrics_snapshot_object(snapshot: MetricsSnapshot) -> Result<js_sys::Object, JsValue> {
    let obj = js_sys::Object::new();
    set_number(&obj, "maxAbsDivergence", snapshot.max_abs_div)?;
    set_u32(&obj, "fluidCellCount", snapshot.fluid_cells)?;
    set_u32(&obj, "divClampFires", snapshot.div_clamp_fires)?;
    set_u32(&obj, "pressureClampFires", snapshot.pressure_clamp_fires)?;
    set_u32(&obj, "massOverflowFires", snapshot.mass_overflow_fires)?;
    set_number(
        &obj,
        "projectionResidualMaxAbsDivergence",
        snapshot.projection_residual_max_abs_div,
    )?;
    set_number(
        &obj,
        "projectionResidualMeanAbsDivergence",
        snapshot.projection_residual_mean_abs_div,
    )?;
    set_u32(
        &obj,
        "projectionResidualCellCount",
        snapshot.projection_residual_cells,
    )?;
    set_number(&obj, "meanTds", snapshot.mean_tds)?;
    set_number(&obj, "cupTds", snapshot.cup_tds)?;
    set_number(&obj, "extractionYield", snapshot.extraction_yield)?;
    Ok(obj)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSim3D {
    sim: AppSim,
    renderer: Renderer,
    camera: OrbitCamera,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSim3D {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WasmSim3D, JsValue> {
        console_error_panic_hook::set_once();
        let settings = SceneSpec::DefaultV60.settings();
        let renderer = Renderer::new(canvas, &settings).await?;
        let sim = AppSim::new(renderer.device(), renderer.queue(), SceneSpec::DefaultV60);
        let camera = OrbitCamera::new(sim.settings().bounds_size);
        Ok(Self {
            sim,
            renderer,
            camera,
        })
    }

    pub fn reset(&mut self) {
        self.sim
            .reset(self.renderer.device(), self.renderer.queue());
        self.camera = OrbitCamera::new(self.sim.settings().bounds_size);
    }

    #[wasm_bindgen(js_name = loadDefaultScene)]
    pub fn load_default_scene(&mut self) {
        self.rebuild(SceneSpec::DefaultV60);
    }

    #[wasm_bindgen(js_name = loadBenchmarkFreeStream)]
    pub fn load_benchmark_free_stream(&mut self) {
        self.rebuild(SceneSpec::BenchmarkFreeStream);
    }

    #[wasm_bindgen(js_name = loadBenchmarkCenterPour)]
    pub fn load_benchmark_center_pour(&mut self) {
        self.rebuild(SceneSpec::BenchmarkCenterPour);
    }

    #[wasm_bindgen(js_name = loadBenchmarkFilterWaterBlock)]
    pub fn load_benchmark_filter_water_block(&mut self) {
        self.rebuild(SceneSpec::Debug(DebugScene::FilterWaterBlock));
    }

    #[wasm_bindgen(js_name = loadDebugScene)]
    pub fn load_debug_scene(&mut self, scene_id: &str) -> Result<(), JsValue> {
        let scene = DebugScene::from_id(scene_id)
            .ok_or_else(|| JsValue::from_str(&format!("unknown debug scene: {scene_id}")))?;
        self.rebuild(SceneSpec::Debug(scene));
        Ok(())
    }

    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, frame_time: f32) {
        let device = self.renderer.device().clone();
        let queue = self.renderer.queue().clone();
        self.sim.step_frame(&device, &queue, frame_time);
    }

    #[wasm_bindgen(js_name = setWaterVelocityMetersPerSecond)]
    pub fn set_water_velocity_m_s(&mut self, speed_m_s: f32) {
        self.sim.set_exit_speed_m_s(speed_m_s);
    }

    #[wasm_bindgen(js_name = setSpoutPosition)]
    pub fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.sim.set_spout_position(x, y, z);
    }

    #[wasm_bindgen(js_name = waterVelocityMetersPerSecond)]
    pub fn water_velocity_m_s(&self) -> f32 {
        self.sim.exit_speed_m_s()
    }

    #[wasm_bindgen(js_name = spoutX)]
    pub fn spout_x(&self) -> f32 {
        self.sim.spout_position().x
    }

    #[wasm_bindgen(js_name = spoutY)]
    pub fn spout_y(&self) -> f32 {
        self.sim.spout_position().y
    }

    #[wasm_bindgen(js_name = spoutZ)]
    pub fn spout_z(&self) -> f32 {
        self.sim.spout_position().z
    }

    #[wasm_bindgen(js_name = flowRate)]
    pub fn flow_rate(&self) -> f32 {
        self.sim.flow_rate_ml_s()
    }

    #[wasm_bindgen(js_name = exitSpeed)]
    pub fn exit_speed(&self) -> f32 {
        self.sim.exit_speed()
    }

    #[wasm_bindgen(js_name = exitSpeedMetersPerSecond)]
    pub fn exit_speed_m_s(&self) -> f32 {
        self.sim.exit_speed_m_s()
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        self.renderer.render_3d(&self.sim, self.camera)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(width, height);
    }

    #[wasm_bindgen(js_name = resizeWithCssSize)]
    pub fn resize_with_css_size(
        &mut self,
        width: u32,
        height: u32,
        css_width: f32,
        css_height: f32,
    ) {
        self.renderer
            .resize_with_css_size(width, height, css_width, css_height);
    }

    #[wasm_bindgen(js_name = orbitCamera)]
    pub fn orbit_camera(&mut self, delta_x: f32, delta_y: f32) {
        self.camera.orbit(delta_x, delta_y);
    }

    #[wasm_bindgen(js_name = zoomCamera)]
    pub fn zoom_camera(&mut self, delta: f32) {
        self.camera.zoom(delta, self.sim.settings().bounds_size);
    }

    #[wasm_bindgen(js_name = panCamera)]
    pub fn pan_camera(&mut self, right: f32, up: f32, forward: f32) {
        self.camera
            .pan(right, up, forward, self.sim.settings().bounds_size);
    }

    #[wasm_bindgen(js_name = cameraYaw)]
    pub fn camera_yaw(&self) -> f32 {
        self.camera.yaw
    }

    #[wasm_bindgen(js_name = cameraPitch)]
    pub fn camera_pitch(&self) -> f32 {
        self.camera.pitch
    }

    #[wasm_bindgen(js_name = particleCount)]
    pub fn particle_count(&self) -> usize {
        self.sim.particle_count()
    }

    #[wasm_bindgen(js_name = waterSlotsUsed)]
    pub fn water_slots_used(&self) -> u32 {
        self.sim.water_slots_used()
    }

    #[wasm_bindgen(js_name = bedParticleCount)]
    pub fn bed_particle_count(&self) -> u32 {
        self.sim.bed_particle_count()
    }

    #[wasm_bindgen(js_name = maxParticles)]
    pub fn max_particles(&self) -> u32 {
        self.sim.max_particles()
    }

    #[wasm_bindgen(js_name = simTime)]
    pub fn sim_time(&self) -> f32 {
        self.sim.total_time()
    }

    #[wasm_bindgen(js_name = frameEmittedMass)]
    pub fn frame_emitted_mass(&self) -> f32 {
        self.sim.frame_emitted_mass()
    }

    #[wasm_bindgen(js_name = frameEmittedMl)]
    pub fn frame_emitted_ml(&self) -> f32 {
        self.sim.frame_emitted_mass() / materials::water::MASS_UNITS_PER_ML
    }

    #[wasm_bindgen(js_name = totalEmittedMass)]
    pub fn total_emitted_mass(&self) -> f32 {
        self.sim.total_emitted_mass()
    }

    #[wasm_bindgen(js_name = totalEmittedMl)]
    pub fn total_emitted_ml(&self) -> f32 {
        self.sim.total_emitted_mass() / materials::water::MASS_UNITS_PER_ML
    }

    #[wasm_bindgen(js_name = frameDroppedParticles)]
    pub fn frame_dropped_particles(&self) -> u32 {
        self.sim.frame_dropped_particles()
    }

    #[wasm_bindgen(js_name = totalDroppedParticles)]
    pub fn total_dropped_particles(&self) -> u32 {
        self.sim.total_dropped_particles()
    }

    #[wasm_bindgen(js_name = hasBed)]
    pub fn has_bed(&self) -> bool {
        self.sim.settings().bed.is_some()
    }

    #[wasm_bindgen(js_name = refreshMetrics)]
    pub async fn refresh_metrics(&mut self) -> Result<(), JsValue> {
        Ok(())
    }

    #[wasm_bindgen(js_name = sampleMetrics)]
    pub fn sample_metrics(&self, _delay_frames: u32) -> js_sys::Promise {
        let snapshot = self.sim.latest_metrics();
        wasm_bindgen_futures::future_to_promise(async move {
            Ok(metrics_snapshot_object(snapshot)?.into())
        })
    }

    #[wasm_bindgen(js_name = waterDiagnostics)]
    pub async fn water_diagnostics(&self) -> Result<JsValue, JsValue> {
        Ok(water_diagnostics_object(self.sim.water_diagnostics())?.into())
    }

    #[wasm_bindgen(js_name = maxAbsDivergence)]
    pub fn max_abs_divergence(&self) -> f32 {
        self.sim.latest_metrics().max_abs_div
    }

    #[wasm_bindgen(js_name = fluidCellCount)]
    pub fn fluid_cell_count(&self) -> u32 {
        self.sim.latest_metrics().fluid_cells
    }

    #[wasm_bindgen(js_name = divClampFires)]
    pub fn div_clamp_fires(&self) -> u32 {
        self.sim.latest_metrics().div_clamp_fires
    }

    #[wasm_bindgen(js_name = pressureClampFires)]
    pub fn pressure_clamp_fires(&self) -> u32 {
        self.sim.latest_metrics().pressure_clamp_fires
    }

    #[wasm_bindgen(js_name = massOverflowFires)]
    pub fn mass_overflow_fires(&self) -> u32 {
        self.sim.latest_metrics().mass_overflow_fires
    }

    #[wasm_bindgen(js_name = projectionResidualMaxAbsDivergence)]
    pub fn projection_residual_max_abs_divergence(&self) -> f32 {
        self.sim.latest_metrics().projection_residual_max_abs_div
    }

    #[wasm_bindgen(js_name = projectionResidualMeanAbsDivergence)]
    pub fn projection_residual_mean_abs_divergence(&self) -> f32 {
        self.sim.latest_metrics().projection_residual_mean_abs_div
    }

    #[wasm_bindgen(js_name = projectionResidualCellCount)]
    pub fn projection_residual_cell_count(&self) -> u32 {
        self.sim.latest_metrics().projection_residual_cells
    }

    #[wasm_bindgen(js_name = lastPressureRbgsPairs)]
    pub fn last_pressure_rbgs_pairs(&self) -> u32 {
        self.sim.last_iterations()
    }

    #[wasm_bindgen(js_name = setPressureResidualAdaptation)]
    pub fn set_pressure_residual_adaptation(&mut self, target: f32, max_pairs: u32) {
        self.sim.set_density_residual_adaptation(target, max_pairs);
    }

    #[wasm_bindgen(js_name = meanTds)]
    pub fn mean_tds(&self) -> f32 {
        self.sim.latest_metrics().mean_tds
    }

    #[wasm_bindgen(js_name = cupTds)]
    pub fn cup_tds(&self) -> f32 {
        self.sim.latest_metrics().cup_tds
    }

    #[wasm_bindgen(js_name = extractionYield)]
    pub fn extraction_yield(&self) -> f32 {
        self.sim.latest_metrics().extraction_yield
    }

    #[wasm_bindgen(js_name = estimatedCupTds)]
    pub fn estimated_cup_tds(&self) -> f32 {
        self.sim.estimated_cup_tds()
    }

    #[wasm_bindgen(js_name = estimatedExtractionYield)]
    pub fn estimated_extraction_yield(&self) -> f32 {
        self.sim.estimated_extraction_yield()
    }
}

#[cfg(target_arch = "wasm32")]
impl WasmSim3D {
    fn rebuild(&mut self, scene: SceneSpec) {
        self.sim
            .rebuild(self.renderer.device(), self.renderer.queue(), scene);
        self.camera = OrbitCamera::new(self.sim.settings().bounds_size);
    }
}

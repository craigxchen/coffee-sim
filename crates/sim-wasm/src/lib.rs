#![cfg_attr(test, allow(dead_code))]

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod emission;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod engine;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod models;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod profiling;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod solvers;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod ui;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod utils;
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) mod mpm_3d {
    #[allow(unused_imports)]
    pub(crate) use crate::solvers::mpm::*;
}
#[cfg(target_arch = "wasm32")]
mod renderer;
#[cfg(test)]
mod renderer_shader_tests;

#[cfg(target_arch = "wasm32")]
use engine::Simulator;
#[cfg(target_arch = "wasm32")]
use renderer::{OrbitCamera, Renderer};
#[cfg(target_arch = "wasm32")]
use solvers::base::{CommonMetrics, SceneSpec, SolverId};
#[cfg(target_arch = "wasm32")]
use solvers::mpm::MpmSettings;
#[cfg(target_arch = "wasm32")]
use solvers::registry::info_for;
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
fn set_str(obj: &js_sys::Object, key: &str, value: &str) -> Result<(), JsValue> {
    js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_str(value)).map(|_| ())
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
    set_number(&obj, "xMeters", v.x * mpm_3d::units::METERS_PER_SIM_UNIT)?;
    set_number(&obj, "yMeters", v.y * mpm_3d::units::METERS_PER_SIM_UNIT)?;
    set_number(&obj, "zMeters", v.z * mpm_3d::units::METERS_PER_SIM_UNIT)?;
    Ok(obj)
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
fn water_diagnostics_object(
    diagnostics: mpm_3d::WaterDiagnostics,
) -> Result<js_sys::Object, JsValue> {
    let obj = js_sys::Object::new();
    set_bool(&obj, "allFinite", diagnostics.all_finite)?;
    set_number(&obj, "simTimeSeconds", diagnostics.sim_time_s)?;
    set_u32(&obj, "activeCount", diagnostics.active_count)?;
    set_u32(&obj, "poolCount", diagnostics.pool_count)?;
    set_number(&obj, "activeMass", diagnostics.active_mass)?;
    set_number(&obj, "activeMassMl", diagnostics.active_mass_ml)?;
    set_number(&obj, "emittedMl", diagnostics.emitted_ml)?;
    set_number(&obj, "restVolumeMl", diagnostics.rest_volume_ml)?;
    set_number(&obj, "currentVolumeMl", diagnostics.current_volume_ml)?;
    set_number(&obj, "meanJ", diagnostics.mean_j)?;
    set_number(&obj, "kineticEnergy", diagnostics.kinetic_energy)?;
    set_number(
        &obj,
        "rmsSpeedMetersPerSecond",
        diagnostics.rms_speed * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "verticalRmsSpeedMetersPerSecond",
        diagnostics.vertical_rms_speed * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "meanVerticalSpeedMetersPerSecond",
        diagnostics.mean_vertical_speed * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "lateralRmsSpeedMetersPerSecond",
        diagnostics.lateral_rms_speed * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "maxSpeedMetersPerSecond",
        diagnostics.max_speed * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "maxUpwardSpeedMetersPerSecond",
        diagnostics.max_upward_speed * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &obj,
        "maxDownwardSpeedMetersPerSecond",
        diagnostics.max_downward_speed * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(&obj, "upwardMomentum", diagnostics.upward_momentum)?;
    set_number(&obj, "downwardMomentum", diagnostics.downward_momentum)?;
    set_number(
        &obj,
        "verticalDipoleMetersPerSecond",
        diagnostics.vertical_dipole_magnitude * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_object(
        &obj,
        "verticalDipole",
        &vec3_object(diagnostics.vertical_dipole)?,
    )?;
    set_number(&obj, "momentumMagnitude", diagnostics.momentum_magnitude)?;
    set_object(&obj, "momentum", &vec3_object(diagnostics.momentum)?)?;
    set_object(&obj, "centroid", &vec3_object(diagnostics.centroid)?)?;
    set_object(&obj, "min", &vec3_object(diagnostics.min)?)?;
    set_object(&obj, "max", &vec3_object(diagnostics.max)?)?;
    set_object(&obj, "extent", &vec3_object(diagnostics.extent)?)?;

    let surface = js_sys::Object::new();
    set_u32(&surface, "binCount", diagnostics.surface_bin_count)?;
    set_u32(&surface, "possibleBins", diagnostics.surface_possible_bins)?;
    set_number(&surface, "meanY", diagnostics.surface_mean_y)?;
    set_number(&surface, "rmsY", diagnostics.surface_rms_y)?;
    set_number(&surface, "minY", diagnostics.surface_min_y)?;
    set_number(&surface, "maxY", diagnostics.surface_max_y)?;
    set_number(&surface, "peakToPeakY", diagnostics.surface_peak_to_peak_y)?;
    set_number(
        &surface,
        "rmsMeters",
        diagnostics.surface_rms_y * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &surface,
        "peakToPeakMeters",
        diagnostics.surface_peak_to_peak_y * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(&surface, "tiltX", diagnostics.surface_tilt.x)?;
    set_number(&surface, "tiltZ", diagnostics.surface_tilt.z)?;
    set_number(&surface, "tilt", diagnostics.surface_tilt_magnitude)?;
    set_number(
        &surface,
        "tiltHeightMeters",
        diagnostics.surface_tilt_height_y * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &surface,
        "residualRmsMeters",
        diagnostics.surface_residual_rms_y * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    set_number(
        &surface,
        "residualPeakToPeakMeters",
        diagnostics.surface_residual_peak_to_peak_y * mpm_3d::units::METERS_PER_SIM_UNIT,
    )?;
    let coverage = if diagnostics.surface_possible_bins > 0 {
        diagnostics.surface_bin_count as f32 / diagnostics.surface_possible_bins as f32
    } else {
        0.0
    };
    set_number(&surface, "coverage", coverage)?;
    set_object(&obj, "surface", &surface)?;

    let pressure = js_sys::Object::new();
    set_u32(
        &pressure,
        "sampleCount",
        diagnostics.hydrostatic_sample_count,
    )?;
    set_number(&pressure, "depthMeters", diagnostics.hydrostatic_depth_m)?;
    set_number(
        &pressure,
        "topPressurePa",
        diagnostics.hydrostatic_top_pressure_pa,
    )?;
    set_number(
        &pressure,
        "bottomPressurePa",
        diagnostics.hydrostatic_bottom_pressure_pa,
    )?;
    set_number(
        &pressure,
        "deltaPressurePa",
        diagnostics.hydrostatic_delta_pressure_pa,
    )?;
    set_number(
        &pressure,
        "gradientPaPerMeter",
        diagnostics.hydrostatic_gradient_pa_per_m,
    )?;
    set_bool(
        &pressure,
        "bottomHigher",
        diagnostics.hydrostatic_bottom_higher,
    )?;
    set_object(&obj, "hydrostaticPressure", &pressure)?;
    set_number(
        &obj,
        "dissolvedSoluteMass",
        diagnostics.dissolved_solute_mass,
    )?;
    set_number(&obj, "meanTds", diagnostics.mean_tds)?;
    set_number(&obj, "cupWaterMass", diagnostics.cup_water_mass)?;
    set_number(&obj, "cupSoluteMass", diagnostics.cup_solute_mass)?;
    set_number(&obj, "cupTds", diagnostics.cup_tds)?;
    set_number(&obj, "extractionYield", diagnostics.extraction_yield)?;
    Ok(obj)
}

#[cfg(target_arch = "wasm32")]
fn metrics_snapshot_object(snapshot: mpm_3d::MetricsSnapshot) -> Result<js_sys::Object, JsValue> {
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
fn common_metrics_object(snapshot: CommonMetrics) -> Result<js_sys::Object, JsValue> {
    let obj = js_sys::Object::new();
    set_number(&obj, "maxAbsDivergence", snapshot.max_abs_divergence)?;
    set_u32(&obj, "fluidCellCount", snapshot.fluid_cell_count)?;
    set_u32(&obj, "divClampFires", snapshot.div_clamp_fires)?;
    set_u32(&obj, "pressureClampFires", snapshot.pressure_clamp_fires)?;
    set_u32(&obj, "massOverflowFires", snapshot.mass_overflow_fires)?;
    set_number(
        &obj,
        "projectionResidualMaxAbsDivergence",
        snapshot.projection_residual_max_abs_divergence,
    )?;
    set_number(
        &obj,
        "projectionResidualMeanAbsDivergence",
        snapshot.projection_residual_mean_abs_divergence,
    )?;
    set_u32(
        &obj,
        "projectionResidualCellCount",
        snapshot.projection_residual_cell_count,
    )?;
    set_number(&obj, "meanTds", snapshot.mean_tds)?;
    set_number(&obj, "cupTds", snapshot.cup_tds)?;
    set_number(&obj, "extractionYield", snapshot.extraction_yield)?;
    Ok(obj)
}

#[cfg(target_arch = "wasm32")]
fn common_water_diagnostics_object(snapshot: CommonMetrics) -> Result<js_sys::Object, JsValue> {
    let obj = js_sys::Object::new();
    set_bool(&obj, "allFinite", true)?;
    set_number(&obj, "simTimeSeconds", snapshot.sim_time_s)?;
    set_u32(&obj, "activeCount", snapshot.water_slots_used)?;
    set_u32(&obj, "poolCount", snapshot.water_slots_used)?;
    set_number(&obj, "activeMass", snapshot.total_emitted_mass)?;
    set_number(&obj, "activeMassMl", snapshot.total_emitted_ml)?;
    set_number(&obj, "emittedMl", snapshot.total_emitted_ml)?;
    set_number(&obj, "restVolumeMl", snapshot.total_emitted_ml)?;
    set_number(&obj, "currentVolumeMl", snapshot.total_emitted_ml)?;
    set_number(&obj, "meanJ", 1.0)?;
    set_number(&obj, "kineticEnergy", 0.0)?;
    set_number(&obj, "rmsSpeedMetersPerSecond", 0.0)?;
    set_number(&obj, "verticalRmsSpeedMetersPerSecond", 0.0)?;
    set_number(&obj, "meanVerticalSpeedMetersPerSecond", 0.0)?;
    set_number(&obj, "lateralRmsSpeedMetersPerSecond", 0.0)?;
    set_number(&obj, "maxSpeedMetersPerSecond", 0.0)?;
    set_number(&obj, "maxUpwardSpeedMetersPerSecond", 0.0)?;
    set_number(&obj, "maxDownwardSpeedMetersPerSecond", 0.0)?;
    set_number(&obj, "upwardMomentum", 0.0)?;
    set_number(&obj, "downwardMomentum", 0.0)?;
    set_number(&obj, "verticalDipoleMetersPerSecond", 0.0)?;
    set_object(
        &obj,
        "verticalDipole",
        &vec3_object(coffee_sim_core::Vec3::ZERO)?,
    )?;
    set_number(&obj, "momentumMagnitude", 0.0)?;
    set_object(&obj, "momentum", &vec3_object(coffee_sim_core::Vec3::ZERO)?)?;
    set_object(&obj, "centroid", &vec3_object(coffee_sim_core::Vec3::ZERO)?)?;
    set_object(&obj, "min", &vec3_object(coffee_sim_core::Vec3::ZERO)?)?;
    set_object(&obj, "max", &vec3_object(coffee_sim_core::Vec3::ZERO)?)?;
    set_object(&obj, "extent", &vec3_object(coffee_sim_core::Vec3::ZERO)?)?;
    let surface = js_sys::Object::new();
    set_u32(&surface, "binCount", 0)?;
    set_u32(&surface, "possibleBins", 0)?;
    set_number(&surface, "meanY", 0.0)?;
    set_number(&surface, "rmsY", 0.0)?;
    set_number(&surface, "minY", 0.0)?;
    set_number(&surface, "maxY", 0.0)?;
    set_number(&surface, "peakToPeakY", 0.0)?;
    set_number(&surface, "rmsMeters", 0.0)?;
    set_number(&surface, "peakToPeakMeters", 0.0)?;
    set_number(&surface, "tiltX", 0.0)?;
    set_number(&surface, "tiltZ", 0.0)?;
    set_number(&surface, "tilt", 0.0)?;
    set_number(&surface, "tiltHeightMeters", 0.0)?;
    set_number(&surface, "residualRmsMeters", 0.0)?;
    set_number(&surface, "residualPeakToPeakMeters", 0.0)?;
    set_number(&surface, "coverage", 0.0)?;
    set_object(&obj, "surface", &surface)?;
    let pressure = js_sys::Object::new();
    set_u32(&pressure, "sampleCount", 0)?;
    set_number(&pressure, "depthMeters", 0.0)?;
    set_number(&pressure, "topPressurePa", 0.0)?;
    set_number(&pressure, "bottomPressurePa", 0.0)?;
    set_number(&pressure, "deltaPressurePa", 0.0)?;
    set_number(&pressure, "gradientPaPerMeter", 0.0)?;
    set_bool(&pressure, "bottomHigher", false)?;
    set_object(&obj, "hydrostaticPressure", &pressure)?;
    set_number(&obj, "dissolvedSoluteMass", 0.0)?;
    set_number(&obj, "meanTds", snapshot.mean_tds)?;
    set_number(&obj, "cupWaterMass", 0.0)?;
    set_number(&obj, "cupSoluteMass", 0.0)?;
    set_number(&obj, "cupTds", snapshot.cup_tds)?;
    set_number(&obj, "extractionYield", snapshot.extraction_yield)?;
    Ok(obj)
}

// ── 3D WebGPU App ────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSim3D {
    sim: Simulator,
    renderer: Renderer,
    camera: OrbitCamera,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmSim3D {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WasmSim3D, JsValue> {
        console_error_panic_hook::set_once();

        let initial_scene = SceneSpec::CenterPour;
        let geometry = MpmSettings::benchmark_center_pour().render_scene_geometry();
        let renderer = Renderer::new(canvas, &geometry).await?;
        let sim = Simulator::new(
            renderer.device(),
            renderer.queue(),
            SolverId::Mpm,
            initial_scene,
        )
        .map_err(|err| JsValue::from_str(&err))?;
        let camera = OrbitCamera::new(sim.snapshot().render.bounds_size());
        Ok(Self {
            sim,
            renderer,
            camera,
        })
    }

    pub fn reset(&mut self) {
        self.sim
            .reset(self.renderer.queue(), self.renderer.device());
        self.camera = OrbitCamera::new(self.sim.snapshot().render.bounds_size());
    }

    #[wasm_bindgen(js_name = availableSolvers)]
    pub fn available_solvers(&self) -> Result<js_sys::Array, JsValue> {
        let solvers = js_sys::Array::new();
        for &id in SolverId::all() {
            let info = info_for(id);
            let obj = js_sys::Object::new();
            set_str(&obj, "id", id.id())?;
            set_str(&obj, "name", info.name)?;
            set_bool(&obj, "experimental", info.experimental)?;
            solvers.push(&obj);
        }
        Ok(solvers)
    }

    #[wasm_bindgen(js_name = activeSolver)]
    pub fn active_solver(&self) -> String {
        self.sim.active_id().id().to_string()
    }

    #[wasm_bindgen(js_name = setSolver)]
    pub fn set_solver(&mut self, solver_id: &str) -> Result<(), JsValue> {
        let id = SolverId::from_id(solver_id)
            .ok_or_else(|| JsValue::from_str(&format!("unknown solver: {solver_id}")))?;
        self.sim
            .switch_solver(self.renderer.device(), self.renderer.queue(), id)
            .map_err(|err| JsValue::from_str(&err))?;
        self.camera = OrbitCamera::new(self.sim.snapshot().render.bounds_size());
        Ok(())
    }

    #[wasm_bindgen(js_name = loadDefaultScene)]
    pub fn load_default_scene(&mut self) -> Result<(), JsValue> {
        self.load_scene(SceneSpec::CenterPour)
    }

    #[wasm_bindgen(js_name = loadBenchmarkFreeStream)]
    pub fn load_benchmark_free_stream(&mut self) -> Result<(), JsValue> {
        self.load_scene(SceneSpec::FreeStream)
    }

    #[wasm_bindgen(js_name = loadBenchmarkCenterPour)]
    pub fn load_benchmark_center_pour(&mut self) -> Result<(), JsValue> {
        self.load_scene(SceneSpec::CenterPour)
    }

    #[wasm_bindgen(js_name = loadBenchmarkFilterWaterBlock)]
    pub fn load_benchmark_filter_water_block(&mut self) -> Result<(), JsValue> {
        self.load_scene(SceneSpec::Debug {
            id: "filter-water-block".to_string(),
        })
    }

    #[wasm_bindgen(js_name = loadDebugScene)]
    pub fn load_debug_scene(&mut self, scene_id: &str) -> Result<(), JsValue> {
        self.load_scene(SceneSpec::Debug {
            id: scene_id.to_string(),
        })
    }

    #[wasm_bindgen(js_name = stepFrame)]
    pub fn step_frame(&mut self, frame_time: f32) {
        self.sim
            .step_frame(self.renderer.device(), self.renderer.queue(), frame_time);
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
        self.sim.common_metrics().exit_speed_m_s
    }

    #[wasm_bindgen(js_name = spoutX)]
    pub fn spout_x(&self) -> f32 {
        self.sim.common_metrics().spout_position.x
    }

    #[wasm_bindgen(js_name = spoutY)]
    pub fn spout_y(&self) -> f32 {
        self.sim.common_metrics().spout_position.y
    }

    #[wasm_bindgen(js_name = spoutZ)]
    pub fn spout_z(&self) -> f32 {
        self.sim.common_metrics().spout_position.z
    }

    #[wasm_bindgen(js_name = flowRate)]
    pub fn flow_rate(&self) -> f32 {
        self.sim.common_metrics().flow_rate_ml_s
    }

    #[wasm_bindgen(js_name = exitSpeed)]
    pub fn exit_speed(&self) -> f32 {
        self.sim.common_metrics().exit_speed
    }

    #[wasm_bindgen(js_name = exitSpeedMetersPerSecond)]
    pub fn exit_speed_m_s(&self) -> f32 {
        self.sim.common_metrics().exit_speed_m_s
    }

    pub fn render(&mut self) -> Result<(), JsValue> {
        let snapshot = self.sim.snapshot();
        self.renderer.render_3d(&snapshot.render, self.camera)
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
        self.camera
            .zoom(delta, self.sim.snapshot().render.bounds_size());
    }

    #[wasm_bindgen(js_name = panCamera)]
    pub fn pan_camera(&mut self, right: f32, up: f32, forward: f32) {
        self.camera
            .pan(right, up, forward, self.sim.snapshot().render.bounds_size());
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
        self.sim.common_metrics().particle_count
    }

    #[wasm_bindgen(js_name = waterSlotsUsed)]
    pub fn water_slots_used(&self) -> u32 {
        self.sim.common_metrics().water_slots_used
    }

    #[wasm_bindgen(js_name = bedParticleCount)]
    pub fn bed_particle_count(&self) -> u32 {
        self.sim.common_metrics().bed_particle_count
    }

    #[wasm_bindgen(js_name = maxParticles)]
    pub fn max_particles(&self) -> u32 {
        self.sim.common_metrics().max_particles
    }

    #[wasm_bindgen(js_name = simTime)]
    pub fn sim_time(&self) -> f32 {
        self.sim.common_metrics().sim_time_s
    }

    #[wasm_bindgen(js_name = frameEmittedMass)]
    pub fn frame_emitted_mass(&self) -> f32 {
        self.sim.common_metrics().frame_emitted_mass
    }

    #[wasm_bindgen(js_name = frameEmittedMl)]
    pub fn frame_emitted_ml(&self) -> f32 {
        self.sim.common_metrics().frame_emitted_ml
    }

    #[wasm_bindgen(js_name = totalEmittedMass)]
    pub fn total_emitted_mass(&self) -> f32 {
        self.sim.common_metrics().total_emitted_mass
    }

    #[wasm_bindgen(js_name = totalEmittedMl)]
    pub fn total_emitted_ml(&self) -> f32 {
        self.sim.common_metrics().total_emitted_ml
    }

    #[wasm_bindgen(js_name = frameDroppedParticles)]
    pub fn frame_dropped_particles(&self) -> u32 {
        self.sim.common_metrics().frame_dropped_particles
    }

    #[wasm_bindgen(js_name = totalDroppedParticles)]
    pub fn total_dropped_particles(&self) -> u32 {
        self.sim.common_metrics().total_dropped_particles
    }

    #[wasm_bindgen(js_name = hasBed)]
    pub fn has_bed(&self) -> bool {
        self.sim.common_metrics().has_bed
    }

    #[wasm_bindgen(js_name = refreshMetrics)]
    pub async fn refresh_metrics(&mut self) -> Result<(), JsValue> {
        Ok(())
    }

    #[wasm_bindgen(js_name = sampleMetrics)]
    pub fn sample_metrics(&self, delay_frames: u32) -> js_sys::Promise {
        if let Some(metrics) = self.sim.metrics_buffer() {
            let device = self.renderer.device().clone();
            let queue = self.renderer.queue().clone();
            let has_bed = self.sim.common_metrics().has_bed;
            return wasm_bindgen_futures::future_to_promise(async move {
                let snapshot = mpm_3d::MpmSim3D::sample_metrics_after_delay(
                    device,
                    queue,
                    metrics,
                    has_bed,
                    delay_frames,
                )
                .await?;
                Ok(metrics_snapshot_object(snapshot)?.into())
            });
        }

        let snapshot = self.sim.common_metrics();
        wasm_bindgen_futures::future_to_promise(async move {
            Ok(common_metrics_object(snapshot)?.into())
        })
    }

    #[wasm_bindgen(js_name = waterDiagnostics)]
    pub async fn water_diagnostics(&self) -> Result<JsValue, JsValue> {
        Ok(common_water_diagnostics_object(self.sim.common_metrics())?.into())
    }

    #[wasm_bindgen(js_name = maxAbsDivergence)]
    pub fn max_abs_divergence(&self) -> f32 {
        self.sim.common_metrics().max_abs_divergence
    }

    #[wasm_bindgen(js_name = fluidCellCount)]
    pub fn fluid_cell_count(&self) -> u32 {
        self.sim.common_metrics().fluid_cell_count
    }

    #[wasm_bindgen(js_name = divClampFires)]
    pub fn div_clamp_fires(&self) -> u32 {
        self.sim.common_metrics().div_clamp_fires
    }

    #[wasm_bindgen(js_name = pressureClampFires)]
    pub fn pressure_clamp_fires(&self) -> u32 {
        self.sim.common_metrics().pressure_clamp_fires
    }

    #[wasm_bindgen(js_name = massOverflowFires)]
    pub fn mass_overflow_fires(&self) -> u32 {
        self.sim.common_metrics().mass_overflow_fires
    }

    #[wasm_bindgen(js_name = projectionResidualMaxAbsDivergence)]
    pub fn projection_residual_max_abs_divergence(&self) -> f32 {
        self.sim
            .common_metrics()
            .projection_residual_max_abs_divergence
    }

    #[wasm_bindgen(js_name = projectionResidualMeanAbsDivergence)]
    pub fn projection_residual_mean_abs_divergence(&self) -> f32 {
        self.sim
            .common_metrics()
            .projection_residual_mean_abs_divergence
    }

    #[wasm_bindgen(js_name = projectionResidualCellCount)]
    pub fn projection_residual_cell_count(&self) -> u32 {
        self.sim.common_metrics().projection_residual_cell_count
    }

    #[wasm_bindgen(js_name = lastPressureRbgsPairs)]
    pub fn last_pressure_rbgs_pairs(&self) -> u32 {
        self.sim.common_metrics().last_pressure_pairs
    }

    #[wasm_bindgen(js_name = setPressureResidualAdaptation)]
    pub fn set_pressure_residual_adaptation(&mut self, target: f32, max_pairs: u32) {
        self.sim.set_pressure_residual_adaptation(target, max_pairs);
    }

    #[wasm_bindgen(js_name = meanTds)]
    pub fn mean_tds(&self) -> f32 {
        self.sim.common_metrics().mean_tds
    }

    #[wasm_bindgen(js_name = cupTds)]
    pub fn cup_tds(&self) -> f32 {
        self.sim.common_metrics().cup_tds
    }

    #[wasm_bindgen(js_name = extractionYield)]
    pub fn extraction_yield(&self) -> f32 {
        self.sim.common_metrics().extraction_yield
    }

    #[wasm_bindgen(js_name = estimatedCupTds)]
    pub fn estimated_cup_tds(&self) -> f32 {
        self.sim.common_metrics().estimated_cup_tds
    }

    #[wasm_bindgen(js_name = estimatedExtractionYield)]
    pub fn estimated_extraction_yield(&self) -> f32 {
        self.sim.common_metrics().estimated_extraction_yield
    }
}

#[cfg(target_arch = "wasm32")]
impl WasmSim3D {
    fn load_scene(&mut self, scene: SceneSpec) -> Result<(), JsValue> {
        self.sim
            .load_scene(self.renderer.queue(), self.renderer.device(), scene)
            .map_err(|err| JsValue::from_str(&err))?;
        self.camera = OrbitCamera::new(self.sim.snapshot().render.bounds_size());
        Ok(())
    }
}

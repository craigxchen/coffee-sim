use crate::models::calibration::{CalibrationControl, CalibrationProfile};
use crate::solvers::base::{
    CalibrationApplyResult, CalibrationSupport, CommonMetrics, DiagnosticsPoll, DiagnosticsTicket,
    FrameContext, FrameSnapshot, FrameSolver, ResetContext, SceneSpec, SolverId,
};
use crate::solvers::registry::build_solver_with_mpm_v2_calibration;
use crate::solvers::twofield_runtime::{
    MPM_V2_CALIBRATION_DRAINAGE_SCENE_ID, MPM_V2_LATE_BREW_BENCHMARK_SCENE_ID,
};

#[derive(Clone, Debug, Default)]
struct CalibrationState {
    mpm_v2: CalibrationProfile,
}

impl CalibrationState {
    fn for_build(&self, id: SolverId) -> Option<&CalibrationProfile> {
        (id == SolverId::MpmV2).then_some(&self.mpm_v2)
    }

    fn capture(&mut self, id: SolverId, profile: Option<&CalibrationProfile>) {
        if id == SolverId::MpmV2 {
            if let Some(profile) = profile {
                self.mpm_v2 = profile.clone();
            }
        }
    }

    fn reset_mpm_v2_defaults(&mut self) {
        self.mpm_v2 = CalibrationProfile::default();
    }
}

pub(crate) struct Simulator {
    active_id: SolverId,
    scene: SceneSpec,
    metrics: CommonMetrics,
    solver: Box<dyn FrameSolver>,
    calibration: CalibrationState,
}

fn scene_for_solver(scene: &SceneSpec, id: SolverId) -> SceneSpec {
    match scene {
        SceneSpec::Debug { id: debug_id }
            if id == SolverId::MpmV2
                && matches!(
                    debug_id.as_str(),
                    MPM_V2_LATE_BREW_BENCHMARK_SCENE_ID | MPM_V2_CALIBRATION_DRAINAGE_SCENE_ID
                ) =>
        {
            scene.clone()
        }
        SceneSpec::Debug { id: debug_id }
            if matches!(
                debug_id.as_str(),
                MPM_V2_LATE_BREW_BENCHMARK_SCENE_ID | MPM_V2_CALIBRATION_DRAINAGE_SCENE_ID
            ) =>
        {
            SceneSpec::CenterPour
        }
        SceneSpec::Debug { .. } if id == SolverId::Mpm => scene.clone(),
        SceneSpec::Debug { .. } => SceneSpec::CenterPour,
        _ => scene.clone(),
    }
}

impl Simulator {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: SolverId,
        scene: SceneSpec,
    ) -> Result<Self, String> {
        let scene = scene_for_solver(&scene, id);
        let calibration = CalibrationState::default();
        let solver = build_solver_with_mpm_v2_calibration(
            id,
            device,
            queue,
            &scene,
            calibration.for_build(id),
        )?;
        let metrics = solver.snapshot().metrics;
        Ok(Self {
            active_id: id,
            scene,
            metrics,
            solver,
            calibration,
        })
    }

    pub(crate) fn active_id(&self) -> SolverId {
        self.active_id
    }

    pub(crate) fn switch_solver(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: SolverId,
    ) -> Result<(), String> {
        let active_profile = self.solver.calibration_profile().cloned();
        self.calibration
            .capture(self.active_id, active_profile.as_ref());
        let scene = scene_for_solver(&self.scene, id);
        let solver = build_solver_with_mpm_v2_calibration(
            id,
            device,
            queue,
            &scene,
            self.calibration.for_build(id),
        )?;
        let metrics = solver.snapshot().metrics;
        self.active_id = id;
        self.scene = scene;
        self.metrics = metrics;
        self.solver = solver;
        Ok(())
    }

    pub(crate) fn reset(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        let ctx = ResetContext { device, queue };
        self.metrics = self.solver.reset(ctx);
    }

    pub(crate) fn load_scene(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        scene: SceneSpec,
    ) -> Result<(), String> {
        let scene = scene_for_solver(&scene, self.active_id);
        let ctx = ResetContext { device, queue };
        self.metrics = self.solver.reset_scene(ctx, &scene)?;
        self.scene = scene;
        let active_profile = self.solver.calibration_profile().cloned();
        self.calibration
            .capture(self.active_id, active_profile.as_ref());
        Ok(())
    }

    pub(crate) fn step_frame(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        self.metrics = self.solver.step_frame(FrameContext { device, queue, dt });
    }

    pub(crate) fn snapshot(&self) -> FrameSnapshot<'_> {
        self.solver.snapshot()
    }

    pub(crate) fn common_metrics(&self) -> CommonMetrics {
        self.metrics
    }

    pub(crate) fn request_diagnostics_snapshot(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
    ) -> Result<DiagnosticsTicket, String> {
        let ticket = self
            .solver
            .request_diagnostics_snapshot(ResetContext { device, queue })?;
        self.metrics = self.solver.snapshot().metrics;
        Ok(ticket)
    }

    pub(crate) fn poll_diagnostics(&mut self) -> DiagnosticsPoll {
        let status = self.solver.poll_diagnostics();
        self.metrics = self.solver.snapshot().metrics;
        status
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn metrics_buffer(&self) -> Option<wgpu::Buffer> {
        self.solver.metrics_buffer()
    }

    pub(crate) fn set_exit_speed_m_s(&mut self, speed_m_s: f32) {
        self.solver.set_water_velocity_m_s(speed_m_s);
        self.metrics = self.solver.snapshot().metrics;
    }

    pub(crate) fn set_spout_position(&mut self, x: f32, y: f32, z: f32) {
        self.solver.set_spout_position(x, y, z);
        self.metrics = self.solver.snapshot().metrics;
    }

    pub(crate) fn set_pressure_residual_adaptation(&mut self, target: f32, max_pairs: u32) {
        self.solver
            .set_pressure_residual_adaptation(target, max_pairs);
        self.metrics = self.solver.snapshot().metrics;
    }

    pub(crate) fn calibration_profile(&self) -> Option<&CalibrationProfile> {
        self.solver.calibration_profile()
    }

    pub(crate) fn calibration_support(&self, control: CalibrationControl) -> CalibrationSupport {
        self.solver.calibration_support(control)
    }

    pub(crate) fn set_calibration_value(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        id: &str,
        value: f32,
    ) -> Result<CalibrationApplyResult, String> {
        let ctx = ResetContext { device, queue };
        let result = self.solver.set_calibration_value(ctx, id, value)?;
        self.metrics = self.solver.snapshot().metrics;
        let active_profile = self.solver.calibration_profile().cloned();
        self.calibration
            .capture(self.active_id, active_profile.as_ref());
        Ok(result)
    }

    pub(crate) fn reset_calibration_defaults(
        &mut self,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
    ) -> Result<bool, String> {
        let ctx = ResetContext { device, queue };
        let simulation_reset = self.solver.reset_calibration_defaults(ctx)?;
        if self.active_id == SolverId::MpmV2 {
            self.calibration.reset_mpm_v2_defaults();
        }
        self.metrics = self.solver.snapshot().metrics;
        Ok(simulation_reset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_scene_routing_keeps_only_solver_owned_debug_workloads() {
        let legacy = SceneSpec::Debug {
            id: "filter-water-block".to_string(),
        };
        assert_eq!(scene_for_solver(&legacy, SolverId::Mpm), legacy);
        assert_eq!(
            scene_for_solver(&legacy, SolverId::MpmV2),
            SceneSpec::CenterPour
        );

        let benchmark = SceneSpec::Debug {
            id: MPM_V2_LATE_BREW_BENCHMARK_SCENE_ID.to_string(),
        };
        assert_eq!(scene_for_solver(&benchmark, SolverId::MpmV2), benchmark);
        assert_eq!(
            scene_for_solver(&benchmark, SolverId::Xpbd),
            SceneSpec::CenterPour
        );
        assert_eq!(
            scene_for_solver(&benchmark, SolverId::Mpm),
            SceneSpec::CenterPour
        );

        let calibration = SceneSpec::Debug {
            id: MPM_V2_CALIBRATION_DRAINAGE_SCENE_ID.to_string(),
        };
        assert_eq!(scene_for_solver(&calibration, SolverId::MpmV2), calibration);
        assert_eq!(
            scene_for_solver(&calibration, SolverId::Xpbd),
            SceneSpec::CenterPour
        );
    }

    #[test]
    fn mpm_v2_calibration_state_survives_solver_and_scene_routing_until_explicit_reset() {
        let mut tuned = CalibrationProfile::default();
        tuned.set_by_id("drag_multiplier", 2.25).unwrap();
        tuned
            .set_by_id("reference_bed_porosity_fraction", 0.47)
            .unwrap();

        let mut state = CalibrationState::default();
        state.capture(SolverId::MpmV2, Some(&tuned));
        assert_eq!(state.for_build(SolverId::MpmV2), Some(&tuned));
        assert_eq!(state.for_build(SolverId::Mpm), None);
        assert_eq!(state.for_build(SolverId::Xpbd), None);

        // Visiting another solver or routing another scene must not overwrite MPM-v2's saved
        // profile. Only a profile reported by an active MPM-v2 solver is authoritative.
        let unrelated = CalibrationProfile::default();
        state.capture(SolverId::Xpbd, Some(&unrelated));
        let _ = scene_for_solver(&SceneSpec::FreeStream, SolverId::MpmV2);
        assert_eq!(state.for_build(SolverId::MpmV2), Some(&tuned));

        state.reset_mpm_v2_defaults();
        assert_eq!(
            state.for_build(SolverId::MpmV2),
            Some(&CalibrationProfile::default())
        );
    }
}

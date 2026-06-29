use super::base::{FrameSolver, Paradigm, SceneSpec, SolverId, SolverInfo, Stability};
use super::mpm::MpmSim3D;
use super::twofield_runtime::TwofieldFrameSolver;
use super::xpbd_runtime::XpbdFrameSolver;

const MPM_INFO: SolverInfo = SolverInfo {
    id: SolverId::Mpm,
    name: "MPM",
    paradigm: Paradigm::ForceBased,
    owns_grid: true,
    stability: Stability::CflLimited { c: 0.5 },
    experimental: false,
};

const XPBD_INFO: SolverInfo = SolverInfo {
    id: SolverId::Xpbd,
    name: "XPBD",
    paradigm: Paradigm::PositionBased,
    owns_grid: false,
    stability: Stability::CflLimited { c: 0.5 },
    experimental: true,
};

const TWOFIELD_INFO: SolverInfo = SolverInfo {
    id: SolverId::Twofield,
    name: "Two-field",
    paradigm: Paradigm::Hybrid,
    owns_grid: true,
    stability: Stability::CflLimited { c: 0.5 },
    experimental: true,
};

pub(crate) fn info_for(id: SolverId) -> SolverInfo {
    match id {
        SolverId::Mpm => MPM_INFO,
        SolverId::Xpbd => XPBD_INFO,
        SolverId::Twofield => TWOFIELD_INFO,
    }
}

pub(crate) fn build_solver(
    id: SolverId,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &SceneSpec,
) -> Result<Box<dyn FrameSolver>, String> {
    match id {
        SolverId::Mpm => {
            let settings = match scene {
                SceneSpec::CenterPour => super::mpm::MpmSettings::benchmark_center_pour(),
                SceneSpec::FreeStream => super::mpm::MpmSettings::benchmark_free_stream(),
                SceneSpec::Debug { id } => super::mpm::DebugScene::from_id(id)
                    .ok_or_else(|| format!("unknown debug scene: {id}"))?
                    .settings(),
            };
            let mut solver = MpmSim3D::new(device, queue, settings);
            if let SceneSpec::Debug { id } = scene {
                let debug_scene = super::mpm::DebugScene::from_id(id)
                    .ok_or_else(|| format!("unknown debug scene: {id}"))?;
                debug_scene.seed(&mut solver, queue);
            }
            Ok(Box::new(solver))
        }
        SolverId::Xpbd => Ok(Box::new(XpbdFrameSolver::new(device, queue, scene)?)),
        SolverId::Twofield => Ok(Box::new(TwofieldFrameSolver::new(device, queue, scene)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::base::FrameContext;
    use crate::ui::ParticleRenderSource;

    fn request_adapter() -> Option<wgpu::Adapter> {
        if std::env::var_os("COFFEE_SIM_SKIP_GPU_TESTS").is_some() {
            return None;
        }

        let instance = wgpu::Instance::default();
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()
    }

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let adapter = request_adapter()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("coffee-sim registry seam test device"),
            required_features: wgpu::Features::empty(),
            required_limits: super::super::mpm::required_limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()
    }

    #[test]
    fn every_registered_solver_has_info() {
        for &id in SolverId::all() {
            let info = info_for(id);
            assert_eq!(info.id, id);
            assert!(!info.name.is_empty());
        }
    }

    #[test]
    fn solver_ids_parse_registered_values() {
        for &id in SolverId::all() {
            assert_eq!(SolverId::from_id(id.id()), Some(id));
        }
        assert_eq!(SolverId::from_id("nope"), None);
    }

    #[test]
    fn registered_solvers_build_and_snapshot_through_common_render_view() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("no GPU adapter available; skipping registry render-view seam smoke");
            return;
        };

        for &id in SolverId::all() {
            let mut solver =
                build_solver(id, &device, &queue, &SceneSpec::CenterPour).expect("build solver");
            let metrics = solver.step_frame(FrameContext {
                device: &device,
                queue: &queue,
                dt: 1.0 / 60.0,
            });
            let snapshot = solver.snapshot();
            assert_eq!(snapshot.render.particle_count(), metrics.particle_count);
            assert!(snapshot.render.render_radius() > 0.0);
            assert!(snapshot.render.bounds_size().x > 0.0);
            match snapshot.render.particle_source() {
                ParticleRenderSource::Packed { render_buffer } => {
                    let _ = render_buffer;
                }
                ParticleRenderSource::Canonical {
                    positions,
                    velocities,
                    phases,
                } => {
                    let _ = (positions, velocities, phases);
                }
            }
        }
    }

    #[test]
    fn engine_switches_between_registered_solvers_and_falls_back_from_debug_scene() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("no GPU adapter available; skipping engine solver-switch smoke");
            return;
        };

        let mut sim = crate::engine::Simulator::new(
            &device,
            &queue,
            SolverId::Mpm,
            SceneSpec::Debug {
                id: "filter-water-block".to_string(),
            },
        )
        .expect("build MPM debug scene");
        assert_eq!(sim.active_id(), SolverId::Mpm);

        for &id in &[SolverId::Xpbd, SolverId::Twofield] {
            sim.switch_solver(&device, &queue, id)
                .expect("switch falls back from MPM debug scene");
            assert_eq!(sim.active_id(), id);
            sim.step_frame(&device, &queue, 1.0 / 60.0);
            assert!(sim.snapshot().render.particle_count() > 0);
        }

        sim.switch_solver(&device, &queue, SolverId::Mpm)
            .expect("switch back to MPM");
        assert_eq!(sim.active_id(), SolverId::Mpm);
        sim.step_frame(&device, &queue, 1.0 / 60.0);
        assert!(sim.snapshot().render.particle_count() > 0);
    }
}

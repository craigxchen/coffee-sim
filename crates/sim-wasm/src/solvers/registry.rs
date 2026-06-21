use super::base::{Paradigm, Solver, SolverId, SolverInfo, Stability};
use super::mpm::{MpmSettings, MpmSim3D};

const MPM_INFO: SolverInfo = SolverInfo {
    id: SolverId::Mpm,
    name: "MPM",
    paradigm: Paradigm::ForceBased,
    owns_grid: true,
    stability: Stability::CflLimited { c: 0.5 },
};

pub(crate) enum SolverBuildConfig {
    Mpm(MpmSettings),
}

pub(crate) fn info_for(id: SolverId) -> SolverInfo {
    match id {
        SolverId::Mpm => MPM_INFO,
    }
}

pub(crate) fn build_solver(
    id: SolverId,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: SolverBuildConfig,
) -> Box<dyn Solver> {
    match (id, config) {
        (SolverId::Mpm, SolverBuildConfig::Mpm(settings)) => {
            Box::new(MpmSim3D::new(device, queue, settings))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_solver_has_info() {
        for &id in SolverId::all() {
            let info = info_for(id);
            assert_eq!(info.id, id);
            assert!(!info.name.is_empty());
        }
    }
}

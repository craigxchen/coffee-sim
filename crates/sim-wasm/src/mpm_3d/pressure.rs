use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PressureSolverKind {
    Rbgs,
    JacobiCg,
}

impl PressureSolverKind {
    pub(crate) const ALL: &'static [Self] = &[Self::Rbgs, Self::JacobiCg];

    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Rbgs => "rbgs",
            Self::JacobiCg => "jacobi-cg",
        }
    }
}

impl fmt::Display for PressureSolverKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for PressureSolverKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "rbgs" | "mpm-rbgs" | "pressure-rbgs" => Ok(Self::Rbgs),
            "cg" | "jacobi-cg" | "mpm-cg" | "pressure-cg" => Ok(Self::JacobiCg),
            other => Err(format!(
                "unknown pressure solver '{other}'; available solvers: {}",
                Self::ALL
                    .iter()
                    .map(|kind| kind.id())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PressureContext {
    pub kind: PressureSolverKind,
    pub cell_wg: u32,
    pub rbgs_pairs: u32,
    pub cg_iterations: u32,
}

pub(crate) struct PressurePipelines {
    pub(crate) rbgs: RbgsPressureSolver,
    pub(crate) jacobi_cg: JacobiCgPressureSolver,
}

impl PressurePipelines {
    #[cfg(test)]
    pub(crate) fn iterations_per_substep(&self, ctx: PressureContext) -> u32 {
        match ctx.kind {
            PressureSolverKind::Rbgs => ctx.rbgs_pairs,
            PressureSolverKind::JacobiCg => ctx.cg_iterations,
        }
    }

    pub(crate) fn classify_pipeline_for(&self, kind: PressureSolverKind) -> &wgpu::ComputePipeline {
        match kind {
            PressureSolverKind::Rbgs => &self.rbgs.classify_cells,
            PressureSolverKind::JacobiCg => &self.jacobi_cg.classify_cells,
        }
    }

    pub(crate) fn project_pipeline_for(&self, kind: PressureSolverKind) -> &wgpu::ComputePipeline {
        match kind {
            PressureSolverKind::Rbgs => &self.rbgs.project_pressure,
            PressureSolverKind::JacobiCg => &self.jacobi_cg.project_pressure,
        }
    }

    pub(crate) fn residual_pipeline_for(&self, kind: PressureSolverKind) -> &wgpu::ComputePipeline {
        match kind {
            PressureSolverKind::Rbgs => &self.rbgs.pressure_residual,
            PressureSolverKind::JacobiCg => &self.jacobi_cg.pressure_residual,
        }
    }

    pub(crate) fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, ctx: PressureContext) {
        match ctx.kind {
            PressureSolverKind::Rbgs => self.rbgs.encode_solve(pass, ctx),
            PressureSolverKind::JacobiCg => self.jacobi_cg.encode_solve(pass, ctx),
        }
    }

    #[cfg(test)]
    pub(crate) fn estimated_timestamp_scopes_for(&self, kind: PressureSolverKind) -> u32 {
        match kind {
            PressureSolverKind::Rbgs => {
                // classify, solve, project, residual
                4
            }
            PressureSolverKind::JacobiCg => {
                // classify, solve, project, residual
                4
            }
        }
    }
}

pub(crate) struct RbgsPressureSolver {
    pub(crate) classify_cells: wgpu::ComputePipeline,
    pub(crate) pressure_rbgs_red: wgpu::ComputePipeline,
    pub(crate) pressure_rbgs_black: wgpu::ComputePipeline,
    pub(crate) project_pressure: wgpu::ComputePipeline,
    pub(crate) pressure_residual: wgpu::ComputePipeline,
}

impl RbgsPressureSolver {
    pub(crate) fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, ctx: PressureContext) {
        // The current pressure scratch contract lives in the grid lanes:
        // classify_cells writes cell kind/divergence, RBGS updates pressure,
        // project_pressure consumes that pressure, and pressure_residual samples
        // the corrected velocity before packing reuses the same scratch lanes.
        for _ in 0..ctx.rbgs_pairs {
            pass.set_pipeline(&self.pressure_rbgs_red);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&self.pressure_rbgs_black);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
        }
    }
}

pub(crate) struct JacobiCgPressureSolver {
    pub(crate) classify_cells: wgpu::ComputePipeline,
    pub(crate) pressure_cg_init: wgpu::ComputePipeline,
    pub(crate) pressure_cg_matvec: wgpu::ComputePipeline,
    pub(crate) pressure_cg_apply_alpha: wgpu::ComputePipeline,
    pub(crate) pressure_cg_update_dir: wgpu::ComputePipeline,
    pub(crate) pressure_cg_finish_iteration: wgpu::ComputePipeline,
    pub(crate) project_pressure: wgpu::ComputePipeline,
    pub(crate) pressure_residual: wgpu::ComputePipeline,
}

impl JacobiCgPressureSolver {
    pub(crate) fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, ctx: PressureContext) {
        pass.set_pipeline(&self.pressure_cg_init);
        pass.dispatch_workgroups(ctx.cell_wg, 1, 1);

        for _ in 0..ctx.cg_iterations {
            pass.set_pipeline(&self.pressure_cg_matvec);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&self.pressure_cg_apply_alpha);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&self.pressure_cg_update_dir);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&self.pressure_cg_finish_iteration);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PressureSolverKind;

    #[test]
    fn pressure_solver_kind_parses_registered_solvers() {
        assert_eq!("rbgs".parse(), Ok(PressureSolverKind::Rbgs));
        assert_eq!("cg".parse(), Ok(PressureSolverKind::JacobiCg));
        assert_eq!("jacobi-cg".parse(), Ok(PressureSolverKind::JacobiCg));
        assert_eq!(
            PressureSolverKind::ALL,
            &[PressureSolverKind::Rbgs, PressureSolverKind::JacobiCg]
        );
    }
}

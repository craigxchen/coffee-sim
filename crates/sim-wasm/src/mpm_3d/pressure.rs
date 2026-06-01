use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PressureSolverKind {
    Rbgs,
}

impl PressureSolverKind {
    pub(crate) const ALL: &'static [Self] = &[Self::Rbgs];

    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Rbgs => "rbgs",
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
    pub cell_wg: u32,
    pub rbgs_pairs: u32,
}

pub(crate) enum PressurePipelines {
    Rbgs(RbgsPressureSolver),
}

impl PressurePipelines {
    #[cfg(test)]
    pub(crate) fn iterations_per_substep(&self, ctx: PressureContext) -> u32 {
        match self {
            Self::Rbgs(_) => ctx.rbgs_pairs,
        }
    }

    #[cfg(test)]
    pub(crate) fn estimated_timestamp_scopes(&self) -> u32 {
        // classify, solve, project, residual
        4
    }

    pub(crate) fn classify_pipeline(&self) -> &wgpu::ComputePipeline {
        match self {
            Self::Rbgs(solver) => &solver.classify_cells,
        }
    }

    pub(crate) fn project_pipeline(&self) -> &wgpu::ComputePipeline {
        match self {
            Self::Rbgs(solver) => &solver.project_pressure,
        }
    }

    pub(crate) fn residual_pipeline(&self) -> &wgpu::ComputePipeline {
        match self {
            Self::Rbgs(solver) => &solver.pressure_residual,
        }
    }

    pub(crate) fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, ctx: PressureContext) {
        match self {
            Self::Rbgs(solver) => solver.encode_solve(pass, ctx),
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

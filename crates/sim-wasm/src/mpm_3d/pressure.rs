use std::fmt;
use std::str::FromStr;

use super::state::{MpmBuffers, METRIC_PRESSURE_ACTIVE_WORKGROUPS_X_IDX};
use super::{MpmDispatch, MpmPassLabel, MpmScheduleOp};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PressureSolverKind {
    Rbgs,
    JacobiCg,
    SparseCg,
}

impl PressureSolverKind {
    pub(crate) const ALL: &'static [Self] = &[Self::Rbgs, Self::JacobiCg, Self::SparseCg];

    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Rbgs => "rbgs",
            Self::JacobiCg => "jacobi-cg",
            Self::SparseCg => "sparse-cg",
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
            "sparse-cg" | "mpm-sparse-cg" | "pressure-sparse-cg" => Ok(Self::SparseCg),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PressureOperatorKind {
    Collocated,
    Staggered,
}

impl PressureOperatorKind {
    pub(crate) const ALL: &'static [Self] = &[Self::Collocated];

    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::Collocated => "collocated",
            Self::Staggered => "staggered",
        }
    }
}

impl fmt::Display for PressureOperatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for PressureOperatorKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "collocated" | "cell-centered" | "cell_centered" => Ok(Self::Collocated),
            "staggered" => Err(
                "pressure operator 'staggered' is known from codex/perf-60hz-tier1 \
                 but is not ported to the modular pressure boundary yet"
                    .to_string(),
            ),
            other => Err(format!(
                "unknown pressure operator '{other}'; available operators: {}",
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
    pub operator: PressureOperatorKind,
    pub cell_wg: u32,
    pub rbgs_pairs: u32,
    pub cg_iterations: u32,
}

pub(crate) struct PressurePipelines {
    pub(crate) rbgs: RbgsPressureSolver,
    pub(crate) jacobi_cg: JacobiCgPressureSolver,
    pub(crate) staged_staggered: StagedStaggeredPressurePipelines,
}

impl PressurePipelines {
    #[cfg(test)]
    pub(crate) fn iterations_per_substep(&self, ctx: PressureContext) -> u32 {
        match ctx.kind {
            PressureSolverKind::Rbgs => ctx.rbgs_pairs,
            PressureSolverKind::JacobiCg | PressureSolverKind::SparseCg => ctx.cg_iterations,
        }
    }

    pub(crate) fn classify_pipeline_for(&self, ctx: PressureContext) -> &wgpu::ComputePipeline {
        match (ctx.operator, ctx.kind) {
            (PressureOperatorKind::Collocated, PressureSolverKind::Rbgs) => {
                &self.rbgs.classify_cells
            }
            (
                PressureOperatorKind::Collocated,
                PressureSolverKind::JacobiCg | PressureSolverKind::SparseCg,
            ) => &self.jacobi_cg.classify_cells,
            (PressureOperatorKind::Staggered, _) => &self.staged_staggered.classify_cells,
        }
    }

    pub(crate) fn project_pipeline_for(&self, ctx: PressureContext) -> &wgpu::ComputePipeline {
        match (ctx.operator, ctx.kind) {
            (PressureOperatorKind::Collocated, PressureSolverKind::Rbgs) => {
                &self.rbgs.project_pressure
            }
            (
                PressureOperatorKind::Collocated,
                PressureSolverKind::JacobiCg | PressureSolverKind::SparseCg,
            ) => &self.jacobi_cg.project_pressure,
            (PressureOperatorKind::Staggered, _) => &self.staged_staggered.project_pressure,
        }
    }

    pub(crate) fn residual_pipeline_for(&self, ctx: PressureContext) -> &wgpu::ComputePipeline {
        match (ctx.operator, ctx.kind) {
            (PressureOperatorKind::Collocated, PressureSolverKind::Rbgs) => {
                &self.rbgs.pressure_residual
            }
            (
                PressureOperatorKind::Collocated,
                PressureSolverKind::JacobiCg | PressureSolverKind::SparseCg,
            ) => &self.jacobi_cg.pressure_residual,
            (PressureOperatorKind::Staggered, _) => &self.staged_staggered.pressure_residual,
        }
    }

    pub(crate) fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, ctx: PressureContext) {
        match (ctx.operator, ctx.kind) {
            (_, PressureSolverKind::Rbgs) => self.rbgs.encode_solve(pass, ctx),
            (PressureOperatorKind::Collocated, PressureSolverKind::JacobiCg) => {
                self.jacobi_cg.encode_solve(pass, ctx)
            }
            (PressureOperatorKind::Collocated, PressureSolverKind::SparseCg) => {
                self.jacobi_cg.encode_sparse_solve(pass, ctx)
            }
            (PressureOperatorKind::Staggered, PressureSolverKind::JacobiCg) => self
                .staged_staggered
                .encode_cg_solve(pass, ctx, &self.jacobi_cg),
            (PressureOperatorKind::Staggered, PressureSolverKind::SparseCg) => {
                self.jacobi_cg.encode_sparse_solve(pass, ctx)
            }
        }
    }

    pub(crate) fn encode_solve_ops<'a, RunOp>(
        &'a self,
        buffers: &'a MpmBuffers,
        ctx: PressureContext,
        mut run_op: RunOp,
    ) where
        RunOp: FnMut(MpmScheduleOp<'a>),
    {
        match (ctx.operator, ctx.kind) {
            (PressureOperatorKind::Collocated, PressureSolverKind::SparseCg)
            | (PressureOperatorKind::Staggered, PressureSolverKind::SparseCg) => {
                self.jacobi_cg.encode_sparse_solve_ops(buffers, ctx, run_op)
            }
            (PressureOperatorKind::Staggered, PressureSolverKind::JacobiCg) => self
                .staged_staggered
                .encode_cg_solve_ops(ctx, &self.jacobi_cg, run_op),
            (_, PressureSolverKind::Rbgs)
            | (PressureOperatorKind::Collocated, PressureSolverKind::JacobiCg) => {
                run_op(MpmScheduleOp::PressureSolve {
                    label: MpmPassLabel::PressureSolve,
                    pressure: self,
                    ctx,
                })
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn estimated_timestamp_scopes(&self, ctx: PressureContext) -> u32 {
        match ctx.kind {
            PressureSolverKind::Rbgs => {
                // classify, solve, project, residual
                4
            }
            PressureSolverKind::JacobiCg => {
                // classify, solve, project, residual
                4
            }
            PressureSolverKind::SparseCg => {
                // classify, init, finalize, warmstart, four kernels per CG iteration,
                // project, residual. Buffer copies are not timestamped.
                6 + (4 * ctx.cg_iterations)
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

pub(crate) struct StagedStaggeredPressurePipelines {
    pub(crate) classify_cells: wgpu::ComputePipeline,
    pub(crate) pressure_cg_init: wgpu::ComputePipeline,
    pub(crate) pressure_cg_matvec: wgpu::ComputePipeline,
    pub(crate) project_pressure: wgpu::ComputePipeline,
    pub(crate) pressure_residual: wgpu::ComputePipeline,
}

impl StagedStaggeredPressurePipelines {
    pub(crate) fn encode_cg_solve(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        ctx: PressureContext,
        common: &JacobiCgPressureSolver,
    ) {
        pass.set_pipeline(&self.pressure_cg_init);
        pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
        pass.set_pipeline(&common.pressure_cg_warmstart);
        pass.dispatch_workgroups(ctx.cell_wg, 1, 1);

        for _ in 0..ctx.cg_iterations {
            pass.set_pipeline(&self.pressure_cg_matvec);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&common.pressure_cg_apply_alpha);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&common.pressure_cg_update_dir);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&common.pressure_cg_finish_iteration);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub(crate) fn encode_cg_solve_ops<'a, RunOp>(
        &'a self,
        ctx: PressureContext,
        common: &'a JacobiCgPressureSolver,
        mut run_op: RunOp,
    ) where
        RunOp: FnMut(MpmScheduleOp<'a>),
    {
        run_op(MpmScheduleOp::Pipeline {
            label: MpmPassLabel::PressureSolve,
            pipeline: &self.pressure_cg_init,
            dispatch: MpmDispatch::Direct(ctx.cell_wg),
        });
        run_op(MpmScheduleOp::Pipeline {
            label: MpmPassLabel::PressureSolve,
            pipeline: &common.pressure_cg_warmstart,
            dispatch: MpmDispatch::Direct(ctx.cell_wg),
        });

        for _ in 0..ctx.cg_iterations {
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &self.pressure_cg_matvec,
                dispatch: MpmDispatch::Direct(ctx.cell_wg),
            });
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &common.pressure_cg_apply_alpha,
                dispatch: MpmDispatch::Direct(ctx.cell_wg),
            });
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &common.pressure_cg_update_dir,
                dispatch: MpmDispatch::Direct(ctx.cell_wg),
            });
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &common.pressure_cg_finish_iteration,
                dispatch: MpmDispatch::Direct(1),
            });
        }
    }
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
    pub(crate) pressure_cg_warmstart: wgpu::ComputePipeline,
    pub(crate) pressure_cg_matvec: wgpu::ComputePipeline,
    pub(crate) pressure_cg_apply_alpha: wgpu::ComputePipeline,
    pub(crate) pressure_cg_update_dir: wgpu::ComputePipeline,
    pub(crate) pressure_cg_finish_iteration: wgpu::ComputePipeline,
    pub(crate) pressure_active_finalize_dispatch: wgpu::ComputePipeline,
    pub(crate) pressure_cg_sparse_matvec: wgpu::ComputePipeline,
    pub(crate) pressure_cg_sparse_apply_alpha: wgpu::ComputePipeline,
    pub(crate) pressure_cg_sparse_update_dir: wgpu::ComputePipeline,
    pub(crate) project_pressure: wgpu::ComputePipeline,
    pub(crate) pressure_residual: wgpu::ComputePipeline,
}

impl JacobiCgPressureSolver {
    pub(crate) fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, ctx: PressureContext) {
        pass.set_pipeline(&self.pressure_cg_init);
        pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
        pass.set_pipeline(&self.pressure_cg_warmstart);
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

    pub(crate) fn encode_sparse_solve(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        ctx: PressureContext,
    ) {
        pass.set_pipeline(&self.pressure_cg_init);
        pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
        pass.set_pipeline(&self.pressure_active_finalize_dispatch);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.pressure_cg_warmstart);
        pass.dispatch_workgroups(ctx.cell_wg, 1, 1);

        for _ in 0..ctx.cg_iterations {
            pass.set_pipeline(&self.pressure_cg_sparse_matvec);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&self.pressure_cg_sparse_apply_alpha);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&self.pressure_cg_sparse_update_dir);
            pass.dispatch_workgroups(ctx.cell_wg, 1, 1);
            pass.set_pipeline(&self.pressure_cg_finish_iteration);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub(crate) fn encode_sparse_solve_ops<'a, RunOp>(
        &'a self,
        buffers: &'a MpmBuffers,
        ctx: PressureContext,
        mut run_op: RunOp,
    ) where
        RunOp: FnMut(MpmScheduleOp<'a>),
    {
        run_op(MpmScheduleOp::Pipeline {
            label: MpmPassLabel::PressureSolve,
            pipeline: &self.pressure_cg_init,
            dispatch: MpmDispatch::Direct(ctx.cell_wg),
        });
        run_op(MpmScheduleOp::Pipeline {
            label: MpmPassLabel::PressureSolve,
            pipeline: &self.pressure_active_finalize_dispatch,
            dispatch: MpmDispatch::Direct(1),
        });
        run_op(MpmScheduleOp::Pipeline {
            label: MpmPassLabel::PressureSolve,
            pipeline: &self.pressure_cg_warmstart,
            dispatch: MpmDispatch::Direct(ctx.cell_wg),
        });
        run_op(MpmScheduleOp::CopyBufferToBuffer {
            src: &buffers.metrics,
            src_offset: (METRIC_PRESSURE_ACTIVE_WORKGROUPS_X_IDX * std::mem::size_of::<u32>())
                as u64,
            dst: &buffers.pressure_dispatch_args,
            dst_offset: 0,
            size: (3 * std::mem::size_of::<u32>()) as u64,
        });
        let sparse_dispatch = MpmDispatch::Indirect {
            buffer: &buffers.pressure_dispatch_args,
            offset: 0,
        };
        for _ in 0..ctx.cg_iterations {
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &self.pressure_cg_sparse_matvec,
                dispatch: sparse_dispatch,
            });
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &self.pressure_cg_sparse_apply_alpha,
                dispatch: sparse_dispatch,
            });
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &self.pressure_cg_sparse_update_dir,
                dispatch: sparse_dispatch,
            });
            run_op(MpmScheduleOp::Pipeline {
                label: MpmPassLabel::PressureSolve,
                pipeline: &self.pressure_cg_finish_iteration,
                dispatch: MpmDispatch::Direct(1),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PressureOperatorKind, PressureSolverKind};

    #[test]
    fn pressure_solver_kind_parses_registered_solvers() {
        assert_eq!("rbgs".parse(), Ok(PressureSolverKind::Rbgs));
        assert_eq!("cg".parse(), Ok(PressureSolverKind::JacobiCg));
        assert_eq!("jacobi-cg".parse(), Ok(PressureSolverKind::JacobiCg));
        assert_eq!("sparse-cg".parse(), Ok(PressureSolverKind::SparseCg));
        assert_eq!(
            PressureSolverKind::ALL,
            &[
                PressureSolverKind::Rbgs,
                PressureSolverKind::JacobiCg,
                PressureSolverKind::SparseCg
            ]
        );
    }

    #[test]
    fn pressure_operator_kind_parses_current_operator_and_rejects_unported_staggered() {
        assert_eq!("collocated".parse(), Ok(PressureOperatorKind::Collocated));
        assert_eq!(
            "cell-centered".parse(),
            Ok(PressureOperatorKind::Collocated)
        );
        assert_eq!(
            PressureOperatorKind::ALL,
            &[PressureOperatorKind::Collocated]
        );
        assert_eq!(PressureOperatorKind::Staggered.id(), "staggered");
        let err = "staggered"
            .parse::<PressureOperatorKind>()
            .expect_err("staggered operator is intentionally not runnable yet");
        assert!(err.contains("not ported"));
    }
}

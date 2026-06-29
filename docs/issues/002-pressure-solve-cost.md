# Reduce pressure solve cost without depending on sparse-grid work

## Tags

- `physics:solver-performance`
- `physics:numerics`

## Severity

High

## Problem

The pressure projection uses many red/black Gauss-Seidel pressure pairs per
substep. With 10 substeps, this multiplies quickly, and each pair runs red and
black passes over the grid. Sparse active-cell work is already in progress, so
this issue should focus on pressure-specific convergence and scheduling rather
than filing another sparse-grid task.

## Proposed fix

Evaluate pressure-specific improvements that are compatible with either dense or
future sparse execution:

1. Use residual metrics to adapt pressure iteration count more aggressively.
2. Investigate cheaper convergence accelerators, such as warm-starting pressure
   from the previous substep or over-relaxation.
3. Prototype a pressure solver variant with better convergence per pass, such as
   a coarse correction or multigrid-inspired path, if it can fit WebGPU binding
   and memory limits.
4. Keep pressure correctness observable through residual metrics and existing
   physics regression tests.

## Acceptance criteria

- Production frames perform fewer pressure passes when residuals are already
  acceptable.
- Pressure residual metrics remain stable or improve for canonical debug scenes.
- The fix does not depend on adding a second issue for active-cell/sparse-grid
  work.

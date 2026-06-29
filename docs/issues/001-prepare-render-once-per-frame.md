# Run `prepare_render` once per frame instead of once per substep

## Tags

- `physics:solver-performance`
- `ui:render-performance`
- `ui:diagnostics`

## Severity

High

## Problem

`MpmSim3D::step_frame` runs `prepare_render` inside the substep loop. The
default V60 settings use 10 substeps, but the browser renders only once after
`stepFrame` returns. The first nine `render_data` writes in a normal frame are
therefore overwritten before they can be displayed.

The pass also currently performs diagnostics atomics for active water, solute,
and cup totals, so the render-packing work and metrics work are coupled.

## Proposed fix

1. Move visual render packing out of the substep loop and run it once after all
   simulation substeps complete.
2. Split diagnostics accumulation from render packing before or during the move:
   - `prepare_render`: writes only visual instance data to `render_data`.
   - `accumulate_metrics` or equivalent: updates debug/coffee metrics at a
     deliberate cadence.
3. Preserve existing metric semantics explicitly. Choose whether metrics report
   final-frame state, last-substep state, or accumulated frame state, and encode
   that decision in names/tests.
4. Add regression coverage that verifies `render_data` is updated after the
   final substep and inactive particles are still hidden.

## Acceptance criteria

- `prepare_render` is dispatched once per `step_frame` call, not once per
  substep.
- Debug/TDS/extraction metrics remain available and documented.
- Rendering still uses the GPU-produced `render_data` buffer directly.
- Relevant tests and profiler output show no correctness regression.

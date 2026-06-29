# Decouple diagnostics from hot simulation/render passes

## Tags

- `ui:diagnostics`
- `physics:solver-performance`

## Severity

Medium

## Problem

Diagnostics metrics use GPU atomics and are intertwined with hot passes such as
render preparation. The browser throttles metric readback, but the GPU still
performs some metrics work during normal simulation steps.

## Proposed fix

1. Add an explicit metrics-enabled uniform or pass schedule.
2. Run expensive diagnostics only when the debug panel, timeseries drawer, or
   scripted evaluation needs them.
3. Split visual render packing from metrics accumulation.
4. Document whether metrics represent the final substep, whole frame, or last
   sampled frame.

## Acceptance criteria

- Normal non-debug frames avoid unnecessary diagnostics atomics.
- Debug/timeseries metrics remain exact enough for regression and evaluation.
- Metrics readback continues to avoid staging-buffer map races.

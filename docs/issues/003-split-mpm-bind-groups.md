# Split the monolithic MPM bind group by pass family

## Tags

- `physics:memory-layout`
- `ui:device-compatibility`

## Severity

High

## Problem

The current MPM compute bind group exposes almost all simulation buffers to all
compute pipelines. The project already requests a higher storage-buffer limit
because the pipeline is near adapter limits. This makes future solver work
fragile: adding one more storage buffer can break validation on constrained
WebGPU adapters.

## Proposed fix

Split compute resources by pass family instead of binding everything
everywhere:

1. Particle transfer bind group: particles, affine, grid, grid velocities.
2. Grid/pressure bind group: grid, grid velocities, classification/SDF, metrics.
3. Bed/extraction bind group: particles or bed particles, bed lookup, bed delta,
   bed extraction state.
4. Render-packing bind group: particles, affine, bed extraction state,
   `render_data`, metrics if still needed.

This may require splitting the WGSL shader into pass-family modules or using
multiple bind group layouts.

## Acceptance criteria

- No individual compute pipeline requires unnecessary storage buffers.
- Required WebGPU limits are equal to or lower than the current limit request.
- Pipeline validation tests cover the new layout.

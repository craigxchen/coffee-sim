# Move inflow emission onto the GPU

## Tags

- `physics:inflow`
- `physics:solver-performance`

## Severity

Medium

## Problem

Water inflow emission is currently CPU/Rust driven and writes particle state to
GPU buffers before each substep. That is simple and deterministic, but it keeps
particle allocation and emission on the CPU side of an otherwise GPU-resident
simulation loop.

## Proposed fix

Prototype GPU-side emission:

1. Keep spout parameters in uniforms or a small storage buffer.
2. Use a GPU atomic counter or free-list to allocate particle slots.
3. Emit particle, affine, and extraction/solute defaults directly in compute.
4. Report emitted and dropped counts through metrics/readback.

## Acceptance criteria

- Emission rate and particle placement match CPU emission within tolerances.
- Dropped-particle accounting remains visible to the UI.
- CPU queue writes per substep are reduced.

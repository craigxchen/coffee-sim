# Reduce fixed-point atomic P2G contention

## Tags

- `physics:solver-performance`
- `physics:numerics`
- `physics:memory-layout`

## Severity

High

## Problem

The P2G pass scatters each water particle to a 3x3x3 stencil and atomically adds
fixed-point mass, momentum, and volume lanes into the grid. This is portable in
WebGPU but expensive in dense regions, and it depends on fixed-point headroom to
avoid integer overflow.

## Proposed fix

Evaluate ways to reduce global atomic pressure without compromising MPM
behavior:

1. Particle binning by cell or tile so neighboring particles are processed more
   coherently.
2. Tile-local accumulation in workgroup memory followed by fewer global atomic
   writes.
3. Reduced or reorganized grid lanes if pressure/packing redesigns make some
   lanes less hot.
4. Explicit profiling before and after, with dense-bed and center-pour scenes.

## Acceptance criteria

- P2G produces equivalent mass/momentum transfer within test tolerances.
- Dense regions show lower P2G GPU time or lower atomic-overflow risk.
- Fixed-point bounds and overflow metrics remain documented.

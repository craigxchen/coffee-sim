# Issue Tags

This folder contains the source bodies for GitHub issues created with
`scripts/create-github-issues.sh`. Each issue lists tags from the categories
below.

## Physics-related

- `physics:solver-performance` — changes that reduce compute cost in core MPM
  transfer, grid, pressure, or scheduling paths without changing user-facing UI.
- `physics:numerics` — changes that affect pressure, viscosity, stability,
  fixed-point ranges, or solver convergence.
- `physics:memory-layout` — changes to GPU buffer layout, bind groups, particle
  layout, or capacity management that affect simulation throughput.
- `physics:inflow` — changes to spout emission, particle allocation, or inlet
  scheduling.

## Coffee-extraction modelling

- `coffee:bed-coupling` — changes to porous-bed lookup, bed-water exchange,
  bed impulse transfer, or bed motion.
- `coffee:extraction-state` — changes to solute, saturation, retained water, or
  extraction metrics/state ownership.

## UI/UX related

- `ui:render-performance` — changes that reduce visual rendering cost without
  changing simulation truth.
- `ui:diagnostics` — changes to debug metrics, readback cadence, profiling, or
  HUD/timeseries overhead.
- `ui:device-compatibility` — changes that improve behavior on constrained
  WebGPU adapters, integrated GPUs, or browser memory budgets.

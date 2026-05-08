# MPM Architecture

## Overview

The active simulator is the WebGPU MPM stack in
[`crates/sim-wasm/src/mpm_3d`](../crates/sim-wasm/src/mpm_3d).
The browser app is a WASM wrapper around that stack.

High-level ownership:
- `sim-core`: shared math/types used by the browser simulation
- `sim-wasm/src/lib.rs`: WASM-facing API and scene loaders
- `sim-wasm/src/renderer.rs`: render pipeline and camera controls
- `sim-wasm/src/mpm_3d/*`: simulation state, passes, scene setup, tests
- `sim-wasm/www-3d/*`: browser UI and scene controls

## Authoritative State

Simulation truth lives in the GPU simulation state, not the renderer.

Authoritative runtime state:
- water and bed particles: `particles`
- affine/APIC state: `affine`
- grid accumulators and velocities: `grid`, `grid_vel`
- bed auxiliary state: `bed_extract`, `bed_lookup`, `bed_delta`
- static field state: SDF texture, cached SDF classification texture
- static filter support geometry: CPU `filter_mesh` render vertices

Rendering is downstream of that state via `render_data`.

## Pass Order

Each frame is orchestrated by [`MpmSim3D::step_frame`](../crates/sim-wasm/src/mpm_3d/mod.rs).

Current pass shape:
1. update uniforms and scene-dependent state
2. emit inflow particles
3. clear hot grid buffers
4. rebuild bed lookup / clear bed-coupling scratch as needed
5. `p2g`
6. `grid_update`
7. `boundary_project`
8. `classify_cells`
9. pressure projection passes
10. `g2p`
11. bed-coupling / extraction-related passes
12. `prepare_render`

Important invariant:
- simulation passes own physical behavior
- UI, renderer, and debug HUD must not invent physical state

## Module Map

- `mod.rs`
  - top-level settings
  - scene presets
  - per-frame orchestration
  - buffer upload and readback helpers
- `state.rs`
  - `MpmUniforms`
  - buffer allocation
  - SDF and cached cell-solid classification generation
  - device limits and fixed-point constants
- `shader.rs`
  - WGSL kernels for particle-grid transfer, grid solve, coupling, and render packing
  - the hottest path and the main source of solver behavior
- `pipelines.rs`
  - pipeline and bind-group creation for the WGSL passes
- `inflow.rs`
  - kettle/spout parameterization
  - inflow state machine and particle emission
- `bed.rs`
  - bed particle initialization
  - bed property seeding and helper logic
- `filter.rs`
  - geometric filter configuration
- `filter_mesh.rs`
  - static filter support mesh and upload-ready vertex state
- `physics_tests.rs`
  - headless GPU regression tests against the MPM stack

## Buffer Ownership

Hot simulation buffers:
- `particles`
- `affine`
- `grid`
- `grid_vel`
- `bed_lookup`
- `bed_delta`

Medium-frequency state:
- `bed_extract`
- `metrics`
- filter mesh render vertices

Static/cold state:
- `sdf_texture`
- `sdf_class_texture`

Key ownership rule:
- per-particle long-lived material state should stay on particles unless profiling proves a grid cache is necessary
- per-cell hot state should be minimized and explicit

## Browser API Surface

`WasmSim3D` currently exposes:
- scene loading and reset
- stepping
- kettle-angle and spout controls
- camera manipulation
- simulation metrics for the UI/debug panel

The browser app in `www-3d/main.js` owns:
- scene selection
- sidebar controls
- animation loop
- debug stats toggles

The browser app must stay a thin controller over `WasmSim3D`, not a second simulation layer.

## Current Invariants

- `origin/main` is the authoritative mainline branch
- the WebGPU app in `www-3d` is the primary product surface
- tests in `physics_tests.rs` are the meaningful regression surface for the solver
- `CHANGELOG.md` must track shipped or snapshot-worthy behavior changes
- known realism gaps should be recorded explicitly rather than hidden behind heuristic tuning

## Known Architectural Gaps

- free-flight jet cohesion is still weak
- bed mechanics and bed hydraulics are still under active iteration
- filter contact is still an approximation rather than a full contact solve
- extraction remains provisional

For active planning and validation priorities, see [`docs/ROADMAP.md`](ROADMAP.md).

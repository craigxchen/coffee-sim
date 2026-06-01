# coffeesim — `utils/` Plan (shared infrastructure)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/utils/` · **Status:** plan (Phase 0 foundation)
**Depends on:** `wgpu`. Depended on by everything; depends on nothing internal.

## Purpose
The low-level infrastructure shared by solvers and the engine, owned by neither. Establishing this is most of Phase 0.

## Scope
**In:** GPU context/resources, the shared spatial hash, SDF helpers, kernel functions/math, seeded RNG. **Out:** physics (`models`), orchestration (`engine`), any solver dynamics.

## Components

### `gpu.rs` — `GpuContext`
Wraps `wgpu`: device + queue, buffer/bind-group/pipeline creation helpers, the surface for the dev window. Targets native (Metal/Vulkan/DX12) now; the **same code path compiles to WASM + WebGPU** later (no rewrite — the cross-platform payoff of `wgpu`). Requests the `TIMESTAMP_QUERY` feature when present (for `profiler`).

### `hash.rs` — shared spatial hash
A single uniform-grid hash, **binned once per step for all particle species** (cell = SPH support radius; counting/radix sort; cell-start/cell-count arrays). This is the cross-solver neighbor infrastructure — XPBD and SPH+MPM both query it for water-water, grain-grain, and water-grain neighbors. Not owned by any solver.

### `sdf.rs` — SDF helpers
Sample/gradient of a signed distance field; **position-level collision projection** (push a particle out along the gradient) for position-based solvers; baked dripper SDFs from `geometry/`. Robust, unconditionally stable boundary handling — the reason the primary avoids SPH boundary-particle machinery.

### `kernels.rs` — kernels + math
SPH/MPM smoothing kernels (cubic/quadratic B-spline, poly6, spiky) and gradients; small math helpers (quaternion/rotation, safe normalize, clamps). Shared so solvers don't re-derive them.

### `rng.rs` — seeded RNG
Deterministic, seeded PRNG (particle jitter, sampling). Determinism underpins reproducible runs and fair `compare_solvers` results.

## Build phases & gates
1. `GpuContext` + dev window. *Gate:* clears a frame; reports adapter + `TIMESTAMP_QUERY` availability.
2. `hash` with a sort + neighbor query. *Gate:* correct neighbor lists vs a brute-force check on a small set; profiled.
3. `sdf` sample/gradient + collision projection. *Gate:* particles collide correctly with a V60 SDF; no leakage at holes.
4. `kernels` + `rng`. *Gate:* kernel partition-of-unity / gradient checks pass; RNG reproducible across runs.

## Open questions
- Hash cell size vs MPM grid cell size compatibility (choose grid `Δx` as a clean fraction of `h` so grid solvers can reuse the same structure).
- Radix vs counting sort on the target GPUs (and the eventual mobile tiers) — measure with `profiler`.
- Buffer-layout conventions shared with `BrewStateView` (mandated `vec4<f32>` layouts for zero-cost vis interop — coordinate with `solvers.md`).

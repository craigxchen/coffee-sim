# Coffee Sim

`coffee-sim` is an interactive, real-time **pour-over coffee simulator** built in Rust with
`wgpu` + WGSL — water poured through a deformable coffee bed in a dripper, with believable
flow, bed deformation, and an extraction model whose causal levers (grind, temperature,
ratio, pour technique, freshness, evenness) drive the right outcomes.

It is organized as a **modular library of complete fluid-simulation solvers** behind one
seam (`Solver`), sharing one physics layer (`models`) so methods can be implemented and
compared fairly. One WGSL codebase targets native desktop now and WASM + WebGPU later.

> **Status: rewrite in progress.** Phase 0 (the scaffold/harness) is in place: the solver
> seam, the registry + runtime solver-switch, the headless `GpuContext`, and the local
> profiler. Solver physics (XPBD water core first) comes next. The previous MPM
> implementation it supersedes lives on `origin/*` branches.

## Quick Start

Prerequisites: [Rust](https://rustup.rs/) and a GPU with a Metal/Vulkan/DX12 backend.

```bash
git clone https://github.com/craigxchen/coffee-sim.git
cd coffee-sim
cargo run --example phase0_noop   # headless Phase 0 demo (skips if no GPU adapter)
```

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo run --example phase0_noop
```

## Project Layout

Single crate; the repo root is the workspace + package. Modular blocks under `src/`:

- `engine/` — orchestration: `Scene`, `Simulator`, `State`, the solver registry + catalog.
- `solvers/` — the `Solver` seam and each complete method (XPBD, MPM, SPH+MPM).
- `models/` — shared physics + materials, imported by every solver.
- `emission/` — particle emission (coffee + water).
- `ui/` — rendering, controls, overlays (consumes canonical state only).
- `profiling/` — local timestamp + dispatch profiler.
- `utils/` — `GpuContext`, spatial hash, SDF, kernels, RNG, geometry, config.

## More Info

- [Architecture](docs/ARCHITECTURE.md) — authoritative top-level design.
- [Sub-module plans](docs/plans/) — per-block detail.
- [KEEP.md](KEEP.md) — salvaged reference values (calibration, units, geometry) from v1.

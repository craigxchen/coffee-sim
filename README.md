# Coffee Sim

`coffee-sim` is an interactive, real-time **pour-over coffee simulator** built in Rust with
`wgpu` + WGSL — water poured through a deformable coffee bed in a dripper, with believable
flow, bed deformation, and an extraction model whose causal levers (grind, temperature,
ratio, pour technique, freshness, evenness) drive the right outcomes.

It is organized as a **modular library of complete fluid-simulation solvers** behind one
seam (`Solver`), sharing one physics layer (`models`) so methods can be implemented and
compared fairly. One WGSL codebase targets native desktop now and WASM + WebGPU later.

> **Status: solver development in progress.** XPBD, two-field, PB-MPM, and seam-blend
> experiments share a registry, renderer, and validation harness. The new `mac` solver
> is a CPU reference for coupled water/bed motion, absorption, and extraction.
> `mac_gpu` runs that model in WebGPU compute, with a coarse interactive V60 preset.
> The previous MPM implementation remains on `origin/*`.

## Quick Start

Prerequisites: [Rust](https://rustup.rs/) and a GPU with a Metal/Vulkan/DX12 backend.

```bash
git clone https://github.com/craigxchen/coffee-sim.git
cd coffee-sim
cargo run --example phase0_noop   # headless Phase 0 demo (skips if no GPU adapter)
cargo run --release --example mac_pourover # small MAC coffee+water reference brew (CPU)
```

## Web app

```bash
wasm-pack build --target web --out-dir www/pkg
python3 -m http.server 8080 --directory www
```

Open [the MAC WebGPU solver](http://localhost:8080/?solver=mac_gpu), or select
**MAC porous mixture (WebGPU)** in the solver dropdown. See the
[GPU backend notes](docs/plans/solver_mac_gpu.md) for validation and numerical limits.

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
- [MAC porous-mixture reference](docs/plans/solver_mac.md) — independent `mac` solver, coupling equations, configuration, and validation limits.
- [KEEP.md](KEEP.md) — salvaged reference values (calibration, units, geometry) from v1.

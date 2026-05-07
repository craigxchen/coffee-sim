# Coffee Sim

[![CI](https://github.com/craigxchen/coffee-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/craigxchen/coffee-sim/actions/workflows/ci.yml)
[![Long Horizon Diagnostics](https://github.com/craigxchen/coffee-sim/actions/workflows/long-horizon.yml/badge.svg)](https://github.com/craigxchen/coffee-sim/actions/workflows/long-horizon.yml)

`coffee-sim` is a browser-first pour-over coffee simulator written in Rust.
The active product is the WebGPU-powered 3D MPM demo in
[`crates/sim-wasm/www-3d`](crates/sim-wasm/www-3d).

## Current State

The current mainline architecture is:
- an MLS-MPM-style free-water solver running on WebGPU
- a coffee-bed material layer coupled on the same simulation grid
- a V60 + filter + carafe browser scene with live controls and debug stats

What is implemented today:
- browser-native WASM app with `wgpu` and `wasm-bindgen`
- pressure-projected water solver under [`mpm_3d`](crates/sim-wasm/src/mpm_3d)
- kettle-driven inflow with adjustable angle and spout placement
- filter collision scaffold and bed coupling groundwork
- headless GPU physics tests for regression coverage

What is still in progress:
- more realistic free-flight jet cohesion
- more physical bed deformation, compaction, and drawdown
- richer filter / drainage / extraction behavior

## Repository Layout

```text
coffee-sim/
├── crates/
│   ├── sim-core/          # Shared math/types used by the browser solver
│   └── sim-wasm/          # WebGPU renderer + WASM-facing simulation app
│       ├── src/
│       │   ├── lib.rs
│       │   ├── renderer.rs
│       │   └── mpm_3d/
│       │       ├── mod.rs
│       │       ├── state.rs
│       │       ├── shader.rs
│       │       ├── pipelines.rs
│       │       ├── inflow.rs
│       │       ├── bed.rs
│       │       ├── filter.rs
│       │       ├── filter_mesh.rs
│       │       └── physics_tests.rs
│       └── www-3d/
├── docs/
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── CHANGELOG.md
└── AGENTS.md
```

## Running The Web App

Prerequisites:
- Rust toolchain via `rustup`
- `wasm-pack`
- a browser with WebGPU enabled

Build the WASM bundle:

```bash
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

Serve the app locally:

```bash
cd crates/sim-wasm/www-3d
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

Controls:
- drag to orbit
- scroll to zoom
- `W/A/S/D` to translate the camera
- scene buttons to switch between default, free-stream, and center-pour presets
- pause / reset / debug toggle in the sidebar
- kettle-angle and spout controls to steer the inflow

## Development Notes

The main simulation entrypoints are:
- [`lib.rs`](crates/sim-wasm/src/lib.rs): WASM-facing API and scene loaders
- [`renderer.rs`](crates/sim-wasm/src/renderer.rs): WebGPU rendering and camera behavior
- [`mod.rs`](crates/sim-wasm/src/mpm_3d/mod.rs): settings, stepping, and pass orchestration
- [`state.rs`](crates/sim-wasm/src/mpm_3d/state.rs): buffers, uniforms, and static field generation
- [`shader.rs`](crates/sim-wasm/src/mpm_3d/shader.rs): WGSL simulation passes

Useful checks:

```bash
cargo test -p coffee-sim-wasm --lib
cargo check -p coffee-sim-wasm --target wasm32-unknown-unknown
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

Before landing a meaningful branch, also run:

```bash
cargo fmt --check
cargo clippy -p coffee-sim-wasm -- -D warnings
```

## Continuous Integration

GitHub Actions runs the standard quality gate on pushes to `main` and on pull
requests:

```bash
cargo fmt --check
cargo clippy -p coffee-sim-wasm -- -D warnings
cargo test -p coffee-sim-wasm --lib
cargo check -p coffee-sim-wasm --target wasm32-unknown-unknown
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

The fast unit lane sets `COFFEE_SIM_SKIP_GPU_TESTS=1` so PR feedback does not
depend on a slow software GPU. CI also runs a small headless GPU smoke test.
Long-horizon physics diagnostics run on a nightly schedule and can be launched
manually from Actions. These use accelerated headless stepping, not browser
wall-clock playback. The known free-surface shape target is reported there as a
diagnostic until the pooled-water collapse issue is fixed.

## Documentation

Use these files as the current source of truth:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): as-built implementation map
- [`docs/ROADMAP.md`](docs/ROADMAP.md): current physics, validation, and performance direction
- [`CHANGELOG.md`](CHANGELOG.md): shipped or snapshot-worthy behavior changes

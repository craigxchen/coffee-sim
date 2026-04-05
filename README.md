# Coffee Sim

`coffee-sim` is a browser-first coffee brewing simulator written in Rust. The
current focus is an interactive 3D pour-over prototype: a controllable kettle
stream, a deformable coffee bed, a V60/carafe scene, and a GPU simulation path
that runs in the browser through WebGPU.

## Current State

The main experience today is the 3D WebGPU app in
[`crates/sim-wasm/www-3d`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/www-3d),
powered by the MPM-based simulation code in
[`crates/sim-wasm/src/mpm_3d`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d).

What works today:

- Browser-native 3D simulation with Rust + `wgpu` + `wasm-bindgen`
- Adjustable kettle angle driving flow rate and exit speed
- Water stream entering a V60 scene with boundary collision
- Coffee-bed particle layer with initial wetting / deformation response
- Carafe accumulation and particle-based rendering
- Orbit / zoom camera controls and live simulation stats

What is still in progress:

- More realistic coffee-bed disruption and drawdown behavior
- Better scene-boundary authoring beyond the current primitive setup
- Physically richer water-bed coupling and extraction modeling
- More faithful pooled-water rendering than simple particle billboards

## Repository Layout

```text
coffee-sim/
├── crates/
│   ├── sim-core/          # Shared Rust simulation/math code
│   ├── sim-python/        # Python bindings / validation hooks
│   └── sim-wasm/          # Browser-facing WASM app
│       ├── src/
│       │   ├── lib.rs
│       │   ├── renderer.rs
│       │   └── mpm_3d/
│       │       ├── mod.rs
│       │       ├── state.rs
│       │       ├── shader.rs
│       │       ├── pipelines.rs
│       │       ├── inflow.rs
│       │       └── bed.rs
│       └── www-3d/
├── examples/
├── python/
├── tests/
├── validation/
└── PLAN.md
```

## Running The 3D App

Prerequisites:

- Rust toolchain via [rustup](https://rustup.rs/)
- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/)
- A browser with WebGPU enabled

Build the WASM bundle:

```bash
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

Serve the app locally:

```bash
cd crates/sim-wasm/www-3d
python3 -m http.server 8080
```

Then open [http://localhost:8080](http://localhost:8080).

Controls:

- Drag to orbit
- Scroll to zoom
- Use the kettle-angle slider to change the stream
- Use pause / reset to inspect behavior

## Development Notes

The current browser simulation is centered on the `mpm_3d` module:

- [`mod.rs`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/mod.rs):
  simulation settings, stepping, and pass orchestration
- [`state.rs`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/state.rs):
  GPU buffers, uniforms, and scene SDF generation
- [`shader.rs`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/shader.rs):
  WGSL compute passes for P2G / grid update / G2P / bed coupling
- [`inflow.rs`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/inflow.rs):
  spout emission and flow-rate mapping
- [`bed.rs`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/mpm_3d/bed.rs):
  coffee-bed particle initialization
- [`renderer.rs`](/Users/cxc/Github/coffee-sim/crates/sim-wasm/src/renderer.rs):
  WebGPU rendering and camera behavior

Useful checks:

```bash
cargo check -p coffee-sim-wasm
cargo check -p coffee-sim-wasm --target wasm32-unknown-unknown
```

## Planning

The longer-term direction is tracked in [PLAN.md](/Users/cxc/Github/coffee-sim/PLAN.md).
That document is the living design note for the browser-native MPM path, scene
boundaries, coffee-bed modeling, and future extraction work.

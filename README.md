# Coffee Extraction Simulator

A physically-based coffee brewing simulator in Rust. The long-term goal is a
browser-based interactive 3D experience where you control a gooseneck kettle and
pour water over coffee grounds in real time. The immediate focus is getting the
fluid dynamics right.

## What Works Today

### 3D WebGPU Fluid Simulation

A real-time SPH (Smoothed Particle Hydrodynamics) fluid simulation running
entirely on the GPU via WebGPU compute shaders. A blob of ~4000 water particles
spawns above a V60 cone, falls under gravity, and flows through the funnel into
a carafe below.

```bash
# Build the WASM package
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg

# Serve locally
cd crates/sim-wasm/www-3d
python3 -m http.server 8080
# Open http://localhost:8080
```

Controls: drag to orbit, scroll to zoom, pause/play/reset buttons.

### 2D Browser Demo

A simpler 2D cross-section view with scripted pour recipes (spiral, center,
pulse, edge-heavy).

```bash
wasm-pack build crates/sim-wasm --target web --release --out-dir pkg
cd crates/sim-wasm/www
python3 -m http.server 8080
```

## Architecture

```
coffee-sim/
├── crates/
│   ├── sim-core/           # Pure Rust simulation library (no I/O, no rendering)
│   │   └── src/
│   │       ├── lib.rs      #   2D CPU particle sim (ParticleSim) with wall/drain collision
│   │       ├── sph.rs      #   SPH kernels, Vec2/Vec3, spatial hashing
│   │       ├── pour.rs     #   Pour script representation (timed pour commands)
│   │       └── constants.rs#   Physical constants
│   │
│   ├── sim-python/         # PyO3 bindings (for validation/plotting)
│   │   └── src/lib.rs
│   │
│   └── sim-wasm/           # Browser targets (wasm-bindgen)
│       └── src/
│           ├── lib.rs      #   WasmSim (2D) + WasmSim3D entry points
│           ├── gpu_sim_3d.rs#  3D WebGPU compute SPH simulation
│           └── renderer.rs #   3D WebGPU renderer (billboard particles + wireframe)
│
├── python/                 # Python orchestration & visualization
│   ├── sim.py, config.py, viz.py, presets.py, analysis.py
│
├── validation/             # Physics validation scripts
│   └── validate_1d.py
│
└── examples/               # Example scripts
    ├── v60_pourover.py
    ├── espresso_9bar.py
    └── comparison.py
```

### Key Design Principle

`sim-core` has zero dependencies on PyO3, wasm-bindgen, or rendering. It compiles
to native (for testing), WASM (for the browser), and links into Python (via PyO3).
All three targets share identical physics code.

## How the 3D Simulation Works

The 3D sim (`gpu_sim_3d.rs`) is a direct port of
[Sebastian Lague's Fluid-Sim](https://github.com/SebLague/Fluid-Sim) to Rust/WebGPU.
It runs the full SPH pipeline in WGSL compute shaders:

1. **External forces** — apply gravity, compute predicted positions
2. **Spatial hashing** — clear grid, assign particles to cells
3. **Density** — accumulate SPH density from neighbors (spiky kernel)
4. **Pressure** — compute pressure force from density gradient (symmetric formulation)
5. **Viscosity** — smooth velocity field (poly6 kernel)
6. **Position update** — integrate velocity, then handle collisions
7. **Render data** — copy positions to vertex-readable buffer

Steps 1–6 repeat `iterations_per_frame` times (default 3) per display frame.

### Collision / Boundaries

Particle collision is handled directly in the WGSL shader with hardcoded geometry:

- **V60 truncated cone** — interior constraint, particles stay inside. The cone
  narrows from radius 4.5 at the top to 0.8 at the bottom over a height of 6 units.
- **Carafe cylinder** — radius 3.0, sits below the cone.
- **Outer bounding box** — safety bounds that contain the full scene.

The wireframe overlay is rendered separately using the same geometry parameters.

### SPH Parameters

The simulation uses the same proven parameter set as the upstream Fluid-Sim:

| Parameter | Value |
|---|---|
| Smoothing radius | 0.24 |
| Target density | 430 |
| Pressure multiplier | 230 |
| Near pressure multiplier | 2.0 |
| Viscosity strength | 0.004 |
| Gravity | -10 |
| Collision damping | 0.95 |
| Iterations/frame | 3 |

Particles spawn in a 4×4×4 cube centered at (0, 6, 0), above the cone opening.

## Build Instructions

### Prerequisites

- Rust toolchain ([rustup](https://rustup.rs/))
- `wasm-pack` (`cargo install wasm-pack`)
- Python >= 3.12 + [uv](https://docs.astral.sh/uv/) (for validation scripts)

### WASM (3D browser sim)

```bash
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
cd crates/sim-wasm/www-3d && python3 -m http.server 8080
```

### Python bindings (validation)

```bash
uv venv && source .venv/bin/activate
uv pip install numpy matplotlib maturin pytest
cd crates/sim-python && maturin develop --release
```

### Tests

```bash
cargo test --workspace
```

## License

See LICENSE file for details.

# Coffee Sim

`coffee-sim` is a browser-based pour-over coffee simulation. It runs a
Rust/WebGPU MPM solver through WebAssembly and renders a V60-style scene with
live pour controls.

## Quick Start

Prerequisites:

- Rust: <https://rustup.rs/>
- Python 3, for a simple local file server
- a browser with WebGPU support

Clone the repo and install the WebAssembly tooling:

```bash
git clone https://github.com/craigxchen/coffee-sim.git
cd coffee-sim
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --locked
```

Build the browser bundle:

```bash
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

Serve the app:

```bash
cd crates/sim-wasm/www-3d
python3 -m http.server 8080
```

Open <http://localhost:8080>.

## Controls

- Drag to orbit the camera.
- Scroll to zoom.
- Use `W/A/S/D` to pan.
- Use the scene, kettle-angle, and spout controls in the sidebar.
- Use pause, reset, and debug from the sidebar when needed.

## More Info

- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Changelog](CHANGELOG.md)

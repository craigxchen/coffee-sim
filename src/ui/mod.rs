//! Rendering + controls + debug overlays. Solver-agnostic: consumes ONLY the canonical
//! state (`ParticleBuffers`/`Metrics`/`Profile`) and never writes simulation state (one-way
//! data flow). The `wgpu`/WGSL renderer is portable; windowing/event-loop is native (lives
//! in the app/example).
//!
//! v1: a windowed sphere-impostor particle renderer with a CAD orbit camera and an
//! orientation cube. Screen-space fluid / scorecard / debug overlays come later.

pub mod camera;
pub mod render;
pub mod wireframe;

pub use camera::OrbitCamera;
pub use render::Renderer;
pub use wireframe::{solid_wireframe, LineVertex};

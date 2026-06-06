//! `coffee-sim` — an interactive, real-time pour-over coffee simulator.
//!
//! Organized as a modular library of complete fluid-simulation solvers behind one seam
//! (`solvers::Solver`), sharing one physics layer (`models`) so methods compare fairly.
//! See `docs/ARCHITECTURE.md` for the top-level design and `docs/plans/` for sub-modules.
//!
//! This is the Phase 0 scaffold: the harness everything hangs off (no solver physics).

pub mod emission;
pub mod engine;
pub mod models;
pub mod profiling;
pub mod solvers;
pub mod ui;
pub mod utils;
pub mod web_controls;

#[cfg(target_arch = "wasm32")]
mod web;

// Seam re-exports for ergonomic top-level use.
pub use emission::{EmissionInput, PourEvent};
pub use engine::registry::{build_solver, info_for, Catalog, SolverId};
pub use engine::{Metrics, Scene, Simulator, State};
pub use models::Materials;
pub use profiling::{Profile, Profiler};
pub use solvers::base::{Paradigm, Solver, SolverInfo, Stability};
pub use utils::buffers::ParticleBuffers;
pub use utils::config::Config;
pub use utils::gpu::GpuContext;

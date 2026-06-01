//! Scene-level orchestration: build a brew, drive the **one** active solver through the
//! frame loop, and emit canonical state/metrics/profile for `ui` and `profiling`. Owns no
//! physics. Also home to the solver registry + the solver-description catalog.

pub mod registry;
pub mod scene;
pub mod simulator;
pub mod state;

pub use scene::Scene;
pub use simulator::Simulator;
pub use state::{Metrics, State};

//! The canonical per-frame snapshot decoupling solvers from rendering/profiling.

use crate::profiling::Profile;
use crate::utils::buffers::ParticleBuffers;

/// Brew-outcome metrics. Zeros for a no-op solver.
#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub extraction_yield: f32,
    pub tds: f32,
    pub drawdown_time: f32,
    pub evenness: f32,
    pub particle_count: u32,
    pub iteration_count: u32,
}

/// What `ui` and `profiling` see each frame — never a solver's internals, which is what
/// makes them solver-agnostic.
#[derive(Clone, Default)]
pub struct State {
    pub particles: ParticleBuffers,
    pub metrics: Metrics,
    pub profile: Profile,
}

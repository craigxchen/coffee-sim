//! The seam: the `Solver` trait every complete method implements, plus the data types
//! that describe a solver. Descriptions are loaded from `engine/solvers.json` — the trait
//! deliberately does **not** carry them, so adding/amending a solver is a data edit.

use serde::Deserialize;

use crate::emission::EmissionInput;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

/// Numerical family of a solver. Data only — drives no behavior in the seam.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Paradigm {
    PositionBased,
    ForceBased,
    Hybrid,
}

/// Stability class. `CflLimited` carries the CFL number `c`.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum Stability {
    Unconditional,
    CflLimited { c: f32 },
}

/// A solver's description, deserialized from `engine/solvers.json`. Never returned by the
/// trait — so the catalog is the single place solver metadata lives.
#[derive(Deserialize, Clone, Debug)]
pub struct SolverInfo {
    pub name: String,
    pub paradigm: Paradigm,
    pub owns_grid: bool,
    pub stability: Stability,
}

/// The contract every complete coffee+water method implements, so `engine`, `ui`, and
/// `profiling` drive any solver uniformly.
///
/// Contract:
/// - `step()` advances a **whole frame**; internal substepping, coupling, and extraction
///   are the solver's business.
/// - `step()` must be **deterministic** given identical state + input + seed.
/// - `step()` must never leave the sim in a non-finite state.
/// - `particles()`/`metrics()`/`profile()` are read-only and cheap; they must not trigger
///   a GPU sync that stalls the frame (read the prior frame's resolved results).
pub trait Solver {
    /// Build the solver for a scene. Not callable on a trait object (`Self: Sized`); the
    /// registry dispatches per `SolverId`.
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self
    where
        Self: Sized;

    /// Reset to the scene's initial state without rebuilding GPU resources.
    fn reset(&mut self, scene: &Scene);

    /// Advance one frame.
    fn step(&mut self, dt: f32, input: &EmissionInput);

    /// Canonical particle buffers for rendering.
    fn particles(&self) -> ParticleBuffers;

    /// Brew-outcome metrics (yield, TDS, drawdown, evenness).
    fn metrics(&self) -> Metrics;

    /// Per-pass GPU timestamps + dispatches-per-frame.
    fn profile(&self) -> Profile;
}

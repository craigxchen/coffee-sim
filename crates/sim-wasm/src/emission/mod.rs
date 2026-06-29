//! Particle emission (coffee + water) as its own block, so emitting each species can be
//! tested independently to verify their physics in isolation.
//!
//! `EmissionInput` is the per-frame control handed to the active solver; [`pour`] is the recipe
//! layer (time-windowed flow + spatial pattern) a driver samples to produce it.

pub mod pour;

/// A discrete pour event delivered alongside the continuous controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PourEvent {
    #[default]
    None,
    StartBloom,
    StopBloom,
    Reset,
}

/// Per-frame pour/emission control input fed to `Solver::step`.
#[derive(Clone, Copy, Debug)]
pub struct EmissionInput {
    /// Kettle spout position (world space).
    pub kettle_pos: [f32; 3],
    /// Volumetric flow rate (scene units / s).
    pub flow_rate: f32,
    /// Pour angle (radians).
    pub pour_angle: f32,
    /// Discrete event for this frame, if any.
    pub event: PourEvent,
}

impl Default for EmissionInput {
    fn default() -> Self {
        Self {
            kettle_pos: [0.0, 0.0, 0.0],
            flow_rate: 0.0,
            pour_angle: 0.0,
            event: PourEvent::None,
        }
    }
}

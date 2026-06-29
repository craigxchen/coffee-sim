//! The canonical, solver-agnostic particle buffers exposed to `ui` and `profiling`.

use std::sync::Arc;

use wgpu::Buffer;

/// Handles to a solver's particle state in a **mandated canonical layout**, so `ui` can
/// render any solver's output with zero conversion regardless of the solver's internal
/// representation. A no-op solver returns the empty default (`particle_count == 0`, all
/// handles `None`).
///
/// `Arc<Buffer>` so the same GPU buffer can be shared by the solver, `ui`, and overlays
/// without copies.
#[derive(Clone, Default)]
pub struct ParticleBuffers {
    pub particle_count: u32,
    pub position: Option<Arc<Buffer>>,
    pub velocity: Option<Arc<Buffer>>,
    pub phase_tag: Option<Arc<Buffer>>,
    pub concentration: Option<Arc<Buffer>>,
    pub temperature: Option<Arc<Buffer>>,
    pub moisture: Option<Arc<Buffer>>,
    /// Optional sampled fields for debug overlays.
    pub alpha_s: Option<Arc<Buffer>>,
    pub pressure: Option<Arc<Buffer>>,
}

impl ParticleBuffers {
    /// The empty view a no-op (or freshly-built) solver returns.
    pub fn empty() -> Self {
        Self::default()
    }
}

//! Rendering + controls + debug overlays. Solver-agnostic: consumes ONLY the canonical
//! state (`ParticleBuffers` / `Metrics` / `Profile`) and **never writes simulation state**
//! — a strict one-way data flow (solver → `State` → `ui`/`profiling`).
//!
//! Phase 0: stub. The renderer + window/surface arrive in the vis phase (Phase 0 runs
//! headless).

/// Render the current brew state.
pub fn render() {
    todo!("rendering + window surface land with the vis phase")
}

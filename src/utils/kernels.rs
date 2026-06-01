//! SPH/MPM smoothing kernels (cubic/quadratic B-spline, poly6, spiky) and gradients,
//! plus small math helpers — shared so solvers don't re-derive them.
//!
//! Phase 0: stub. Kernels land with the XPBD water core. The validated quadratic
//! B-spline weights from v1 are recorded in `KEEP.md` §3.

/// Cubic B-spline smoothing kernel `W(r, h)`.
pub fn cubic_bspline(_r: f32, _h: f32) -> f32 {
    todo!("kernels land with the XPBD water core")
}

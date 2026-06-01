//! Signed-distance-field helpers: sample/gradient + position-level collision projection
//! (push a particle out along the gradient) for position-based solvers — robust,
//! unconditionally stable boundary handling, which is why the primary avoids SPH
//! boundary-particle machinery.
//!
//! Phase 0: stub. The real implementation (and baked dripper SDFs from `geometry`) lands
//! with the boundary-handling work. Reference math to re-validate is in `KEEP.md` §3.

/// Sample the signed distance to the nearest surface at world position `p`.
pub fn sample(_p: [f32; 3]) -> f32 {
    todo!("SDF sampling lands with the boundary/geometry phase")
}

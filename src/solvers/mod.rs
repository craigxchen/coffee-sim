//! The modular library of complete fluid-simulation solvers. Each solver owns its
//! internal coupling and extraction; the seam lives in [`base`]. Solver *descriptions*
//! are data (`engine/solvers.json`), not trait code.

pub mod base;
pub mod noop;
pub mod pass_recorder;
pub mod pbmpm;
pub mod twofield;
pub mod xpbd;

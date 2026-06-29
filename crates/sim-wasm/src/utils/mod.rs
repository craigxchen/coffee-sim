//! Shared infrastructure + config, owned by no block: the `GpuContext`, the shared
//! uniform-grid spatial hash, SDF helpers, kernels, seeded RNG, geometry builders, and
//! the canonical `ParticleBuffers`. Depends on `wgpu`; depends on nothing internal.

pub mod buffers;
pub mod config;
pub mod geometry;
pub mod gpu;
pub mod hash;
pub mod kernels;
pub mod rng;
pub mod sdf;

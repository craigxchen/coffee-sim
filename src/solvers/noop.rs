//! Two zero-physics solvers used to validate the harness end to end: they exercise the
//! `Solver` seam, the registry, and the runtime solver-switch (between `noop_a`/`noop_b`).
//! They do nothing each step and expose empty state — but they own a [`Profiler`] and
//! drive it per frame, demonstrating the intended solver-owns-its-profiler pattern.

use crate::emission::EmissionInput;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::{Profile, Profiler};
use crate::solvers::base::Solver;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

/// No-op solver A — does nothing; empty state.
pub struct NoopSolverA {
    profiler: Profiler,
}

impl Solver for NoopSolverA {
    fn build(_scene: &Scene, _mats: &Materials, _cfg: &Config, gpu: &GpuContext) -> Self {
        Self {
            profiler: Profiler::new(gpu.timestamps_supported),
        }
    }
    fn reset(&mut self, _scene: &Scene) {}
    fn step(&mut self, _dt: f32, _input: &EmissionInput) {
        self.profiler.begin_frame();
        // No dispatches, no passes — the harness must still report a (zeroed) profile.
    }
    fn particles(&self) -> ParticleBuffers {
        ParticleBuffers::empty()
    }
    fn metrics(&self) -> Metrics {
        Metrics::default()
    }
    fn profile(&self) -> Profile {
        self.profiler.snapshot()
    }
}

/// No-op solver B — identical behavior, different description (see `engine/solvers.json`),
/// so the runtime solver-switch can be exercised between two variants.
pub struct NoopSolverB {
    profiler: Profiler,
}

impl Solver for NoopSolverB {
    fn build(_scene: &Scene, _mats: &Materials, _cfg: &Config, gpu: &GpuContext) -> Self {
        Self {
            profiler: Profiler::new(gpu.timestamps_supported),
        }
    }
    fn reset(&mut self, _scene: &Scene) {}
    fn step(&mut self, _dt: f32, _input: &EmissionInput) {
        self.profiler.begin_frame();
    }
    fn particles(&self) -> ParticleBuffers {
        ParticleBuffers::empty()
    }
    fn metrics(&self) -> Metrics {
        Metrics::default()
    }
    fn profile(&self) -> Profile {
        self.profiler.snapshot()
    }
}

//! `Simulator` — drives the **one** active solver through the frame loop and pumps
//! canonical state out to `ui`/`profiling`. Owns the `GpuContext`; rebuilds a different
//! solver on the **same** `Scene` for the runtime solver-switch.

use crate::emission::EmissionInput;
use crate::engine::registry::{build_solver, info_for, SolverId};
use crate::engine::scene::Scene;
use crate::engine::state::State;
use crate::models::Materials;
use crate::solvers::base::{Solver, SolverInfo};
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

/// Holds the active solver + everything needed to rebuild it on the same scene.
pub struct Simulator {
    gpu: GpuContext,
    scene: Scene,
    materials: Materials,
    config: Config,
    active_id: SolverId,
    active_info: SolverInfo,
    solver: Box<dyn Solver>,
    frame: u64,
}

impl Simulator {
    /// Build a simulator with `id` as the initial active solver.
    pub fn new(
        gpu: GpuContext,
        scene: Scene,
        materials: Materials,
        config: Config,
        id: SolverId,
    ) -> Self {
        let solver = build_solver(id, &scene, &materials, &config, &gpu);
        let active_info = info_for(id);
        Self {
            gpu,
            scene,
            materials,
            config,
            active_id: id,
            active_info,
            solver,
            frame: 0,
        }
    }

    pub fn active_id(&self) -> SolverId {
        self.active_id
    }

    pub fn active_info(&self) -> &SolverInfo {
        &self.active_info
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Advance one frame. The solver does its own internal substepping; we just pass the
    /// frame `dt`, then pull the canonical snapshot.
    pub fn step(&mut self, dt: f32, input: &EmissionInput) -> State {
        self.solver.step(dt, input);
        self.frame += 1;
        State {
            particles: self.solver.particles(),
            metrics: self.solver.metrics(),
            profile: self.solver.profile(),
        }
    }

    /// Rebuild a different solver on the **same** scene — the runtime solver-switch.
    pub fn switch_solver(&mut self, id: SolverId) {
        self.solver = build_solver(id, &self.scene, &self.materials, &self.config, &self.gpu);
        self.active_id = id;
        self.active_info = info_for(id);
    }
}

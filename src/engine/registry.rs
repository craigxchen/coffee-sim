//! The solver registry + the embedded solver-description catalog.
//!
//! Adding a solver is three small, local edits: one row in `solvers.json`, one `SolverId`
//! variant, and one arm in [`build_solver`] — never a new fork. This is the seam that
//! prevents the per-solver branch sprawl that forced v1's modular rewrite.

use std::collections::HashMap;

use crate::engine::scene::Scene;
use crate::models::Materials;
use crate::solvers::base::{Solver, SolverInfo};
use crate::solvers::noop::{NoopSolverA, NoopSolverB};
use crate::solvers::xpbd::XpbdSolver;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

/// Solver-description catalog, embedded at build time.
const SOLVERS_JSON: &str = include_str!("solvers.json");

/// Identifies a registered solver. `id()` is its key in `solvers.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverId {
    NoopA,
    NoopB,
    Xpbd,
}

impl SolverId {
    pub fn id(self) -> &'static str {
        match self {
            SolverId::NoopA => "noop_a",
            SolverId::NoopB => "noop_b",
            SolverId::Xpbd => "xpbd",
        }
    }

    /// Every registered solver, for UI dropdowns and comparison sweeps.
    pub fn all() -> &'static [SolverId] {
        &[SolverId::NoopA, SolverId::NoopB, SolverId::Xpbd]
    }
}

/// Parsed solver descriptions keyed by `SolverId::id()`.
pub struct Catalog {
    entries: HashMap<String, SolverInfo>,
}

impl Catalog {
    /// Parse the embedded catalog. Panics on malformed JSON — a build-time invariant.
    pub fn load() -> Self {
        let entries: HashMap<String, SolverInfo> =
            serde_json::from_str(SOLVERS_JSON).expect("engine/solvers.json is valid JSON");
        Self { entries }
    }

    /// Look up a solver's description. Panics if a `SolverId` has no catalog row — a
    /// startup invariant (every registered solver must be described).
    pub fn info(&self, id: SolverId) -> SolverInfo {
        self.entries
            .get(id.id())
            .unwrap_or_else(|| panic!("solvers.json missing an entry for {:?} ({})", id, id.id()))
            .clone()
    }
}

/// Look up a solver's description without building it.
pub fn info_for(id: SolverId) -> SolverInfo {
    Catalog::load().info(id)
}

/// Build the active solver for `id` on the given scene (powers the runtime solver-switch).
pub fn build_solver(
    id: SolverId,
    scene: &Scene,
    mats: &Materials,
    cfg: &Config,
    gpu: &GpuContext,
) -> Box<dyn Solver> {
    match id {
        SolverId::NoopA => Box::new(NoopSolverA::build(scene, mats, cfg, gpu)),
        SolverId::NoopB => Box::new(NoopSolverB::build(scene, mats, cfg, gpu)),
        SolverId::Xpbd => Box::new(XpbdSolver::build(scene, mats, cfg, gpu)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::base::Paradigm;

    #[test]
    fn catalog_parses_and_every_solver_id_resolves() {
        let catalog = Catalog::load();
        for &id in SolverId::all() {
            // Panics if any registered solver is missing a catalog row.
            let info = catalog.info(id);
            assert!(!info.name.is_empty(), "{id:?} has an empty name");
        }
    }

    #[test]
    fn noop_variants_have_distinct_descriptions() {
        // The runtime switch must observe a real difference between the two variants.
        let a = info_for(SolverId::NoopA);
        let b = info_for(SolverId::NoopB);
        assert_eq!(a.paradigm, Paradigm::PositionBased);
        assert_eq!(b.paradigm, Paradigm::ForceBased);
        assert_ne!(a.paradigm, b.paradigm);
        assert_ne!(a.owns_grid, b.owns_grid);
    }
}

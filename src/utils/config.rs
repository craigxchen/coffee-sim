//! Solver/simulation configuration.

/// Declarative configuration handed to a solver at build time.
///
/// Carries the deterministic `seed` — determinism is load-bearing for reproducible runs
/// and fair cross-solver comparison. Unit-calibration constants (see `KEEP.md` §2) land
/// here as the sim grows.
#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0x00C0_FFEE_5EED,
        }
    }
}

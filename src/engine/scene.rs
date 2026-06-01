//! `Scene` — the user-facing brew description, immutable input to `Solver::build`/`reset`.
//!
//! Phase 0: a minimal placeholder. Real fields (dripper geometry SDF, dose/ratio,
//! water temperature, the pour schedule) arrive with `geometry` and `models`; reference
//! recipe values are in `KEEP.md` §1.

/// Immutable brew description handed to a solver at build/reset.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    /// Coffee dose (grams).
    pub dose_g: f32,
    /// Brew water (milliliters).
    pub water_ml: f32,
}

impl Scene {
    /// A default V60 brew (reference recipe; see `KEEP.md` §1).
    pub fn v60() -> Self {
        Self {
            dose_g: 15.0,
            water_ml: 250.0,
        }
    }
}

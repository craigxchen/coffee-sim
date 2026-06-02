//! `Scene` — the user-facing brew description, immutable input to `Solver::build`/`reset`.
//!
//! Phase 1.1 carries what the PBF water core needs: the simulation box, an initial water
//! block (the dam), and gravity. Real V60 dripper geometry / pour schedule arrive with
//! `geometry`; reference recipe values are in `KEEP.md` §1.

/// Which material a scene seeds. Scenes are single-species for now (water-only OR grain-only):
/// mixed water+grain needs interphase coupling (a later step), without which the two would just
/// interpenetrate. The solver shares one particle system + grid across both, tagged per particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Water,
    Grain,
}

/// Immutable brew/scene description handed to a solver at build/reset.
#[derive(Clone, Debug)]
pub struct Scene {
    /// Coffee dose (grams) — placeholder for the recipe-driven scenes.
    pub dose_g: f32,
    /// Brew water (milliliters) — placeholder for the recipe-driven scenes.
    pub water_ml: f32,

    /// Gravity vector (scene units / s²).
    pub gravity: [f32; 3],
    /// Axis-aligned simulation domain (inclusive bounds).
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
    /// Axis-aligned region seeded with particles at build/reset (water dam, or grain column).
    pub water_block_min: [f32; 3],
    pub water_block_max: [f32; 3],
    /// Which material the seed block is (selects the solver's per-frame pass sequence).
    pub species: Species,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [0.0, 0.0, 0.0],
            box_max: [32.0, 40.0, 32.0],
            // A 16³ block against one wall — a dam-break that's lively but not a torture
            // column (a too-tall column over-compresses the pool and triggers squeeze-out
            // eruptions). Settles to a shallow pool over the floor.
            water_block_min: [1.0, 1.0, 8.0],
            water_block_max: [17.0, 17.0, 24.0],
            species: Species::Water,
        }
    }
}

impl Scene {
    /// A default V60 brew (reference recipe; see `KEEP.md` §1).
    pub fn v60() -> Self {
        Self {
            dose_g: 15.0,
            water_ml: 250.0,
            ..Self::default()
        }
    }

    /// A dam-break: a tall water column released in a box. The water-core gate scene.
    pub fn dam_break() -> Self {
        Self::default()
    }

    /// A dry-bed drop: a grain block released from a height onto the flat floor. The fall/impact
    /// energy lets the grains flow and settle into a stable heap at a believable angle of repose
    /// (a pre-built column would instead stand metastably under friction). The granular-bed gate.
    pub fn bed_drop() -> Self {
        Self {
            // A wider floor than the water box so the heap settles without ever touching a wall
            // (wall contact would confine the pile and corrupt the repose). A ~12-wide block,
            // centered, dropped from ~14 units up: it free-falls, hits the floor, and spreads.
            box_max: [48.0, 40.0, 48.0],
            water_block_min: [18.0, 14.0, 18.0],
            water_block_max: [30.0, 28.0, 30.0],
            species: Species::Grain,
            ..Self::default()
        }
    }
}

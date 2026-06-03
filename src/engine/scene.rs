//! `Scene` — the user-facing brew description, immutable input to `Solver::build`/`reset`.
//!
//! Carries the simulation box, gravity, and a list of **seed regions** (each a block of one
//! species). Single-species scenes have one region; the coupling scenes seed both a grain bed and
//! a water column. Real V60 dripper geometry / pour schedule arrive with `geometry`; reference
//! recipe values are in `KEEP.md` §1.

/// Which material a seed region is. The solver shares one particle system + grid across both,
/// tagged per particle by `phase`; the per-frame pass sequence is chosen from which species are present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Water,
    Grain,
}

/// An axis-aligned block seeded with one species at build/reset.
#[derive(Clone, Copy, Debug)]
pub struct SeedRegion {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub species: Species,
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
    /// Regions seeded with particles at build/reset (one per species block).
    pub regions: Vec<SeedRegion>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [0.0, 0.0, 0.0],
            box_max: [32.0, 40.0, 32.0],
            // A 16³ water block against one wall — a dam-break that's lively but not a torture
            // column (a too-tall column over-compresses the pool and triggers squeeze-out
            // eruptions). Settles to a shallow pool over the floor.
            regions: vec![SeedRegion {
                min: [1.0, 1.0, 8.0],
                max: [17.0, 17.0, 24.0],
                species: Species::Water,
            }],
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

    /// A dry-bed drop: a low, wide grain block poured onto the flat floor — it slumps into a
    /// stable static heap. The granular-bed gate scene. Wider floor so the heap never touches a
    /// wall (wall contact would confine the pile and corrupt the repose).
    pub fn bed_drop() -> Self {
        Self {
            box_max: [48.0, 40.0, 48.0],
            regions: vec![SeedRegion {
                min: [15.0, 2.0, 15.0],
                max: [33.0, 12.0, 33.0],
                species: Species::Grain,
            }],
            ..Self::default()
        }
    }

    /// A pour-over: a grain bed on the floor with a water column above it — the first **mixed**
    /// water+grain scene, the water/bed-coupling gate. The bed settles, then water pools on /
    /// drains through it (drainage needs the drag step; exclusion alone keeps water off the floor).
    pub fn pour_over() -> Self {
        Self {
            box_max: [48.0, 40.0, 48.0],
            regions: vec![
                // Grain bed: a low wide block that settles into a bed (~7 tall, well clear of walls).
                SeedRegion {
                    min: [15.0, 2.0, 15.0],
                    max: [33.0, 12.0, 33.0],
                    species: Species::Grain,
                },
                // Water column centered above the bed, released onto it.
                SeedRegion {
                    min: [19.0, 18.0, 19.0],
                    max: [29.0, 32.0, 29.0],
                    species: Species::Water,
                },
            ],
            ..Self::default()
        }
    }
}

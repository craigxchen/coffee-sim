//! `Scene` — the user-facing brew description, immutable input to `Solver::build`/`reset`.
//!
//! Carries the simulation box, gravity, a list of **seed regions** (each a block of one species),
//! and any static **solids** (analytic SDF geometry — a V60 dripper, a cup) the particles collide
//! with. Single-species scenes have one region and no solids; the coupling scenes seed both a grain
//! bed and a water column; the V60 scene adds dripper geometry. Reference recipe values are in
//! `KEEP.md` §1.

pub use crate::utils::sdf::{SdfPrimitive, SolidKind};

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
    /// Water delivered by a continuous pour over the brew (mL). `0` = no pour (water comes only from
    /// the seeded regions; the solver sizes its particle pool to exactly the seed). `> 0` opts a scene
    /// into pour emission and sizes the pool with headroom for the dosed water. (Phase: pour emission.)
    pub pour_water_ml: f32,

    /// Gravity vector (scene units / s²).
    pub gravity: [f32; 3],
    /// Axis-aligned simulation domain (inclusive bounds).
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
    /// Regions seeded with particles at build/reset (one per species block).
    pub regions: Vec<SeedRegion>,
    /// Static solid geometry (analytic SDF) the particles collide with — empty for the AABB-only
    /// scenes. When non-empty, seeding rejects lattice points that fall outside a solid's cavity for
    /// their species (so a bed starts inside the dripper, not through its wall).
    pub solids: Vec<SdfPrimitive>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 0.0,
            pour_water_ml: 0.0,
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
            solids: Vec::new(),
        }
    }
}

impl Scene {
    /// A V60 brew: the dripper geometry (support cone + grains-only filter + cup) with a coffee bed
    /// seeded inside the cone and a water column above it. Origin-centered domain. Bed/water blocks
    /// are seeded generously and **cone-aware rejection** (in `seed_block`) trims them to the cavity,
    /// so no grain starts through the filter wall. Dimensions are reference values (`KEEP.md` §2/§4).
    pub fn v60() -> Self {
        Self {
            dose_g: 15.0,
            water_ml: 250.0,
            pour_water_ml: 0.0, // fixed-column V60; the continuous-pour variant is `v60_pour()`
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            regions: vec![
                // Coffee bed: a block spanning the lower cone; rejection trims it to the (grains-only)
                // filter cavity so the bed is cone-shaped and starts inside the paper.
                SeedRegion {
                    min: [-2.5, -2.8, -2.5],
                    max: [2.5, 0.2, 2.5],
                    species: Species::Grain,
                },
                // Water column above the bed; rejection trims it to the support-cone cavity.
                SeedRegion {
                    min: [-2.0, 0.6, -2.0],
                    max: [2.0, 2.6, 2.0],
                    species: Species::Water,
                },
            ],
        }
    }

    /// A V60 **pour** brew: the dripper geometry + a seeded coffee bed, with water arriving via a
    /// continuous pour (no seeded water column). `declares_pour()` is true (`pour_water_ml > 0`), so
    /// the solver sizes its particle pool with headroom for the dose and runs the water passes; a
    /// driver feeds `EmissionInput` from a pour recipe. Same bed/geometry as [`Scene::v60`].
    pub fn v60_pour() -> Self {
        Self {
            dose_g: 15.0,
            water_ml: 0.0,
            pour_water_ml: 250.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Coffee bed only; water is poured in. Rejection trims it to the filter cavity.
            regions: vec![SeedRegion {
                min: [-2.5, -2.8, -2.5],
                max: [2.5, 0.2, 2.5],
                species: Species::Grain,
            }],
        }
    }

    /// A V60 **pour into the empty dripper** — the same support cone + grains-only filter + cup
    /// geometry as [`Scene::v60_pour`] but with **NO coffee bed**: water is poured into the clean
    /// cone and drains through the filter apex into the cup (the filter only blocks grains, so it
    /// passes water freely). `declares_pour()` is true, so the live velocity/spout controls drive the
    /// pour. Lets you watch the pour stream + drainage geometry without the grounds.
    pub fn v60_pour_water_only() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 0.0,
            pour_water_ml: 250.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            regions: vec![], // water only — no coffee bed
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

    /// A dam-break against a porous **sand wall**: a tall water column on the left collapses and
    /// surges into a vertical slab of grains spanning the box. Because there is open space on the
    /// *far* side, water actually flows **through** the wall (the far side is the outlet — no drain
    /// hole needed), driven by the surge + the head that builds behind the wall. Showcases the
    /// coupling: percolation through a porous barrier, with the wall eroding/holding under the load.
    pub fn dam_through_sand() -> Self {
        Self {
            box_min: [0.0, 0.0, 0.0],
            box_max: [28.0, 22.0, 10.0],
            regions: vec![
                // Vertical sand wall across the box. With coarse grains (grain_diameter > the water
                // spacing) this is a few big grains thick, with pores the fine water threads.
                SeedRegion {
                    min: [14.0, 0.0, 0.0],
                    max: [20.0, 14.0, 10.0],
                    species: Species::Grain,
                },
                // Water dam on the left, released toward the wall (builds head to drive flow).
                SeedRegion {
                    min: [1.0, 1.0, 1.0],
                    max: [11.0, 14.0, 9.0],
                    species: Species::Water,
                },
            ],
            ..Self::default()
        }
    }

    /// A pour-over: a grain bed on the floor with a water column above it — the first **mixed**
    /// water+grain scene, the water/bed-coupling gate. The bed settles, then water pools on /
    /// drains through it under the porosity field and Darcy drag.
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

    /// Whether this scene injects water over the brew (pour emission). When true the solver sizes its
    /// particle pool with headroom for `pour_water_ml`; when false the pool is exactly the seed.
    pub fn declares_pour(&self) -> bool {
        self.pour_water_ml > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v60_has_geometry_enclosed_by_the_domain() {
        let s = Scene::v60();
        assert_eq!(s.solids.len(), 3, "support cone + grains-only filter + cup");
        // The domain encloses the cup floor (-8) and the cone top (+3).
        assert!(
            s.box_min[1] <= -8.0,
            "domain floor reaches below the cup floor"
        );
        assert!(s.box_max[1] >= 3.0, "domain top reaches above the cone top");
        assert!(!s.regions.is_empty(), "v60 seeds a bed + a water column");
    }
}

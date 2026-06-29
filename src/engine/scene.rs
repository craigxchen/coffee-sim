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

    /// DEBUG SCENE (ported from v1's `cup-volume` / `hydrostatic-column` debug scenes): the V60
    /// cup pre-filled STATICALLY with water at rest, **no pour** — water is seeded directly in the
    /// cup cavity (rejection trims the slab to the cup) and just settles under gravity. The
    /// static-fill counterpart to `v60_pour_water_only`: same geometry and (web) resolution, so it
    /// isolates whether cup over-compression is pour-jet-driven (this scene stays at rest density)
    /// or confinement/fill-driven (this scene over-compresses too). The slab fills the cup
    /// cross-section ~1.6 units deep, sized to land near the poured cup's settled particle count.
    pub fn v60_cup_static_full() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 250.0,
            pour_water_ml: 0.0, // STATIC: no pour, water is seeded directly in the cup
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Full-cross-section water slab inside the cup (floor -8, rim -3.5, radius 3); rejection
            // trims it to the cup cavity. Depth chosen so the settled count ≈ the poured cup's.
            regions: vec![SeedRegion {
                min: [-2.85, -7.85, -2.85],
                max: [2.85, -6.2, 2.85],
                species: Species::Water,
            }],
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

    // === Debug-scene catalog (ported by INTENT from the `main` MPM branch's DebugScene enum) ===
    //
    // Two families:
    //  * Filter/bed scenes use the full V60 dripper (support cone + grains-only filter + cup), like
    //    `v60`/`v60_pour`. Water is seeded inside the cone (cone-aware rejection trims it).
    //  * Cup-water scenes isolate the cup: they keep the same dripper geometry (so the cup wall/floor
    //    are present) but seed water ONLY inside the cup cavity (rim -3.5, floor -8, radius 3) with no
    //    bed — the rewrite analogue of main's `cup_only_water_scene` (filter/bed stripped).
    // Where a `main` scene is deeply MPM-specific (per-cell pressure-solve tuning), only the geometry/
    // seed intent is ported; solver-tuning knobs live in `web::setup_for`, not the `Scene`.

    /// DEBUG (`filter-water-block`): a still block of water resting inside the filter cone over the
    /// coffee bed, no pour — the static filter-confinement counterpart to [`Scene::v60`]. Same V60
    /// geometry + bed; the water column just settles. (main: `seed_filter_water_block`.)
    pub fn debug_filter_water_block() -> Self {
        Self::v60()
    }

    /// DEBUG (`off-center-filter-wall-pour`): the V60 pour brew, but the live spout is parked toward
    /// the filter wall (off-axis) so the stream runs down one side. Same geometry/bed as
    /// [`Scene::v60_pour`]; `web::setup_for` parks the spout off-center.
    pub fn debug_off_center_filter_wall_pour() -> Self {
        Self::v60_pour()
    }

    /// DEBUG (`seeded-paper-wall-sheet`): a thin sheet of water clinging to the filter wall, no pour —
    /// tests wall contact / sheet drainage on the grains-only filter. Approximated as a tall, thin
    /// water slab offset against one side of the filter cone (rejection trims it to the cavity); the
    /// bed is dropped so the sheet drains freely. (main: `seed_paper_wall_sheet`, an arc of particles
    /// hugging the wall.)
    pub fn debug_seeded_paper_wall_sheet() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 60.0,
            pour_water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Thin slab offset to +x, hugging the filter wall; rejection trims it to the cavity so it
            // starts as a sheet on the wall rather than a centered column.
            regions: vec![SeedRegion {
                min: [0.6, -1.0, -2.2],
                max: [2.4, 2.2, 2.2],
                species: Species::Water,
            }],
        }
    }

    /// DEBUG (`filter-apex-drain`): water seeded low in the filter cone near the apex with NO bed —
    /// it drains straight through the support-cone outlet into the cup. Same dripper as
    /// [`Scene::v60_pour_water_only`] minus the pour; a small low water plug seeds the drain.
    /// (main: `seed_filter_apex_drain`.)
    pub fn debug_filter_apex_drain() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 40.0,
            pour_water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // A small plug low in the cone (just above the apex); rejection trims it to the cavity.
            regions: vec![SeedRegion {
                min: [-1.2, -2.6, -1.2],
                max: [1.2, -0.4, 1.2],
                species: Species::Water,
            }],
        }
    }

    /// DEBUG (`cup-wall-floor-corner-contact`): cup-only water resting in a wedge against one wall +
    /// the floor — the wall/floor corner-contact stability case. Seeds a water slab biased to +x and
    /// low in the cup cavity (rejection trims it to the cylinder). No bed, no pour.
    /// (main: `seed_cup_wall_floor_corner_contact`.)
    pub fn debug_cup_wall_floor_corner_contact() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 120.0,
            pour_water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Slab in the +x half of the cup floor (corner against the +x wall); cup rejection trims.
            regions: vec![SeedRegion {
                min: [0.3, -7.85, -2.85],
                max: [2.85, -6.4, 2.85],
                species: Species::Water,
            }],
        }
    }

    /// DEBUG (`asymmetric-cup-mound-settle`): cup-only water seeded as an off-center mound that
    /// settles into a level pool — tests asymmetric free-surface relaxation. Approximated as a water
    /// slab offset to one quadrant of the cup floor (rejection trims it to the cylinder). No bed/pour.
    /// (main: `seed_asymmetric_cup_mound`, an offset ellipsoid.)
    pub fn debug_asymmetric_cup_mound_settle() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 120.0,
            pour_water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Off-center mound: a taller slab biased to the (+x, -z) quadrant; cup rejection trims it.
            regions: vec![SeedRegion {
                min: [-0.4, -7.85, -2.85],
                max: [2.85, -4.8, 0.6],
                species: Species::Water,
            }],
        }
    }

    /// DEBUG (`hydrostatic-column`): cup-only water as a tall, narrow on-axis column — the classic
    /// hydrostatic-pressure / no-spurious-drift check. Seeds a thin centered column in the cup cavity.
    /// No bed, no pour. (main: `seed_hydrostatic_column`.)
    pub fn debug_hydrostatic_column() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 120.0,
            pour_water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Narrow centered column (radius ~1) standing tall in the cup; cup rejection trims it.
            regions: vec![SeedRegion {
                min: [-1.0, -7.85, -1.0],
                max: [1.0, -4.0, 1.0],
                species: Species::Water,
            }],
        }
    }

    /// DEBUG (`dam-break-slosh`): cup-only water filling one half of the cup, released so it surges
    /// across and sloshes back. The cup analogue of [`Scene::dam_break`]. Seeds a half-cup slab biased
    /// to -x in the cup cavity (rejection trims it). No bed, no pour.
    /// (main: `seed_dam_break_slosh`.)
    pub fn debug_dam_break_slosh() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 120.0,
            pour_water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Half-cup dam in the -x half, ~2.3 units tall; cup rejection trims it to the cylinder.
            regions: vec![SeedRegion {
                min: [-2.85, -7.85, -2.85],
                max: [-0.2, -5.5, 2.85],
                species: Species::Water,
            }],
        }
    }

    /// DEBUG (`sparse-free-jet`): a thin, slow pour into the EMPTY cup — a sparse free stream that
    /// falls and pools. Same dripper as [`Scene::v60_pour_water_only`] (no bed); `web::setup_for` runs
    /// it with a thin nozzle + low velocity so the stream stays sparse. (main: `debug_sparse_free_jet`.)
    pub fn debug_sparse_free_jet() -> Self {
        Self::v60_pour_water_only()
    }

    /// DEBUG (`high-velocity-jet-impact`): a fast pour plunging onto a shallow seeded pool in the cup —
    /// the jet-impact / crater case. Reuses the cup pool seed (like [`Scene::v60_cup_static_full`]) AND
    /// declares a pour so the (fast) emitter drives the impact. No bed.
    /// (main: `seed_high_velocity_jet_impact_pool` + a high-speed inflow.)
    pub fn debug_high_velocity_jet_impact() -> Self {
        Self {
            dose_g: 0.0,
            water_ml: 100.0,
            pour_water_ml: 150.0, // declares a pour so the fast jet emitter runs; headroom for inflow
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            // Shallow target pool on the cup floor; cup rejection trims it to the cylinder.
            regions: vec![SeedRegion {
                min: [-2.85, -7.85, -2.85],
                max: [2.85, -6.9, 2.85],
                species: Species::Water,
            }],
        }
    }

    /// DEBUG (`uniform-bed-saturation`): the full V60 bed pre-seeded with water filling its pore
    /// space (a water column co-located with the bed) — the uniform-saturation / drainage case. Same
    /// geometry/bed as [`Scene::v60`]; the water column spans the bed instead of resting above it.
    /// No pour. (main: `seed_uniform_bed_saturation`.)
    pub fn debug_uniform_bed_saturation() -> Self {
        Self {
            dose_g: 15.0,
            water_ml: 250.0,
            pour_water_ml: 0.0,
            gravity: [0.0, -20.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            solids: crate::utils::geometry::v60_dripper(),
            regions: vec![
                // Coffee bed (same block as v60; rejection trims to the filter cavity).
                SeedRegion {
                    min: [-2.5, -2.8, -2.5],
                    max: [2.5, 0.2, 2.5],
                    species: Species::Grain,
                },
                // Water seeded over the SAME y-span as the bed, so it starts inside the pore space
                // (uniform saturation) rather than as a column resting on top.
                SeedRegion {
                    min: [-2.5, -2.8, -2.5],
                    max: [2.5, 0.4, 2.5],
                    species: Species::Water,
                },
            ],
        }
    }

    /// DEBUG (`permeability-comparison`): the V60 pour brew with a TIGHTER bed (lower permeability) so
    /// water ponds and drains slowly — the permeability/drawdown stress case. Same geometry/bed/pour
    /// as [`Scene::v60_pour`]; `web::setup_for` lowers the bed porosity + stiffens the Darcy drag.
    /// (main: `debug_permeability_comparison`, Kozeny–Carman at a finer grind.)
    pub fn debug_permeability_comparison() -> Self {
        Self::v60_pour()
    }

    /// DEBUG (`particle-capacity-stress`): the V60 pour brew driven at a high flow rate to push the
    /// particle pool — the capacity/throughput stress case. Same geometry/bed as [`Scene::v60_pour`]
    /// with a larger `pour_water_ml` budget so the pool is sized big; `web::setup_for` runs a fat,
    /// fast nozzle. (main: `debug_particle_capacity_stress`, max_particles 32k + high inflow.)
    pub fn debug_particle_capacity_stress() -> Self {
        Self {
            pour_water_ml: 500.0, // big pool headroom for the high-flow stress
            ..Self::v60_pour()
        }
    }

    /// DEBUG (`sand-wall`, NEW — no `main` equivalent): a plain box with a vertical wall of GRAINS on
    /// one side and a block of water on the other, released to surge into/through the wall. Tests
    /// water/solid coupling (percolation + the wall eroding/holding). Reuses the
    /// [`Scene::dam_through_sand`] recipe at a comparison footprint; no pour, no cup geometry.
    pub fn sand_wall() -> Self {
        Self {
            gravity: [0.0, -20.0, 0.0],
            box_min: [0.0, 0.0, 0.0],
            box_max: [30.0, 22.0, 12.0],
            regions: vec![
                // Vertical sand column spanning the box depth, set right of center.
                SeedRegion {
                    min: [16.0, 0.0, 0.0],
                    max: [22.0, 15.0, 12.0],
                    species: Species::Grain,
                },
                // Water block on the left, released toward the wall (builds head to drive flow).
                SeedRegion {
                    min: [1.0, 1.0, 1.0],
                    max: [12.0, 15.0, 11.0],
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

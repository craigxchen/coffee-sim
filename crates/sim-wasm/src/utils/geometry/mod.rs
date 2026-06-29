//! Dripper / filter / cup geometry presets, built from analytic SDF primitives (`utils::sdf`).
//!
//! Geometry is **analytic**, not a baked SDF texture (a circular cone on a Cartesian grid aliases —
//! v1's baked 128³ texture trapped water on the wall). A preset returns a list of `SdfPrimitive`s the
//! solver evaluates per particle. The V60 cone dimensions and bed-seating math salvaged from v1 are
//! recorded in `KEEP.md` §4 — treat them as reference values to re-validate against the new solver's
//! own contact gates (`KEEP.md` §5), not gospel.

use crate::utils::sdf::{SdfPrimitive, SolidKind, MASK_ALL, MASK_GRAIN};
use glam::Vec3;

/// Build the V60 dripper geometry: a rigid support cone (open apex outlet, both species), a
/// grains-only filter-paper cone (closed tip — traps grounds, passes water), and a catch cup.
///
/// Dimensions are scene units (`KEEP.md` §2/§4); cone radii are **outer** wall radii (the cavity
/// surface is `outer − thickness`, clamped to `hole_radius`). The filter sits nested inside the
/// support cone; water exits the support apex outlet and falls into the cup.
pub fn v60_dripper() -> Vec<SdfPrimitive> {
    vec![
        // Rigid support cone: outer apex radius 0.42 (= outlet) at y=-3 → outer top radius 4.6834 at
        // y=+3. Open apex outlet (water funnels through; grains never reach it — the filter catches
        // them). Blocks both species on the wall.
        SdfPrimitive {
            kind: SolidKind::Cone {
                center: Vec3::ZERO,
                apex_y: -3.0,
                top_y: 3.0,
                apex_r: 0.42,
                top_r: 4.6834,
                thickness: 0.05,
                hole_radius: 0.42,
                apex_open: true,
            },
            species_mask: MASK_ALL,
            friction: 0.3,
        },
        // Filter paper: grains-only, closed tip (KEEP.md §4, filter center (0,-0.35,0) → absolute
        // y-extent [-3.37, +2.40], outer top radius 4.10, tip radius 0, thickness 0.08). The closed
        // tip traps grounds; water ignores it (not in the mask) and threads the pores.
        SdfPrimitive {
            kind: SolidKind::Cone {
                center: Vec3::ZERO,
                apex_y: -3.37,
                top_y: 2.40,
                apex_r: 0.0,
                top_r: 4.10,
                thickness: 0.08,
                hole_radius: 0.0,
                apex_open: false,
            },
            species_mask: MASK_GRAIN,
            friction: 0.6,
        },
        // Catch cup: an open-top cylinder below the dripper that water pools into.
        SdfPrimitive {
            kind: SolidKind::Cylinder {
                center: Vec3::ZERO,
                floor_y: -8.0,
                rim_y: -3.5,
                radius: 3.0,
            },
            species_mask: MASK_ALL,
            friction: 0.2,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sdf::nearest;

    const WATER: u32 = 0;
    const GRAIN: u32 = 1;

    #[test]
    fn v60_has_support_cone_filter_and_cup() {
        let solids = v60_dripper();
        assert_eq!(solids.len(), 3);
        // support cone: open outlet, both species
        match solids[0].kind {
            SolidKind::Cone { apex_open, .. } => assert!(apex_open, "support cone outlet is open"),
            _ => panic!("solid 0 should be the support cone"),
        }
        assert_eq!(solids[0].species_mask, MASK_ALL);
        // filter: closed tip, grains only
        match solids[1].kind {
            SolidKind::Cone {
                apex_open, top_r, ..
            } => {
                assert!(!apex_open, "filter tip is closed");
                assert!(
                    (top_r - 4.10).abs() < 1e-4,
                    "filter top radius from KEEP §4"
                );
            }
            _ => panic!("solid 1 should be the filter cone"),
        }
        assert_eq!(solids[1].species_mask, MASK_GRAIN, "filter is grains-only");
        // cup
        assert!(matches!(solids[2].kind, SolidKind::Cylinder { .. }));
    }

    #[test]
    fn centerline_is_inside_for_both_species() {
        let solids = v60_dripper();
        let p = Vec3::new(0.0, 0.0, 0.0); // on the axis, mid-cone
        assert!(
            nearest(&solids, p, WATER).signed > 0.0,
            "water inside the cavity"
        );
        assert!(
            nearest(&solids, p, GRAIN).signed > 0.0,
            "grain inside the cavity"
        );
    }

    #[test]
    fn filter_blocks_grains_but_passes_water() {
        let solids = v60_dripper();
        // At y=0 the filter inner radius (~2.31) is tighter than the support (~2.50): a point at
        // r=2.4 is outside the filter cavity but inside the support cone.
        let p = Vec3::new(2.4, 0.0, 0.0);
        assert!(
            nearest(&solids, p, GRAIN).signed < 0.0,
            "grain is blocked by the filter wall"
        );
        assert!(
            nearest(&solids, p, WATER).signed > 0.0,
            "water passes the filter (inside the support cone)"
        );
    }

    #[test]
    fn water_drains_through_the_apex_outlet() {
        let solids = v60_dripper();
        // In the gap between the support apex (-3.0) and the cup rim (-3.5): water is unconstrained.
        let p = Vec3::new(0.0, -3.2, 0.0);
        assert_eq!(
            nearest(&solids, p, WATER).signed,
            crate::utils::sdf::FREE,
            "water free-falls through the outlet gap"
        );
    }

    #[test]
    fn grains_do_not_leak_past_the_closed_filter_tip() {
        let solids = v60_dripper();
        // Just below the filter tip (-3.37) on the axis: a grain is forbidden (closed tip).
        let p = Vec3::new(0.0, -3.4, 0.0);
        let g = nearest(&solids, p, GRAIN);
        assert!(
            g.signed < 0.0,
            "grain trapped by the closed filter tip: {}",
            g.signed
        );
        assert!(
            g.grad.y > 0.0,
            "pushed back up toward the cavity: {:?}",
            g.grad
        );
    }
}

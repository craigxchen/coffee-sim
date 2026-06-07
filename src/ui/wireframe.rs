//! CPU line geometry for the SDF solid boundaries (V60 dripper cone, filter, catch cup) so the
//! viewer can draw a wireframe of the collision geometry it otherwise only pushes particles out of.
//! Pure: produces a flat `LineList` vertex list from `&[SdfPrimitive]`; the GPU side (`render.rs`) is
//! a trivial unlit line pipeline. Mirrors `sdf.rs` exactly — cones/cylinders use `center.x`/`center.z`
//! for the radial offset but **absolute** y extents (`apex_y`/`top_y`/`rim_y`/`floor_y`); `center.y`
//! is unused, matching the cavity evaluation — so the wireframe coincides with where particles stop.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::utils::sdf::{SdfPrimitive, SolidKind, MASK_GRAIN};

/// One endpoint of a line segment. The buffer is a `LineList`, so vertices come in pairs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

/// Segments per ring (the circular cross-sections).
const RING_SEGMENTS: usize = 48;
/// Number of longitude lines joining a primitive's rings.
const LONGITUDES: usize = 16;
/// Below this radius a ring is degenerate (a point) and is skipped — e.g. the filter cone's closed
/// tip (`apex_r == 0`). Longitudes still converge to the tip so the cone shape still reads.
const RING_EPS: f32 = 1.0e-4;

/// Structural solids (the dripper support cone, the catch cup): warm ceramic white.
const CERAMIC: [f32; 3] = [0.80, 0.78, 0.72];
/// The grains-only filter cone: paper tan, so the two near-coincident V60 cones read as a filter
/// nested inside the dripper rather than as z-fighting doubled lines.
const PAPER: [f32; 3] = [0.78, 0.68, 0.50];

/// Build the wireframe line list (world-space, colored) for every solid in a scene. Cones become
/// stacked rings joined by longitudes; cylinders become rim/floor rings joined by longitudes. Empty
/// input (or solid-free scenes) yields an empty list, so the caller can skip the pass.
pub fn solid_wireframe(solids: &[SdfPrimitive]) -> Vec<LineVertex> {
    let mut out = Vec::new();
    for s in solids {
        let color = if s.species_mask == MASK_GRAIN {
            PAPER
        } else {
            CERAMIC
        };
        match s.kind {
            SolidKind::Cone {
                center,
                apex_y,
                top_y,
                apex_r,
                top_r,
                ..
            } => cone_lines(&mut out, center, apex_y, top_y, apex_r, top_r, color),
            SolidKind::Cylinder {
                center,
                floor_y,
                rim_y,
                radius,
            } => cylinder_lines(&mut out, center, floor_y, rim_y, radius, color),
        }
    }
    out
}

/// A truncated cone: rings at four heights (apex → top) joined by `LONGITUDES` slanted lines.
fn cone_lines(
    out: &mut Vec<LineVertex>,
    center: Vec3,
    apex_y: f32,
    top_y: f32,
    apex_r: f32,
    top_r: f32,
    color: [f32; 3],
) {
    // Intermediate rings make a tall cone read clearly; endpoints included.
    for &t in &[0.0_f32, 0.3333, 0.6667, 1.0] {
        let y = lerp(apex_y, top_y, t);
        let r = lerp(apex_r, top_r, t);
        ring(out, center, r, y, color);
    }
    for j in 0..LONGITUDES {
        let theta = (j as f32) / (LONGITUDES as f32) * std::f32::consts::TAU;
        let a = ring_point(center, apex_r, apex_y, theta);
        let b = ring_point(center, top_r, top_y, theta);
        push_segment(out, a, b, color);
    }
}

/// An open-top cylinder (cup): rings at the floor and rim joined by `LONGITUDES` vertical lines.
fn cylinder_lines(
    out: &mut Vec<LineVertex>,
    center: Vec3,
    floor_y: f32,
    rim_y: f32,
    radius: f32,
    color: [f32; 3],
) {
    ring(out, center, radius, floor_y, color);
    ring(out, center, radius, rim_y, color);
    for j in 0..LONGITUDES {
        let theta = (j as f32) / (LONGITUDES as f32) * std::f32::consts::TAU;
        let a = ring_point(center, radius, floor_y, theta);
        let b = ring_point(center, radius, rim_y, theta);
        push_segment(out, a, b, color);
    }
}

/// A circle of `RING_SEGMENTS` segments at `(center.xz, y)` and the given radius. Skipped when the
/// radius is ~0 (a degenerate point — e.g. a closed cone tip).
fn ring(out: &mut Vec<LineVertex>, center: Vec3, radius: f32, y: f32, color: [f32; 3]) {
    if radius <= RING_EPS {
        return;
    }
    for i in 0..RING_SEGMENTS {
        let t0 = (i as f32) / (RING_SEGMENTS as f32) * std::f32::consts::TAU;
        let t1 = ((i + 1) as f32) / (RING_SEGMENTS as f32) * std::f32::consts::TAU;
        let a = ring_point(center, radius, y, t0);
        let b = ring_point(center, radius, y, t1);
        push_segment(out, a, b, color);
    }
}

/// A point on a ring: radial offset about `center.xz` at absolute height `y` (matching `sdf.rs`,
/// which ignores `center.y`).
#[inline]
fn ring_point(center: Vec3, radius: f32, y: f32, theta: f32) -> Vec3 {
    Vec3::new(
        center.x + radius * theta.cos(),
        y,
        center.z + radius * theta.sin(),
    )
}

#[inline]
fn push_segment(out: &mut Vec<LineVertex>, a: Vec3, b: Vec3, color: [f32; 3]) {
    out.push(LineVertex {
        pos: a.to_array(),
        color,
    });
    out.push(LineVertex {
        pos: b.to_array(),
        color,
    });
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::geometry::v60_dripper;
    use crate::utils::sdf::{SolidKind, MASK_ALL, MASK_GRAIN};

    const TOL: f32 = 1.0e-3;

    fn radius_xz(v: &LineVertex, cx: f32, cz: f32) -> f32 {
        let dx = v.pos[0] - cx;
        let dz = v.pos[2] - cz;
        (dx * dx + dz * dz).sqrt()
    }

    /// A cone primitive helper (params mirror the support cone unless overridden).
    fn cone(center: Vec3, apex_r: f32, top_r: f32, mask: u32) -> SdfPrimitive {
        SdfPrimitive {
            kind: SolidKind::Cone {
                center,
                apex_y: -3.0,
                top_y: 3.0,
                apex_r,
                top_r,
                thickness: 0.05,
                hole_radius: 0.42,
                apex_open: true,
            },
            species_mask: mask,
            friction: 0.3,
        }
    }

    #[test]
    fn v60_wireframe_is_nonempty_even_and_finite() {
        let v = solid_wireframe(&v60_dripper());
        assert!(!v.is_empty(), "v60 has solids → wireframe is non-empty");
        assert_eq!(v.len() % 2, 0, "LineList: vertices come in pairs");
        assert!(
            v.iter()
                .all(|p| p.pos.iter().all(|c| c.is_finite())
                    && p.color.iter().all(|c| c.is_finite())),
            "all coordinates and colors finite"
        );
    }

    #[test]
    fn cone_top_and_apex_rings_at_expected_radius_and_height() {
        // Support cone: top_r 4.6834 @ y=+3, apex_r 0.42 @ y=-3.
        let v = solid_wireframe(&[cone(Vec3::ZERO, 0.42, 4.6834, MASK_ALL)]);
        assert!(
            v.iter()
                .any(|p| (radius_xz(p, 0.0, 0.0) - 4.6834).abs() < TOL
                    && (p.pos[1] - 3.0).abs() < TOL),
            "a vertex sits on the top rim (r≈top_r at y=top_y)"
        );
        assert!(
            v.iter().any(
                |p| (radius_xz(p, 0.0, 0.0) - 0.42).abs() < TOL && (p.pos[1] + 3.0).abs() < TOL
            ),
            "a vertex sits on the apex ring (r≈apex_r at y=apex_y)"
        );
    }

    #[test]
    fn cylinder_rings_at_rim_and_floor() {
        let cup = SdfPrimitive {
            kind: SolidKind::Cylinder {
                center: Vec3::ZERO,
                floor_y: -8.0,
                rim_y: -3.5,
                radius: 3.0,
            },
            species_mask: MASK_ALL,
            friction: 0.2,
        };
        let v = solid_wireframe(&[cup]);
        assert!(
            v.iter()
                .any(|p| (radius_xz(p, 0.0, 0.0) - 3.0).abs() < TOL && (p.pos[1] + 3.5).abs() < TOL),
            "rim ring at radius 3.0, y=-3.5"
        );
        assert!(
            v.iter()
                .any(|p| (radius_xz(p, 0.0, 0.0) - 3.0).abs() < TOL && (p.pos[1] + 8.0).abs() < TOL),
            "floor ring at radius 3.0, y=-8.0"
        );
    }

    #[test]
    fn world_center_offset_translates_every_vertex_in_xz_only() {
        // center.x/center.z shift the geometry; center.y is unused (matches sdf.rs).
        let centered = solid_wireframe(&[cone(Vec3::ZERO, 0.42, 4.6834, MASK_ALL)]);
        let offset = solid_wireframe(&[cone(Vec3::new(10.0, 99.0, 5.0), 0.42, 4.6834, MASK_ALL)]);
        assert_eq!(centered.len(), offset.len(), "same vertex count");
        for (c, o) in centered.iter().zip(offset.iter()) {
            assert!(
                (o.pos[0] - (c.pos[0] + 10.0)).abs() < TOL,
                "x shifted by center.x"
            );
            assert!(
                (o.pos[1] - c.pos[1]).abs() < TOL,
                "y unchanged (center.y ignored)"
            );
            assert!(
                (o.pos[2] - (c.pos[2] + 5.0)).abs() < TOL,
                "z shifted by center.z"
            );
        }
    }

    #[test]
    fn filter_and_structural_use_distinct_colors() {
        let filter = solid_wireframe(&[cone(Vec3::ZERO, 0.0, 4.10, MASK_GRAIN)]);
        let structural = solid_wireframe(&[cone(Vec3::ZERO, 0.42, 4.6834, MASK_ALL)]);
        assert_eq!(filter[0].color, PAPER, "grains-only filter is paper tan");
        assert_eq!(structural[0].color, CERAMIC, "structural solid is ceramic");
        assert_ne!(filter[0].color, structural[0].color);
    }

    #[test]
    fn empty_input_yields_empty() {
        assert!(solid_wireframe(&[]).is_empty());
    }

    #[test]
    fn closed_tip_skips_apex_ring_but_keeps_the_tip_longitude() {
        // apex_r == 0: the degenerate apex ring is skipped, but longitudes still reach the tip.
        let closed = solid_wireframe(&[cone(Vec3::ZERO, 0.0, 4.10, MASK_GRAIN)]);
        let open = solid_wireframe(&[cone(Vec3::ZERO, 1.5, 4.10, MASK_GRAIN)]);
        assert!(
            closed.len() < open.len(),
            "closed tip omits the apex ring ({} < {})",
            closed.len(),
            open.len()
        );
        assert!(
            closed.iter().all(|p| p.pos.iter().all(|c| c.is_finite())),
            "no NaN at the degenerate tip"
        );
        assert!(
            closed
                .iter()
                .any(|p| radius_xz(p, 0.0, 0.0) < TOL && (p.pos[1] + 3.0).abs() < TOL),
            "a longitude endpoint sits at the tip (r≈0 at apex_y)"
        );
    }
}

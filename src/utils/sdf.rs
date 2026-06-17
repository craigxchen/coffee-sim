//! Signed-distance-field helpers: analytic cavity primitives (truncated cone, cylinder) with
//! sample/gradient + a species-filtered union, for position-level collision projection (push a
//! particle out along the gradient) in the position-based solver — robust, unconditionally stable
//! boundary handling, which is why the primary avoids SPH boundary-particle machinery.
//!
//! Geometry is **analytic**, not a baked SDF texture: a circular cone on a Cartesian grid aliases
//! (v1's baked 128³ texture trapped water on the wall), so a handful of primitives are evaluated
//! directly. Each solid is a **cavity**: the allowed free-space region is the interior, so
//! `sample > 0` inside and `< 0` through the wall, and the gradient always points *into* the cavity.
//! The CPU math here is the reference mirror of the WGSL functions in the solver; reference values to
//! re-validate are in `KEEP.md` §3-4.

use glam::{Vec2, Vec3};

/// Below this length a direction is treated as degenerate and normalizes to zero (guards NaNs in
/// gradients / contact normals — `KEEP.md` §3).
const EPS: f32 = 1.0e-6;

/// Sentinel "no constraint" distance for free regions (open cone top, apex outlet, above a cup rim).
/// Large and positive so the union's min-pick never selects a free solid and the push-out never fires.
pub const FREE: f32 = 1.0e30;

/// Species-mask bits, indexed by `phase` (0 = water, 1 = grain): `mask & (1 << phase)`.
pub const MASK_WATER: u32 = 1; // phase 0
pub const MASK_GRAIN: u32 = 2; // phase 1
pub const MASK_ALL: u32 = MASK_WATER | MASK_GRAIN;

/// A static solid the particles collide with. Each kind is a **cavity** (interior allowed).
#[derive(Clone, Copy, Debug)]
pub enum SolidKind {
    /// Truncated-cone cavity on the +Y axis. `apex_r`/`top_r` are **outer** wall radii; the cavity
    /// (collision) surface is `inner_r(y) = max(lerp(apex_r, top_r, t) − thickness, hole_radius)`,
    /// used identically by the distance segment and the inside test. The top is always open (free
    /// entry). The apex is either an **open outlet** (`apex_open = true`: water falls through below
    /// the apex / the `hole_radius` funnel) or a **closed tip** (`apex_open = false`: the converging
    /// surface plus the tip trap grains — below the apex is forbidden).
    Cone {
        center: Vec3,
        apex_y: f32,
        top_y: f32,
        apex_r: f32,
        top_r: f32,
        thickness: f32,
        hole_radius: f32,
        apex_open: bool,
    },
    /// Open-top cylinder cavity (a cup): side wall + floor constrain; the rim is open (enter from
    /// the top). `rim_y > floor_y`.
    Cylinder {
        center: Vec3,
        floor_y: f32,
        rim_y: f32,
        radius: f32,
    },
    /// DIAGNOSTIC: regular N-gon prism cup (flat side faces, vertical axis). `apothem` is the
    /// inradius (face distance from the axis); `sides` = N (4 = axis-aligned square, 8 = octagon).
    /// Isolates whether the curved-cylinder wall over-pack is curvature, off-grid-normal, or the
    /// SDF-wall path itself. Same open-top + floor as `Cylinder`.
    PolyCup {
        center: Vec3,
        floor_y: f32,
        rim_y: f32,
        apothem: f32,
        sides: u32,
    },
}

/// A solid plus its collision properties: which species it blocks and its boundary friction.
#[derive(Clone, Copy, Debug)]
pub struct SdfPrimitive {
    pub kind: SolidKind,
    /// Species this solid blocks, as `MASK_*` bits. A grains-only filter sets `MASK_GRAIN` so water
    /// passes through it.
    pub species_mask: u32,
    /// Tangential Coulomb friction coefficient at this boundary (used for grains, like `floor_mu`).
    pub friction: f32,
}

/// The nearest binding constraint at a point for one species: the signed distance (interior
/// positive), the gradient (toward the cavity interior), and the active solid's friction.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub signed: f32,
    pub grad: Vec3,
    pub friction: f32,
}

impl SdfPrimitive {
    /// Does this solid constrain the given `phase` (0 = water, 1 = grain)?
    #[inline]
    pub fn applies_to(&self, phase: u32) -> bool {
        self.species_mask & (1u32 << phase) != 0
    }

    /// Signed distance (interior positive) and gradient (unit, toward the cavity interior) at `p`.
    pub fn cavity(&self, p: Vec3) -> (f32, Vec3) {
        match self.kind {
            SolidKind::Cone {
                center,
                apex_y,
                top_y,
                apex_r,
                top_r,
                thickness,
                hole_radius,
                apex_open,
            } => cone_cavity(
                center,
                apex_y,
                top_y,
                apex_r,
                top_r,
                thickness,
                hole_radius,
                apex_open,
                p,
            ),
            SolidKind::Cylinder {
                center,
                floor_y,
                rim_y,
                radius,
            } => cyl_cavity(center, floor_y, rim_y, radius, p),
            SolidKind::PolyCup {
                center,
                floor_y,
                rim_y,
                apothem,
                sides,
            } => poly_cavity(center, floor_y, rim_y, apothem, sides, p),
        }
    }

    /// Signed distance only (interior positive).
    #[inline]
    pub fn sample(&self, p: Vec3) -> f32 {
        self.cavity(p).0
    }

    /// Gradient only (unit, toward the cavity interior; zero only at non-differentiable points).
    #[inline]
    pub fn gradient(&self, p: Vec3) -> Vec3 {
        self.cavity(p).1
    }
}

/// Species-filtered union: the most-penetrated (minimum signed-distance) solid that blocks `phase`,
/// with its gradient and friction. This is an intersection-of-cavities composition — a particle must
/// be inside *all* applicable cavities, so the binding constraint each step is the min. Returns a
/// `FREE` contact (no constraint) when no solid applies.
pub fn nearest(solids: &[SdfPrimitive], p: Vec3, phase: u32) -> Contact {
    let mut best = Contact {
        signed: FREE,
        grad: Vec3::ZERO,
        friction: 0.0,
    };
    for s in solids {
        if !s.applies_to(phase) {
            continue;
        }
        let (signed, grad) = s.cavity(p);
        if signed < best.signed {
            best = Contact {
                signed,
                grad,
                friction: s.friction,
            };
        }
    }
    best
}

// --- internals ------------------------------------------------------------------------------------

#[inline]
fn safe_normalize2(v: Vec2) -> Vec2 {
    let len = v.length();
    if len > EPS {
        v / len
    } else {
        Vec2::ZERO
    }
}

#[inline]
fn safe_normalize3(v: Vec3) -> Vec3 {
    let len = v.length();
    if len > EPS {
        v / len
    } else {
        Vec3::ZERO
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Closest point on segment `[a, b]` to `p`, and whether it landed strictly on the segment interior
/// (vs clamped to an endpoint). The interior flag picks the gradient fallback for exact-contact points.
fn closest_on_segment(p: Vec2, a: Vec2, b: Vec2) -> (Vec2, bool) {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < EPS {
        return (a, false);
    }
    let t = (p - a).dot(ab) / len2;
    let tc = t.clamp(0.0, 1.0);
    (a + ab * tc, t > EPS && t < 1.0 - EPS)
}

/// Inward (toward smaller-r cavity interior) unit normal of the wall segment in the `(r, y)` plane.
/// For a dripper the segment widens upward (`b.r > a.r`), so the interior is on the `(−Δy, +Δr)` side.
#[inline]
fn inward_segment_normal(a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    safe_normalize2(Vec2::new(-ab.y, ab.x))
}

#[allow(clippy::too_many_arguments)]
fn cone_cavity(
    center: Vec3,
    apex_y: f32,
    top_y: f32,
    apex_r: f32,
    top_r: f32,
    thickness: f32,
    hole_radius: f32,
    apex_open: bool,
    p: Vec3,
) -> (f32, Vec3) {
    let rel = Vec2::new(p.x - center.x, p.z - center.z);
    let r = rel.length();
    let y = p.y;
    // Axis guard: zero the radial direction at r≈0 BEFORE building the 3D gradient.
    let rhat = if r < EPS { Vec2::ZERO } else { rel / r };

    if y > top_y {
        return (FREE, Vec3::Y); // open top → free entry (both cone kinds)
    }
    if apex_open && y < apex_y {
        return (FREE, Vec3::NEG_Y); // outlet → water falls through below the apex
    }

    // ONE radius contract: outer radii − thickness (clamped to the hole) define the cavity surface,
    // used for both the distance segment and the inside test (so sample==0 and the wall coincide).
    let inner_apex = (apex_r - thickness).max(hole_radius);
    let inner_top = (top_r - thickness).max(hole_radius);
    let a = Vec2::new(inner_apex, apex_y);
    let b = Vec2::new(inner_top, top_y);
    let (cp, on_interior) = closest_on_segment(Vec2::new(r, y), a, b);
    let d2d = (Vec2::new(r, y) - cp).length();

    let height = top_y - apex_y;
    let t = if height > EPS {
        (y - apex_y) / height
    } else {
        0.0
    };
    let inner = (lerp(apex_r, top_r, t) - thickness).max(hole_radius);
    let inside = y >= apex_y && r <= inner; // below the apex ⇒ outside (closed tip forbidden)
    let signed = if inside { d2d } else { -d2d };

    // Gradient always points into the cavity. dir = p − closest is zero exactly on the wall/tip;
    // on the smooth wall the analytic segment normal recovers it (a zero gradient there would
    // silently defeat the push-out), and only the non-differentiable tip/rim corner stays zero.
    let dir = Vec2::new(r, y) - cp;
    let n2d = if dir.length() > EPS {
        safe_normalize2(if inside { dir } else { -dir })
    } else if on_interior {
        inward_segment_normal(a, b)
    } else {
        Vec2::ZERO
    };
    let grad = safe_normalize3(Vec3::new(n2d.x * rhat.x, n2d.y, n2d.x * rhat.y));
    (signed, grad)
}

fn cyl_cavity(center: Vec3, floor_y: f32, rim_y: f32, radius: f32, p: Vec3) -> (f32, Vec3) {
    let rel = Vec2::new(p.x - center.x, p.z - center.z);
    let r = rel.length();
    let y = p.y;
    if y > rim_y {
        return (FREE, Vec3::Y); // open rim → enter from the top
    }
    let d_side = radius - r; // + inside the side wall
    let d_floor = y - floor_y; // + above the floor
    if d_side <= d_floor {
        // side wall nearest: increasing interior distance points toward the axis (−r̂)
        let rhat = if r < EPS { Vec2::ZERO } else { rel / r };
        let grad = safe_normalize3(Vec3::new(-rhat.x, 0.0, -rhat.y));
        (d_side, grad)
    } else {
        (d_floor, Vec3::Y) // floor nearest: push up
    }
}

/// Regular N-gon prism cup cavity (DIAGNOSTIC; mirror of common.wgsl::poly_cavity). The N face
/// outward normals are at angles 2πk/N (k=0 → +x), so `sides=4` is an axis-aligned square. Inner
/// side distance = apothem − max_k(rel·n_k); the gradient is the inward normal of the nearest face.
fn poly_cavity(
    center: Vec3,
    floor_y: f32,
    rim_y: f32,
    apothem: f32,
    sides: u32,
    p: Vec3,
) -> (f32, Vec3) {
    let rel = Vec2::new(p.x - center.x, p.z - center.z);
    let y = p.y;
    if y > rim_y {
        return (FREE, Vec3::Y); // open rim → enter from the top
    }
    let n = sides.max(3);
    let mut maxproj = f32::NEG_INFINITY;
    let mut bestn = Vec2::new(1.0, 0.0);
    for k in 0..n {
        let ang = std::f32::consts::TAU * (k as f32) / (n as f32);
        let nk = Vec2::new(ang.cos(), ang.sin());
        let proj = rel.dot(nk);
        if proj > maxproj {
            maxproj = proj;
            bestn = nk;
        }
    }
    let d_side = apothem - maxproj; // + inside the nearest face
    let d_floor = y - floor_y;
    if d_side <= d_floor {
        let grad = safe_normalize3(Vec3::new(-bestn.x, 0.0, -bestn.y));
        (d_side, grad)
    } else {
        (d_floor, Vec3::Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-4;

    /// A grains-only filter cone: closed tip at (r=0, y=−3), widening to outer r=4 at y=+3,
    /// thickness 0 (inner == outer for clean Euclidean checks), no hole.
    fn filter_cone() -> SdfPrimitive {
        SdfPrimitive {
            kind: SolidKind::Cone {
                center: Vec3::ZERO,
                apex_y: -3.0,
                top_y: 3.0,
                apex_r: 0.0,
                top_r: 4.0,
                thickness: 0.0,
                hole_radius: 0.0,
                apex_open: false,
            },
            species_mask: MASK_GRAIN,
            friction: 0.5,
        }
    }

    /// A support cone with an open apex outlet (water passes below the hole).
    fn support_cone() -> SdfPrimitive {
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
        }
    }

    fn cup() -> SdfPrimitive {
        SdfPrimitive {
            kind: SolidKind::Cylinder {
                center: Vec3::ZERO,
                floor_y: -8.0,
                rim_y: -3.5,
                radius: 3.0,
            },
            species_mask: MASK_ALL,
            friction: 0.2,
        }
    }

    #[test]
    fn cone_interior_positive_wall_negative_surface_zero() {
        let c = filter_cone();
        // inner radius at y=0 is lerp(0,4,0.5) = 2.
        assert!(
            c.sample(Vec3::new(1.0, 0.0, 0.0)) > 0.0,
            "centerward point inside"
        );
        assert!(
            c.sample(Vec3::new(3.0, 0.0, 0.0)) < 0.0,
            "past the wall outside"
        );
        assert!(
            c.sample(Vec3::new(2.0, 0.0, 0.0)).abs() < TOL,
            "on the surface ~0"
        );
    }

    #[test]
    fn cone_distance_is_euclidean_not_radial() {
        // At (r=1, y=0): radial gap is inner−r = 2−1 = 1; true perpendicular distance is smaller.
        let c = filter_cone();
        let signed = c.sample(Vec3::new(1.0, 0.0, 0.0));
        let radial = 2.0 - 1.0;
        assert!(
            signed > 0.0 && signed < radial - TOL,
            "euclidean {signed} < radial {radial}"
        );
    }

    #[test]
    fn cone_gradient_climbs_signed_distance_on_both_sides() {
        let c = filter_cone();
        let eps = 1.0e-3;
        for p in [Vec3::new(1.5, 0.0, 0.0), Vec3::new(2.5, 0.0, 0.0)] {
            let g = c.gradient(p);
            assert!(
                g.is_finite() && (g.length() - 1.0).abs() < 1.0e-3,
                "unit grad at {p:?}"
            );
            // sample(p + eps·grad) > sample(p): gradient points toward increasing signed distance.
            assert!(
                c.sample(p + g * eps) > c.sample(p),
                "grad must increase signed distance at {p:?}"
            );
        }
    }

    #[test]
    fn cone_smooth_wall_gradient_is_finite_nonzero() {
        // Exactly on the smooth wall (p == closest point): gradient must come from the segment
        // normal, not collapse to zero (a zero here would defeat the push-out).
        let c = filter_cone();
        // surface point at y=0 is r=2 on the +x axis.
        let g = c.gradient(Vec3::new(2.0, 0.0, 0.0));
        assert!(g.is_finite(), "no NaN/Inf on the smooth wall");
        assert!(
            g.length() > 0.5,
            "nonzero unit gradient on the smooth wall: {g:?}"
        );
        // points inward (toward smaller x) and up the slope.
        assert!(g.x < 0.0 && g.y > 0.0, "inward+up: {g:?}");
    }

    #[test]
    fn cone_axis_and_tip_are_finite_no_nan() {
        let c = filter_cone();
        for p in [
            Vec3::new(0.0, 0.0, 0.0),    // on the axis, deep inside
            Vec3::new(0.0, -3.001, 0.0), // exactly on the axis just below the closed tip
            Vec3::new(0.0, -3.0, 0.0),   // the tip itself
        ] {
            let (s, g) = c.cavity(p);
            assert!(s.is_finite(), "signed finite at {p:?}");
            assert!(g.is_finite(), "gradient finite (no NaN) at {p:?}");
        }
    }

    #[test]
    fn closed_tip_forbids_below_apex_on_axis_and_off_axis() {
        let c = filter_cone();
        // exact on-axis below the tip: r = inner = 0 must NOT misclassify as inside.
        let (s_axis, g_axis) = c.cavity(Vec3::new(0.0, -3.05, 0.0));
        assert!(
            s_axis < 0.0,
            "below the closed tip is forbidden on-axis: {s_axis}"
        );
        assert!(
            g_axis.y > 0.0,
            "gradient points up toward the tip on-axis: {g_axis:?}"
        );
        // off-axis below the apex whose closest point is on the slanted segment (not the tip):
        let (s_off, _) = c.cavity(Vec3::new(0.6, -2.9, 0.0));
        assert!(
            s_off < 0.0,
            "outside the closed-tip cavity is forbidden off-axis: {s_off}"
        );
    }

    #[test]
    fn open_outlet_frees_water_below_the_apex() {
        let c = support_cone();
        assert_eq!(
            c.sample(Vec3::new(0.0, -3.5, 0.0)),
            FREE,
            "outlet frees below the apex"
        );
        assert_eq!(
            c.sample(Vec3::new(4.0, 5.0, 0.0)),
            FREE,
            "open top frees above"
        );
    }

    #[test]
    fn cone_open_top_is_free() {
        let c = filter_cone();
        assert_eq!(c.sample(Vec3::new(1.0, 4.0, 0.0)), FREE);
    }

    #[test]
    fn cup_inside_positive_outside_negative() {
        let c = cup();
        assert!(c.sample(Vec3::new(0.0, -5.0, 0.0)) > 0.0, "inside the cup");
        assert!(
            c.sample(Vec3::new(3.5, -5.0, 0.0)) < 0.0,
            "past the side wall"
        );
        assert!(c.sample(Vec3::new(0.0, -8.5, 0.0)) < 0.0, "below the floor");
        assert!(
            c.sample(Vec3::new(0.0, -3.0, 0.0)) == FREE,
            "above the open rim"
        );
    }

    #[test]
    fn cup_gradients_point_inward_and_up() {
        let c = cup();
        // near the side wall → push toward the axis (−x here)
        let gs = c.gradient(Vec3::new(2.9, -5.0, 0.0));
        assert!(
            gs.x < 0.0 && gs.is_finite(),
            "side wall pushes inward: {gs:?}"
        );
        // near the floor → push up
        let gf = c.gradient(Vec3::new(0.0, -7.9, 0.0));
        assert!(gf.y > 0.0, "floor pushes up: {gf:?}");
    }

    #[test]
    fn union_skips_nonapplicable_species_and_returns_friction() {
        // A grains-only filter is inactive for water, active for grains.
        let solids = [filter_cone()];
        let p = Vec3::new(3.0, 0.0, 0.0); // past the filter wall
        assert_eq!(
            nearest(&solids, p, 0).signed,
            FREE,
            "water ignores the grains-only filter"
        );
        let grain = nearest(&solids, p, 1);
        assert!(grain.signed < 0.0, "grain is blocked by the filter");
        assert!(
            (grain.friction - 0.5).abs() < TOL,
            "returns the active solid's friction"
        );
    }

    #[test]
    fn union_takes_the_most_penetrated_nested_solid() {
        // Nested grain solids: a tighter inner cone + a looser outer cone both bind grains; the
        // union returns the min (most-penetrated) signed distance.
        let mut inner = filter_cone(); // inner r at y=0 is 2
        inner.friction = 0.9;
        let mut outer = filter_cone();
        if let SolidKind::Cone { ref mut top_r, .. } = outer.kind {
            *top_r = 6.0; // inner at y=0 becomes 3 → looser
        }
        outer.friction = 0.1;
        let solids = [outer, inner];
        // At (r=2.5, y=0): inside the looser (3) but outside the tighter (2) → min is the tighter.
        let c = nearest(&solids, Vec3::new(2.5, 0.0, 0.0), 1);
        assert!(c.signed < 0.0, "bound by the tighter nested cone");
        assert!(
            (c.friction - 0.9).abs() < TOL,
            "friction from the binding (tighter) solid"
        );
    }
}

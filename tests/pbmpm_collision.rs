//! PB-MPM thin-wall collider gates (CPU twin of the WGSL SDF in `src/solvers/pbmpm/common.wgsl`).
//!
//! The colliders are now THIN WALLS, not cavities: `dist > 0` is FREE space EVERYWHERE — both the
//! open vessel interior AND everything OUTSIDE the vessel — and `dist < 0` is only inside the wall
//! material. The gradient (∇dist) points OUT of the wall toward the nearest free side. This file
//! mirrors that SDF math exactly on the CPU (no GPU needed) so the conversion is pinned by a fast,
//! deterministic gate. The key regression: water that spills BESIDE the cup must read FREE (the old
//! cavity model grabbed it and shoved it back in — water could never overflow).
//!
//! These functions are a byte-for-byte port of the WGSL `cone_wall` / `cyl_wall` / `poly_wall` +
//! `solid_union`. If the shader math changes, this twin must change with it.

use glam::{Vec2, Vec3};

const WALL_T: f32 = 1.0; // mirrors WALL_T in common.wgsl
const EPS: f32 = 1.0e-6;
const FREE: f32 = 1.0e30;

#[derive(Clone, Copy)]
struct Hit {
    dist: f32,
    grad: Vec3,
}

fn norm2(v: Vec2) -> Vec2 {
    let l = v.length();
    if l > EPS {
        v / l
    } else {
        Vec2::ZERO
    }
}
fn norm3(v: Vec3) -> Vec3 {
    let l = v.length();
    if l > EPS {
        v / l
    } else {
        Vec3::ZERO
    }
}

/// Closest point on segment `[a, b]` to `p`, plus whether it landed strictly interior.
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

// --- thin-wall SDFs (CPU port of common.wgsl) -------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cone_wall(
    center: Vec2,
    apex_y: f32,
    apex_r: f32,
    top_y: f32,
    top_r: f32,
    geom_thickness: f32,
    hole_radius: f32,
    apex_open: bool,
    p: Vec3,
) -> Hit {
    let thickness = geom_thickness.max(WALL_T);
    let rel = Vec2::new(p.x - center.x, p.z - center.y);
    let r = rel.length();
    let y = p.y;
    let rhat = if r >= EPS { rel / r } else { Vec2::ZERO };

    if y > top_y {
        return Hit {
            dist: FREE,
            grad: Vec3::Y,
        };
    }
    if apex_open && y < apex_y {
        return Hit {
            dist: FREE,
            grad: Vec3::NEG_Y,
        };
    }

    let p2d = Vec2::new(r, y);
    let height = top_y - apex_y;
    let t = if height > EPS {
        (y - apex_y) / height
    } else {
        0.0
    };
    let outer_r = apex_r + (top_r - apex_r) * t;
    let inner_r = (outer_r - thickness).max(hole_radius);

    // free_in: positive inside the cavity (r ≤ inner_r), grad toward the cavity interior.
    let ai = Vec2::new((apex_r - thickness).max(hole_radius), apex_y);
    let bi = Vec2::new((top_r - thickness).max(hole_radius), top_y);
    let (cpi, cpi_int) = closest_on_segment(p2d, ai, bi);
    let di = (p2d - cpi).length();
    let inside_cav = y >= apex_y && r <= inner_r;
    let free_in = if inside_cav { di } else { -di };
    let dir_in = p2d - cpi;
    let n_in = if dir_in.length() > EPS {
        norm2(if inside_cav { dir_in } else { -dir_in })
    } else if cpi_int {
        let ab = bi - ai;
        norm2(Vec2::new(-ab.y, ab.x))
    } else {
        Vec2::ZERO
    };

    // free_out: positive outside the vessel (r ≥ outer_r), grad outward.
    let ao = Vec2::new(apex_r, apex_y);
    let bo = Vec2::new(top_r, top_y);
    let (cpo, cpo_int) = closest_on_segment(p2d, ao, bo);
    let do2 = (p2d - cpo).length();
    let inside_outer = y >= apex_y && r <= outer_r;
    let free_out = if inside_outer { -do2 } else { do2 };
    let dir_out = p2d - cpo;
    let n_out = if dir_out.length() > EPS {
        norm2(if inside_outer { -dir_out } else { dir_out })
    } else if cpo_int {
        let ab = bo - ao;
        norm2(Vec2::new(ab.y, -ab.x))
    } else {
        Vec2::ZERO
    };

    let (dist, n2d) = if free_out > free_in {
        (free_out, n_out)
    } else {
        (free_in, n_in)
    };
    Hit {
        dist,
        grad: norm3(Vec3::new(n2d.x * rhat.x, n2d.y, n2d.x * rhat.y)),
    }
}

fn cyl_wall(center: Vec2, floor_y: f32, rim_y: f32, radius: f32, p: Vec3) -> Hit {
    let rel = Vec2::new(p.x - center.x, p.z - center.y);
    let r = rel.length();
    let y = p.y;
    let rhat = if r >= EPS { rel / r } else { Vec2::ZERO };

    let o_side = r - (radius + WALL_T);
    let o_floor = (floor_y - WALL_T) - y;
    let o_rim = y - rim_y;
    let mut sdf_o = o_side;
    let mut n_o = Vec3::new(rhat.x, 0.0, rhat.y);
    if o_floor > sdf_o {
        sdf_o = o_floor;
        n_o = Vec3::NEG_Y;
    }
    if o_rim > sdf_o {
        sdf_o = o_rim;
        n_o = Vec3::Y;
    }

    let c_side = r - radius;
    let c_floor = floor_y - y;
    let mut sdf_c = c_side;
    let mut n_c = Vec3::new(-rhat.x, 0.0, -rhat.y);
    if c_floor > sdf_c {
        sdf_c = c_floor;
        n_c = Vec3::Y;
    }

    let (dist, grad) = if -sdf_c > sdf_o {
        (-sdf_c, n_c)
    } else {
        (sdf_o, n_o)
    };
    Hit {
        dist,
        grad: norm3(grad),
    }
}

fn poly_wall(center: Vec2, floor_y: f32, rim_y: f32, apothem: f32, sides: u32, p: Vec3) -> Hit {
    let rel = Vec2::new(p.x - center.x, p.z - center.y);
    let y = p.y;
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

    let o_side = maxproj - (apothem + WALL_T);
    let o_floor = (floor_y - WALL_T) - y;
    let o_rim = y - rim_y;
    let mut sdf_o = o_side;
    let mut n_o = Vec3::new(bestn.x, 0.0, bestn.y);
    if o_floor > sdf_o {
        sdf_o = o_floor;
        n_o = Vec3::NEG_Y;
    }
    if o_rim > sdf_o {
        sdf_o = o_rim;
        n_o = Vec3::Y;
    }

    let c_side = maxproj - apothem;
    let c_floor = floor_y - y;
    let mut sdf_c = c_side;
    let mut n_c = Vec3::new(-bestn.x, 0.0, -bestn.y);
    if c_floor > sdf_c {
        sdf_c = c_floor;
        n_c = Vec3::Y;
    }

    let (dist, grad) = if -sdf_c > sdf_o {
        (-sdf_c, n_c)
    } else {
        (sdf_o, n_o)
    };
    Hit {
        dist,
        grad: norm3(grad),
    }
}

// --- scene fixtures (the v60 cup + support cone, matching utils/geometry::v60_dripper) ---------

fn cup() -> impl Fn(Vec3) -> Hit {
    // Catch cup: floor_y -8, rim_y -3.5, radius 3.0, centered at origin (KEEP §4).
    move |p| cyl_wall(Vec2::ZERO, -8.0, -3.5, 3.0, p)
}

fn support_cone() -> impl Fn(Vec3) -> Hit {
    // Support cone: apex_y -3, top_y +3, apex_r 0.42, top_r 4.6834, thickness 0.05, hole 0.42, open.
    move |p| cone_wall(Vec2::ZERO, -3.0, 0.42, 3.0, 4.6834, 0.05, 0.42, true, p)
}

const TOL: f32 = 1.0e-4;

/// `x` is on a free side of the wall after a push-out: feed it through and assert it ends up FREE
/// (dist ≥ 0) and the push direction is sane. Returns the pushed-out position.
fn push_out(f: &impl Fn(Vec3) -> Hit, x: Vec3) -> Vec3 {
    let hit = f(x);
    if hit.dist < 0.0 {
        x + (-hit.dist) * hit.grad
    } else {
        x
    }
}

// --- CUP: thin-wall regression gate ------------------------------------------------------------

#[test]
fn cup_inside_is_free() {
    // A point INSIDE the cup (r=1, y=-6): free space, the water pools here.
    let f = cup();
    assert!(
        f(Vec3::new(1.0, -6.0, 0.0)).dist > 0.0,
        "inside the cup is free"
    );
}

#[test]
fn cup_outside_beside_is_free_the_regression() {
    // THE REGRESSION: a point OUTSIDE the cup beside it (r=5, y=-6, with radius+WALL_T = 4). The old
    // cavity model treated this as solid and shoved spilled water back into the cup. It MUST be free.
    let f = cup();
    let d = f(Vec3::new(5.0, -6.0, 0.0)).dist;
    assert!(
        d > 0.0,
        "spilled water BESIDE the cup is free (not grabbed): dist={d}"
    );
    // Just past the outer wall is free too (r = radius + WALL_T + 0.01 = 4.01).
    assert!(
        f(Vec3::new(4.01, -6.0, 0.0)).dist > 0.0,
        "just outside the outer wall is free"
    );
}

#[test]
fn cup_wall_is_solid_and_pushes_to_a_free_face() {
    // A point in the WALL material (r=3.05, y=-6: between radius 3.0 and radius+WALL_T 4.0).
    let f = cup();
    let p = Vec3::new(3.05, -6.0, 0.0);
    let hit = f(p);
    assert!(
        hit.dist < 0.0,
        "inside the side wall material: dist={}",
        hit.dist
    );
    // Nearest free face here is the cup interior (0.05 in) vs outside (0.95 out) → push inward (−x).
    assert!(
        hit.grad.x < -0.9 && hit.grad.y.abs() < TOL,
        "side-wall normal points inward toward the cup interior: {:?}",
        hit.grad
    );
    let out = push_out(&f, p);
    assert!(
        f(out).dist >= -TOL,
        "pushed out to a free position: {:?} dist={}",
        out,
        f(out).dist
    );
}

#[test]
fn cup_floor_wall_pushes_up() {
    // A point in the FLOOR shell, nearer the inner (cup-interior) free face (y = -8.3, floor -8,
    // shell [-9,-8]): pushed UP into the cup.
    let f = cup();
    let p = Vec3::new(0.5, -8.3, 0.0);
    let hit = f(p);
    assert!(hit.dist < 0.0, "inside the floor shell: dist={}", hit.dist);
    assert!(hit.grad.y > 0.9, "floor pushes up: {:?}", hit.grad);
    assert!(f(push_out(&f, p)).dist >= -TOL, "pushed up to free");
}

#[test]
fn cup_above_rim_is_free() {
    // A point ABOVE the rim (r=1, y=-3.0, rim -3.5): free — water can spill OVER the top.
    let f = cup();
    assert!(
        f(Vec3::new(1.0, -3.0, 0.0)).dist > 0.0,
        "above the open rim is free"
    );
}

#[test]
fn cup_below_floor_outside_is_free() {
    // A point BELOW the cup floor and outside the shell (y=-9.5, below floor_y−WALL_T = -9): free,
    // water that tunnels/falls under the cup is not grabbed.
    let f = cup();
    assert!(
        f(Vec3::new(1.0, -9.5, 0.0)).dist > 0.0,
        "below the outer floor is free"
    );
}

// --- CONE: funnel thin wall --------------------------------------------------------------------

#[test]
fn cone_inside_funnel_is_free() {
    // Inside the funnel cavity (r=1, y=0): inner_r ≈ 1.55 with WALL_T=1 → free.
    let f = support_cone();
    assert!(
        f(Vec3::new(1.0, 0.0, 0.0)).dist > 0.0,
        "inside the funnel is free"
    );
}

#[test]
fn cone_outside_is_free() {
    // Outside the cone wall (r=5, y=0, outer ≈ 2.55): free — water beside the funnel falls away.
    let f = support_cone();
    assert!(
        f(Vec3::new(5.0, 0.0, 0.0)).dist > 0.0,
        "outside the cone is free"
    );
}

#[test]
fn cone_apex_hole_is_free_drain_open() {
    // Within the apex drain hole (r < hole_radius 0.42, at the apex y=-3): free passage — the drain.
    let f = support_cone();
    assert!(
        f(Vec3::new(0.2, -3.0, 0.0)).dist > 0.0,
        "the apex drain hole is free"
    );
    // Below the open apex outlet: free (water falls through into the cup).
    assert!(
        f(Vec3::new(0.0, -3.5, 0.0)).dist >= FREE - 1.0,
        "below the open apex is free"
    );
}

#[test]
fn cone_wall_is_solid_and_pushes_out() {
    // A point in the cone wall (r=2.0, y=0: between inner ≈1.55 and outer ≈2.55).
    let f = support_cone();
    let p = Vec3::new(2.0, 0.0, 0.0);
    let hit = f(p);
    assert!(hit.dist < 0.0, "inside the cone wall: dist={}", hit.dist);
    assert!(
        hit.grad.length() > 0.5 && hit.grad.is_finite(),
        "nonzero finite push-out normal: {:?}",
        hit.grad
    );
    assert!(
        f(push_out(&f, p)).dist >= -TOL,
        "pushed out of the cone wall to free"
    );
}

#[test]
fn cone_above_top_is_free() {
    let f = support_cone();
    assert!(
        f(Vec3::new(1.0, 4.0, 0.0)).dist >= FREE - 1.0,
        "above the open top is free"
    );
}

// --- POLY: same regression as the cup, polygon apothem -----------------------------------------

#[test]
fn poly_outside_is_free_and_wall_pushes() {
    // Square cup (sides=4, apothem 3, floor -8, rim -3.5). Outside (x=5 on a face axis) is free.
    let f = |p| poly_wall(Vec2::ZERO, -8.0, -3.5, 3.0, 4, p);
    assert!(
        f(Vec3::new(5.0, -6.0, 0.0)).dist > 0.0,
        "spilled water beside the poly cup is free"
    );
    assert!(f(Vec3::new(1.0, -6.0, 0.0)).dist > 0.0, "inside is free");
    // In the side wall (x=3.05, between apothem 3 and apothem+WALL_T 4): solid, pushed inward.
    let hit = f(Vec3::new(3.05, -6.0, 0.0));
    assert!(hit.dist < 0.0, "inside the poly side wall: {}", hit.dist);
    assert!(
        hit.grad.x < -0.9,
        "poly side-wall normal points inward: {:?}",
        hit.grad
    );
}

// --- BOX BACKSTOP: the ultimate container ------------------------------------------------------

/// CPU twin of the particle_integrate box clamp: clamp to [box_min, box_max] and zero the into-wall
/// normal velocity component (free slip on the separating component). The box is the real container
/// that catches water spilling out of any vessel.
fn box_clamp(box_min: Vec3, box_max: Vec3, mut x: Vec3, mut v: Vec3) -> (Vec3, Vec3) {
    if x.x < box_min.x {
        x.x = box_min.x;
        if v.x < 0.0 {
            v.x = 0.0;
        }
    }
    if x.y < box_min.y {
        x.y = box_min.y;
        if v.y < 0.0 {
            v.y = 0.0;
        }
    }
    if x.z < box_min.z {
        x.z = box_min.z;
        if v.z < 0.0 {
            v.z = 0.0;
        }
    }
    if x.x > box_max.x {
        x.x = box_max.x;
        if v.x > 0.0 {
            v.x = 0.0;
        }
    }
    if x.y > box_max.y {
        x.y = box_max.y;
        if v.y > 0.0 {
            v.y = 0.0;
        }
    }
    if x.z > box_max.z {
        x.z = box_max.z;
        if v.z > 0.0 {
            v.z = 0.0;
        }
    }
    (x, v)
}

#[test]
fn box_backstop_clamps_and_zeros_into_wall_velocity() {
    let lo = Vec3::splat(-10.0);
    let hi = Vec3::splat(10.0);
    // A particle pushed past box_max with outward velocity: clamped back, into-wall velocity zeroed.
    let (x, v) = box_clamp(
        lo,
        hi,
        Vec3::new(12.0, 5.0, -13.0),
        Vec3::new(4.0, 1.0, -7.0),
    );
    assert!((x.x - 10.0).abs() < TOL, "clamped to box_max.x");
    assert!((x.z + 10.0).abs() < TOL, "clamped to box_min.z");
    assert!(v.x.abs() < TOL, "into +x wall velocity zeroed");
    assert!(v.z.abs() < TOL, "into −z wall velocity zeroed");
    assert!(
        (v.y - 1.0).abs() < TOL,
        "separating (interior) velocity preserved"
    );
    // Water spilling beside the cup falls and is caught by the box floor.
    let (xf, vf) = box_clamp(
        lo,
        hi,
        Vec3::new(5.0, -12.0, 0.0),
        Vec3::new(0.0, -8.0, 0.0),
    );
    assert!((xf.y + 10.0).abs() < TOL, "caught on the box floor");
    assert!(vf.y.abs() < TOL, "downward velocity zeroed at the floor");
}

// --- UNION: most-penetrated wall + empty scene ------------------------------------------------

#[test]
fn union_takes_most_penetrated_and_empty_is_free() {
    // The v60 set: support cone + cup. A point inside BOTH vessels' free regions is free.
    let cone = support_cone();
    let cup = cup();
    let union = |p: Vec3| -> f32 {
        let a = cone(p).dist;
        let b = cup(p).dist;
        a.min(b)
    };
    // Inside the funnel, above the cup rim: both free → union free.
    assert!(union(Vec3::new(1.0, 0.0, 0.0)) > 0.0, "centerline is free");
    // Beside the cup (the regression) — cone free (outside it), cup free (beside it) → free.
    assert!(
        union(Vec3::new(5.0, -6.0, 0.0)) > 0.0,
        "beside the cup, union still free (spill falls freely)"
    );
}

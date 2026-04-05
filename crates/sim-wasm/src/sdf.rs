use coffee_sim_core::sph::Vec3;

pub(crate) const SDF_NO_CONSTRAINT: f32 = 999.0;
pub(crate) const SDF_RES: u32 = 128;
pub(crate) const WALL_THICKNESS: f32 = 0.4;

#[derive(Clone, Debug)]
pub(crate) enum Shape {
    TruncatedCone {
        center: Vec3,
        top_radius: f32,
        bot_radius: f32,
        top_y: f32,
        bot_y: f32,
    },
    Cylinder {
        center: Vec3,
        radius: f32,
        top_y: f32,
        bot_y: f32,
    },
}

pub(crate) fn sdf_interior(shape: &Shape, p: Vec3) -> f32 {
    match shape {
        Shape::TruncatedCone {
            center,
            top_radius,
            bot_radius,
            top_y,
            bot_y,
        } => {
            // Open top and bottom — radial wall only within Y range.
            let local_y = p.y - center.y;
            if local_y > *top_y || local_y < *bot_y {
                return SDF_NO_CONSTRAINT;
            }
            let height = top_y - bot_y;
            if height < 1e-10 {
                return SDF_NO_CONSTRAINT;
            }
            let t = (local_y - bot_y) / height;
            let cone_r = bot_radius + (top_radius - bot_radius) * t;
            let dx = p.x - center.x;
            let dz = p.z - center.z;
            let r = (dx * dx + dz * dz).sqrt();
            cone_r - r
        }
        Shape::Cylinder {
            center,
            radius,
            top_y,
            bot_y,
        } => {
            // Open top, closed bottom (floor) — radial wall + floor.
            let local_y = p.y - center.y;
            if local_y > *top_y {
                return SDF_NO_CONSTRAINT;
            }
            let dx = p.x - center.x;
            let dz = p.z - center.z;
            let r = (dx * dx + dz * dz).sqrt();
            let radial_dist = radius - r;
            let floor_dist = local_y - bot_y;
            radial_dist.min(floor_dist)
        }
    }
}

fn obstacle_aabb(shape: &Shape, margin: f32) -> (Vec3, Vec3) {
    match shape {
        Shape::TruncatedCone {
            center,
            top_radius,
            bot_radius,
            top_y,
            bot_y,
        } => {
            let max_r = top_radius.max(*bot_radius) + margin;
            (
                Vec3::new(
                    center.x - max_r,
                    center.y + bot_y - margin,
                    center.z - max_r,
                ),
                Vec3::new(
                    center.x + max_r,
                    center.y + top_y + margin,
                    center.z + max_r,
                ),
            )
        }
        Shape::Cylinder {
            center,
            radius,
            top_y,
            bot_y,
        } => {
            let r = radius + margin;
            (
                Vec3::new(center.x - r, center.y + bot_y - margin, center.z - r),
                Vec3::new(center.x + r, center.y + top_y + margin, center.z + r),
            )
        }
    }
}

fn voxel_center(ix: usize, iy: usize, iz: usize, bounds: Vec3, res: usize) -> Vec3 {
    let res_f = res as f32;
    Vec3::new(
        -bounds.x * 0.5 + (ix as f32 + 0.5) * bounds.x / res_f,
        -bounds.y * 0.5 + (iy as f32 + 0.5) * bounds.y / res_f,
        -bounds.z * 0.5 + (iz as f32 + 0.5) * bounds.z / res_f,
    )
}

pub(crate) fn generate_sdf(shapes: &[Shape], bounds: Vec3, res: u32) -> Vec<f32> {
    let n = res as usize;
    let mut data = vec![SDF_NO_CONSTRAINT; n * n * n];

    for shape in shapes {
        let (aabb_min, aabb_max) = obstacle_aabb(shape, 1.0);

        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let p = voxel_center(ix, iy, iz, bounds, n);
                    if p.x < aabb_min.x
                        || p.x > aabb_max.x
                        || p.y < aabb_min.y
                        || p.y > aabb_max.y
                        || p.z < aabb_min.z
                        || p.z > aabb_max.z
                    {
                        continue;
                    }
                    let sd = sdf_interior(shape, p) - WALL_THICKNESS * 0.5;
                    let idx = iz * n * n + iy * n + ix;
                    if data[idx] >= SDF_NO_CONSTRAINT - 1.0 {
                        data[idx] = sd;
                    } else {
                        data[idx] = data[idx].max(sd);
                    }
                }
            }
        }
    }
    data
}

#[allow(dead_code)]
pub(crate) fn sample_sdf_trilinear(data: &[f32], bounds: Vec3, res: u32, p: Vec3) -> f32 {
    let n = res as f32;
    let uvx = (p.x + bounds.x * 0.5) / bounds.x * n - 0.5;
    let uvy = (p.y + bounds.y * 0.5) / bounds.y * n - 0.5;
    let uvz = (p.z + bounds.z * 0.5) / bounds.z * n - 0.5;

    let ix = uvx.floor() as i32;
    let iy = uvy.floor() as i32;
    let iz = uvz.floor() as i32;
    let fx = uvx - ix as f32;
    let fy = uvy - iy as f32;
    let fz = uvz - iz as f32;

    let n_i = res as i32;
    let fetch = |x: i32, y: i32, z: i32| -> f32 {
        let cx = x.clamp(0, n_i - 1) as usize;
        let cy = y.clamp(0, n_i - 1) as usize;
        let cz = z.clamp(0, n_i - 1) as usize;
        data[cz * (res as usize) * (res as usize) + cy * (res as usize) + cx]
    };

    let c000 = fetch(ix, iy, iz);
    let c100 = fetch(ix + 1, iy, iz);
    let c010 = fetch(ix, iy + 1, iz);
    let c110 = fetch(ix + 1, iy + 1, iz);
    let c001 = fetch(ix, iy, iz + 1);
    let c101 = fetch(ix + 1, iy, iz + 1);
    let c011 = fetch(ix, iy + 1, iz + 1);
    let c111 = fetch(ix + 1, iy + 1, iz + 1);

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;
    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;
    c0 * (1.0 - fz) + c1 * fz
}

#[allow(dead_code)]
pub(crate) fn sdf_gradient(data: &[f32], bounds: Vec3, res: u32, p: Vec3) -> Vec3 {
    let voxel = Vec3::new(
        bounds.x / res as f32,
        bounds.y / res as f32,
        bounds.z / res as f32,
    );
    let eps = voxel.x.max(voxel.y).max(voxel.z);
    let dx = sample_sdf_trilinear(data, bounds, res, Vec3::new(p.x + eps, p.y, p.z))
        - sample_sdf_trilinear(data, bounds, res, Vec3::new(p.x - eps, p.y, p.z));
    let dy = sample_sdf_trilinear(data, bounds, res, Vec3::new(p.x, p.y + eps, p.z))
        - sample_sdf_trilinear(data, bounds, res, Vec3::new(p.x, p.y - eps, p.z));
    let dz = sample_sdf_trilinear(data, bounds, res, Vec3::new(p.x, p.y, p.z + eps))
        - sample_sdf_trilinear(data, bounds, res, Vec3::new(p.x, p.y, p.z - eps));
    Vec3::new(dx, dy, dz)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v60_shapes() -> Vec<Shape> {
        vec![
            Shape::TruncatedCone {
                center: Vec3::ZERO,
                top_radius: 4.5,
                bot_radius: 0.8,
                top_y: 3.0,
                bot_y: -3.0,
            },
            Shape::Cylinder {
                center: Vec3::ZERO,
                radius: 3.0,
                top_y: -3.5,
                bot_y: -8.0,
            },
        ]
    }

    const BOUNDS: Vec3 = Vec3 {
        x: 14.0,
        y: 20.0,
        z: 14.0,
    };
    const SPOUT_ORIGIN: Vec3 = Vec3 {
        x: -4.8,
        y: 6.15,
        z: 0.0,
    };
    const SPOUT_DIR: Vec3 = Vec3 {
        x: 0.7310553,
        y: -0.6823183,
        z: 0.0,
    };

    // ── Analytical SDF tests ──

    #[test]
    fn cone_center_is_inside() {
        let cone = &v60_shapes()[0];
        let val = sdf_interior(cone, Vec3::new(0.0, 0.0, 0.0));
        assert!(
            val > 0.0,
            "center of cone should be positive (inside), got {val}"
        );
    }

    #[test]
    fn cone_wall_is_outside() {
        // At y=0, cone radius = mix(0.8, 4.5, 0.5) = 2.65
        // Point at r=3.0 should be outside
        let cone = &v60_shapes()[0];
        let val = sdf_interior(cone, Vec3::new(3.0, 0.0, 0.0));
        assert!(val < 0.0, "outside cone wall should be negative, got {val}");
    }

    #[test]
    fn cone_surface_is_near_zero() {
        // At y=0, cone radius = 2.65. Point right at the wall.
        // (analytical SDF without wall thickness)
        let cone = &v60_shapes()[0];
        let val = sdf_interior(cone, Vec3::new(2.65, 0.0, 0.0));
        assert!(
            val.abs() < 0.1,
            "at cone surface, SDF should be near zero, got {val}"
        );
    }

    #[test]
    fn wall_thickness_shifts_boundary_inward() {
        // With wall thickness, the grid SDF zero-crossing should be inside the geometric wall
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        // At y=0, cone radius = 2.65. Point at r=2.5 (inside by 0.15) should be
        // negative in the grid due to wall thickness of 0.4 (half = 0.2 offset)
        let val = sample_sdf_trilinear(&sdf, BOUNDS, 64, Vec3::new(2.5, 0.0, 0.0));
        assert!(
            val < 0.0,
            "near cone wall should be negative with thickness, got {val}"
        );
    }

    #[test]
    fn cone_above_top_is_no_constraint() {
        let cone = &v60_shapes()[0];
        let val = sdf_interior(cone, Vec3::new(0.0, 4.0, 0.0));
        assert!(
            val > 100.0,
            "above open cone top should be no-constraint (positive), got {val}"
        );
    }

    #[test]
    fn cone_below_bottom_is_no_constraint() {
        let cone = &v60_shapes()[0];
        let val = sdf_interior(cone, Vec3::new(0.0, -4.0, 0.0));
        assert!(
            val > 100.0,
            "below open cone bottom should be no-constraint (positive), got {val}"
        );
    }

    #[test]
    fn cylinder_center_is_inside() {
        let cyl = &v60_shapes()[1];
        let val = sdf_interior(cyl, Vec3::new(0.0, -5.75, 0.0));
        assert!(
            val > 0.0,
            "center of cylinder should be positive, got {val}"
        );
    }

    #[test]
    fn cylinder_wall_is_outside() {
        let cyl = &v60_shapes()[1];
        let val = sdf_interior(cyl, Vec3::new(4.0, -5.75, 0.0));
        assert!(
            val < 0.0,
            "outside cylinder wall should be negative, got {val}"
        );
    }

    #[test]
    fn cylinder_below_floor_is_outside() {
        let cyl = &v60_shapes()[1];
        let val = sdf_interior(cyl, Vec3::new(0.0, -9.0, 0.0));
        assert!(
            val < 0.0,
            "below cylinder floor should be negative, got {val}"
        );
    }

    // ── Grid SDF tests ──

    #[test]
    fn free_space_is_positive() {
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 32);
        // Point at (0, 9, 0) — well above the cone, free space
        let val = sample_sdf_trilinear(&sdf, BOUNDS, 32, Vec3::new(0.0, 9.0, 0.0));
        assert!(
            val > 100.0,
            "free space should be large positive (no collision), got {val}"
        );
    }

    #[test]
    fn grid_cone_center_positive() {
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        let val = sample_sdf_trilinear(&sdf, BOUNDS, 64, Vec3::new(0.0, 0.0, 0.0));
        assert!(
            val > 0.0,
            "grid SDF at cone center should be positive, got {val}"
        );
    }

    #[test]
    fn grid_cylinder_center_positive() {
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        let val = sample_sdf_trilinear(&sdf, BOUNDS, 64, Vec3::new(0.0, -5.75, 0.0));
        assert!(
            val > 0.0,
            "grid SDF at cylinder center should be positive, got {val}"
        );
    }

    #[test]
    fn spout_path_is_outside_funnel() {
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 128);
        for distance in [0.26_f32, 0.5, 1.0, 1.5, 2.0] {
            let p = Vec3::new(
                SPOUT_ORIGIN.x + SPOUT_DIR.x * distance,
                SPOUT_ORIGIN.y + SPOUT_DIR.y * distance,
                SPOUT_ORIGIN.z,
            );
            let val = sample_sdf_trilinear(&sdf, BOUNDS, 128, p);
            assert!(
                val > 0.0,
                "spout path should stay in free space at distance {distance}, got {val} at {p:?}"
            );
        }
    }

    #[test]
    fn grid_outside_cone_negative() {
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        // At y=0, cone radius ~2.65. Point at r=4.0 is outside.
        let val = sample_sdf_trilinear(&sdf, BOUNDS, 64, Vec3::new(4.0, 0.0, 0.0));
        assert!(
            val < 0.0,
            "grid SDF outside cone should be negative, got {val}"
        );
    }

    // ── Gradient tests ──

    #[test]
    fn gradient_points_inward_at_cone_wall() {
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        // Just outside the cone wall at y=0, x=3.0 (cone radius ~2.65)
        let grad = sdf_gradient(&sdf, BOUNDS, 64, Vec3::new(3.0, 0.0, 0.0));
        // Gradient should point inward (negative x direction)
        assert!(
            grad.x < 0.0,
            "gradient x should point inward (negative), got {}",
            grad.x
        );
        assert!(
            grad.x.abs() > grad.y.abs(),
            "radial gradient should dominate over vertical"
        );
    }

    #[test]
    fn gradient_points_inward_at_cylinder_wall() {
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        // Just outside cylinder wall at z=3.5
        let grad = sdf_gradient(&sdf, BOUNDS, 64, Vec3::new(0.0, -5.75, 3.5));
        assert!(
            grad.z < 0.0,
            "gradient z should point inward (negative), got {}",
            grad.z
        );
    }

    // ── Combined shape tests ──

    #[test]
    fn cone_cylinder_gap_is_unconstrained() {
        // The gap between cone bottom (y=-3) and cylinder top (y=-3.5) at y=-3.25
        // is outside both shapes' Y ranges — no constraint, particles fall through
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        let val = sample_sdf_trilinear(&sdf, BOUNDS, 64, Vec3::new(0.0, -3.25, 0.0));
        assert!(
            val > 0.0,
            "cone-cylinder gap should be positive (no constraint), got {val}"
        );
    }

    #[test]
    fn cone_bottom_narrow_is_inside() {
        // At y=-2.9, cone radius = mix(0.8, 4.5, 0.1/6) ≈ 0.86
        // Point at r=0.3 should be inside
        let sdf = generate_sdf(&v60_shapes(), BOUNDS, 64);
        let val = sample_sdf_trilinear(&sdf, BOUNDS, 64, Vec3::new(0.3, -2.9, 0.0));
        assert!(
            val > 0.0,
            "inside narrow cone bottom should be positive, got {val}"
        );
    }
}

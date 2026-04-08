use std::collections::HashMap;

use coffee_sim_core::sph::Vec3;

use super::FilterConfig;

#[derive(Clone)]
pub(crate) struct BedConfig {
    pub center: Vec3,
    pub top_y: f32,
    pub bot_y: f32,
    pub top_radius: f32,
    pub bot_radius: f32,
    pub num_particles: u32,
    pub initial_porosity: f32,
    pub initial_permeability: f32,
    pub extractable_mass: f32,
}

impl Default for BedConfig {
    fn default() -> Self {
        Self {
            // Keep the default bed comfortably inside the V60 cone:
            // absolute y range ~= [-2.6, 0.1], with radii that stay inset from
            // the dripper wall all the way down to the outlet.
            center: Vec3::new(0.0, -0.35, 0.0),
            top_y: 0.45,
            bot_y: -2.25,
            top_radius: 2.65,
            bot_radius: 0.95,
            num_particles: 12_000,
            initial_porosity: 0.4,
            initial_permeability: 1.0,
            extractable_mass: 0.15,
        }
    }
}

impl BedConfig {
    pub(crate) fn seated_in_filter(filter: &FilterConfig) -> Self {
        let mut bed = BedConfig {
            center: filter.center,
            ..Self::default()
        };

        // `filter.bot_y`/`top_y` are stored relative to `filter.center.y`
        // (see `FilterMesh::new`), so convert the filter interior to absolute
        // coordinates before clamping the bed against it.
        let filter_bot_abs = filter.center.y + filter.bot_y;
        let filter_top_abs = filter.center.y + filter.top_y;

        // `f32::clamp` panics when `min > max`, so fall back to `(min + max) * 0.5`
        // whenever the filter is too narrow to host the bed with the requested margins.
        let (top_min, top_max) = order_bounds(filter_bot_abs + 0.6, filter_top_abs - 0.35);
        let top_abs = (bed.center.y + bed.top_y).clamp(top_min, top_max);

        let (bot_min, bot_max) = order_bounds(filter_bot_abs + 0.4, top_abs - 1.4);
        let bot_abs = (bed.center.y + bed.bot_y).clamp(bot_min, bot_max);

        bed.top_y = top_abs - bed.center.y;
        bed.bot_y = bot_abs - bed.center.y;

        // `inner_radius_at_y` expects the `y` argument in the same frame as
        // `filter.top_y`/`filter.bot_y` (relative to `filter.center.y`), not in
        // absolute world coordinates.
        let top_local = top_abs - filter.center.y;
        let bot_local = bot_abs - filter.center.y;

        let (top_r_min, top_r_max) = order_bounds(
            filter.opening_radius() + 0.8,
            filter.top_radius - filter.thickness - 0.1,
        );
        bed.top_radius =
            (filter.inner_radius_at_y(top_local) - 0.18).clamp(top_r_min, top_r_max);

        let (bot_r_min, bot_r_max) =
            order_bounds(filter.opening_radius() + 0.32, bed.top_radius - 0.25);
        bed.bot_radius =
            (filter.inner_radius_at_y(bot_local) - 0.12).clamp(bot_r_min, bot_r_max);

        bed
    }
}

fn order_bounds(min: f32, max: f32) -> (f32, f32) {
    if min <= max {
        (min, max)
    } else {
        let mid = (min + max) * 0.5;
        (mid, mid)
    }
}

pub(crate) struct BedInit {
    pub particles: Vec<[f32; 8]>,
    pub affines: Vec<[f32; 12]>,
    pub bed_extracts: Vec<[f32; 8]>,
    pub cell_lookup: Vec<i32>,
    pub bed_support_count: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct LatticeKey {
    ix: i32,
    iy: i32,
    iz: i32,
}

pub(crate) fn init_bed_particles(
    config: &BedConfig,
    grid_dims: [u32; 3],
    bounds_size: Vec3,
) -> BedInit {
    let height = config.top_y - config.bot_y;
    if height < 1e-6 {
        return BedInit {
            particles: vec![],
            affines: vec![],
            bed_extracts: vec![],
            cell_lookup: vec![-1; (grid_dims[0] * grid_dims[1] * grid_dims[2]) as usize],
            bed_support_count: vec![],
        };
    }

    let avg_radius = (config.top_radius + config.bot_radius) * 0.5;
    let volume = std::f32::consts::PI * avg_radius * avg_radius * height / 3.0
        * (1.0
            + config.bot_radius / avg_radius
            + (config.bot_radius / avg_radius).powi(2));
    let spacing = (volume / config.num_particles.max(1) as f32).cbrt();

    let nx = ((config.top_radius * 2.0) / spacing).ceil() as i32;
    let ny = (height / spacing).ceil() as i32;
    let nz = nx;

    let mut particles = Vec::new();
    let mut affines = Vec::new();
    let mut bed_extracts = Vec::new();
    let mut lattice_to_particle = HashMap::new();

    for iy in 0..ny {
        let y = config.center.y + config.bot_y + (iy as f32 + 0.5) * spacing;
        let t = (y - (config.center.y + config.bot_y)) / height;
        let max_r = config.bot_radius + (config.top_radius - config.bot_radius) * t;

        for ix in 0..nx {
            let x = config.center.x - max_r + (ix as f32 + 0.5) * spacing;
            for iz in 0..nz {
                let z = config.center.z - max_r + (iz as f32 + 0.5) * spacing;

                let dx = x - config.center.x;
                let dz = z - config.center.z;
                let r = (dx * dx + dz * dz).sqrt();
                if r > max_r {
                    continue;
                }

                let particle_index = particles.len() as i32;
                lattice_to_particle.insert(LatticeKey { ix, iy, iz }, particle_index);

                // Particle: pos(x,y,z,J=1), vel(0,0,0,mass=1)
                particles.push([x, y, z, 1.0, 0.0, 0.0, 0.0, 1.0]);
                // Phase=1.0 means bed particle.
                affines.push([0.0, 0.0, 0.0, 1.0, x, y, z, 0.0, y, 0.0, 0.0, 0.0]);
                // BedExtract: bed(pore_water, porosity, permeability, compaction),
                //             extract(extractable, dissolved, temp, saturation)
                bed_extracts.push([
                    0.0,
                    config.initial_porosity,
                    config.initial_permeability,
                    0.0,
                    config.extractable_mass,
                    0.0,
                    93.0,
                    0.0,
                ]);
            }
        }
    }

    let target_n = config.num_particles as usize;
    if particles.len() > target_n {
        particles.truncate(target_n);
        affines.truncate(target_n);
        bed_extracts.truncate(target_n);
        lattice_to_particle.retain(|_, idx| (*idx as usize) < target_n);
    }

    let cell_lookup = build_cell_lookup(
        config,
        spacing,
        nx,
        ny,
        nz,
        &lattice_to_particle,
        grid_dims,
        bounds_size,
    );
    let bed_support_count = build_support_counts(&cell_lookup, particles.len());

    BedInit {
        particles,
        affines,
        bed_extracts,
        cell_lookup,
        bed_support_count,
    }
}

fn build_support_counts(cell_lookup: &[i32], num_particles: usize) -> Vec<u32> {
    let mut counts = vec![0_u32; num_particles];
    for &entry in cell_lookup {
        if entry >= 0 {
            let idx = entry as usize;
            if idx < counts.len() {
                counts[idx] += 1;
            }
        }
    }
    counts
}

fn build_cell_lookup(
    config: &BedConfig,
    spacing: f32,
    nx: i32,
    ny: i32,
    nz: i32,
    lattice_to_particle: &HashMap<LatticeKey, i32>,
    grid_dims: [u32; 3],
    bounds_size: Vec3,
) -> Vec<i32> {
    let [gx, gy, gz] = grid_dims;
    let mut lookup = vec![-1; (gx * gy * gz) as usize];
    let grid_origin = Vec3::new(-bounds_size.x * 0.5, -bounds_size.y * 0.5, -bounds_size.z * 0.5);
    let dx = bounds_size.x / gx as f32;
    let height = config.top_y - config.bot_y;
    let bed_bottom = config.center.y + config.bot_y;

    for iz in 0..gz {
        for iy in 0..gy {
            for ix in 0..gx {
                let pos = Vec3::new(
                    grid_origin.x + (ix as f32 + 0.5) * dx,
                    grid_origin.y + (iy as f32 + 0.5) * dx,
                    grid_origin.z + (iz as f32 + 0.5) * dx,
                );

                if pos.y < bed_bottom || pos.y > config.center.y + config.top_y {
                    continue;
                }

                let t = ((pos.y - bed_bottom) / height).clamp(0.0, 1.0);
                let max_r = config.bot_radius + (config.top_radius - config.bot_radius) * t;
                let dxr = pos.x - config.center.x;
                let dzr = pos.z - config.center.z;
                if (dxr * dxr + dzr * dzr).sqrt() > max_r {
                    continue;
                }

                let iy_guess = (((pos.y - bed_bottom) / spacing) - 0.5).round() as i32;
                let ix_guess = (((pos.x - (config.center.x - max_r)) / spacing) - 0.5).round() as i32;
                let iz_guess = (((pos.z - (config.center.z - max_r)) / spacing) - 0.5).round() as i32;

                let mut best = -1;
                let mut best_dist2 = f32::INFINITY;
                for dy in -1..=1 {
                    for dx_idx in -1..=1 {
                        for dz_idx in -1..=1 {
                            let key = LatticeKey {
                                ix: (ix_guess + dx_idx).clamp(0, nx.saturating_sub(1)),
                                iy: (iy_guess + dy).clamp(0, ny.saturating_sub(1)),
                                iz: (iz_guess + dz_idx).clamp(0, nz.saturating_sub(1)),
                            };
                            let Some(&particle_idx) = lattice_to_particle.get(&key) else {
                                continue;
                            };
                            let py = bed_bottom + (key.iy as f32 + 0.5) * spacing;
                            let py_t = ((py - bed_bottom) / height).clamp(0.0, 1.0);
                            let py_r = config.bot_radius + (config.top_radius - config.bot_radius) * py_t;
                            let px = config.center.x - py_r + (key.ix as f32 + 0.5) * spacing;
                            let pz = config.center.z - py_r + (key.iz as f32 + 0.5) * spacing;
                            let ddx = pos.x - px;
                            let ddy = pos.y - py;
                            let ddz = pos.z - pz;
                            let dist2 = ddx * ddx + ddy * ddy + ddz * ddz;
                            if dist2 < best_dist2 {
                                best_dist2 = dist2;
                                best = particle_idx;
                            }
                        }
                    }
                }

                let flat = (iz * gx * gy + iy * gx + ix) as usize;
                lookup[flat] = best;
            }
        }
    }

    lookup
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> BedConfig {
        BedConfig {
            num_particles: 200,
            ..BedConfig::default()
        }
    }

    #[test]
    fn empty_bed_when_height_collapses() {
        let mut cfg = small_config();
        cfg.top_y = 0.0;
        cfg.bot_y = 0.0;
        let init = init_bed_particles(&cfg, [16, 16, 16], Vec3::new(8.0, 8.0, 8.0));
        assert!(init.particles.is_empty());
        assert!(init.affines.is_empty());
        assert!(init.bed_extracts.is_empty());
        assert_eq!(init.cell_lookup.len(), 16 * 16 * 16);
        assert!(init.bed_support_count.is_empty());
        assert!(init.cell_lookup.iter().all(|v| *v == -1));
    }

    #[test]
    fn produces_at_most_target_particles() {
        let cfg = small_config();
        let init = init_bed_particles(&cfg, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));
        assert!(init.particles.len() <= cfg.num_particles as usize);
        assert_eq!(init.particles.len(), init.affines.len());
        assert_eq!(init.particles.len(), init.bed_extracts.len());
        assert_eq!(init.particles.len(), init.bed_support_count.len());
    }

    #[test]
    fn bed_particles_carry_water_phase_marker() {
        let cfg = small_config();
        let init = init_bed_particles(&cfg, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));
        for affine in &init.affines {
            // affine[3] is the phase slot (col0.w); bed particles must be 1.0
            assert_eq!(affine[3], 1.0);
        }
    }

    #[test]
    fn bed_extracts_initialised_within_bounds() {
        let cfg = small_config();
        let init = init_bed_particles(&cfg, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));
        for extract in &init.bed_extracts {
            // bed: pore_water, porosity, permeability, compaction
            assert_eq!(extract[0], 0.0);
            assert!((extract[1] - cfg.initial_porosity).abs() < 1e-6);
            assert!((extract[2] - cfg.initial_permeability).abs() < 1e-6);
            assert_eq!(extract[3], 0.0);
            // extract: extractable, dissolved, temp, saturation
            assert!((extract[4] - cfg.extractable_mass).abs() < 1e-6);
            assert_eq!(extract[5], 0.0);
            assert_eq!(extract[7], 0.0);
        }
    }

    #[test]
    fn cell_lookup_only_indexes_existing_particles() {
        let cfg = small_config();
        let init = init_bed_particles(&cfg, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));
        let n = init.particles.len() as i32;
        for value in &init.cell_lookup {
            assert!(*value < n, "lookup {} >= particle count {}", value, n);
            assert!(*value >= -1);
        }
        let any_indexed = init.cell_lookup.iter().any(|v| *v >= 0);
        assert!(any_indexed, "expected at least one bed-occupied cell");
    }

    #[test]
    fn bed_support_count_matches_lookup_fanout() {
        let cfg = small_config();
        let init = init_bed_particles(&cfg, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));
        let mut recomputed = vec![0_u32; init.particles.len()];
        for &entry in &init.cell_lookup {
            if entry >= 0 {
                recomputed[entry as usize] += 1;
            }
        }
        assert_eq!(init.bed_support_count, recomputed);
    }

    #[test]
    fn default_bed_sits_inside_filter_interior() {
        let filter = FilterConfig::default();
        let bed = BedConfig::seated_in_filter(&filter);

        let filter_bot_abs = filter.center.y + filter.bot_y;
        let filter_top_abs = filter.center.y + filter.top_y;

        let top_abs = bed.center.y + bed.top_y;
        let bot_abs = bed.center.y + bed.bot_y;

        // `radius_at_y` expects the input in the same frame as `filter.top_y`/
        // `filter.bot_y` — i.e. relative to `filter.center.y`. Convert the
        // absolute bed top/bot into that frame before sampling the inner cone.
        let bed_top_local = top_abs - filter.center.y;
        let bed_bot_local = bot_abs - filter.center.y;
        assert!(bed.top_radius < filter.inner_radius_at_y(bed_top_local));
        assert!(bed.bot_radius < filter.inner_radius_at_y(bed_bot_local));
        assert!(bot_abs > filter_bot_abs);
        assert!(top_abs < filter_top_abs);
    }

    #[test]
    fn seated_in_filter_does_not_panic_on_narrow_filter() {
        // Pathologically narrow filter that cannot actually host a bed:
        // - vertical range is 0.2 (< 0.95), which previously caused the first
        //   clamp to panic with `min > max`.
        // - top radius is below the minimum the second clamp expects.
        let narrow = FilterConfig {
            top_y: 0.1,
            bot_y: -0.1,
            top_radius: 0.3,
            bot_radius: 0.2,
            thickness: 0.02,
            hole_radius: 0.05,
            ..FilterConfig::default()
        };
        let bed = BedConfig::seated_in_filter(&narrow);
        assert!(bed.top_y.is_finite());
        assert!(bed.bot_y.is_finite());
        assert!(bed.top_radius.is_finite());
        assert!(bed.bot_radius.is_finite());
    }

    #[test]
    fn seated_in_filter_uses_filter_center_offset() {
        // A filter whose center is offset vertically must still place the bed
        // above the filter apex in absolute coordinates. The earlier
        // implementation clamped against `filter.bot_y`/`top_y` as if they
        // were world coordinates even though `FilterMesh::new` treats them as
        // relative to `filter.center.y`.
        let filter = FilterConfig {
            center: Vec3::new(0.0, 2.0, 0.0),
            ..FilterConfig::default()
        };
        let bed = BedConfig::seated_in_filter(&filter);

        let bed_top_abs = bed.center.y + bed.top_y;
        let bed_bot_abs = bed.center.y + bed.bot_y;
        let filter_top_abs = filter.center.y + filter.top_y;
        let filter_bot_abs = filter.center.y + filter.bot_y;

        assert!(bed_top_abs <= filter_top_abs - 0.3);
        assert!(bed_bot_abs >= filter_bot_abs + 0.3);
    }
}

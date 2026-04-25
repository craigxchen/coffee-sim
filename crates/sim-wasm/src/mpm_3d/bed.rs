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
    pub mean_grind_size: f32,
    pub grind_size_spread: f32,
    pub fines_fraction: f32,
    pub spawn_drop_height: f32,
    pub spawn_radius_scale: f32,
    pub spawn_jitter: f32,
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
            initial_permeability: 0.002,
            mean_grind_size: 1.0,
            grind_size_spread: 0.32,
            fines_fraction: 0.18,
            spawn_drop_height: 2.8,
            spawn_radius_scale: 0.82,
            spawn_jitter: 0.18,
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
        let (top_min, top_max) = order_bounds(filter_bot_abs + 0.35, filter_top_abs - 0.35);
        let top_abs = ((bed.center.y + bed.top_y) - 0.22).clamp(top_min, top_max);

        // Start the dry bed already packed close to the filter apex instead of
        // with a large flat-bottom gap that would require granular settling to
        // fill. The current solid model intentionally keeps dry grounds fairly
        // immobile, so the initial geometry needs to be seated against the
        // filter rather than expecting later motion to do that packing.
        let (bot_min, bot_max) = order_bounds(filter_bot_abs + 0.22, top_abs - 1.6);
        let bot_abs = bot_min.min(bot_max);

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
        bed.top_radius = (filter.inner_radius_at_y(top_local) - 0.18).clamp(top_r_min, top_r_max);

        let (bot_r_min, bot_r_max) =
            order_bounds(filter.opening_radius() + 0.06, bed.top_radius - 0.25);
        bed.bot_radius = (filter.inner_radius_at_y(bot_local) - 0.04).clamp(bot_r_min, bot_r_max);

        // A preloaded bed should start seated in the filter, not as a second
        // "grounds being poured" scenario. Poured grounds can use a different
        // config path with nonzero drop height and spawn jitter.
        bed.spawn_drop_height = 0.0;
        bed.spawn_radius_scale = 1.0;
        bed.spawn_jitter = 0.0;

        bed
    }
}

fn uses_preloaded_spawn(config: &BedConfig) -> bool {
    config.spawn_drop_height.abs() <= f32::EPSILON
        && (config.spawn_radius_scale - 1.0).abs() <= f32::EPSILON
        && config.spawn_jitter.abs() <= f32::EPSILON
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
    pub bed_extracts: Vec<[f32; 20]>,
    pub cell_lookup: Vec<i32>,
    pub bed_support_count: Vec<u32>,
}

#[derive(Clone, Copy)]
struct BedHydraulicProps {
    porosity: f32,
    permeability: f32,
    capacity_scale: f32,
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
        * (1.0 + config.bot_radius / avg_radius + (config.bot_radius / avg_radius).powi(2));
    let spacing = (volume / config.num_particles.max(1) as f32).cbrt();

    let nx = ((config.top_radius * 2.0) / spacing).ceil() as i32;
    let ny = (height / spacing).ceil() as i32;
    let nz = nx;

    let mut particles = Vec::new();
    let mut affines = Vec::new();
    let mut bed_extracts = Vec::new();
    let mut lattice_to_particle = HashMap::new();

    let preloaded_spawn = uses_preloaded_spawn(config);

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
                let hydraulic = sample_bed_hydraulics(config, ix, iy, iz);
                let spawn_jitter = sample_spawn_jitter(config, ix, iy, iz, spacing);
                let spawn_scale = if preloaded_spawn {
                    1.0
                } else {
                    (config.spawn_radius_scale + (1.0 - t) * 0.06).clamp(0.55, 1.1)
                };
                let spawn_x = config.center.x + dx * spawn_scale + spawn_jitter.x;
                let spawn_y = if preloaded_spawn {
                    y
                } else {
                    y + config.spawn_drop_height + spawn_jitter.y + (1.0 - t) * spacing * 0.35
                };
                let spawn_z = config.center.z + dz * spawn_scale + spawn_jitter.z;

                // Particle: pos(x,y,z,J=1), vel(0,0,0,mass=1)
                particles.push([spawn_x, spawn_y, spawn_z, 1.0, 0.0, 0.0, 0.0, 1.0]);
                // Phase=1.0 means bed particle. col1/col2 hold the APIC C
                // matrix after the first G2P pass, so zero-init them.
                affines.push([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                // BedExtract:
                //   bed(pore_water, porosity, permeability, capacity_scale)
                //   extract(extractable, dissolved, settled_activation, saturation_cache)
                //   mech0.xyz = elastic F col0, mech0.w = compaction
                //   mech1.xyz = elastic F col1, mech1.w = rest porosity
                //   mech2.xyz = elastic F col2, mech2.w = rest permeability
                bed_extracts.push([
                    0.0,
                    hydraulic.porosity,
                    hydraulic.permeability,
                    hydraulic.capacity_scale,
                    config.extractable_mass,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    hydraulic.porosity,
                    0.0,
                    0.0,
                    1.0,
                    hydraulic.permeability,
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

fn sample_bed_hydraulics(config: &BedConfig, ix: i32, iy: i32, iz: i32) -> BedHydraulicProps {
    let seed = mix_seed(ix, iy, iz);
    let base_noise = triangular_noise(seed);
    let grind_scale = sample_grind_scale(config, seed, base_noise);

    let porosity = (config.initial_porosity * (0.84 + 0.22 * grind_scale)).clamp(0.24, 0.58);
    let permeability = (config.initial_permeability * grind_scale.powf(2.35)).clamp(0.0002, 0.02);
    let capacity_scale = (1.10 - 0.28 * (grind_scale - 1.0)).clamp(0.72, 1.38);

    BedHydraulicProps {
        porosity,
        permeability,
        capacity_scale,
    }
}

fn sample_grind_scale(config: &BedConfig, seed: u64, base_noise: f32) -> f32 {
    let mut grind_scale =
        (config.mean_grind_size * (1.0 + config.grind_size_spread * base_noise)).clamp(0.35, 2.4);
    if hash_to_unit(seed ^ 0x517c_c1b7_d2e4_f91a) < config.fines_fraction.clamp(0.0, 0.75) {
        grind_scale *= 0.42;
    }
    grind_scale.clamp(0.25, 2.2)
}

fn sample_spawn_jitter(config: &BedConfig, ix: i32, iy: i32, iz: i32, spacing: f32) -> Vec3 {
    let seed = mix_seed(ix, iy, iz) ^ 0x6eed_0e9d_13f2_8a5b;
    let jitter_scale = config.spawn_jitter.max(0.0) * spacing;
    let x = triangular_noise(seed ^ 0x243f_6a88_85a3_08d3) * jitter_scale;
    let y = triangular_noise(seed ^ 0x1319_8a2e_0370_7344) * jitter_scale * 0.35;
    let z = triangular_noise(seed ^ 0xa409_3822_299f_31d0) * jitter_scale;
    Vec3::new(x, y, z)
}

fn triangular_noise(seed: u64) -> f32 {
    let a = hash_to_unit(seed ^ 0x9e37_79b9_7f4a_7c15);
    let b = hash_to_unit(seed ^ 0xbf58_476d_1ce4_e5b9);
    (a + b - 1.0).clamp(-1.0, 1.0)
}

fn mix_seed(ix: i32, iy: i32, iz: i32) -> u64 {
    let x = ix as u32 as u64;
    let y = iy as u32 as u64;
    let z = iz as u32 as u64;
    x.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ y.wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ z.wrapping_mul(0x94d0_49bb_1331_11eb)
}

fn hash_to_unit(seed: u64) -> f32 {
    let mut x = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    ((x >> 40) as u32) as f32 / ((1u32 << 24) as f32)
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
    let grid_origin = Vec3::new(
        -bounds_size.x * 0.5,
        -bounds_size.y * 0.5,
        -bounds_size.z * 0.5,
    );
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
                let ix_guess =
                    (((pos.x - (config.center.x - max_r)) / spacing) - 0.5).round() as i32;
                let iz_guess =
                    (((pos.z - (config.center.z - max_r)) / spacing) - 0.5).round() as i32;

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
                            let py_r =
                                config.bot_radius + (config.top_radius - config.bot_radius) * py_t;
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
            // bed: pore_water, porosity, permeability, capacity_scale
            assert_eq!(extract[0], 0.0);
            assert!(extract[1] >= 0.24 && extract[1] <= 0.58);
            assert!(extract[2] >= 0.0002 && extract[2] <= 0.02);
            assert!(extract[3] >= 0.72 && extract[3] <= 1.38);
            // extract: extractable, dissolved, settled activation, saturation cache
            assert!((extract[4] - cfg.extractable_mass).abs() < 1e-6);
            assert_eq!(extract[5], 0.0);
            assert_eq!(extract[6], 0.0);
            assert_eq!(extract[7], 0.0);
            // mech: F starts at identity for dry bed particles
            assert_eq!(extract[8], 1.0);
            assert_eq!(extract[13], 1.0);
            assert_eq!(extract[18], 1.0);
            assert!((extract[15] - extract[1]).abs() < 1e-6);
            assert!((extract[19] - extract[2]).abs() < 1e-6);
            assert_eq!(extract[11], 0.0);
        }
    }

    #[test]
    fn bed_hydraulics_are_non_uniform_by_default() {
        let cfg = small_config();
        let init = init_bed_particles(&cfg, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));
        let min_perm = init
            .bed_extracts
            .iter()
            .map(|extract| extract[2])
            .fold(f32::MAX, f32::min);
        let max_perm = init
            .bed_extracts
            .iter()
            .map(|extract| extract[2])
            .fold(f32::MIN, f32::max);
        let min_cap = init
            .bed_extracts
            .iter()
            .map(|extract| extract[3])
            .fold(f32::MAX, f32::min);
        let max_cap = init
            .bed_extracts
            .iter()
            .map(|extract| extract[3])
            .fold(f32::MIN, f32::max);

        assert!(max_perm > min_perm);
        assert!(max_cap > min_cap);
    }

    #[test]
    fn coarser_grind_raises_mean_permeability() {
        let mut fine = small_config();
        fine.mean_grind_size = 0.75;
        fine.fines_fraction = 0.28;
        let mut coarse = small_config();
        coarse.mean_grind_size = 1.35;
        coarse.fines_fraction = 0.08;

        let fine_init = init_bed_particles(&fine, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));
        let coarse_init = init_bed_particles(&coarse, [32, 32, 32], Vec3::new(14.0, 20.0, 14.0));

        let fine_mean_perm = fine_init
            .bed_extracts
            .iter()
            .map(|extract| extract[2])
            .sum::<f32>()
            / fine_init.bed_extracts.len().max(1) as f32;
        let coarse_mean_perm = coarse_init
            .bed_extracts
            .iter()
            .map(|extract| extract[2])
            .sum::<f32>()
            / coarse_init.bed_extracts.len().max(1) as f32;
        let fine_mean_capacity = fine_init
            .bed_extracts
            .iter()
            .map(|extract| extract[3])
            .sum::<f32>()
            / fine_init.bed_extracts.len().max(1) as f32;
        let coarse_mean_capacity = coarse_init
            .bed_extracts
            .iter()
            .map(|extract| extract[3])
            .sum::<f32>()
            / coarse_init.bed_extracts.len().max(1) as f32;

        assert!(coarse_mean_perm > fine_mean_perm);
        assert!(fine_mean_capacity > coarse_mean_capacity);
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
        assert!(bot_abs <= filter_bot_abs + 0.26);
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
        assert!(bed_bot_abs >= filter_bot_abs + 0.05);
        assert!(bed_bot_abs <= filter_bot_abs + 0.26);
    }

    #[test]
    fn seated_in_filter_starts_as_preloaded_bed() {
        let filter = FilterConfig::default();
        let bed = BedConfig::seated_in_filter(&filter);

        assert_eq!(bed.spawn_drop_height, 0.0);
        assert_eq!(bed.spawn_radius_scale, 1.0);
        assert_eq!(bed.spawn_jitter, 0.0);
    }

    #[test]
    fn preloaded_spawn_does_not_add_vertical_lift() {
        let filter = FilterConfig::default();
        let cfg = BedConfig::seated_in_filter(&filter);
        let init = init_bed_particles(&cfg, [48, 48, 48], Vec3::new(16.0, 16.0, 16.0));
        let max_y = init
            .particles
            .iter()
            .map(|particle| particle[1])
            .fold(f32::MIN, f32::max);

        assert!(
            max_y <= cfg.center.y + cfg.top_y,
            "preloaded spawn still injected an upward lift: max_y={max_y} top_y={}",
            cfg.center.y + cfg.top_y,
        );
    }
}

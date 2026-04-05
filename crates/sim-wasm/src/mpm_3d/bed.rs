use std::collections::HashMap;

use coffee_sim_core::sph::Vec3;

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
            num_particles: 5_000,
            initial_porosity: 0.4,
            initial_permeability: 1.0,
            extractable_mass: 0.15,
        }
    }
}

pub(crate) struct BedInit {
    pub particles: Vec<[f32; 8]>,
    pub affines: Vec<[f32; 12]>,
    pub bed_extracts: Vec<[f32; 8]>,
    pub cell_lookup: Vec<i32>,
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
                affines.push([0.0, 0.0, 0.0, 1.0, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0]);
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

    BedInit {
        particles,
        affines,
        bed_extracts,
        cell_lookup,
    }
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

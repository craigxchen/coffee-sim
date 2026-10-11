//! Deterministic particle/MAC reference implementation. No device or readback required.

use glam::DVec3;

use crate::emission::{EmissionInput, PourEvent};
use crate::engine::{scene::Species, Scene};
use crate::models::{extraction, permeability, wetting, Materials};
use crate::utils::{config::Config, rng::Rng, sdf};

use super::grid::{Grid, Mobility};

#[derive(Clone, Debug)]
pub struct WaterPoint {
    pub position: DVec3,
    pub velocity: DVec3,
    pub volume: f64,
    pub solute: f64,
    /// Sensible heat in the repository's normalized temperature units.
    pub heat: f64,
}

#[derive(Clone, Debug)]
pub struct GrainPoint {
    pub position: DVec3,
    pub velocity: DVec3,
    pub dry_volume: f64,
    pub initial_mass: f64,
    pub absorbed: f64,
    pub capacity: f64,
    pub fast: f64,
    pub slow: f64,
    pub insoluble: f64,
    /// Dissolved material carried into the intragranular water by absorption.
    pub retained_solute: f64,
    pub heat: f64,
}

impl GrainPoint {
    pub fn moisture(&self) -> f64 {
        if self.capacity > 0.0 {
            (self.absorbed / self.capacity).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn mass(&self, rho: f64) -> f64 {
        self.insoluble + self.fast + self.slow + self.retained_solute + rho * self.absorbed
    }
}

/// All inventories include material already drained through the optional open bottom.
/// Water and solute are separate conserved substances; absorbed water is counted once.
#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub time: f64,
    pub free_water: f64,
    pub absorbed_water: f64,
    pub drained_water: f64,
    pub emitted_water: f64,
    pub dissolved_solute: f64,
    pub retained_solute: f64,
    pub remaining_solute: f64,
    pub drained_solute: f64,
    pub water_error: f64,
    pub solute_error: f64,
    /// Complementarity residual, normalized as a fraction of cell volume per substep.
    pub pressure_residual: f64,
    pub pressure_iterations: u32,
    pub substeps: u32,
    pub momentum: DVec3,
    pub drained_momentum: DVec3,
    /// Accumulated impulses delivered to fixed boundaries or a frozen skeleton.
    pub boundary_impulse: DVec3,
    /// Independent global momentum balance. A nonzero value exposes transfer/operator
    /// error; it is never reclassified as an external boundary reaction.
    pub momentum_error: DVec3,
    pub heat: f64,
}

pub struct MacSimulation {
    pub(super) scene: Scene,
    pub(super) mats: Materials,
    pub(super) cfg: Config,
    pub(super) grid: Grid,
    pub(super) water: Vec<WaterPoint>,
    pub(super) grains: Vec<GrainPoint>,
    time: f64,
    initial_water: f64,
    initial_solute: f64,
    emitted_water: f64,
    drained_water: f64,
    drained_solute: f64,
    drained_heat: f64,
    drained_momentum: DVec3,
    boundary_impulse: DVec3,
    gravity_impulse: DVec3,
    emitted_momentum: DVec3,
    pressure_residual: f64,
    pressure_iterations: u32,
    substeps: u32,
    rng: Rng,
}

impl MacSimulation {
    pub fn new(scene: &Scene, mats: &Materials, cfg: &Config) -> Self {
        cfg.mac.validate();
        assert!(mats.particle_spacing.is_finite() && mats.particle_spacing > 0.0);
        assert!(mats.grain_diameter.is_finite() && mats.grain_diameter > 0.0);
        assert!(mats.particle_mass.is_finite() && mats.particle_mass > 0.0);
        assert!(mats.grain_mass.is_finite() && mats.grain_mass > 0.0);
        assert!(scene
            .box_min
            .iter()
            .chain(&scene.box_max)
            .chain(&scene.gravity)
            .all(|x| x.is_finite()));
        let h = if cfg.mac.cell_size > 0.0 {
            cfg.mac.cell_size
        } else {
            2.0 * mats.particle_spacing as f64
        };
        let grid = Grid::new(scene, h, cfg.mac.open_bottom);
        let mut s = Self {
            scene: scene.clone(),
            mats: mats.clone(),
            cfg: cfg.clone(),
            grid,
            water: Vec::new(),
            grains: Vec::new(),
            time: 0.0,
            initial_water: 0.0,
            initial_solute: 0.0,
            emitted_water: 0.0,
            drained_water: 0.0,
            drained_solute: 0.0,
            drained_heat: 0.0,
            drained_momentum: DVec3::ZERO,
            boundary_impulse: DVec3::ZERO,
            gravity_impulse: DVec3::ZERO,
            emitted_momentum: DVec3::ZERO,
            pressure_residual: 0.0,
            pressure_iterations: 0,
            substeps: 0,
            rng: Rng::new(cfg.seed),
        };
        // Half-open, cell-centered seeding avoids double-counting region boundary layers.
        for region in &scene.regions {
            let spacing = match region.species {
                Species::Water => mats.particle_spacing,
                Species::Grain => mats.grain_diameter,
            } as f64;
            let lo = DVec3::from_array(region.min.map(f64::from));
            let hi = DVec3::from_array(region.max.map(f64::from));
            let count = ((hi - lo) / spacing)
                .floor()
                .to_array()
                .map(|v| v.max(0.0) as usize);
            for z in 0..count[2] {
                for y in 0..count[1] {
                    for x in 0..count[0] {
                        let mut p = lo
                            + spacing * DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
                        for a in 0..3 {
                            p[a] += spacing
                                * cfg.seed_jitter as f64
                                * (2.0 * s.rng.next_f32() as f64 - 1.0);
                        }
                        let phase = u32::from(region.species == Species::Grain);
                        if !s.inside(p)
                            || sdf::nearest(&scene.solids, p.as_vec3(), phase).signed < 0.0
                        {
                            continue;
                        }
                        if phase == 0 {
                            s.push_water(p, DVec3::ZERO, spacing.powi(3));
                        } else {
                            let volume = std::f64::consts::PI / 6.0 * spacing.powi(3);
                            let mass = mats.grain_mass as f64;
                            let (fast, slow) = extraction::split(
                                mats.soluble_fraction * mats.grain_mass,
                                mats.fast_fraction,
                            );
                            // The actual dry mass and liquid density determine the
                            // gravimetric capacity; a separately tuned density ratio
                            // must not silently disagree with the mechanical masses.
                            let density_ratio = mass / (s.rho() * volume);
                            let capacity =
                                wetting::capacity(volume as f32, mats.r_max, density_ratio as f32)
                                    as f64;
                            let absorbed = capacity * cfg.mac.initial_moisture;
                            let mut grain = GrainPoint {
                                position: p,
                                velocity: DVec3::ZERO,
                                dry_volume: volume,
                                initial_mass: mass,
                                absorbed,
                                capacity,
                                fast: fast as f64,
                                slow: slow as f64,
                                insoluble: mass - fast as f64 - slow as f64,
                                retained_solute: 0.0,
                                heat: 0.0,
                            };
                            grain.heat = s.grain_heat_capacity(&grain) * mats.t_amb as f64;
                            s.grains.push(grain);
                        }
                    }
                }
            }
        }
        s.initial_water = s.water.iter().map(|p| p.volume).sum::<f64>()
            + s.grains.iter().map(|g| g.absorbed).sum::<f64>();
        s.initial_solute = s.grains.iter().map(|g| g.fast + g.slow).sum();
        s.scatter();
        s
    }

    pub fn water(&self) -> &[WaterPoint] {
        &self.water
    }
    pub fn grains(&self) -> &[GrainPoint] {
        &self.grains
    }
    pub fn cell_size(&self) -> f64 {
        self.grid.h
    }

    /// Pore saturation, distinct from the grains' intragranular moisture.
    pub fn saturation_at(&self, p: DVec3) -> f64 {
        self.grid
            .stencil(p, None)
            .iter()
            .map(|&(i, w)| {
                let c = self.grid.cells[i];
                w * (c.water / (self.grid.h.powi(3) - c.solid).max(1e-12)).clamp(0.0, 1.0)
            })
            .sum()
    }

    pub fn diagnostics(&self) -> Diagnostics {
        let free_water = self.water.iter().map(|p| p.volume).sum::<f64>();
        let absorbed_water = self.grains.iter().map(|g| g.absorbed).sum::<f64>();
        let dissolved_solute = self.water.iter().map(|p| p.solute).sum::<f64>();
        let retained_solute = self.grains.iter().map(|g| g.retained_solute).sum::<f64>();
        let remaining_solute = self.grains.iter().map(|g| g.fast + g.slow).sum::<f64>();
        Diagnostics {
            time: self.time,
            free_water,
            absorbed_water,
            drained_water: self.drained_water,
            emitted_water: self.emitted_water,
            dissolved_solute,
            retained_solute,
            remaining_solute,
            drained_solute: self.drained_solute,
            water_error: free_water + absorbed_water + self.drained_water
                - self.initial_water
                - self.emitted_water,
            solute_error: dissolved_solute
                + retained_solute
                + remaining_solute
                + self.drained_solute
                - self.initial_solute,
            pressure_residual: self.pressure_residual,
            pressure_iterations: self.pressure_iterations,
            substeps: self.substeps,
            momentum: self.momentum(),
            drained_momentum: self.drained_momentum,
            boundary_impulse: self.boundary_impulse,
            momentum_error: self.momentum() + self.drained_momentum + self.boundary_impulse
                - self.gravity_impulse
                - self.emitted_momentum,
            heat: self.water.iter().map(|p| p.heat).sum::<f64>()
                + self.grains.iter().map(|g| g.heat).sum::<f64>()
                + self.drained_heat,
        }
    }

    pub(super) fn rho(&self) -> f64 {
        self.mats.particle_mass as f64 / (self.mats.particle_spacing as f64).powi(3)
    }
    pub(super) fn water_mass(&self, p: &WaterPoint) -> f64 {
        self.rho() * p.volume + p.solute
    }
    pub(super) fn water_heat_capacity(&self, p: &WaterPoint) -> f64 {
        self.rho() * p.volume * self.mats.cp_water as f64 + p.solute * self.mats.cp_grain as f64
    }
    pub(super) fn grain_heat_capacity(&self, g: &GrainPoint) -> f64 {
        (g.mass(self.rho()) - self.rho() * g.absorbed) * self.mats.cp_grain as f64
            + self.rho() * g.absorbed * self.mats.cp_water as f64
    }
    pub(super) fn grain_volume(&self, g: &GrainPoint) -> f64 {
        g.dry_volume * (1.0 + self.cfg.mac.swelling_ratio * g.moisture())
    }
    fn momentum(&self) -> DVec3 {
        self.water
            .iter()
            .map(|p| self.water_mass(p) * p.velocity)
            .sum::<DVec3>()
            + self
                .grains
                .iter()
                .map(|g| g.mass(self.rho()) * g.velocity)
                .sum::<DVec3>()
    }
    fn inside(&self, p: DVec3) -> bool {
        (0..3).all(|a| p[a] >= self.scene.box_min[a] as f64 && p[a] < self.scene.box_max[a] as f64)
    }
    fn push_water(&mut self, position: DVec3, velocity: DVec3, volume: f64) {
        self.water.push(WaterPoint {
            position,
            velocity,
            volume,
            solute: 0.0,
            heat: volume * self.rho() * self.mats.cp_water as f64 * self.mats.pour_t as f64,
        });
    }

    /// Advance the WHOLE requested interval. CFL is re-evaluated after each accepted
    /// substep, including gravity and inlet speed. No silent dt truncation or speed clamp.
    pub fn advance(&mut self, dt: f64, input: &EmissionInput) {
        if input.event == PourEvent::Reset {
            *self = Self::new(&self.scene.clone(), &self.mats.clone(), &self.cfg.clone());
            return;
        }
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.pressure_iterations = 0;
        self.pressure_residual = 0.0;
        self.substeps = 0;
        let mut remaining = dt;
        while remaining > dt * 1e-12 {
            let speed = self
                .water
                .iter()
                .map(|p| p.velocity.length())
                .chain(self.grains.iter().map(|g| g.velocity.length()))
                .fold(self.inlet_speed(input), f64::max);
            let g = DVec3::from_array(self.scene.gravity.map(f64::from)).length();
            let distance = self.cfg.mac.cfl * self.grid.h.min(self.mats.grain_diameter as f64);
            let safe_dt =
                2.0 * distance / (speed + (speed * speed + 2.0 * g * distance).sqrt()).max(1e-12);
            let step = remaining.min(self.cfg.mac.max_dt).min(safe_dt);
            self.emit(step, input);
            // Absorption changes both particle masses and pore space BEFORE pressure assembly.
            self.exchange(step);
            self.scatter();
            self.mechanics(step);
            self.move_particles(step);
            self.time += step;
            remaining -= step;
            self.substeps += 1;
        }
        let pressure: Vec<f64> = self.grid.cells.iter().map(|c| c.pressure).collect();
        self.scatter();
        for (c, p) in self.grid.cells.iter_mut().zip(pressure) {
            c.pressure = p;
        }
    }

    fn inlet_speed(&self, input: &EmissionInput) -> f64 {
        if !input.flow_rate.is_finite() || input.flow_rate <= 0.0 {
            return 0.0;
        }
        input.flow_rate as f64
            / (std::f64::consts::PI
                * (self.cfg.nozzle_radius as f64).powi(2)
                * self.cfg.discharge_coeff as f64)
                .max(1e-12)
    }

    fn emit(&mut self, dt: f64, input: &EmissionInput) {
        if input.flow_rate <= 0.0
            || !input.flow_rate.is_finite()
            || !input.pour_angle.is_finite()
            || !input.kettle_pos.iter().all(|x| x.is_finite())
        {
            return;
        }
        let center = DVec3::from_array(input.kettle_pos.map(f64::from));
        if !self.inside(center)
            || sdf::nearest(&self.scene.solids, center.as_vec3(), 0).signed < 0.0
        {
            return;
        }
        let volume = input.flow_rate as f64 * dt;
        let count = (volume / (self.mats.particle_spacing as f64).powi(3))
            .ceil()
            .max(1.0) as usize;
        let angle = input.pour_angle as f64;
        let velocity = self.inlet_speed(input) * DVec3::new(angle.sin(), -angle.cos(), 0.0);
        for _ in 0..count {
            let radius = self.cfg.nozzle_radius as f64 * (self.rng.next_f32() as f64).sqrt();
            let azimuth = std::f64::consts::TAU * self.rng.next_f32() as f64;
            let p = center + radius * DVec3::new(azimuth.cos(), 0.0, azimuth.sin());
            // A clipped nozzle deposits its exact requested volume at the valid center.
            let p = if self.inside(p)
                && sdf::nearest(&self.scene.solids, p.as_vec3(), 0).signed >= 0.0
            {
                p
            } else {
                center
            };
            self.push_water(p, velocity, volume / count as f64);
        }
        self.emitted_water += volume;
        self.emitted_momentum += self.rho() * volume * velocity;
    }

    pub(super) fn scatter(&mut self) {
        let rho = self.rho();
        self.grid.clear();
        for p in &self.water {
            for (i, w) in self.grid.stencil(p.position, None) {
                self.grid.cells[i].water += w * p.volume;
            }
            let mass = self.water_mass(p);
            for axis in 0..3 {
                for (i, w) in self.grid.stencil(p.position, Some(axis)) {
                    let f = &mut self.grid.faces[i];
                    f.mass[0] += w * mass;
                    f.volume[0] += w * p.volume;
                    f.momentum[0] += w * mass * p.velocity[axis];
                }
            }
        }
        for g in &self.grains {
            let volume = self.grain_volume(g);
            for (i, w) in self.grid.stencil(g.position, None) {
                self.grid.cells[i].solid += w * volume;
            }
            for axis in 0..3 {
                for (i, w) in self.grid.stencil(g.position, Some(axis)) {
                    let f = &mut self.grid.faces[i];
                    f.mass[1] += w * g.mass(rho);
                    f.volume[1] += w * volume;
                    f.momentum[1] += w * g.mass(rho) * g.velocity[axis];
                }
            }
        }
    }

    fn mechanics(&mut self, dt: f64) {
        let gravity = DVec3::from_array(self.scene.gravity.map(f64::from));
        let total_mass = self.water.iter().map(|p| self.water_mass(p)).sum::<f64>()
            + self.grains.iter().map(|g| g.mass(self.rho())).sum::<f64>();
        let h = self.grid.h;
        for f in &mut self.grid.faces {
            let phi = (1.0 - f.volume[1] / h.powi(3)).clamp(0.02, 1.0);
            let beta = if f.mass[1] > 0.0 {
                permeability::darcy_drag_rate(
                    self.mats.grain_diameter,
                    phi as f32,
                    self.cfg.drag_scale,
                ) as f64
            } else {
                0.0
            };
            f.mobility = Mobility::new(
                f.mass[0],
                f.mass[1],
                dt * f.mass[0] * beta,
                f.blocked[0],
                f.blocked[1] || self.cfg.mac.frozen_bed,
            );
            f.drag_dt = if f.mass[0] > 0.0 && f.mass[1] > 0.0 {
                dt * f.mass[0] * beta
            } else {
                0.0
            };
            // Pressure integrates traction over a FACE, not over the noisy particle
            // census. Phase fractions partition that face, even at a free surface.
            // Keeping the total face measure constant also makes total pressure
            // impulses telescope, instead of inventing a p*grad(particle density) force.
            let total_volume = f.volume[0] + f.volume[1];
            if total_volume > 0.0 {
                for v in &mut f.volume {
                    *v *= h.powi(3) / total_volume;
                }
            }
            let mut force = [f.mass[0] * gravity[f.axis], f.mass[1] * gravity[f.axis]];
            if self.cfg.mac.capillary_pressure > 0.0 && f.mass[0] > 0.0 && f.mass[1] > 0.0 {
                let suction = |i: Option<usize>| {
                    i.map_or(0.0, |j| {
                        let c = self.grid.cells[j];
                        if c.solid <= 0.0 {
                            return 0.0;
                        }
                        let sat = (c.water / (h.powi(3) - c.solid).max(1e-12)).clamp(0.0, 1.0);
                        -self.cfg.mac.capillary_pressure * (1.0 - sat)
                    })
                };
                let capillary = -f.volume[0] / h * (suction(f.right) - suction(f.left));
                force[0] += capillary;
                force[1] -= capillary;
            }
            f.velocity = f
                .mobility
                .apply(f.momentum[0] + dt * force[0], f.momentum[1] + dt * force[1]);
            f.force = force;
            let response = f.mobility.apply(f.volume[0], f.volume[1]);
            f.coefficient = dt / (h * h) * (f.volume[0] * response[0] + f.volume[1] * response[1]);
        }
        let (iterations, residual) = self.grid.project(
            dt,
            self.cfg.mac.pressure_iterations,
            self.cfg.mac.pressure_tolerance,
        );
        self.pressure_iterations += iterations;
        self.pressure_residual = self.pressure_residual.max(residual);
        self.gravity_impulse += dt * total_mass * gravity;
        for f in &self.grid.faces {
            let dp = f.right.map_or(0.0, |i| self.grid.cells[i].pressure)
                - f.left.map_or(0.0, |i| self.grid.cells[i].pressure);
            for phase in 0..2 {
                if f.blocked[phase] || (phase == 1 && self.cfg.mac.frozen_bed) {
                    self.boundary_impulse[f.axis] += f.momentum[phase] + dt * f.force[phase]
                        - dt / h * f.volume[phase] * dp
                        + f.drag_dt * (f.velocity[1 - phase] - f.velocity[phase])
                        - f.mass[phase] * f.velocity[phase];
                }
            }
        }
        // Water gathers the grid impulse (FLIP), preserving unresolved ballistic
        // velocity differences. Grain PIC intentionally dissipates bed motion.
        // Both use the adjoint mass-weighted transfer and conserve momentum.
        for p in &mut self.water {
            for a in 0..3 {
                p.velocity[a] += self
                    .grid
                    .stencil(p.position, Some(a))
                    .iter()
                    .map(|&(i, w)| {
                        let f = &self.grid.faces[i];
                        w * (f.velocity[0] - f.momentum[0] / f.mass[0].max(1e-30))
                    })
                    .sum::<f64>();
            }
        }
        for g in &mut self.grains {
            for a in 0..3 {
                g.velocity[a] = self
                    .grid
                    .stencil(g.position, Some(a))
                    .iter()
                    .map(|&(i, w)| w * self.grid.faces[i].velocity[1])
                    .sum();
            }
        }
    }

    fn move_particles(&mut self, dt: f64) {
        let rho = self.rho();
        for p in &mut self.water {
            p.position += dt * p.velocity;
            let old = p.velocity;
            collide(
                &self.scene,
                &mut p.position,
                &mut p.velocity,
                0,
                0.0,
                self.cfg.mac.open_bottom,
                0.0,
            );
            self.boundary_impulse += (old - p.velocity) * (rho * p.volume + p.solute);
        }
        if !self.cfg.mac.frozen_bed {
            for g in &mut self.grains {
                g.position += dt * g.velocity;
            }
            self.contacts();
        }
        let mut keep = Vec::with_capacity(self.water.len());
        for p in self.water.drain(..) {
            if self.cfg.mac.open_bottom && p.position.y < self.scene.box_min[1] as f64 {
                self.drained_water += p.volume;
                self.drained_solute += p.solute;
                self.drained_heat += p.heat;
                self.drained_momentum += (rho * p.volume + p.solute) * p.velocity;
            } else if p.volume > 0.0 || p.solute > 0.0 {
                keep.push(p);
            }
        }
        self.water = keep;
    }

    /// Coarse grain parcels use frictional sphere contacts. A spatial bucket lookup keeps
    /// the reference from doing all-pairs work. Every pair impulse is antisymmetric.
    fn contacts(&mut self) {
        use std::collections::BTreeMap;
        let rho = self.rho();
        let radii: Vec<f64> = self
            .grains
            .iter()
            .map(|g| (3.0 * self.grain_volume(g) / (4.0 * std::f64::consts::PI)).cbrt())
            .collect();
        let pitch = radii.iter().copied().fold(0.0, f64::max) * 2.0;
        if pitch <= 0.0 {
            return;
        }
        for _ in 0..self.cfg.mac.contact_iterations {
            let mut bins: BTreeMap<[i32; 3], Vec<usize>> = BTreeMap::new();
            let key = |p: DVec3| (p / pitch).floor().to_array().map(|x| x as i32);
            for (i, g) in self.grains.iter().enumerate() {
                bins.entry(key(g.position)).or_default().push(i);
            }
            for i in 0..self.grains.len() {
                let k = key(self.grains[i].position);
                for z in -1..=1 {
                    for y in -1..=1 {
                        for x in -1..=1 {
                            let Some(neighbors) = bins.get(&[k[0] + x, k[1] + y, k[2] + z]) else {
                                continue;
                            };
                            for &j in neighbors {
                                if j <= i {
                                    continue;
                                }
                                let delta = self.grains[j].position - self.grains[i].position;
                                let distance = delta.length();
                                let overlap = radii[i] + radii[j] - distance;
                                if overlap <= 0.0 {
                                    continue;
                                }
                                let n = if distance > 1e-12 {
                                    delta / distance
                                } else {
                                    DVec3::X
                                };
                                let (mi, mj) = (self.grains[i].mass(rho), self.grains[j].mass(rho));
                                let inv_sum = 1.0 / mi + 1.0 / mj;
                                let displacement = overlap / inv_sum;
                                self.grains[i].position -= n * (displacement / mi);
                                self.grains[j].position += n * (displacement / mj);
                                let rel = self.grains[j].velocity - self.grains[i].velocity;
                                let normal = (-rel.dot(n)).max(0.0) / inv_sum;
                                let tangent = rel - rel.dot(n) * n;
                                let friction = (tangent.length() / inv_sum)
                                    .min(self.mats.friction_mu as f64 * normal);
                                let impulse = normal * n - tangent.normalize_or_zero() * friction;
                                self.grains[i].velocity -= impulse / mi;
                                self.grains[j].velocity += impulse / mj;
                            }
                        }
                    }
                }
            }
            for (i, g) in self.grains.iter_mut().enumerate() {
                let old = g.velocity;
                collide(
                    &self.scene,
                    &mut g.position,
                    &mut g.velocity,
                    1,
                    radii[i],
                    false,
                    self.mats.floor_mu as f64,
                );
                self.boundary_impulse += (old - g.velocity) * g.mass(rho);
            }
        }
    }
}

fn collide(
    scene: &Scene,
    p: &mut DVec3,
    v: &mut DVec3,
    phase: u32,
    radius: f64,
    open_bottom: bool,
    floor_friction: f64,
) {
    for a in 0..3 {
        let before = v[a];
        let lo = scene.box_min[a] as f64 + radius;
        let hi = scene.box_max[a] as f64 - radius - 1e-9;
        if p[a] < lo && !(a == 1 && open_bottom) {
            p[a] = lo;
            v[a] = v[a].max(0.0);
        }
        if p[a] > hi {
            p[a] = hi;
            v[a] = v[a].min(0.0);
        }
        if phase == 1 {
            let mut tangent = *v;
            tangent[a] = 0.0;
            let loss = tangent.length().min(floor_friction * (v[a] - before).abs());
            *v -= tangent.normalize_or_zero() * loss;
        }
    }
    for _ in 0..4 {
        let c = sdf::nearest(&scene.solids, p.as_vec3(), phase);
        if c.signed as f64 >= radius {
            break;
        }
        let n = c.grad.as_dvec3();
        *p += (radius - c.signed as f64 + 1e-7) * n;
        let normal = v.dot(n).min(0.0);
        *v -= normal * n;
        if phase == 1 {
            let tangent = *v - v.dot(n) * n;
            *v -= tangent.normalize_or_zero() * tangent.length().min(-normal * c.friction as f64);
        }
    }
}

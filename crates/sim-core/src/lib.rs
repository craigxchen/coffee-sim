pub mod constants;
pub mod pour;
pub mod sph;

use sph::*;

pub use pour::{PourCommand, PourPattern, PourScript};

// ── Wall segment (line boundary) ──────────────────────────

#[derive(Clone, Debug)]
pub struct Wall {
    pub a: Vec2, // start point
    pub b: Vec2, // end point
    // Normal points inward (toward the fluid side)
    pub normal: Vec2,
}

impl Wall {
    pub fn new(a: Vec2, b: Vec2) -> Self {
        let edge = b - a;
        // Left-hand normal (for walls where fluid is on the left side)
        let n = Vec2::new(-edge.y, edge.x);
        let len = n.length();
        let normal = if len > EPSILON { n / len } else { Vec2::ZERO };
        Self { a, b, normal }
    }

    pub fn new_inward(a: Vec2, b: Vec2, inward_point: Vec2) -> Self {
        let mut w = Self::new(a, b);
        // Make sure normal points toward the inward_point
        let mid = Vec2::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
        let to_inside = inward_point - mid;
        if w.normal.dot(to_inside) < 0.0 {
            w.normal = -w.normal;
        }
        w
    }

    /// Signed distance from point to wall line. Positive = on normal side (inside).
    pub fn signed_distance(&self, p: Vec2) -> f32 {
        (p - self.a).dot(self.normal)
    }

    /// Project point onto the wall segment, clamped to [a, b].
    pub fn closest_point(&self, p: Vec2) -> Vec2 {
        let ab = self.b - self.a;
        let len_sq = ab.length_squared();
        if len_sq < EPSILON {
            return self.a;
        }
        let t = ((p - self.a).dot(ab) / len_sq).clamp(0.0, 1.0);
        self.a + ab * t
    }
}

// ── Simulation settings ───────────────────────────────────

#[derive(Clone, Debug)]
pub struct SimSettings {
    pub gravity: f32,
    pub smoothing_radius: f32,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub near_pressure_multiplier: f32,
    pub viscosity_strength: f32,
    pub collision_damping: f32,
    pub iterations_per_frame: usize,
    pub max_particles: usize,
    pub bounds_size: Vec2, // outer bounding box (particles can't escape)
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            gravity: -12.0, // negative Y = down (matching Fluid-Sim convention: Y up)
            smoothing_radius: 0.35,
            target_density: 55.0,
            pressure_multiplier: 500.0,
            near_pressure_multiplier: 5.0,
            viscosity_strength: 0.06,
            collision_damping: 0.95,
            iterations_per_frame: 3,
            max_particles: 5000,
            bounds_size: Vec2::new(14.0, 14.0),
        }
    }
}

pub struct StepResult {
    pub water_count: usize,
    pub total_water_in: f32,
    pub total_water_out: f32,
    pub brew_time: f32,
}

// ── Particle simulation ───────────────────────────────────

pub struct ParticleSim {
    pub positions: Vec<Vec2>,
    pub predicted_positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub densities: Vec<Vec2>,
    velocity_scratch: Vec<Vec2>,

    spatial_keys: Vec<u32>,
    spatial_offsets: Vec<usize>,

    pub walls: Vec<Wall>,
    /// Drain segments: water particles passing through these are removed
    pub drains: Vec<Wall>,

    pub brew_time: f32,
    pub total_water_in: f32,
    pub total_water_out: f32,

    pub settings: SimSettings,
}

impl ParticleSim {
    pub fn new(settings: SimSettings) -> Self {
        Self {
            positions: Vec::new(),
            predicted_positions: Vec::new(),
            velocities: Vec::new(),
            densities: Vec::new(),
            velocity_scratch: Vec::new(),
            spatial_keys: Vec::new(),
            spatial_offsets: Vec::new(),
            walls: Vec::new(),
            drains: Vec::new(),
            brew_time: 0.0,
            total_water_in: 0.0,
            total_water_out: 0.0,
            settings,
        }
    }

    /// Set up V60 cone walls. Y-up coordinate system.
    /// V60 cross-section: wide at top, narrow at bottom.
    pub fn setup_v60(&mut self, top_half_width: f32, bot_half_width: f32, height: f32) {
        let center = Vec2::ZERO; // center of V60 at origin

        // Top of cone is at y = height/2, bottom at y = -height/2
        let top_y = height * 0.5;
        let bot_y = -height * 0.5;

        let tl = Vec2::new(-top_half_width, top_y);
        let tr = Vec2::new(top_half_width, top_y);
        let bl = Vec2::new(-bot_half_width, bot_y);
        let br = Vec2::new(bot_half_width, bot_y);

        // Left wall: from top-left down to bottom-left. Normal points right (inward).
        self.walls.push(Wall::new_inward(tl, bl, center));
        // Right wall: from top-right down to bottom-right. Normal points left (inward).
        self.walls.push(Wall::new_inward(br, tr, center));
        // Bottom wall: from bottom-left to bottom-right. Normal points up (inward).
        // This is the filter — we'll make it a drain instead.
        self.drains.push(Wall::new_inward(bl, br, center));
    }

    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Spawn water particles at a given position with a configurable initial velocity.
    pub fn emit_water_with_velocity(&mut self, count: usize, x: f32, y: f32, velocity: Vec2) {
        let mut rng = SimpleRng::new((self.brew_time * 10000.0) as u64 + 7);
        let spread = self.settings.smoothing_radius * 0.12;

        for _ in 0..count {
            if self.positions.len() >= self.settings.max_particles {
                break;
            }
            let px = x + (rng.next_f32() - 0.5) * spread;
            let py = y + (rng.next_f32() - 0.5) * spread * 0.3;
            let pos = Vec2::new(px, py);
            self.positions.push(pos);
            self.predicted_positions.push(pos);
            self.velocities.push(velocity);
            self.densities.push(Vec2::ZERO);
            self.spatial_keys.push(0);
            self.total_water_in += 1.0;
        }
    }

    /// Spawn water particles with the default downward injection velocity.
    pub fn emit_water(&mut self, count: usize, x: f32, y: f32) {
        self.emit_water_with_velocity(count, x, y, Vec2::new(0.0, -3.0));
    }

    pub fn step_frame(
        &mut self,
        frame_time: f32,
        pour_x: f32,
        pour_y: f32,
        emit_count: usize,
    ) -> StepResult {
        self.step_frame_with_velocity(frame_time, pour_x, pour_y, Vec2::new(0.0, -3.0), emit_count)
    }

    pub fn step_frame_with_velocity(
        &mut self,
        frame_time: f32,
        pour_x: f32,
        pour_y: f32,
        emit_velocity: Vec2,
        emit_count: usize,
    ) -> StepResult {
        let max_dt = 1.0 / 60.0;
        let dt = frame_time.min(max_dt);
        let iters = self.settings.iterations_per_frame.max(1);
        let sub_dt = dt / iters as f32;

        for i in 0..iters {
            // Emit water spread across substeps
            if i == 0 {
                self.emit_water_with_velocity(emit_count, pour_x, pour_y, emit_velocity);
            }
            self.run_step(sub_dt);
        }

        self.drain_particles();
        self.brew_time += dt;

        StepResult {
            water_count: self.positions.len(),
            total_water_in: self.total_water_in,
            total_water_out: self.total_water_out,
            brew_time: self.brew_time,
        }
    }

    fn drain_particles(&mut self) {
        let mut i = 0;
        while i < self.positions.len() {
            let pos = self.positions[i];
            let mut drained = false;

            // Check if below any drain wall
            for drain in &self.drains {
                if drain.signed_distance(pos) < -0.05 {
                    drained = true;
                    break;
                }
            }

            // Also drain if outside outer bounds
            let hb = self.settings.bounds_size * 0.5;
            if pos.x.abs() > hb.x || pos.y.abs() > hb.y {
                drained = true;
            }

            if drained {
                self.total_water_out += 1.0;
                self.remove_particle(i);
            } else {
                i += 1;
            }
        }
    }

    fn remove_particle(&mut self, i: usize) {
        self.positions.swap_remove(i);
        self.predicted_positions.swap_remove(i);
        self.velocities.swap_remove(i);
        self.densities.swap_remove(i);
        self.spatial_keys.swap_remove(i);
    }

    fn run_step(&mut self, dt: f32) {
        if self.positions.is_empty() {
            return;
        }
        self.apply_gravity(dt);
        self.update_spatial_lookup();
        self.calculate_densities();
        self.apply_pressure_forces(dt);
        self.apply_viscosity(dt);
        self.update_positions(dt);
    }

    fn apply_gravity(&mut self, dt: f32) {
        let gravity = Vec2::new(0.0, self.settings.gravity);
        for i in 0..self.positions.len() {
            self.velocities[i] += gravity * dt;
            self.predicted_positions[i] =
                self.positions[i] + self.velocities[i] * PREDICTION_FACTOR;
        }
    }

    fn update_spatial_lookup(&mut self) {
        let n = self.positions.len();
        let table_size = n.max(1);
        let mut entries: Vec<(u32, usize)> = Vec::with_capacity(n);

        for i in 0..n {
            let cell = get_cell(self.predicted_positions[i], self.settings.smoothing_radius);
            let hash = hash_cell(cell);
            entries.push((hash % table_size as u32, i));
        }

        entries.sort_unstable_by_key(|(k, _)| *k);

        self.spatial_offsets.clear();
        self.spatial_offsets.resize(table_size, usize::MAX);

        let mut sp = Vec::with_capacity(n);
        let mut spp = Vec::with_capacity(n);
        let mut sv = Vec::with_capacity(n);
        let mut sk = Vec::with_capacity(n);

        for (si, (key, oi)) in entries.into_iter().enumerate() {
            if self.spatial_offsets[key as usize] == usize::MAX {
                self.spatial_offsets[key as usize] = si;
            }
            sp.push(self.positions[oi]);
            spp.push(self.predicted_positions[oi]);
            sv.push(self.velocities[oi]);
            sk.push(key);
        }

        self.positions = sp;
        self.predicted_positions = spp;
        self.velocities = sv;
        self.spatial_keys = sk;
    }

    fn calculate_densities(&mut self) {
        let n = self.positions.len();
        self.densities.resize(n, Vec2::ZERO);
        for i in 0..n {
            self.densities[i] = self.density_at(self.predicted_positions[i]);
        }
    }

    fn density_at(&self, position: Vec2) -> Vec2 {
        let origin = get_cell(position, self.settings.smoothing_radius);
        let r_sq = self.settings.smoothing_radius * self.settings.smoothing_radius;
        let n = self.positions.len().max(1);
        let (mut d, mut nd) = (0.0_f32, 0.0_f32);

        for offset in NEIGHBOUR_OFFSETS {
            let key = hash_cell((origin.0 + offset.0, origin.1 + offset.1)) % n as u32;
            let mut cur = self
                .spatial_offsets
                .get(key as usize)
                .copied()
                .unwrap_or(usize::MAX);
            if cur == usize::MAX {
                continue;
            }
            while cur < self.positions.len() {
                if self.spatial_keys[cur] != key {
                    break;
                }
                let delta = self.predicted_positions[cur] - position;
                let d2 = delta.length_squared();
                if d2 <= r_sq {
                    let dst = d2.sqrt();
                    d += spiky_kernel_pow2(dst, self.settings.smoothing_radius);
                    nd += spiky_kernel_pow3(dst, self.settings.smoothing_radius);
                }
                cur += 1;
            }
        }
        Vec2::new(d, nd)
    }

    fn apply_pressure_forces(&mut self, dt: f32) {
        let n = self.positions.len();
        self.velocity_scratch.clone_from(&self.velocities);
        let r_sq = self.settings.smoothing_radius * self.settings.smoothing_radius;
        let ts = n.max(1);

        for i in 0..n {
            let den = self.densities[i].x.max(EPSILON);
            let nden = self.densities[i].y.max(EPSILON);
            let pres = (den - self.settings.target_density) * self.settings.pressure_multiplier;
            let npres = self.settings.near_pressure_multiplier * nden;
            let pos = self.predicted_positions[i];
            let origin = get_cell(pos, self.settings.smoothing_radius);
            let mut force = Vec2::ZERO;

            for offset in NEIGHBOUR_OFFSETS {
                let key = hash_cell((origin.0 + offset.0, origin.1 + offset.1)) % ts as u32;
                let mut cur = self
                    .spatial_offsets
                    .get(key as usize)
                    .copied()
                    .unwrap_or(usize::MAX);
                if cur == usize::MAX {
                    continue;
                }
                while cur < n {
                    if self.spatial_keys[cur] != key {
                        break;
                    }
                    if cur == i {
                        cur += 1;
                        continue;
                    }
                    let to = self.predicted_positions[cur] - pos;
                    let d2 = to.length_squared();
                    if d2 > r_sq {
                        cur += 1;
                        continue;
                    }
                    let dst = d2.sqrt();
                    let dir = if dst > EPSILON {
                        to / dst
                    } else {
                        Vec2::new(0.0, 1.0)
                    };
                    let nd2 = self.densities[cur].x.max(EPSILON);
                    let nnd = self.densities[cur].y.max(EPSILON);
                    let np =
                        (nd2 - self.settings.target_density) * self.settings.pressure_multiplier;
                    let nnp = self.settings.near_pressure_multiplier * nnd;
                    force += dir
                        * derivative_spiky_pow2(dst, self.settings.smoothing_radius)
                        * (pres + np)
                        * 0.5
                        / nd2;
                    force += dir
                        * derivative_spiky_pow3(dst, self.settings.smoothing_radius)
                        * (npres + nnp)
                        * 0.5
                        / nnd;
                    cur += 1;
                }
            }
            self.velocity_scratch[i] += force / den * dt;
        }
        std::mem::swap(&mut self.velocities, &mut self.velocity_scratch);
    }

    fn apply_viscosity(&mut self, dt: f32) {
        let n = self.positions.len();
        self.velocity_scratch.clone_from(&self.velocities);
        let r_sq = self.settings.smoothing_radius * self.settings.smoothing_radius;
        let ts = n.max(1);

        for i in 0..n {
            let pos = self.predicted_positions[i];
            let vel = self.velocities[i];
            let origin = get_cell(pos, self.settings.smoothing_radius);
            let mut vf = Vec2::ZERO;

            for offset in NEIGHBOUR_OFFSETS {
                let key = hash_cell((origin.0 + offset.0, origin.1 + offset.1)) % ts as u32;
                let mut cur = self
                    .spatial_offsets
                    .get(key as usize)
                    .copied()
                    .unwrap_or(usize::MAX);
                if cur == usize::MAX {
                    continue;
                }
                while cur < n {
                    if self.spatial_keys[cur] != key {
                        break;
                    }
                    if cur == i {
                        cur += 1;
                        continue;
                    }
                    let to = self.predicted_positions[cur] - pos;
                    let d2 = to.length_squared();
                    if d2 <= r_sq {
                        vf += (self.velocities[cur] - vel)
                            * smoothing_kernel_poly6(d2.sqrt(), self.settings.smoothing_radius);
                    }
                    cur += 1;
                }
            }
            self.velocity_scratch[i] += vf * self.settings.viscosity_strength * dt;
        }
        std::mem::swap(&mut self.velocities, &mut self.velocity_scratch);
    }

    fn update_positions(&mut self, dt: f32) {
        for i in 0..self.positions.len() {
            self.positions[i] += self.velocities[i] * dt;
            self.handle_collisions(i);
        }
    }

    fn handle_collisions(&mut self, i: usize) {
        let damp = self.settings.collision_damping;
        let mut pos = self.positions[i];
        let mut vel = self.velocities[i];

        // Outer bounds (same as Fluid-Sim)
        let hb = self.settings.bounds_size * 0.5;
        let edge = hb - pos.abs();
        if edge.x <= 0.0 {
            pos.x = hb.x * signed_unit(pos.x);
            vel.x *= -damp;
        }
        if edge.y <= 0.0 {
            pos.y = hb.y * signed_unit(pos.y);
            vel.y *= -damp;
        }

        // Wall segment collisions
        for wall in &self.walls {
            let dist = wall.signed_distance(pos);
            if dist < 0.0 {
                // Penetrating — push back along normal
                pos += wall.normal * (-dist + 0.001);
                // Reflect velocity component along normal
                let vn = vel.dot(wall.normal);
                if vn < 0.0 {
                    vel = vel - wall.normal * (1.0 + damp) * vn;
                }
            }
        }

        self.positions[i] = pos;
        self.velocities[i] = vel;
    }
}

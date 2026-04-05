use coffee_sim_core::sph::Vec3;

use super::state::MpmBuffers;

pub(crate) const PARTICLES_PER_ML: f32 = 80.0;

#[derive(Clone, Copy)]
pub(crate) struct SpoutSettings {
    pub origin: Vec3,
    pub direction: Vec3,
    pub nozzle_radius: f32,
    pub stem_radius: f32,
    pub activation_angle_deg: f32,
    pub full_flow_angle_deg: f32,
    pub max_flow_rate_ml_s: f32,
    pub max_exit_speed: f32,
    pub stem_length: f32,
}

impl Default for SpoutSettings {
    fn default() -> Self {
        Self {
            origin: Vec3::new(-6.0, 6.6, 2.2),
            direction: Vec3::new(0.71, -0.66, -0.24).normalized(),
            nozzle_radius: 0.18,
            stem_radius: 0.24,
            activation_angle_deg: 8.0,
            full_flow_angle_deg: 56.0,
            max_flow_rate_ml_s: 11.5,
            max_exit_speed: 22.0,
            stem_length: 1.9,
        }
    }
}

pub(crate) struct InflowState {
    kettle_angle_deg: f32,
    flow_rate: f32,
    exit_speed: f32,
    accumulator: f32,
}

impl InflowState {
    pub fn new(angle_deg: f32) -> Self {
        Self {
            kettle_angle_deg: angle_deg,
            flow_rate: 0.0,
            exit_speed: 0.0,
            accumulator: 0.0,
        }
    }

    pub fn set_angle(&mut self, angle_deg: f32) {
        self.kettle_angle_deg = angle_deg;
    }

    pub fn angle(&self) -> f32 {
        self.kettle_angle_deg
    }

    pub fn flow_rate(&self) -> f32 {
        self.flow_rate
    }

    pub fn exit_speed(&self) -> f32 {
        self.exit_speed
    }

    pub fn update(&mut self, spout: &SpoutSettings) {
        self.flow_rate = flow_rate_from_angle(self.kettle_angle_deg, spout);
        self.exit_speed = exit_speed_from_flow_rate(
            self.flow_rate,
            spout.nozzle_radius,
            spout.max_exit_speed,
        );
    }

    pub fn emit_particles(
        &mut self,
        queue: &wgpu::Queue,
        buffers: &MpmBuffers,
        spout: &SpoutSettings,
        dt: f32,
        particle_mass: f32,
        current_water: u32,
        current_bed: u32,
        max_particles: u32,
    ) -> u32 {
        self.update(spout);

        if self.flow_rate < 1e-6 {
            self.accumulator = 0.0;
            return 0;
        }

        let particles_per_sec = self.flow_rate * PARTICLES_PER_ML;
        self.accumulator += particles_per_sec * dt;
        let count = self.accumulator as u32;
        if count == 0 {
            return 0;
        }
        self.accumulator -= count as f32;

        let total = current_water + current_bed;
        let available = max_particles.saturating_sub(total);
        let count = count.min(available);
        if count == 0 {
            return 0;
        }

        // Build tangent basis for the nozzle disk
        let dir = spout.direction;
        let up = if dir.y.abs() < 0.9 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let t1 = dir.cross(up).normalized();
        let t2 = dir.cross(t1).normalized();

        let mut particle_data: Vec<[f32; 8]> = Vec::with_capacity(count as usize);
        let mut affine_data: Vec<[f32; 12]> = Vec::with_capacity(count as usize);

        let vel = Vec3::new(
            dir.x * self.exit_speed,
            dir.y * self.exit_speed,
            dir.z * self.exit_speed,
        );

        // Simple disk distribution using golden angle
        let golden_angle = 2.399_963_f32;
        for i in 0..count {
            let fi = i as f32;
            let r = spout.nozzle_radius * (fi / count as f32).sqrt();
            let theta = fi * golden_angle;
            let offset = Vec3::new(
                t1.x * r * theta.cos() + t2.x * r * theta.sin(),
                t1.y * r * theta.cos() + t2.y * r * theta.sin(),
                t1.z * r * theta.cos() + t2.z * r * theta.sin(),
            );
            let pos = Vec3::new(
                spout.origin.x + offset.x,
                spout.origin.y + offset.y,
                spout.origin.z + offset.z,
            );

            // Particle: pos(x,y,z,J), vel(vx,vy,vz,mass)
            particle_data.push([pos.x, pos.y, pos.z, 1.0, vel.x, vel.y, vel.z, particle_mass]);
            // AffineC: col0(0,0,0,phase), col1(0,0,0,0), col2(0,0,0,0)
            affine_data.push([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        // Append new water particles after all existing particles (water + bed).
        // The shader uses phase (0.0=water, 1.0=bed) to distinguish particle types.
        let particle_offset = (total as u64) * 32;
        let affine_offset = (total as u64) * 48;

        queue.write_buffer(
            &buffers.particles,
            particle_offset,
            bytemuck::cast_slice(&particle_data),
        );
        queue.write_buffer(
            &buffers.affine,
            affine_offset,
            bytemuck::cast_slice(&affine_data),
        );

        count
    }
}

fn flow_rate_from_angle(angle_deg: f32, spout: &SpoutSettings) -> f32 {
    if angle_deg <= spout.activation_angle_deg {
        return 0.0;
    }
    if angle_deg >= spout.full_flow_angle_deg {
        return spout.max_flow_rate_ml_s;
    }
    let t = (angle_deg - spout.activation_angle_deg)
        / (spout.full_flow_angle_deg - spout.activation_angle_deg);
    // Smooth ramp (smoothstep)
    let s = t * t * (3.0 - 2.0 * t);
    s * spout.max_flow_rate_ml_s
}

fn exit_speed_from_flow_rate(flow_rate: f32, nozzle_radius: f32, max_exit_speed: f32) -> f32 {
    if flow_rate < 1e-6 {
        return 0.0;
    }
    let area = std::f32::consts::PI * nozzle_radius * nozzle_radius;
    // flow_rate is mL/s = cm^3/s, area in sim units^2
    // Approximate: speed = flow_rate_factor / area
    // Keep it simple: scale so max flow gives reasonable speed
    (flow_rate / area).min(max_exit_speed)
}

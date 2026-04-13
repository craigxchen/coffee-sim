use coffee_sim_core::sph::Vec3;

use super::state::MpmBuffers;

pub(crate) const MASS_UNITS_PER_ML: f32 = 80.0;
pub(crate) const PARTICLES_PER_ML: f32 = 320.0;

#[derive(Clone, Copy)]
pub(crate) struct SpoutSettings {
    pub origin: Vec3,
    pub direction: Vec3,
    pub nozzle_radius: f32,
    pub stem_radius: f32,
    pub activation_angle_deg: f32,
    pub full_flow_angle_deg: f32,
    pub head_at_activation: f32,
    pub head_at_full_angle: f32,
    pub discharge_coeff: f32,
    pub volume_to_ml: f32,
    pub max_flow_rate_ml_s: f32,
    pub max_exit_speed: f32,
    pub stem_length: f32,
}

impl Default for SpoutSettings {
    fn default() -> Self {
        Self {
            origin: Vec3::new(-3.4, 7.3, 0.9),
            direction: Vec3::new(0.36, -0.92, -0.12).normalized(),
            nozzle_radius: 0.18,
            stem_radius: 0.24,
            activation_angle_deg: 8.0,
            full_flow_angle_deg: 56.0,
            head_at_activation: 0.04,
            head_at_full_angle: 24.0,
            discharge_coeff: 0.92,
            volume_to_ml: 5.4,
            max_flow_rate_ml_s: 11.5,
            max_exit_speed: 22.0,
            stem_length: 1.9,
        }
    }
}

impl SpoutSettings {
    pub fn aim_at(&mut self, target: Vec3) {
        let delta = Vec3::new(
            target.x - self.origin.x,
            target.y - self.origin.y,
            target.z - self.origin.z,
        );
        if delta.length() > 1e-5 {
            self.direction = delta.normalized();
        }
    }
}

pub(crate) struct InflowState {
    kettle_angle_deg: f32,
    flow_rate: f32,
    exit_speed: f32,
    accumulator: f32,
    emission_sequence: u32,
}

pub(crate) struct EmissionResult {
    pub emitted: u32,
    pub dropped: u32,
}

impl InflowState {
    pub fn new(angle_deg: f32) -> Self {
        Self {
            kettle_angle_deg: angle_deg,
            flow_rate: 0.0,
            exit_speed: 0.0,
            accumulator: 0.0,
            emission_sequence: 0,
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
        let head = effective_head_from_angle(self.kettle_angle_deg, spout);
        self.exit_speed = exit_speed_from_head(head, spout.max_exit_speed);
        self.flow_rate = flow_rate_from_speed(
            self.exit_speed,
            spout.nozzle_radius,
            spout.discharge_coeff,
            spout.volume_to_ml,
            spout.max_flow_rate_ml_s,
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
    ) -> EmissionResult {
        self.update(spout);

        if self.flow_rate < 1e-6 {
            self.accumulator = 0.0;
            return EmissionResult {
                emitted: 0,
                dropped: 0,
            };
        }

        let particles_per_sec = self.flow_rate * PARTICLES_PER_ML;
        self.accumulator += particles_per_sec * dt;
        let requested = self.accumulator as u32;
        let count = requested;
        if count == 0 {
            return EmissionResult {
                emitted: 0,
                dropped: 0,
            };
        }
        self.accumulator -= count as f32;

        let total = current_water + current_bed;
        let available = max_particles.saturating_sub(total);
        let count = count.min(available);
        if count == 0 {
            return EmissionResult {
                emitted: 0,
                dropped: requested,
            };
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

        let emit_speed = self.exit_speed();
        let vel = Vec3::new(
            dir.x * emit_speed,
            dir.y * emit_speed,
            dir.z * emit_speed,
        );

        // Emit from a slightly contracted jet core rather than the full nozzle
        // disk. This approximates the vena-contracta region just below the
        // lip and avoids an unrealistically flared free stream when we do not
        // model sub-nozzle cohesion.
        let emit_origin = Vec3::new(
            spout.origin.x + dir.x * 0.18,
            spout.origin.y + dir.y * 0.18,
            spout.origin.z + dir.z * 0.18,
        );
        let jet_radius = spout.nozzle_radius * 0.58;

        // Low-discrepancy disk distribution with a persistent sample sequence
        // so the nozzle does not replay the same few beams every frame.
        let golden_angle = 2.399_963_f32;
        let radial_irrational = 0.754_877_7_f32;
        for i in 0..count {
            let sample = self.emission_sequence.wrapping_add(i);
            let sf = sample as f32;
            let radial_u = ((sf + 0.5) * radial_irrational).fract();
            let r = jet_radius * radial_u.sqrt();
            let theta = sf * golden_angle;
            let offset = Vec3::new(
                t1.x * r * theta.cos() + t2.x * r * theta.sin(),
                t1.y * r * theta.cos() + t2.y * r * theta.sin(),
                t1.z * r * theta.cos() + t2.z * r * theta.sin(),
            );
            let pos = Vec3::new(
                emit_origin.x + offset.x,
                emit_origin.y + offset.y,
                emit_origin.z + offset.z,
            );

            // Particle: pos(x,y,z,J), vel(vx,vy,vz,mass)
            particle_data.push([pos.x, pos.y, pos.z, 1.0, vel.x, vel.y, vel.z, particle_mass]);
            // AffineC: col0(0,0,0,phase), col1(0,0,0,0), col2(0,0,0,0)
            affine_data.push([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        self.emission_sequence = self.emission_sequence.wrapping_add(count);

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

        EmissionResult {
            emitted: count,
            dropped: requested.saturating_sub(count),
        }
    }
}

fn effective_head_from_angle(angle_deg: f32, spout: &SpoutSettings) -> f32 {
    if angle_deg <= spout.activation_angle_deg {
        return 0.0;
    }
    if angle_deg >= spout.full_flow_angle_deg {
        return spout.head_at_full_angle;
    }
    let t = (angle_deg - spout.activation_angle_deg)
        / (spout.full_flow_angle_deg - spout.activation_angle_deg);
    let s = t * t * (3.0 - 2.0 * t);
    spout.head_at_activation + s * (spout.head_at_full_angle - spout.head_at_activation)
}

fn exit_speed_from_head(head: f32, max_exit_speed: f32) -> f32 {
    if head < 1e-6 {
        return 0.0;
    }
    let speed = (2.0 * 10.0 * head).sqrt();
    speed.min(max_exit_speed)
}

fn flow_rate_from_speed(
    exit_speed: f32,
    nozzle_radius: f32,
    discharge_coeff: f32,
    volume_to_ml: f32,
    max_flow_rate_ml_s: f32,
) -> f32 {
    if exit_speed < 1e-6 {
        return 0.0;
    }
    let area = std::f32::consts::PI * nozzle_radius * nozzle_radius;
    let volumetric_flow = discharge_coeff * area * exit_speed;
    (volumetric_flow * volume_to_ml).min(max_flow_rate_ml_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_head_is_zero_below_activation_and_monotonic() {
        let spout = SpoutSettings::default();

        assert_eq!(effective_head_from_angle(0.0, &spout), 0.0);
        assert_eq!(
            effective_head_from_angle(spout.activation_angle_deg, &spout),
            0.0
        );

        let low = effective_head_from_angle(16.0, &spout);
        let mid = effective_head_from_angle(30.0, &spout);
        let high = effective_head_from_angle(48.0, &spout);

        assert!(low > 0.0);
        assert!(mid > low);
        assert!(high > mid);
        assert!(
            (effective_head_from_angle(spout.full_flow_angle_deg, &spout)
                - spout.head_at_full_angle)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn exit_speed_from_head_is_monotonic_and_capped() {
        let low = exit_speed_from_head(0.25, 22.0);
        let mid = exit_speed_from_head(2.0, 22.0);
        let high = exit_speed_from_head(40.0, 22.0);

        assert!(low > 0.0);
        assert!(mid > low);
        assert!(high >= mid);
        assert!(high <= 22.0);
        assert_eq!(exit_speed_from_head(0.0, 22.0), 0.0);
    }

    #[test]
    fn flow_rate_from_speed_is_monotonic_and_capped() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;
        let low = flow_rate_from_speed(4.0, nozzle_radius, 0.92, 5.4, 11.5);
        let mid = flow_rate_from_speed(10.0, nozzle_radius, 0.92, 5.4, 11.5);
        let high = flow_rate_from_speed(40.0, nozzle_radius, 0.92, 5.4, 11.5);

        assert!(low > 0.0);
        assert!(mid > low);
        assert!(high >= mid);
        assert!(high <= 11.5);
        assert_eq!(
            flow_rate_from_speed(0.0, nozzle_radius, 0.92, 5.4, 11.5),
            0.0
        );
    }

    #[test]
    fn inflow_update_produces_consistent_head_speed_and_flow() {
        let spout = SpoutSettings::default();
        let mut inflow = InflowState::new(42.0);
        inflow.update(&spout);

        assert!(inflow.exit_speed() > 0.0);
        assert!(inflow.flow_rate() > 0.0);
        assert!(inflow.exit_speed() <= spout.max_exit_speed);
        assert!(inflow.flow_rate() <= spout.max_flow_rate_ml_s);
    }
}

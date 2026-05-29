use coffee_sim_core::Vec3;

use super::{brew_config::DEFAULT_BREW, state::MpmBuffers, units};

pub(crate) const MASS_UNITS_PER_ML: f32 = DEFAULT_BREW.water_mass_units_per_ml;
pub(crate) const PARTICLES_PER_ML: f32 = DEFAULT_BREW.water_particles_per_ml;

#[derive(Clone, Copy)]
pub(crate) struct SpoutSettings {
    pub origin: Vec3,
    pub direction: Vec3,
    pub nozzle_radius: f32,
    pub stem_radius: f32,
    pub discharge_coeff: f32,
    pub volume_to_ml: f32,
    pub max_flow_rate_ml_s: f32,
    pub max_exit_speed: f32,
    pub stem_length: f32,
}

impl Default for SpoutSettings {
    fn default() -> Self {
        Self {
            origin: Vec3::new(0.0, 7.3, 0.0),
            direction: vertical_emission_direction(),
            nozzle_radius: 0.18,
            stem_radius: 0.24,
            discharge_coeff: 0.92,
            volume_to_ml: units::ML_PER_SIM_UNIT_CUBED,
            max_flow_rate_ml_s: DEFAULT_BREW.max_flow_rate_ml_s,
            max_exit_speed: units::HIGH_POUR_EXIT_SPEED_SIM_UNITS,
            stem_length: 1.9,
        }
    }
}

impl SpoutSettings {
    pub fn emission_direction(&self) -> Vec3 {
        vertical_emission_direction()
    }

    pub fn translate_origin_to(&mut self, origin: Vec3) {
        self.origin = origin;
        self.direction = self.emission_direction();
    }
}

pub(crate) struct InflowState {
    flow_rate: f32,
    exit_speed: f32,
    accumulator: f32,
}

pub(crate) struct EmissionResult {
    pub emitted: u32,
    pub dropped: u32,
}

impl InflowState {
    pub fn new(exit_speed: f32) -> Self {
        let mut inflow = Self {
            flow_rate: 0.0,
            exit_speed: 0.0,
            accumulator: 0.0,
        };
        inflow.set_exit_speed(exit_speed);
        inflow
    }

    pub fn set_exit_speed(&mut self, exit_speed: f32) {
        self.exit_speed = exit_speed.clamp(0.0, units::MAX_WATER_SPEED_SIM_UNITS);
    }

    pub fn flow_rate(&self) -> f32 {
        self.flow_rate
    }

    pub fn exit_speed(&self) -> f32 {
        self.exit_speed
    }

    pub fn update(&mut self, spout: &SpoutSettings) {
        self.exit_speed = self.exit_speed.clamp(0.0, spout.max_exit_speed);
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

        let mut particle_data: Vec<[f32; 8]> = Vec::with_capacity(count as usize);
        let mut affine_data: Vec<[f32; 12]> = Vec::with_capacity(count as usize);

        let dir = spout.emission_direction();
        let emit_speed = self.exit_speed();

        // Emit material samples across the 2D nozzle aperture with one shared
        // vertical jet velocity. Flow rate controls sample count; the aperture
        // radius controls only the initial cross-section.
        let emit_origin = Vec3::new(
            spout.origin.x + dir.x * 0.18,
            spout.origin.y + dir.y * 0.18,
            spout.origin.z + dir.z * 0.18,
        );

        for i in 0..count {
            let emission_age = aperture_emission_age(i, count, dt);
            let vel = aperture_jet_velocity(dir, emit_speed);
            let aperture_offset =
                aperture_sample_offset(i, count, current_water, spout.nozzle_radius);
            let pos = aperture_sample_position(emit_origin, aperture_offset, vel, emission_age);

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

        EmissionResult {
            emitted: count,
            dropped: requested.saturating_sub(count),
        }
    }
}

fn aperture_jet_velocity(dir: Vec3, emit_speed: f32) -> Vec3 {
    dir * emit_speed
}

fn aperture_emission_age(local_index: u32, count: u32, dt: f32) -> f32 {
    debug_assert!(local_index < count);
    ((local_index as f32 + 0.5) / count as f32) * dt
}

fn aperture_sample_position(
    emit_origin: Vec3,
    aperture_offset: Vec3,
    velocity: Vec3,
    emission_age: f32,
) -> Vec3 {
    Vec3::new(
        emit_origin.x + aperture_offset.x + velocity.x * emission_age,
        emit_origin.y + aperture_offset.y + velocity.y * emission_age,
        emit_origin.z + aperture_offset.z + velocity.z * emission_age,
    )
}

fn aperture_sample_offset(
    local_index: u32,
    batch_count: u32,
    batch_seed: u32,
    nozzle_radius: f32,
) -> Vec3 {
    debug_assert!(local_index < batch_count);
    if batch_count % 2 == 1 && local_index == 0 {
        return Vec3::ZERO;
    }

    let paired_index = if batch_count % 2 == 1 {
        local_index - 1
    } else {
        local_index
    };
    let pair_index = paired_index / 2;
    let pair_side = paired_index % 2;
    let sample_key = batch_seed
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(pair_index);
    let angle_hash = hash_u32(sample_key ^ 0x85eb_ca6b);
    let radius_hash = hash_u32(sample_key ^ 0xc2b2_ae35);
    // Push samples off the dead-center axis. Without this floor a single
    // sample at the exact origin pins the jet to a one-particle-wide column
    // and visible ringing develops along the central streamline. Sampling
    // uniformly in area (sqrt of the fraction) then keeps the disk coverage
    // even outside the small inner exclusion.
    let radial_fraction = unit_float_from_hash(radius_hash).max(0.12);
    let theta = unit_float_from_hash(angle_hash) * std::f32::consts::TAU;
    let radius = nozzle_radius * radial_fraction.sqrt();
    let side = if pair_side == 0 { 1.0 } else { -1.0 };
    Vec3::new(
        side * radius * theta.cos(),
        0.0,
        side * radius * theta.sin(),
    )
}

fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn unit_float_from_hash(hash: u32) -> f32 {
    ((hash >> 8) as f32) * (1.0 / 16_777_216.0)
}

fn vertical_emission_direction() -> Vec3 {
    Vec3::new(0.0, -1.0, 0.0)
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
    fn flow_rate_from_speed_is_monotonic_and_capped() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;
        let low = flow_rate_from_speed(4.0, nozzle_radius, 0.92, 5.4, 4.0);
        let mid = flow_rate_from_speed(10.0, nozzle_radius, 0.92, 5.4, 4.0);
        let high = flow_rate_from_speed(40.0, nozzle_radius, 0.92, 5.4, 4.0);

        assert!(low > 0.0);
        assert!(mid > low);
        assert!(high >= mid);
        assert!(high <= 4.0);
        assert_eq!(
            flow_rate_from_speed(0.0, nozzle_radius, 0.92, 5.4, 4.0),
            0.0
        );
    }

    #[test]
    fn inflow_update_uses_direct_speed_and_flow() {
        let spout = SpoutSettings::default();
        let mut inflow = InflowState::new(units::GENTLE_POUR_EXIT_SPEED_SIM_UNITS);
        inflow.update(&spout);

        assert!((inflow.exit_speed() - units::GENTLE_POUR_EXIT_SPEED_SIM_UNITS).abs() < 1e-6);
        assert!(inflow.flow_rate() > 0.0);
        assert!(inflow.exit_speed() <= spout.max_exit_speed);
        assert!(inflow.flow_rate() <= spout.max_flow_rate_ml_s);
    }

    #[test]
    fn inflow_update_clamps_to_spout_speed_limit() {
        let spout = SpoutSettings::default();
        let mut inflow = InflowState::new(units::MAX_WATER_SPEED_SIM_UNITS);
        inflow.update(&spout);

        assert!((inflow.exit_speed() - spout.max_exit_speed).abs() < 1e-6);
        assert!(inflow.flow_rate() <= spout.max_flow_rate_ml_s);
    }

    #[test]
    fn spout_controls_preserve_vertical_jet_direction() {
        let mut spout = SpoutSettings::default();
        let before_origin = spout.origin;
        let before_direction = spout.direction;

        let new_origin = before_origin + Vec3::new(-0.3, 0.0, 0.0);
        spout.translate_origin_to(new_origin);

        assert!((spout.origin.x - new_origin.x).abs() < 1e-5);
        assert!((spout.origin.y - new_origin.y).abs() < 1e-5);
        assert!((spout.origin.z - new_origin.z).abs() < 1e-5);
        assert!((spout.direction.x - before_direction.x).abs() < 1e-5);
        assert!((spout.direction.y - before_direction.y).abs() < 1e-5);
        assert!((spout.direction.z - before_direction.z).abs() < 1e-5);
        assert!((spout.direction.x).abs() < 1e-5);
        assert!((spout.direction.y + 1.0).abs() < 1e-5);
        assert!((spout.direction.z).abs() < 1e-5);
    }

    #[test]
    fn aperture_jet_velocity_is_vertical() {
        let dir = SpoutSettings::default().emission_direction();
        let speed = 12.0;

        let velocity = aperture_jet_velocity(dir, speed);

        assert!(velocity.x.abs() < 1e-6);
        assert!((velocity.y + speed).abs() < 1e-6);
        assert!(velocity.z.abs() < 1e-6);
        assert!((velocity.dot(dir) - speed).abs() < 1e-5);
        assert!((velocity.length() - speed).abs() < 1e-5);
    }

    #[test]
    fn aperture_samples_fill_nozzle_disk_without_vertical_offset() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;

        let mut max_x = 0.0_f32;
        let mut max_z = 0.0_f32;
        for i in 0..32 {
            let offset = aperture_sample_offset(i, 32, 17, nozzle_radius);
            assert!(offset.y.abs() < 1e-6);
            assert!(offset.length() <= nozzle_radius + 1e-6);
            max_x = max_x.max(offset.x.abs());
            max_z = max_z.max(offset.z.abs());
        }

        assert!(max_x > nozzle_radius * 0.5);
        assert!(max_z > nozzle_radius * 0.5);
    }

    #[test]
    fn aperture_sample_angles_do_not_advance_as_a_visible_spiral() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;
        let mut previous_angle = 0.0_f32;
        let mut min_turn_delta = f32::MAX;
        let mut max_turn_delta = 0.0_f32;

        for i in (0..16).step_by(2) {
            let offset = aperture_sample_offset(i, 16, 23, nozzle_radius);
            let angle = offset.z.atan2(offset.x);
            if i > 0 {
                let mut delta = (angle - previous_angle).abs();
                if delta > std::f32::consts::PI {
                    delta = std::f32::consts::TAU - delta;
                }
                min_turn_delta = min_turn_delta.min(delta);
                max_turn_delta = max_turn_delta.max(delta);
            }
            previous_angle = angle;
        }

        assert!(
            max_turn_delta - min_turn_delta > 0.5,
            "sequential aperture angles should be scrambled, not one fixed turn"
        );
    }

    #[test]
    fn aperture_batches_have_zero_lateral_centroid() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;

        for count in 1..10 {
            let mut sum = Vec3::ZERO;
            for i in 0..count {
                sum = sum + aperture_sample_offset(i, count, 101, nozzle_radius);
            }

            assert!(
                sum.x.abs() < 1e-6 && sum.y.abs() < 1e-6 && sum.z.abs() < 1e-6,
                "batch count {count} had lateral aperture bias: {sum:?}"
            );
        }
    }

    #[test]
    fn emission_age_advects_samples_along_jet_axis() {
        // Samples are forward-extrapolated by `emission_age * velocity`, so
        // larger ages land further along the jet (downward, for the default
        // vertical spout). Lateral coordinates are untouched.
        let dir = SpoutSettings::default().emission_direction();
        let velocity = aperture_jet_velocity(dir, 12.0);
        let origin = Vec3::new(0.0, 7.0, 0.0);
        let aperture_offset = Vec3::new(0.1, 0.0, -0.1);

        let younger = aperture_sample_position(origin, aperture_offset, velocity, 0.001);
        let older = aperture_sample_position(origin, aperture_offset, velocity, 0.004);

        assert!(younger.y < origin.y);
        assert!(older.y < younger.y);
        assert!((younger.x - aperture_offset.x).abs() < 1e-6);
        assert!((younger.z - aperture_offset.z).abs() < 1e-6);
    }

    #[test]
    fn emitted_samples_are_spaced_along_the_same_jet_axis() {
        let dt = 1.0 / 60.0;
        let count = 6;
        let mut previous_age = 0.0;

        for i in 0..count {
            let age = aperture_emission_age(i, count, dt);
            assert!(age > 0.0 && age < dt);
            assert!(age > previous_age);
            previous_age = age;
        }
    }
}

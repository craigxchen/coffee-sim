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
    slug_sample_cursor: u64,
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
            slug_sample_cursor: 0,
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

        // Emit material samples as coherent cylindrical slug layers: flow rate
        // controls sample count, while the nozzle shape controls the repeating
        // cross-section and the cursor preserves continuity across frames.
        let emit_origin = Vec3::new(
            spout.origin.x + dir.x * 0.18,
            spout.origin.y + dir.y * 0.18,
            spout.origin.z + dir.z * 0.18,
        );
        let first_slug_sample = self.slug_sample_cursor;

        for i in 0..count {
            let slug_sample = first_slug_sample + i as u64;
            let emission_age = slug_emission_age(slug_sample, first_slug_sample, count, dt);
            let vel = aperture_jet_velocity(dir, emit_speed);
            let aperture_offset = slug_sample_offset(slug_sample, spout.nozzle_radius);
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
        self.slug_sample_cursor = self.slug_sample_cursor.wrapping_add(count as u64);

        EmissionResult {
            emitted: count,
            dropped: requested.saturating_sub(count),
        }
    }
}

fn aperture_jet_velocity(dir: Vec3, emit_speed: f32) -> Vec3 {
    dir * emit_speed
}

const SLUG_LAYER_SAMPLES: u64 = 13;
const SLUG_INNER_RADIUS_FRACTION: f32 = 0.38;
const SLUG_MIDDLE_RADIUS_FRACTION: f32 = 0.62;
const SLUG_OUTER_RADIUS_FRACTION: f32 = 0.84;
const SLUG_LAYER_TWIST: f32 = 2.399_963_1;

fn slug_emission_age(slug_sample: u64, first_slug_sample: u64, count: u32, dt: f32) -> f32 {
    debug_assert!(count > 0);
    debug_assert!(slug_sample >= first_slug_sample);
    debug_assert!(slug_sample < first_slug_sample + count as u64);

    let first_layer = slug_sample_layer(first_slug_sample);
    let last_layer = slug_sample_layer(first_slug_sample + count as u64 - 1);
    let layer_count = last_layer - first_layer + 1;
    let local_layer = slug_sample_layer(slug_sample) - first_layer;

    ((local_layer as f32 + 0.5) / layer_count as f32) * dt
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

fn slug_sample_offset(slug_sample: u64, nozzle_radius: f32) -> Vec3 {
    let slot = slug_sample % SLUG_LAYER_SAMPLES;
    if slot == 0 {
        return Vec3::ZERO;
    }

    let pair_slot = slot - 1;
    let pair_index = pair_slot / 2;
    let pair_side = pair_slot % 2;
    let layer_phase = slug_layer_phase(slug_sample_layer(slug_sample));
    let pair_angle = layer_phase + pair_index as f32 * (std::f32::consts::TAU / 6.0);
    let radial_fraction = if pair_index < 2 {
        SLUG_INNER_RADIUS_FRACTION
    } else if pair_index < 4 {
        SLUG_MIDDLE_RADIUS_FRACTION
    } else {
        SLUG_OUTER_RADIUS_FRACTION
    };
    let radius = nozzle_radius * radial_fraction;
    let side = if pair_side == 0 { 1.0 } else { -1.0 };

    Vec3::new(
        side * radius * pair_angle.cos(),
        0.0,
        side * radius * pair_angle.sin(),
    )
}

fn slug_sample_layer(slug_sample: u64) -> u64 {
    slug_sample / SLUG_LAYER_SAMPLES
}

fn slug_layer_phase(layer: u64) -> f32 {
    (layer as f32 * SLUG_LAYER_TWIST).rem_euclid(std::f32::consts::TAU)
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
    fn slug_samples_fill_nozzle_disk_without_vertical_offset() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;

        let mut max_x = 0.0_f32;
        let mut max_z = 0.0_f32;
        for i in 0..SLUG_LAYER_SAMPLES * 4 {
            let offset = slug_sample_offset(i, nozzle_radius);
            assert!(offset.y.abs() < 1e-6);
            assert!(offset.length() <= nozzle_radius + 1e-6);
            max_x = max_x.max(offset.x.abs());
            max_z = max_z.max(offset.z.abs());
        }

        assert!(max_x > nozzle_radius * 0.5);
        assert!(max_z > nozzle_radius * 0.5);
    }

    #[test]
    fn slug_layers_twist_deterministically_along_the_stream() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;

        let first_layer_offset = slug_sample_offset(1, nozzle_radius);
        let second_layer_offset = slug_sample_offset(SLUG_LAYER_SAMPLES + 1, nozzle_radius);
        let repeat_first_layer_offset = slug_sample_offset(1, nozzle_radius);

        let first_angle = first_layer_offset.z.atan2(first_layer_offset.x);
        let second_angle = second_layer_offset.z.atan2(second_layer_offset.x);
        let mut delta = (second_angle - first_angle).abs();
        if delta > std::f32::consts::PI {
            delta = std::f32::consts::TAU - delta;
        }

        assert!(delta > 0.5);
        assert!((repeat_first_layer_offset.x - first_layer_offset.x).abs() < 1e-6);
        assert!((repeat_first_layer_offset.z - first_layer_offset.z).abs() < 1e-6);
    }

    #[test]
    fn slug_layers_have_zero_lateral_centroid() {
        let nozzle_radius = SpoutSettings::default().nozzle_radius;

        for layer in 0..4 {
            let mut sum = Vec3::ZERO;
            let layer_start = layer * SLUG_LAYER_SAMPLES;
            for i in 0..SLUG_LAYER_SAMPLES {
                sum = sum + slug_sample_offset(layer_start + i, nozzle_radius);
            }

            assert!(
                sum.x.abs() < 1e-6 && sum.y.abs() < 1e-6 && sum.z.abs() < 1e-6,
                "slug layer {layer} had lateral aperture bias: {sum:?}"
            );
        }
    }

    #[test]
    fn emission_age_backdates_samples_downstream() {
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
        let count = (SLUG_LAYER_SAMPLES * 2) as u32;
        let first_sample = 0;
        let mut previous_age = 0.0;

        for i in 0..count {
            let age = slug_emission_age(first_sample + i as u64, first_sample, count, dt);
            assert!(age > 0.0 && age < dt);
            assert!(age >= previous_age);
            previous_age = age;
        }
    }

    #[test]
    fn emission_age_groups_samples_by_slug_layer() {
        let dt = 1.0 / 60.0;
        let count = (SLUG_LAYER_SAMPLES * 2) as u32;
        let first_sample = 0;

        let first_layer_age = slug_emission_age(0, first_sample, count, dt);
        let same_layer_age = slug_emission_age(SLUG_LAYER_SAMPLES - 1, first_sample, count, dt);
        let second_layer_age = slug_emission_age(SLUG_LAYER_SAMPLES, first_sample, count, dt);

        assert!((same_layer_age - first_layer_age).abs() < 1e-6);
        assert!(second_layer_age > first_layer_age);
    }
}

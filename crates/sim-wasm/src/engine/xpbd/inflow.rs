use coffee_sim_core::Vec3;

use crate::materials::water::{particle_mass_units, PARTICLES_PER_ML};
use crate::scene::{sim_speed_from_meters_per_second, SpoutSettings};

#[derive(Clone, Debug)]
pub(crate) struct InflowState {
    flow_rate: f32,
    exit_speed: f32,
    accumulator: f32,
    cursor: u64,
}

pub(crate) struct Emission {
    pub particles: Vec<([f32; 4], [f32; 4], [f32; 4], [f32; 4])>,
    pub requested: u32,
}

impl InflowState {
    pub(crate) fn new(exit_speed_m_s: f32) -> Self {
        Self {
            flow_rate: 0.0,
            exit_speed: sim_speed_from_meters_per_second(exit_speed_m_s),
            accumulator: 0.0,
            cursor: 0,
        }
    }

    pub(crate) fn set_exit_speed_m_s(&mut self, speed_m_s: f32, spout: &SpoutSettings) {
        self.exit_speed =
            sim_speed_from_meters_per_second(speed_m_s).clamp(0.0, spout.max_exit_speed);
        self.update(spout);
    }

    pub(crate) fn exit_speed(&self) -> f32 {
        self.exit_speed
    }

    pub(crate) fn flow_rate(&self) -> f32 {
        self.flow_rate
    }

    pub(crate) fn update(&mut self, spout: &SpoutSettings) {
        self.exit_speed = self.exit_speed.clamp(0.0, spout.max_exit_speed);
        let area = std::f32::consts::PI * spout.nozzle_radius * spout.nozzle_radius;
        self.flow_rate = (area * self.exit_speed * spout.discharge_coeff * spout.volume_to_ml)
            .min(spout.max_flow_rate_ml_s);
    }

    pub(crate) fn emit(&mut self, spout: &SpoutSettings, dt: f32, radius: f32) -> Emission {
        self.update(spout);
        if self.flow_rate < 1e-6 {
            self.accumulator = 0.0;
            return Emission {
                particles: Vec::new(),
                requested: 0,
            };
        }
        self.accumulator += self.flow_rate * PARTICLES_PER_ML * dt;
        let requested = self.accumulator.floor() as u32;
        self.accumulator -= requested as f32;
        let mut particles = Vec::with_capacity(requested as usize);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        for i in 0..requested {
            let slot = self.cursor + i as u64;
            let angle = slot as f32 * 2.399_963_1;
            let ring = ((slot % 7) as f32 / 6.0).sqrt();
            let r = spout.nozzle_radius * 0.85 * ring;
            let pos = Vec3::new(
                spout.origin.x + angle.cos() * r,
                spout.origin.y - 0.18,
                spout.origin.z + angle.sin() * r,
            );
            let vel = dir * self.exit_speed;
            particles.push((
                [pos.x, pos.y, pos.z, 0.0],
                [vel.x, vel.y, vel.z, particle_mass_units()],
                [radius, 92.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ));
        }
        self.cursor = self.cursor.wrapping_add(requested as u64);
        Emission {
            particles,
            requested,
        }
    }
}

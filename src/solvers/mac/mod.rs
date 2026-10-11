//! MAC porous-mixture reference solver, selectable through the ordinary Solver registry.
//!
//! Physics is computed on the CPU in f64, on native and WASM alike. GPU buffers only
//! expose the canonical renderer layout. This is an explicit correctness reference, not
//! a claim of 200k-particle real-time performance. See docs/plans/solver_mac.md.

mod core;
mod exchange;
mod grid;
#[cfg(test)]
mod tests;

pub use core::{Diagnostics, GrainPoint, MacSimulation, WaterPoint};

use std::sync::Arc;

use crate::emission::EmissionInput;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::solvers::base::Solver;
use crate::utils::{buffers::ParticleBuffers, config::Config, gpu::GpuContext};

#[derive(Clone, Debug)]
pub struct MacConfig {
    /// 0 derives h = 2 * particle_spacing. All other sizes are in scene units.
    pub cell_size: f64,
    /// GPU backend pool and fixed red/black pressure sweeps.
    pub gpu_particle_capacity: u32,
    pub gpu_pressure_iterations: u32,
    pub max_dt: f64,
    pub cfl: f64,
    pub pressure_iterations: u32,
    pub pressure_tolerance: f64,
    pub contact_iterations: u32,
    /// Diagnostic fixed-skeleton mode; the normal solver advances both phases.
    pub frozen_bed: bool,
    /// Open y-min to liquid only; analytic scene filters also retain grains.
    pub open_bottom: bool,
    pub initial_moisture: f64,
    /// Fractional external grain-volume increase at full moisture. Absorbed mass is
    /// independent: internal pore filling need not expand the grain by the same volume.
    pub swelling_ratio: f64,
    /// Magnitude of the simple retention law p_c = -P*(1-S). 0 disables capillarity.
    pub capillary_pressure: f64,
    pub thermal_exchange: bool,
}

impl Default for MacConfig {
    fn default() -> Self {
        Self {
            cell_size: 0.0,
            gpu_particle_capacity: 4096,
            gpu_pressure_iterations: 64,
            max_dt: 1.0 / 120.0,
            cfl: 0.35,
            pressure_iterations: 400,
            pressure_tolerance: 1e-6,
            contact_iterations: 8,
            frozen_bed: false,
            open_bottom: false,
            initial_moisture: 0.0,
            swelling_ratio: 0.55,
            capillary_pressure: 0.0,
            thermal_exchange: true,
        }
    }
}

impl MacConfig {
    fn validate(&self) {
        assert!(self.cell_size.is_finite() && self.cell_size >= 0.0);
        assert!(self.max_dt.is_finite() && self.max_dt > 0.0);
        assert!(self.cfl.is_finite() && self.cfl > 0.0 && self.cfl <= 0.5);
        assert!(self.pressure_iterations > 0 && self.contact_iterations > 0);
        assert!(self.pressure_tolerance.is_finite() && self.pressure_tolerance > 0.0);
        assert!(self.initial_moisture.is_finite() && (0.0..=1.0).contains(&self.initial_moisture));
        assert!(self.swelling_ratio.is_finite() && self.swelling_ratio >= 0.0);
        assert!(self.capillary_pressure.is_finite() && self.capillary_pressure >= 0.0);
    }
}

pub struct MacSolver {
    simulation: MacSimulation,
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: ParticleBuffers,
    capacity: usize,
    cached_metrics: Metrics,
    cached_diagnostics: Diagnostics,
}

impl MacSolver {
    pub fn simulation(&self) -> &MacSimulation {
        &self.simulation
    }
    pub fn diagnostics(&self) -> Diagnostics {
        self.cached_diagnostics.clone()
    }

    fn upload(&mut self) {
        let count = self.simulation.water.len() + self.simulation.grains.len();
        if count > self.capacity || self.buffers.position.is_none() {
            self.capacity = count.max(1).next_power_of_two();
            let buffer = |label, stride| {
                Some(Arc::new(self.device.create_buffer(
                    &wgpu::BufferDescriptor {
                        label: Some(label),
                        size: (self.capacity * stride) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                )))
            };
            self.buffers = ParticleBuffers {
                particle_count: count as u32,
                position: buffer("mac-position", 16),
                velocity: buffer("mac-velocity", 16),
                phase_tag: buffer("mac-phase", 4),
                concentration: buffer("mac-chemistry", 16),
                temperature: None,
                moisture: buffer("mac-moisture", 4),
                alpha_s: buffer("mac-solid-fraction", 4),
                pressure: buffer("mac-pressure", 4),
            };
            // Existing canonical chemistry: water [c,T,0,0], grain [fast,slow,T,0].
            // Also supply normalized moisture; capacity derives from actual grain mass.
            self.buffers.temperature = self.buffers.concentration.clone();
        }
        self.buffers.particle_count = count as u32;
        self.update_metrics();
        if count == 0 {
            return;
        }
        let mut pos = Vec::with_capacity(count);
        let mut vel = Vec::with_capacity(count);
        let mut phase = Vec::with_capacity(count);
        let mut concentration = Vec::with_capacity(count);
        let mut moisture = Vec::with_capacity(count);
        let mut alpha = Vec::with_capacity(count);
        let mut pressure = Vec::with_capacity(count);
        let s = &self.simulation;
        let nominal = (s.mats.particle_spacing as f64).powi(3);
        for p in &s.water {
            pos.push([
                p.position.x as f32,
                p.position.y as f32,
                p.position.z as f32,
                (p.volume / nominal) as f32,
            ]);
            vel.push([
                p.velocity.x as f32,
                p.velocity.y as f32,
                p.velocity.z as f32,
                0.0,
            ]);
            phase.push(0u32);
            moisture.push(0.0f32);
            concentration.push([
                (p.solute / s.water_mass(p).max(1e-30)) as f32,
                (p.heat / s.water_heat_capacity(p).max(1e-30)) as f32,
                0.0,
                0.0,
            ]);
        }
        for g in &s.grains {
            pos.push([
                g.position.x as f32,
                g.position.y as f32,
                g.position.z as f32,
                g.absorbed as f32,
            ]);
            vel.push([
                g.velocity.x as f32,
                g.velocity.y as f32,
                g.velocity.z as f32,
                0.0,
            ]);
            phase.push(1u32);
            moisture.push(g.moisture() as f32);
            concentration.push([
                g.fast as f32,
                g.slow as f32,
                (g.heat / s.grain_heat_capacity(g).max(1e-30)) as f32,
                0.0,
            ]);
        }
        for p in s
            .water
            .iter()
            .map(|p| p.position)
            .chain(s.grains.iter().map(|g| g.position))
        {
            let stencil = s.grid.stencil(p, None);
            alpha.push(
                stencil
                    .iter()
                    .map(|&(i, w)| w * s.grid.cells[i].solid / s.grid.h.powi(3))
                    .sum::<f64>() as f32,
            );
            pressure.push(
                stencil
                    .iter()
                    .map(|&(i, w)| w * s.grid.cells[i].pressure)
                    .sum::<f64>() as f32,
            );
        }
        for (buffer, data) in [
            (&self.buffers.position, bytemuck::cast_slice(&pos)),
            (&self.buffers.velocity, bytemuck::cast_slice(&vel)),
            (&self.buffers.phase_tag, bytemuck::cast_slice(&phase)),
            (
                &self.buffers.concentration,
                bytemuck::cast_slice(&concentration),
            ),
            (&self.buffers.alpha_s, bytemuck::cast_slice(&alpha)),
            (&self.buffers.moisture, bytemuck::cast_slice(&moisture)),
            (&self.buffers.pressure, bytemuck::cast_slice(&pressure)),
        ] {
            self.queue.write_buffer(buffer.as_ref().unwrap(), 0, data);
        }
    }
    fn update_metrics(&mut self) {
        let d = self.simulation.diagnostics();
        let dose = self
            .simulation
            .grains
            .iter()
            .map(|g| g.initial_mass)
            .sum::<f64>();
        let extracted = d.dissolved_solute + d.retained_solute + d.drained_solute;
        let brewed = d.dissolved_solute + d.drained_solute;
        let water_mass = (d.free_water + d.drained_water) * self.simulation.rho();
        let mut drawdown = if d.time > 0.0 {
            self.cached_metrics.drawdown_time
        } else {
            0.0
        };
        let cup_water: f64 = self
            .simulation
            .water
            .iter()
            .filter(|p| {
                self.simulation
                    .scene
                    .solids
                    .iter()
                    .any(|solid| match solid.kind {
                        crate::utils::sdf::SolidKind::Cylinder {
                            center,
                            floor_y,
                            rim_y,
                            radius,
                        } => {
                            let dx = p.position.x - center.x as f64;
                            let dz = p.position.z - center.z as f64;
                            dx * dx + dz * dz < (radius as f64).powi(2)
                                && p.position.y >= floor_y as f64
                                && p.position.y <= rim_y as f64
                        }
                        _ => false,
                    })
            })
            .map(|p| p.volume)
            .sum();
        // Match the existing first-drain metric: time at which 2% reaches the outlet/cup.
        if drawdown == 0.0
            && cup_water + d.drained_water > 0.02 * (d.free_water + d.drained_water).max(1e-30)
        {
            drawdown = d.time as f32;
        }
        self.cached_metrics = Metrics {
            extraction_yield: (extracted / dose.max(1e-30)) as f32,
            tds: (brewed / (water_mass + brewed).max(1e-30)) as f32,
            particle_count: self.buffers.particle_count,
            iteration_count: d.pressure_iterations,
            drawdown_time: drawdown,
            evenness: self.flow_evenness(),
        };
        self.cached_diagnostics = d;
    }

    fn flow_evenness(&self) -> f32 {
        use glam::DVec3;
        let s = &self.simulation;
        if s.grains.is_empty() {
            return 1.0;
        }
        let lo = s
            .grains
            .iter()
            .fold(DVec3::splat(f64::INFINITY), |v, g| v.min(g.position));
        let hi = s
            .grains
            .iter()
            .fold(DVec3::splat(f64::NEG_INFINITY), |v, g| v.max(g.position));
        let bin = |p: DVec3| {
            let x = (((p.x - lo.x) / (hi.x - lo.x).max(1e-12) * 4.0) as usize).min(3);
            let z = (((p.z - lo.z) / (hi.z - lo.z).max(1e-12) * 4.0) as usize).min(3);
            x + 4 * z
        };
        let mut occupied = [false; 16];
        let mut flux = [0.0; 16];
        for g in &s.grains {
            occupied[bin(g.position)] = true;
        }
        for p in &s.water {
            if p.position.y >= lo.y - s.grid.h && p.position.y <= hi.y + s.grid.h {
                flux[bin(p.position)] += p.volume * (-p.velocity.y).max(0.0);
            }
        }
        let values: Vec<f64> = flux
            .into_iter()
            .zip(occupied)
            .filter_map(|(f, used)| used.then_some(f))
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if mean <= 1e-12 {
            return 1.0;
        }
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        (1.0 / (1.0 + variance.sqrt() / mean)) as f32
    }
}

impl Solver for MacSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        let mut solver = Self {
            simulation: MacSimulation::new(scene, mats, cfg),
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            buffers: ParticleBuffers::empty(),
            capacity: 0,
            cached_metrics: Metrics::default(),
            cached_diagnostics: Diagnostics::default(),
        };
        solver.upload();
        solver
    }
    fn reset(&mut self, scene: &Scene) {
        self.simulation = MacSimulation::new(scene, &self.simulation.mats, &self.simulation.cfg);
        self.upload();
    }
    fn step(&mut self, dt: f32, input: &EmissionInput) {
        self.simulation.advance(dt as f64, input);
        self.upload();
    }
    fn particles(&self) -> ParticleBuffers {
        self.buffers.clone()
    }
    fn metrics(&self) -> Metrics {
        self.cached_metrics.clone()
    }
    // There are no GPU physics dispatches. Never mislabel CPU time as GPU timestamps.
    fn profile(&self) -> Profile {
        Profile::default()
    }
}

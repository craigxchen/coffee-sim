//! Heat transfer solver.
//!
//! Solves the advection-diffusion equation for temperature:
//!   ∂T/∂t + u · ∇T = α_th · ∇²T
//!
//! Boundary conditions:
//! - Top surface: Newton cooling to ambient air
//! - Sides: heat loss through filter/dripper walls
//! - Bottom: outflow carries heat away

use ndarray::Array3;
use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray, PyReadonlyArray3};

use crate::grid::SimulationGrid;
use crate::utils::WATER_THERMAL_DIFFUSIVITY;

/// Heat transfer coefficient for convective cooling to air (W/(m²·K)).
const H_CONV_AIR: f64 = 10.0;
/// Heat transfer coefficient through dripper walls (W/(m²·K)).
const H_CONV_WALL: f64 = 5.0;
/// Ambient temperature (K).
const T_AMBIENT: f64 = 293.15; // 20°C
/// Water specific heat capacity (J/(kg·K)).
const WATER_CP: f64 = 4186.0;
/// Water density for thermal calcs (kg/m³).
const WATER_RHO: f64 = 971.8;

/// Thermal solver for the coffee bed.
#[pyclass]
pub struct ThermalSolver {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    /// Temperature field (K).
    pub temperature: Array3<f64>,
    /// Thermal diffusivity (m²/s).
    alpha: f64,
}

#[pymethods]
impl ThermalSolver {
    #[new]
    #[pyo3(signature = (grid, initial_temp_c=None))]
    pub fn new(grid: &SimulationGrid, initial_temp_c: Option<f64>) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let t_init = initial_temp_c.unwrap_or(93.0) + 273.15; // Convert °C to K

        ThermalSolver {
            nx,
            ny,
            nz,
            dx: grid.dx,
            temperature: Array3::from_elem((nx, ny, nz), t_init),
            alpha: WATER_THERMAL_DIFFUSIVITY,
        }
    }

    /// Advance temperature by one timestep.
    ///
    /// # Arguments
    /// * `dt` - Timestep (seconds)
    /// * `vel_x`, `vel_y`, `vel_z` - Velocity components
    /// * `t_inlet_c` - Inlet water temperature (°C)
    /// * `grid` - Simulation grid
    pub fn step(
        &mut self,
        dt: f64,
        vel_x: PyReadonlyArray3<'_, f64>,
        vel_y: PyReadonlyArray3<'_, f64>,
        vel_z: PyReadonlyArray3<'_, f64>,
        t_inlet_c: f64,
        grid: &SimulationGrid,
    ) {
        let vx = vel_x.as_array();
        let vy = vel_y.as_array();
        let vz = vel_z.as_array();
        let t_inlet_k = t_inlet_c + 273.15;

        self.advect_diffuse_temp(dt, &vx, &vy, &vz, t_inlet_k, grid);
    }

    /// Get the temperature field as a numpy array (in Kelvin).
    pub fn temperature_field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.temperature.clone().into_pyarray_bound(py)
    }

    /// Get the temperature field in Celsius as a numpy array.
    pub fn temperature_field_celsius<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let celsius = self.temperature.mapv(|t| t - 273.15);
        celsius.into_pyarray_bound(py)
    }

    /// Average temperature in the bed (°C).
    pub fn avg_temperature_celsius(&self, grid: &SimulationGrid) -> f64 {
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if grid.inside_bed[[i, j, k]] {
                        total += self.temperature[[i, j, k]];
                        count += 1;
                    }
                }
            }
        }
        if count == 0 {
            return 0.0;
        }
        (total / count as f64) - 273.15
    }
}

impl ThermalSolver {
    /// Advection-diffusion for temperature with upwind scheme and boundary cooling.
    fn advect_diffuse_temp(
        &mut self,
        dt: f64,
        vx: &ndarray::ArrayView3<'_, f64>,
        vy: &ndarray::ArrayView3<'_, f64>,
        vz: &ndarray::ArrayView3<'_, f64>,
        t_inlet_k: f64,
        grid: &SimulationGrid,
    ) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let dx = self.dx;
        let alpha = self.alpha;

        let t_old = self.temperature.clone();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !grid.inside_bed[[i, j, k]] {
                        // Outside bed: ambient or inlet temperature
                        if k == nz - 1 {
                            self.temperature[[i, j, k]] = t_inlet_k;
                        } else {
                            self.temperature[[i, j, k]] = T_AMBIENT;
                        }
                        continue;
                    }

                    let t = t_old[[i, j, k]];
                    let ux = vx[[i, j, k]];
                    let uy = vy[[i, j, k]];
                    let uz = vz[[i, j, k]];

                    // Upwind advection
                    let adv_x = if ux > 0.0 && i > 0 {
                        ux * (t - t_old[[i - 1, j, k]]) / dx
                    } else if ux < 0.0 && i < nx - 1 {
                        ux * (t_old[[i + 1, j, k]] - t) / dx
                    } else {
                        0.0
                    };

                    let adv_y = if uy > 0.0 && j > 0 {
                        uy * (t - t_old[[i, j - 1, k]]) / dx
                    } else if uy < 0.0 && j < ny - 1 {
                        uy * (t_old[[i, j + 1, k]] - t) / dx
                    } else {
                        0.0
                    };

                    let adv_z = if uz > 0.0 && k > 0 {
                        uz * (t - t_old[[i, j, k - 1]]) / dx
                    } else if uz < 0.0 && k < nz - 1 {
                        uz * (t_old[[i, j, k + 1]] - t) / dx
                    } else {
                        0.0
                    };

                    // Diffusion (central differences)
                    let t_xm = if i > 0 { t_old[[i - 1, j, k]] } else { t };
                    let t_xp = if i < nx - 1 { t_old[[i + 1, j, k]] } else { t };
                    let t_ym = if j > 0 { t_old[[i, j - 1, k]] } else { t };
                    let t_yp = if j < ny - 1 { t_old[[i, j + 1, k]] } else { t };
                    let t_zm = if k > 0 { t_old[[i, j, k - 1]] } else { t };
                    let t_zp = if k < nz - 1 { t_old[[i, j, k + 1]] } else { t };

                    let diff =
                        alpha * (t_xp + t_xm + t_yp + t_ym + t_zp + t_zm - 6.0 * t) / (dx * dx);

                    // Boundary cooling: apply to voxels at the boundary of the bed
                    let mut cooling = 0.0;
                    let is_top = k == nz - 1
                        || (k < nz - 1 && !grid.inside_bed[[i, j, k + 1]]);
                    let is_side = i == 0
                        || i == nx - 1
                        || j == 0
                        || j == ny - 1
                        || (i > 0 && !grid.inside_bed[[i - 1, j, k]])
                        || (i < nx - 1 && !grid.inside_bed[[i + 1, j, k]])
                        || (j > 0 && !grid.inside_bed[[i, j - 1, k]])
                        || (j < ny - 1 && !grid.inside_bed[[i, j + 1, k]]);

                    if is_top {
                        // Newton cooling at top surface
                        cooling += H_CONV_AIR * (T_AMBIENT - t) / (WATER_RHO * WATER_CP * dx);
                    }
                    if is_side {
                        cooling += H_CONV_WALL * (T_AMBIENT - t) / (WATER_RHO * WATER_CP * dx);
                    }

                    // Inlet temperature at top of bed
                    if k == nz - 1 {
                        // Top layer receives inlet water
                        self.temperature[[i, j, k]] = t_inlet_k;
                    } else {
                        self.temperature[[i, j, k]] =
                            t - dt * (adv_x + adv_y + adv_z) + dt * diff + dt * cooling;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_cooling() {
        let (nx, ny, nz) = (5, 5, 5);
        let dx = 0.001;
        let mut grid = SimulationGrid::new(nx, ny, nz, dx);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    grid.inside_bed[[i, j, k]] = true;
                    grid.porosity[[i, j, k]] = 0.4;
                    grid.particle_diameter[[i, j, k]] = 500e-6;
                }
            }
        }

        let mut solver = ThermalSolver {
            nx,
            ny,
            nz,
            dx,
            temperature: Array3::from_elem((nx, ny, nz), 366.15), // 93°C
            alpha: WATER_THERMAL_DIFFUSIVITY,
        };

        let vx = Array3::zeros((nx, ny, nz));
        let vy = Array3::zeros((nx, ny, nz));
        let vz = Array3::zeros((nx, ny, nz));

        // Step many times with no flow and lower inlet temp to drive cooling
        for _ in 0..100 {
            solver.advect_diffuse_temp(0.1, &vx.view(), &vy.view(), &vz.view(), 353.15, &grid);
        }

        // Boundary voxel should cool toward inlet/ambient
        let edge_t = solver.temperature[[0, 2, 2]];
        assert!(
            edge_t < 366.15,
            "Edge should cool: T = {edge_t}"
        );
        assert!(
            edge_t > 280.0,
            "Should not cool below ambient: T = {edge_t}"
        );
    }
}

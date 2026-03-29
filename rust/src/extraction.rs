//! Extraction kinetics and concentration transport.
//!
//! Two-phase extraction model:
//! 1. Fast surface wash: readily accessible solubles dissolve quickly.
//! 2. Slow diffusion-limited extraction: solubles inside cells diffuse out slowly.
//!
//! Concentration transport via advection-diffusion:
//!   ∂C/∂t + u · ∇C = D · ∇²C + S_extraction

use ndarray::Array3;
use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray, PyReadonlyArray3};

use crate::grid::SimulationGrid;
use crate::utils::*;

/// Extraction solver: manages extraction kinetics and concentration field.
#[pyclass]
pub struct ExtractionSolver {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,

    /// Concentration of dissolved solubles in liquid phase (kg/m³).
    pub concentration: Array3<f64>,
    /// Remaining fast-extractable mass per voxel (kg/m³ of bed).
    pub mass_fast: Array3<f64>,
    /// Remaining slow-extractable mass per voxel (kg/m³ of bed).
    pub mass_slow: Array3<f64>,
    /// Initial fast-extractable mass per voxel.
    mass_fast_0: Array3<f64>,
    /// Initial slow-extractable mass per voxel.
    mass_slow_0: Array3<f64>,

    /// Total mass extracted so far (kg).
    total_extracted: f64,
    /// Total initial coffee mass in bed (kg).
    total_coffee_mass: f64,

    /// Effective diffusivity (m²/s).
    diffusivity: f64,
    /// CO₂ remaining fraction per voxel (for bloom).
    pub co2: Array3<f64>,
}

#[pymethods]
impl ExtractionSolver {
    #[new]
    pub fn new(grid: &SimulationGrid, params: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let dx = grid.dx;
        let get_f64 = |key: &str, default: f64| -> f64 {
            params.get_item(key).ok().flatten()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(default)
        };

        let coffee_mass_kg = get_f64("coffee_mass_g", 20.0) * 1e-3;
        let fast_frac = get_f64("fast_fraction", SOLUBLES_FAST_FRACTION);
        let slow_frac = get_f64("slow_fraction", SOLUBLES_SLOW_FRACTION);
        let co2_content = get_f64("co2_content", CO2_CONTENT_FRESH);
        let diffusivity = DIFFUSIVITY_SOLUBLES / BED_TORTUOSITY;

        // Count bed voxels to distribute coffee mass
        let bed_count = grid.inside_bed.iter().filter(|&&v| v).count();
        let mass_per_voxel = if bed_count > 0 {
            coffee_mass_kg / bed_count as f64
        } else {
            0.0
        };

        let mut mass_fast = Array3::zeros((nx, ny, nz));
        let mut mass_slow = Array3::zeros((nx, ny, nz));
        let mut co2 = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if grid.inside_bed[[i, j, k]] {
                        mass_fast[[i, j, k]] = mass_per_voxel * fast_frac;
                        mass_slow[[i, j, k]] = mass_per_voxel * slow_frac;
                        co2[[i, j, k]] = mass_per_voxel * co2_content;
                    }
                }
            }
        }

        let mass_fast_0 = mass_fast.clone();
        let mass_slow_0 = mass_slow.clone();

        Ok(ExtractionSolver {
            nx,
            ny,
            nz,
            dx,
            concentration: Array3::zeros((nx, ny, nz)),
            mass_fast,
            mass_slow,
            mass_fast_0,
            mass_slow_0,
            total_extracted: 0.0,
            total_coffee_mass: coffee_mass_kg,
            diffusivity,
            co2,
        })
    }

    /// Advance extraction by one timestep.
    ///
    /// # Arguments
    /// * `dt` - Timestep (seconds)
    /// * `vel_x`, `vel_y`, `vel_z` - Velocity components
    /// * `temperature` - Temperature field (Kelvin)
    /// * `grid` - The simulation grid
    pub fn step(
        &mut self,
        dt: f64,
        vel_x: PyReadonlyArray3<'_, f64>,
        vel_y: PyReadonlyArray3<'_, f64>,
        vel_z: PyReadonlyArray3<'_, f64>,
        temperature: PyReadonlyArray3<'_, f64>,
        grid: &SimulationGrid,
    ) {
        let vx = vel_x.as_array();
        let vy = vel_y.as_array();
        let vz = vel_z.as_array();
        let temp = temperature.as_array();

        // Step 1: Update extraction kinetics (source terms)
        self.update_extraction(dt, &temp, grid);

        // Step 2: Advect and diffuse concentration
        self.advect_diffuse(dt, &vx, &vy, &vz, grid);

        // Step 3: Update CO₂ (bloom decay)
        self.update_co2(dt);
    }

    /// Get the concentration field as a numpy array.
    pub fn concentration_field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.concentration.clone().into_pyarray_bound(py)
    }

    /// Get per-voxel extraction yield field (fraction extracted).
    pub fn extraction_yield_field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let mut ey = Array3::zeros((self.nx, self.ny, self.nz));
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let m0 = self.mass_fast_0[[i, j, k]] + self.mass_slow_0[[i, j, k]];
                    if m0 > 0.0 {
                        let m_remaining = self.mass_fast[[i, j, k]] + self.mass_slow[[i, j, k]];
                        ey[[i, j, k]] = 1.0 - m_remaining / m0;
                    }
                }
            }
        }
        ey.into_pyarray_bound(py)
    }

    /// Compute TDS of outflow at the bottom face (g/100mL = %).
    pub fn outflow_tds(&self, grid: &SimulationGrid) -> f64 {
        let k = 0; // bottom layer
        let mut total_conc = 0.0;
        let mut count = 0;
        for i in 0..self.nx {
            for j in 0..self.ny {
                if grid.inside_bed[[i, j, k]] {
                    total_conc += self.concentration[[i, j, k]];
                    count += 1;
                }
            }
        }
        if count == 0 {
            return 0.0;
        }
        // concentration is in kg/m³, convert to g/100mL = g/(0.1L) = g/(1e-4 m³)
        // TDS% = (kg/m³) * (1000 g/kg) / (1e4 per 100mL) * 100 = (kg/m³) * 0.01 * 100
        let avg_conc_kg_m3 = total_conc / count as f64;
        // Convert to g/mL: kg/m³ * 1e-3 = g/mL (since 1 kg/m³ = 0.001 g/mL)
        // TDS% = g/100mL = (g/mL) * 100
        let tds_percent = avg_conc_kg_m3 * 1e-3 * 100.0;
        tds_percent
    }

    /// Overall extraction yield as a percentage of total coffee mass.
    pub fn total_extraction_yield(&self) -> f64 {
        if self.total_coffee_mass <= 0.0 {
            return 0.0;
        }
        self.total_extracted / self.total_coffee_mass * 100.0
    }

    /// Get the CO₂ field as a numpy array.
    pub fn co2_field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.co2.clone().into_pyarray_bound(py)
    }

    /// Sync CO₂ field from extraction solver into grid's co2_fraction.
    pub fn sync_co2_to_grid(&self, grid: &mut SimulationGrid) {
        grid.co2_fraction.assign(&self.co2);
    }
}

impl ExtractionSolver {
    /// Update extraction kinetics: dissolve solubles from coffee particles.
    fn update_extraction(
        &mut self,
        dt: f64,
        temperature: &ndarray::ArrayView3<'_, f64>,
        grid: &SimulationGrid,
    ) {
        let voxel_vol = self.dx.powi(3);

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if !grid.inside_bed[[i, j, k]] {
                        continue;
                    }

                    let t_k = temperature[[i, j, k]];
                    let arr_factor = arrhenius_factor(t_k);
                    let c_sat = saturation_concentration(t_k) * 1000.0; // convert g/mL to kg/m³
                    let c_local = self.concentration[[i, j, k]];
                    let a_s = grid.surface_area[[i, j, k]];
                    let porosity = grid.porosity[[i, j, k]];

                    if c_local >= c_sat || a_s <= 0.0 {
                        continue;
                    }

                    let driving_force = c_sat - c_local;

                    // Fast extraction
                    // rate has units kg/(m³·s); multiply by voxel_vol to get kg/s per voxel
                    let m_fast = self.mass_fast[[i, j, k]];
                    let m_fast_0 = self.mass_fast_0[[i, j, k]];
                    let dm_fast = if m_fast > 0.0 && m_fast_0 > 0.0 {
                        let rate = K_FAST_REF * arr_factor * driving_force * a_s * (m_fast / m_fast_0);
                        (rate * dt * voxel_vol).min(m_fast)
                    } else {
                        0.0
                    };

                    // Slow extraction
                    let m_slow = self.mass_slow[[i, j, k]];
                    let m_slow_0 = self.mass_slow_0[[i, j, k]];
                    let dm_slow = if m_slow > 0.0 && m_slow_0 > 0.0 {
                        let rate = K_SLOW_REF * arr_factor * driving_force * a_s * (m_slow / m_slow_0);
                        (rate * dt * voxel_vol).min(m_slow)
                    } else {
                        0.0
                    };

                    let dm_total = dm_fast + dm_slow;
                    self.mass_fast[[i, j, k]] -= dm_fast;
                    self.mass_slow[[i, j, k]] -= dm_slow;

                    // Add dissolved mass to concentration field
                    // dm_total is in kg per voxel, concentration is kg/m³
                    if porosity > 0.0 {
                        self.concentration[[i, j, k]] += dm_total / (voxel_vol * porosity);
                    }

                    self.total_extracted += dm_total;
                }
            }
        }
    }

    /// Advection-diffusion of concentration field using upwind scheme.
    fn advect_diffuse(
        &mut self,
        dt: f64,
        vx: &ndarray::ArrayView3<'_, f64>,
        vy: &ndarray::ArrayView3<'_, f64>,
        vz: &ndarray::ArrayView3<'_, f64>,
        grid: &SimulationGrid,
    ) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let dx = self.dx;
        let d = self.diffusivity;

        let c_old = self.concentration.clone();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !grid.inside_bed[[i, j, k]] {
                        self.concentration[[i, j, k]] = 0.0;
                        continue;
                    }

                    let c = c_old[[i, j, k]];
                    let ux = vx[[i, j, k]];
                    let uy = vy[[i, j, k]];
                    let uz = vz[[i, j, k]];

                    // Upwind advection in x
                    let adv_x = if ux > 0.0 && i > 0 {
                        ux * (c - c_old[[i - 1, j, k]]) / dx
                    } else if ux < 0.0 && i < nx - 1 {
                        ux * (c_old[[i + 1, j, k]] - c) / dx
                    } else {
                        0.0
                    };

                    // Upwind advection in y
                    let adv_y = if uy > 0.0 && j > 0 {
                        uy * (c - c_old[[i, j - 1, k]]) / dx
                    } else if uy < 0.0 && j < ny - 1 {
                        uy * (c_old[[i, j + 1, k]] - c) / dx
                    } else {
                        0.0
                    };

                    // Upwind advection in z with inflow/outflow BCs
                    let adv_z = if uz > 0.0 {
                        // Upward flow: upstream is below
                        let c_below = if k > 0 { c_old[[i, j, k - 1]] } else { c };
                        uz * (c - c_below) / dx
                    } else if uz < 0.0 {
                        // Downward flow: upstream is above
                        // At top of bed: fresh water enters (C = 0)
                        let c_above = if k < nz - 1 { c_old[[i, j, k + 1]] } else { 0.0 };
                        uz * (c_above - c) / dx
                    } else {
                        0.0
                    };

                    // Diffusion (central differences)
                    let c_xm = if i > 0 { c_old[[i - 1, j, k]] } else { c };
                    let c_xp = if i < nx - 1 { c_old[[i + 1, j, k]] } else { c };
                    let c_ym = if j > 0 { c_old[[i, j - 1, k]] } else { c };
                    let c_yp = if j < ny - 1 { c_old[[i, j + 1, k]] } else { c };
                    let c_zm = if k > 0 { c_old[[i, j, k - 1]] } else { c };
                    let c_zp = if k < nz - 1 { c_old[[i, j, k + 1]] } else { c };

                    let diff = d * (c_xp + c_xm + c_yp + c_ym + c_zp + c_zm - 6.0 * c)
                        / (dx * dx);

                    self.concentration[[i, j, k]] =
                        (c - dt * (adv_x + adv_y + adv_z) + dt * diff).max(0.0);
                }
            }
        }
    }

    /// Decay CO₂ over time (first-order kinetics).
    fn update_co2(&mut self, dt: f64) {
        let decay = (-CO2_RELEASE_RATE * dt).exp();
        self.co2.mapv_inplace(|v| v * decay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_reduces_mass() {
        let (nx, ny, nz) = (3, 3, 3);
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
        grid.recompute_derived_fields();

        // Create solver with some coffee mass
        let mut solver = ExtractionSolver {
            nx,
            ny,
            nz,
            dx,
            concentration: Array3::zeros((nx, ny, nz)),
            mass_fast: Array3::from_elem((nx, ny, nz), 0.001),
            mass_slow: Array3::from_elem((nx, ny, nz), 0.0005),
            mass_fast_0: Array3::from_elem((nx, ny, nz), 0.001),
            mass_slow_0: Array3::from_elem((nx, ny, nz), 0.0005),
            total_extracted: 0.0,
            total_coffee_mass: 0.020,
            diffusivity: DIFFUSIVITY_SOLUBLES / BED_TORTUOSITY,
            co2: Array3::zeros((nx, ny, nz)),
        };

        let temp = Array3::from_elem((nx, ny, nz), T_REF);
        solver.update_extraction(0.1, &temp.view(), &grid);

        assert!(solver.total_extracted > 0.0, "Should have extracted something");
        assert!(solver.mass_fast[[1, 1, 1]] < 0.001, "Fast mass should decrease");
    }
}

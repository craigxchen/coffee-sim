//! 3D voxel grid for the coffee bed.
//!
//! Each voxel stores local porosity, permeability, specific surface area,
//! effective particle diameter, and whether it is inside the bed geometry.

use ndarray::Array3;
use pyo3::prelude::*;
use numpy::{PyArray3, PyReadonlyArray3, IntoPyArray};

/// 3D structured Cartesian grid for the coffee bed simulation.
#[pyclass]
pub struct SimulationGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64, // voxel size in meters

    /// Local porosity field (0..1). 0 = solid, 1 = pure fluid.
    pub porosity: Array3<f64>,
    /// Base porosity (before compression/CO₂/fines modifications).
    pub porosity_base: Array3<f64>,
    /// Local permeability field (m²), computed via Kozeny-Carman.
    pub permeability: Array3<f64>,
    /// Local specific surface area (m²/m³).
    pub surface_area: Array3<f64>,
    /// Local effective particle diameter (m).
    pub particle_diameter: Array3<f64>,
    /// Mask: true if the voxel is inside the bed geometry.
    pub inside_bed: Array3<bool>,
    /// CO₂ gas volume fraction (for bloom modeling).
    pub co2_fraction: Array3<f64>,
}

#[pymethods]
impl SimulationGrid {
    #[new]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        let default_porosity = 0.4;
        SimulationGrid {
            nx,
            ny,
            nz,
            dx,
            porosity: Array3::from_elem((nx, ny, nz), default_porosity),
            porosity_base: Array3::from_elem((nx, ny, nz), default_porosity),
            permeability: Array3::zeros((nx, ny, nz)),
            surface_area: Array3::zeros((nx, ny, nz)),
            particle_diameter: Array3::zeros((nx, ny, nz)),
            inside_bed: Array3::from_elem((nx, ny, nz), false),
            co2_fraction: Array3::zeros((nx, ny, nz)),
        }
    }

    /// Return the porosity field as a numpy array.
    pub fn get_porosity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.porosity.clone().into_pyarray_bound(py)
    }

    /// Return the permeability field as a numpy array.
    pub fn get_permeability<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.permeability.clone().into_pyarray_bound(py)
    }

    /// Return the specific surface area field as a numpy array.
    pub fn get_surface_area<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.surface_area.clone().into_pyarray_bound(py)
    }

    /// Return the particle diameter field as a numpy array.
    pub fn get_particle_diameter<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.particle_diameter.clone().into_pyarray_bound(py)
    }

    /// Return the inside-bed mask as a numpy array of booleans.
    pub fn get_inside_bed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        self.inside_bed.clone().into_pyarray_bound(py)
    }

    /// Grid dimensions as (nx, ny, nz).
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Voxel size in meters.
    pub fn voxel_size(&self) -> f64 {
        self.dx
    }

    /// Total number of bed voxels.
    pub fn bed_voxel_count(&self) -> usize {
        self.inside_bed.iter().filter(|&&v| v).count()
    }

    /// Recompute permeability and surface area from current porosity.
    pub fn recompute(&mut self) {
        self.recompute_derived_fields();
    }

    /// Apply bed compression: ε = ε_base * (1 - α*P), then recompute permeability.
    pub fn apply_compression(&mut self, pressure: PyReadonlyArray3<'_, f64>, alpha: f64) {
        let p = pressure.as_array();
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.inside_bed[[i, j, k]] {
                        let eps_base = self.porosity_base[[i, j, k]];
                        let eps_new = eps_base * (1.0 - alpha * p[[i, j, k]]);
                        self.porosity[[i, j, k]] = eps_new.clamp(0.15, 0.65);
                    }
                }
            }
        }
        self.recompute_derived_fields();
    }

    /// Apply CO₂ flow impedance: ε_eff = ε_base - φ_gas, then recompute.
    pub fn apply_co2_impedance(&mut self) {
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.inside_bed[[i, j, k]] {
                        let eps_base = self.porosity_base[[i, j, k]];
                        let co2 = self.co2_fraction[[i, j, k]];
                        let eps_eff = (eps_base - co2).max(0.05);
                        self.porosity[[i, j, k]] = eps_eff;
                    }
                }
            }
        }
        self.recompute_derived_fields();
    }

    /// Apply fines migration porosity delta and recompute.
    pub fn apply_fines_migration(&mut self, delta: PyReadonlyArray3<'_, f64>) {
        let d = delta.as_array();
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.inside_bed[[i, j, k]] {
                        let eps = (self.porosity[[i, j, k]] + d[[i, j, k]]).clamp(0.10, 0.70);
                        self.porosity[[i, j, k]] = eps;
                        self.porosity_base[[i, j, k]] = eps;
                    }
                }
            }
        }
        self.recompute_derived_fields();
    }
}

impl SimulationGrid {
    /// Update permeability and surface area from porosity and particle diameter fields.
    pub fn recompute_derived_fields(&mut self) {
        use crate::utils::{kozeny_carman, specific_surface_area};
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.inside_bed[[i, j, k]] {
                        let eps = self.porosity[[i, j, k]];
                        let dp = self.particle_diameter[[i, j, k]];
                        self.permeability[[i, j, k]] = kozeny_carman(eps, dp);
                        self.surface_area[[i, j, k]] = specific_surface_area(eps, dp);
                    }
                }
            }
        }
    }

    /// Snapshot porosity into porosity_base (call after bed generation).
    pub fn snapshot_porosity_base(&mut self) {
        self.porosity_base.assign(&self.porosity);
    }
}

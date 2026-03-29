//! Fluid dynamics solver: Darcy's law pressure solver and velocity computation.
//!
//! Solves the variable-coefficient Laplace equation for pressure:
//!   ∇ · ((k/μ) ∇P) = 0
//! using Preconditioned Conjugate Gradient (PCG) with Jacobi preconditioner.
//!
//! Supports both gravity-driven pourover and pump-driven espresso regimes.

use ndarray::Array3;
use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray};

use crate::grid::SimulationGrid;
use crate::utils::{WATER_DENSITY, GRAVITY, water_viscosity};

/// Fluid solver operating on the simulation grid.
#[pyclass]
pub struct FluidSolver {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    /// Pressure field (Pa).
    pub pressure: Array3<f64>,
    /// Velocity components (m/s).
    pub vel_x: Array3<f64>,
    pub vel_y: Array3<f64>,
    pub vel_z: Array3<f64>,
    /// Hydraulic conductivity: k/μ at each voxel.
    conductivity: Array3<f64>,
    /// Whether to use Ergun correction (espresso regime).
    use_ergun: bool,
}

/// Boundary condition specification.
pub struct BoundaryConditions {
    /// Pressure at the top of the bed (Pa). For pourover: ρgH.
    pub p_top: f64,
    /// Pressure at the bottom (Pa). Typically 0 (atmospheric).
    pub p_bottom: f64,
    /// Whether sides are no-flux (true) or have some permeability.
    pub no_flux_sides: bool,
}

#[pymethods]
impl FluidSolver {
    #[new]
    pub fn new(grid: &SimulationGrid) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        FluidSolver {
            nx,
            ny,
            nz,
            dx: grid.dx,
            pressure: Array3::zeros((nx, ny, nz)),
            vel_x: Array3::zeros((nx, ny, nz)),
            vel_y: Array3::zeros((nx, ny, nz)),
            vel_z: Array3::zeros((nx, ny, nz)),
            conductivity: Array3::zeros((nx, ny, nz)),
            use_ergun: false,
        }
    }

    /// Enable/disable Ergun equation correction for espresso.
    pub fn set_ergun(&mut self, enabled: bool) {
        self.use_ergun = enabled;
    }

    /// Solve the pressure field given boundary conditions.
    ///
    /// bc_dict keys: "p_top" (float, Pa), "p_bottom" (float, Pa), "no_flux_sides" (bool)
    #[pyo3(signature = (grid, bc_dict, _temperature=None))]
    pub fn solve_pressure<'py>(
        &mut self,
        py: Python<'py>,
        grid: &SimulationGrid,
        bc_dict: &Bound<'_, pyo3::types::PyDict>,
        _temperature: Option<&Bound<'_, PyArray3<f64>>>,
    ) -> Bound<'py, PyArray3<f64>> {
        let p_top = bc_dict
            .get_item("p_top").ok().flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(WATER_DENSITY * GRAVITY * 0.05); // 5cm water column default
        let p_bottom = bc_dict
            .get_item("p_bottom").ok().flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(0.0);

        let bc = BoundaryConditions {
            p_top,
            p_bottom,
            no_flux_sides: true,
        };

        if self.use_ergun {
            // Picard iteration for Ergun (nonlinear) equation
            self.compute_conductivity(grid, _temperature);
            self.solve_pcg(grid, &bc);
            self.compute_velocity(grid);

            for _iter in 0..4 {
                self.compute_conductivity_ergun(grid);
                self.solve_pcg(grid, &bc);
                self.compute_velocity(grid);
            }
        } else {
            // Standard Darcy (linear)
            self.compute_conductivity(grid, _temperature);
            self.solve_pcg(grid, &bc);
            self.compute_velocity(grid);
        }

        self.pressure.clone().into_pyarray_bound(py)
    }

    /// Get velocity components as (vx, vy, vz) numpy arrays.
    pub fn get_velocity<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
    ) {
        (
            self.vel_x.clone().into_pyarray_bound(py),
            self.vel_y.clone().into_pyarray_bound(py),
            self.vel_z.clone().into_pyarray_bound(py),
        )
    }

    /// Compute maximum velocity magnitude in the bed (for CFL condition).
    pub fn max_velocity(&self, grid: &SimulationGrid) -> f64 {
        let mut max_v: f64 = 0.0;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if grid.inside_bed[[i, j, k]] {
                        let v = (self.vel_x[[i, j, k]].powi(2)
                            + self.vel_y[[i, j, k]].powi(2)
                            + self.vel_z[[i, j, k]].powi(2))
                        .sqrt();
                        max_v = max_v.max(v);
                    }
                }
            }
        }
        max_v
    }

    /// Compute total outflow rate (m³/s) at the bottom face.
    pub fn outflow_rate(&self, grid: &SimulationGrid) -> f64 {
        let k = 0; // bottom layer
        let mut total = 0.0;
        let area = self.dx * self.dx;
        for i in 0..self.nx {
            for j in 0..self.ny {
                if grid.inside_bed[[i, j, k]] {
                    // Outflow is downward velocity (negative z) at bottom
                    total += (-self.vel_z[[i, j, k]]).max(0.0) * area;
                }
            }
        }
        total
    }
}

impl FluidSolver {
    /// Compute Ergun-corrected conductivity using current velocity (Picard linearization).
    /// k_eff = k_darcy / (1 + β*(1-ε)*ρ*|u|*k_darcy / (ε³*d_p*μ))
    fn compute_conductivity_ergun(&mut self, grid: &SimulationGrid) {
        let mu = water_viscosity(crate::utils::T_REF);
        let rho = WATER_DENSITY;
        let ergun_b = crate::utils::ERGUN_B;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if grid.inside_bed[[i, j, k]] {
                        let perm = grid.permeability[[i, j, k]];
                        let eps = grid.porosity[[i, j, k]];
                        let dp = grid.particle_diameter[[i, j, k]];

                        let vel_mag = (self.vel_x[[i, j, k]].powi(2)
                            + self.vel_y[[i, j, k]].powi(2)
                            + self.vel_z[[i, j, k]].powi(2))
                        .sqrt();

                        let eps3 = eps * eps * eps;
                        let inertial = if eps3 > 0.0 && dp > 0.0 {
                            ergun_b * (1.0 - eps) * rho * vel_mag * perm / (eps3 * dp * mu)
                        } else {
                            0.0
                        };

                        let k_eff = perm / (1.0 + inertial);
                        // Under-relax: 70% new, 30% old
                        let old_cond = self.conductivity[[i, j, k]];
                        self.conductivity[[i, j, k]] = 0.7 * (k_eff / mu) + 0.3 * old_cond;
                    } else {
                        self.conductivity[[i, j, k]] = 0.0;
                    }
                }
            }
        }
    }

    /// Compute the hydraulic conductivity field k/μ.
    fn compute_conductivity(
        &mut self,
        grid: &SimulationGrid,
        _temperature: Option<&Bound<'_, PyArray3<f64>>>,
    ) {
        // Default: use constant viscosity at reference temperature
        let mu = water_viscosity(crate::utils::T_REF);

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if grid.inside_bed[[i, j, k]] {
                        let perm = grid.permeability[[i, j, k]];
                        self.conductivity[[i, j, k]] = perm / mu;
                    } else {
                        self.conductivity[[i, j, k]] = 0.0;
                    }
                }
            }
        }
    }

    /// Preconditioned Conjugate Gradient solver for the pressure Laplace equation.
    ///
    /// Solves: ∇ · (K ∇P) = 0  where K = k/μ
    /// with Dirichlet BCs at top (k=nz-1) and bottom (k=0),
    /// and Neumann (no-flux) on sides.
    fn solve_pcg(&mut self, grid: &SimulationGrid, bc: &BoundaryConditions) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let dx = self.dx;
        let dx2 = dx * dx;
        let max_iter = 2000;
        let tol = 1e-8;

        // Initialize pressure with linear gradient guess
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let frac = k as f64 / (nz - 1).max(1) as f64;
                    self.pressure[[i, j, k]] = bc.p_bottom + (bc.p_top - bc.p_bottom) * frac;
                }
            }
        }

        // Compute residual r = b - A*x
        // For Laplace equation with zero RHS, r = -A*x
        let mut r = Array3::zeros((nx, ny, nz));
        let mut z_vec = Array3::zeros((nx, ny, nz));
        let mut p = Array3::zeros((nx, ny, nz));
        let mut ap = Array3::zeros((nx, ny, nz));

        // Jacobi preconditioner: diagonal of A
        let mut diag = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !grid.inside_bed[[i, j, k]] || k == 0 || k == nz - 1 {
                        diag[[i, j, k]] = 1.0;
                        continue;
                    }
                    let mut d = 0.0;
                    // Sum conductivities to neighbors
                    if i > 0 { d += self.avg_conductivity(i, j, k, i-1, j, k); }
                    if i < nx-1 { d += self.avg_conductivity(i, j, k, i+1, j, k); }
                    if j > 0 { d += self.avg_conductivity(i, j, k, i, j-1, k); }
                    if j < ny-1 { d += self.avg_conductivity(i, j, k, i, j+1, k); }
                    if k > 0 { d += self.avg_conductivity(i, j, k, i, j, k-1); }
                    if k < nz-1 { d += self.avg_conductivity(i, j, k, i, j, k+1); }
                    diag[[i, j, k]] = if d > 0.0 { d / dx2 } else { 1.0 };
                }
            }
        }

        // Compute initial residual
        self.apply_laplacian(grid, &self.pressure.clone(), &mut r, bc);
        // r = -A*x (since b=0 for Laplace)
        r.mapv_inplace(|v| -v);

        // Apply preconditioner: z = M^{-1} r
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    z_vec[[i, j, k]] = r[[i, j, k]] / diag[[i, j, k]];
                }
            }
        }
        p.assign(&z_vec);

        let mut rz_old = dot(&r, &z_vec);

        for _iter in 0..max_iter {
            // ap = A * p
            self.apply_laplacian(grid, &p, &mut ap, bc);

            let pap = dot(&p, &ap);
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rz_old / pap;

            // x = x + alpha * p
            // r = r - alpha * ap
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        self.pressure[[i, j, k]] += alpha * p[[i, j, k]];
                        r[[i, j, k]] -= alpha * ap[[i, j, k]];
                    }
                }
            }

            let r_norm = dot(&r, &r).sqrt();
            if r_norm < tol {
                break;
            }

            // z = M^{-1} r
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        z_vec[[i, j, k]] = r[[i, j, k]] / diag[[i, j, k]];
                    }
                }
            }

            let rz_new = dot(&r, &z_vec);
            let beta = rz_new / rz_old;

            // p = z + beta * p
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        p[[i, j, k]] = z_vec[[i, j, k]] + beta * p[[i, j, k]];
                    }
                }
            }

            rz_old = rz_new;
        }

        // Enforce BCs
        for i in 0..nx {
            for j in 0..ny {
                self.pressure[[i, j, 0]] = bc.p_bottom;
                self.pressure[[i, j, nz - 1]] = bc.p_top;
            }
        }
    }

    /// Apply the discrete Laplacian operator: out = ∇ · (K ∇ P).
    fn apply_laplacian(
        &self,
        grid: &SimulationGrid,
        p_field: &Array3<f64>,
        out: &mut Array3<f64>,
        _bc: &BoundaryConditions,
    ) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let dx2 = self.dx * self.dx;

        out.fill(0.0);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Dirichlet BCs: fixed pressure at top and bottom
                    if k == 0 || k == nz - 1 || !grid.inside_bed[[i, j, k]] {
                        out[[i, j, k]] = 0.0;
                        continue;
                    }

                    let p_c = p_field[[i, j, k]];
                    let mut lap = 0.0;

                    // x-direction
                    if i > 0 {
                        let kc = self.avg_conductivity(i, j, k, i - 1, j, k);
                        lap += kc * (p_field[[i - 1, j, k]] - p_c);
                    }
                    if i < nx - 1 {
                        let kc = self.avg_conductivity(i, j, k, i + 1, j, k);
                        lap += kc * (p_field[[i + 1, j, k]] - p_c);
                    }
                    // Neumann BC at x-boundaries: zero flux (ghost = interior)
                    if i == 0 {
                        // No flux: effectively p_ghost = p[0], so contribution = 0
                    }
                    if i == nx - 1 {
                        // No flux
                    }

                    // y-direction
                    if j > 0 {
                        let kc = self.avg_conductivity(i, j, k, i, j - 1, k);
                        lap += kc * (p_field[[i, j - 1, k]] - p_c);
                    }
                    if j < ny - 1 {
                        let kc = self.avg_conductivity(i, j, k, i, j + 1, k);
                        lap += kc * (p_field[[i, j + 1, k]] - p_c);
                    }

                    // z-direction
                    if k > 0 {
                        let kc = self.avg_conductivity(i, j, k, i, j, k - 1);
                        lap += kc * (p_field[[i, j, k - 1]] - p_c);
                    }
                    if k < nz - 1 {
                        let kc = self.avg_conductivity(i, j, k, i, j, k + 1);
                        lap += kc * (p_field[[i, j, k + 1]] - p_c);
                    }

                    out[[i, j, k]] = lap / dx2;
                }
            }
        }
    }

    /// Harmonic average of conductivity between two neighboring voxels.
    #[inline]
    fn avg_conductivity(
        &self,
        i1: usize, j1: usize, k1: usize,
        i2: usize, j2: usize, k2: usize,
    ) -> f64 {
        let k1v = self.conductivity[[i1, j1, k1]];
        let k2v = self.conductivity[[i2, j2, k2]];
        if k1v + k2v > 0.0 {
            2.0 * k1v * k2v / (k1v + k2v)
        } else {
            0.0
        }
    }

    /// Compute velocity from pressure gradient using Darcy's law: u = -(k/μ) ∇P
    /// Also includes gravity in z-direction: u_z = -(k/μ)(∂P/∂z - ρg)
    fn compute_velocity(&mut self, grid: &SimulationGrid) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let dx = self.dx;

        self.vel_x.fill(0.0);
        self.vel_y.fill(0.0);
        self.vel_z.fill(0.0);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !grid.inside_bed[[i, j, k]] {
                        continue;
                    }
                    let cond = self.conductivity[[i, j, k]];
                    if cond <= 0.0 {
                        continue;
                    }

                    // Central differences for pressure gradient
                    let dpdx = if i > 0 && i < nx - 1 {
                        (self.pressure[[i + 1, j, k]] - self.pressure[[i - 1, j, k]]) / (2.0 * dx)
                    } else if i == 0 && nx > 1 {
                        (self.pressure[[i + 1, j, k]] - self.pressure[[i, j, k]]) / dx
                    } else if i == nx - 1 && nx > 1 {
                        (self.pressure[[i, j, k]] - self.pressure[[i - 1, j, k]]) / dx
                    } else {
                        0.0
                    };

                    let dpdy = if j > 0 && j < ny - 1 {
                        (self.pressure[[i, j + 1, k]] - self.pressure[[i, j - 1, k]]) / (2.0 * dx)
                    } else if j == 0 && ny > 1 {
                        (self.pressure[[i, j + 1, k]] - self.pressure[[i, j, k]]) / dx
                    } else if j == ny - 1 && ny > 1 {
                        (self.pressure[[i, j, k]] - self.pressure[[i, j - 1, k]]) / dx
                    } else {
                        0.0
                    };

                    let dpdz = if k > 0 && k < nz - 1 {
                        (self.pressure[[i, j, k + 1]] - self.pressure[[i, j, k - 1]]) / (2.0 * dx)
                    } else if k == 0 && nz > 1 {
                        (self.pressure[[i, j, k + 1]] - self.pressure[[i, j, k]]) / dx
                    } else if k == nz - 1 && nz > 1 {
                        (self.pressure[[i, j, k]] - self.pressure[[i, j, k - 1]]) / dx
                    } else {
                        0.0
                    };

                    // Darcy's law: u = -(k/μ) * ∇P
                    // With gravity correction in z: dP/dz includes hydrostatic component
                    self.vel_x[[i, j, k]] = -cond * dpdx;
                    self.vel_y[[i, j, k]] = -cond * dpdy;
                    // Darcy's law with gravity: u_z = -(k/μ)(dP/dz + ρg)
                    // ρg drives flow downward (negative z), adding to the pressure gradient
                    self.vel_z[[i, j, k]] = -cond * (dpdz + WATER_DENSITY * GRAVITY);
                }
            }
        }
    }
}

/// Dot product of two 3D arrays.
fn dot(a: &Array3<f64>, b: &Array3<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::SimulationGrid;

    #[test]
    fn test_pressure_solver_linear() {
        // A uniform bed should produce a linear pressure profile
        let (nx, ny, nz) = (5, 5, 10);
        let dx = 0.001; // 1mm
        let mut grid = SimulationGrid::new(nx, ny, nz, dx);

        // Set uniform bed properties
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

        let mut solver = FluidSolver::new(&grid);
        let bc = BoundaryConditions {
            p_top: 1000.0,
            p_bottom: 0.0,
            no_flux_sides: true,
        };

        solver.compute_conductivity(&grid, None);
        solver.solve_pcg(&grid, &bc);

        // Check that pressure is approximately linear in z
        let i = nx / 2;
        let j = ny / 2;
        for k in 1..nz - 1 {
            let expected = bc.p_bottom + (bc.p_top - bc.p_bottom) * (k as f64 / (nz - 1) as f64);
            let actual = solver.pressure[[i, j, k]];
            let rel_err = (actual - expected).abs() / bc.p_top;
            assert!(rel_err < 0.1, "k={k}: expected {expected}, got {actual}");
        }
    }
}

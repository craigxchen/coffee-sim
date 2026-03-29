//! Grind size distribution and bed packing.
//!
//! Models coffee grounds as a bimodal log-normal distribution of spherical
//! particles, packed into a filter geometry using random sequential deposition.

use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::LogNormal;

use crate::grid::SimulationGrid;
use crate::utils::{kozeny_carman, specific_surface_area};

/// Grind size distribution parameters.
#[derive(Clone, Debug)]
pub struct GrindParams {
    pub d_main_m: f64,      // median main grind size in meters
    pub d_fines_m: f64,     // median fines size in meters
    pub fines_fraction: f64, // mass fraction of fines
    pub sigma_main: f64,    // log-normal spread for main peak
    pub sigma_fines: f64,   // log-normal spread for fines peak
}

impl GrindParams {
    pub fn from_dict(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<Self> {
        let get_f64 = |key: &str, default: f64| -> f64 {
            dict.get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<f64>().ok())
                .unwrap_or(default)
        };

        Ok(GrindParams {
            d_main_m: get_f64("d_main_um", 700.0) * 1e-6,
            d_fines_m: get_f64("d_fines_um", 50.0) * 1e-6,
            fines_fraction: get_f64("fines_fraction", 0.15),
            sigma_main: get_f64("sigma_main", 0.2),
            sigma_fines: get_f64("sigma_fines", 0.3),
        })
    }
}

/// Sample a particle diameter from the bimodal log-normal distribution.
fn sample_particle_diameter(params: &GrindParams, rng: &mut impl Rng) -> f64 {
    let is_fine = rng.gen::<f64>() < params.fines_fraction;
    let (mu, sigma) = if is_fine {
        (params.d_fines_m.ln(), params.sigma_fines)
    } else {
        (params.d_main_m.ln(), params.sigma_main)
    };
    let dist = LogNormal::new(mu, sigma).unwrap();
    rng.sample(dist)
}

/// Filter geometry types.
#[derive(Clone, Debug)]
pub enum FilterGeometry {
    /// V60-style cone: truncated cone, 60° included angle.
    V60Cone {
        top_radius_m: f64,
        bottom_radius_m: f64,
    },
    /// Flat-bottom (e.g., Kalita Wave): cylinder.
    FlatBottom { radius_m: f64 },
    /// Espresso basket: cylinder with specific dimensions.
    EspressoBasket { radius_m: f64 },
}

impl FilterGeometry {
    pub fn from_str(geometry: &str, grid: &SimulationGrid) -> Self {
        let domain_width = grid.nx as f64 * grid.dx;
        let _domain_height = grid.nz as f64 * grid.dx;
        match geometry {
            "v60" => {
                // 60° included angle truncated cone
                let top_radius = domain_width * 0.45;
                let bottom_radius = domain_width * 0.05;
                FilterGeometry::V60Cone {
                    top_radius_m: top_radius,
                    bottom_radius_m: bottom_radius,
                }
            }
            "kalita" => FilterGeometry::FlatBottom {
                radius_m: domain_width * 0.40,
            },
            "espresso" => FilterGeometry::EspressoBasket {
                radius_m: 0.029, // 58mm diameter / 2
            },
            _ => {
                // Default to cylinder filling the domain
                FilterGeometry::FlatBottom {
                    radius_m: domain_width * 0.45,
                }
            }
        }
    }

    /// Check if a point (x, y, z) in meters is inside the filter geometry.
    /// Origin is at the center-bottom of the domain.
    fn contains(&self, x: f64, y: f64, z: f64, domain_height: f64) -> bool {
        let r = (x * x + y * y).sqrt();
        match self {
            FilterGeometry::V60Cone {
                top_radius_m,
                bottom_radius_m,
            } => {
                // Linear interpolation of radius from bottom to top
                let frac = z / domain_height;
                let radius_at_z =
                    bottom_radius_m + (top_radius_m - bottom_radius_m) * frac;
                r < radius_at_z
            }
            FilterGeometry::FlatBottom { radius_m } | FilterGeometry::EspressoBasket { radius_m } => {
                r < *radius_m
            }
        }
    }
}

/// Bed generator: creates packed particle beds in the grid.
#[pyclass]
pub struct BedGenerator;

#[pymethods]
impl BedGenerator {
    #[new]
    pub fn new() -> Self {
        BedGenerator
    }

    /// Generate a packed bed in the given grid.
    ///
    /// # Arguments
    /// * `geometry` - Filter geometry type: "v60", "kalita", "espresso"
    /// * `grind_params` - Dict with keys: d_main_um, d_fines_um, fines_fraction, sigma_main, sigma_fines
    /// * `grid` - The simulation grid to fill
    /// * `seed` - Random seed for reproducibility
    #[staticmethod]
    #[pyo3(signature = (geometry, grind_params, grid, seed=None))]
    pub fn generate(
        geometry: &str,
        grind_params: &Bound<'_, pyo3::types::PyDict>,
        grid: &mut SimulationGrid,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let params = GrindParams::from_dict(grind_params)?;
        let seed = seed.unwrap_or(42);
        generate_bed(grid, &params, geometry, seed);
        Ok(())
    }

    /// Migrate fines: small particles detach from high-velocity regions
    /// and re-lodge downstream, modifying local porosity.
    #[staticmethod]
    pub fn migrate_fines(
        grid: &mut SimulationGrid,
        vel_x: numpy::PyReadonlyArray3<'_, f64>,
        vel_y: numpy::PyReadonlyArray3<'_, f64>,
        vel_z: numpy::PyReadonlyArray3<'_, f64>,
        dt: f64,
    ) {
        let vx = vel_x.as_array();
        let vy = vel_y.as_array();
        let vz = vel_z.as_array();
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let detach_rate = 0.01; // probability scale factor
        let fines_threshold = 100e-6; // only particles < 100μm migrate

        let mut delta = ndarray::Array3::<f64>::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if !grid.inside_bed[[i, j, k]] {
                        continue;
                    }
                    let dp = grid.particle_diameter[[i, j, k]];
                    if dp > fines_threshold || dp <= 0.0 {
                        continue;
                    }

                    let vel_mag = (vx[[i, j, k]].powi(2)
                        + vy[[i, j, k]].powi(2)
                        + vz[[i, j, k]].powi(2))
                    .sqrt();

                    let detach_prob = (detach_rate * vel_mag * dt).min(0.01);
                    if detach_prob <= 0.0 {
                        continue;
                    }

                    // Porosity increase at source (particle leaves)
                    let porosity_change = detach_prob * 0.01; // small change
                    delta[[i, j, k]] += porosity_change;

                    // Deposit downstream (lower k = toward bottom)
                    if k > 0 && grid.inside_bed[[i, j, k - 1]] {
                        delta[[i, j, k - 1]] -= porosity_change;
                    } else if k > 1 && grid.inside_bed[[i, j, k - 2]] {
                        delta[[i, j, k - 2]] -= porosity_change;
                    }
                }
            }
        }

        // Apply delta to grid
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if grid.inside_bed[[i, j, k]] {
                        let eps = (grid.porosity[[i, j, k]] + delta[[i, j, k]]).clamp(0.10, 0.70);
                        grid.porosity[[i, j, k]] = eps;
                        grid.porosity_base[[i, j, k]] = eps;
                    }
                }
            }
        }
        grid.recompute_derived_fields();
    }
}

/// Internal bed generation using random sequential deposition.
pub fn generate_bed(grid: &mut SimulationGrid, params: &GrindParams, geometry_name: &str, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let filter = FilterGeometry::from_str(geometry_name, grid);

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;

    let cx = (nx as f64 * dx) / 2.0; // center x
    let cy = (ny as f64 * dx) / 2.0; // center y
    let domain_height = nz as f64 * dx;

    // First pass: mark which voxels are inside the bed geometry
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = (i as f64 + 0.5) * dx - cx;
                let y = (j as f64 + 0.5) * dx - cy;
                let z = (k as f64 + 0.5) * dx;
                grid.inside_bed[[i, j, k]] = filter.contains(x, y, z, domain_height);
            }
        }
    }

    // Random sequential deposition: assign particle sizes to each bed voxel
    // and compute local porosity with some spatial variation.
    // Use a stochastic approach: each voxel gets a local particle size sampled
    // from the bimodal distribution, and porosity varies around a base value.
    let base_porosity = match geometry_name {
        "espresso" => 0.35, // tighter packing
        _ => 0.40,          // typical pourover
    };

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if !grid.inside_bed[[i, j, k]] {
                    grid.porosity[[i, j, k]] = 1.0; // outside bed = pure fluid
                    grid.permeability[[i, j, k]] = 0.0;
                    grid.surface_area[[i, j, k]] = 0.0;
                    grid.particle_diameter[[i, j, k]] = 0.0;
                    continue;
                }

                // Sample particle diameter for this voxel
                let dp = sample_particle_diameter(params, &mut rng);
                grid.particle_diameter[[i, j, k]] = dp;

                // Add spatial porosity variation (±10% of base)
                let variation: f64 = rng.gen_range(-0.1..0.1);
                let porosity = (base_porosity * (1.0 + variation)).clamp(0.15, 0.65);
                grid.porosity[[i, j, k]] = porosity;

                // Gravity-driven settling: slightly lower porosity at bottom
                let depth_frac = 1.0 - (k as f64 / nz as f64);
                let settling = 1.0 - 0.05 * depth_frac; // up to 5% denser at bottom
                grid.porosity[[i, j, k]] *= settling;

                // Compute derived fields
                let eps = grid.porosity[[i, j, k]];
                grid.permeability[[i, j, k]] = kozeny_carman(eps, dp);
                grid.surface_area[[i, j, k]] = specific_surface_area(eps, dp);
            }
        }
    }

    // Snapshot base porosity for compression/CO₂ modifications
    grid.snapshot_porosity_base();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bimodal_distribution() {
        let params = GrindParams {
            d_main_m: 700e-6,
            d_fines_m: 50e-6,
            fines_fraction: 0.15,
            sigma_main: 0.2,
            sigma_fines: 0.3,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let mut fines_count = 0;
        let n = 10000;
        for _ in 0..n {
            let d = sample_particle_diameter(&params, &mut rng);
            if d < 200e-6 {
                fines_count += 1;
            }
            assert!(d > 0.0);
        }
        // Fines fraction should be roughly 15%
        let frac = fines_count as f64 / n as f64;
        assert!(frac > 0.05 && frac < 0.35, "fines fraction = {frac}");
    }
}

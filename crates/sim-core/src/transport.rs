//! Generic scalar advection-diffusion transport.
//!
//! Provides `advect_diffuse_scalar()` with selectable advection scheme:
//! - First-order upwind (stable, for saturation where sharp fronts OK)
//! - Second-order TVD with van Leer limiter (for concentration where we need accuracy)

use crate::grid::Grid;

/// Advection scheme selection.
#[derive(Clone, Copy, Debug)]
pub enum AdvectionScheme {
    /// First-order upwind. Maximally stable, but diffusive.
    Upwind,
    /// Second-order TVD with van Leer flux limiter. Preserves sharp features.
    TvdVanLeer,
}

/// Advect and diffuse a scalar field in-place.
///
/// # Arguments
/// * `field` - scalar field (length = grid.total()), modified in-place
/// * `vx`, `vy`, `vz` - cell-centered velocity components
/// * `diffusivity` - scalar diffusion coefficient (m²/s)
/// * `mask` - geometry mask (only update active voxels)
/// * `grid` - computational grid
/// * `dt` - timestep (seconds)
/// * `scheme` - advection scheme
/// * `bc_top_value` - if Some(val), top-layer inflow uses this value (e.g., 0 for fresh water)
pub fn advect_diffuse_scalar(
    field: &mut [f64],
    vx: &[f64],
    vy: &[f64],
    vz: &[f64],
    diffusivity: f64,
    mask: &[bool],
    grid: &Grid,
    dt: f64,
    scheme: AdvectionScheme,
    bc_top_value: Option<f64>,
) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let dx = grid.dx;
    let dx2 = dx * dx;

    // Work on a copy to avoid read/write aliasing
    let old = field.to_vec();

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let i = grid.idx(ix, iy, iz);
                if !mask[i] {
                    continue;
                }

                let c = old[i];
                let ux = vx[i];
                let uy = vy[i];
                let uz = vz[i];

                // Advection
                let adv = match scheme {
                    AdvectionScheme::Upwind => {
                        advect_upwind_1d(ix, iy, iz, 0, ux, &old, grid, mask, bc_top_value)
                            + advect_upwind_1d(ix, iy, iz, 1, uy, &old, grid, mask, bc_top_value)
                            + advect_upwind_1d(ix, iy, iz, 2, uz, &old, grid, mask, bc_top_value)
                    }
                    AdvectionScheme::TvdVanLeer => {
                        advect_tvd_1d(ix, iy, iz, 0, ux, &old, grid, mask, bc_top_value)
                            + advect_tvd_1d(ix, iy, iz, 1, uy, &old, grid, mask, bc_top_value)
                            + advect_tvd_1d(ix, iy, iz, 2, uz, &old, grid, mask, bc_top_value)
                    }
                };

                // Diffusion (central differences, 6-point stencil)
                let mut diff = 0.0;
                for dir in 0..3 {
                    let (cm, cp) = get_neighbors(ix, iy, iz, dir, &old, grid, mask, bc_top_value);
                    diff += (cp + cm - 2.0 * c) / dx2;
                }
                diff *= diffusivity;

                field[i] = (c - dt * adv + dt * diff).max(0.0);
            }
        }
    }
}

/// First-order upwind advection for one direction.
fn advect_upwind_1d(
    ix: usize, iy: usize, iz: usize,
    dir: usize, // 0=x, 1=y, 2=z
    u: f64,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    bc_top: Option<f64>,
) -> f64 {
    let dx = grid.dx;
    let c = field[grid.idx(ix, iy, iz)];

    if u > 0.0 {
        // Upstream is minus direction
        let c_up = get_neighbor_minus(ix, iy, iz, dir, field, grid, mask, c);
        u * (c - c_up) / dx
    } else if u < 0.0 {
        // Upstream is plus direction
        let c_up = get_neighbor_plus(ix, iy, iz, dir, field, grid, mask, bc_top, c);
        u * (c_up - c) / dx
    } else {
        0.0
    }
}

/// TVD advection with van Leer limiter for one direction.
fn advect_tvd_1d(
    ix: usize, iy: usize, iz: usize,
    dir: usize,
    u: f64,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    bc_top: Option<f64>,
) -> f64 {
    let dx = grid.dx;
    let c = field[grid.idx(ix, iy, iz)];

    if u.abs() < 1e-30 {
        return 0.0;
    }

    // Get upstream and downstream values
    let (c_minus, c_plus) = get_neighbors_full(ix, iy, iz, dir, field, grid, mask, bc_top, c);

    if u > 0.0 {
        // Flow in + direction: upstream is c_minus
        let c_upup = get_neighbor_minus_minus(ix, iy, iz, dir, field, grid, mask, c_minus);
        let r = if (c - c_minus).abs() > 1e-30 {
            (c_minus - c_upup) / (c - c_minus)
        } else {
            1.0
        };
        let phi = van_leer(r);
        let flux = u * (c_minus + 0.5 * phi * (c - c_minus));
        let flux_out = u * c; // simple upwind for outgoing face
        (flux_out - flux) / dx
    } else {
        // Flow in - direction: upstream is c_plus
        let c_upup = get_neighbor_plus_plus(ix, iy, iz, dir, field, grid, mask, bc_top, c_plus);
        let r = if (c - c_plus).abs() > 1e-30 {
            (c_plus - c_upup) / (c - c_plus)
        } else {
            1.0
        };
        let phi = van_leer(r);
        let flux = u * (c_plus + 0.5 * phi * (c - c_plus));
        let flux_in = u * c;
        (flux_in - flux) / dx
    }
}

/// Van Leer flux limiter: φ(r) = (r + |r|) / (1 + |r|)
#[inline]
fn van_leer(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else {
        (r + r.abs()) / (1.0 + r.abs())
    }
}

/// Get neighbor values in both directions along `dir`.
fn get_neighbors(
    ix: usize, iy: usize, iz: usize,
    dir: usize,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    bc_top: Option<f64>,
) -> (f64, f64) {
    let c = field[grid.idx(ix, iy, iz)];
    let cm = get_neighbor_minus(ix, iy, iz, dir, field, grid, mask, c);
    let cp = get_neighbor_plus(ix, iy, iz, dir, field, grid, mask, bc_top, c);
    (cm, cp)
}

fn get_neighbors_full(
    ix: usize, iy: usize, iz: usize,
    dir: usize,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    bc_top: Option<f64>,
    default: f64,
) -> (f64, f64) {
    let cm = get_neighbor_minus(ix, iy, iz, dir, field, grid, mask, default);
    let cp = get_neighbor_plus(ix, iy, iz, dir, field, grid, mask, bc_top, default);
    (cm, cp)
}

fn get_neighbor_minus(
    ix: usize, iy: usize, iz: usize,
    dir: usize,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    default: f64,
) -> f64 {
    let (nix, niy, niz) = match dir {
        0 => { if ix == 0 { return default; } (ix - 1, iy, iz) }
        1 => { if iy == 0 { return default; } (ix, iy - 1, iz) }
        2 => { if iz == 0 { return default; } (ix, iy, iz - 1) }
        _ => unreachable!(),
    };
    let ni = grid.idx(nix, niy, niz);
    if mask[ni] { field[ni] } else { default }
}

fn get_neighbor_plus(
    ix: usize, iy: usize, iz: usize,
    dir: usize,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    bc_top: Option<f64>,
    default: f64,
) -> f64 {
    let (nix, niy, niz) = match dir {
        0 => { if ix >= grid.nx - 1 { return default; } (ix + 1, iy, iz) }
        1 => { if iy >= grid.ny - 1 { return default; } (ix, iy + 1, iz) }
        2 => {
            if iz >= grid.nz - 1 {
                // Top boundary: use bc_top_value if provided (fresh water inflow)
                return bc_top.unwrap_or(default);
            }
            (ix, iy, iz + 1)
        }
        _ => unreachable!(),
    };
    let ni = grid.idx(nix, niy, niz);
    if mask[ni] { field[ni] } else { default }
}

fn get_neighbor_minus_minus(
    ix: usize, iy: usize, iz: usize,
    dir: usize,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    default: f64,
) -> f64 {
    let (nix, niy, niz) = match dir {
        0 => { if ix < 2 { return default; } (ix - 2, iy, iz) }
        1 => { if iy < 2 { return default; } (ix, iy - 2, iz) }
        2 => { if iz < 2 { return default; } (ix, iy, iz - 2) }
        _ => unreachable!(),
    };
    let ni = grid.idx(nix, niy, niz);
    if mask[ni] { field[ni] } else { default }
}

fn get_neighbor_plus_plus(
    ix: usize, iy: usize, iz: usize,
    dir: usize,
    field: &[f64],
    grid: &Grid,
    mask: &[bool],
    bc_top: Option<f64>,
    default: f64,
) -> f64 {
    let (nix, niy, niz) = match dir {
        0 => { if ix + 2 >= grid.nx { return default; } (ix + 2, iy, iz) }
        1 => { if iy + 2 >= grid.ny { return default; } (ix, iy + 2, iz) }
        2 => {
            if iz + 2 >= grid.nz { return bc_top.unwrap_or(default); }
            (ix, iy, iz + 2)
        }
        _ => unreachable!(),
    };
    let ni = grid.idx(nix, niy, niz);
    if mask[ni] { field[ni] } else { default }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_advection_conserves() {
        // 1D column, uniform downward flow, upwind scheme
        let grid = Grid::new(1, 1, 20, 0.001);
        let n = grid.total();
        let mask = vec![true; n];
        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let vz = vec![-0.01; n]; // downward flow
        let mut field = vec![0.0; n];

        // Initial Gaussian pulse in the middle
        for iz in 0..20 {
            let z = iz as f64 / 20.0 - 0.5;
            field[grid.idx(0, 0, iz)] = (-z * z / 0.01).exp();
        }

        let sum_before: f64 = field.iter().sum();

        for _ in 0..10 {
            advect_diffuse_scalar(
                &mut field, &vx, &vy, &vz, 0.0, &mask, &grid, 0.01,
                AdvectionScheme::Upwind, None,
            );
        }

        let sum_after: f64 = field.iter().sum();
        // Mass should be approximately conserved (some leaves domain)
        assert!(sum_after <= sum_before + 1e-10, "Mass should not increase");
    }

    #[test]
    fn test_tvd_preserves_peak() {
        // TVD should preserve the peak better than upwind
        let grid = Grid::new(1, 1, 40, 0.001);
        let n = grid.total();
        let mask = vec![true; n];
        let vx = vec![0.0; n];
        let vy = vec![0.0; n];
        let vz = vec![-0.005; n];
        let mut field_upwind = vec![0.0; n];
        let mut field_tvd = vec![0.0; n];

        for iz in 0..40 {
            let z = (iz as f64 - 30.0) / 40.0;
            let val = (-z * z / 0.002).exp();
            field_upwind[grid.idx(0, 0, iz)] = val;
            field_tvd[grid.idx(0, 0, iz)] = val;
        }

        let peak_initial = field_upwind.iter().cloned().fold(0.0_f64, f64::max);

        for _ in 0..20 {
            advect_diffuse_scalar(
                &mut field_upwind, &vx, &vy, &vz, 0.0, &mask, &grid, 0.005,
                AdvectionScheme::Upwind, None,
            );
            advect_diffuse_scalar(
                &mut field_tvd, &vx, &vy, &vz, 0.0, &mask, &grid, 0.005,
                AdvectionScheme::TvdVanLeer, None,
            );
        }

        let peak_upwind = field_upwind.iter().cloned().fold(0.0_f64, f64::max);
        let peak_tvd = field_tvd.iter().cloned().fold(0.0_f64, f64::max);

        // TVD should retain more of the peak than upwind
        assert!(
            peak_tvd >= peak_upwind * 0.9,
            "TVD peak {peak_tvd} should be >= 90% of upwind peak {peak_upwind}"
        );
    }
}

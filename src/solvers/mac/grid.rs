//! MAC topology and the local elimination used by BOTH pressure assembly and correction.
//! Scalar inventories live at cell centers; each face owns two normal velocities.

use glam::DVec3;

use crate::engine::Scene;
use crate::utils::sdf;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Cell {
    pub water: f64,
    pub solid: f64,
    pub pressure: f64,
    pub rhs: f64,
    pub diagonal: f64,
}

/// Inverse of the two-phase backward-Euler momentum block. Fixed/absent degrees of
/// freedom have zero mobility. Using this same block in A and the update is essential:
/// damping velocities first and projecting with bare 1/m does NOT recover Darcy flow.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Mobility {
    pub ww: f64,
    pub ws: f64,
    pub ss: f64,
}

impl Mobility {
    pub fn new(mw: f64, ms: f64, drag_dt: f64, water_fixed: bool, solid_fixed: bool) -> Self {
        let w = mw > 0.0 && !water_fixed;
        let s = ms > 0.0 && !solid_fixed;
        let d = if mw > 0.0 && ms > 0.0 { drag_dt } else { 0.0 };
        match (w, s) {
            (true, true) => {
                // Expanded determinant avoids cancellation when d >> either mass.
                let det = mw * ms + d * (mw + ms);
                Self {
                    ww: (ms + d) / det,
                    ws: d / det,
                    ss: (mw + d) / det,
                }
            }
            (true, false) => Self {
                ww: 1.0 / (mw + d),
                ..Self::default()
            },
            (false, true) => Self {
                ss: 1.0 / (ms + d),
                ..Self::default()
            },
            _ => Self::default(),
        }
    }

    pub fn apply(self, w: f64, s: f64) -> [f64; 2] {
        [self.ww * w + self.ws * s, self.ws * w + self.ss * s]
    }
}

#[derive(Clone, Debug)]
pub(super) struct Face {
    pub axis: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub blocked: [bool; 2],
    pub mass: [f64; 2],
    pub volume: [f64; 2],
    pub momentum: [f64; 2],
    pub velocity: [f64; 2],
    pub mobility: Mobility,
    pub coefficient: f64,
    pub force: [f64; 2],
    pub drag_dt: f64,
}

#[derive(Clone)]
pub(super) struct Grid {
    pub origin: DVec3,
    pub h: f64,
    pub dims: [usize; 3],
    pub cells: Vec<Cell>,
    pub faces: Vec<Face>,
    pub cell_faces: Vec<[(usize, f64); 6]>,
    offsets: [usize; 3],
}

impl Grid {
    pub fn new(scene: &Scene, requested_h: f64, open_bottom: bool) -> Self {
        let origin = DVec3::from_array(scene.box_min.map(f64::from));
        let extent = DVec3::from_array(scene.box_max.map(f64::from)) - origin;
        let dims = extent.to_array().map(|x| (x / requested_h).ceil() as usize);
        assert!(
            dims.iter().all(|&d| d > 0),
            "MAC domain must have positive extents"
        );
        let n = dims.iter().product::<usize>();
        assert!(
            n <= 2_000_000,
            "MAC reference grid exceeds two million cells; increase mac.cell_size"
        );
        let mut grid = Self {
            origin,
            h: requested_h,
            dims,
            cells: vec![Cell::default(); n],
            faces: Vec::new(),
            cell_faces: vec![[(0, 0.0); 6]; n],
            offsets: [0; 3],
        };
        for axis in 0..3 {
            grid.offsets[axis] = grid.faces.len();
            let mut fd = dims;
            fd[axis] += 1;
            for z in 0..fd[2] {
                for y in 0..fd[1] {
                    for x in 0..fd[0] {
                        let c = [x, y, z];
                        let mut l = c;
                        l[axis] = l[axis].saturating_sub(1);
                        let left = (c[axis] > 0).then(|| grid.index(l));
                        let right = (c[axis] < dims[axis]).then(|| grid.index(c));
                        let mut p = DVec3::new(x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
                        p[axis] -= 0.5;
                        p = origin + requested_h * p;
                        let boundary = left.is_none() || right.is_none();
                        let drain = open_bottom && axis == 1 && left.is_none();
                        let blocked = [
                            (boundary && !drain)
                                || sdf::nearest(&scene.solids, p.as_vec3(), 0).signed < 0.0,
                            boundary || sdf::nearest(&scene.solids, p.as_vec3(), 1).signed < 0.0,
                        ];
                        let f = grid.faces.len();
                        if let Some(i) = left {
                            grid.cell_faces[i][2 * axis + 1] = (f, 1.0);
                        }
                        if let Some(i) = right {
                            grid.cell_faces[i][2 * axis] = (f, -1.0);
                        }
                        grid.faces.push(Face {
                            axis,
                            left,
                            right,
                            blocked,
                            mass: [0.0; 2],
                            volume: [0.0; 2],
                            momentum: [0.0; 2],
                            velocity: [0.0; 2],
                            mobility: Mobility::default(),
                            coefficient: 0.0,
                            force: [0.0; 2],
                            drag_dt: 0.0,
                        });
                    }
                }
            }
        }
        grid
    }

    fn index(&self, c: [usize; 3]) -> usize {
        (c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]
    }

    /// Trilinear partition of unity, renormalized only at the outer domain. Identical
    /// weights are used for scatter/gather and exchange, including at domain corners.
    pub fn stencil(&self, p: DVec3, axis: Option<usize>) -> Vec<(usize, f64)> {
        let mut q = (p - self.origin) / self.h - DVec3::splat(0.5);
        let mut dims = self.dims;
        let mut offset = 0;
        if let Some(a) = axis {
            q[a] += 0.5;
            dims[a] += 1;
            offset = self.offsets[a];
        }
        let base = q.floor();
        let t = q - base;
        let mut result = Vec::with_capacity(8);
        let mut sum = 0.0;
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let d = [x, y, z];
                    let c = [
                        base.x as isize + x,
                        base.y as isize + y,
                        base.z as isize + z,
                    ];
                    if (0..3).any(|a| c[a] < 0 || c[a] >= dims[a] as isize) {
                        continue;
                    }
                    let weight: f64 = (0..3)
                        .map(|a| if d[a] == 0 { 1.0 - t[a] } else { t[a] })
                        .product();
                    if weight <= 0.0 {
                        continue;
                    }
                    let i = offset
                        + (c[2] as usize * dims[1] + c[1] as usize) * dims[0]
                        + c[0] as usize;
                    result.push((i, weight));
                    sum += weight;
                }
            }
        }
        if sum > 0.0 {
            for (_, w) in &mut result {
                *w /= sum;
            }
        }
        result
    }

    pub fn clear(&mut self) {
        self.cells.fill(Cell::default());
        for f in &mut self.faces {
            f.mass = [0.0; 2];
            f.volume = [0.0; 2];
            f.momentum = [0.0; 2];
            f.velocity = [0.0; 2];
            f.coefficient = 0.0;
        }
    }

    /// Solve the unilateral pore-volume constraint: p >= 0, A p >= b,
    /// p*(A p-b) = 0. Underfilled cells have slack (air at atmospheric pressure).
    /// A = dt D B H^-1 B^T D^T, assembled from precisely the face correction below.
    /// No density clipping and no artificial pressure at dry cells.
    pub fn project(&mut self, dt: f64, iterations: u32, tolerance: f64) -> (u32, f64) {
        for (i, c) in self.cells.iter_mut().enumerate() {
            let mut div = 0.0;
            for &(fi, sign) in &self.cell_faces[i] {
                let f = &self.faces[fi];
                div += sign * (f.volume[0] * f.velocity[0] + f.volume[1] * f.velocity[1]) / self.h;
                c.diagonal += f.coefficient;
            }
            c.rhs = (c.water + c.solid - self.h.powi(3)) / dt - div;
        }
        let mut used = 0;
        let mut residual = 0.0;
        for iteration in 0..iterations {
            for i in 0..self.cells.len() {
                if self.cells[i].water <= 1e-15 || self.cells[i].diagonal <= 1e-30 {
                    continue;
                }
                let mut b = self.cells[i].rhs;
                for &(fi, _) in &self.cell_faces[i] {
                    let f = &self.faces[fi];
                    let other = if f.left == Some(i) { f.right } else { f.left };
                    b += f.coefficient * other.map_or(0.0, |j| self.cells[j].pressure);
                }
                self.cells[i].pressure = (b / self.cells[i].diagonal).max(0.0);
            }
            used = iteration + 1;
            residual = self.residual(dt);
            if residual <= tolerance {
                break;
            }
        }
        for f in &mut self.faces {
            let dp = f.right.map_or(0.0, |i| self.cells[i].pressure)
                - f.left.map_or(0.0, |i| self.cells[i].pressure);
            let change = f.mobility.apply(f.volume[0], f.volume[1]);
            for (phase, v) in f.velocity.iter_mut().enumerate() {
                *v -= dt / self.h * dp * change[phase];
            }
        }
        (used, residual)
    }

    fn residual(&self, dt: f64) -> f64 {
        let mut max: f64 = 0.0;
        for (i, c) in self.cells.iter().enumerate() {
            if c.water <= 1e-15 {
                continue;
            }
            let mut r = c.rhs - c.diagonal * c.pressure;
            for &(fi, _) in &self.cell_faces[i] {
                let f = &self.faces[fi];
                let other = if f.left == Some(i) { f.right } else { f.left };
                r += f.coefficient * other.map_or(0.0, |j| self.cells[j].pressure);
            }
            let violation = if c.pressure > 0.0 {
                r.abs()
            } else {
                r.max(0.0)
            };
            max = max.max(violation * dt / self.h.powi(3));
        }
        max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stiff_mutual_drag_conserves_momentum_and_dissipates_relative_motion() {
        for d in [0.0, 0.1, 10.0, 1e12] {
            let (mw, ms) = (2.0, 7.0);
            let m = Mobility::new(mw, ms, d, false, false);
            let [w, s] = m.apply(mw * 3.0, -ms);
            assert!((mw * w + ms * s - (mw * 3.0 - ms)).abs() < 1e-12);
            assert!((w - s).abs() <= 4.0 + 1e-12);
            assert!(mw * w * w + ms * s * s <= mw * 9.0 + ms + 1e-12);
        }
    }

    #[test]
    fn fixed_bed_darcy_balance_is_independent_of_timestep() {
        for dt in [0.001, 0.1, 10.0] {
            let (mass, beta, force) = (2.0, 13.0, 5.0);
            let exact = force / (mass * beta);
            let m = Mobility::new(mass, 7.0, dt * mass * beta, false, true);
            let [v, _] = m.apply(mass * exact + dt * force, 0.0);
            assert!((v - exact).abs() < 1e-14);
        }
    }

    #[test]
    fn mac_stencils_partition_unity_at_faces_and_corners() {
        let scene = Scene {
            box_max: [3.0; 3],
            regions: vec![],
            ..Scene::default()
        };
        let g = Grid::new(&scene, 1.0, false);
        for p in [
            DVec3::splat(0.001),
            DVec3::new(1.2, 2.3, 0.2),
            DVec3::splat(2.999),
        ] {
            for axis in [None, Some(0), Some(1), Some(2)] {
                assert!(
                    (g.stencil(p, axis).iter().map(|(_, w)| w).sum::<f64>() - 1.0).abs() < 1e-14
                );
            }
        }
    }
}

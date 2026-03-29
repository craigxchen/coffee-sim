/// 3D structured Cartesian grid.
///
/// Voxels are indexed as flat arrays: `idx = ix * ny * nz + iy * nz + iz`.
/// World-space origin is at the center-bottom of the domain: (0, 0, 0) = bed center, z=0 at bottom.
#[derive(Clone, Debug)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64, // uniform voxel spacing (meters)
}

impl Grid {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        Grid { nx, ny, nz, dx }
    }

    /// Flat index for voxel (ix, iy, iz).
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix * self.ny * self.nz + iy * self.nz + iz
    }

    /// Total number of voxels.
    #[inline]
    pub fn total(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// 2D index for heightfield (ix, iy).
    #[inline]
    pub fn idx2d(&self, ix: usize, iy: usize) -> usize {
        ix * self.ny + iy
    }

    /// Total number of 2D columns.
    #[inline]
    pub fn total2d(&self) -> usize {
        self.nx * self.ny
    }

    /// World-space center of voxel (ix, iy, iz).
    /// Origin at center-bottom: x,y centered, z=0 at bottom.
    pub fn position(&self, ix: usize, iy: usize, iz: usize) -> (f64, f64, f64) {
        let x = (ix as f64 + 0.5 - self.nx as f64 / 2.0) * self.dx;
        let y = (iy as f64 + 0.5 - self.ny as f64 / 2.0) * self.dx;
        let z = (iz as f64 + 0.5) * self.dx;
        (x, y, z)
    }

    /// Domain height in meters.
    pub fn height(&self) -> f64 {
        self.nz as f64 * self.dx
    }

    /// Domain half-width (radius) in meters.
    pub fn radius(&self) -> f64 {
        self.nx as f64 * self.dx / 2.0
    }

    /// Check if indices are in bounds.
    #[inline]
    pub fn in_bounds(&self, ix: usize, iy: usize, iz: usize) -> bool {
        ix < self.nx && iy < self.ny && iz < self.nz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idx_roundtrip() {
        let g = Grid::new(10, 8, 6, 0.001);
        assert_eq!(g.total(), 480);
        assert_eq!(g.idx(0, 0, 0), 0);
        assert_eq!(g.idx(9, 7, 5), 479);
    }

    #[test]
    fn test_position_center() {
        let g = Grid::new(10, 10, 10, 0.002);
        let (x, y, z) = g.position(5, 5, 0);
        // ix=5, center=5.0 → (5.5 - 5.0)*0.002 = 0.001
        assert!((x - 0.001).abs() < 1e-10);
        assert!((y - 0.001).abs() < 1e-10);
        assert!((z - 0.001).abs() < 1e-10);
    }
}

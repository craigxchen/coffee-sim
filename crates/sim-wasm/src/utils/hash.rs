//! Shared uniform-grid spatial hash, binned once per step for all particle species.
//!
//! Phase 0 ships a minimal, correct **CPU** implementation (build + neighbor query),
//! verified against brute force. The GPU radix-sort version (cell-start/cell-count
//! arrays) lands when a solver needs it; this keeps the cross-solver neighbor contract
//! pinned now.

use std::collections::HashMap;

/// Uniform-grid neighbor structure over 3D points. Cell size should be the neighbor
/// search radius (e.g. the SPH support radius), so the 3×3×3 block around a query cell is
/// a superset of all points within one cell size.
pub struct SpatialHash {
    cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    fn cell_of(&self, p: [f32; 3]) -> (i32, i32, i32) {
        (
            (p[0] / self.cell_size).floor() as i32,
            (p[1] / self.cell_size).floor() as i32,
            (p[2] / self.cell_size).floor() as i32,
        )
    }

    /// Rebuild the hash from a fresh set of points; the returned indices match `points`.
    pub fn rebuild(&mut self, points: &[[f32; 3]]) {
        self.cells.clear();
        for (i, &p) in points.iter().enumerate() {
            self.cells
                .entry(self.cell_of(p))
                .or_default()
                .push(i as u32);
        }
    }

    /// Candidate neighbor indices: everything in the 3×3×3 block of cells around `p`.
    /// This is a superset of points within `cell_size`; callers apply the exact radius
    /// test. Order is unspecified.
    pub fn neighbors(&self, p: [f32; 3]) -> Vec<u32> {
        let (cx, cy, cz) = self.cell_of(p);
        let mut out = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        out.extend_from_slice(bucket);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::rng::Rng;

    fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    }

    #[test]
    fn candidates_superset_of_true_neighbors() {
        let mut rng = Rng::new(42);
        let cell = 0.25_f32;
        let points: Vec<[f32; 3]> = (0..400)
            .map(|_| {
                [
                    rng.next_f32() * 2.0 - 1.0,
                    rng.next_f32() * 2.0 - 1.0,
                    rng.next_f32() * 2.0 - 1.0,
                ]
            })
            .collect();

        let mut hash = SpatialHash::new(cell);
        hash.rebuild(&points);

        // Every brute-force neighbor within `cell` must appear in the candidate set.
        for (i, &q) in points.iter().enumerate() {
            let candidates: std::collections::HashSet<u32> =
                hash.neighbors(q).into_iter().collect();
            for (j, &p) in points.iter().enumerate() {
                if i != j && dist2(q, p) <= cell * cell {
                    assert!(
                        candidates.contains(&(j as u32)),
                        "true neighbor {j} of {i} missing from candidate set",
                    );
                }
            }
        }
    }
}

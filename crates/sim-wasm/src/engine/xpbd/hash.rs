use coffee_sim_core::Vec3;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub(crate) struct NeighborHash {
    pub cell_size: f32,
}

pub(crate) type CellCoord = (i32, i32, i32);
pub(crate) type ParticleBuckets = HashMap<CellCoord, Vec<usize>>;

impl NeighborHash {
    pub(crate) fn new(cell_size: f32) -> Self {
        Self { cell_size }
    }

    pub(crate) fn estimate_active_cells(&self, positions: &[[f32; 4]]) -> u32 {
        let mut cells = HashSet::new();
        for p in positions {
            if p[3] >= 2.0 {
                continue;
            }
            cells.insert(self.cell(Vec3::new(p[0], p[1], p[2])));
        }
        cells.len() as u32
    }

    pub(crate) fn build_type_buckets(&self, positions: &[[f32; 4]], particle_type: f32) -> ParticleBuckets {
        let mut buckets = HashMap::new();
        for (i, p) in positions.iter().enumerate() {
            if (p[3] - particle_type).abs() > 0.1 {
                continue;
            }
            buckets
                .entry(self.cell(Vec3::new(p[0], p[1], p[2])))
                .or_insert_with(Vec::new)
                .push(i);
        }
        buckets
    }

    pub(crate) fn neighbor_cells(cell: CellCoord) -> impl Iterator<Item = CellCoord> {
        (-1..=1).flat_map(move |dx| {
            (-1..=1).flat_map(move |dy| {
                (-1..=1).map(move |dz| (cell.0 + dx, cell.1 + dy, cell.2 + dz))
            })
        })
    }

    fn cell(&self, p: Vec3) -> CellCoord {
        (
            (p.x / self.cell_size).floor() as i32,
            (p.y / self.cell_size).floor() as i32,
            (p.z / self.cell_size).floor() as i32,
        )
    }
}

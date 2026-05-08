use coffee_sim_core::Vec3;

use super::filter::FilterConfig;

const RING_COUNT: usize = 10;
const SEGMENT_COUNT: usize = 32;

// Compile-time guards: the sync loops below divide by `(RING_COUNT - 1)` and
// assume `SEGMENT_COUNT >= 3` for a non-degenerate ring topology.
const _: () = assert!(RING_COUNT >= 2);
const _: () = assert!(SEGMENT_COUNT >= 3);

/// Maximum number of vertices the fill buffer emits. Used by the renderer so
/// its GPU vertex buffer is always sized to match the CPU mesh output.
pub(crate) const MAX_FILL_VERTEX_COUNT: usize = (RING_COUNT - 1) * SEGMENT_COUNT * 6;

/// Maximum number of vertices the wireframe render buffer emits. Derived from
/// the edge count: each ring has `SEGMENT_COUNT` ring edges, and every
/// non-terminal ring adds `SEGMENT_COUNT` vertical + `SEGMENT_COUNT` diagonal
/// edges. Each edge expands to two line-list vertices.
pub(crate) const MAX_RENDER_VERTEX_COUNT: usize =
    (RING_COUNT * SEGMENT_COUNT + (RING_COUNT - 1) * SEGMENT_COUNT * 2) * 2;

#[derive(Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
}

pub(crate) struct FilterMesh {
    positions: Vec<Vec3>,
    edges: Vec<Edge>,
    render_vertices: Vec<[f32; 3]>,
    fill_vertices: Vec<[f32; 3]>,
}

impl FilterMesh {
    pub(crate) fn new(config: &FilterConfig) -> Self {
        let mut rest_positions = Vec::with_capacity(RING_COUNT * SEGMENT_COUNT);
        for ring in 0..RING_COUNT {
            let ring_t = ring as f32 / (RING_COUNT - 1) as f32;
            let y = config.bot_y + (config.top_y - config.bot_y) * ring_t;
            let radius = config.radius_at_y(y);
            for seg in 0..SEGMENT_COUNT {
                let angle = 2.0 * std::f32::consts::PI * seg as f32 / SEGMENT_COUNT as f32;
                rest_positions.push(Vec3::new(
                    config.center.x + radius * angle.cos(),
                    config.center.y + y,
                    config.center.z + radius * angle.sin(),
                ));
            }
        }

        let mut edges = Vec::new();
        for ring in 0..RING_COUNT {
            for seg in 0..SEGMENT_COUNT {
                let a = ring * SEGMENT_COUNT + seg;
                let next_seg = ring * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;
                edges.push(Edge { a, b: next_seg });

                if ring + 1 < RING_COUNT {
                    let below = (ring + 1) * SEGMENT_COUNT + seg;
                    edges.push(Edge { a, b: below });

                    let diag = (ring + 1) * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;
                    edges.push(Edge { a, b: diag });
                }
            }
        }

        let positions = rest_positions;

        let mut mesh = Self {
            positions,
            edges,
            render_vertices: Vec::new(),
            fill_vertices: Vec::new(),
        };
        // Start the render mesh on the exact paper cone. The live simulation
        // currently keeps this mesh static, so construction-time sag would
        // only create a visual mismatch against the rigid V60 support.
        mesh.sync_render_vertices();
        mesh
    }

    pub(crate) fn render_vertices(&self) -> &[[f32; 3]] {
        &self.render_vertices
    }

    pub(crate) fn fill_vertices(&self) -> &[[f32; 3]] {
        &self.fill_vertices
    }

    fn sync_render_vertices(&mut self) {
        self.render_vertices.clear();
        self.render_vertices.reserve(self.edges.len() * 2);
        for edge in &self.edges {
            let a = self.positions[edge.a];
            let b = self.positions[edge.b];
            self.render_vertices.push([a.x, a.y, a.z]);
            self.render_vertices.push([b.x, b.y, b.z]);
        }

        self.fill_vertices.clear();
        self.fill_vertices
            .reserve((RING_COUNT - 1) * SEGMENT_COUNT * 6);
        for ring in 0..(RING_COUNT - 1) {
            for seg in 0..SEGMENT_COUNT {
                let a = ring * SEGMENT_COUNT + seg;
                let next_seg = ring * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;
                let below = (ring + 1) * SEGMENT_COUNT + seg;
                let diag = (ring + 1) * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;

                let p00 = self.positions[a];
                let p01 = self.positions[next_seg];
                let p10 = self.positions[below];
                let p11 = self.positions[diag];

                self.fill_vertices.extend_from_slice(&[
                    [p00.x, p00.y, p00.z],
                    [p10.x, p10.y, p10.z],
                    [p11.x, p11.y, p11.z],
                    [p00.x, p00.y, p00.z],
                    [p11.x, p11.y, p11.z],
                    [p01.x, p01.y, p01.z],
                ]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_mesh_starts_on_filter_cone() {
        let config = FilterConfig::default();
        let mesh = FilterMesh::new(&config);
        for pos in &mesh.positions {
            let local_y = pos.y - config.center.y;
            let expected = config.radius_at_y(local_y);
            let dx = pos.x - config.center.x;
            let dz = pos.z - config.center.z;
            let r = (dx * dx + dz * dz).sqrt();
            assert!((r - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn render_vertices_follow_edges() {
        let mesh = FilterMesh::new(&FilterConfig::default());
        assert_eq!(mesh.render_vertices().len(), mesh.edges.len() * 2);
    }

    #[test]
    fn max_vertex_counts_match_sync_output() {
        let mesh = FilterMesh::new(&FilterConfig::default());
        assert_eq!(mesh.fill_vertices().len(), MAX_FILL_VERTEX_COUNT);
        assert_eq!(mesh.render_vertices().len(), MAX_RENDER_VERTEX_COUNT);
    }
}

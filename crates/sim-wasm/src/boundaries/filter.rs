use coffee_sim_core::Vec3;

const RING_COUNT: usize = 10;
const SEGMENT_COUNT: usize = 32;

pub(crate) const MAX_FILL_VERTEX_COUNT: usize = (RING_COUNT - 1) * SEGMENT_COUNT * 6;
pub(crate) const MAX_RENDER_VERTEX_COUNT: usize =
    (RING_COUNT * SEGMENT_COUNT + (RING_COUNT - 1) * SEGMENT_COUNT * 2) * 2;

#[derive(Clone, Debug)]
pub(crate) struct FilterConfig {
    pub center: Vec3,
    pub top_y: f32,
    pub bot_y: f32,
    pub top_radius: f32,
    pub bot_radius: f32,
    pub thickness: f32,
    pub hole_radius: f32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            center: Vec3::new(0.0, -0.35, 0.0),
            top_y: 2.75,
            bot_y: -3.02,
            top_radius: 4.10,
            bot_radius: 0.0,
            thickness: 0.08,
            hole_radius: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    a: usize,
    b: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FilterMesh {
    positions: Vec<Vec3>,
    edges: Vec<Edge>,
    render_vertices: Vec<[f32; 3]>,
    fill_vertices: Vec<[f32; 3]>,
}

impl FilterMesh {
    pub(crate) fn new(config: &FilterConfig) -> Self {
        let mut positions = Vec::with_capacity(RING_COUNT * SEGMENT_COUNT);
        for ring in 0..RING_COUNT {
            let ring_t = ring as f32 / (RING_COUNT - 1) as f32;
            let y = config.bot_y + (config.top_y - config.bot_y) * ring_t;
            let radius = config.radius_at_y(y);
            for seg in 0..SEGMENT_COUNT {
                let angle = 2.0 * std::f32::consts::PI * seg as f32 / SEGMENT_COUNT as f32;
                positions.push(Vec3::new(
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
                let next = ring * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;
                edges.push(Edge { a, b: next });

                if ring + 1 < RING_COUNT {
                    let below = (ring + 1) * SEGMENT_COUNT + seg;
                    let diag = (ring + 1) * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;
                    edges.push(Edge { a, b: below });
                    edges.push(Edge { a, b: diag });
                }
            }
        }

        let mut mesh = Self {
            positions,
            edges,
            render_vertices: Vec::new(),
            fill_vertices: Vec::new(),
        };
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
        self.fill_vertices.reserve(MAX_FILL_VERTEX_COUNT);
        for ring in 0..(RING_COUNT - 1) {
            for seg in 0..SEGMENT_COUNT {
                let a = ring * SEGMENT_COUNT + seg;
                let next = ring * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;
                let below = (ring + 1) * SEGMENT_COUNT + seg;
                let diag = (ring + 1) * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;

                let p00 = self.positions[a];
                let p01 = self.positions[next];
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
    fn filter_mesh_counts_match_renderer_contract() {
        let mesh = FilterMesh::new(&FilterConfig::default());
        assert_eq!(mesh.fill_vertices().len(), MAX_FILL_VERTEX_COUNT);
        assert_eq!(mesh.render_vertices().len(), MAX_RENDER_VERTEX_COUNT);
    }
}

impl FilterConfig {
    pub(crate) fn radius_at_y(&self, y: f32) -> f32 {
        let height = (self.top_y - self.bot_y).max(1e-6);
        let t = ((y - self.bot_y) / height).clamp(0.0, 1.0);
        self.bot_radius + (self.top_radius - self.bot_radius) * t
    }

    pub(crate) fn inner_radius_at_y(&self, y: f32) -> f32 {
        (self.radius_at_y(y) - self.thickness).max(self.hole_radius)
    }
}

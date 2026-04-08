use coffee_sim_core::sph::Vec3;

use super::filter::FilterConfig;

const RING_COUNT: usize = 10;
const SEGMENT_COUNT: usize = 32;
const EDGE_ITERS: usize = 5;
const DAMPING: f32 = 0.18;
const GRAVITY: f32 = -0.55;
const REST_STIFFNESS: f32 = 3.25;
const EDGE_STIFFNESS: f32 = 0.86;

#[derive(Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
    rest_length: f32,
}

pub(crate) struct FilterMesh {
    config: FilterConfig,
    rest_positions: Vec<Vec3>,
    positions: Vec<Vec3>,
    prev_positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    pinned: Vec<bool>,
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
                edges.push(Edge {
                    a,
                    b: next_seg,
                    rest_length: (rest_positions[a] - rest_positions[next_seg]).length(),
                });

                if ring + 1 < RING_COUNT {
                    let below = (ring + 1) * SEGMENT_COUNT + seg;
                    edges.push(Edge {
                        a,
                        b: below,
                        rest_length: (rest_positions[a] - rest_positions[below]).length(),
                    });

                    let diag = (ring + 1) * SEGMENT_COUNT + (seg + 1) % SEGMENT_COUNT;
                    edges.push(Edge {
                        a,
                        b: diag,
                        rest_length: (rest_positions[a] - rest_positions[diag]).length(),
                    });
                }
            }
        }

        let positions = rest_positions.clone();
        let prev_positions = positions.clone();
        let velocities = vec![Vec3::ZERO; positions.len()];
        let pinned = (0..positions.len())
            .map(|idx| idx < SEGMENT_COUNT)
            .collect::<Vec<_>>();

        let mut mesh = Self {
            config: config.clone(),
            rest_positions,
            positions,
            prev_positions,
            velocities,
            pinned,
            edges,
            render_vertices: Vec::new(),
            fill_vertices: Vec::new(),
        };
        mesh.sync_render_vertices();
        mesh
    }

    pub(crate) fn step(&mut self, dt: f32, load: f32) {
        let dt = dt.max(0.0).min(1.0 / 20.0);
        if dt <= 0.0 {
            return;
        }

        let load = load.clamp(0.0, 2.0);
        self.prev_positions.copy_from_slice(&self.positions);

        let height = (self.config.top_y - self.config.bot_y).max(1e-6);
        let center = self.config.center;
        let sag_strength = (0.10 + 0.42 * load).clamp(0.06, 0.72);

        for (i, pos) in self.positions.iter_mut().enumerate() {
            if self.pinned[i] {
                *pos = self.rest_positions[i];
                self.velocities[i] = Vec3::ZERO;
                continue;
            }

            let rest = self.rest_positions[i];
            let depth_t = ((rest.y - (center.y + self.config.bot_y)) / height).clamp(0.0, 1.0);
            let profile = depth_t.powf(1.45);
            let inward = 1.0 - 0.10 * profile;
            let target = Vec3::new(
                center.x + (rest.x - center.x) * inward,
                rest.y - sag_strength * profile,
                center.z + (rest.z - center.z) * inward,
            );

            let load_force = (target - *pos) * REST_STIFFNESS;
            let gravity = Vec3::new(0.0, GRAVITY * (0.35 + 0.65 * profile), 0.0);
            self.velocities[i] = (self.velocities[i] + (load_force + gravity) * dt)
                * (1.0 / (1.0 + DAMPING * dt));
            *pos = *pos + self.velocities[i] * dt;
        }

        for _ in 0..EDGE_ITERS {
            for edge in &self.edges {
                let a_pinned = self.pinned[edge.a];
                let b_pinned = self.pinned[edge.b];
                if a_pinned && b_pinned {
                    continue;
                }

                let pa = self.positions[edge.a];
                let pb = self.positions[edge.b];
                let delta = pb - pa;
                let dist = delta.length();
                if dist < 1e-6 {
                    continue;
                }

                let diff = (dist - edge.rest_length) / dist;
                let correction = delta * (0.5 * EDGE_STIFFNESS * diff);

                match (a_pinned, b_pinned) {
                    (false, false) => {
                        self.positions[edge.a] = self.positions[edge.a] + correction;
                        self.positions[edge.b] = self.positions[edge.b] - correction;
                    }
                    (true, false) => {
                        self.positions[edge.b] = self.positions[edge.b] - correction * 2.0;
                    }
                    (false, true) => {
                        self.positions[edge.a] = self.positions[edge.a] + correction * 2.0;
                    }
                    (true, true) => {}
                }
            }
        }

        for (i, pos) in self.positions.iter_mut().enumerate() {
            if self.pinned[i] {
                *pos = self.rest_positions[i];
                self.velocities[i] = Vec3::ZERO;
                continue;
            }

            let rest = self.rest_positions[i];
            if pos.y > rest.y {
                pos.y = rest.y;
            }

            let delta = *pos - self.prev_positions[i];
            self.velocities[i] = delta / dt;
        }

        self.sync_render_vertices();
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
    fn top_rim_vertices_are_pinned() {
        let mesh = FilterMesh::new(&FilterConfig::default());
        assert!(mesh.pinned.iter().take(SEGMENT_COUNT).all(|p| *p));
    }

    #[test]
    fn load_creates_sag_but_preserves_pins() {
        let mut mesh = FilterMesh::new(&FilterConfig::default());
        let before = mesh.positions.clone();
        mesh.step(1.0 / 60.0, 1.0);
        for (after, prior) in mesh.positions[..SEGMENT_COUNT]
            .iter()
            .zip(before[..SEGMENT_COUNT].iter())
        {
            assert!((after.x - prior.x).abs() < 1e-6);
            assert!((after.y - prior.y).abs() < 1e-6);
            assert!((after.z - prior.z).abs() < 1e-6);
        }
        let lower_ring_start = (RING_COUNT - 1) * SEGMENT_COUNT;
        assert!(mesh.positions[lower_ring_start].y < before[lower_ring_start].y);
    }

    #[test]
    fn render_vertices_follow_edges() {
        let mesh = FilterMesh::new(&FilterConfig::default());
        assert_eq!(mesh.render_vertices().len(), mesh.edges.len() * 2);
    }
}

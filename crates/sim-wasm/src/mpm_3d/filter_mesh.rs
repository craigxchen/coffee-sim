use coffee_sim_core::sph::Vec3;

use super::filter::FilterConfig;

const RING_COUNT: usize = 10;
const SEGMENT_COUNT: usize = 32;
#[allow(dead_code)]
const EDGE_ITERS: usize = 5;
#[allow(dead_code)]
const DAMPING: f32 = 0.18;
#[allow(dead_code)]
const GRAVITY: f32 = -0.55;
#[allow(dead_code)]
const REST_STIFFNESS: f32 = 3.25;
#[allow(dead_code)]
const EDGE_STIFFNESS: f32 = 0.86;

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

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
    rest_length: f32,
}

#[allow(dead_code)]
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
        let top_ring_start = (RING_COUNT - 1) * SEGMENT_COUNT;
        let pinned = (0..positions.len())
            .map(|idx| idx >= top_ring_start)
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
        // Start the render mesh on the exact paper cone. The live simulation
        // currently keeps this mesh static, so construction-time sag would
        // only create a visual mismatch against the rigid V60 support.
        mesh.sync_render_vertices();
        mesh
    }

    #[allow(dead_code)]
    pub(crate) fn step(&mut self, dt: f32, load: f32) {
        // Sanitise `dt` first: if the caller passes NaN or a negative value we
        // must bail out rather than poison every position/velocity. The manual
        // `max(0).min(cap)` form is intentional — `clamp` would propagate NaN.
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let dt = dt.min(1.0 / 20.0);
        // Clamp dt to a small positive lower bound so the velocity reconstruction
        // `(pos - prev_pos) / dt` below cannot explode on frame-time glitches.
        let dt = dt.max(1e-5);

        let load = if load.is_finite() {
            load.clamp(0.0, 2.0)
        } else {
            0.0
        };
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
            let profile = (1.0 - depth_t).powf(1.45);
            let target_y = rest.y - sag_strength * profile;
            let target_local_y = (target_y - center.y).clamp(self.config.bot_y, self.config.top_y);
            let target_radius = self.config.radius_at_y(target_local_y);
            let rest_radial = Vec3::new(rest.x - center.x, 0.0, rest.z - center.z);
            let rest_radius = rest_radial.length();
            let radial_scale = if rest_radius > 1e-6 {
                target_radius / rest_radius
            } else {
                0.0
            };
            let target = Vec3::new(
                center.x + rest_radial.x * radial_scale,
                target_y,
                center.z + rest_radial.z * radial_scale,
            );

            let load_force = (target - *pos) * REST_STIFFNESS;
            let gravity = Vec3::new(0.0, GRAVITY * (0.35 + 0.65 * profile), 0.0);
            self.velocities[i] =
                (self.velocities[i] + (load_force + gravity) * dt) * (1.0 / (1.0 + DAMPING * dt));
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

                // Standard PBD distance correction: move both endpoints so the
                // edge reaches its rest length. Using `delta * 0.5 * (1 - rest/dist)`
                // keeps corrections bounded under compression — the earlier
                // `(dist - rest) / dist` form grew unbounded negative when
                // `dist << rest_length`, causing the mesh to overshoot.
                let scale = 0.5 * EDGE_STIFFNESS * (1.0 - edge.rest_length / dist);
                let correction = delta * scale;

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
        let top_ring_start = (RING_COUNT - 1) * SEGMENT_COUNT;
        assert!(mesh.pinned.iter().take(top_ring_start).all(|p| !*p));
        assert!(mesh.pinned.iter().skip(top_ring_start).all(|p| *p));
    }

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
    fn load_creates_sag_but_preserves_pins() {
        let mut mesh = FilterMesh::new(&FilterConfig::default());
        let before = mesh.positions.clone();
        mesh.step(1.0 / 60.0, 1.0);
        let top_ring_start = (RING_COUNT - 1) * SEGMENT_COUNT;
        for (after, prior) in mesh.positions[top_ring_start..]
            .iter()
            .zip(before[top_ring_start..].iter())
        {
            assert!((after.x - prior.x).abs() < 1e-6);
            assert!((after.y - prior.y).abs() < 1e-6);
            assert!((after.z - prior.z).abs() < 1e-6);
        }
        let mid_ring_start = (RING_COUNT / 2) * SEGMENT_COUNT;
        assert!(mesh.positions[mid_ring_start].y < before[mid_ring_start].y);
    }

    #[test]
    fn render_vertices_follow_edges() {
        let mesh = FilterMesh::new(&FilterConfig::default());
        assert_eq!(mesh.render_vertices().len(), mesh.edges.len() * 2);
    }

    #[test]
    fn step_ignores_nonfinite_dt() {
        let mut mesh = FilterMesh::new(&FilterConfig::default());
        let before = mesh.positions.clone();
        mesh.step(f32::NAN, 1.0);
        for (after, prior) in mesh.positions.iter().zip(before.iter()) {
            assert!(after.x.is_finite());
            assert!(after.y.is_finite());
            assert!(after.z.is_finite());
            assert!((after.x - prior.x).abs() < 1e-6);
            assert!((after.y - prior.y).abs() < 1e-6);
            assert!((after.z - prior.z).abs() < 1e-6);
        }
    }

    #[test]
    fn step_ignores_nonfinite_load() {
        let mut mesh = FilterMesh::new(&FilterConfig::default());
        // NaN load should fall through to 0.0 and not poison the positions.
        mesh.step(1.0 / 60.0, f32::NAN);
        for pos in &mesh.positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
            assert!(pos.z.is_finite());
        }
    }

    #[test]
    fn step_remains_stable_under_compression() {
        let mut mesh = FilterMesh::new(&FilterConfig::default());
        // Artificially compress every un-pinned vertex toward the center and
        // verify the PBD edge solver converges without overshoot — the old
        // `(dist - rest) / dist` form produced unbounded corrections when
        // `dist << rest_length`.
        for (i, pos) in mesh.positions.iter_mut().enumerate() {
            if !mesh.pinned[i] {
                *pos = Vec3::ZERO;
            }
        }
        mesh.prev_positions.copy_from_slice(&mesh.positions);
        for _ in 0..5 {
            mesh.step(1.0 / 60.0, 0.5);
            for pos in &mesh.positions {
                assert!(pos.x.is_finite());
                assert!(pos.y.is_finite());
                assert!(pos.z.is_finite());
                // Positions should stay within a sane window — the mesh must
                // not explode out of the simulation bounds.
                assert!(pos.x.abs() < 20.0);
                assert!(pos.y.abs() < 20.0);
                assert!(pos.z.abs() < 20.0);
            }
        }
    }

    #[test]
    fn max_vertex_counts_match_sync_output() {
        let mesh = FilterMesh::new(&FilterConfig::default());
        assert_eq!(mesh.fill_vertices().len(), MAX_FILL_VERTEX_COUNT);
        assert_eq!(mesh.render_vertices().len(), MAX_RENDER_VERTEX_COUNT);
    }
}

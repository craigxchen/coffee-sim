use coffee_sim_core::Vec3;

const STANDARD_RENDER_GRID_DX: f32 = 14.0 / 80.0;
pub(crate) const STANDARD_WATER_RENDER_RADIUS: f32 = STANDARD_RENDER_GRID_DX * 0.18;
pub(crate) const STANDARD_GRAIN_RENDER_RADIUS: f32 = STANDARD_RENDER_GRID_DX * 0.62;

#[derive(Clone, Copy, Debug)]
pub(crate) enum RenderObstacle {
    TruncatedCone {
        center: Vec3,
        top_radius: f32,
        bot_radius: f32,
        top_y: f32,
        bot_y: f32,
    },
    Cylinder {
        center: Vec3,
        radius: f32,
        top_y: f32,
        bot_y: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RenderSpout {
    pub(crate) origin: Vec3,
    pub(crate) direction: Vec3,
    pub(crate) stem_length: f32,
    pub(crate) stem_radius: f32,
    pub(crate) nozzle_radius: f32,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct RenderSceneGeometry {
    pub(crate) bounds_size: Vec3,
    pub(crate) render_radius: f32,
    pub(crate) obstacles: Vec<RenderObstacle>,
    pub(crate) spout: RenderSpout,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RenderMaterial {
    pub(crate) grain_radius_scale: f32,
    pub(crate) moisture_inv_cap: f32,
}

impl Default for RenderMaterial {
    fn default() -> Self {
        Self {
            grain_radius_scale: 1.0,
            moisture_inv_cap: 0.0,
        }
    }
}

impl RenderMaterial {
    pub(crate) fn standard_coffee_particles(moisture_inv_cap: f32) -> Self {
        Self {
            grain_radius_scale: STANDARD_GRAIN_RENDER_RADIUS / STANDARD_WATER_RENDER_RADIUS,
            moisture_inv_cap,
        }
    }
}

pub(crate) struct RenderView<'a> {
    bounds_size: Vec3,
    render_radius: f32,
    material: RenderMaterial,
    particle_count: usize,
    particle_source: ParticleRenderSource<'a>,
    filter_render_vertices: Option<&'a [[f32; 3]]>,
    filter_fill_vertices: Option<&'a [[f32; 3]]>,
    static_filter_mesh_key: Option<u64>,
}

pub(crate) enum ParticleRenderSource<'a> {
    Packed {
        render_buffer: &'a wgpu::Buffer,
    },
    Canonical {
        positions: &'a wgpu::Buffer,
        velocities: &'a wgpu::Buffer,
        phases: &'a wgpu::Buffer,
    },
}

impl<'a> RenderView<'a> {
    pub(crate) fn new(
        bounds_size: Vec3,
        render_radius: f32,
        particle_count: usize,
        render_buffer: &'a wgpu::Buffer,
        filter_render_vertices: Option<&'a [[f32; 3]]>,
        filter_fill_vertices: Option<&'a [[f32; 3]]>,
        static_filter_mesh_key: Option<u64>,
    ) -> Self {
        Self {
            bounds_size,
            render_radius,
            material: RenderMaterial::default(),
            particle_count,
            particle_source: ParticleRenderSource::Packed { render_buffer },
            filter_render_vertices,
            filter_fill_vertices,
            static_filter_mesh_key,
        }
    }

    pub(crate) fn new_canonical(
        bounds_size: Vec3,
        render_radius: f32,
        particle_count: usize,
        positions: &'a wgpu::Buffer,
        velocities: &'a wgpu::Buffer,
        phases: &'a wgpu::Buffer,
        material: RenderMaterial,
    ) -> Self {
        Self {
            bounds_size,
            render_radius,
            material,
            particle_count,
            particle_source: ParticleRenderSource::Canonical {
                positions,
                velocities,
                phases,
            },
            filter_render_vertices: None,
            filter_fill_vertices: None,
            static_filter_mesh_key: None,
        }
    }

    pub(crate) fn bounds_size(&self) -> Vec3 {
        self.bounds_size
    }

    pub(crate) fn render_radius(&self) -> f32 {
        self.render_radius
    }

    pub(crate) fn grain_radius_scale(&self) -> f32 {
        self.material.grain_radius_scale
    }

    pub(crate) fn moisture_inv_cap(&self) -> f32 {
        self.material.moisture_inv_cap
    }

    pub(crate) fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub(crate) fn render_buffer(&self) -> &wgpu::Buffer {
        match self.particle_source {
            ParticleRenderSource::Packed { render_buffer } => render_buffer,
            ParticleRenderSource::Canonical { .. } => {
                panic!("canonical render view does not expose a packed render buffer")
            }
        }
    }

    pub(crate) fn particle_source(&self) -> &ParticleRenderSource<'a> {
        &self.particle_source
    }

    pub(crate) fn filter_render_vertices(&self) -> Option<&'a [[f32; 3]]> {
        self.filter_render_vertices
    }

    pub(crate) fn filter_fill_vertices(&self) -> Option<&'a [[f32; 3]]> {
        self.filter_fill_vertices
    }

    pub(crate) fn static_filter_mesh_key(&self) -> Option<u64> {
        self.static_filter_mesh_key
    }
}

use crate::solvers::mpm::MpmSettings;

pub(crate) struct RenderView<'a> {
    settings: &'a MpmSettings,
    particle_count: usize,
    render_buffer: &'a wgpu::Buffer,
    filter_render_vertices: Option<&'a [[f32; 3]]>,
    filter_fill_vertices: Option<&'a [[f32; 3]]>,
    static_filter_mesh_key: Option<u64>,
}

impl<'a> RenderView<'a> {
    pub(crate) fn new(
        settings: &'a MpmSettings,
        particle_count: usize,
        render_buffer: &'a wgpu::Buffer,
        filter_render_vertices: Option<&'a [[f32; 3]]>,
        filter_fill_vertices: Option<&'a [[f32; 3]]>,
        static_filter_mesh_key: Option<u64>,
    ) -> Self {
        Self {
            settings,
            particle_count,
            render_buffer,
            filter_render_vertices,
            filter_fill_vertices,
            static_filter_mesh_key,
        }
    }

    pub(crate) fn settings(&self) -> &MpmSettings {
        self.settings
    }

    pub(crate) fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub(crate) fn render_buffer(&self) -> &wgpu::Buffer {
        self.render_buffer
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

use std::borrow::Cow;

use super::shader::MPM_COMPUTE_SHADER;
use super::state::MpmBuffers;

pub(crate) struct MpmPipelines {
    pub bind_group: wgpu::BindGroup,
    pub clear_grid: wgpu::ComputePipeline,
    pub p2g: wgpu::ComputePipeline,
    pub grid_update: wgpu::ComputePipeline,
    pub classify_cells: wgpu::ComputePipeline,
    pub pressure_rbgs_red: wgpu::ComputePipeline,
    pub pressure_rbgs_black: wgpu::ComputePipeline,
    pub project_pressure: wgpu::ComputePipeline,
    pub boundary_project: wgpu::ComputePipeline,
    pub g2p: wgpu::ComputePipeline,
    pub bed_coupling: wgpu::ComputePipeline,
    pub extraction_advect: wgpu::ComputePipeline,
    pub bed_dynamics: wgpu::ComputePipeline,
    pub prepare_render: wgpu::ComputePipeline,
}

impl MpmPipelines {
    pub fn new(device: &wgpu::Device, buffers: &MpmBuffers) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mpm bind group layout"),
            entries: &[
                // 0: uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: particles
                storage_entry(1),
                // 2: affine
                storage_entry(2),
                // 3: grid atomics
                storage_entry(3),
                // 4: grid_vel
                storage_entry(4),
                // 5: sdf texture
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 6: render_data
                storage_entry(6),
                // 7: bed_extract
                storage_entry(7),
                // 8: bed_lookup
                read_only_storage_entry(8),
                // 9: bed_delta
                storage_entry(9),
            ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mpm bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.affine.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.grid_vel.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&buffers.sdf_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.render_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.bed_extract.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.bed_lookup.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.bed_delta.as_entire_binding(),
                },
            ],
        });

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mpm compute shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MPM_COMPUTE_SHADER)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mpm pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            bind_group,
            clear_grid: make("clear_grid"),
            p2g: make("p2g"),
            grid_update: make("grid_update"),
            classify_cells: make("classify_cells"),
            pressure_rbgs_red: make("pressure_rbgs_red"),
            pressure_rbgs_black: make("pressure_rbgs_black"),
            project_pressure: make("project_pressure"),
            boundary_project: make("boundary_project"),
            g2p: make("g2p"),
            bed_coupling: make("bed_coupling"),
            extraction_advect: make("extraction_advect"),
            bed_dynamics: make("bed_dynamics"),
            prepare_render: make("prepare_render"),
        }
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn read_only_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

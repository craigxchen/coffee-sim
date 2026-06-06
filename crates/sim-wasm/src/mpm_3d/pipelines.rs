use std::borrow::Cow;

use super::shader::MPM_COMPUTE_SHADER;
use super::state::MpmBuffers;
use super::PressureTier;

pub(crate) struct MpmPipelines {
    pub bind_group: wgpu::BindGroup,
    /// group(1) holding the active-tile list (Indirect tier only). Bound during
    /// compaction and the indirect RBGS solve. `None` on the Sparse tier.
    pub active_tile_bind_group: Option<wgpu::BindGroup>,
    /// group(2) holding the indirect-args buffer (Indirect tier only). Bound
    /// ONLY during the compaction pass — never during the solve (KTD3). `None`
    /// on the Sparse tier.
    pub indirect_args_bind_group: Option<wgpu::BindGroup>,
    pub metrics_clear: wgpu::ComputePipeline,
    pub sparse_tiles_clear: wgpu::ComputePipeline,
    pub bed_lookup_clear: wgpu::ComputePipeline,
    pub bed_lookup_scatter: wgpu::ComputePipeline,
    pub p2g: wgpu::ComputePipeline,
    pub grid_update: wgpu::ComputePipeline,
    pub viscosity_prepare: wgpu::ComputePipeline,
    pub viscosity_apply: wgpu::ComputePipeline,
    pub classify_cells: wgpu::ComputePipeline,
    pub pressure_rbgs_red: wgpu::ComputePipeline,
    pub pressure_rbgs_black: wgpu::ComputePipeline,
    pub pressure_rbgs_red_sparse: wgpu::ComputePipeline,
    pub pressure_rbgs_black_sparse: wgpu::ComputePipeline,
    /// Indirect-tier pipelines (`None` on the Sparse tier). `indirect_args_init`
    /// and `sparse_compaction` use the group(0)+group(1)+group(2) layout; the
    /// red/black indirect sweeps use the group(0)+group(1) layout.
    pub indirect_args_init: Option<wgpu::ComputePipeline>,
    pub sparse_compaction: Option<wgpu::ComputePipeline>,
    pub pressure_rbgs_red_indirect: Option<wgpu::ComputePipeline>,
    pub pressure_rbgs_black_indirect: Option<wgpu::ComputePipeline>,
    pub project_pressure: wgpu::ComputePipeline,
    pub pressure_residual: wgpu::ComputePipeline,
    pub boundary_project: wgpu::ComputePipeline,
    pub packing_prepare: wgpu::ComputePipeline,
    pub packing_apply: wgpu::ComputePipeline,
    pub g2p: wgpu::ComputePipeline,
    pub bed_coupling: wgpu::ComputePipeline,
    pub extraction_advect: wgpu::ComputePipeline,
    pub bed_dynamics: wgpu::ComputePipeline,
    pub prepare_render: wgpu::ComputePipeline,
}

impl MpmPipelines {
    pub fn new(device: &wgpu::Device, buffers: &MpmBuffers, tier: PressureTier) -> Self {
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
                // 8: bed_lookup (rebuilt each substep via atomic scatter)
                storage_entry(8),
                // 9: bed_delta
                storage_entry(9),
                // 10: metrics (projection residual / clamp counters).
                storage_entry(10),
                // 11: cached cell-solid classification for classify_cells
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 12: sparse-pressure tile metadata (flags + active count)
                storage_entry(12),
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
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.metrics.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&buffers.sdf_class_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.sparse_tiles.as_entire_binding(),
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

        // Indirect tier only: build the second/third bind groups, their layouts,
        // and the compaction + indirect RBGS pipelines. On the Sparse tier these
        // are `None`, the group(1)/group(2) layouts are never created, and every
        // pipeline keeps the single-group(0) layout — byte-for-byte v1.
        let (
            active_tile_bind_group,
            indirect_args_bind_group,
            indirect_args_init,
            sparse_compaction,
            pressure_rbgs_red_indirect,
            pressure_rbgs_black_indirect,
        ) = if tier == PressureTier::Indirect {
            let active_tile_list = buffers
                .active_tile_list
                .as_ref()
                .expect("active_tile_list allocated on the Indirect tier");
            let indirect_args = buffers
                .indirect_args
                .as_ref()
                .expect("indirect_args allocated on the Indirect tier");

            // group(1): active-tile list (read_write storage).
            let active_tile_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mpm active tile bind group layout"),
                    entries: &[storage_entry(0)],
                });
            let active_tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mpm active tile bind group"),
                layout: &active_tile_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: active_tile_list.as_entire_binding(),
                }],
            });

            // group(2): indirect args (read_write storage), bound only by
            // compaction.
            let indirect_args_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mpm indirect args bind group layout"),
                    entries: &[storage_entry(0)],
                });
            let indirect_args_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mpm indirect args bind group"),
                layout: &indirect_args_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: indirect_args.as_entire_binding(),
                }],
            });

            // Compaction reads flags via group(0), writes the list via group(1),
            // and writes the args via group(2).
            let compaction_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("mpm compaction pipeline layout"),
                    bind_group_layouts: &[
                        Some(&bind_group_layout),
                        Some(&active_tile_layout),
                        Some(&indirect_args_layout),
                    ],
                    immediate_size: 0,
                });
            // Indirect RBGS reads the list via group(1); indirect_args is the
            // dispatch source, never bound here.
            let indirect_solve_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("mpm indirect solve pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout), Some(&active_tile_layout)],
                    immediate_size: 0,
                });

            let make_with = |entry: &str, layout: &wgpu::PipelineLayout| -> wgpu::ComputePipeline {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(layout),
                    module: &shader_module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };

            (
                Some(active_tile_bind_group),
                Some(indirect_args_bind_group),
                Some(make_with("indirect_args_init", &compaction_layout)),
                Some(make_with("sparse_compaction", &compaction_layout)),
                Some(make_with(
                    "pressure_rbgs_red_indirect",
                    &indirect_solve_layout,
                )),
                Some(make_with(
                    "pressure_rbgs_black_indirect",
                    &indirect_solve_layout,
                )),
            )
        } else {
            (None, None, None, None, None, None)
        };

        Self {
            bind_group,
            active_tile_bind_group,
            indirect_args_bind_group,
            metrics_clear: make("metrics_clear"),
            sparse_tiles_clear: make("sparse_tiles_clear"),
            bed_lookup_clear: make("bed_lookup_clear"),
            bed_lookup_scatter: make("bed_lookup_scatter"),
            p2g: make("p2g"),
            grid_update: make("grid_update"),
            viscosity_prepare: make("viscosity_prepare"),
            viscosity_apply: make("viscosity_apply"),
            classify_cells: make("classify_cells"),
            pressure_rbgs_red: make("pressure_rbgs_red"),
            pressure_rbgs_black: make("pressure_rbgs_black"),
            pressure_rbgs_red_sparse: make("pressure_rbgs_red_sparse"),
            pressure_rbgs_black_sparse: make("pressure_rbgs_black_sparse"),
            indirect_args_init,
            sparse_compaction,
            pressure_rbgs_red_indirect,
            pressure_rbgs_black_indirect,
            project_pressure: make("project_pressure"),
            pressure_residual: make("pressure_residual"),
            boundary_project: make("boundary_project"),
            packing_prepare: make("packing_prepare"),
            packing_apply: make("packing_apply"),
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

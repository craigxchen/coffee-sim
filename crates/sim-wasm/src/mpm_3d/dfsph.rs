//! DFSPH backend resources.
//!
//! This module is the landing zone for the DFSPH water solver work that is
//! being ported into the modular profiler framework. DFSPH is a backend-level
//! solver, not an MPM pressure-solver variant: it owns different neighbor-hash
//! resources and will eventually provide its own schedule for the shared
//! profiler runner.
//!
//! Current compatibility contract from `codex/dfsph-water`:
//! - binding 0: shared MPM/DFSPH uniforms
//! - binding 3: DFSPH water neighbor hash
//! - binding 13: active pressure-tile indirect metadata

use super::state::MpmBuffers;

#[allow(dead_code)]
pub(crate) struct DfsphPipelines {
    pub active_tile_bind_group: wgpu::BindGroup,
}

impl DfsphPipelines {
    pub(crate) fn new(device: &wgpu::Device, buffers: &MpmBuffers) -> Self {
        let active_tile_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dfsph active pressure tile bind group layout"),
                entries: &[
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
                    storage_entry(3),
                    storage_entry(13),
                ],
            });

        let active_tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dfsph active pressure tile bind group"),
            layout: &active_tile_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.water_hash.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.pressure_indirect.as_entire_binding(),
                },
            ],
        });

        Self {
            active_tile_bind_group,
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

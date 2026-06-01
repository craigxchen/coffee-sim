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
    pub bind_group: wgpu::BindGroup,
    pub water_hash_clear: wgpu::ComputePipeline,
    pub water_hash_scatter: wgpu::ComputePipeline,
    pub density_factor: wgpu::ComputePipeline,
    pub active_tile_bind_group: wgpu::BindGroup,
}

impl DfsphPipelines {
    pub(crate) fn new(device: &wgpu::Device, buffers: &MpmBuffers) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dfsph water bind group layout"),
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
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dfsph water bind group"),
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
                    resource: buffers.water_hash.as_entire_binding(),
                },
            ],
        });

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

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dfsph water shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(DFSPH_WATER_SHADER)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dfsph water pipeline layout"),
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
            water_hash_clear: make("water_hash_clear"),
            water_hash_scatter: make("water_hash_scatter"),
            density_factor: make("dfsph_density_factor"),
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

const DFSPH_WATER_SHADER: &str = r#"
struct MpmUniforms {
    grid_dims: vec4<u32>,
    counts: vec4<u32>,
    sim_params: vec4<f32>,
    grid_origin: vec4<f32>,
    bounds_max: vec4<f32>,
    fluid_params: vec4<f32>,
    fp_params: vec4<f32>,
    inflow_origin: vec4<f32>,
    inflow_dir: vec4<f32>,
    inflow_params: vec4<f32>,
    sdf_params: vec4<f32>,
    bed_params: vec4<f32>,
    extraction_params: vec4<f32>,
    solute_params: vec4<f32>,
    time_params: vec4<f32>,
    clamp_params: vec4<f32>,
    projection_params: vec4<f32>,
};

struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

struct AffineC {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: MpmUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> affine: array<AffineC>;
@group(0) @binding(3) var<storage, read_write> water_hash: array<atomic<i32>>;

const DFSPH_MAX_NEIGHBORS_PER_CELL: u32 = 4096u;
const DFSPH_EPS: f32 = 1.0e-6;

fn gx() -> u32 { return u.grid_dims.x; }
fn gy() -> u32 { return u.grid_dims.y; }
fn gz() -> u32 { return u.grid_dims.z; }
fn total_cells() -> u32 { return u.grid_dims.w; }
fn num_water() -> u32 { return u.counts.x; }
fn num_bed() -> u32 { return u.counts.y; }
fn max_particles() -> u32 { return u.counts.z; }
fn num_particles() -> u32 { return u.counts.x + u.counts.y; }
fn dx() -> f32 { return u.sim_params.z; }
fn inv_dx() -> f32 { return u.sim_params.w; }
fn p_vol() -> f32 { return u.fluid_params.w; }

fn water_hash_slots() -> u32 {
    return total_cells() + max_particles();
}

fn water_hash_head_idx(cell: u32) -> u32 {
    return cell;
}

fn water_hash_next_idx(pid: u32) -> u32 {
    return total_cells() + pid;
}

fn water_particle_id(local_water_id: u32) -> u32 {
    return num_bed() + local_water_id;
}

fn is_water_phase(phase: f32) -> bool {
    return abs(phase) < 0.5;
}

fn inactive_mass_threshold() -> f32 {
    return 1.0e-8;
}

fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return ix + gx() * (iy + gy() * iz);
}

fn world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor((pos - u.grid_origin.xyz) * inv_dx()));
}

fn dfsph_support_radius() -> f32 {
    return max(dx() * 2.0, 1.0e-6);
}

fn dfsph_particle_volume(pid: u32) -> f32 {
    return p_vol() * max(particles[pid].vel.w, 0.0);
}

fn cubic_kernel(r: f32) -> f32 {
    let h = dfsph_support_radius();
    let q = clamp(r / h, 0.0, 1.0);
    let a = 1.0 - q;
    return a * a * a / max(h * h * h, DFSPH_EPS);
}

fn cubic_kernel_grad(r: vec3<f32>) -> vec3<f32> {
    let len = length(r);
    if len <= DFSPH_EPS {
        return vec3<f32>(0.0);
    }
    let h = dfsph_support_radius();
    let q = clamp(len / h, 0.0, 1.0);
    let a = 1.0 - q;
    let d = -3.0 * a * a / max(h * h * h * h, DFSPH_EPS);
    return d * r / len;
}

@compute @workgroup_size(64)
fn water_hash_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= water_hash_slots() { return; }
    atomicStore(&water_hash[idx], -1);
}

@compute @workgroup_size(64)
fn water_hash_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if pid >= num_particles() { return; }
    if !is_water_phase(affine[pid].col0.w) || particles[pid].vel.w <= inactive_mass_threshold() {
        return;
    }

    let cell = world_to_cell(particles[pid].pos.xyz);
    if cell.x < 0 || cell.y < 0 || cell.z < 0 { return; }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { return; }

    let head_idx = water_hash_head_idx(cell_index(u32(cell.x), u32(cell.y), u32(cell.z)));
    let old_head = atomicExchange(&water_hash[head_idx], i32(pid));
    atomicStore(&water_hash[water_hash_next_idx(pid)], old_head);
}

@compute @workgroup_size(64)
fn dfsph_density_factor(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if pid >= num_particles() { return; }
    if !is_water_phase(affine[pid].col0.w) || particles[pid].vel.w <= inactive_mass_threshold() {
        return;
    }

    let xp = particles[pid].pos.xyz;
    let home = world_to_cell(xp);
    var density = dfsph_particle_volume(pid) * cubic_kernel(0.0);
    var grad_i = vec3<f32>(0.0);
    var sum_grad = 0.0;

    for (var oz = -1i; oz <= 1i; oz = oz + 1i) {
        for (var oy = -1i; oy <= 1i; oy = oy + 1i) {
            for (var ox = -1i; ox <= 1i; ox = ox + 1i) {
                let cell = home + vec3<i32>(ox, oy, oz);
                if cell.x < 0 || cell.y < 0 || cell.z < 0 { continue; }
                if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { continue; }
                let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
                var neighbor_pid = atomicLoad(&water_hash[water_hash_head_idx(ci)]);
                var guard = 0u;
                loop {
                    if neighbor_pid < 0 || guard >= DFSPH_MAX_NEIGHBORS_PER_CELL { break; }
                    let j = u32(neighbor_pid);
                    if j != pid && j < num_particles()
                        && is_water_phase(affine[j].col0.w)
                        && particles[j].vel.w > inactive_mass_threshold() {
                        let r = xp - particles[j].pos.xyz;
                        let r_len = length(r);
                        if r_len <= dfsph_support_radius() {
                            let volume_j = dfsph_particle_volume(j);
                            density += volume_j * cubic_kernel(r_len);
                            let grad_j = -volume_j * cubic_kernel_grad(r);
                            grad_i -= grad_j;
                            sum_grad += dot(grad_j, grad_j);
                        }
                    }
                    neighbor_pid = atomicLoad(&water_hash[water_hash_next_idx(j)]);
                    guard = guard + 1u;
                }
            }
        }
    }

    sum_grad += dot(grad_i, grad_i);
    var factor = 0.0;
    if sum_grad > DFSPH_EPS {
        factor = -1.0 / sum_grad;
    }
    affine[pid].col1.w = density;
    affine[pid].col2.w = factor;
}
"#;

#[cfg(test)]
mod tests {
    use super::DFSPH_WATER_SHADER;

    #[test]
    fn dfsph_water_shader_parses() {
        let module = naga::front::wgsl::parse_str(DFSPH_WATER_SHADER)
            .expect("DFSPH water WGSL should parse");
        let entry_names: Vec<_> = module
            .entry_points
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert!(entry_names.contains(&"water_hash_clear"));
        assert!(entry_names.contains(&"water_hash_scatter"));
        assert!(entry_names.contains(&"dfsph_density_factor"));
    }
}

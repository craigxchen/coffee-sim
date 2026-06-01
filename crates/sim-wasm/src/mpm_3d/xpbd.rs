//! GPU XPBD backend scaffolding.
//!
//! The `xpbd-solver-rewrite` branch defines an XPBD backend shape, but its
//! checked-in WGSL entry points are stubs. This module starts the GPU port in
//! the modular profiler branch with real compute kernels over the existing MPM
//! particle buffers, and is exposed to the profiler as the `xpbd:gpu` backend.

use super::state::MpmBuffers;

pub(crate) struct XpbdPipelines {
    pub bind_group: wgpu::BindGroup,
    pub hash_clear: wgpu::ComputePipeline,
    pub hash_scatter: wgpu::ComputePipeline,
    pub predict: wgpu::ComputePipeline,
    pub solve_density: wgpu::ComputePipeline,
    pub apply_density: wgpu::ComputePipeline,
    pub solve_bounds: wgpu::ComputePipeline,
    pub velocity_update: wgpu::ComputePipeline,
}

impl XpbdPipelines {
    pub(crate) fn new(device: &wgpu::Device, buffers: &MpmBuffers) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xpbd gpu bind group layout"),
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
            label: Some("xpbd gpu bind group"),
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

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xpbd gpu shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(XPBD_GPU_SHADER)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xpbd gpu pipeline layout"),
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
            hash_clear: make("xpbd_hash_clear"),
            hash_scatter: make("xpbd_hash_scatter"),
            predict: make("xpbd_predict"),
            solve_density: make("xpbd_solve_density"),
            apply_density: make("xpbd_apply_density"),
            solve_bounds: make("xpbd_solve_bounds"),
            velocity_update: make("xpbd_velocity_update"),
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

const XPBD_GPU_SHADER: &str = r#"
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
@group(0) @binding(3) var<storage, read_write> hash: array<atomic<i32>>;

const XPBD_MAX_NEIGHBORS_PER_CELL: u32 = 128u;
const XPBD_EPS: f32 = 1.0e-6;

fn gx() -> u32 { return u.grid_dims.x; }
fn gy() -> u32 { return u.grid_dims.y; }
fn gz() -> u32 { return u.grid_dims.z; }
fn total_cells() -> u32 { return u.grid_dims.w; }
fn num_water() -> u32 { return u.counts.x; }
fn num_bed() -> u32 { return u.counts.y; }
fn max_particles() -> u32 { return u.counts.z; }
fn num_particles() -> u32 { return u.counts.x + u.counts.y; }
fn dt() -> f32 { return u.sim_params.x; }
fn gravity() -> f32 { return u.sim_params.y; }
fn inv_dx() -> f32 { return u.sim_params.w; }
fn particle_radius() -> f32 { return max(u.inflow_params.z, u.sim_params.z * 0.35); }
fn vel_cap() -> f32 { return u.fp_params.z; }

fn water_particle_id(local_water_id: u32) -> u32 {
    return num_bed() + local_water_id;
}

fn is_water_phase(phase: f32) -> bool {
    return abs(phase) < 0.5;
}

fn active_water(pid: u32) -> bool {
    return pid < num_particles()
        && is_water_phase(affine[pid].col0.w)
        && particles[pid].vel.w > 1.0e-8;
}

fn hash_slots() -> u32 {
    return total_cells() + max_particles();
}

fn hash_head_idx(cell: u32) -> u32 {
    return cell;
}

fn hash_next_idx(pid: u32) -> u32 {
    return total_cells() + pid;
}

fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return ix + gx() * (iy + gy() * iz);
}

fn world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor((pos - u.grid_origin.xyz) * inv_dx()));
}

@compute @workgroup_size(64)
fn xpbd_hash_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= hash_slots() { return; }
    atomicStore(&hash[idx], -1);
}

@compute @workgroup_size(64)
fn xpbd_hash_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if !active_water(pid) { return; }
    let cell = world_to_cell(particles[pid].pos.xyz);
    if cell.x < 0 || cell.y < 0 || cell.z < 0 { return; }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { return; }
    let head = hash_head_idx(cell_index(u32(cell.x), u32(cell.y), u32(cell.z)));
    let old = atomicExchange(&hash[head], i32(pid));
    atomicStore(&hash[hash_next_idx(pid)], old);
}

@compute @workgroup_size(64)
fn xpbd_predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if !active_water(pid) { return; }
    let old_pos = particles[pid].pos.xyz;
    var vel = particles[pid].vel.xyz + vec3<f32>(0.0, gravity() * dt(), 0.0);
    let speed = length(vel);
    if speed > vel_cap() {
        vel *= vel_cap() / speed;
    }
    affine[pid].col2 = vec4<f32>(old_pos, affine[pid].col2.w);
    particles[pid].pos = vec4<f32>(old_pos + vel * dt(), particles[pid].pos.w);
    particles[pid].vel = vec4<f32>(vel, particles[pid].vel.w);
}

@compute @workgroup_size(64)
fn xpbd_solve_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if !active_water(pid) { return; }

    let pos = particles[pid].pos.xyz;
    let home = world_to_cell(pos);
    let min_dist = particle_radius() * 1.55;
    var correction = vec3<f32>(0.0);
    var count = 0.0;

    for (var oz = -1i; oz <= 1i; oz = oz + 1i) {
        for (var oy = -1i; oy <= 1i; oy = oy + 1i) {
            for (var ox = -1i; ox <= 1i; ox = ox + 1i) {
                let cell = home + vec3<i32>(ox, oy, oz);
                if cell.x < 0 || cell.y < 0 || cell.z < 0 { continue; }
                if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { continue; }
                let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
                var neighbor_pid = atomicLoad(&hash[hash_head_idx(ci)]);
                var guard = 0u;
                loop {
                    if neighbor_pid < 0 || guard >= XPBD_MAX_NEIGHBORS_PER_CELL { break; }
                    let j = u32(neighbor_pid);
                    if j != pid && active_water(j) {
                        let delta = pos - particles[j].pos.xyz;
                        let dist = length(delta);
                        if dist > XPBD_EPS && dist < min_dist {
                            let n = delta / dist;
                            correction += n * (min_dist - dist) * 0.35;
                            count += 1.0;
                        }
                    }
                    neighbor_pid = atomicLoad(&hash[hash_next_idx(j)]);
                    guard = guard + 1u;
                }
            }
        }
    }

    if count > 0.0 {
        affine[pid].col1 = vec4<f32>(correction / count, affine[pid].col1.w);
    } else {
        affine[pid].col1 = vec4<f32>(0.0, 0.0, 0.0, affine[pid].col1.w);
    }
}

@compute @workgroup_size(64)
fn xpbd_apply_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if !active_water(pid) { return; }
    let correction = affine[pid].col1.xyz;
    particles[pid].pos = vec4<f32>(particles[pid].pos.xyz + correction, particles[pid].pos.w);
    affine[pid].col1 = vec4<f32>(0.0, 0.0, 0.0, affine[pid].col1.w);
}

@compute @workgroup_size(64)
fn xpbd_solve_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if !active_water(pid) { return; }
    let radius = particle_radius();
    let lo = u.grid_origin.xyz + vec3<f32>(radius);
    let hi = u.bounds_max.xyz - vec3<f32>(radius);
    let pos = clamp(particles[pid].pos.xyz, lo, hi);
    particles[pid].pos = vec4<f32>(pos, particles[pid].pos.w);
}

@compute @workgroup_size(64)
fn xpbd_velocity_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= num_water() { return; }
    let pid = water_particle_id(gid.x);
    if !active_water(pid) { return; }
    let old_pos = affine[pid].col2.xyz;
    var vel = (particles[pid].pos.xyz - old_pos) / max(dt(), 1.0e-6);
    let speed = length(vel);
    if speed > vel_cap() {
        vel *= vel_cap() / speed;
    }
    particles[pid].vel = vec4<f32>(vel, particles[pid].vel.w);
}
"#;

#[cfg(test)]
mod tests {
    use super::XPBD_GPU_SHADER;

    #[test]
    fn xpbd_gpu_shader_parses() {
        let module =
            naga::front::wgsl::parse_str(XPBD_GPU_SHADER).expect("XPBD GPU WGSL should parse");
        let entry_names: Vec<_> = module
            .entry_points
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert!(entry_names.contains(&"xpbd_hash_clear"));
        assert!(entry_names.contains(&"xpbd_hash_scatter"));
        assert!(entry_names.contains(&"xpbd_predict"));
        assert!(entry_names.contains(&"xpbd_solve_density"));
        assert!(entry_names.contains(&"xpbd_apply_density"));
        assert!(entry_names.contains(&"xpbd_solve_bounds"));
        assert!(entry_names.contains(&"xpbd_velocity_update"));
    }
}

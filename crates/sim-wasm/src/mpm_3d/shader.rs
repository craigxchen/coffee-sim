pub(crate) const MPM_COMPUTE_SHADER: &str = r#"

// ── Structs ──

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
    time_params: vec4<f32>,
    clamp_params: vec4<f32>,
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

struct BedExtract {
    bed: vec4<f32>,
    extract: vec4<f32>,
};

struct ContactResult {
    pos: vec3<f32>,
    vel: vec3<f32>,
};

// ── Bindings ──

@group(0) @binding(0) var<uniform> u: MpmUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> affine: array<AffineC>;
@group(0) @binding(3) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> grid_vel: array<vec4<f32>>;
@group(0) @binding(5) var sdf_texture: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> render_data: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> bed_extract: array<BedExtract>;
@group(0) @binding(8) var<storage, read_write> bed_lookup: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> bed_delta: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> metrics: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> filter_mesh: array<vec4<f32>>;
@group(0) @binding(12) var sdf_class_tex: texture_3d<u32>;

const FILTER_RING_COUNT = 10u;
const FILTER_SEGMENT_COUNT = 32u;

// Metrics slot layout — keep in sync with `METRICS_SLOT_COUNT` in state.rs.
const METRIC_MAX_ABS_DIV_IDX: u32 = 0u;
const METRIC_FLUID_CELLS_IDX: u32 = 1u;
const METRIC_DIV_CLAMP_FIRES_IDX: u32 = 2u;
const METRIC_PRESSURE_CLAMP_FIRES_IDX: u32 = 3u;
const METRIC_MASS_OVERFLOW_FIRES_IDX: u32 = 4u;
const BED_NEIGHBOR_SAMPLE_COUNT: u32 = 27u;

// ── Helpers ──

fn gx() -> u32 { return u.grid_dims.x; }
fn gy() -> u32 { return u.grid_dims.y; }
fn gz() -> u32 { return u.grid_dims.z; }
fn total_cells() -> u32 { return u.grid_dims.w; }
fn num_bed() -> u32 { return u.counts.y; }
fn num_particles() -> u32 { return u.counts.x + u.counts.y; }
fn use_sdf_cache() -> bool { return u.counts.w > 0u; }
fn dt() -> f32 { return u.sim_params.x; }
fn gravity() -> f32 { return u.sim_params.y; }
fn dx() -> f32 { return u.sim_params.z; }
fn inv_dx() -> f32 { return u.sim_params.w; }
fn bulk_K() -> f32 { return u.fluid_params.x; }
fn viscosity() -> f32 { return u.fluid_params.y; }
fn nominal_mass() -> f32 { return u.fluid_params.z; }
fn p_vol() -> f32 { return u.fluid_params.w; }
fn fp_scale() -> f32 { return u.fp_params.x; }
fn inv_fp_scale() -> f32 { return u.fp_params.y; }
fn vel_cap() -> f32 { return u.fp_params.z; }
fn sdf_res() -> f32 { return u.sdf_params.x; }
fn friction() -> f32 { return u.sdf_params.y; }
fn restitution() -> f32 { return u.sdf_params.z; }
fn contact_offset() -> f32 { return u.sdf_params.w; }
fn uniform_porosity() -> f32 { return u.bed_params.x; }
fn absorption_rate() -> f32 { return u.bed_params.y; }
fn max_saturation() -> f32 { return u.bed_params.z; }
fn has_filter() -> bool { return u.bed_params.w > 0.5; }
fn extraction_rate() -> f32 { return u.extraction_params.x; }
fn K_bed() -> f32 { return u.extraction_params.y; }
fn mu_fluid() -> f32 { return u.extraction_params.z; }
fn uniform_permeability() -> f32 { return u.extraction_params.w; }
fn bed_storage_redistribution_rate() -> f32 { return 0.18; }
fn inactive_mass_threshold() -> f32 { return nominal_mass() * 0.10; }
fn div_clamp_limit() -> f32 { return u.clamp_params.x; }
fn pressure_clamp_limit() -> f32 { return u.clamp_params.y; }
fn metrics_div_fp_scale() -> f32 { return u.clamp_params.z; }
fn metrics_div_inv_fp_scale() -> f32 { return u.clamp_params.w; }
fn bed_compaction_response(saturation: f32) -> f32 {
    return smoothstep(0.05, 0.55, saturation);
}
fn determinant_from_cols(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> f32 {
    return dot(c0, cross(c1, c2));
}
fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return iz * gx() * gy() + iy * gx() + ix;
}

fn grid_mass_idx(cell: u32) -> u32 { return cell; }
fn grid_mom_x_idx(cell: u32) -> u32 { return total_cells() + cell; }
fn grid_mom_y_idx(cell: u32) -> u32 { return 2u * total_cells() + cell; }
fn grid_mom_z_idx(cell: u32) -> u32 { return 3u * total_cells() + cell; }
fn grid_solid_mass_idx(cell: u32) -> u32 { return 4u * total_cells() + cell; }
fn grid_solid_mom_x_idx(cell: u32) -> u32 { return 5u * total_cells() + cell; }
fn grid_solid_mom_y_idx(cell: u32) -> u32 { return 6u * total_cells() + cell; }
fn grid_solid_mom_z_idx(cell: u32) -> u32 { return 7u * total_cells() + cell; }
fn grid_vel_solid_idx(cell: u32) -> u32 { return total_cells() + cell; }
fn scratch_pressure_idx(cell: u32) -> u32 { return grid_mass_idx(cell); }
fn scratch_div_idx(cell: u32) -> u32 { return grid_mom_x_idx(cell); }
// Slot 2 (`grid_mom_y_idx`) is reused only as the mass/momentum accumulator
// during `p2g`; projection no longer keeps a per-cell residual so no scratch
// alias is defined for it. Add one back if/when an iterative residual probe
// needs the slot during the projection pass.
fn scratch_kind_idx(cell: u32) -> u32 { return grid_mom_z_idx(cell); }
fn scratch_absorbed_idx(cell: u32) -> u32 { return grid_mom_y_idx(cell); }
// A quadratic-B-spline particle deposits at most `nominal_mass * 0.75^3 ≈
// 0.42 * nominal_mass` to its peak cell. The threshold must stay strictly
// below that peak or isolated particles never register as fluid. Matching
// `inactive_mass_threshold()` at 0.1 * nominal_mass means "enough mass to
// still exist" ⇔ "enough mass to produce a fluid cell", which is
// semantically consistent and keeps ghost-splat noise below the bar.
fn occupancy_mass_threshold() -> f32 { return nominal_mass() * 0.1; }

const CELL_AIR: i32 = 0;
const CELL_SURFACE_FLUID: i32 = 1;
const CELL_INTERIOR_FLUID: i32 = 2;
const CELL_BED_COUPLED: i32 = 3;
const CELL_SOLID: i32 = 4;

fn cell_kind_load(cell: u32) -> i32 {
    return atomicLoad(&grid[scratch_kind_idx(cell)]);
}

fn pressure_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[scratch_pressure_idx(cell)])) * inv_fp_scale();
}

fn pressure_store(cell: u32, value: f32) {
    let limit = pressure_clamp_limit();
    let clamped = clamp(value, -limit, limit);
    if clamped != value {
        atomicAdd(&metrics[METRIC_PRESSURE_CLAMP_FIRES_IDX], 1u);
    }
    atomicStore(&grid[scratch_pressure_idx(cell)], i32(clamped * fp_scale()));
}

fn divergence_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[scratch_div_idx(cell)])) * inv_fp_scale();
}

fn divergence_store(cell: u32, value: f32) {
    let limit = div_clamp_limit();
    let clamped = clamp(value, -limit, limit);
    if clamped != value {
        atomicAdd(&metrics[METRIC_DIV_CLAMP_FIRES_IDX], 1u);
    }
    atomicStore(&grid[scratch_div_idx(cell)], i32(clamped * fp_scale()));
}

fn bed_lookup_load(cell: u32) -> i32 {
    return atomicLoad(&bed_lookup[cell]);
}

fn sdf_class_is_solid(cell: vec3<i32>) -> bool {
    return textureLoad(sdf_class_tex, cell, 0).r != 0u;
}

fn is_fluid_kind(kind: i32) -> bool {
    // Surface cells must participate in the pressure solve so hydrostatic
    // pressure can build up in shallow puddles. Adjacent CELL_AIR cells
    // provide the Dirichlet p=0 BC via the neighbor-loop sum in
    // `pressure_update` (air neighbors count toward the denominator but
    // contribute 0 to the numerator), so excluding surface cells here would
    // zero them out and let gravity compress thin pools to a single layer.
    //
    // BED_COUPLED cells are excluded: the bed skeleton carries its own solid
    // velocity field, and water/solid coupling happens through Darcy drag
    // instead of treating the bed region as incompressible fluid.
    return kind == CELL_INTERIOR_FLUID
        || kind == CELL_SURFACE_FLUID;
}

fn is_solid_kind(kind: i32) -> bool {
    return kind == CELL_SOLID;
}

fn world_to_cell(position: vec3<f32>) -> vec3<i32> {
    let grid_pos = (position - u.grid_origin.xyz) * inv_dx();
    return vec3<i32>(floor(grid_pos));
}

fn load_sdf_texel(c: vec3<i32>) -> f32 {
    let mx = vec3<i32>(i32(sdf_res()) - 1);
    return textureLoad(sdf_texture, clamp(c, vec3<i32>(0), mx), 0).r;
}

fn sample_sdf(position: vec3<f32>) -> f32 {
    let bounds_size = u.bounds_max.xyz * 2.0;
    let res = sdf_res();
    let uv = (position + u.bounds_max.xyz) / bounds_size * res - vec3<f32>(0.5);
    let base = vec3<i32>(floor(uv));
    let f = fract(uv);
    let c000 = load_sdf_texel(base);
    let c100 = load_sdf_texel(base + vec3<i32>(1, 0, 0));
    let c010 = load_sdf_texel(base + vec3<i32>(0, 1, 0));
    let c110 = load_sdf_texel(base + vec3<i32>(1, 1, 0));
    let c001 = load_sdf_texel(base + vec3<i32>(0, 0, 1));
    let c101 = load_sdf_texel(base + vec3<i32>(1, 0, 1));
    let c011 = load_sdf_texel(base + vec3<i32>(0, 1, 1));
    let c111 = load_sdf_texel(base + vec3<i32>(1, 1, 1));
    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    let c0 = mix(c00, c10, f.y);
    let c1 = mix(c01, c11, f.y);
    return mix(c0, c1, f.z);
}

fn sdf_gradient(position: vec3<f32>) -> vec3<f32> {
    let eps = dx();
    let gx_val = sample_sdf(position + vec3<f32>(eps, 0.0, 0.0))
               - sample_sdf(position - vec3<f32>(eps, 0.0, 0.0));
    let gy_val = sample_sdf(position + vec3<f32>(0.0, eps, 0.0))
               - sample_sdf(position - vec3<f32>(0.0, eps, 0.0));
    let gz_val = sample_sdf(position + vec3<f32>(0.0, 0.0, eps))
               - sample_sdf(position - vec3<f32>(0.0, 0.0, eps));
    let g = vec3<f32>(gx_val, gy_val, gz_val);
    let len = length(g);
    if len < 1e-8 {
        return vec3<f32>(0.0);
    }
    return g / len;
}

fn resolve_radial_barrier(
    position: vec3<f32>,
    velocity: vec3<f32>,
    center: vec2<f32>,
    max_radius: f32,
) -> ContactResult {
    var out_pos = position;
    var out_vel = velocity;

    let radial = out_pos.xz - center;
    let r = length(radial);
    if r > max_radius && r > 1e-6 {
        let outward = radial / r;
        out_pos.x = center.x + outward.x * max_radius;
        out_pos.z = center.y + outward.y * max_radius;

        let vn = dot(out_vel.xz, outward);
        if vn > 0.0 {
            let tangential = out_vel.xz - outward * vn;
            out_vel.x = tangential.x * (1.0 - friction() * 0.35);
            out_vel.z = tangential.y * (1.0 - friction() * 0.35);
        }
    }

    return ContactResult(out_pos, out_vel);
}

fn resolve_filter_mesh_contact(pos: vec3<f32>, vel: vec3<f32>) -> ContactResult {
    let mesh_bot_y = filter_mesh[0].y;
    let mesh_top_y = filter_mesh[(FILTER_RING_COUNT - 1u) * FILTER_SEGMENT_COUNT].y;

    if pos.y > mesh_top_y || pos.y < mesh_bot_y - 0.5 {
        return ContactResult(pos, vel);
    }

    let ring_t = clamp((pos.y - mesh_bot_y) / (mesh_top_y - mesh_bot_y), 0.0, 1.0);
    let ring_f = ring_t * f32(FILTER_RING_COUNT - 1u);
    let ring_lo = u32(floor(ring_f));
    let ring_hi = min(ring_lo + 1u, FILTER_RING_COUNT - 1u);
    let ring_frac = ring_f - floor(ring_f);

    let angle = atan2(pos.z, pos.x);
    let seg_f = ((angle / (2.0 * 3.14159265) + 1.0) % 1.0) * f32(FILTER_SEGMENT_COUNT);
    let seg_lo = u32(floor(seg_f)) % FILTER_SEGMENT_COUNT;
    let seg_hi = (seg_lo + 1u) % FILTER_SEGMENT_COUNT;
    let seg_frac = seg_f - floor(seg_f);

    let v00 = filter_mesh[ring_lo * FILTER_SEGMENT_COUNT + seg_lo].xyz;
    let v01 = filter_mesh[ring_lo * FILTER_SEGMENT_COUNT + seg_hi].xyz;
    let v10 = filter_mesh[ring_hi * FILTER_SEGMENT_COUNT + seg_lo].xyz;
    let v11 = filter_mesh[ring_hi * FILTER_SEGMENT_COUNT + seg_hi].xyz;

    let ring_r_lo = mix(length(v00.xz), length(v01.xz), seg_frac);
    let ring_r_hi = mix(length(v10.xz), length(v11.xz), seg_frac);
    let ring_y_lo = mix(v00.y, v01.y, seg_frac);
    let ring_y_hi = mix(v10.y, v11.y, seg_frac);
    let mesh_r = mix(ring_r_lo, ring_r_hi, ring_frac);

    var out_pos = pos;
    var out_vel = vel;

    let radial = out_pos.xz;
    let radial_len = length(radial);
    if mesh_r > 0.1 && radial_len > 1e-6 {
        let barrier_r = mesh_r - contact_offset();
        if radial_len > barrier_r {
            let outward = radial / radial_len;
            let dy = max(abs(ring_y_hi - ring_y_lo), 1e-5);
            let dr_dy = (ring_r_hi - ring_r_lo) / dy;
            // Keep position correction radial so particles can settle onto the
            // filter instead of being lifted above it by an over-large normal
            // projection. Use the local cone normal only for the velocity
            // response so the contact still provides upward support.
            let surface_normal = normalize(vec3<f32>(outward.x, -dr_dy, outward.y));
            let penetration = radial_len - barrier_r;
            out_pos.x -= outward.x * penetration;
            out_pos.z -= outward.y * penetration;

            let vn = dot(out_vel, surface_normal);
            if vn > 0.0 {
                out_vel = out_vel - surface_normal * vn;
                let vt = out_vel - surface_normal * dot(out_vel, surface_normal);
                let vt_len = length(vt);
                if vt_len > 1e-6 {
                    let friction_impulse = min(friction() * abs(vn), vt_len);
                    out_vel = out_vel - vt * (friction_impulse / vt_len);
                }
            }
        }
    }

    // Apex floor: only keep particles from falling through the bottom tip.
    // The radial barrier handles the cone walls; this handles the point.
    if out_pos.y < mesh_bot_y + contact_offset() {
        out_pos.y = mesh_bot_y + contact_offset();
        if out_vel.y < 0.0 {
            out_vel.y = 0.0;
            out_vel.x *= 1.0 - friction() * 0.55;
            out_vel.z *= 1.0 - friction() * 0.55;
        }
    }

    return ContactResult(out_pos, out_vel);
}

fn resolve_scene_obstacles(position: vec3<f32>, velocity: vec3<f32>, is_bed: bool) -> ContactResult {
    var out_pos = position;
    var out_vel = velocity;

    // V60 dripper interior. Radial barrier keeps particles inside the cone.
    let cone_top_y = 3.0;
    let cone_bot_y = -3.0;
    if out_pos.y <= cone_top_y && out_pos.y >= cone_bot_y {
        let t = clamp((out_pos.y - cone_bot_y) / (cone_top_y - cone_bot_y), 0.0, 1.0);
        let cone_radius = mix(0.8, 4.5, t) - contact_offset();
        let cone_contact = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), cone_radius);
        out_pos = cone_contact.pos;
        out_vel = cone_contact.vel;
    }

    // Paper filter (bed particles only): the filter mesh is a deformable
    // collision surface uploaded from the CPU cloth sim each frame.
    if is_bed && has_filter() {
        let fc = resolve_filter_mesh_contact(out_pos, out_vel);
        out_pos = fc.pos;
        out_vel = fc.vel;
    }

    // Carafe interior. Keep pooled water inside the cup walls and above the
    // floor so accumulation reads as actual contained volume.
    if out_pos.y <= -3.5 {
        let cup_radius = 3.0 - contact_offset();
        let cup_contact = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), cup_radius);
        out_pos = cup_contact.pos;
        out_vel = cup_contact.vel;

        let floor_y = -8.0 + contact_offset();
        if out_pos.y < floor_y {
            out_pos.y = floor_y;
            if out_vel.y < 0.0 {
                out_vel.y = 0.0;
                out_vel.x *= 1.0 - friction() * 0.55;
                out_vel.z *= 1.0 - friction() * 0.55;
            }
        }
    }

    return ContactResult(out_pos, out_vel);
}

fn resolve_sdf_contact(position: vec3<f32>, velocity: vec3<f32>, is_bed: bool) -> ContactResult {
    var out_pos = position;
    var out_vel = velocity;

    let sdf_val = sample_sdf(out_pos);
    if sdf_val < contact_offset() {
        let n = sdf_gradient(out_pos);
        if length(n) > 1e-6 {
            out_pos += n * (contact_offset() - sdf_val);
            let vn = dot(out_vel, n);
            if vn < 0.0 {
                out_vel = out_vel - n * vn * (1.0 + restitution());
                let vt = out_vel - n * dot(out_vel, n);
                let vt_len = length(vt);
                if vt_len > 1e-6 {
                    let friction_impulse = min(friction() * abs(vn), vt_len);
                    out_vel = out_vel - vt * (friction_impulse / vt_len);
                }
            }
        }
    }

    let hard_contact = resolve_scene_obstacles(out_pos, out_vel, is_bed);
    return ContactResult(hard_contact.pos, hard_contact.vel);
}

fn clamp_velocity(v_in: vec3<f32>) -> vec3<f32> {
    var v = v_in;
    let speed = length(v);
    if speed > vel_cap() {
        v = v * (vel_cap() / speed);
    }
    return v;
}

fn project_solid_velocity_against_filter(velocity: vec3<f32>, cell_pos: vec3<f32>) -> vec3<f32> {
    if !has_filter() { return velocity; }

    let mesh_bot_y = filter_mesh[0].y;
    let mesh_top_y = filter_mesh[(FILTER_RING_COUNT - 1u) * FILTER_SEGMENT_COUNT].y;

    if cell_pos.y > mesh_top_y || cell_pos.y < mesh_bot_y - 0.5 {
        return velocity;
    }

    let ring_t = clamp((cell_pos.y - mesh_bot_y) / (mesh_top_y - mesh_bot_y), 0.0, 1.0);
    let ring_f = ring_t * f32(FILTER_RING_COUNT - 1u);
    let ring_lo = u32(floor(ring_f));
    let ring_hi = min(ring_lo + 1u, FILTER_RING_COUNT - 1u);
    let ring_frac = ring_f - floor(ring_f);

    let angle = atan2(cell_pos.z, cell_pos.x);
    let seg_f = ((angle / (2.0 * 3.14159265) + 1.0) % 1.0) * f32(FILTER_SEGMENT_COUNT);
    let seg_lo = u32(floor(seg_f)) % FILTER_SEGMENT_COUNT;
    let seg_hi = (seg_lo + 1u) % FILTER_SEGMENT_COUNT;
    let seg_frac = seg_f - floor(seg_f);

    let v00 = filter_mesh[ring_lo * FILTER_SEGMENT_COUNT + seg_lo].xyz;
    let v01 = filter_mesh[ring_lo * FILTER_SEGMENT_COUNT + seg_hi].xyz;
    let v10 = filter_mesh[ring_hi * FILTER_SEGMENT_COUNT + seg_lo].xyz;
    let v11 = filter_mesh[ring_hi * FILTER_SEGMENT_COUNT + seg_hi].xyz;

    let ring_r_lo = mix(length(v00.xz), length(v01.xz), seg_frac);
    let ring_r_hi = mix(length(v10.xz), length(v11.xz), seg_frac);
    let ring_y_lo = mix(v00.y, v01.y, seg_frac);
    let ring_y_hi = mix(v10.y, v11.y, seg_frac);
    let mesh_r = mix(ring_r_lo, ring_r_hi, ring_frac);

    var v = velocity;
    let radial = cell_pos.xz;
    let radial_len = length(radial);

    if mesh_r > dx() && radial_len > 1e-6 {
        // Wide region: full surface-normal projection against the cone wall.
        let barrier_r = mesh_r - contact_offset();
        if radial_len > barrier_r {
            let outward = radial / radial_len;
            let dy = max(abs(ring_y_hi - ring_y_lo), 1e-5);
            let dr_dy = (ring_r_hi - ring_r_lo) / dy;
            let surface_normal = normalize(vec3<f32>(outward.x, -dr_dy, outward.y));
            let vn = dot(v, surface_normal);
            if vn > 0.0 {
                v = v - surface_normal * vn;
            }
        }
    } else if radial_len > 1e-6 && radial_len > mesh_r {
        // Apex region: filter is narrower than a grid cell. Only zero the
        // outward radial velocity to prevent sideways escape. Leave the
        // downward component intact so the bed can settle into the tip.
        let radial_dir = radial / radial_len;
        let radial_v = dot(vec2<f32>(v.x, v.z), radial_dir);
        if radial_v > 0.0 {
            v.x -= radial_dir.x * radial_v;
            v.z -= radial_dir.y * radial_v;
        }
    }

    // Apex floor: prevent solid velocity from pushing bed through the tip.
    if cell_pos.y < mesh_bot_y + dx() && v.y < 0.0 {
        v.y = 0.0;
    }

    return v;
}

fn project_grid_velocity(
    velocity: vec3<f32>,
    cell_pos: vec3<f32>,
    sdf_val: f32,
    normal: vec3<f32>,
    bmin: vec3<f32>,
    bmax: vec3<f32>,
) -> vec3<f32> {
    var v = velocity;

    if sdf_val < contact_offset() {
        let vn = dot(v, normal);
        if vn < 0.0 {
            v = v - normal * vn * (1.0 + restitution());
            let vt = v - normal * dot(v, normal);
            let vt_len = length(vt);
            if vt_len > 1e-6 {
                let friction_impulse = min(friction() * abs(vn), vt_len);
                v = v - vt * (friction_impulse / vt_len);
            }
        }
    }

    if cell_pos.x < bmin.x && v.x < 0.0 { v.x = 0.0; }
    if cell_pos.x > bmax.x && v.x > 0.0 { v.x = 0.0; }
    if cell_pos.y < bmin.y && v.y < 0.0 { v.y = 0.0; }
    if cell_pos.y > bmax.y && v.y > 0.0 { v.y = 0.0; }
    if cell_pos.z < bmin.z && v.z < 0.0 { v.z = 0.0; }
    if cell_pos.z > bmax.z && v.z > 0.0 { v.z = 0.0; }

    return v;
}

// ── clear_grid ──

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    atomicStore(&grid[grid_mass_idx(idx)], 0);
    atomicStore(&grid[grid_mom_x_idx(idx)], 0);
    atomicStore(&grid[grid_mom_y_idx(idx)], 0);
    atomicStore(&grid[grid_mom_z_idx(idx)], 0);
    atomicStore(&grid[grid_solid_mass_idx(idx)], 0);
    atomicStore(&grid[grid_solid_mom_x_idx(idx)], 0);
    atomicStore(&grid[grid_solid_mom_y_idx(idx)], 0);
    atomicStore(&grid[grid_solid_mom_z_idx(idx)], 0);
    grid_vel[idx] = vec4<f32>(0.0);
    grid_vel[grid_vel_solid_idx(idx)] = vec4<f32>(0.0);
}

// ── p2g ──

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= num_particles() { return; }

    let p = particles[pid];
    let a = affine[pid];
    let xp = p.pos.xyz;
    let vp = p.vel.xyz;
    let J = p.pos.w;
    let mass_p = p.vel.w;
    let phase = a.col0.w;
    if mass_p <= inactive_mass_threshold() {
        return;
    }

    let origin = u.grid_origin.xyz;
    let grid_pos = (xp - origin) * inv_dx();
    let base = vec3<i32>(floor(grid_pos - 0.5));
    let fx = grid_pos - vec3<f32>(base);

    // Quadratic B-spline weights
    var wx: array<f32, 3>;
    var wy: array<f32, 3>;
    var wz: array<f32, 3>;
    wx[0] = 0.5 * (1.5 - fx.x) * (1.5 - fx.x);
    wx[1] = 0.75 - (fx.x - 1.0) * (fx.x - 1.0);
    wx[2] = 0.5 * (fx.x - 0.5) * (fx.x - 0.5);
    wy[0] = 0.5 * (1.5 - fx.y) * (1.5 - fx.y);
    wy[1] = 0.75 - (fx.y - 1.0) * (fx.y - 1.0);
    wy[2] = 0.5 * (fx.y - 0.5) * (fx.y - 0.5);
    wz[0] = 0.5 * (1.5 - fx.z) * (1.5 - fx.z);
    wz[1] = 0.75 - (fx.z - 1.0) * (fx.z - 1.0);
    wz[2] = 0.5 * (fx.z - 0.5) * (fx.z - 0.5);

    // Affine term starts as `mass_p * C`; bed particles also add an
    // isotropic bulk-stress contribution through the same MLS-MPM channel.
    let C0 = a.col0.xyz;
    let C1 = a.col1.xyz;
    let C2 = a.col2.xyz;
    let is_bed = phase >= 0.5;
    var bed_saturation = 0.0;
    if is_bed && pid < num_bed() {
        bed_saturation = bed_extract[pid].extract.w;
    }
    var aff_col0 = vec3<f32>(mass_p * C0.x, mass_p * C0.y, mass_p * C0.z);
    var aff_col1 = vec3<f32>(mass_p * C1.x, mass_p * C1.y, mass_p * C1.z);
    var aff_col2 = vec3<f32>(mass_p * C2.x, mass_p * C2.y, mass_p * C2.z);
    if is_bed {
        let compaction = bed_compaction_response(bed_saturation);
        let K_eff = mix(K_bed() * 6.0, K_bed(), compaction);
        let stress_term = dt() * p_vol() * K_eff * (1.0 - J) * 4.0 * inv_dx() * inv_dx();
        aff_col0.x += stress_term;
        aff_col1.y += stress_term;
        aff_col2.z += stress_term;
    }

    let fp = fp_scale();
    let cell_dx = dx();

    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            for (var k = 0u; k < 3u; k++) {
                let offset = vec3<i32>(vec3<u32>(i, j, k));
                let cell = base + offset;

                if cell.x < 0 || cell.y < 0 || cell.z < 0 { continue; }
                if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { continue; }

                let w = wx[i] * wy[j] * wz[k];
                let dpos = (vec3<f32>(offset) - fx) * cell_dx;

                let mass_contrib = w * mass_p;
                let mom = w * (mass_p * vp + vec3<f32>(
                    dot(aff_col0, dpos),
                    dot(aff_col1, dpos),
                    dot(aff_col2, dpos),
                ));

                let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
                // Overflow probe: each per-cell per-axis term must stay below
                // `i32::MAX`. The tightest channel in practice is momentum,
                // which is mass_contrib * v_cap scaled by FP. We log any single
                // contribution that comes within ~50% of the int limit so
                // accumulation headroom stays visible from the HUD once the
                // readback path is re-enabled.
                let limit_m = 1.0e9;
                let mass_fp = mass_contrib * fp;
                let mom_x_fp = mom.x * fp;
                let mom_y_fp = mom.y * fp;
                let mom_z_fp = mom.z * fp;
                if abs(mass_fp) > limit_m || abs(mom_x_fp) > limit_m
                    || abs(mom_y_fp) > limit_m || abs(mom_z_fp) > limit_m {
                    atomicAdd(&metrics[METRIC_MASS_OVERFLOW_FIRES_IDX], 1u);
                }
                if is_bed {
                    atomicAdd(&grid[grid_solid_mass_idx(ci)], i32(mass_fp));
                    atomicAdd(&grid[grid_solid_mom_x_idx(ci)], i32(mom_x_fp));
                    atomicAdd(&grid[grid_solid_mom_y_idx(ci)], i32(mom_y_fp));
                    atomicAdd(&grid[grid_solid_mom_z_idx(ci)], i32(mom_z_fp));
                } else {
                    atomicAdd(&grid[grid_mass_idx(ci)], i32(mass_fp));
                    atomicAdd(&grid[grid_mom_x_idx(ci)], i32(mom_x_fp));
                    atomicAdd(&grid[grid_mom_y_idx(ci)], i32(mom_y_fp));
                    atomicAdd(&grid[grid_mom_z_idx(ci)], i32(mom_z_fp));
                }
            }
        }
    }
}

// ── grid_update ──

@compute @workgroup_size(64)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let inv_fp = inv_fp_scale();
    let mass_w = f32(atomicLoad(&grid[grid_mass_idx(idx)])) * inv_fp;
    let mass_s = f32(atomicLoad(&grid[grid_solid_mass_idx(idx)])) * inv_fp;
    if mass_w <= 1e-6 && mass_s <= 1e-6 {
        return;
    }
    var v_w = vec3<f32>(0.0);
    if mass_w > 1e-6 {
        v_w = vec3<f32>(
            f32(atomicLoad(&grid[grid_mom_x_idx(idx)])) * inv_fp / mass_w,
            f32(atomicLoad(&grid[grid_mom_y_idx(idx)])) * inv_fp / mass_w,
            f32(atomicLoad(&grid[grid_mom_z_idx(idx)])) * inv_fp / mass_w,
        );
        v_w.y += gravity() * dt();
    }

    var v_s = vec3<f32>(0.0);
    if mass_s > 1e-6 {
        v_s = vec3<f32>(
            f32(atomicLoad(&grid[grid_solid_mom_x_idx(idx)])) * inv_fp / mass_s,
            f32(atomicLoad(&grid[grid_solid_mom_y_idx(idx)])) * inv_fp / mass_s,
            f32(atomicLoad(&grid[grid_solid_mom_z_idx(idx)])) * inv_fp / mass_s,
        );
        v_s.y += gravity() * dt();
    }

    var absorbed_frac = 0.0;
    if mass_w > 1e-6 && mass_s > 1e-6 {
        let n = uniform_porosity();
        let K = max(uniform_permeability(), 1e-4);
        let beta = dt() * n * n * mu_fluid() / K;
        let drag = beta * (v_s - v_w)
            / (1.0 + beta * (1.0 / mass_w + 1.0 / mass_s));
        v_w += drag / mass_w;
        v_s -= drag / mass_s;
    }

    // Grid-level absorption: transfer water mass into bed pore storage.
    // This runs before G2P so the velocity/pressure fields reflect the sink.
    var mass_w_post = mass_w;
    if mass_w > 1e-6 && mass_s > 1e-6 {
        let iz_val = idx / (gx() * gy());
        let rem = idx % (gx() * gy());
        let iy_val = rem / gx();
        let ix_val = rem % gx();

        var neighbor_ids: array<i32, 27>;
        var neighbor_caps: array<f32, 27>;
        var neighbor_sats: array<f32, 27>;
        for (var init_i = 0u; init_i < BED_NEIGHBOR_SAMPLE_COUNT; init_i++) {
            neighbor_ids[init_i] = -1;
            neighbor_caps[init_i] = 0.0;
            neighbor_sats[init_i] = 0.0;
        }

        var neighbor_count = 0u;
        for (var di = -1; di <= 1; di++) {
            for (var dj = -1; dj <= 1; dj++) {
                for (var dk = -1; dk <= 1; dk++) {
                    let cx = i32(ix_val) + di;
                    let cy = i32(iy_val) + dj;
                    let cz = i32(iz_val) + dk;
                    if cx < 0 || cy < 0 || cz < 0 { continue; }
                    if u32(cx) >= gx() || u32(cy) >= gy() || u32(cz) >= gz() { continue; }

                    let ci = cell_index(u32(cx), u32(cy), u32(cz));
                    let bed_idx = bed_lookup_load(ci);
                    if bed_idx < 0 || u32(bed_idx) >= num_bed() { continue; }

                    var found = false;
                    for (var existing = 0u; existing < neighbor_count; existing++) {
                        if neighbor_ids[existing] == bed_idx {
                            found = true;
                            break;
                        }
                    }
                    if found || neighbor_count >= BED_NEIGHBOR_SAMPLE_COUNT {
                        continue;
                    }

                    let bid = u32(bed_idx);
                    neighbor_ids[neighbor_count] = bed_idx;
                    neighbor_caps[neighbor_count] =
                        max(max_saturation() - bed_extract[bid].bed.x, 0.0);
                    neighbor_sats[neighbor_count] = bed_extract[bid].extract.w;
                    neighbor_count += 1u;
                }
            }
        }

        if neighbor_count > 0u {
            var sat_sum = 0.0;
            var cap_sum = 0.0;
            for (var sample_i = 0u; sample_i < neighbor_count; sample_i++) {
                sat_sum += neighbor_sats[sample_i];
                cap_sum += neighbor_caps[sample_i];
            }

            if cap_sum > 1e-6 {
                let saturation = sat_sum / f32(neighbor_count);
                let abs_rate = absorption_rate() * (1.0 - saturation) * dt();
                let m_abs = min(min(mass_w * clamp(abs_rate, 0.0, 0.3), mass_w * 0.5), cap_sum);
                if m_abs > 1e-6 {
                    mass_w_post = mass_w - m_abs;
                    absorbed_frac = m_abs / mass_w;

                    for (var sample_i = 0u; sample_i < neighbor_count; sample_i++) {
                        let cap_i = neighbor_caps[sample_i];
                        if cap_i <= 1e-6 { continue; }
                        let share = m_abs * cap_i / cap_sum;
                        atomicAdd(&bed_delta[u32(neighbor_ids[sample_i])], i32(share * fp_scale()));
                    }
                }
            }
        }
    }

    // Store absorbed fraction in scratch slot for bed_coupling to read.
    atomicStore(&grid[scratch_absorbed_idx(idx)], i32(absorbed_frac * fp_scale()));

    grid_vel[idx] = vec4<f32>(clamp_velocity(v_w), mass_w_post);
    grid_vel[grid_vel_solid_idx(idx)] = vec4<f32>(clamp_velocity(v_s), mass_s);
}

// ── classify_cells ──

@compute @workgroup_size(64)
fn classify_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let mass = grid_vel[idx].w;
    pressure_store(idx, 0.0);

    // Hoist index decomposition so the SDF probe, the has_air_neighbor
    // loop, and the divergence stencil can all reuse it.
    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();

    // SDF solid classification. Sign convention (see state.rs
    // `generate_sdf_data`):
    //   sdf > 0  → open fluid domain (inside the cup)
    //   sdf = 0  → on the wall surface
    //   sdf < 0  → inside wall material or exterior ambient space
    // Marking sdf<0 cells as CELL_SOLID lets the pressure solve treat them
    // with a Neumann BC (∂p/∂n = 0) via the ghost-mirror trick in
    // pressure_update / project_pressure, instead of the Dirichlet p=0
    // they'd get if lumped with air.
    let cell_center = u.grid_origin.xyz
        + (vec3<f32>(f32(ix_val), f32(iy_val), f32(iz_val)) + vec3<f32>(0.5)) * dx();
    let self_is_solid = select(
        sample_sdf(cell_center) < 0.0,
        sdf_class_is_solid(vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val))),
        use_sdf_cache(),
    );
    if self_is_solid {
        atomicStore(&grid[scratch_kind_idx(idx)], CELL_SOLID);
        divergence_store(idx, 0.0);
        return;
    }

    let solid_mass = grid_vel[grid_vel_solid_idx(idx)].w;

    if mass <= occupancy_mass_threshold() {
        if solid_mass > 1e-6 {
            atomicStore(&grid[scratch_kind_idx(idx)], CELL_BED_COUPLED);
            divergence_store(idx, 0.0);
            return;
        }
        atomicStore(&grid[scratch_kind_idx(idx)], CELL_AIR);
        divergence_store(idx, 0.0);
        return;
    }

    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0),
        vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1),
        vec3<i32>(0, 0, 1),
    );

    var has_air_neighbor = false;
    for (var n = 0u; n < 6u; n++) {
        let neighbor = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val)) + offsets[n];
        if neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0
            || u32(neighbor.x) >= gx() || u32(neighbor.y) >= gy() || u32(neighbor.z) >= gz() {
            has_air_neighbor = true;
            break;
        }

        let neighbor_idx = cell_index(u32(neighbor.x), u32(neighbor.y), u32(neighbor.z));
        if grid_vel[neighbor_idx].w <= occupancy_mass_threshold() {
            has_air_neighbor = true;
            break;
        }
    }

    atomicStore(
        &grid[scratch_kind_idx(idx)],
        select(CELL_INTERIOR_FLUID, CELL_SURFACE_FLUID, has_air_neighbor),
    );

    // Central-difference divergence using cell-centered velocities. No-flow
    // boundaries (off-grid faces and CELL_SOLID neighbors) use a ghost-mirror
    // on the normal velocity component: v_ghost.n = -v_self.n. That makes
    // the central difference cancel to zero at a quiescent wall cell, which
    // is the matching RHS treatment for the Neumann LHS handling in
    // pressure_update / project_pressure. Previously initializing these to 0
    // injected a spurious sink (v_self.n - 0) / 2dx at every wall-adjacent
    // fluid cell and pushed fluid away from the cup floor.
    //
    // Neighbor-solid detection samples the static SDF directly rather than
    // calling cell_kind_load, because classify_cells is the dispatch that
    // writes cell_kind — reading a neighbor's kind here races against other
    // workgroups. The SDF is read-only so neighbor SDF probes are race-free.
    let self_vel = grid_vel[idx].xyz;
    let dx_vec = dx();
    var vxm = -self_vel.x;
    var vxp = -self_vel.x;
    var vym = -self_vel.y;
    var vyp = -self_vel.y;
    var vzm = -self_vel.z;
    var vzp = -self_vel.z;
    if ix_val > 0u
        && select(
            sample_sdf(cell_center + vec3<f32>(-dx_vec, 0.0, 0.0)) >= 0.0,
            !sdf_class_is_solid(vec3<i32>(i32(ix_val) - 1, i32(iy_val), i32(iz_val))),
            use_sdf_cache(),
        ) {
        vxm = grid_vel[cell_index(ix_val - 1u, iy_val, iz_val)].x;
    }
    if ix_val + 1u < gx()
        && select(
            sample_sdf(cell_center + vec3<f32>(dx_vec, 0.0, 0.0)) >= 0.0,
            !sdf_class_is_solid(vec3<i32>(i32(ix_val) + 1, i32(iy_val), i32(iz_val))),
            use_sdf_cache(),
        ) {
        vxp = grid_vel[cell_index(ix_val + 1u, iy_val, iz_val)].x;
    }
    if iy_val > 0u
        && select(
            sample_sdf(cell_center + vec3<f32>(0.0, -dx_vec, 0.0)) >= 0.0,
            !sdf_class_is_solid(vec3<i32>(i32(ix_val), i32(iy_val) - 1, i32(iz_val))),
            use_sdf_cache(),
        ) {
        vym = grid_vel[cell_index(ix_val, iy_val - 1u, iz_val)].y;
    }
    if iy_val + 1u < gy()
        && select(
            sample_sdf(cell_center + vec3<f32>(0.0, dx_vec, 0.0)) >= 0.0,
            !sdf_class_is_solid(vec3<i32>(i32(ix_val), i32(iy_val) + 1, i32(iz_val))),
            use_sdf_cache(),
        ) {
        vyp = grid_vel[cell_index(ix_val, iy_val + 1u, iz_val)].y;
    }
    if iz_val > 0u
        && select(
            sample_sdf(cell_center + vec3<f32>(0.0, 0.0, -dx_vec)) >= 0.0,
            !sdf_class_is_solid(vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val) - 1)),
            use_sdf_cache(),
        ) {
        vzm = grid_vel[cell_index(ix_val, iy_val, iz_val - 1u)].z;
    }
    if iz_val + 1u < gz()
        && select(
            sample_sdf(cell_center + vec3<f32>(0.0, 0.0, dx_vec)) >= 0.0,
            !sdf_class_is_solid(vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val) + 1)),
            use_sdf_cache(),
        ) {
        vzp = grid_vel[cell_index(ix_val, iy_val, iz_val + 1u)].z;
    }

    let div = 0.5 * inv_dx() * ((vxp - vxm) + (vyp - vym) + (vzp - vzm));

    divergence_store(idx, div);

    // Observability: track the worst-case cell divergence and the fluid-cell
    // footprint of the active substep. `atomicMax` on u32 gives the peak FP
    // encoding; the HUD decodes via `METRICS_DIV_FP_SCALE`.
    let abs_div = abs(div);
    let fp_div = u32(clamp(abs_div * metrics_div_fp_scale(), 0.0, f32(0x7fffffffu)));
    atomicMax(&metrics[METRIC_MAX_ABS_DIV_IDX], fp_div);
    atomicAdd(&metrics[METRIC_FLUID_CELLS_IDX], 1u);
}

// ── pressure_rbgs ──

fn pressure_update(idx: u32, target_parity: u32) {
    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    if ((ix_val + iy_val + iz_val) & 1u) != target_parity {
        return;
    }

    let kind = cell_kind_load(idx);
    if !is_fluid_kind(kind) {
        pressure_store(idx, 0.0);
        return;
    }

    var neighbor_count = 0.0;
    var pressure_sum = 0.0;
    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0),
        vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1),
        vec3<i32>(0, 0, 1),
    );

    for (var n = 0u; n < 6u; n++) {
        let neighbor = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val)) + offsets[n];
        // Off-grid neighbors act as Neumann (no-flow) — skip from both
        // numerator and denominator. Preserves existing behavior.
        if neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0
            || u32(neighbor.x) >= gx() || u32(neighbor.y) >= gy() || u32(neighbor.z) >= gz() {
            continue;
        }

        let neighbor_idx = cell_index(u32(neighbor.x), u32(neighbor.y), u32(neighbor.z));
        let neighbor_kind = cell_kind_load(neighbor_idx);

        // Solid neighbor → Neumann BC via ghost-mirror (p_ghost = p_here).
        // The standard 7-point Laplacian with a mirror ghost drops the
        // solid face from both the numerator and denominator of the
        // averaging update, so we `continue` before touching neighbor_count.
        if is_solid_kind(neighbor_kind) {
            continue;
        }

        neighbor_count += 1.0;
        // Air neighbor → Dirichlet p=0, contributes 0 to pressure_sum.
        // Fluid neighbor → contributes its pressure.
        if is_fluid_kind(neighbor_kind) {
            pressure_sum += pressure_load(neighbor_idx);
        }
    }

    if neighbor_count <= 0.0 {
        pressure_store(idx, 0.0);
        return;
    }

    let rhs = divergence_load(idx) / max(dt(), 1e-6);
    let p_new = (pressure_sum - dx() * dx() * rhs) / neighbor_count;
    pressure_store(idx, p_new);
}

@compute @workgroup_size(64)
fn pressure_rbgs_red(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }
    pressure_update(idx, 0u);
}

@compute @workgroup_size(64)
fn pressure_rbgs_black(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }
    pressure_update(idx, 1u);
}

// ── project_pressure ──

@compute @workgroup_size(64)
fn project_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let gv = grid_vel[idx];
    if gv.w < 1e-6 { return; }

    let kind = cell_kind_load(idx);
    if !is_fluid_kind(kind) {
        return;
    }

    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();

    let p_here = pressure_load(idx);
    var p_xm = p_here;
    var p_xp = p_here;
    var p_ym = p_here;
    var p_yp = p_here;
    var p_zm = p_here;
    var p_zp = p_here;
    // Skip the overwrite when the neighbor is solid so the mirror
    // pressure (initialized to p_here) is preserved — this gives
    // ∂p/∂n = 0 across the wall face and zero contribution to grad_p.
    if ix_val > 0u {
        let ci = cell_index(ix_val - 1u, iy_val, iz_val);
        if !is_solid_kind(cell_kind_load(ci)) {
            p_xm = pressure_load(ci);
        }
    }
    if ix_val + 1u < gx() {
        let ci = cell_index(ix_val + 1u, iy_val, iz_val);
        if !is_solid_kind(cell_kind_load(ci)) {
            p_xp = pressure_load(ci);
        }
    }
    if iy_val > 0u {
        let ci = cell_index(ix_val, iy_val - 1u, iz_val);
        if !is_solid_kind(cell_kind_load(ci)) {
            p_ym = pressure_load(ci);
        }
    }
    if iy_val + 1u < gy() {
        let ci = cell_index(ix_val, iy_val + 1u, iz_val);
        if !is_solid_kind(cell_kind_load(ci)) {
            p_yp = pressure_load(ci);
        }
    }
    if iz_val > 0u {
        let ci = cell_index(ix_val, iy_val, iz_val - 1u);
        if !is_solid_kind(cell_kind_load(ci)) {
            p_zm = pressure_load(ci);
        }
    }
    if iz_val + 1u < gz() {
        let ci = cell_index(ix_val, iy_val, iz_val + 1u);
        if !is_solid_kind(cell_kind_load(ci)) {
            p_zp = pressure_load(ci);
        }
    }

    let grad_p = 0.5 * inv_dx() * vec3<f32>(
        p_xp - p_xm,
        p_yp - p_ym,
        p_zp - p_zm,
    );

    var v = gv.xyz - dt() * grad_p;
    let speed = length(v);
    if speed > vel_cap() {
        v = v * (vel_cap() / speed);
    }
    grid_vel[idx] = vec4<f32>(v, gv.w);
}

// ── boundary_project ──

@compute @workgroup_size(64)
fn boundary_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let gv_w = grid_vel[idx];
    let solid_idx = grid_vel_solid_idx(idx);
    let gv_s = grid_vel[solid_idx];
    if gv_w.w <= 1e-6 && gv_s.w <= 1e-6 {
        return;
    }

    // Reconstruct cell position from flat index
    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let origin = u.grid_origin.xyz;
    let cell_pos = origin + vec3<f32>(f32(ix_val), f32(iy_val), f32(iz_val)) * dx();

    // SDF collision
    let sdf_val = sample_sdf(cell_pos);
    let n = select(vec3<f32>(0.0), sdf_gradient(cell_pos), sdf_val < contact_offset());

    // Box boundary
    let margin = 2.0 * dx();
    let bmin = u.grid_origin.xyz + vec3<f32>(margin);
    let bmax = u.bounds_max.xyz - vec3<f32>(margin);

    if gv_w.w > 1e-6 {
        let v_w = project_grid_velocity(gv_w.xyz, cell_pos, sdf_val, n, bmin, bmax);
        grid_vel[idx] = vec4<f32>(v_w, gv_w.w);
    }

    if gv_s.w > 1e-6 {
        var v_s = project_grid_velocity(gv_s.xyz, cell_pos, sdf_val, n, bmin, bmax);
        v_s = project_solid_velocity_against_filter(v_s, cell_pos);
        grid_vel[solid_idx] = vec4<f32>(v_s, gv_s.w);
    }
}

// ── g2p ──

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= num_particles() { return; }

    let p = particles[pid];
    let xp = p.pos.xyz;
    let J_old = p.pos.w;
    let mass_p = p.vel.w;
    let phase = affine[pid].col0.w;
    if mass_p <= inactive_mass_threshold() {
        particles[pid].vel.w = 0.0;
        return;
    }

    let origin = u.grid_origin.xyz;
    let grid_pos = (xp - origin) * inv_dx();
    let base = vec3<i32>(floor(grid_pos - 0.5));
    let fx = grid_pos - vec3<f32>(base);

    var wx: array<f32, 3>;
    var wy: array<f32, 3>;
    var wz: array<f32, 3>;
    wx[0] = 0.5 * (1.5 - fx.x) * (1.5 - fx.x);
    wx[1] = 0.75 - (fx.x - 1.0) * (fx.x - 1.0);
    wx[2] = 0.5 * (fx.x - 0.5) * (fx.x - 0.5);
    wy[0] = 0.5 * (1.5 - fx.y) * (1.5 - fx.y);
    wy[1] = 0.75 - (fx.y - 1.0) * (fx.y - 1.0);
    wy[2] = 0.5 * (fx.y - 0.5) * (fx.y - 0.5);
    wz[0] = 0.5 * (1.5 - fx.z) * (1.5 - fx.z);
    wz[1] = 0.75 - (fx.z - 1.0) * (fx.z - 1.0);
    wz[2] = 0.5 * (fx.z - 0.5) * (fx.z - 0.5);

    var new_v = vec3<f32>(0.0);
    var new_C0 = vec3<f32>(0.0);
    var new_C1 = vec3<f32>(0.0);
    var new_C2 = vec3<f32>(0.0);
    var supported_weight = 0.0;
    let is_bed = phase >= 0.5;

    let B = 4.0 * inv_dx() * inv_dx();
    let cell_dx = dx();

    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            for (var k = 0u; k < 3u; k++) {
                let offset = vec3<i32>(vec3<u32>(i, j, k));
                let cell = base + offset;

                if cell.x < 0 || cell.y < 0 || cell.z < 0 { continue; }
                if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { continue; }

                let w = wx[i] * wy[j] * wz[k];
                let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
                let vel_idx = select(ci, grid_vel_solid_idx(ci), is_bed);
                let grid_v = grid_vel[vel_idx].xyz;
                let dpos = (vec3<f32>(offset) - fx) * cell_dx;
                let grid_mass = grid_vel[vel_idx].w;

                if grid_mass > 1e-6 {
                    new_v += w * grid_v;
                    // APIC: C = B * sum(w * v * dpos^T)
                    new_C0 += w * B * grid_v * dpos.x;
                    new_C1 += w * B * grid_v * dpos.y;
                    new_C2 += w * B * grid_v * dpos.z;
                    supported_weight += w;
                }
            }
        }
    }

    if supported_weight > 1e-6 {
        let inv_supported = 1.0 / supported_weight;
        new_v *= inv_supported;
        new_C0 *= inv_supported;
        new_C1 *= inv_supported;
        new_C2 *= inv_supported;
    } else {
        // If a particle sees no active grid support at all this substep,
        // preserve its ballistic motion instead of collapsing it to zero.
        new_v = vec3<f32>(p.vel.x, p.vel.y + gravity() * dt(), p.vel.z);
        new_C0 = vec3<f32>(0.0);
        new_C1 = vec3<f32>(0.0);
        new_C2 = vec3<f32>(0.0);
    }

    if is_bed {
        // Mild APIC regularization for the solid phase: the current bed model
        // has volumetric elasticity but no full shear/plastic constitutive
        // response yet, so damping only the affine subgrid mode is a narrower
        // stabilizer than damping particle velocity directly.
        let apic_regularization = 0.85;
        new_C0 *= apic_regularization;
        new_C1 *= apic_regularization;
        new_C2 *= apic_regularization;
    }

    var J_new = 1.0;
    if is_bed {
        let dt_c0 = new_C0 * dt();
        let dt_c1 = new_C1 * dt();
        let dt_c2 = new_C2 * dt();
        let F_col0 = vec3<f32>(1.0 + dt_c0.x, dt_c0.y, dt_c0.z);
        let F_col1 = vec3<f32>(dt_c1.x, 1.0 + dt_c1.y, dt_c1.z);
        let F_col2 = vec3<f32>(dt_c2.x, dt_c2.y, 1.0 + dt_c2.z);
        let detF = max(determinant_from_cols(F_col0, F_col1, F_col2), 0.0);
        J_new = clamp(J_old * detF, 0.5, 1.5);
    }

    // Advect
    var new_pos = xp + new_v * dt();

    // Particle-level boundary projection closes the gap left by the grid-only
    // collision pass so the dripper wall behaves like a hard barrier.
    let mid_pos = mix(xp, new_pos, 0.5);
    var contact = resolve_sdf_contact(mid_pos, new_v, is_bed);
    new_v = contact.vel;
    contact = resolve_sdf_contact(new_pos, new_v, is_bed);
    new_pos = contact.pos;
    new_v = contact.vel;

    // Clamp to domain
    let margin = dx() * 0.5;
    let lo = u.grid_origin.xyz + vec3<f32>(margin);
    let hi = u.bounds_max.xyz - vec3<f32>(margin);
    new_pos = clamp(new_pos, lo, hi);

    particles[pid].pos = vec4<f32>(new_pos, J_new);
    particles[pid].vel = vec4<f32>(new_v, mass_p);

    affine[pid].col0 = vec4<f32>(new_C0, phase);
    affine[pid].col1 = vec4<f32>(new_C1, 0.0);
    affine[pid].col2 = vec4<f32>(new_C2, 0.0);
}

// ── bed_coupling ──

@compute @workgroup_size(64)
fn bed_coupling(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= num_particles() { return; }

    let phase = affine[pid].col0.w;
    if phase >= 0.5 { return; }

    let mass_p = particles[pid].vel.w;
    if mass_p <= inactive_mass_threshold() { return; }

    let pos = particles[pid].pos.xyz;
    let cell = world_to_cell(pos);
    if cell.x < 0 || cell.y < 0 || cell.z < 0 { return; }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { return; }

    let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    let frac = f32(atomicLoad(&grid[scratch_absorbed_idx(ci)])) * inv_fp_scale();
    if frac <= 1e-6 { return; }

    let absorbed = mass_p * clamp(frac, 0.0, 0.95);
    let remaining = mass_p - absorbed;
    if remaining <= inactive_mass_threshold() {
        particles[pid].vel = vec4<f32>(vec3<f32>(0.0), 0.0);
    } else {
        particles[pid].vel = vec4<f32>(particles[pid].vel.xyz, remaining);
    }
}

// ── extraction_advect ──

@compute @workgroup_size(64)
fn bed_redistribute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x;
    if bid >= num_bed() { return; }

    let self_pos = particles[bid].pos.xyz;
    let self_pore = bed_extract[bid].bed.x;
    if self_pore <= 1e-6 { return; }

    let cell = world_to_cell(self_pos);
    if cell.x < 0 || cell.y < 0 || cell.z < 0 { return; }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { return; }

    var seen_ids: array<i32, 27>;
    for (var i = 0u; i < 27u; i++) {
        seen_ids[i] = -1;
    }

    var seen_count = 0u;
    var best_neighbor = -1;
    var best_neighbor_pore = 0.0;
    var best_neighbor_capacity = 0.0;
    var best_score = 0.0;

    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var dk = -1; dk <= 1; dk++) {
                let c = cell + vec3<i32>(di, dj, dk);
                if c.x < 0 || c.y < 0 || c.z < 0 { continue; }
                if u32(c.x) >= gx() || u32(c.y) >= gy() || u32(c.z) >= gz() { continue; }

                let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
                let neighbor_id = bed_lookup_load(ci);
                if neighbor_id < 0 || u32(neighbor_id) >= num_bed() || neighbor_id == i32(bid) {
                    continue;
                }

                var duplicate = false;
                for (var s = 0u; s < seen_count; s++) {
                    if seen_ids[s] == neighbor_id {
                        duplicate = true;
                        break;
                    }
                }
                if duplicate || seen_count >= 27u {
                    continue;
                }
                seen_ids[seen_count] = neighbor_id;
                seen_count += 1u;

                let nid = u32(neighbor_id);
                let neighbor_pos = particles[nid].pos.xyz;
                let neighbor_pore = bed_extract[nid].bed.x;
                let neighbor_capacity = max(max_saturation() - neighbor_pore, 0.0);
                if neighbor_capacity <= 1e-6 { continue; }

                let pore_gradient = self_pore - neighbor_pore;
                if pore_gradient <= 1e-4 { continue; }

                let downward_bias = max(self_pos.y - neighbor_pos.y, 0.0);
                let score = pore_gradient + downward_bias * 0.15;
                if score > best_score {
                    best_score = score;
                    best_neighbor = neighbor_id;
                    best_neighbor_pore = neighbor_pore;
                    best_neighbor_capacity = neighbor_capacity;
                }
            }
        }
    }

    if best_neighbor < 0 { return; }

    let pore_gradient = max(self_pore - best_neighbor_pore, 0.0);
    let transfer = min(
        min(self_pore * bed_storage_redistribution_rate() * dt(), best_neighbor_capacity * 0.2),
        pore_gradient * 0.25,
    );
    if transfer <= 1e-6 { return; }

    let delta_fp = i32(transfer * fp_scale());
    if delta_fp <= 0 { return; }

    atomicAdd(&bed_delta[bid], -delta_fp);
    atomicAdd(&bed_delta[u32(best_neighbor)], delta_fp);
}

@compute @workgroup_size(64)
fn extraction_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x;
    if bid >= num_bed() { return; }

    var be = bed_extract[bid];
    let absorbed = f32(atomicExchange(&bed_delta[bid], 0)) * inv_fp_scale();
    if absorbed > 0.0 {
        be.bed.x = min(be.bed.x + absorbed, max_saturation());
        be.extract.w = be.bed.x / max(max_saturation(), 1e-6);
    }
    let sat = be.extract.w;

    if sat > 0.01 {
        let flux = extraction_rate() * be.extract.x * sat * dt();
        be.extract.x = max(be.extract.x - flux, 0.0);
        be.extract.y += flux;
    }

    bed_extract[bid] = be;
}

// ── prepare_render ──

@compute @workgroup_size(64)
fn prepare_render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= num_particles() { return; }

    let p = particles[pid];
    let phase = affine[pid].col0.w;
    if p.vel.w <= inactive_mass_threshold() {
        render_data[pid] = vec4<f32>(0.0, -1e6, 0.0, -999.0);
        return;
    }

    var color_t = 0.0;
    if phase < 0.5 {
        let speed = length(p.vel.xyz);
        color_t = clamp(speed / 10.0, 0.0, 2.0);
    } else {
        let bed_idx = pid;
        var sat = 0.0;
        if bed_idx < num_bed() {
            sat = bed_extract[bed_idx].extract.w;
        }
        color_t = -1.0 - sat;
    }

    render_data[pid] = vec4<f32>(p.pos.xyz, color_t);
}

// ── metrics_clear ──

const METRICS_SLOT_COUNT: u32 = 8u;

@compute @workgroup_size(8)
fn metrics_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= METRICS_SLOT_COUNT { return; }
    atomicStore(&metrics[idx], 0u);
}

// ── bed_lookup_clear ──

@compute @workgroup_size(64)
fn bed_lookup_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }
    atomicStore(&bed_lookup[idx], -1);
}

// ── bed_lookup_scatter ──
//
// Each bed particle stamps its id into a 3x3x3 cell neighborhood around its
// center using `atomicMax`. The highest bed id wins deterministically, which
// matches the CPU-side `build_cell_lookup` ordering and keeps the lookup
// current as bed particles deform from their rest positions.
@compute @workgroup_size(64)
fn bed_lookup_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x;
    if bid >= num_bed() { return; }

    // Bed particles occupy the leading slots in the particle array (bed
    // first, water appended). The bed array has length `num_bed` and starts
    // at index 0.
    let pid = bid;
    let phase = affine[pid].col0.w;
    if phase < 0.5 {
        // Defensive: bed particle should always have phase >= 0.5.
        return;
    }

    let pos = particles[pid].pos.xyz;
    let cell = world_to_cell(pos);
    let id = i32(bid);

    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var dk = -1; dk <= 1; dk++) {
                let c = cell + vec3<i32>(di, dj, dk);
                if c.x < 0 || c.y < 0 || c.z < 0 { continue; }
                if u32(c.x) >= gx() || u32(c.y) >= gy() || u32(c.z) >= gz() { continue; }
                let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
                atomicMax(&bed_lookup[ci], id);
            }
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::MPM_COMPUTE_SHADER;

    #[test]
    fn shader_parses_with_naga() {
        let module = naga::front::wgsl::parse_str(MPM_COMPUTE_SHADER)
            .expect("mpm compute shader should parse");
        assert!(!module.entry_points.is_empty());
    }
}

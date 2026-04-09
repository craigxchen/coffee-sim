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

// Metrics slot layout — keep in sync with `METRICS_SLOT_COUNT` in state.rs.
const METRIC_MAX_ABS_DIV_IDX: u32 = 0u;
const METRIC_FLUID_CELLS_IDX: u32 = 1u;
const METRIC_DIV_CLAMP_FIRES_IDX: u32 = 2u;
const METRIC_PRESSURE_CLAMP_FIRES_IDX: u32 = 3u;
const METRIC_MASS_OVERFLOW_FIRES_IDX: u32 = 4u;

// ── Helpers ──

fn gx() -> u32 { return u.grid_dims.x; }
fn gy() -> u32 { return u.grid_dims.y; }
fn gz() -> u32 { return u.grid_dims.z; }
fn total_cells() -> u32 { return u.grid_dims.w; }
fn num_bed() -> u32 { return u.counts.y; }
fn num_particles() -> u32 { return u.counts.x + u.counts.y; }
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
fn drag_coeff() -> f32 { return u.bed_params.x; }
fn absorption_rate() -> f32 { return u.bed_params.y; }
fn max_saturation() -> f32 { return u.bed_params.z; }
fn extraction_rate() -> f32 { return u.extraction_params.x; }
fn bed_spring() -> f32 { return u.extraction_params.y; }
fn bed_damping() -> f32 { return u.extraction_params.z; }
fn bed_impact() -> f32 { return u.extraction_params.w; }
fn inactive_mass_threshold() -> f32 { return nominal_mass() * 0.10; }
fn div_clamp_limit() -> f32 { return u.clamp_params.x; }
fn pressure_clamp_limit() -> f32 { return u.clamp_params.y; }
fn metrics_div_fp_scale() -> f32 { return u.clamp_params.z; }
fn metrics_div_inv_fp_scale() -> f32 { return u.clamp_params.w; }

fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return iz * gx() * gy() + iy * gx() + ix;
}

fn grid_mass_idx(cell: u32) -> u32 { return cell; }
fn grid_mom_x_idx(cell: u32) -> u32 { return total_cells() + cell; }
fn grid_mom_y_idx(cell: u32) -> u32 { return 2u * total_cells() + cell; }
fn grid_mom_z_idx(cell: u32) -> u32 { return 3u * total_cells() + cell; }
fn scratch_pressure_idx(cell: u32) -> u32 { return grid_mass_idx(cell); }
fn scratch_div_idx(cell: u32) -> u32 { return grid_mom_x_idx(cell); }
// Slot 2 (`grid_mom_y_idx`) is reused only as the mass/momentum accumulator
// during `p2g`; projection no longer keeps a per-cell residual so no scratch
// alias is defined for it. Add one back if/when an iterative residual probe
// needs the slot during the projection pass.
fn scratch_kind_idx(cell: u32) -> u32 { return grid_mom_z_idx(cell); }
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

fn is_fluid_kind(kind: i32) -> bool {
    // Surface cells must participate in the pressure solve so hydrostatic
    // pressure can build up in shallow puddles. Adjacent CELL_AIR cells
    // provide the Dirichlet p=0 BC via the neighbor-loop sum in
    // `pressure_update` (air neighbors count toward the denominator but
    // contribute 0 to the numerator), so excluding surface cells here would
    // zero them out and let gravity compress thin pools to a single layer.
    return kind == CELL_INTERIOR_FLUID
        || kind == CELL_SURFACE_FLUID
        || kind == CELL_BED_COUPLED;
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

    // Paper filter (bed particles only): the filter is porous — water
    // passes through the paper, but coffee particles are trapped above.
    // Below the V60 cone the filter narrows to a point at y=-3.37.
    let filter_bot_y = -3.37;
    if is_bed {
        if out_pos.y < cone_bot_y && out_pos.y >= filter_bot_y {
            let ft = (out_pos.y - filter_bot_y) / (cone_bot_y - filter_bot_y);
            let filter_r = mix(0.0, 0.26, ft) - contact_offset();
            if filter_r > 0.0 {
                let fc = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), filter_r);
                out_pos = fc.pos;
                out_vel = fc.vel;
            }
        }

        // Filter apex: floor keeps bed particles from falling through.
        if out_pos.y <= filter_bot_y + contact_offset() && out_pos.y > filter_bot_y - 0.5 {
            out_pos.y = filter_bot_y + contact_offset();
            if out_vel.y < 0.0 {
                out_vel.y = 0.0;
                out_vel.x *= 1.0 - friction() * 0.55;
                out_vel.z *= 1.0 - friction() * 0.55;
            }
        }
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

// ── clear_grid ──

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    atomicStore(&grid[grid_mass_idx(idx)], 0);
    atomicStore(&grid[grid_mom_x_idx(idx)], 0);
    atomicStore(&grid[grid_mom_y_idx(idx)], 0);
    atomicStore(&grid[grid_mom_z_idx(idx)], 0);
    grid_vel[idx] = vec4<f32>(0.0);
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
    if phase >= 0.5 || mass_p <= inactive_mass_threshold() {
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

    // Affine = mass_p*C (J-stress disabled under pressure projection)
    let C0 = a.col0.xyz;
    let C1 = a.col1.xyz;
    let C2 = a.col2.xyz;
    let aff_col0 = vec3<f32>(mass_p * C0.x, mass_p * C0.y, mass_p * C0.z);
    let aff_col1 = vec3<f32>(mass_p * C1.x, mass_p * C1.y, mass_p * C1.z);
    let aff_col2 = vec3<f32>(mass_p * C2.x, mass_p * C2.y, mass_p * C2.z);

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
                atomicAdd(&grid[grid_mass_idx(ci)], i32(mass_fp));
                atomicAdd(&grid[grid_mom_x_idx(ci)], i32(mom_x_fp));
                atomicAdd(&grid[grid_mom_y_idx(ci)], i32(mom_y_fp));
                atomicAdd(&grid[grid_mom_z_idx(ci)], i32(mom_z_fp));
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
    let mass = f32(atomicLoad(&grid[grid_mass_idx(idx)])) * inv_fp;

    if mass < 1e-6 {
        grid_vel[idx] = vec4<f32>(0.0);
        return;
    }

    var v = vec3<f32>(
        f32(atomicLoad(&grid[grid_mom_x_idx(idx)])) * inv_fp / mass,
        f32(atomicLoad(&grid[grid_mom_y_idx(idx)])) * inv_fp / mass,
        f32(atomicLoad(&grid[grid_mom_z_idx(idx)])) * inv_fp / mass,
    );

    v.y += gravity() * dt();

    let speed = length(v);
    if speed > vel_cap() {
        v = v * (vel_cap() / speed);
    }

    grid_vel[idx] = vec4<f32>(v, mass);
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
    if sample_sdf(cell_center) < 0.0 {
        atomicStore(&grid[scratch_kind_idx(idx)], CELL_SOLID);
        divergence_store(idx, 0.0);
        return;
    }

    if mass <= occupancy_mass_threshold() {
        atomicStore(&grid[scratch_kind_idx(idx)], CELL_AIR);
        divergence_store(idx, 0.0);
        return;
    }

    if bed_lookup_load(idx) >= 0 {
        atomicStore(&grid[scratch_kind_idx(idx)], CELL_BED_COUPLED);
    } else {
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
    }

    let kind = cell_kind_load(idx);
    if !is_fluid_kind(kind) {
        divergence_store(idx, 0.0);
        return;
    }

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
        && sample_sdf(cell_center + vec3<f32>(-dx_vec, 0.0, 0.0)) >= 0.0 {
        vxm = grid_vel[cell_index(ix_val - 1u, iy_val, iz_val)].x;
    }
    if ix_val + 1u < gx()
        && sample_sdf(cell_center + vec3<f32>(dx_vec, 0.0, 0.0)) >= 0.0 {
        vxp = grid_vel[cell_index(ix_val + 1u, iy_val, iz_val)].x;
    }
    if iy_val > 0u
        && sample_sdf(cell_center + vec3<f32>(0.0, -dx_vec, 0.0)) >= 0.0 {
        vym = grid_vel[cell_index(ix_val, iy_val - 1u, iz_val)].y;
    }
    if iy_val + 1u < gy()
        && sample_sdf(cell_center + vec3<f32>(0.0, dx_vec, 0.0)) >= 0.0 {
        vyp = grid_vel[cell_index(ix_val, iy_val + 1u, iz_val)].y;
    }
    if iz_val > 0u
        && sample_sdf(cell_center + vec3<f32>(0.0, 0.0, -dx_vec)) >= 0.0 {
        vzm = grid_vel[cell_index(ix_val, iy_val, iz_val - 1u)].z;
    }
    if iz_val + 1u < gz()
        && sample_sdf(cell_center + vec3<f32>(0.0, 0.0, dx_vec)) >= 0.0 {
        vzp = grid_vel[cell_index(ix_val, iy_val, iz_val + 1u)].z;
    }

    let div = 0.5 * inv_dx() * ((vxp - vxm) + (vyp - vym) + (vzp - vzm));

    // Sink-augmented divergence for bed-coupled cells (PLAN.md formula):
    //   div(u) = -m_abs / (rho * V * dt)
    // Adding a positive sink tells the pressure solver to allow convergent
    // flow at bed cells rather than fighting it to zero. After projection
    // (vel -= dt * grad_p), the dt factors cancel via the 1/dt normalization
    // in pressure_update, so div(u_new) = -sink exactly.
    var sink = 0.0;
    if kind == CELL_BED_COUPLED {
        let bed_idx = bed_lookup_load(idx);
        if bed_idx >= 0 {
            let be = bed_extract[u32(bed_idx)];
            let saturation = be.extract.w;
            let abs_frac = clamp(absorption_rate() * (1.0 - saturation) * dt(), 0.0, 0.25);
            let predicted_abs = mass * abs_frac;
            let ref_mass = nominal_mass() * 8.0;
            sink = predicted_abs / (ref_mass * dt());
        }
    }
    divergence_store(idx, div + sink);

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

    let gv = grid_vel[idx];
    if gv.w < 1e-6 { return; }

    var v = gv.xyz;

    // Reconstruct cell position from flat index
    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let origin = u.grid_origin.xyz;
    let cell_pos = origin + vec3<f32>(f32(ix_val), f32(iy_val), f32(iz_val)) * dx();

    // SDF collision
    let sdf_val = sample_sdf(cell_pos);
    if sdf_val < contact_offset() {
        let n = sdf_gradient(cell_pos);
        let vn = dot(v, n);
        if vn < 0.0 {
            v = v - n * vn * (1.0 + restitution());
            // Friction: reduce tangential component
            let vt = v - n * dot(v, n);
            let vt_len = length(vt);
            if vt_len > 1e-6 {
                let friction_impulse = min(friction() * abs(vn), vt_len);
                v = v - vt * (friction_impulse / vt_len);
            }
        }
    }

    // Box boundary
    let margin = 2.0 * dx();
    let bmin = u.grid_origin.xyz + vec3<f32>(margin);
    let bmax = u.bounds_max.xyz - vec3<f32>(margin);

    if cell_pos.x < bmin.x && v.x < 0.0 { v.x = 0.0; }
    if cell_pos.x > bmax.x && v.x > 0.0 { v.x = 0.0; }
    if cell_pos.y < bmin.y && v.y < 0.0 { v.y = 0.0; }
    if cell_pos.y > bmax.y && v.y > 0.0 { v.y = 0.0; }
    if cell_pos.z < bmin.z && v.z < 0.0 { v.z = 0.0; }
    if cell_pos.z > bmax.z && v.z > 0.0 { v.z = 0.0; }

    grid_vel[idx] = vec4<f32>(v, gv.w);
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
    if phase >= 0.5 {
        return;
    }
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
    var local_grid_mass = 0.0;

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
                let grid_v = grid_vel[ci].xyz;
                let dpos = (vec3<f32>(offset) - fx) * cell_dx;
                let grid_mass = grid_vel[ci].w;

                if grid_mass > 1e-6 {
                    new_v += w * grid_v;
                    // APIC: C = B * sum(w * v * dpos^T)
                    new_C0 += w * B * grid_v * dpos.x;
                    new_C1 += w * B * grid_v * dpos.y;
                    new_C2 += w * B * grid_v * dpos.z;
                    supported_weight += w;
                    local_grid_mass += w * grid_mass;
                }
            }
        }
    }

    // Sparse jets suffer strong PIC-style dissipation because empty stencil nodes
    // contribute zero velocity. When support is weak, preserve more of the
    // particle's previous ballistic motion instead of letting the stream stall.
    let support_ratio = clamp(supported_weight, 0.0, 1.0);
    if supported_weight > 1e-6 {
        let inv_supported = 1.0 / supported_weight;
        new_v *= inv_supported;
        new_C0 *= inv_supported;
        new_C1 *= inv_supported;
        new_C2 *= inv_supported;
    }
    let in_cup_volume = xp.y < -3.5 && dot(xp.xz, xp.xz) < (3.0 + contact_offset()) * (3.0 + contact_offset());
    if support_ratio < 0.999 {
        let ballistic_v = vec3<f32>(p.vel.x, p.vel.y + gravity() * dt(), p.vel.z);
        let preserve = clamp((1.0 - support_ratio) * 1.15, 0.0, 0.95);
        new_v = mix(new_v, ballistic_v, preserve);
        let affine_damp = 1.0 - preserve * 0.75;
        new_C0 *= affine_damp;
        new_C1 *= affine_damp;
        new_C2 *= affine_damp;
    }

    // Even with full stencil support, a thin free stream below the dripper can be
    // severely under-dense. PIC/APIC transfer then numerically diffuses momentum.
    // Preserve more ballistic motion when the particle is airborne and local mass
    // support is low compared with a compact fluid region.
    let home_cell = world_to_cell(xp);
    var bed_near = false;
    if home_cell.x >= 0 && home_cell.y >= 0 && home_cell.z >= 0
        && u32(home_cell.x) < gx() && u32(home_cell.y) < gy() && u32(home_cell.z) < gz() {
        let home_idx = cell_index(u32(home_cell.x), u32(home_cell.y), u32(home_cell.z));
        bed_near = bed_lookup_load(home_idx) >= 0;
    }
    let airborne = !bed_near && sample_sdf(xp) > contact_offset() * 2.0;
    if airborne {
        let dense_mass = nominal_mass() * 4.0;
        let density_ratio = clamp(local_grid_mass / max(dense_mass, 1e-6), 0.0, 1.0);
        let ballistic_v = vec3<f32>(p.vel.x, p.vel.y + gravity() * dt(), p.vel.z);
        let preserve = clamp((1.0 - density_ratio) * 0.72, 0.0, 0.88);
        new_v = mix(new_v, ballistic_v, preserve);
        let affine_damp = 1.0 - preserve * 0.65;
        new_C0 *= affine_damp;
        new_C1 *= affine_damp;
        new_C2 *= affine_damp;
    }

    let J_new = 1.0;

    // Advect
    var new_pos = xp + new_v * dt();

    // Particle-level boundary projection closes the gap left by the grid-only
    // collision pass so the dripper wall behaves like a hard barrier.
    let mid_pos = mix(xp, new_pos, 0.5);
    var contact = resolve_sdf_contact(mid_pos, new_v, false);
    new_v = contact.vel;
    contact = resolve_sdf_contact(new_pos, new_v, false);
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

    if phase >= 0.5 {
        return;
    }

    let pos = particles[pid].pos.xyz;
    let cell = world_to_cell(pos);
    if cell.x < 0 || cell.y < 0 || cell.z < 0 { return; }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { return; }

    let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    let bed_idx = bed_lookup_load(ci);
    if bed_idx < 0 || u32(bed_idx) >= num_bed() {
        return;
    }

    var be = bed_extract[u32(bed_idx)];
    let saturation = be.extract.w;
    let permeability = be.bed.z;

    let drag = drag_coeff() * (1.0 - permeability * 0.1) * dt();
    let v = particles[pid].vel.xyz;
    particles[pid].vel = vec4<f32>(
        v / (1.0 + max(drag, 0.0)),
        particles[pid].vel.w,
    );

    let capacity = max(max_saturation() - be.bed.x, 0.0);
    if capacity <= 1e-6 {
        return;
    }

    let mass_p = particles[pid].vel.w;
    let abs_rate = absorption_rate() * (1.0 - saturation) * dt();
    let speed = length(particles[pid].vel.xyz);
    var absorbed = min(min(mass_p * clamp(abs_rate, 0.0, 0.25), mass_p * 0.5), capacity);
    let remaining_after_partial = mass_p - absorbed;
    let retire_threshold = nominal_mass() * 0.22;
    if remaining_after_partial > 0.0 && remaining_after_partial <= retire_threshold {
        absorbed = min(mass_p, capacity);
    } else if saturation > 0.55 && speed < 1.35 {
        let almost_absorbed = min(mass_p, capacity);
        if mass_p - almost_absorbed <= nominal_mass() * 0.35 {
            absorbed = almost_absorbed;
        }
    }
    if absorbed <= 1e-6 {
        return;
    }

    let remaining = mass_p - absorbed;
    if remaining <= inactive_mass_threshold() {
        particles[pid].vel = vec4<f32>(vec3<f32>(0.0), 0.0);
    } else {
        particles[pid].vel = vec4<f32>(particles[pid].vel.xyz, remaining);
    }
    atomicAdd(&bed_delta[u32(bed_idx)], i32(absorbed * fp_scale()));
}

// ── extraction_advect ──

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

// ── bed_dynamics ──

@compute @workgroup_size(64)
fn bed_dynamics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x;
    if bid >= num_bed() { return; }

    let pid = bid;
    let p = particles[pid];
    var rest = affine[pid].col1.xyz;
    let pos = p.pos.xyz;
    let mass_p = p.vel.w;

    let origin = u.grid_origin.xyz;
    let grid_pos = (pos - origin) * inv_dx();
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

    var water_v = vec3<f32>(0.0);
    var water_mass = 0.0;

    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            for (var k = 0u; k < 3u; k++) {
                let offset = vec3<i32>(vec3<u32>(i, j, k));
                let cell = base + offset;

                if cell.x < 0 || cell.y < 0 || cell.z < 0 { continue; }
                if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { continue; }

                let w = wx[i] * wy[j] * wz[k];
                let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
                let gv = grid_vel[ci];
                water_v += w * gv.xyz;
                water_mass += w * gv.w;
            }
        }
    }

    let sat = bed_extract[bid].extract.w;
    let mobility = clamp((1.0 - sat) * (0.35 + water_mass * 0.12), 0.0, 1.0);
    let surface_factor = clamp((rest.y + 3.0) / 3.5, 0.12, 1.0);
    let spring = bed_spring() * (0.8 + sat * 0.5);
    let damping = clamp(1.0 - bed_damping() * dt(), 0.0, 1.0);

    var vel = p.vel.xyz;
    vel += (rest - pos) * spring * dt();

    let impact_v = vec3<f32>(water_v.x * 0.25, min(water_v.y, 0.0) * 0.9, water_v.z * 0.25);
    vel += impact_v * bed_impact() * mobility * surface_factor * dt();
    vel *= damping;

    var new_pos = pos + vel * dt();
    var offset = new_pos - rest;
    let lateral_len = length(offset.xz);
    let max_lateral = dx() * 0.9 * surface_factor;
    if lateral_len > max_lateral && lateral_len > 1e-6 {
        let lateral_dir = offset.xz / lateral_len;
        offset.x = lateral_dir.x * max_lateral;
        offset.z = lateral_dir.y * max_lateral;
        vel.x *= 0.4;
        vel.z *= 0.4;
    }
    offset.y = clamp(offset.y, -dx() * (1.75 * surface_factor + 0.2), dx() * 0.18);
    new_pos = rest + offset;

    // Plastic compaction: once the bed is indented enough, lower the remembered
    // local rest height so the crater relaxes slowly instead of springing fully back.
    let compression = max(rest.y - new_pos.y, 0.0);
    let plastic_threshold = dx() * 0.18;
    if compression > plastic_threshold {
        let excess = compression - plastic_threshold;
        let plasticity = clamp(
            (0.18 + sat * 0.55 + mobility * 0.45) * surface_factor * dt() * 6.0,
            0.0,
            0.18,
        );
        rest.y -= excess * plasticity;
    }

    // Very slow rebound toward the original packed state for drier regions so
    // old craters soften over time rather than staying perfectly frozen forever.
    let packed_rest = affine[pid].col2.x;
    if packed_rest != 0.0 {
        let rebound = clamp((1.0 - sat) * dt() * 0.08, 0.0, 0.01);
        rest.y = mix(rest.y, packed_rest, rebound);
    }

    let contact = resolve_sdf_contact(new_pos, vel, true);
    new_pos = contact.pos;
    vel = contact.vel;

    particles[pid].pos = vec4<f32>(new_pos, p.pos.w);
    particles[pid].vel = vec4<f32>(vel, mass_p);
    affine[pid].col1 = vec4<f32>(rest, 0.0);
    bed_extract[bid].bed.w = max((rest.y - new_pos.y) / max(dx(), 1e-6), 0.0);
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

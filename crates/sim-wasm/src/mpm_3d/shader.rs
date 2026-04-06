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
@group(0) @binding(8) var<storage, read> bed_lookup: array<i32>;
@group(0) @binding(9) var<storage, read_write> bed_delta: array<atomic<i32>>;

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

fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return iz * gx() * gy() + iy * gx() + ix;
}

fn grid_mass_idx(cell: u32) -> u32 { return cell; }
fn grid_mom_x_idx(cell: u32) -> u32 { return total_cells() + cell; }
fn grid_mom_y_idx(cell: u32) -> u32 { return 2u * total_cells() + cell; }
fn grid_mom_z_idx(cell: u32) -> u32 { return 3u * total_cells() + cell; }

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

fn resolve_scene_obstacles(position: vec3<f32>, velocity: vec3<f32>) -> ContactResult {
    var out_pos = position;
    var out_vel = velocity;

    // V60 dripper interior. Open at the top and bottom, but particles should
    // stay inside the cone wall as they travel toward the outlet.
    let cone_top_y = 3.0;
    let cone_bot_y = -3.0;
    if out_pos.y <= cone_top_y && out_pos.y >= cone_bot_y {
        let t = clamp((out_pos.y - cone_bot_y) / (cone_top_y - cone_bot_y), 0.0, 1.0);
        let cone_radius = mix(0.8, 4.5, t) - contact_offset();
        let cone_contact = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), cone_radius);
        out_pos = cone_contact.pos;
        out_vel = cone_contact.vel;
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

fn resolve_sdf_contact(position: vec3<f32>, velocity: vec3<f32>) -> ContactResult {
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

    let hard_contact = resolve_scene_obstacles(out_pos, out_vel);
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

    let stress = -dt() * 4.0 * inv_dx() * inv_dx() * p_vol() * bulk_K() * (J - 1.0);

    // Affine = stress*I + mass_p*C
    let C0 = a.col0.xyz;
    let C1 = a.col1.xyz;
    let C2 = a.col2.xyz;
    let aff_col0 = vec3<f32>(stress + mass_p * C0.x, mass_p * C0.y, mass_p * C0.z);
    let aff_col1 = vec3<f32>(mass_p * C1.x, stress + mass_p * C1.y, mass_p * C1.z);
    let aff_col2 = vec3<f32>(mass_p * C2.x, mass_p * C2.y, stress + mass_p * C2.z);

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
                atomicAdd(&grid[grid_mass_idx(ci)], i32(mass_contrib * fp));
                atomicAdd(&grid[grid_mom_x_idx(ci)], i32(mom.x * fp));
                atomicAdd(&grid[grid_mom_y_idx(ci)], i32(mom.y * fp));
                atomicAdd(&grid[grid_mom_z_idx(ci)], i32(mom.z * fp));
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

                new_v += w * grid_v;
                // APIC: C = B * sum(w * v * dpos^T)
                new_C0 += w * B * grid_v * dpos.x;
                new_C1 += w * B * grid_v * dpos.y;
                new_C2 += w * B * grid_v * dpos.z;
            }
        }
    }

    // Extra dissipation and volume recovery in the carafe help the under-
    // resolved pool settle into a rising body of water instead of collapsing
    // into a thin, self-overlapping particle column.
    let in_carafe = xp.y < -3.6 && sample_sdf(xp) > 0.0;
    if in_carafe {
        let depth = clamp((-3.6 - xp.y) / 4.4, 0.0, 1.0);
        let lateral = clamp(1.0 - (6.0 + depth * 5.0) * dt(), 0.78, 1.0);
        let vertical = clamp(1.0 - (9.0 + depth * 8.0) * dt(), 0.68, 1.0);
        new_v = vec3<f32>(new_v.x * lateral, new_v.y * vertical, new_v.z * lateral);
        new_C0 *= 1.0 - clamp(viscosity() * dt() * (1.6 + depth), 0.0, 0.38);
        new_C1 *= 1.0 - clamp(viscosity() * dt() * (1.9 + depth), 0.0, 0.42);
        new_C2 *= 1.0 - clamp(viscosity() * dt() * (1.6 + depth), 0.0, 0.38);
    }

    // Update J (fluid volume ratio)
    let trace_C = new_C0.x + new_C1.y + new_C2.z;
    var J_new = J_old * (1.0 + dt() * trace_C);
    J_new = clamp(J_new, 0.1, 10.0);
    if in_carafe {
        let depth = clamp((-3.6 - xp.y) / 4.4, 0.0, 1.0);
        let recover = clamp((0.14 + depth * 0.2) * bulk_K() * dt() / 900.0, 0.0, 0.32);
        J_new = mix(J_new, 1.0, recover);
    }

    // Advect
    var new_pos = xp + new_v * dt();

    // Particle-level boundary projection closes the gap left by the grid-only
    // collision pass so the dripper wall behaves like a hard barrier.
    let mid_pos = mix(xp, new_pos, 0.5);
    var contact = resolve_sdf_contact(mid_pos, new_v);
    new_v = contact.vel;
    contact = resolve_sdf_contact(new_pos, new_v);
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
    let bed_idx = bed_lookup[ci];
    if bed_idx < 0 || u32(bed_idx) >= num_bed() {
        return;
    }

    var be = bed_extract[u32(bed_idx)];
    let saturation = be.extract.w;
    let permeability = be.bed.z;

    let drag = drag_coeff() * (1.0 - permeability * 0.1) * dt();
    let v = particles[pid].vel.xyz;
    particles[pid].vel = vec4<f32>(
        v * (1.0 - clamp(drag, 0.0, 0.95)),
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

    let contact = resolve_sdf_contact(new_pos, vel);
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

"#;

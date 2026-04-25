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
    mech0: vec4<f32>,
    mech1: vec4<f32>,
    mech2: vec4<f32>,
};

struct ContactResult {
    pos: vec3<f32>,
    vel: vec3<f32>,
};

struct RenderParticle {
    primary: vec4<f32>,
    aux: vec4<f32>,
};

struct FilterMeshVertex {
    current: vec4<f32>,
    previous: vec4<f32>,
};

struct FilterSupportSample {
    valid: bool,
    mesh_bot_y: f32,
    mesh_top_y: f32,
    surface_y: f32,
    surface_radius: f32,
    surface_normal: vec3<f32>,
    surface_velocity: vec3<f32>,
};

// ── Bindings ──

@group(0) @binding(0) var<uniform> u: MpmUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> affine: array<AffineC>;
@group(0) @binding(3) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> grid_vel: array<vec4<f32>>;
@group(0) @binding(5) var sdf_texture: texture_3d<f32>;
@group(0) @binding(6) var<storage, read_write> render_data: array<RenderParticle>;
@group(0) @binding(7) var<storage, read_write> bed_extract: array<BedExtract>;
@group(0) @binding(8) var<storage, read_write> bed_lookup: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> bed_delta: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> metrics: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> filter_mesh: array<FilterMeshVertex>;
@group(0) @binding(12) var sdf_class_tex: texture_3d<u32>;

const FILTER_RING_COUNT = 10u;
const FILTER_SEGMENT_COUNT = 32u;
const BED_NEIGHBOR_SAMPLE_COUNT: u32 = 27u;

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
fn use_sdf_cache() -> bool { return u.counts.w > 0u; }
fn dt() -> f32 { return u.sim_params.x; }
fn gravity() -> f32 { return u.sim_params.y; }
fn dx() -> f32 { return u.sim_params.z; }
fn inv_dx() -> f32 { return u.sim_params.w; }
fn sim_time() -> f32 { return u.time_params.x; }
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
fn bed_filter_contact_band() -> f32 { return max(contact_offset() * 1.5, dx() * 0.35); }
fn inactive_mass_threshold() -> f32 { return nominal_mass() * 0.10; }
fn div_clamp_limit() -> f32 { return u.clamp_params.x; }
fn pressure_clamp_limit() -> f32 { return u.clamp_params.y; }
fn metrics_div_fp_scale() -> f32 { return u.clamp_params.z; }
fn metrics_div_inv_fp_scale() -> f32 { return u.clamp_params.w; }
fn bed_compaction_response(saturation: f32) -> f32 {
    return smoothstep(0.05, 0.55, saturation);
}
fn bed_particle_capacity_scale(bid: u32) -> f32 {
    return max(bed_extract[bid].bed.w, 0.2);
}
fn bed_particle_capacity(bid: u32) -> f32 {
    return max(max_saturation() * bed_particle_capacity_scale(bid), 1e-6);
}
fn bed_compaction(bid: u32) -> f32 {
    return clamp(bed_extract[bid].mech0.w, 0.0, 0.12);
}
fn bed_rest_porosity(bid: u32) -> f32 {
    return clamp(max(bed_extract[bid].mech1.w, 0.0), 0.24, 0.58);
}
fn bed_rest_permeability(bid: u32) -> f32 {
    return clamp(max(bed_extract[bid].mech2.w, 0.0), 0.0002, 0.02);
}
fn bed_particle_porosity(bid: u32) -> f32 {
    return clamp(bed_extract[bid].bed.y, 0.18, 0.58);
}
fn bed_particle_permeability(bid: u32) -> f32 {
    return clamp(bed_extract[bid].bed.z, 0.0002, 0.02);
}
fn bed_settled_activation(bid: u32) -> f32 {
    return clamp(bed_extract[bid].extract.z, 0.0, 1.0);
}
fn bed_particle_saturation(bid: u32) -> f32 {
    return clamp(bed_extract[bid].bed.x / bed_particle_capacity(bid), 0.0, 1.0);
}
fn water_particle_radius() -> f32 {
    return dx() * 0.18;
}
fn bed_particle_render_radius(bid: u32) -> f32 {
    let permeability_ratio =
        bed_particle_permeability(bid) / max(uniform_permeability(), 1e-5);
    let size_scale = clamp(pow(permeability_ratio, 0.18), 0.68, 1.42);
    return dx() * 0.434 * size_scale;
}
fn bed_F_col0(bid: u32) -> vec3<f32> {
    return bed_extract[bid].mech0.xyz;
}
fn bed_F_col1(bid: u32) -> vec3<f32> {
    return bed_extract[bid].mech1.xyz;
}
fn bed_F_col2(bid: u32) -> vec3<f32> {
    return bed_extract[bid].mech2.xyz;
}
fn bed_shear_modulus() -> f32 {
    let poisson = 0.27;
    return 3.0 * K_bed() * (1.0 - 2.0 * poisson) / (2.0 * (1.0 + poisson));
}
fn bed_lambda_from_bulk() -> f32 {
    return K_bed() - 2.0 * bed_shear_modulus() / 3.0;
}
fn identity3() -> mat3x3<f32> {
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
}
fn safe_mat_cols(c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(c0, c1, c2);
}
fn orthonormal_basis_from_F(c0: vec3<f32>, c1: vec3<f32>) -> mat3x3<f32> {
    let r0 = normalize(select(vec3<f32>(1.0, 0.0, 0.0), c0, length(c0) > 1e-6));
    let c1_ortho = c1 - r0 * dot(r0, c1);
    let r1 = normalize(select(vec3<f32>(0.0, 1.0, 0.0), c1_ortho, length(c1_ortho) > 1e-6));
    let r2 = normalize(cross(r0, r1));
    return mat3x3<f32>(r0, r1, r2);
}
fn compaction_dryness_factor(saturation: f32) -> f32 {
    return 1.0 - smoothstep(0.10, 0.55, saturation);
}
fn compaction_support_factor(support_ratio: f32) -> f32 {
    return clamp((support_ratio - 0.35) / 0.65, 0.0, 1.0);
}
fn compaction_compression_factor(Je: f32) -> f32 {
    return clamp((0.98 - Je) / 0.20, 0.0, 1.0);
}
fn compaction_motion_factor(vertical_speed: f32) -> f32 {
    return smoothstep(0.02, 0.18, max(-vertical_speed, 0.0));
}
fn compaction_rest_factor(speed: f32) -> f32 {
    return 1.0 - smoothstep(0.04, 0.18, speed);
}
fn dry_settle_compaction_window() -> f32 {
    return 1.0 - smoothstep(2.0, 3.0, sim_time());
}
fn settled_support_ratio_factor(support_ratio: f32) -> f32 {
    return clamp((support_ratio - 0.40) / 0.30, 0.0, 1.0);
}
fn settled_support_mass_factor(local_grid_mass: f32) -> f32 {
    return clamp(
        (local_grid_mass - nominal_mass() * 0.14) / max(nominal_mass() * 0.60, 1e-6),
        0.0,
        1.0,
    );
}
fn settled_rest_factor(speed: f32) -> f32 {
    return 1.0 - smoothstep(0.01, 0.05, speed);
}
fn settled_drain_factor(local_fluid_mass: f32) -> f32 {
    return 1.0
        - smoothstep(
            occupancy_mass_threshold() * 0.02,
            nominal_mass() * 0.008,
            local_fluid_mass,
        );
}
fn raw_dry_contact_strength(
    saturation: f32,
    support_ratio: f32,
    local_grid_mass: f32,
) -> f32 {
    let dryness = 1.0 - smoothstep(0.0, 0.02, saturation);
    let support = clamp((support_ratio - 0.40) / 0.40, 0.0, 1.0);
    let supported_mass = clamp(
        (local_grid_mass - nominal_mass() * 0.08) / max(nominal_mass() * 0.65, 1e-6),
        0.0,
        1.0,
    );
    return dryness * support * supported_mass;
}
fn settled_activation_target(
    saturation: f32,
    compaction: f32,
    support_ratio: f32,
    local_grid_mass: f32,
    local_fluid_mass: f32,
    speed: f32,
) -> f32 {
    let dryness = 1.0 - smoothstep(0.001, 0.004, saturation);
    let packed = smoothstep(0.0015, 0.008, compaction);
    return dryness
        * packed
        * settled_support_ratio_factor(support_ratio)
        * settled_support_mass_factor(local_grid_mass)
        * settled_drain_factor(local_fluid_mass)
        * settled_rest_factor(speed);
}
fn update_settled_activation(
    settled_old: f32,
    saturation: f32,
    compaction: f32,
    support_ratio: f32,
    local_grid_mass: f32,
    local_fluid_mass: f32,
    speed: f32,
    support_strength: f32,
) -> f32 {
    let base_target = settled_activation_target(
        saturation,
        compaction,
        support_ratio,
        local_grid_mass,
        local_fluid_mass,
        speed,
    );
    let settle_target = max(
        base_target,
        0.35 * support_strength * settled_drain_factor(local_fluid_mass),
    );
    let rewetted = smoothstep(0.004, 0.012, saturation);
    let fluidized = smoothstep(
        occupancy_mass_threshold() * 0.01,
        nominal_mass() * 0.015,
        local_fluid_mass,
    );
    let destabilized = max(rewetted, fluidized);
    if destabilized > 1e-4 {
        let decay = min(dt() * (4.0 + 8.0 * destabilized), 0.35);
        return mix(settled_old, 0.0, decay);
    }
    let rise = min(dt() * (0.04 + 0.04 * settle_target), 0.01);
    let settled_new = mix(settled_old, settle_target, rise);
    if saturation > 5e-4 || local_fluid_mass > occupancy_mass_threshold() * 0.001 {
        return min(settled_new, 0.22);
    }
    return settled_new;
}
fn settled_contact_strength(
    settled_activation: f32,
    raw_contact_strength: f32,
) -> f32 {
    return smoothstep(0.12, 0.50, settled_activation) * raw_contact_strength;
}
fn settled_rest_activation(
    settled_activation: f32,
    support_ratio: f32,
    speed: f32,
    compaction: f32,
) -> f32 {
    let settled = smoothstep(0.10, 0.55, settled_activation);
    let supported = settled_support_ratio_factor(support_ratio);
    let near_rest = 1.0 - smoothstep(0.06, 0.22, speed);
    let packed = smoothstep(0.004, 0.016, compaction);
    return settled * supported * near_rest * packed;
}
fn runtime_porosity(rest_porosity: f32, compaction: f32) -> f32 {
    return clamp(rest_porosity * (1.0 - 1.20 * compaction), 0.18, 0.58);
}
fn runtime_permeability(rest_perm: f32, compaction: f32) -> f32 {
    return clamp(rest_perm * exp(-4.5 * compaction), 0.0002, 0.02);
}
fn filter_mesh_pos(idx: u32) -> vec3<f32> {
    return filter_mesh[idx].current.xyz;
}
fn filter_mesh_prev_pos(idx: u32) -> vec3<f32> {
    return filter_mesh[idx].previous.xyz;
}
fn filter_contact_band_strength(distance_to_surface: f32) -> f32 {
    let band = bed_filter_contact_band();
    return 1.0 - smoothstep(band, band * 2.0, distance_to_surface);
}
fn filter_wall_contact_strength(
    position: vec3<f32>,
    support: FilterSupportSample,
    dry_support_strength: f32,
) -> f32 {
    if !support.valid || dry_support_strength <= 1e-5 {
        return 0.0;
    }
    let radial_len = length(position.xz);
    if radial_len <= 1e-6 || support.surface_radius <= 0.1 {
        return 0.0;
    }
    let barrier_r = support.surface_radius - contact_offset();
    let wall_distance = abs(barrier_r - radial_len);
    return dry_support_strength * filter_contact_band_strength(wall_distance);
}
fn sample_filter_support(position: vec3<f32>) -> FilterSupportSample {
    let mesh_bot_y = filter_mesh_pos(0u).y;
    let mesh_top_y = filter_mesh_pos((FILTER_RING_COUNT - 1u) * FILTER_SEGMENT_COUNT).y;
    if position.y > mesh_top_y || position.y < mesh_bot_y - 0.5 {
        return FilterSupportSample(
            false,
            mesh_bot_y,
            mesh_top_y,
            0.0,
            0.0,
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0),
        );
    }

    let ring_t = clamp((position.y - mesh_bot_y) / max(mesh_top_y - mesh_bot_y, 1e-5), 0.0, 1.0);
    let ring_f = ring_t * f32(FILTER_RING_COUNT - 1u);
    let ring_lo = u32(floor(ring_f));
    let ring_hi = min(ring_lo + 1u, FILTER_RING_COUNT - 1u);
    let ring_frac = ring_f - floor(ring_f);

    let angle = atan2(position.z, position.x);
    let seg_f = ((angle / (2.0 * 3.14159265) + 1.0) % 1.0) * f32(FILTER_SEGMENT_COUNT);
    let seg_lo = u32(floor(seg_f)) % FILTER_SEGMENT_COUNT;
    let seg_hi = (seg_lo + 1u) % FILTER_SEGMENT_COUNT;
    let seg_frac = seg_f - floor(seg_f);

    let v00 = filter_mesh_pos(ring_lo * FILTER_SEGMENT_COUNT + seg_lo);
    let v01 = filter_mesh_pos(ring_lo * FILTER_SEGMENT_COUNT + seg_hi);
    let v10 = filter_mesh_pos(ring_hi * FILTER_SEGMENT_COUNT + seg_lo);
    let v11 = filter_mesh_pos(ring_hi * FILTER_SEGMENT_COUNT + seg_hi);
    let p00 = filter_mesh_prev_pos(ring_lo * FILTER_SEGMENT_COUNT + seg_lo);
    let p01 = filter_mesh_prev_pos(ring_lo * FILTER_SEGMENT_COUNT + seg_hi);
    let p10 = filter_mesh_prev_pos(ring_hi * FILTER_SEGMENT_COUNT + seg_lo);
    let p11 = filter_mesh_prev_pos(ring_hi * FILTER_SEGMENT_COUNT + seg_hi);

    let ring_r_lo = mix(length(v00.xz), length(v01.xz), seg_frac);
    let ring_r_hi = mix(length(v10.xz), length(v11.xz), seg_frac);
    let ring_y_lo = mix(v00.y, v01.y, seg_frac);
    let ring_y_hi = mix(v10.y, v11.y, seg_frac);
    let current_surface = mix(mix(v00, v01, seg_frac), mix(v10, v11, seg_frac), ring_frac);
    let previous_surface = mix(mix(p00, p01, seg_frac), mix(p10, p11, seg_frac), ring_frac);

    let radial = position.xz;
    let radial_len = length(radial);
    let outward = select(vec2<f32>(1.0, 0.0), radial / radial_len, radial_len > 1e-6);
    let dy = max(abs(ring_y_hi - ring_y_lo), 1e-5);
    let dr_dy = (ring_r_hi - ring_r_lo) / dy;
    let surface_normal = normalize(vec3<f32>(outward.x, -dr_dy, outward.y));
    let surface_velocity = (current_surface - previous_surface) / max(dt(), 1e-5);

    return FilterSupportSample(
        true,
        mesh_bot_y,
        mesh_top_y,
        current_surface.y,
        length(current_surface.xz),
        surface_normal,
        surface_velocity,
    );
}
fn project_dry_support_velocity(
    velocity: vec3<f32>,
    support_velocity: vec3<f32>,
    support_normal: vec3<f32>,
    support_strength: f32,
    allow_static: bool,
) -> vec3<f32> {
    let strength = clamp(support_strength, 0.0, 1.0);
    if strength <= 1e-5 {
        return velocity;
    }

    var rel = velocity - support_velocity;
    let outward_speed = max(dot(rel, support_normal), 0.0);
    if outward_speed > 0.0 {
        rel -= support_normal * outward_speed;
    }

    let tangential = rel - support_normal * dot(rel, support_normal);
    let tangential_speed = length(tangential);
    if tangential_speed > 1e-6 {
        let static_band = mix(0.02, 0.18, strength);
        if allow_static && tangential_speed <= static_band {
            rel -= tangential;
        } else {
            let kinetic_scale = min((0.45 + 0.35 * friction()) * strength, 0.97);
            rel -= tangential * kinetic_scale;
        }
    }

    return rel + support_velocity;
}
fn bed_fixed_corotated_stress(bid: u32, J: f32) -> mat3x3<f32> {
    let F0 = bed_F_col0(bid);
    let F1 = bed_F_col1(bid);
    let F2 = bed_F_col2(bid);
    let F = safe_mat_cols(F0, F1, F2);
    let R = orthonormal_basis_from_F(F0, F1);
    let mu = bed_shear_modulus();
    let lambda = bed_lambda_from_bulk();
    let PFt = (2.0 * mu) * (F - R) * transpose(F) + lambda * (J - 1.0) * J * identity3();
    return PFt;
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

fn static_cone_support_normal(position: vec3<f32>) -> vec3<f32> {
    let cone_top_y = 3.0;
    let cone_bot_y = -3.0;
    let radial = position.xz;
    let radial_len = length(radial);
    let outward = select(vec2<f32>(1.0, 0.0), radial / radial_len, radial_len > 1e-6);
    let dr_dy = (4.5 - 0.8) / max(cone_top_y - cone_bot_y, 1e-6);
    return normalize(vec3<f32>(outward.x, -dr_dy, outward.y));
}

fn resolve_filter_mesh_contact(pos: vec3<f32>, vel: vec3<f32>, dry_support_strength: f32) -> ContactResult {
    let support = sample_filter_support(pos);
    if !support.valid {
        return ContactResult(pos, vel);
    }

    var out_pos = pos;
    var out_vel = vel;

    let radial = out_pos.xz;
    let radial_len = length(radial);
    if support.surface_radius > 0.1 && radial_len > 1e-6 {
        let barrier_r = support.surface_radius - contact_offset();
        let wall_strength = filter_wall_contact_strength(out_pos, support, dry_support_strength);
        if wall_strength > 1e-4 {
            out_vel = project_dry_support_velocity(
                out_vel,
                support.surface_velocity,
                support.surface_normal,
                wall_strength,
                true,
            );
        }
        if radial_len > barrier_r {
            let outward = radial / radial_len;
            let penetration = radial_len - barrier_r;
            out_pos.x -= outward.x * penetration;
            out_pos.z -= outward.y * penetration;
            if wall_strength <= 1e-4 {
                let vn = dot(out_vel, support.surface_normal);
                if vn > 0.0 {
                    out_vel = out_vel - support.surface_normal * vn;
                    let vt = out_vel - support.surface_normal * dot(out_vel, support.surface_normal);
                    let vt_len = length(vt);
                    if vt_len > 1e-6 {
                        let friction_impulse = min(friction() * abs(vn) * 2.4, vt_len);
                        out_vel = out_vel - vt * (friction_impulse / vt_len);
                    }
                    out_vel *= 0.35;
                    if vel.y < 0.0 && out_vel.y > 0.0 {
                        out_vel.y = 0.0;
                    }
                }
            }
        }
    }

    // Apex floor: only keep particles from falling through the bottom tip.
    // The radial barrier handles the cone walls; this handles the point.
    let floor_y = support.mesh_bot_y + contact_offset();
    let floor_distance = abs(floor_y - out_pos.y);
    let floor_strength = dry_support_strength * filter_contact_band_strength(floor_distance);
    if floor_strength > 1e-4 && out_pos.y <= floor_y + bed_filter_contact_band() {
        out_vel = project_dry_support_velocity(
            out_vel,
            vec3<f32>(0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            floor_strength,
            true,
        );
    }
    if out_pos.y < floor_y {
        out_pos.y = floor_y;
        if floor_strength <= 1e-4 && out_vel.y < 0.0 {
            out_vel.y = 0.0;
            out_vel.x *= 0.35 * (1.0 - friction() * 0.55);
            out_vel.z *= 0.35 * (1.0 - friction() * 0.55);
        }
    }

    return ContactResult(out_pos, out_vel);
}

fn resolve_scene_obstacles(
    position: vec3<f32>,
    velocity: vec3<f32>,
    is_bed: bool,
    dry_support_strength: f32,
) -> ContactResult {
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
        let fc = resolve_filter_mesh_contact(out_pos, out_vel, dry_support_strength);
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

fn resolve_sdf_contact(
    position: vec3<f32>,
    velocity: vec3<f32>,
    is_bed: bool,
    dry_support_strength: f32,
) -> ContactResult {
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

    let hard_contact = resolve_scene_obstacles(out_pos, out_vel, is_bed, dry_support_strength);
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

fn project_solid_velocity_against_filter(velocity: vec3<f32>, cell_pos: vec3<f32>, cell_idx: u32) -> vec3<f32> {
    if !has_filter() { return velocity; }

    let support = sample_filter_support(cell_pos);
    if !support.valid {
        return velocity;
    }

    var v = velocity;
    var dry_support_strength = 0.0;
    let bed_idx = bed_lookup_load(cell_idx);
    if bed_idx >= 0 {
        let bid = u32(bed_idx);
        let saturation = bed_particle_saturation(bid);
        let compaction = bed_compaction(bid);
        let dryness = 1.0 - smoothstep(0.0, 0.08, saturation);
        let packing = smoothstep(0.0, 0.012, compaction);
        dry_support_strength = dryness * packing;
    }
    let radial = cell_pos.xz;
    let radial_len = length(radial);

    if support.surface_radius > dx() && radial_len > 1e-6 {
        let barrier_r = support.surface_radius - contact_offset();
        let wall_strength = filter_wall_contact_strength(cell_pos, support, dry_support_strength);
        if wall_strength > 1e-4 {
            v = project_dry_support_velocity(
                v,
                support.surface_velocity,
                support.surface_normal,
                wall_strength,
                true,
            );
        }
        if radial_len > barrier_r {
            if wall_strength <= 1e-4 {
                let vn = dot(v, support.surface_normal);
                if vn > 0.0 {
                    v = v - support.surface_normal * vn;
                    let vt = v - support.surface_normal * dot(v, support.surface_normal);
                    let vt_len = length(vt);
                    if vt_len > 1e-6 {
                        let friction_impulse = min(friction() * abs(vn) * 2.0, vt_len);
                        v = v - vt * (friction_impulse / vt_len);
                    }
                    v *= 0.35;
                    if velocity.y < 0.0 && v.y > 0.0 {
                        v.y = 0.0;
                    }
                }
            }
        }
    } else if radial_len > 1e-6 && radial_len > support.surface_radius {
        let radial_dir = radial / radial_len;
        let radial_v = dot(vec2<f32>(v.x, v.z), radial_dir);
        if radial_v > 0.0 {
            v.x -= radial_dir.x * radial_v;
            v.z -= radial_dir.y * radial_v;
        }
    }

    let floor_y = support.mesh_bot_y + contact_offset();
    if cell_pos.y < floor_y + dx() && v.y < 0.0 {
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
    var bed_porosity = uniform_porosity();
    if is_bed && pid < num_bed() {
        bed_porosity = bed_particle_porosity(pid);
    }
    var aff_col0 = vec3<f32>(mass_p * C0.x, mass_p * C0.y, mass_p * C0.z);
    var aff_col1 = vec3<f32>(mass_p * C1.x, mass_p * C1.y, mass_p * C1.z);
    var aff_col2 = vec3<f32>(mass_p * C2.x, mass_p * C2.y, mass_p * C2.z);
    if is_bed {
        let PFt = bed_fixed_corotated_stress(pid, J);
        let stress_affine = PFt * (-dt() * p_vol() * 4.0 * inv_dx() * inv_dx());
        aff_col0 += vec3<f32>(stress_affine[0].x, stress_affine[0].y, stress_affine[0].z);
        aff_col1 += vec3<f32>(stress_affine[1].x, stress_affine[1].y, stress_affine[1].z);
        aff_col2 += vec3<f32>(stress_affine[2].x, stress_affine[2].y, stress_affine[2].z);
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
    var v_w = vec3<f32>(0.0);
    if mass_w > 1e-6 {
        v_w = vec3<f32>(
            f32(atomicLoad(&grid[grid_mom_x_idx(idx)])) * inv_fp / mass_w,
            f32(atomicLoad(&grid[grid_mom_y_idx(idx)])) * inv_fp / mass_w,
            f32(atomicLoad(&grid[grid_mom_z_idx(idx)])) * inv_fp / mass_w,
        );
        v_w.y += gravity() * dt();
    }

    let mass_s = f32(atomicLoad(&grid[grid_solid_mass_idx(idx)])) * inv_fp;
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
    var mass_w_post = mass_w;
    if mass_w > 1e-6 && mass_s > 1e-6 {
        let iz_val = idx / (gx() * gy());
        let rem = idx % (gx() * gy());
        let iy_val = rem / gx();
        let ix_val = rem % gx();

        var neighbor_ids: array<i32, 27>;
        var neighbor_caps: array<f32, 27>;
        var neighbor_sats: array<f32, 27>;
        var neighbor_porosities: array<f32, 27>;
        var neighbor_perms: array<f32, 27>;
        for (var init_i = 0u; init_i < BED_NEIGHBOR_SAMPLE_COUNT; init_i++) {
            neighbor_ids[init_i] = -1;
            neighbor_caps[init_i] = 0.0;
            neighbor_sats[init_i] = 0.0;
            neighbor_porosities[init_i] = uniform_porosity();
            neighbor_perms[init_i] = uniform_permeability();
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
                    let local_capacity = bed_particle_capacity(bid);
                    neighbor_ids[neighbor_count] = bed_idx;
                    neighbor_caps[neighbor_count] =
                        max(local_capacity - bed_extract[bid].bed.x, 0.0);
                    neighbor_sats[neighbor_count] = bed_particle_saturation(bid);
                    neighbor_porosities[neighbor_count] = bed_particle_porosity(bid);
                    neighbor_perms[neighbor_count] = bed_particle_permeability(bid);
                    neighbor_count += 1u;
                }
            }
        }

        if neighbor_count > 0u {
            var sat_sum = 0.0;
            var cap_sum = 0.0;
            var porosity_sum = 0.0;
            var inv_perm_sum = 0.0;
            for (var sample_i = 0u; sample_i < neighbor_count; sample_i++) {
                sat_sum += neighbor_sats[sample_i];
                cap_sum += neighbor_caps[sample_i];
                porosity_sum += neighbor_porosities[sample_i];
                inv_perm_sum += 1.0 / max(neighbor_perms[sample_i], 1e-5);
            }

            let saturation = sat_sum / f32(neighbor_count);
            let local_porosity = porosity_sum / f32(neighbor_count);
            let local_perm = f32(neighbor_count) / max(inv_perm_sum, 1e-6);

            let beta = dt() * local_porosity * local_porosity * mu_fluid() / max(local_perm, 1e-5);
            let drag = beta * (v_s - v_w)
                / (1.0 + beta * (1.0 / mass_w + 1.0 / mass_s));
            v_w += drag / mass_w;
            v_s -= drag / mass_s;

            if cap_sum > 1e-6 {
                let transport_scale =
                    clamp(local_perm / max(uniform_permeability(), 1e-5), 0.08, 1.8);
                let abs_rate = absorption_rate() * transport_scale * (1.0 - saturation) * dt();
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
        } else {
            let beta =
                dt() * uniform_porosity() * uniform_porosity() * mu_fluid()
                    / max(uniform_permeability(), 1e-5);
            let drag = beta * (v_s - v_w)
                / (1.0 + beta * (1.0 / mass_w + 1.0 / mass_s));
            v_w += drag / mass_w;
            v_s -= drag / mass_s;
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
    if solid_mass > 1e-6 {
        atomicStore(&grid[scratch_kind_idx(idx)], CELL_BED_COUPLED);
        divergence_store(idx, 0.0);
        return;
    }

    if mass <= occupancy_mass_threshold() {
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
    // workgroups. The cached mask is generated from the same cell-center SDF
    // probe and avoids repeated texture interpolation in the hot path.
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

    let gv_w = grid_vel[idx];
    if gv_w.w > 1e-6 {
        let v_w = project_grid_velocity(gv_w.xyz, cell_pos, sdf_val, n, bmin, bmax);
        grid_vel[idx] = vec4<f32>(v_w, gv_w.w);
    }

    let solid_idx = grid_vel_solid_idx(idx);
    let gv_s = grid_vel[solid_idx];
    if gv_s.w > 1e-6 {
        var v_s = project_grid_velocity(gv_s.xyz, cell_pos, sdf_val, n, bmin, bmax);
        v_s = project_solid_velocity_against_filter(v_s, cell_pos, idx);
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
    var local_grid_mass = 0.0;
    var local_fluid_mass = 0.0;
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
                if is_bed {
                    local_fluid_mass += w * grid_vel[ci].w;
                }

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

    var J_new = 1.0;
    var dry_floor_strength = 0.0;
    if is_bed {
        let F_old0 = bed_F_col0(pid);
        let F_old1 = bed_F_col1(pid);
        let F_old2 = bed_F_col2(pid);
        let compaction_old = bed_compaction(pid);
        let settled_old = bed_settled_activation(pid);
        let rest_porosity = bed_rest_porosity(pid);
        let rest_perm = bed_rest_permeability(pid);
        let saturation = bed_particle_saturation(pid);
        let F_old = mat3x3<f32>(F_old0, F_old1, F_old2);
        let F_step = mat3x3<f32>(
            vec3<f32>(1.0 + dt() * new_C0.x, dt() * new_C0.y, dt() * new_C0.z),
            vec3<f32>(dt() * new_C1.x, 1.0 + dt() * new_C1.y, dt() * new_C1.z),
            vec3<f32>(dt() * new_C2.x, dt() * new_C2.y, 1.0 + dt() * new_C2.z),
        );
        var F_new = F_step * F_old;
        var F_new0 = F_new[0];
        var F_new1 = F_new[1];
        var F_new2 = F_new[2];
        let detF_raw = max(determinant_from_cols(F_new0, F_new1, F_new2), 1e-5);
        let detF_clamped = clamp(detF_raw, 0.85, 1.45);
        if abs(detF_clamped - detF_raw) > 1e-5 {
            let scale = pow(detF_clamped / detF_raw, 1.0 / 3.0);
            F_new0 *= scale;
            F_new1 *= scale;
            F_new2 *= scale;
            F_new = mat3x3<f32>(F_new0, F_new1, F_new2);
        }
        let support_factor = compaction_support_factor(support_ratio);
        let compression_factor = compaction_compression_factor(detF_clamped);
        let dryness_factor = compaction_dryness_factor(saturation);
        let motion_factor = compaction_motion_factor(new_v.y);
        let rest_factor = compaction_rest_factor(length(new_v));
        let phase_factor = dry_settle_compaction_window();
        let compaction_drive = max(motion_factor, 0.75 * rest_factor);
        let delta_compaction = min(
            dt() * 1.10 * support_factor * compression_factor * dryness_factor * compaction_drive
                * phase_factor,
            0.01,
        );
        let compaction_new = clamp(compaction_old + delta_compaction, 0.0, 0.12);
        let raw_contact_strength =
            raw_dry_contact_strength(saturation, support_ratio, local_grid_mass);
        let near_rest_support = settled_rest_factor(length(new_v));
        let static_support =
            settled_support_ratio_factor(support_ratio) * near_rest_support;
        let packed_support = raw_contact_strength * max(
            smoothstep(0.004, 0.016, compaction_new),
            0.55 * static_support,
        );
        let settled_new = update_settled_activation(
            settled_old,
            saturation,
            compaction_new,
            support_ratio,
            local_grid_mass,
            local_fluid_mass,
            length(new_v),
            packed_support,
        );
        dry_floor_strength = packed_support;
        let support_hold = max(packed_support, raw_contact_strength * static_support);
        let relax_activation = max(0.85 * support_hold, 0.45 * raw_contact_strength * near_rest_support);
        let relax_factor = min(dt() * 3.2 * relax_activation, 0.12);
        if relax_factor > 0.0 {
            F_new0 = mix(F_new0, vec3<f32>(1.0, 0.0, 0.0), relax_factor);
            F_new1 = mix(F_new1, vec3<f32>(0.0, 1.0, 0.0), relax_factor);
            F_new2 = mix(F_new2, vec3<f32>(0.0, 0.0, 1.0), relax_factor);
            let det_relaxed_raw = max(determinant_from_cols(F_new0, F_new1, F_new2), 1e-5);
            let det_relaxed_clamped = clamp(det_relaxed_raw, 0.85, 1.45);
            if abs(det_relaxed_clamped - det_relaxed_raw) > 1e-5 {
                let scale = pow(det_relaxed_clamped / det_relaxed_raw, 1.0 / 3.0);
                F_new0 *= scale;
                F_new1 *= scale;
                F_new2 *= scale;
            }
        }
        // A dry supported pile should shed residual affine flow as it comes to
        // rest instead of continuing to circulate APIC shear internally.
        let affine_rest_damping = min(dt() * 24.0 * relax_activation, 0.94);
        if affine_rest_damping > 0.0 {
            new_C0 *= 1.0 - affine_rest_damping;
            new_C1 *= 1.0 - affine_rest_damping;
            new_C2 *= 1.0 - affine_rest_damping;
        }
        if support_hold > 1e-4 {
            let support_damping = min(dt() * 8.0 * support_hold, 0.24);
            new_v.x *= 1.0 - support_damping;
            new_v.z *= 1.0 - support_damping;
            if new_v.y < 0.0 {
                let vertical_hold = min(dt() * 14.0 * support_hold, 0.40);
                new_v.y *= 1.0 - vertical_hold;
            }
        }
        if packed_support > 1e-4 {
            let support_damping = min(dt() * 6.0 * packed_support, 0.16);
            new_v.x *= 1.0 - support_damping;
            new_v.z *= 1.0 - support_damping;
            if new_v.y < 0.0 {
                let vertical_hold = min(dt() * 12.0 * packed_support, 0.30);
                new_v.y *= 1.0 - vertical_hold;
            }
        }
        let rest_damping = min(dt() * 12.0 * relax_activation, 0.34);
        if rest_damping > 0.0 {
            new_v *= 1.0 - rest_damping;
            if length(new_v) < 0.06 && relax_activation > 0.22 {
                new_v = vec3<f32>(0.0);
                new_C0 = vec3<f32>(0.0);
                new_C1 = vec3<f32>(0.0);
                new_C2 = vec3<f32>(0.0);
            }
        }
        let cone_hold_activation =
            smoothstep(0.01, 0.03, compaction_new) * smoothstep(0.10, 0.35, settled_new);
        if has_filter() && support_hold > 1e-4 && cone_hold_activation > 1e-4
            && xp.y >= -3.0 && xp.y <= 3.0 {
            let cone_support = clamp(0.85 * support_hold * cone_hold_activation, 0.0, 1.0);
            new_v = project_dry_support_velocity(
                new_v,
                vec3<f32>(0.0),
                static_cone_support_normal(xp),
                cone_support,
                true,
            );
        }
        var be = bed_extract[pid];
        be.mech0 = vec4<f32>(F_new0, compaction_new);
        be.mech1 = vec4<f32>(F_new1, rest_porosity);
        be.mech2 = vec4<f32>(F_new2, rest_perm);
        be.bed.y = runtime_porosity(rest_porosity, compaction_new);
        be.bed.z = runtime_permeability(rest_perm, compaction_new);
        be.extract.z = settled_new;
        be.extract.w = bed_particle_saturation(pid);
        bed_extract[pid] = be;
        J_new = clamp(determinant_from_cols(F_new0, F_new1, F_new2), 0.85, 1.45);
    }

    // Advect
    var new_pos = xp + new_v * dt();

    // Particle-level boundary projection closes the gap left by the grid-only
    // collision pass so the dripper wall behaves like a hard barrier.
    let mid_pos = mix(xp, new_pos, 0.5);
    var contact = resolve_sdf_contact(mid_pos, new_v, is_bed, dry_floor_strength);
    new_v = contact.vel;
    contact = resolve_sdf_contact(new_pos, new_v, is_bed, dry_floor_strength);
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
    let saturation = bed_particle_saturation(u32(bed_idx));
    let capacity = max(bed_particle_capacity(u32(bed_idx)) - be.bed.x, 0.0);
    if capacity <= 1e-6 {
        return;
    }

    let mass_p = particles[pid].vel.w;
    let permeability_scale = clamp(
        bed_particle_permeability(u32(bed_idx)) / max(uniform_permeability(), 1e-5),
        0.05,
        1.8,
    );
    let abs_rate = absorption_rate() * permeability_scale * (1.0 - saturation) * dt();
    let speed = length(particles[pid].vel.xyz);
    var absorbed = min(min(mass_p * clamp(abs_rate, 0.0, 0.25), mass_p * 0.5), capacity);
    let remaining_after_partial = mass_p - absorbed;
    let retire_threshold = nominal_mass() * 0.22;
    if remaining_after_partial > 0.0 && remaining_after_partial <= retire_threshold {
        if capacity >= mass_p {
            absorbed = mass_p;
        } else {
            let safe_partial = max(mass_p - inactive_mass_threshold(), 0.0);
            absorbed = min(min(absorbed, safe_partial), capacity);
        }
    } else if saturation > 0.55 && speed < 1.35 {
        let almost_absorbed = min(mass_p, capacity);
        if mass_p - almost_absorbed <= nominal_mass() * 0.35 {
            if capacity >= mass_p {
                absorbed = mass_p;
            } else {
                let safe_partial = max(mass_p - inactive_mass_threshold(), 0.0);
                absorbed = min(min(absorbed, safe_partial), capacity);
            }
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
fn bed_redistribute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x;
    if bid >= num_bed() { return; }

    let self_pos = particles[bid].pos.xyz;
    let self_pore = bed_extract[bid].bed.x;
    let self_perm = bed_particle_permeability(bid);
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
    var best_neighbor_perm = 0.0;
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
                let neighbor_capacity = max(bed_particle_capacity(nid) - neighbor_pore, 0.0);
                if neighbor_capacity <= 1e-6 { continue; }
                let neighbor_perm = bed_particle_permeability(nid);

                let pore_gradient = self_pore - neighbor_pore;
                if pore_gradient <= 1e-4 { continue; }

                let downward_bias = max(self_pos.y - neighbor_pos.y, 0.0);
                let perm_bridge =
                    clamp(min(self_perm, neighbor_perm) / max(uniform_permeability(), 1e-5), 0.05, 2.0);
                let score = pore_gradient * perm_bridge + downward_bias * 0.15;
                if score > best_score {
                    best_score = score;
                    best_neighbor = neighbor_id;
                    best_neighbor_pore = neighbor_pore;
                    best_neighbor_capacity = neighbor_capacity;
                    best_neighbor_perm = neighbor_perm;
                }
            }
        }
    }

    if best_neighbor < 0 { return; }

    let pore_gradient = max(self_pore - best_neighbor_pore, 0.0);
    let transfer_scale =
        clamp(min(self_perm, best_neighbor_perm) / max(uniform_permeability(), 1e-5), 0.04, 1.6);
    let transfer = min(
        min(
            self_pore * bed_storage_redistribution_rate() * transfer_scale * dt(),
            best_neighbor_capacity * 0.2,
        ),
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
    let local_capacity = bed_particle_capacity(bid);
    if abs(absorbed) > 0.0 {
        be.bed.x = clamp(be.bed.x + absorbed, 0.0, local_capacity);
    }
    be.extract.w = be.bed.x / local_capacity;
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
        render_data[pid].primary = vec4<f32>(0.0, -1e6, 0.0, -999.0);
        render_data[pid].aux = vec4<f32>(0.0);
        return;
    }

    var color_t = 0.0;
    var radius = water_particle_radius();
    if phase < 0.5 {
        let speed = length(p.vel.xyz);
        color_t = clamp(speed / 10.0, 0.0, 2.0);
    } else {
        let bed_idx = pid;
        var sat = 0.0;
        if bed_idx < num_bed() {
            sat = bed_particle_saturation(bed_idx);
            radius = bed_particle_render_radius(bed_idx);
        }
        color_t = -1.0 - sat;
    }

    render_data[pid].primary = vec4<f32>(p.pos.xyz, color_t);
    render_data[pid].aux = vec4<f32>(radius, 0.0, 0.0, 0.0);
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

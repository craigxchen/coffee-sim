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

struct BedExtract {
    bed: vec4<f32>,
    extract: vec4<f32>,
};

struct RenderParticle {
    data0: vec4<f32>,
    data1: vec4<f32>,
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
@group(0) @binding(6) var<storage, read_write> render_data: array<RenderParticle>;
@group(0) @binding(7) var<storage, read_write> bed_extract: array<BedExtract>;
@group(0) @binding(8) var<storage, read_write> bed_lookup: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> bed_delta: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> metrics: array<atomic<u32>>;
@group(0) @binding(11) var sdf_class_tex: texture_3d<u32>;

// Metrics slot layout — keep in sync with `METRICS_SLOT_COUNT` in state.rs.
const OBSTACLE_WALL_THICKNESS: f32 = 0.4;
const METRIC_MAX_ABS_DIV_IDX: u32 = 0u;
const METRIC_FLUID_CELLS_IDX: u32 = 1u;
const METRIC_DIV_CLAMP_FIRES_IDX: u32 = 2u;
const METRIC_PRESSURE_CLAMP_FIRES_IDX: u32 = 3u;
const METRIC_MASS_OVERFLOW_FIRES_IDX: u32 = 4u;
const BED_DELTA_WATER_LANE: u32 = 0u;
const BED_DELTA_IMPULSE_X_LANE: u32 = 1u;
const BED_DELTA_IMPULSE_Y_LANE: u32 = 2u;
const BED_DELTA_IMPULSE_Z_LANE: u32 = 3u;
const BED_REACTION_ALPHA: f32 = 0.04;
const BED_REACTION_IMPULSE_CAP: f32 = 0.012;

// ── Helpers ──

fn gx() -> u32 { return u.grid_dims.x; }
fn gy() -> u32 { return u.grid_dims.y; }
fn gz() -> u32 { return u.grid_dims.z; }
fn total_cells() -> u32 { return u.grid_dims.w; }
fn num_bed() -> u32 { return u.counts.y; }
fn max_particles() -> u32 { return u.counts.z; }
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
fn dripper_outlet_radius() -> f32 { return u.fp_params.w; }
fn dripper_top_radius() -> f32 { return dripper_outlet_radius() + 4.2634315; }
fn sdf_res() -> f32 { return u.sdf_params.x; }
fn friction() -> f32 { return u.sdf_params.y; }
fn restitution() -> f32 { return u.sdf_params.z; }
fn contact_offset() -> f32 { return u.sdf_params.w; }
fn obstacle_wall_half_thickness() -> f32 { return OBSTACLE_WALL_THICKNESS * 0.5; }
fn cup_floor_y() -> f32 { return -8.0 + obstacle_wall_half_thickness() + contact_offset(); }
fn water_kinematic_viscosity_m2_s() -> f32 { return u.bed_params.x; }
fn absorption_rate() -> f32 { return u.bed_params.y; }
fn max_saturation() -> f32 { return u.bed_params.z; }
fn min_bed_permeability_m2() -> f32 { return u.bed_params.w; }
fn extraction_rate() -> f32 { return u.extraction_params.x; }
fn bed_compaction_rate() -> f32 { return u.extraction_params.y; }
fn bed_damping() -> f32 { return u.extraction_params.z; }
fn bed_impact() -> f32 { return u.extraction_params.w; }
fn inactive_mass_threshold() -> f32 { return nominal_mass() * 0.10; }
fn div_clamp_limit() -> f32 { return u.clamp_params.x; }
fn pressure_clamp_limit() -> f32 { return u.clamp_params.y; }
fn metrics_div_fp_scale() -> f32 { return u.clamp_params.z; }
fn metrics_div_inv_fp_scale() -> f32 { return u.clamp_params.w; }
fn projection_j_alpha() -> f32 { return u.projection_params.x; }
fn projection_j_expand_alpha() -> f32 { return u.projection_params.y; }
fn projection_max_rest_volume_fraction() -> f32 { return u.projection_params.z; }
fn bed_surface_void_scale() -> f32 { return u.projection_params.w; }
fn bed_pore_capacity_scale() -> f32 { return u.time_params.z; }
fn bed_pore_overfill_alpha() -> f32 { return u.time_params.w; }
fn water_particle_radius() -> f32 { return dx() * u.inflow_params.y; }
fn bed_particle_radius() -> f32 { return dx() * u.inflow_params.z; }
fn filter_absorption_rate() -> f32 { return u.inflow_params.w; }
fn min_particle_j() -> f32 { return 0.40; }
fn max_particle_j() -> f32 { return 2.00; }
fn clamp_particle_j(value: f32) -> f32 {
    return clamp(value, min_particle_j(), max_particle_j());
}

fn coffee_filter_floor_y() -> f32 {
    let filter_center_y = -0.35;
    let filter_bot_y = filter_center_y - 3.02;
    let filter_top_y = filter_center_y + 2.75;
    let filter_top_radius = 4.10;
    let filter_thickness = 0.08;
    let bed_contact_offset = max(contact_offset(), bed_particle_radius());
    let filter_height = max(filter_top_y - filter_bot_y, 1e-6);
    let filter_slope = filter_top_radius / filter_height;
    return filter_bot_y + (bed_contact_offset + filter_thickness) / max(filter_slope, 1e-6);
}

fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
    return iz * gx() * gy() + iy * gx() + ix;
}

fn grid_mass_idx(cell: u32) -> u32 { return cell; }
fn grid_mom_x_idx(cell: u32) -> u32 { return total_cells() + cell; }
fn grid_mom_y_idx(cell: u32) -> u32 { return 2u * total_cells() + cell; }
fn grid_mom_z_idx(cell: u32) -> u32 { return 3u * total_cells() + cell; }
fn grid_rest_volume_idx(cell: u32) -> u32 { return 4u * total_cells() + cell; }
fn grid_current_volume_idx(cell: u32) -> u32 { return 5u * total_cells() + cell; }
fn scratch_pressure_idx(cell: u32) -> u32 { return grid_mass_idx(cell); }
fn scratch_div_idx(cell: u32) -> u32 { return grid_mom_x_idx(cell); }
// Slot 2 (`grid_mom_y_idx`) is free after `grid_update` consumes p2g
// momentum. It carries the temporary unilateral packing pressure before
// viscosity reuses the momentum lanes as velocity scratch.
fn scratch_packing_idx(cell: u32) -> u32 { return grid_mom_y_idx(cell); }
fn scratch_kind_idx(cell: u32) -> u32 { return grid_mom_z_idx(cell); }
// A quadratic-B-spline particle deposits at most `nominal_mass * 0.75^3 ≈
// 0.42 * nominal_mass` to its peak cell. The threshold must stay strictly
// below that peak or isolated particles never register as fluid. Matching
// `inactive_mass_threshold()` at 0.1 * nominal_mass means "enough mass to
// still exist" ⇔ "enough mass to produce a fluid cell", which is
// semantically consistent and keeps ghost-splat noise below the bar.
fn occupancy_mass_threshold() -> f32 { return nominal_mass() * 0.1; }
fn viscosity_support_mass_threshold() -> f32 { return nominal_mass() * 2.0; }

const CELL_AIR: i32 = 0;
const CELL_SURFACE_FLUID: i32 = 1;
const CELL_INTERIOR_FLUID: i32 = 2;
const CELL_BED_COUPLED: i32 = 3;
const CELL_SOLID: i32 = 4;
const PHASE_WATER_MAX: f32 = 0.5;
const PHASE_SUSPENDED_COFFEE: f32 = 2.0;

fn is_water_phase(phase: f32) -> bool {
    return phase < PHASE_WATER_MAX;
}

fn is_anchored_coffee_phase(phase: f32) -> bool {
    return phase >= PHASE_WATER_MAX && phase < 1.5;
}

fn is_suspended_coffee_phase(phase: f32) -> bool {
    return phase >= 1.5;
}

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

fn packing_pressure_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[scratch_packing_idx(cell)])) * inv_fp_scale();
}

fn packing_pressure_store(cell: u32, value: f32) {
    let limit = pressure_clamp_limit();
    let clamped = clamp(value, 0.0, limit);
    if clamped != value {
        atomicAdd(&metrics[METRIC_PRESSURE_CLAMP_FIRES_IDX], 1u);
    }
    atomicStore(&grid[scratch_packing_idx(cell)], i32(clamped * fp_scale()));
}

fn pressure_or_mirror(cell: vec3<i32>, mirror_pressure: f32) -> f32 {
    if cell.x < 0 || cell.y < 0 || cell.z < 0 {
        return mirror_pressure;
    }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() {
        return mirror_pressure;
    }
    let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    if is_solid_kind(cell_kind_load(ci)) {
        return mirror_pressure;
    }
    return pressure_load(ci);
}

fn pressure_gradient_at_cell(cell: vec3<i32>) -> vec3<f32> {
    if cell.x < 0 || cell.y < 0 || cell.z < 0 {
        return vec3<f32>(0.0);
    }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() {
        return vec3<f32>(0.0);
    }

    let idx = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    let p_here = pressure_load(idx);
    let p_xm = pressure_or_mirror(cell + vec3<i32>(-1, 0, 0), p_here);
    let p_xp = pressure_or_mirror(cell + vec3<i32>(1, 0, 0), p_here);
    let p_ym = pressure_or_mirror(cell + vec3<i32>(0, -1, 0), p_here);
    let p_yp = pressure_or_mirror(cell + vec3<i32>(0, 1, 0), p_here);
    let p_zm = pressure_or_mirror(cell + vec3<i32>(0, 0, -1), p_here);
    let p_zp = pressure_or_mirror(cell + vec3<i32>(0, 0, 1), p_here);

    return 0.5 * inv_dx() * vec3<f32>(
        p_xp - p_xm,
        p_yp - p_ym,
        p_zp - p_zm,
    );
}

fn divergence_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[scratch_div_idx(cell)])) * inv_fp_scale();
}

fn velocity_scratch_load(cell: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(atomicLoad(&grid[grid_mom_x_idx(cell)])) * inv_fp_scale(),
        f32(atomicLoad(&grid[grid_mom_y_idx(cell)])) * inv_fp_scale(),
        f32(atomicLoad(&grid[grid_mom_z_idx(cell)])) * inv_fp_scale(),
    );
}

fn velocity_scratch_store(cell: u32, value: vec3<f32>) {
    var clamped = value;
    let speed = length(clamped);
    if speed > vel_cap() {
        clamped = clamped * (vel_cap() / speed);
    }
    atomicStore(&grid[grid_mom_x_idx(cell)], i32(clamped.x * fp_scale()));
    atomicStore(&grid[grid_mom_y_idx(cell)], i32(clamped.y * fp_scale()));
    atomicStore(&grid[grid_mom_z_idx(cell)], i32(clamped.z * fp_scale()));
}

fn rest_volume_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[grid_rest_volume_idx(cell)])) * inv_fp_scale();
}

fn current_volume_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[grid_current_volume_idx(cell)])) * inv_fp_scale();
}

fn liquid_fill_fraction(cell: u32, kind: i32) -> f32 {
    if kind == CELL_INTERIOR_FLUID || kind == CELL_BED_COUPLED {
        return 1.0;
    }
    if kind != CELL_SURFACE_FLUID {
        return 0.0;
    }

    let cell_volume = max(dx() * dx() * dx(), 1e-8);
    let deposited_fraction = max(rest_volume_load(cell), current_volume_load(cell)) / cell_volume;
    return clamp(deposited_fraction, 0.0, 1.0);
}

fn pressure_face_weight(
    self_kind: i32,
    self_fill: f32,
    neighbor_cell: u32,
    neighbor_kind: i32,
) -> f32 {
    if is_solid_kind(neighbor_kind) {
        return 0.0;
    }
    if self_kind == CELL_BED_COUPLED || neighbor_kind == CELL_BED_COUPLED {
        return 1.0;
    }
    if !is_fluid_kind(neighbor_kind) {
        return self_fill;
    }

    return min(self_fill, liquid_fill_fraction(neighbor_cell, neighbor_kind));
}

fn pressure_weighted_or_mirror(
    cell: vec3<i32>,
    mirror_pressure: f32,
    self_kind: i32,
    self_fill: f32,
) -> f32 {
    if cell.x < 0 || cell.y < 0 || cell.z < 0 {
        return mirror_pressure;
    }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() {
        return mirror_pressure;
    }

    let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    let kind = cell_kind_load(ci);
    if is_solid_kind(kind) {
        return mirror_pressure;
    }

    var neighbor_pressure = 0.0;
    if is_fluid_kind(kind) {
        neighbor_pressure = pressure_load(ci);
    }
    let face_weight = pressure_face_weight(self_kind, self_fill, ci, kind);
    return mix(mirror_pressure, neighbor_pressure, face_weight);
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

fn is_valid_bed_solid_idx(bed_idx: i32) -> bool {
    if bed_idx < 0 || u32(bed_idx) >= num_bed() {
        return false;
    }
    return !is_water_phase(affine[u32(bed_idx)].col0.w);
}

fn bed_velocity_load(cell: u32) -> vec3<f32> {
    let bed_idx = bed_lookup_load(cell);
    if !is_valid_bed_solid_idx(bed_idx) {
        return vec3<f32>(0.0);
    }
    return particles[u32(bed_idx)].vel.xyz;
}

fn bed_matrix_velocity_cell(cell: vec3<i32>) -> vec3<f32> {
    var vel_sum = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var dk = -1; dk <= 1; dk++) {
                let c = cell + vec3<i32>(di, dj, dk);
                if c.x < 0 || c.y < 0 || c.z < 0 { continue; }
                if u32(c.x) >= gx() || u32(c.y) >= gy() || u32(c.z) >= gz() { continue; }

                let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
                let bed_idx = bed_lookup_load(ci);
                if !is_valid_bed_solid_idx(bed_idx) {
                    continue;
                }

                vel_sum += particles[u32(bed_idx)].vel.xyz;
                weight_sum += 1.0;
            }
        }
    }

    if weight_sum <= 0.0 {
        return vec3<f32>(0.0);
    }
    return vel_sum / weight_sum;
}

fn bed_velocity_load_cell(cell: vec3<i32>) -> vec3<f32> {
    if cell.x < 0 || cell.y < 0 || cell.z < 0 {
        return vec3<f32>(0.0);
    }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() {
        return vec3<f32>(0.0);
    }
    return bed_matrix_velocity_cell(cell);
}

fn bed_velocity_divergence(cell: vec3<i32>) -> f32 {
    let v_xm = bed_velocity_load_cell(cell + vec3<i32>(-1, 0, 0)).x;
    let v_xp = bed_velocity_load_cell(cell + vec3<i32>(1, 0, 0)).x;
    let v_ym = bed_velocity_load_cell(cell + vec3<i32>(0, -1, 0)).y;
    let v_yp = bed_velocity_load_cell(cell + vec3<i32>(0, 1, 0)).y;
    let v_zm = bed_velocity_load_cell(cell + vec3<i32>(0, 0, -1)).z;
    let v_zp = bed_velocity_load_cell(cell + vec3<i32>(0, 0, 1)).z;
    return 0.5 * inv_dx() * ((v_xp - v_xm) + (v_yp - v_ym) + (v_zp - v_zm));
}

fn kozeny_porosity_factor(phi_in: f32) -> f32 {
    let phi = clamp(phi_in, 0.08, 0.82);
    let solid = max(1.0 - phi, 1e-4);
    return (phi * phi * phi) / (solid * solid);
}

fn bed_compacted_porosity(bed_idx: u32) -> f32 {
    let be = bed_extract[bed_idx];
    let base_porosity = clamp(be.bed.y, 0.18, 0.70);
    let compaction = clamp(be.bed.w, 0.0, 2.25);
    return clamp(base_porosity - compaction * 0.12, 0.16, base_porosity);
}

fn bed_compacted_permeability(bed_idx: u32) -> f32 {
    let be = bed_extract[bed_idx];
    let base_porosity = clamp(be.bed.y, 0.18, 0.70);
    let compacted_porosity = bed_compacted_porosity(bed_idx);
    let ratio = kozeny_porosity_factor(compacted_porosity)
        / max(kozeny_porosity_factor(base_porosity), 1e-8);
    return max(be.bed.z * ratio, min_bed_permeability_m2());
}

fn bed_delta_idx(lane: u32, bed_idx: u32) -> u32 {
    return lane * max_particles() + bed_idx;
}

fn bed_water_delta_add(bed_idx: u32, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    atomicAdd(&bed_delta[bed_delta_idx(BED_DELTA_WATER_LANE, bed_idx)], i32(amount * fp_scale()));
}

fn bed_impulse_delta_add(bed_idx: u32, impulse: vec3<f32>) {
    let cap = nominal_mass() * vel_cap() * BED_REACTION_IMPULSE_CAP;
    let clamped = clamp(impulse, vec3<f32>(-cap), vec3<f32>(cap));
    atomicAdd(
        &bed_delta[bed_delta_idx(BED_DELTA_IMPULSE_X_LANE, bed_idx)],
        i32(clamped.x * fp_scale()),
    );
    atomicAdd(
        &bed_delta[bed_delta_idx(BED_DELTA_IMPULSE_Y_LANE, bed_idx)],
        i32(clamped.y * fp_scale()),
    );
    atomicAdd(
        &bed_delta[bed_delta_idx(BED_DELTA_IMPULSE_Z_LANE, bed_idx)],
        i32(clamped.z * fp_scale()),
    );
}

fn bed_impulse_delta_add_neighborhood(cell: vec3<i32>, impulse: vec3<f32>) {
    var sample_count = 0.0;
    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var dk = -1; dk <= 1; dk++) {
                let c = cell + vec3<i32>(di, dj, dk);
                if c.x < 0 || c.y < 0 || c.z < 0 { continue; }
                if u32(c.x) >= gx() || u32(c.y) >= gy() || u32(c.z) >= gz() { continue; }

                let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
                if is_valid_bed_solid_idx(bed_lookup_load(ci)) {
                    sample_count += 1.0;
                }
            }
        }
    }
    if sample_count <= 0.0 {
        return;
    }

    let impulse_share = impulse / sample_count;
    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var dk = -1; dk <= 1; dk++) {
                let c = cell + vec3<i32>(di, dj, dk);
                if c.x < 0 || c.y < 0 || c.z < 0 { continue; }
                if u32(c.x) >= gx() || u32(c.y) >= gy() || u32(c.z) >= gz() { continue; }

                let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
                let bed_idx = bed_lookup_load(ci);
                if is_valid_bed_solid_idx(bed_idx) {
                    bed_impulse_delta_add(u32(bed_idx), impulse_share);
                }
            }
        }
    }
}

fn bed_impulse_delta_exchange(bed_idx: u32) -> vec3<f32> {
    let ix = atomicExchange(&bed_delta[bed_delta_idx(BED_DELTA_IMPULSE_X_LANE, bed_idx)], 0);
    let iy = atomicExchange(&bed_delta[bed_delta_idx(BED_DELTA_IMPULSE_Y_LANE, bed_idx)], 0);
    let iz = atomicExchange(&bed_delta[bed_delta_idx(BED_DELTA_IMPULSE_Z_LANE, bed_idx)], 0);
    return vec3<f32>(f32(ix), f32(iy), f32(iz)) * inv_fp_scale();
}

fn deposit_absorbed_bed_water(home_cell: vec3<i32>, home_bed_idx: i32, absorbed: f32) {
    let neighbor_share = absorbed * 0.35;
    let home_share = absorbed - neighbor_share;
    bed_water_delta_add(u32(home_bed_idx), home_share);

    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0),
        vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1),
        vec3<i32>(0, 0, 1),
    );
    var neighbor_ids: array<i32, 6>;
    var neighbor_count = 0u;

    for (var i = 0u; i < 6u; i++) {
        let c = home_cell + offsets[i];
        if c.x < 0 || c.y < 0 || c.z < 0 {
            continue;
        }
        if u32(c.x) >= gx() || u32(c.y) >= gy() || u32(c.z) >= gz() {
            continue;
        }

        let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
        let neighbor_id = bed_lookup_load(ci);
        if neighbor_id < 0 || neighbor_id == home_bed_idx || u32(neighbor_id) >= num_bed() {
            continue;
        }

        var duplicate = false;
        for (var j = 0u; j < neighbor_count; j++) {
            if neighbor_ids[j] == neighbor_id {
                duplicate = true;
            }
        }
        if !duplicate {
            neighbor_ids[neighbor_count] = neighbor_id;
            neighbor_count += 1u;
        }
    }

    if neighbor_count == 0u {
        bed_water_delta_add(u32(home_bed_idx), neighbor_share);
        return;
    }

    let each_neighbor_share = neighbor_share / f32(neighbor_count);
    for (var i = 0u; i < neighbor_count; i++) {
        bed_water_delta_add(u32(neighbor_ids[i]), each_neighbor_share);
    }
}

fn sdf_class_is_solid(cell: vec3<i32>) -> bool {
    return textureLoad(sdf_class_tex, cell, 0).r != 0u;
}

fn is_fluid_kind(kind: i32) -> bool {
    // Surface and bed-coupled cells must participate in the pressure solve so
    // hydrostatic pressure can build up in shallow puddles and water inside the
    // bed remains part of the incompressible solve. Surface/air faces are
    // weighted by the cell's deposited liquid fraction in `pressure_update`
    // instead of being treated as full-cell Dirichlet p=0 faces.
    return kind == CELL_INTERIOR_FLUID
        || kind == CELL_SURFACE_FLUID
        || kind == CELL_BED_COUPLED;
}

fn is_solid_kind(kind: i32) -> bool {
    return kind == CELL_SOLID;
}

fn volume_projection_target_divergence(rest_volume: f32, current_volume: f32) -> f32 {
    let cell_volume = dx() * dx() * dx();
    let rest_volume_eps = max(cell_volume * 1e-5, p_vol() * 1e-3);
    if rest_volume <= rest_volume_eps {
        return 0.0;
    }

    let j_cell = current_volume / rest_volume;
    let compressed_error = clamp(1.0 - j_cell, 0.0, 0.75);
    let expanded_error = clamp(j_cell - 1.0, 0.0, 0.75);
    var target_div = compressed_error * projection_j_alpha()
        - expanded_error * projection_j_expand_alpha();

    // A cell can have an acceptable per-particle J yet contain too much
    // material because many particle kernels overlap there. Treat overpacked
    // cells as an independent positive expansion target, including cells whose
    // current volume is already above rest volume.
    let packed_fraction = max(rest_volume, current_volume) / max(cell_volume, 1e-8);
    let overpack_error = max(packed_fraction - projection_max_rest_volume_fraction(), 0.0);
    target_div = max(target_div, overpack_error * projection_j_alpha());

    return target_div;
}

fn bed_pore_projection_target_divergence(
    bed_idx: u32,
    cell_center: vec3<f32>,
    rest_volume: f32,
    current_volume: f32,
) -> f32 {
    var target_div = volume_projection_target_divergence(rest_volume, current_volume);
    let cell_volume = dx() * dx() * dx();
    let be = bed_extract[bed_idx];
    let base_porosity = bed_compacted_porosity(bed_idx);
    let saturation = clamp(be.extract.w, 0.0, 1.0);
    let compaction = clamp(be.bed.w, 0.0, 2.25);
    let bed_center_y = particles[bed_idx].pos.y;
    // `bed_lookup` stamps neighboring cells around each bed sample. Near the
    // upper bed surface, those stamped cells are only partially occupied by
    // grounds; treating them as full porous cells makes the first impact point
    // reject water like a hard plug and produces a hollow wetting ring. Once
    // that local material is wet/compacted, the same geometric opening has less
    // remaining mobile pore volume, so the pressure solve should start forming
    // a head instead of letting the stream keep entering the bed freely.
    let surface_t = smoothstep(-0.25, 0.75, (cell_center.y - bed_center_y) * inv_dx());
    let open_pore_availability = clamp(
        (1.0 - 0.65 * saturation) * (1.0 - 0.12 * compaction),
        0.20,
        1.0,
    );
    let surface_opening =
        surface_t * clamp(bed_surface_void_scale(), 0.0, 1.0) * open_pore_availability;
    let surface_porosity = min(base_porosity + 0.35, 0.82);
    let effective_porosity = mix(base_porosity, surface_porosity, surface_opening);
    let pore_occupancy = mix(1.0, 0.35, surface_opening);
    let saturation_capacity = mix(1.0, 0.32, saturation);
    let compaction_capacity = mix(1.0, 0.58, clamp(compaction / 1.5, 0.0, 1.0));
    let pore_capacity = effective_porosity
        * cell_volume
        * max(bed_pore_capacity_scale(), 0.0)
        * saturation_capacity
        * compaction_capacity;
    let overfill_fraction = max(current_volume - pore_capacity, 0.0) / max(cell_volume, 1e-8);
    let overfill_response = bed_pore_overfill_alpha()
        * pore_occupancy
        * mix(1.0, 1.45, saturation);
    target_div = max(target_div, overfill_fraction * overfill_response);

    return target_div;
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
    // Keep this analytic fallback parallel to `FilterConfig::default()` and
    // `v60_support_cone`; the SDF texture uses the same obstacle radii.
    let cone_top_y = 3.0;
    let cone_bot_y = -3.0;
    if out_pos.y <= cone_top_y && out_pos.y >= cone_bot_y {
        let t = clamp((out_pos.y - cone_bot_y) / (cone_top_y - cone_bot_y), 0.0, 1.0);
        let cone_radius = mix(dripper_outlet_radius(), dripper_top_radius(), t) - contact_offset();
        let cone_contact = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), cone_radius);
        out_pos = cone_contact.pos;
        out_vel = cone_contact.vel;
    }

    // Paper filter (bed particles only): the filter is porous — water passes
    // through the paper, but coffee particles are trapped inside the actual
    // paper interior. This constraint mirrors `FilterConfig::default()`:
    // center.y=-0.35, top_y=2.75, bot_y=-3.02, top_radius=4.10,
    // bot_radius=0.0, thickness=0.08. The rigid V60 support is the truncated
    // cone; the paper itself comes to a tip and protrudes through that opening.
    let filter_center_y = -0.35;
    let filter_top_y = filter_center_y + 2.75;
    let filter_bot_y = filter_center_y - 3.02;
    let filter_top_radius = 4.10;
    let filter_thickness = 0.08;
    let bed_contact_offset = max(contact_offset(), bed_particle_radius());
    let coffee_floor_y = coffee_filter_floor_y();
    if is_bed {
        if out_pos.y <= filter_top_y && out_pos.y >= filter_bot_y {
            let ft = clamp((out_pos.y - filter_bot_y) / (filter_top_y - filter_bot_y), 0.0, 1.0);
            let outer_r = mix(0.0, filter_top_radius, ft);
            let filter_r = max(outer_r - filter_thickness, 0.0) - bed_contact_offset;
            let fc = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), max(filter_r, 0.0));
            out_pos = fc.pos;
            out_vel = fc.vel;
        }

        // Filter apex: the paper comes to a tip, so a finite-size coffee
        // particle cannot sit at the geometric apex; it rests where the inner
        // cone radius can contain its particle radius.
        if out_pos.y <= coffee_floor_y {
            out_pos.y = coffee_floor_y;
            out_pos.x = 0.0;
            out_pos.z = 0.0;
            if out_vel.y < 0.0 {
                out_vel.y = 0.0;
                out_vel.x *= 1.0 - friction() * 0.55;
                out_vel.z *= 1.0 - friction() * 0.55;
            }
            out_vel.x *= 1.0 - friction() * 0.55;
            out_vel.z *= 1.0 - friction() * 0.55;
        }
    }

    // Carafe interior fallback. The primary contact comes from the SDF, whose
    // effective fluid surface is inset by half the obstacle wall thickness. Keep
    // this analytic guard on the same surface so floor/wall contacts cannot
    // create a second visible boundary layer.
    if out_pos.y <= -3.5 {
        let cup_radius = 3.0 - obstacle_wall_half_thickness() - contact_offset();
        let cup_contact = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), cup_radius);
        out_pos = cup_contact.pos;
        out_vel = cup_contact.vel;

        let floor_y = cup_floor_y();
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

fn resolve_coffee_particle_packing(
    position: vec3<f32>,
    velocity: vec3<f32>,
    self_bid: u32,
) -> ContactResult {
    var closest_delta = vec3<f32>(0.0);
    var closest_dist = 1e9;
    var found = false;
    let center_cell = world_to_cell(position);

    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            for (var dk = -1; dk <= 1; dk++) {
                let c = center_cell + vec3<i32>(di, dj, dk);
                if c.x < 0 || c.y < 0 || c.z < 0 { continue; }
                if u32(c.x) >= gx() || u32(c.y) >= gy() || u32(c.z) >= gz() { continue; }

                let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
                let other_bid = bed_lookup_load(ci);
                if other_bid < 0 || u32(other_bid) == self_bid || u32(other_bid) >= num_bed() {
                    continue;
                }

                let delta = position - particles[u32(other_bid)].pos.xyz;
                let dist = length(delta);
                if dist < closest_dist {
                    closest_dist = dist;
                    closest_delta = delta;
                    found = true;
                }
            }
        }
    }

    var out_pos = position;
    var out_vel = velocity;
    let min_sep = bed_particle_radius() * 1.78;
    if found && closest_dist < min_sep {
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if closest_dist > 1e-5 {
            n = closest_delta / closest_dist;
        }

        let penetration = min_sep - closest_dist;
        out_pos += n * min(penetration * 0.65, dx() * 0.22);
        let vn = dot(out_vel, n);
        if vn < 0.0 {
            out_vel -= n * vn;
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

fn filter_paper_absorption_weight(position: vec3<f32>) -> f32 {
    // Thin paper medium. Water overlapping the inner paper face can be taken
    // into the paper reservoir; this is absorption at the filter material
    // rather than a momentum-only damping zone at the outlet.
    let filter_center_y = -0.35;
    let filter_top_y = filter_center_y + 2.75;
    let filter_bot_y = filter_center_y - 3.02;
    let filter_top_radius = 4.10;
    let filter_thickness = 0.08;
    let particle_r = max(water_particle_radius(), dx() * 0.35);

    if position.y < filter_bot_y - particle_r || position.y > filter_top_y + particle_r {
        return 0.0;
    }

    let ft = clamp((position.y - filter_bot_y) / (filter_top_y - filter_bot_y), 0.0, 1.0);
    let outer_r = mix(0.0, filter_top_radius, ft);
    let inner_r = max(outer_r - filter_thickness, 0.0);
    let r = length(position.xz);
    let side_dist = abs(r - inner_r);
    let side_weight = 1.0 - smoothstep(particle_r, particle_r * 3.0, side_dist);

    // The bottom seam of a folded paper filter is still paper, so particles
    // passing through the truncated outlet can be absorbed by the paper lip.
    let dy = position.y - filter_bot_y;
    let lip_vertical = 1.0 - smoothstep(0.0, particle_r * 5.0, max(dy, 0.0));
    let lip_radial = 1.0 - smoothstep(inner_r, inner_r + particle_r * 3.0, r);
    let lip_weight = lip_vertical * lip_radial;

    return clamp(max(side_weight, lip_weight), 0.0, 1.0);
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
    atomicStore(&grid[grid_rest_volume_idx(idx)], 0);
    atomicStore(&grid[grid_current_volume_idx(idx)], 0);
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
    if !is_water_phase(phase) || mass_p <= inactive_mass_threshold() {
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
    let rest_particle_volume = p_vol() * mass_p / max(nominal_mass(), 1e-6);
    let current_particle_volume = rest_particle_volume * clamp_particle_j(J);

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
                let rest_volume_contrib = w * rest_particle_volume;
                let current_volume_contrib = w * current_particle_volume;
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
                let rest_volume_fp = rest_volume_contrib * fp;
                let current_volume_fp = current_volume_contrib * fp;
                let mom_x_fp = mom.x * fp;
                let mom_y_fp = mom.y * fp;
                let mom_z_fp = mom.z * fp;
                if abs(mass_fp) > limit_m || abs(mom_x_fp) > limit_m
                    || abs(mom_y_fp) > limit_m || abs(mom_z_fp) > limit_m
                    || abs(rest_volume_fp) > limit_m || abs(current_volume_fp) > limit_m {
                    atomicAdd(&metrics[METRIC_MASS_OVERFLOW_FIRES_IDX], 1u);
                }
                atomicAdd(&grid[grid_mass_idx(ci)], i32(mass_fp));
                atomicAdd(&grid[grid_mom_x_idx(ci)], i32(mom_x_fp));
                atomicAdd(&grid[grid_mom_y_idx(ci)], i32(mom_y_fp));
                atomicAdd(&grid[grid_mom_z_idx(ci)], i32(mom_z_fp));
                atomicAdd(&grid[grid_rest_volume_idx(ci)], i32(rest_volume_fp));
                atomicAdd(&grid[grid_current_volume_idx(ci)], i32(current_volume_fp));
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

// ── viscosity ──

@compute @workgroup_size(64)
fn viscosity_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let gv = grid_vel[idx];
    if gv.w <= occupancy_mass_threshold() || viscosity() <= 0.0 {
        velocity_scratch_store(idx, gv.xyz);
        return;
    }

    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let v_here = gv.xyz;
    // Viscosity is a bulk-fluid stress. Sparse streams have too little local
    // support for a stable velocity Laplacian, so gate it by local occupancy
    // rather than by scene location.
    if gv.w <= viscosity_support_mass_threshold() {
        velocity_scratch_store(idx, v_here);
        return;
    }

    let alpha = clamp(viscosity() * dt() / max(dx() * dx(), 1e-6), 0.0, 0.14);
    var neighbor_velocity_sum = vec3<f32>(0.0);
    var neighbor_weight_sum = 0.0;
    var fluid_neighbor_count = 0u;
    let wall_viscosity_weight = 1.0;

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
        if neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0
            || u32(neighbor.x) >= gx() || u32(neighbor.y) >= gy() || u32(neighbor.z) >= gz() {
            // No-slip boundary: outside the simulation domain is stationary
            // support for viscous diffusion, not missing fluid support.
            neighbor_weight_sum += wall_viscosity_weight;
            continue;
        }
        if sdf_class_is_solid(neighbor) {
            // No-slip boundary: static solids dissipate tangential pool motion.
            neighbor_weight_sum += wall_viscosity_weight;
            continue;
        }

        let neighbor_idx = cell_index(u32(neighbor.x), u32(neighbor.y), u32(neighbor.z));
        let neighbor_gv = grid_vel[neighbor_idx];
        if neighbor_gv.w <= occupancy_mass_threshold() {
            continue;
        }
        let neighbor_weight = clamp(
            neighbor_gv.w / max(max(gv.w, nominal_mass()), 1e-6),
            0.0,
            1.0,
        );
        neighbor_velocity_sum += neighbor_gv.xyz * neighbor_weight;
        neighbor_weight_sum += neighbor_weight;
        fluid_neighbor_count += 1u;
    }

    if fluid_neighbor_count < 3u || neighbor_weight_sum <= 1e-6 {
        velocity_scratch_store(idx, v_here);
        return;
    }

    let neighbor_average = neighbor_velocity_sum / neighbor_weight_sum;
    let blend = clamp(alpha * neighbor_weight_sum, 0.0, 0.65);
    velocity_scratch_store(idx, mix(v_here, neighbor_average, blend));
}

@compute @workgroup_size(64)
fn viscosity_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let mass = grid_vel[idx].w;
    if mass <= occupancy_mass_threshold() {
        grid_vel[idx] = vec4<f32>(0.0);
        return;
    }

    grid_vel[idx] = vec4<f32>(velocity_scratch_load(idx), mass);
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
    let rest_volume = rest_volume_load(idx);
    let current_volume = current_volume_load(idx);
    var target_divergence = volume_projection_target_divergence(rest_volume, current_volume);
    if kind == CELL_BED_COUPLED {
        let bed_idx = bed_lookup_load(idx);
        if bed_idx >= 0 && u32(bed_idx) < num_bed() {
            let pore_target = bed_pore_projection_target_divergence(
                u32(bed_idx),
                cell_center,
                rest_volume,
                current_volume,
            );
            let porosity = clamp(bed_compacted_porosity(u32(bed_idx)), 0.08, 1.0);
            let solid_fraction = 1.0 - porosity;
            let solid_div = bed_velocity_divergence(vec3<i32>(
                i32(ix_val),
                i32(iy_val),
                i32(iz_val),
            ));
            target_divergence = (pore_target - solid_fraction * solid_div) / porosity;
        }
    }
    divergence_store(idx, div - target_divergence);

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
    let self_fill = liquid_fill_fraction(idx, kind);
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

        let face_weight = pressure_face_weight(kind, self_fill, neighbor_idx, neighbor_kind);
        if face_weight <= 0.0 {
            continue;
        }

        neighbor_count += face_weight;
        // Air neighbor → weighted Dirichlet p=0, contributes 0 to
        // pressure_sum. Fluid neighbor → weighted pressure contribution.
        if is_fluid_kind(neighbor_kind) {
            pressure_sum += face_weight * pressure_load(neighbor_idx);
        }
    }

    if neighbor_count <= 0.0 {
        pressure_store(idx, 0.0);
        return;
    }

    let rhs = divergence_load(idx) * self_fill / max(dt(), 1e-6);
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
    let self_fill = liquid_fill_fraction(idx, kind);
    let cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));
    // Solid/off-grid neighbors keep the mirror pressure, while air and fluid
    // faces blend toward the neighboring pressure by the liquid face weight.
    let p_xm = pressure_weighted_or_mirror(
        cell + vec3<i32>(-1, 0, 0),
        p_here,
        kind,
        self_fill,
    );
    let p_xp = pressure_weighted_or_mirror(
        cell + vec3<i32>(1, 0, 0),
        p_here,
        kind,
        self_fill,
    );
    let p_ym = pressure_weighted_or_mirror(
        cell + vec3<i32>(0, -1, 0),
        p_here,
        kind,
        self_fill,
    );
    let p_yp = pressure_weighted_or_mirror(
        cell + vec3<i32>(0, 1, 0),
        p_here,
        kind,
        self_fill,
    );
    let p_zm = pressure_weighted_or_mirror(
        cell + vec3<i32>(0, 0, -1),
        p_here,
        kind,
        self_fill,
    );
    let p_zp = pressure_weighted_or_mirror(
        cell + vec3<i32>(0, 0, 1),
        p_here,
        kind,
        self_fill,
    );

    let grad_p = 0.5 * inv_dx() * vec3<f32>(
        p_xp - p_xm,
        p_yp - p_ym,
        p_zp - p_zm,
    );

    var v = gv.xyz - dt() * grad_p;
    if kind == CELL_BED_COUPLED {
        let bed_idx = bed_lookup_load(idx);
        if bed_idx >= 0 && u32(bed_idx) < num_bed() {
            let permeability_m2 = bed_compacted_permeability(u32(bed_idx));
            let darcy_rate = water_kinematic_viscosity_m2_s() / permeability_m2;
            let darcy_damping = 1.0 / (1.0 + max(darcy_rate * dt(), 0.0));
            let bed_cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));
            let bed_v = bed_matrix_velocity_cell(bed_cell);
            let v_before_darcy = v;
            v = bed_v + (v - bed_v) * darcy_damping;
            let water_impulse = gv.w * (v - v_before_darcy);
            bed_impulse_delta_add_neighborhood(bed_cell, -water_impulse * BED_REACTION_ALPHA);
        }
    }
    let speed = length(v);
    if speed > vel_cap() {
        v = v * (vel_cap() / speed);
    }
    grid_vel[idx] = vec4<f32>(v, gv.w);
}

// ── packing pressure ──

fn packing_pressure_or_mirror(cell: vec3<i32>, mirror_pressure: f32) -> f32 {
    if cell.x < 0 || cell.y < 0 || cell.z < 0 {
        return mirror_pressure;
    }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() {
        return mirror_pressure;
    }

    let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    let kind = cell_kind_load(ci);
    if is_solid_kind(kind) {
        return mirror_pressure;
    }
    if !is_fluid_kind(kind) {
        return 0.0;
    }
    return packing_pressure_load(ci);
}

@compute @workgroup_size(64)
fn packing_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let kind = cell_kind_load(idx);
    if !is_fluid_kind(kind) || kind == CELL_BED_COUPLED {
        packing_pressure_store(idx, 0.0);
        return;
    }

    let cell_volume = dx() * dx() * dx();
    let packed_fraction = max(rest_volume_load(idx), current_volume_load(idx))
        / max(cell_volume, 1e-8);
    let packing_target = min(projection_max_rest_volume_fraction(), 1.0);
    let overpack = max(packed_fraction - packing_target, 0.0);
    packing_pressure_store(idx, bulk_K() * overpack);
}

@compute @workgroup_size(64)
fn packing_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= total_cells() { return; }

    let gv = grid_vel[idx];
    if gv.w <= occupancy_mass_threshold() {
        return;
    }

    let kind = cell_kind_load(idx);
    if !is_fluid_kind(kind) {
        return;
    }

    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));

    let p_here = packing_pressure_load(idx);
    let p_xm = packing_pressure_or_mirror(cell + vec3<i32>(-1, 0, 0), p_here);
    let p_xp = packing_pressure_or_mirror(cell + vec3<i32>(1, 0, 0), p_here);
    let p_ym = packing_pressure_or_mirror(cell + vec3<i32>(0, -1, 0), p_here);
    let p_yp = packing_pressure_or_mirror(cell + vec3<i32>(0, 1, 0), p_here);
    let p_zm = packing_pressure_or_mirror(cell + vec3<i32>(0, 0, -1), p_here);
    let p_zp = packing_pressure_or_mirror(cell + vec3<i32>(0, 0, 1), p_here);

    let grad_packing = 0.5 * inv_dx() * vec3<f32>(
        p_xp - p_xm,
        p_yp - p_ym,
        p_zp - p_zm,
    );

    var v = gv.xyz - dt() * grad_packing;
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
    let cell_pos = origin
        + (vec3<f32>(f32(ix_val), f32(iy_val), f32(iz_val)) + vec3<f32>(0.5)) * dx();

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
    if !is_water_phase(phase) {
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
    var bed_overlap_weight = 0.0;
    var bed_velocity_sum = vec3<f32>(0.0);
    var bed_permeability_sum = 0.0;

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
                let bed_idx = bed_lookup_load(ci);
                if num_bed() > 0u && is_valid_bed_solid_idx(bed_idx) {
                    bed_overlap_weight += w;
                    bed_velocity_sum += w * particles[u32(bed_idx)].vel.xyz;
                    bed_permeability_sum += w * bed_compacted_permeability(u32(bed_idx));
                }
            }
        }
    }

    // Sparse jets suffer strong PIC-style dissipation because empty stencil nodes
    // contribute zero velocity. When support is weak, preserve more of the
    // particle's previous ballistic motion instead of letting the stream stall.
    let support_ratio = clamp(supported_weight, 0.0, 1.0);
    var j_update_support = support_ratio;
    if supported_weight > 1e-6 {
        let inv_supported = 1.0 / supported_weight;
        new_v *= inv_supported;
        new_C0 *= inv_supported;
        new_C1 *= inv_supported;
        new_C2 *= inv_supported;
    }
    let in_cup_volume =
        xp.y < -3.5 && dot(xp.xz, xp.xz) < (3.0 + contact_offset()) * (3.0 + contact_offset());
    let dense_support_ratio =
        clamp(local_grid_mass / max(nominal_mass() * 4.0, 1e-6), 0.0, 1.0);
    // Use the particle's interpolation stencil rather than a single home-cell
    // bed lookup so particles exiting the coffee bed do not toggle abruptly
    // between porous and airborne transfer behavior at cell boundaries.
    let porous_overlap = clamp(bed_overlap_weight, 0.0, 1.0);
    var particle_bed_v = vec3<f32>(0.0);
    var particle_bed_permeability = min_bed_permeability_m2();
    if bed_overlap_weight > 1e-6 {
        let inv_bed_overlap = 1.0 / bed_overlap_weight;
        particle_bed_v = bed_velocity_sum * inv_bed_overlap;
        particle_bed_permeability = max(bed_permeability_sum * inv_bed_overlap, min_bed_permeability_m2());
    }
    if support_ratio < 0.999 {
        let ballistic_v = vec3<f32>(p.vel.x, p.vel.y + gravity() * dt(), p.vel.z);
        let free_preserve_gain = select(1.35, 1.15, in_cup_volume);
        let free_preserve_cap = select(0.995, 0.95, in_cup_volume);
        let preserve_gain = mix(free_preserve_gain, 0.18, porous_overlap);
        let preserve_cap = mix(free_preserve_cap, 0.16, porous_overlap);
        let dense_pool_damping = mix(1.0, 0.12, dense_support_ratio);
        let preserve =
            clamp((1.0 - support_ratio) * preserve_gain * dense_pool_damping, 0.0, preserve_cap);
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
    let airborne = porous_overlap <= 0.05 && sample_sdf(xp) > contact_offset() * 2.0;
    if airborne {
        let dense_mass = nominal_mass() * 4.0;
        let density_ratio = clamp(local_grid_mass / max(dense_mass, 1e-6), 0.0, 1.0);
        j_update_support = min(j_update_support, density_ratio);
        let ballistic_v = vec3<f32>(p.vel.x, p.vel.y + gravity() * dt(), p.vel.z);
        let preserve_gain = select(0.98, 0.72, in_cup_volume);
        let preserve_cap = select(0.98, 0.88, in_cup_volume);
        let preserve = clamp((1.0 - density_ratio) * preserve_gain, 0.0, preserve_cap);
        new_v = mix(new_v, ballistic_v, preserve);
        let affine_damp = 1.0 - preserve * 0.65;
        new_C0 *= affine_damp;
        new_C1 *= affine_damp;
        new_C2 *= affine_damp;
    }

    if porous_overlap > 1e-4 {
        let rel_before = new_v - particle_bed_v;
        let darcy_rate = water_kinematic_viscosity_m2_s() / particle_bed_permeability;
        let darcy_damping =
            1.0 / (1.0 + max(darcy_rate * dt() * porous_overlap, 0.0));
        let rel_speed = length(rel_before);
        let inertial_damping =
            1.0 / (1.0 + max(rel_speed * dt() * inv_dx() * porous_overlap * 0.35, 0.0));
        let new_v_before_porous_drag = new_v;
        new_v = particle_bed_v + rel_before * darcy_damping * inertial_damping;
        let water_impulse = mass_p * (new_v - new_v_before_porous_drag);
        bed_impulse_delta_add_neighborhood(
            world_to_cell(xp),
            -water_impulse * BED_REACTION_ALPHA,
        );
        let affine_damp = mix(1.0, darcy_damping * inertial_damping, porous_overlap);
        new_C0 *= affine_damp;
        new_C1 *= affine_damp;
        new_C2 *= affine_damp;
        j_update_support = max(j_update_support, porous_overlap);
    }

    let trace_C = new_C0.x + new_C1.y + new_C2.z;
    let J_old_clamped = clamp_particle_j(select(1.0, J_old, J_old > 0.0));
    let J_apic = clamp_particle_j(J_old_clamped * exp(clamp(dt() * trace_C, -0.35, 0.35)));
    let J_blend = smoothstep(0.35, 0.85, clamp(j_update_support, 0.0, 1.0));
    let J_new = clamp_particle_j(mix(J_old_clamped, J_apic, J_blend));

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

    if !is_water_phase(phase) {
        return;
    }

    let pos = particles[pid].pos.xyz;
    let cell = world_to_cell(pos);
    if cell.x < 0 || cell.y < 0 || cell.z < 0 { return; }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() { return; }

    var mass_p = particles[pid].vel.w;
    // Paper absorption is intentionally disabled by default until the paper
    // has finite retained-water storage and a release path. Keep the codepath
    // dormant for future work, but do not let the paper act as an infinite
    // water sink in long brews.
    let paper_weight = filter_paper_absorption_weight(pos);
    if paper_weight > 1e-4 && filter_absorption_rate() > 0.0 {
        let paper_absorb_fraction = clamp(filter_absorption_rate() * paper_weight * dt(), 0.0, 0.04);
        let paper_absorbed = min(mass_p * paper_absorb_fraction, mass_p * 0.12);
        mass_p = max(mass_p - paper_absorbed, inactive_mass_threshold() * 1.25);
        particles[pid].vel = vec4<f32>(particles[pid].vel.xyz, mass_p);
    }

    let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    let bed_idx = bed_lookup_load(ci);
    if bed_idx < 0 || u32(bed_idx) >= num_bed() {
        return;
    }

    var be = bed_extract[u32(bed_idx)];
    let saturation = be.extract.w;

    let capacity = max(max_saturation() - be.bed.x, 0.0);
    if capacity <= 1e-6 {
        return;
    }

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
    // Do not create a remnant that the inactive-particle path will zero on a
    // later pass unless the bed can be credited for the whole particle mass.
    if mass_p - absorbed <= inactive_mass_threshold() && absorbed < mass_p {
        absorbed = max(mass_p - inactive_mass_threshold() * 1.05, 0.0);
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
    deposit_absorbed_bed_water(cell, bed_idx, absorbed);
}

// ── extraction_advect ──

@compute @workgroup_size(64)
fn extraction_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bid = gid.x;
    if bid >= num_bed() { return; }

    var be = bed_extract[bid];
    let absorbed =
        f32(atomicExchange(&bed_delta[bed_delta_idx(BED_DELTA_WATER_LANE, bid)], 0))
            * inv_fp_scale();
    if absorbed > 0.0 {
        // Multiple water particles can reserve capacity against the same stale
        // bed state within one dispatch. Credit all atomically reported mass
        // here so water loss remains conservative, then clamp only the
        // saturation ratio that gates future absorption.
        be.bed.x += absorbed;
        be.extract.w = clamp(be.bed.x / max(max_saturation(), 1e-6), 0.0, 1.0);
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
    var phase = affine[pid].col0.w;
    var rest = affine[pid].col1.xyz;
    let pos = p.pos.xyz;
    let mass_p = p.vel.w;
    var be = bed_extract[bid];
    var bed_reaction_v = bed_impulse_delta_exchange(bid) / max(mass_p, nominal_mass() * 0.25);
    let max_reaction_speed = min(vel_cap() * 0.012, dx() / max(dt(), 1e-6) * 0.02);
    let reaction_speed = length(bed_reaction_v);
    if reaction_speed > max_reaction_speed && reaction_speed > 1e-6 {
        bed_reaction_v *= max_reaction_speed / reaction_speed;
    }

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
    var anchored_support_weight = 0.0;
    var lower_anchored_support_weight = 0.0;

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

                let support_bid = bed_lookup_load(ci);
                if support_bid >= 0 && u32(support_bid) != bid {
                    anchored_support_weight += w;
                    let cell_center_y = origin.y + (f32(cell.y) + 0.5) * dx();
                    if cell_center_y <= pos.y + dx() * 0.25 {
                        lower_anchored_support_weight += w;
                    }
                }
            }
        }
    }

    if is_suspended_coffee_phase(phase) {
        let water_support = clamp(water_mass / max(nominal_mass() * 2.0, 1e-6), 0.0, 1.0);
        let drag = clamp((8.0 + 18.0 * water_support) * dt(), 0.0, 1.0);
        // Suspended grounds are denser than water: surrounding flow can carry
        // fines, but they should still settle and should fall normally once
        // they leave a supported water region.
        let unsupported_v = p.vel.xyz + bed_reaction_v + vec3<f32>(0.0, gravity() * dt() * 0.85, 0.0);
        let settling_speed = mix(1.10, 0.32, water_support);
        let carried_v = water_v + vec3<f32>(0.0, -settling_speed, 0.0);
        var suspended_v = mix(unsupported_v, carried_v, drag);
        let speed = length(suspended_v);
        if speed > vel_cap() {
            suspended_v = suspended_v * (vel_cap() / speed);
        }

        var suspended_pos = pos + suspended_v * dt();
        let contact = resolve_sdf_contact(suspended_pos, suspended_v, true);
        suspended_pos = contact.pos;
        suspended_v = contact.vel;

        particles[pid].pos = vec4<f32>(suspended_pos, p.pos.w);
        particles[pid].vel = vec4<f32>(suspended_v, mass_p);
        let anchored_contact =
            lower_anchored_support_weight > 0.06 || anchored_support_weight > 0.20;
        let dense_bed_contact = anchored_support_weight > 0.42;
        let redeposit =
            (suspended_pos.y <= coffee_filter_floor_y() + dx() * 0.35 && water_support < 0.35)
            || (
                anchored_contact
                && suspended_v.y <= 0.20
                && (water_support < 0.55 || dense_bed_contact)
            );
        if redeposit {
            phase = 1.0;
        }
        affine[pid].col0 = vec4<f32>(affine[pid].col0.xyz, phase);
        affine[pid].col1 = vec4<f32>(suspended_pos, 0.0);
        be.bed.w *= 0.985;
        bed_extract[bid] = be;
        return;
    }

    let sat = be.extract.w;
    // Saturation should not pin individual grains. Wet grounds have lower
    // effective contact friction under pore pressure, so keep saturated
    // particles mobile instead of treating bound water as a positional anchor.
    let water_load = clamp(water_mass / max(nominal_mass() * 1.5, 1e-6), 0.0, 1.0);
    let wet_mobility = 1.0 + sat * 0.40;
    let mobility = clamp((0.30 + water_load * 0.62) * wet_mobility, 0.0, 1.0);
    let surface_factor = clamp((rest.y + 3.0) / 3.5, 0.12, 1.0);
    let damping = clamp(1.0 - bed_damping() * dt(), 0.0, 1.0);

    var vel = p.vel.xyz + bed_reaction_v;
    let support_release = clamp(1.0 - anchored_support_weight / 0.55, 0.0, 1.0);
    let entrainment_drag =
        clamp((2.0 + 5.0 * sat) * water_load * support_release * dt(), 0.0, 0.08);
    let grain_settling_v = vec3<f32>(0.0, -mix(0.95, 0.35, water_load), 0.0);
    vel = mix(vel, water_v + grain_settling_v, entrainment_drag);

    let lateral_water_v = water_v.xz;
    let lateral_speed = length(lateral_water_v);
    let lateral_dir = select(
        vec2<f32>(0.0),
        lateral_water_v / lateral_speed,
        lateral_speed > 1e-6,
    );
    let lateral_shear_speed = min(lateral_speed, abs(min(water_v.y, 0.0)) * 0.18);
    let impact_v = vec3<f32>(
        lateral_dir.x * lateral_shear_speed * 0.08,
        min(water_v.y, 0.0) * 0.62,
        lateral_dir.y * lateral_shear_speed * 0.08,
    );
    vel += impact_v * bed_impact() * mobility * surface_factor * dt();

    let pressure_grad = pressure_gradient_at_cell(world_to_cell(pos));
    let seepage_force_xz = -pressure_grad.xz;
    let seepage_speed = length(seepage_force_xz);
    let seepage_dir = select(
        vec2<f32>(0.0),
        seepage_force_xz / seepage_speed,
        seepage_speed > 1e-6,
    );
    let pressure_drive = min(seepage_speed, vel_cap() / max(dt(), 1e-6));
    let pore_load = clamp(water_mass * (0.25 + sat), 0.0, 1.0);
    let seepage_lateral_v = seepage_dir * pressure_drive * pore_load * 0.08;
    vel.x += seepage_lateral_v.x * bed_impact() * mobility * surface_factor * dt();
    vel.z += seepage_lateral_v.y * bed_impact() * mobility * surface_factor * dt();

    vel *= damping;

    var new_pos = pos + vel * dt();
    var offset = new_pos - rest;
    let lateral_len = length(offset.xz);
    let lateral_plastic_threshold = dx() * 0.22;
    if lateral_len > lateral_plastic_threshold && lateral_len > 1e-6 {
        let lateral_dir = offset.xz / lateral_len;
        let lateral_excess = lateral_len - lateral_plastic_threshold;
        let lateral_plasticity = clamp(
            (0.04 + sat * 0.35 + mobility * 0.45 + pore_load * 0.25)
                * surface_factor
                * bed_compaction_rate()
                * dt(),
            0.0,
            0.18,
        );
        rest.x += lateral_dir.x * lateral_excess * lateral_plasticity;
        rest.z += lateral_dir.y * lateral_excess * lateral_plasticity;
        offset = new_pos - rest;
    }
    let clamped_lateral_len = length(offset.xz);
    let max_lateral = dx() * 0.9 * surface_factor;
    if clamped_lateral_len > max_lateral && clamped_lateral_len > 1e-6 {
        let lateral_dir = offset.xz / clamped_lateral_len;
        offset.x = lateral_dir.x * max_lateral;
        offset.z = lateral_dir.y * max_lateral;
        vel.x *= 0.4;
        vel.z *= 0.4;
    }
    offset.y = clamp(offset.y, -dx() * (1.75 * surface_factor + 0.2), dx() * 0.04);
    new_pos = rest + offset;
    let packing_contact = resolve_coffee_particle_packing(new_pos, vel, bid);
    new_pos = packing_contact.pos;
    vel = packing_contact.vel;

    // Plastic compaction: once the bed is indented enough, lower the remembered
    // local rest height instead of applying an elastic spring back to the
    // original packed surface. Coffee grounds are treated here as an overdamped
    // porous granular bed; recovery should come from later flow/packing dynamics,
    // not from a shape-memory spring.
    let compression = max(rest.y - new_pos.y, 0.0);
    let plastic_threshold = dx() * 0.18;
    if compression > plastic_threshold {
        let excess = compression - plastic_threshold;
        let plasticity = clamp(
            (0.05 + sat * 0.35 + mobility * 0.55) * surface_factor * bed_compaction_rate() * dt(),
            0.0,
            0.18,
        );
        rest.y -= excess * plasticity;
    }

    let hydraulic_detach =
        sat > 0.82
        && pore_load > 0.65
        && water_load > 0.65
        && support_release > 0.35
        && (
            compression > dx() * 0.65
            || clamped_lateral_len > dx() * 0.65
        );
    let flow_entrained_grain =
        sat > 0.58
        && water_load > 0.74
        && support_release > 0.55
        && length(water_v) > 0.95
        && new_pos.y > coffee_filter_floor_y() + dx() * 0.70;
    let isolated_saturated_grain =
        sat > 0.55
        && water_mass < nominal_mass() * 0.75
        && anchored_support_weight < 0.04
        && lower_anchored_support_weight < 0.03
        && new_pos.y > coffee_filter_floor_y() + dx() * 0.50;
    if hydraulic_detach || flow_entrained_grain || isolated_saturated_grain {
        phase = PHASE_SUSPENDED_COFFEE;
        rest = new_pos;
    }

    let contact = resolve_sdf_contact(new_pos, vel, true);
    new_pos = contact.pos;
    vel = contact.vel;

    particles[pid].pos = vec4<f32>(new_pos, p.pos.w);
    particles[pid].vel = vec4<f32>(vel, mass_p);
    affine[pid].col0 = vec4<f32>(affine[pid].col0.xyz, phase);
    affine[pid].col1 = vec4<f32>(rest, 0.0);
    let geometric_compaction = max((rest.y - new_pos.y) / max(dx(), 1e-6), 0.0);
    be.bed.w = max(be.bed.w * 0.995, geometric_compaction);
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
        render_data[pid] = RenderParticle(
            vec4<f32>(0.0, -1e6, 0.0, -999.0),
            vec4<f32>(0.0),
        );
        return;
    }

    var color_t = 0.0;
    var radius = water_particle_radius();
    if is_water_phase(phase) {
        let speed = length(p.vel.xyz);
        color_t = clamp(speed / 10.0, 0.0, 2.0);
    } else {
        let bed_idx = pid;
        var sat = 0.0;
        if bed_idx < num_bed() {
            sat = bed_extract[bed_idx].extract.w;
            radius = bed_particle_radius();
        }
        color_t = -1.0 - sat;
    }

    render_data[pid] = RenderParticle(
        vec4<f32>(p.pos.xyz, color_t),
        vec4<f32>(radius, 0.0, 0.0, 0.0),
    );
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
    if is_water_phase(phase) {
        return;
    }
    // Mobile coffee is still a solid phase in the mixture. It should keep
    // contributing pore obstruction while it is suspended in a wet channel.

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

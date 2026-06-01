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
@group(0) @binding(12) var<storage, read_write> cg: array<vec4<f32>>;

// Metrics slot layout — keep in sync with `METRICS_SLOT_COUNT` in state.rs.
const OBSTACLE_WALL_THICKNESS: f32 = 0.4;
const METRIC_MAX_ABS_DIV_IDX: u32 = 0u;
const METRIC_FLUID_CELLS_IDX: u32 = 1u;
const METRIC_DIV_CLAMP_FIRES_IDX: u32 = 2u;
const METRIC_PRESSURE_CLAMP_FIRES_IDX: u32 = 3u;
const METRIC_MASS_OVERFLOW_FIRES_IDX: u32 = 4u;
const METRIC_CG_RZ_IDX: u32 = 5u;
const METRIC_CG_PAP_IDX: u32 = 6u;
const METRIC_CG_NEW_RZ_IDX: u32 = 7u;
const METRIC_PRESSURE_INITIAL_RZ_IDX: u32 = 8u;
const METRIC_PRESSURE_FINAL_RZ_IDX: u32 = 9u;
const METRIC_PRESSURE_ACTIVE_COUNT_IDX: u32 = 10u;
const METRIC_PRESSURE_ACTIVE_WORKGROUPS_X_IDX: u32 = 11u;
const METRIC_PRESSURE_ACTIVE_WORKGROUPS_Y_IDX: u32 = 12u;
const METRIC_PRESSURE_ACTIVE_WORKGROUPS_Z_IDX: u32 = 13u;
const METRIC_GRID_ACTIVE_COUNT_IDX: u32 = 14u;
const METRIC_GRID_ACTIVE_WORKGROUPS_X_IDX: u32 = 15u;
const METRIC_GRID_ACTIVE_WORKGROUPS_Y_IDX: u32 = 16u;
const METRIC_GRID_ACTIVE_WORKGROUPS_Z_IDX: u32 = 17u;
const BED_DELTA_WATER_LANE: u32 = 0u;
const BED_DELTA_IMPULSE_X_LANE: u32 = 1u;
const BED_DELTA_IMPULSE_Y_LANE: u32 = 2u;
const BED_DELTA_IMPULSE_Z_LANE: u32 = 3u;
const BED_REACTION_ALPHA: f32 = 0.04;
const BED_REACTION_IMPULSE_CAP: f32 = 0.012;

// ── Pressure CG solver tuning ──
// These guardrails exist because the CG inner products (r·z, pᵀAp) are
// accumulated into u32 atomics — integer-quantized, which loses exact CG
// conjugacy. The alpha cap and the beta clamp/restart compensate for that lost
// conjugacy and are LOAD-BEARING: GPU regression tests diverge if they are
// loosened. Centralized here so the solver's fragile knobs are legible in one
// place. The principled fix for the fragility is float reductions instead of
// fixed-point atomics, after which most of these could be relaxed or removed.
const PRESSURE_MIN_DIAGONAL: f32 = 1e-6;   // drop isolated cells (no fluid faces) from the solve
const PRESSURE_STORABLE_FRACTION: f32 = 0.5; // drop a cell from the solve if its Jacobi pressure estimate exceeds this fraction of the storable ceiling (vanishing-support sparse cell; legitimate hydrostatic pressure sits well below the clamp)
const CG_CONVERGENCE_REL_TOL: f32 = 1e-4;  // converged when weighted residual drops 4 orders vs initial
const CG_RZ_ABS_FLOOR: f32 = 1e-8;         // absolute residual floor for near-zero baselines
const CG_ALPHA_GATE_REL: f32 = 1e-5;       // skip the step when pᵀAp is tiny vs old_rz (near-singular direction)
const CG_RZ_DIVIDE_EPS: f32 = 1e-12;       // divide-by-zero guard for beta = new_rz / old_rz
const CG_MAX_ALPHA: f32 = 12.0;            // step-length cap; alpha ≫ 1 means a quantization-singular direction
const CG_MAX_BETA: f32 = 0.95;             // cap direction-history reuse against quantization-inflated beta
const CG_BETA_RESTART_RATIO: f32 = 0.98;   // restart as steepest descent when the residual stops shrinking

// ── Free-surface / continuum classification ──
const MIN_LATERAL_FLUID_FACES: u32 = 2u;   // ≥2 filled lateral neighbours required to join the pressure domain
const SURFACE_FREEFALL_SPEED: f32 = -10.0; // a surface cell falling faster than this (sim units/s downward) is treated as a ballistic free-stream, not a pressure-domain surface
const CUP_RIM_Y: f32 = -3.5;               // cup mouth plane (matches the cup obstacle top_y); cells above stay particle-resolved
const DENSE_CELL_MASS_FACTOR: f32 = 4.0;   // grid mass (× nominal) that counts as a "dense" / well-packed cell
// Smagorinsky LES coefficient for the sub-grid eddy viscosity — the one knob
// that sets how fast sheared/sloshing flow dissipates. Standard range 0.1–0.2.
const SMAGORINSKY_C: f32 = 0.15;

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
fn pressure_residual_fp_scale() -> f32 { return 1024.0; }
fn projection_j_alpha() -> f32 { return u.projection_params.x; }
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
// momentum. It carries pressure inv_diag during CG, then temporary unilateral
// packing pressure after pressure projection.
fn scratch_packing_idx(cell: u32) -> u32 { return grid_mom_y_idx(cell); }
fn scratch_kind_idx(cell: u32) -> u32 { return grid_mom_z_idx(cell); }
// A quadratic-B-spline particle deposits at most `nominal_mass * 0.75^3 ≈
// 0.42 * nominal_mass` to its peak cell. The threshold must stay strictly
// below that peak or isolated particles never register as fluid. Matching
// `inactive_mass_threshold()` at 0.1 * nominal_mass means "enough mass to
// still exist" ⇔ "enough mass to produce a fluid cell", which is
// semantically consistent and keeps ghost-splat noise below the bar.
fn occupancy_mass_threshold() -> f32 { return nominal_mass() * 0.1; }
fn viscosity_support_mass_threshold() -> f32 { return nominal_mass() * 0.5; }

const CELL_AIR: i32 = 0;
const CELL_SURFACE_FLUID: i32 = 1;
const CELL_INTERIOR_FLUID: i32 = 2;
const CELL_BED_COUPLED: i32 = 3;
const CELL_SOLID: i32 = 4;
const PHASE_WATER_MAX: f32 = 0.5;
const PHASE_SUSPENDED_COFFEE: f32 = 2.0;
const PRESSURE_REDUCE_WORKGROUP_SIZE: u32 = 64u;

var<workgroup> pressure_reduce_values: array<f32, 64>;

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

fn pressure_inv_diag_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[scratch_packing_idx(cell)])) * inv_fp_scale();
}

fn pressure_inv_diag_store(cell: u32, value: f32) {
    atomicStore(&grid[scratch_packing_idx(cell)], i32(max(value, 0.0) * fp_scale()));
}

fn pressure_cached_active_cell(cell: u32) -> bool {
    if grid_vel[cell].w <= occupancy_mass_threshold() {
        return false;
    }
    return pressure_inv_diag_load(cell) > 0.0;
}

fn pressure_field_active_cell(cell: u32) -> bool {
    if grid_vel[cell].w <= occupancy_mass_threshold() {
        return false;
    }

    let iz_val = cell / (gx() * gy());
    let rem = cell % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let kind = current_cell_kind(vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val)));
    return pressure_active_cell(cell, kind);
}

fn active_pressure_list_idx(list_i: u32) -> u32 {
    return total_cells() + list_i;
}

fn active_pressure_count() -> u32 {
    return atomicLoad(&metrics[METRIC_PRESSURE_ACTIVE_COUNT_IDX]);
}

fn active_pressure_cell(list_i: u32) -> u32 {
    return u32(cg[active_pressure_list_idx(list_i)].x);
}

fn active_pressure_cell_store(list_i: u32, cell: u32) {
    cg[active_pressure_list_idx(list_i)] = vec4<f32>(f32(cell), 0.0, 0.0, 0.0);
}

fn active_grid_list_idx(list_i: u32) -> u32 {
    return 2u * total_cells() + list_i;
}

fn active_grid_count() -> u32 {
    return atomicLoad(&metrics[METRIC_GRID_ACTIVE_COUNT_IDX]);
}

fn active_grid_cell(list_i: u32) -> u32 {
    return u32(cg[active_grid_list_idx(list_i)].x);
}

fn active_grid_cell_store(list_i: u32, cell: u32) {
    cg[active_grid_list_idx(list_i)] = vec4<f32>(f32(cell), 0.0, 0.0, 0.0);
}

// Per-cell cached liquid fill fraction (region 4 of `cg`). Assembled once in
// `pressure_linear_system_cell` (init) so the matvec reads neighbours' fill as
// a single load instead of re-running the 6-neighbour scan every iteration.
fn cg_fill_idx(cell: u32) -> u32 {
    return 3u * total_cells() + cell;
}
fn cg_fill_store(cell: u32, value: f32) {
    cg[cg_fill_idx(cell)] = vec4<f32>(value, 0.0, 0.0, 0.0);
}
fn cg_fill_load(cell: u32) -> f32 {
    return cg[cg_fill_idx(cell)].x;
}

// Region 4 of `cg`: persistent cell-center pressure. Survives across substeps
// (the `cg` buffer is never in the per-substep clear set), so the CG can
// warm-start from the previous substep's converged pressure.
fn cg_persist_pressure_idx(cell: u32) -> u32 {
    return 4u * total_cells() + cell;
}
fn persist_pressure_store(cell: u32, value: f32) {
    cg[cg_persist_pressure_idx(cell)] = vec4<f32>(value, 0.0, 0.0, 0.0);
}
fn persist_pressure_load(cell: u32) -> f32 {
    return cg[cg_persist_pressure_idx(cell)].x;
}

// ── Staggered grid: velocity on NODES, pressure on CELL CENTERS ──
// A pressure cell with index `c` is the dual point at world
// `origin + (vec3(c) + 0.5) * dx` (see `cell_center_from_cell`). Its 8 corner
// NODES are `c + (a,b,d)` for a,b,d in {0,1}; conversely a node `n`'s 8
// surrounding cells are `n - (a,b,d)`. The divergence (node velocity -> cell)
// and gradient (cell pressure -> node) are built from this one incidence so
// they are exact transposes (energy-orthogonal projection, no checkerboard).
// `staggered_node_in_bounds` guards corners/cells that fall off the node grid.
fn node_in_bounds(n: vec3<i32>) -> bool {
    return n.x >= 0 && n.y >= 0 && n.z >= 0
        && u32(n.x) < gx() && u32(n.y) < gy() && u32(n.z) < gz();
}

// A pressure cell `c` is geometrically valid iff all 8 of its corner nodes are
// in-bounds, i.e. `cx<gx-1, cy<gy-1, cz<gz-1` and `c>=0`. Cells touching the
// upper boundary are ghost/inactive.
fn staggered_cell_in_bounds(c: vec3<i32>) -> bool {
    return c.x >= 0 && c.y >= 0 && c.z >= 0
        && u32(c.x) + 1u < gx() && u32(c.y) + 1u < gy() && u32(c.z) + 1u < gz();
}

// Per-NODE liquid weight w_n ∈ [0,1] folded identically into D, A = D·Dᵀ and
// G = -Dᵀ so they stay exact transposes. It is the mean liquid fill of the
// node's surrounding fluid cells (air/solid cells count as 0 fill). A node deep
// inside a pool gets w_n ≈ 1; a node on the edge of a sparse falling stream gets
// a small w_n, which suppresses the pressure gradient there and stops the
// projection from converting axial stream momentum into spurious lateral motion.
// Computable in every pass (reads only cell kinds + deposited volume), so D
// (classify), A (CG) and G (project) all see the same weight.
fn staggered_node_fill_weight(n: vec3<i32>) -> f32 {
    var sum = 0.0;
    for (var a = 0; a <= 1; a++) {
        for (var b = 0; b <= 1; b++) {
            for (var d = 0; d <= 1; d++) {
                let c = n - vec3<i32>(a, b, d);
                if !staggered_cell_in_bounds(c) {
                    continue;
                }
                let kind = current_cell_kind(c);
                if !is_fluid_kind(kind) {
                    continue;
                }
                let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
                sum += raw_liquid_fill_fraction(ci, kind);
            }
        }
    }
    let mean_fill = clamp(sum / 8.0, 0.0, 1.0);
    // Sharpen the rolloff so sparse free-surface nodes (mostly air neighbours)
    // are strongly de-weighted while pool-interior nodes stay near 1. Cubing the
    // mean fill keeps the same single w_n in D, A and G (still exact transposes),
    // but pushes a half-air stream-edge node from ~0.4 down to ~0.06.
    return mean_fill * mean_fill * mean_fill;
}

// Pressure of a surrounding cell of a node, for the gradient G = -Dᵀ. Only
// ACTIVE FLUID cells are rows of D, so only they contribute to the transpose;
// every other cell (air = Dirichlet, solid/off-grid = excluded by the node-level
// Neumann v·n = 0) contributes 0. Returning 0 here — rather than a cell-level
// mirror — is what makes G exactly -Dᵀ: the Neumann wall is enforced at the
// solid NODES inside D (`staggered_corner_vel_component` → 0), not by mirroring a
// solid cell's pressure. Keeping G a pure transpose is what guarantees the
// projection is energy-orthogonal (the KEY free-surface energy gate).
fn staggered_cell_pressure(c: vec3<i32>) -> f32 {
    if !staggered_cell_in_bounds(c) {
        return 0.0;
    }
    if sdf_class_is_solid(c) {
        return 0.0;
    }
    let ci = cell_index(u32(c.x), u32(c.y), u32(c.z));
    if pressure_cached_active_cell(ci) {
        return pressure_load(ci);
    }
    return 0.0;
}

// Staggered DIVERGENCE at cell center `c` from the velocities of its 8 corner
// NODES `c + (a,b,d)`. Per axis the +face (a/b/d = 1) minus the −face (= 0).
// Scale `inv_dx * 0.25`. D must be a fixed LINEAR map on the node velocity
// vector so that A = D·Dᵀ and G = −Dᵀ hold exactly: each in-bounds, non-solid
// corner node contributes its OWN velocity component (linear in V); a solid or
// off-grid corner contributes 0 (Neumann wall v·n = 0). The energy-orthogonality
// of the resulting projection — not an air-velocity substitution — is what stops
// the free surface from injecting kinetic energy.
fn staggered_cell_divergence(c: vec3<i32>) -> f32 {
    var dvx = 0.0;
    var dvy = 0.0;
    var dvz = 0.0;
    for (var b = 0; b <= 1; b++) {
        for (var d = 0; d <= 1; d++) {
            // x faces: high corner (a=1) minus low corner (a=0).
            dvx += staggered_corner_vel_component(c + vec3<i32>(1, b, d), 0)
                 - staggered_corner_vel_component(c + vec3<i32>(0, b, d), 0);
            // y faces: group over (a, d).
            dvy += staggered_corner_vel_component(c + vec3<i32>(b, 1, d), 1)
                 - staggered_corner_vel_component(c + vec3<i32>(b, 0, d), 1);
            // z faces: group over (a, b).
            dvz += staggered_corner_vel_component(c + vec3<i32>(b, d, 1), 2)
                 - staggered_corner_vel_component(c + vec3<i32>(b, d, 0), 2);
        }
    }
    return inv_dx() * 0.25 * (dvx + dvy + dvz);
}

// Normal velocity component of a corner node for the divergence stencil, scaled
// by the per-node liquid weight w_n. Off-grid or solid → 0 (Neumann wall).
// Otherwise w_n · (node velocity component). The w_n factor is the SAME one A and
// G use, so D stays linear and A = D·Dᵀ, G = −Dᵀ hold exactly.
fn staggered_corner_vel_component(n: vec3<i32>, axis: i32) -> f32 {
    if !node_in_bounds(n) {
        return 0.0;
    }
    if sdf_class_is_solid(n) {
        return 0.0;
    }
    let ni = cell_index(u32(n.x), u32(n.y), u32(n.z));
    let nv = grid_vel[ni];
    let w = staggered_node_fill_weight(n);
    if axis == 0 {
        return w * nv.x;
    } else if axis == 1 {
        return w * nv.y;
    }
    return w * nv.z;
}

// Staggered GRADIENT at node `n`, the negative transpose of the divergence
// (G = -Dᵀ). For axis x: cells on the +x side of n (c.x = n.x) minus cells on
// the −x side (c.x = n.x-1), grouped over the other two axes. Scale inv_dx*0.25.
// Only active fluid cells contribute (see `staggered_cell_pressure`); air and
// solid cells contribute 0, so G is exactly -Dᵀ.
fn staggered_node_gradient(n: vec3<i32>) -> vec3<f32> {
    var gx_acc = 0.0;
    var gy_acc = 0.0;
    var gz_acc = 0.0;
    for (var b = 0; b <= 1; b++) {
        for (var d = 0; d <= 1; d++) {
            // x: +x side (a=0) minus −x side (a=1).
            gx_acc += staggered_cell_pressure(n - vec3<i32>(0, b, d))
                    - staggered_cell_pressure(n - vec3<i32>(1, b, d));
            // y: +y side minus −y side, grouped over (a,d).
            gy_acc += staggered_cell_pressure(n - vec3<i32>(b, 0, d))
                    - staggered_cell_pressure(n - vec3<i32>(b, 1, d));
            // z: +z side minus −z side, grouped over (a,b).
            gz_acc += staggered_cell_pressure(n - vec3<i32>(b, d, 0))
                    - staggered_cell_pressure(n - vec3<i32>(b, d, 1));
        }
    }
    // The whole node gradient carries the per-node weight w_n (every D[·,n] does),
    // so G = -Dᵀ exactly and sparse-stream boundary nodes get a weak gradient.
    let w = staggered_node_fill_weight(n);
    return w * inv_dx() * 0.25 * vec3<f32>(gx_acc, gy_acc, gz_acc);
}

fn pressure_or_mirror(cell: vec3<i32>, mirror_pressure: f32) -> f32 {
    if cell.x < 0 || cell.y < 0 || cell.z < 0 {
        return mirror_pressure;
    }
    if u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() {
        return mirror_pressure;
    }
    let ci = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    if is_solid_kind(current_cell_kind(cell)) {
        return mirror_pressure;
    }
    if !pressure_field_active_cell(ci) {
        return 0.0;
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
    return cg[cell].xyz;
}

fn velocity_scratch_store(cell: u32, value: vec3<f32>) {
    var clamped = value;
    let speed = length(clamped);
    if speed > vel_cap() {
        clamped = clamped * (vel_cap() / speed);
    }
    let quantized = vec3<f32>(
        f32(i32(clamped.x * fp_scale())) * inv_fp_scale(),
        f32(i32(clamped.y * fp_scale())) * inv_fp_scale(),
        f32(i32(clamped.z * fp_scale())) * inv_fp_scale(),
    );
    cg[cell] = vec4<f32>(quantized, 0.0);
}

fn rest_volume_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[grid_rest_volume_idx(cell)])) * inv_fp_scale();
}

fn current_volume_load(cell: u32) -> f32 {
    return f32(atomicLoad(&grid[grid_current_volume_idx(cell)])) * inv_fp_scale();
}

fn deposited_liquid_fraction(cell: u32) -> f32 {
    let cell_volume = max(dx() * dx() * dx(), 1e-8);
    return max(rest_volume_load(cell), current_volume_load(cell)) / cell_volume;
}

fn raw_liquid_fill_fraction(cell: u32, kind: i32) -> f32 {
    if kind == CELL_INTERIOR_FLUID || kind == CELL_BED_COUPLED {
        return 1.0;
    }
    if kind != CELL_SURFACE_FLUID {
        return 0.0;
    }

    return clamp(deposited_liquid_fraction(cell), 0.0, 1.0);
}

fn liquid_fill_fraction(cell: u32, kind: i32) -> f32 {
    if kind == CELL_INTERIOR_FLUID || kind == CELL_BED_COUPLED {
        return 1.0;
    }
    if kind != CELL_SURFACE_FLUID {
        return 0.0;
    }

    let iz_val = cell / (gx() * gy());
    let rem = cell % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let self_raw_fill = raw_liquid_fill_fraction(cell, kind);
    var fill_sum = self_raw_fill * 2.0;
    var fill_weight = 2.0;
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
            continue;
        }
        let neighbor_idx = cell_index(u32(neighbor.x), u32(neighbor.y), u32(neighbor.z));
        let neighbor_kind = current_cell_kind(neighbor);
        if !is_fluid_kind(neighbor_kind) {
            continue;
        }
        fill_sum += raw_liquid_fill_fraction(neighbor_idx, neighbor_kind);
        fill_weight += 1.0;
    }

    return clamp(fill_sum / max(fill_weight, 1e-6), 0.0, 1.0);
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
    if !pressure_active_cell(neighbor_cell, neighbor_kind) {
        return self_fill;
    }

    return min(self_fill, liquid_fill_fraction(neighbor_cell, neighbor_kind));
}

fn pressure_face_weight_cached(
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
    if !pressure_cached_active_cell(neighbor_cell) {
        return self_fill;
    }

    return min(self_fill, liquid_fill_fraction(neighbor_cell, neighbor_kind));
}

// Identical to `pressure_face_weight_cached` but reads the neighbour's liquid
// fill fraction from the per-cell cache (`cg_fill_load`) instead of recomputing
// the 6-neighbour scan. The cache is populated in init for every active
// pressure cell, which is exactly the set this branch reads (the
// `pressure_cached_active_cell` neighbours), so the result is identical.
//
// Invariant this relies on: init caches `liquid_fill_fraction(cell,
// cell_kind_load(cell))`, while the matvec wants it evaluated at
// `current_cell_kind(neighbour)`. Those kinds are equal for the cached set
// because `classify_cells` runs immediately before `pressure_cg_init` over the
// same active-grid list, so the stored kind matches the live SDF/occupancy
// classification. Keep classify → init adjacent in the dispatch order.
fn pressure_face_weight_fillcached(
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
    if !pressure_cached_active_cell(neighbor_cell) {
        return self_fill;
    }

    return min(self_fill, cg_fill_load(neighbor_cell));
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
    let kind = current_cell_kind(cell);
    if is_solid_kind(kind) {
        return mirror_pressure;
    }

    var neighbor_pressure = 0.0;
    if pressure_cached_active_cell(ci) {
        neighbor_pressure = pressure_load(ci);
    }
    let face_weight = pressure_face_weight_cached(self_kind, self_fill, ci, kind);
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
    return atomicLoad(&bed_lookup[cell]) - 1;
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
    if use_sdf_cache() {
        return textureLoad(sdf_class_tex, cell, 0).r != 0u;
    }
    return sample_sdf(cell_center_from_cell(cell)) < 0.0;
}

fn is_fluid_kind(kind: i32) -> bool {
    return kind == CELL_INTERIOR_FLUID
        || kind == CELL_SURFACE_FLUID
        || kind == CELL_BED_COUPLED;
}

fn current_cell_kind(cell: vec3<i32>) -> i32 {
    if cell.x < 0 || cell.y < 0 || cell.z < 0
        || u32(cell.x) >= gx() || u32(cell.y) >= gy() || u32(cell.z) >= gz() {
        return CELL_AIR;
    }
    if sdf_class_is_solid(cell) {
        return CELL_SOLID;
    }
    let idx = cell_index(u32(cell.x), u32(cell.y), u32(cell.z));
    if grid_vel[idx].w <= occupancy_mass_threshold() {
        return CELL_AIR;
    }
    return cell_kind_load(idx);
}

fn floor_supported_surface_cell(cell: u32) -> bool {
    let gv = grid_vel[cell];
    let quasi_static_speed = sqrt(max(abs(gravity()) * dx(), 1e-8));
    if length(gv.xyz) > quasi_static_speed {
        return false;
    }
    if deposited_liquid_fraction(cell) < 1.0 {
        return false;
    }

    let iz_val = cell / (gx() * gy());
    let rem = cell % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();

    let self_cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));
    if iy_val > 0u {
        let below = vec3<i32>(i32(ix_val), i32(iy_val) - 1, i32(iz_val));
        let floor_band_top = cup_floor_y() + dx() * 2.0;
        if cell_center_from_cell(self_cell).y <= floor_band_top && sdf_class_is_solid(below) {
            return true;
        }
    }

    return false;
}

fn surface_pressure_has_continuum_support(cell: u32) -> bool {
    if floor_supported_surface_cell(cell) {
        return true;
    }

    let iz_val = cell / (gx() * gy());
    let rem = cell % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let self_cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));

    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0),
        vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1),
        vec3<i32>(0, 0, 1),
    );

    var lower_fluid_support = false;
    var lateral_fluid_faces = 0u;

    for (var n = 0u; n < 6u; n++) {
        let neighbor = self_cell + offsets[n];
        if neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0
            || u32(neighbor.x) >= gx() || u32(neighbor.y) >= gy() || u32(neighbor.z) >= gz() {
            continue;
        }
        let neighbor_kind = current_cell_kind(neighbor);
        if neighbor_kind == CELL_BED_COUPLED {
            return true;
        }
        if is_fluid_kind(neighbor_kind) {
            let neighbor_idx = cell_index(u32(neighbor.x), u32(neighbor.y), u32(neighbor.z));
            if raw_liquid_fill_fraction(neighbor_idx, neighbor_kind) <= 0.0 {
                continue;
            }
            if offsets[n].y < 0 {
                lower_fluid_support = true;
            } else if offsets[n].y == 0 {
                lateral_fluid_faces += 1u;
            }
        }
    }

    // A water-only free surface belongs to the continuum pressure domain once
    // it is the boundary of a locally supported liquid volume: at least one
    // fluid cell below carries hydrostatic support, and lateral fluid faces
    // distinguish a pool/sheet surface from an isolated falling stream.
    //
    // BUT fluid-below is NOT floor support: a free-FALLING column self-satisfies
    // it (the column continues below, and a 2+-wide column has lateral fluid
    // faces), which wrongly promoted the laminar pour into the pressure domain.
    // The staggered free-surface gradient then sprayed it sideways (ejecting the
    // pour out of the domain → no particles on screen). A genuine free-falling
    // stream is ballistic (no solid reaction), so additionally require that it
    // is NOT in fast free-fall to join the pressure domain; a settled/decelerated
    // pool surface (small downward speed) still qualifies. The pool BULK stays
    // incompressible via the interior-cell projection regardless.
    let falling = grid_vel[cell].y < SURFACE_FREEFALL_SPEED;
    return lower_fluid_support
        && lateral_fluid_faces >= MIN_LATERAL_FLUID_FACES
        && !falling;
}

fn interior_pressure_has_continuum_support(cell: u32) -> bool {
    if deposited_liquid_fraction(cell) >= 1.0 || floor_supported_surface_cell(cell) {
        return true;
    }

    let iz_val = cell / (gx() * gy());
    let rem = cell % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let self_cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));
    let self_center = cell_center_from_cell(self_cell);
    // Under-filled interior cells need a receiving body of water, not merely
    // other falling samples. The cup boundary is the water-only reservoir; bed
    // support is handled separately through CELL_BED_COUPLED.
    if self_center.y > CUP_RIM_Y {
        return false;
    }

    let offsets = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0),
        vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0),
        vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1),
        vec3<i32>(0, 0, 1),
    );

    var lower_hydro_support = false;
    var lateral_fluid_faces = 0u;

    for (var n = 0u; n < 6u; n++) {
        let neighbor = self_cell + offsets[n];
        if neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0
            || u32(neighbor.x) >= gx() || u32(neighbor.y) >= gy() || u32(neighbor.z) >= gz() {
            continue;
        }

        let neighbor_idx = cell_index(u32(neighbor.x), u32(neighbor.y), u32(neighbor.z));
        let neighbor_kind = current_cell_kind(neighbor);
        if !is_fluid_kind(neighbor_kind)
            || raw_liquid_fill_fraction(neighbor_idx, neighbor_kind) <= 0.0 {
            continue;
        }

        if offsets[n].y < 0 {
            lower_hydro_support = true;
        } else if offsets[n].y == 0 {
            lateral_fluid_faces += 1u;
        }
    }

    return lower_hydro_support && lateral_fluid_faces >= MIN_LATERAL_FLUID_FACES;
}

fn pressure_active_cell(cell: u32, kind: i32) -> bool {
    // Surface cells are part of the liquid domain when they have continuum
    // support from the porous bed or a solid floor under gravity. Isolated
    // falling streams remain particle-resolved free surfaces instead of being
    // promoted into a pressure domain.
    return (kind == CELL_INTERIOR_FLUID && interior_pressure_has_continuum_support(cell))
        || kind == CELL_BED_COUPLED
        || (kind == CELL_SURFACE_FLUID && surface_pressure_has_continuum_support(cell));
}

fn packing_active_cell(cell: u32, kind: i32) -> bool {
    return kind == CELL_INTERIOR_FLUID
        || (kind == CELL_SURFACE_FLUID && surface_pressure_has_continuum_support(cell));
}

fn is_viscous_kind(kind: i32) -> bool {
    // Viscosity is a continuum stress. Surface cells need it too once they
    // have enough neighboring fluid support; otherwise dense cup-pool pockets
    // can keep grid-transfer/packing energy and bubble upward after pour-off.
    // Sparse free-falling jets are still protected by the support gates in
    // `viscosity_prepare`.
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
    var target_div = compressed_error * projection_j_alpha();

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
    // Use analytic obstacles for live contact and pressure classification.
    // The texture SDF is coarse enough that a circular cone picks up a
    // fourfold Cartesian alias, which traps water on the filter wall and then
    // ejects it inward. Keep this parallel to `MpmSettings::default_v60()`.
    var result = 999.0;

    let cone_top_y = 3.0;
    let cone_bot_y = -3.0;
    if position.y <= cone_top_y && position.y >= cone_bot_y {
        let t = clamp((position.y - cone_bot_y) / (cone_top_y - cone_bot_y), 0.0, 1.0);
        let cone_radius = mix(dripper_outlet_radius(), dripper_top_radius(), t);
        result = cone_radius - length(position.xz) - obstacle_wall_half_thickness();
    }

    let cup_top_y = CUP_RIM_Y;
    let cup_bot_y = -8.0;
    if position.y <= cup_top_y {
        let radial_sd = 3.0 - length(position.xz);
        let floor_sd = position.y - cup_bot_y;
        let cup_sd = min(radial_sd, floor_sd) - obstacle_wall_half_thickness();
        result = select(max(result, cup_sd), cup_sd, result >= 998.0);
    }

    return result;
}

fn cell_center_from_cell(cell: vec3<i32>) -> vec3<f32> {
    return u.grid_origin.xyz
        + (vec3<f32>(cell) + vec3<f32>(0.5)) * dx();
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

fn resolve_conical_barrier(
    position: vec3<f32>,
    velocity: vec3<f32>,
    center: vec2<f32>,
    bot_y: f32,
    bot_radius: f32,
    slope: f32,
) -> ContactResult {
    var out_pos = position;
    var out_vel = velocity;

    let radial = out_pos.xz - center;
    let r = length(radial);
    if r > 1e-6 {
        let outward = radial / r;
        let cone_radius = bot_radius + slope * (out_pos.y - bot_y);
        let sdf_val = cone_radius - r;
        if sdf_val < contact_offset() {
            let n = normalize(vec3<f32>(-outward.x, slope, -outward.y));
            out_pos += n * (contact_offset() - sdf_val);

            let vn = dot(out_vel, n);
            let radial_v = dot(out_vel.xz, outward);
            if vn < 0.0 && radial_v > 0.0 {
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

    return ContactResult(out_pos, out_vel);
}

fn resolve_scene_obstacles(position: vec3<f32>, velocity: vec3<f32>, is_bed: bool) -> ContactResult {
    var out_pos = position;
    var out_vel = velocity;

    // V60 dripper interior. The analytic fallback uses the conical surface
    // normal, not a cylindrical radial clamp, so wall impact becomes
    // down-slope film motion instead of a collapse onto a vertical ray.
    // Keep this analytic fallback parallel to `FilterConfig::default()` and
    // `v60_support_cone`; the SDF texture uses the same obstacle radii.
    let cone_top_y = 3.0;
    let cone_bot_y = -3.0;
    let cone_height = cone_top_y - cone_bot_y;
    let cone_slope = (dripper_top_radius() - dripper_outlet_radius()) / cone_height;
    if out_pos.y <= cone_top_y && out_pos.y >= cone_bot_y {
        let outlet_rim_band = dx() * 4.0;
        if out_pos.y <= cone_bot_y + outlet_rim_band {
            let t = clamp((out_pos.y - cone_bot_y) / cone_height, 0.0, 1.0);
            let cone_radius = mix(dripper_outlet_radius(), dripper_top_radius(), t) - contact_offset();
            let cone_contact = resolve_radial_barrier(out_pos, out_vel, vec2<f32>(0.0, 0.0), cone_radius);
            out_pos = cone_contact.pos;
            out_vel = cone_contact.vel;
            let outlet_radial = out_pos.xz;
            let outlet_r = length(outlet_radial);
            if outlet_r > 1e-6 {
                let outlet_outward = outlet_radial / outlet_r;
                let inward_v = min(dot(out_vel.xz, outlet_outward), 0.0);
                out_vel.x -= outlet_outward.x * inward_v;
                out_vel.z -= outlet_outward.y * inward_v;
            }
        } else {
            let cone_contact = resolve_conical_barrier(
                out_pos,
                out_vel,
                vec2<f32>(0.0, 0.0),
                cone_bot_y,
                dripper_outlet_radius(),
                cone_slope,
            );
            out_pos = cone_contact.pos;
            out_vel = cone_contact.vel;
        }
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
    if out_pos.y <= CUP_RIM_Y {
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

    // APIC affine state is stored by spatial-gradient columns:
    // C0=dv/dx, C1=dv/dy, C2=dv/dz. P2G must apply C*dpos with the
    // same orientation that g2p reconstructs below.
    let C0 = a.col0.xyz;
    let C1 = a.col1.xyz;
    let C2 = a.col2.xyz;

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
                let affine_mom = mass_p * (C0 * dpos.x + C1 * dpos.y + C2 * dpos.z);
                let mom = w * (mass_p * vp + affine_mom);

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
    let active_slot = atomicAdd(&metrics[METRIC_GRID_ACTIVE_COUNT_IDX], 1u);
    active_grid_cell_store(active_slot, idx);

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

@compute @workgroup_size(1)
fn grid_active_finalize_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u { return; }
    let count = active_grid_count();
    atomicStore(
        &metrics[METRIC_GRID_ACTIVE_WORKGROUPS_X_IDX],
        max((count + PRESSURE_REDUCE_WORKGROUP_SIZE - 1u) / PRESSURE_REDUCE_WORKGROUP_SIZE, 1u),
    );
    atomicStore(&metrics[METRIC_GRID_ACTIVE_WORKGROUPS_Y_IDX], 1u);
    atomicStore(&metrics[METRIC_GRID_ACTIVE_WORKGROUPS_Z_IDX], 1u);
}

// ── viscosity ──

@compute @workgroup_size(64)
fn viscosity_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }
    let idx = active_grid_cell(list_i);

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
    let kind = cell_kind_load(idx);
    if !is_viscous_kind(kind) {
        velocity_scratch_store(idx, v_here);
        return;
    }
    // Viscosity is a bulk-fluid stress. Sparse streams have too little local
    // support for a stable velocity Laplacian, so gate it by local occupancy
    // rather than by scene location.
    if gv.w <= viscosity_support_mass_threshold() {
        velocity_scratch_store(idx, v_here);
        return;
    }

    var neighbor_velocity_sum = vec3<f32>(0.0);
    var neighbor_weight_sum = 0.0;
    var fluid_neighbor_count = 0u;
    let wall_viscosity_weight = 2.0;

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
            fluid_neighbor_count += 1u;
            continue;
        }
        if sdf_class_is_solid(neighbor) {
            // No-slip boundary: static solids dissipate tangential pool motion.
            neighbor_weight_sum += wall_viscosity_weight;
            fluid_neighbor_count += 1u;
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
    // Strain-dependent effective viscosity. Water's molecular ν (the
    // `viscosity()` uniform) is microscopic, so on its own this diffusion does
    // nothing and a sloshing pool never settles. The real dissipation at this
    // resolution is *turbulent* — energy cascades to sub-grid eddies we cannot
    // resolve — and the sub-grid model for it is the Smagorinsky eddy viscosity
    // nu_t = (Cs * dx)^2 * |S|, with |S| the local strain rate estimated from
    // the cell↔neighbour-average velocity difference (a one-cell |grad v|).
    // This is large in shear/boundary layers (where turbulence dissipates) and
    // ~0 in coherent flow, so it settles the sloshing pool WITHOUT damping the
    // laminar pour stream — something no constant ν can do (only the strain
    // rate distinguishes them). `alpha` (the diffusion number) is still clamped
    // to 0.14 < 1/(2d) = 1/6, the explicit-diffusion CFL stability limit.
    let strain_rate = length(v_here - neighbor_average) / max(dx(), 1e-6);
    let eddy_viscosity = (SMAGORINSKY_C * dx()) * (SMAGORINSKY_C * dx()) * strain_rate;
    let alpha = clamp(
        (viscosity() + eddy_viscosity) * dt() / max(dx() * dx(), 1e-6),
        0.0,
        0.14,
    );
    let blend = clamp(alpha * neighbor_weight_sum, 0.0, 0.65);
    velocity_scratch_store(idx, mix(v_here, neighbor_average, blend));
}

@compute @workgroup_size(64)
fn viscosity_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }
    let idx = active_grid_cell(list_i);

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
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }
    let idx = active_grid_cell(list_i);

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
    let self_is_solid = sdf_class_is_solid(vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val)));
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

    let bed_idx_here = bed_lookup_load(idx);
    if bed_idx_here >= 0 {
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

    // Staggered divergence at the CELL CENTER of cell `idx` (the cell whose
    // lowest corner is node `idx`) from its 8 corner-node velocities. D is a
    // fixed linear map (solid/off-grid corner → 0, Neumann v·n=0); it pairs with
    // the gradient G = -Dᵀ in project_pressure and the Laplacian A = D·Dᵀ in the
    // CG so the projection is energy-orthogonal. A cell touching the upper node
    // boundary (cx=gx-1 etc.) is a ghost cell with no full corner set → div 0.
    let cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));
    if !staggered_cell_in_bounds(cell) {
        divergence_store(idx, 0.0);
        return;
    }
    let div = staggered_cell_divergence(cell);
    divergence_store(idx, div);

    // Observability: track the worst-case cell divergence and the fluid-cell
    // footprint of the active substep. `atomicMax` on u32 gives the peak FP
    // encoding; the HUD decodes via `METRICS_DIV_FP_SCALE`.
    let abs_div = abs(div);
    let fp_div = u32(clamp(abs_div * metrics_div_fp_scale(), 0.0, f32(0x7fffffffu)));
    atomicMax(&metrics[METRIC_MAX_ABS_DIV_IDX], fp_div);
    atomicAdd(&metrics[METRIC_FLUID_CELLS_IDX], 1u);
}

// ── pressure CG ──

fn pressure_projection_target_divergence(
    idx: u32,
    kind: i32,
    cell: vec3<i32>,
    cell_center: vec3<f32>,
) -> f32 {
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
            let solid_div = bed_velocity_divergence(cell);
            target_divergence = (pore_target - solid_fraction * solid_div) / porosity;
        }
    }
    return target_divergence;
}

// Is corner node `n` a valid (in-bounds, non-solid) divergence node? Solid /
// off-grid corners have D[·,n] = 0 (Neumann wall), so they drop from A = D·Dᵀ.
fn staggered_div_node_valid(n: vec3<i32>) -> bool {
    return node_in_bounds(n) && !sdf_class_is_solid(n);
}

// Sign of D[c,n] per axis: +1 if node n is the high corner of cell c on that
// axis (n - c = 1), −1 if the low corner (n - c = 0). D[c,n].axis = 0.25*sign.
fn staggered_div_sign(c: vec3<i32>, n: vec3<i32>) -> vec3<f32> {
    let r = vec3<f32>(n - c); // each component is 0 or 1
    return 2.0 * r - vec3<f32>(1.0);
}

// Diagonal of A = D·Dᵀ at cell c: Σ_n D[c,n]·D[c,n] over the 8 corner nodes.
// Each valid corner contributes 0.0625 * (1 + 1 + 1) = 0.1875; solid/off-grid
// corners contribute 0. So diag ∈ {0 .. 1.5}.
fn staggered_laplacian_diag(c: vec3<i32>) -> f32 {
    var diag = 0.0;
    for (var a = 0; a <= 1; a++) {
        for (var b = 0; b <= 1; b++) {
            for (var d = 0; d <= 1; d++) {
                let n = c + vec3<i32>(a, b, d);
                if !staggered_div_node_valid(n) {
                    continue;
                }
                let s = staggered_div_sign(c, n);
                let w = staggered_node_fill_weight(n);
                diag += 0.0625 * w * w * dot(s, s);
            }
        }
    }
    return diag;
}

// (A·d)_c for A = D·Dᵀ, reading the CG search direction d[c'] = cg[c'].z for the
// active fluid cells c' that share a corner node with c. Built literally as
// Σ_n Σ_c' D[c,n]·D[c',n]·d[c'] (loop c's 8 corner nodes; for each valid node
// loop its 8 surrounding cells), so A is symmetric and equals D·Dᵀ exactly. The
// self term (c' = c) gives the diagonal; inactive (air/solid) c' contribute 0
// (Dirichlet p=0 / node-level Neumann), matching G = -Dᵀ in project_pressure.
fn staggered_laplacian_apply_d(c: vec3<i32>, d_self: f32) -> f32 {
    var q = 0.0;
    for (var a = 0; a <= 1; a++) {
        for (var b = 0; b <= 1; b++) {
            for (var dd = 0; dd <= 1; dd++) {
                let n = c + vec3<i32>(a, b, dd);
                if !staggered_div_node_valid(n) {
                    continue;
                }
                let w = staggered_node_fill_weight(n);
                // Both D[c,n] and D[c',n] carry w_n, so the node contributes w_n².
                let s_cn = 0.25 * w * staggered_div_sign(c, n);
                // n's 8 surrounding cells c' = n - (a',b',d').
                for (var a2 = 0; a2 <= 1; a2++) {
                    for (var b2 = 0; b2 <= 1; b2++) {
                        for (var d2 = 0; d2 <= 1; d2++) {
                            let cp = n - vec3<i32>(a2, b2, d2);
                            var d_cp = 0.0;
                            if all(cp == c) {
                                d_cp = d_self;
                            } else if staggered_cell_in_bounds(cp) && !sdf_class_is_solid(cp) {
                                let cpi = cell_index(u32(cp.x), u32(cp.y), u32(cp.z));
                                if pressure_cached_active_cell(cpi) {
                                    d_cp = cg[cpi].z;
                                }
                            }
                            if d_cp == 0.0 {
                                continue;
                            }
                            // s_cn already carries one w_n; the shared node's w_n²
                            // is completed by the second factor here.
                            let s_cpn = 0.25 * w * staggered_div_sign(cp, n);
                            q += dot(s_cn, s_cpn) * d_cp;
                        }
                    }
                }
            }
        }
    }
    return q;
}

fn pressure_linear_system_cell(idx: u32) -> vec2<f32> {
    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();

    let kind = cell_kind_load(idx);
    if !pressure_active_cell(idx, kind) {
        return vec2<f32>(0.0);
    }

    let cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));
    let self_fill = liquid_fill_fraction(idx, kind);
    // Cache this cell's fill for the matvec (read as a neighbour every CG
    // iteration). The init pass finishes before the matvec pass, so the cache
    // is fully populated for every active pressure cell by the time it is read.
    cg_fill_store(idx, self_fill);

    // Diagonal of A = D·Dᵀ (staggered Laplacian) at this cell.
    let diag = staggered_laplacian_diag(cell);

    let target_divergence = pressure_projection_target_divergence(
        idx,
        kind,
        cell,
        cell_center_from_cell(cell),
    );
    // A·p = -(div - target)/dt. The implemented A is the dimensionless graph
    // operator (no inv_dx²), while `divergence_load` carries inv_dx, so scale the
    // RHS by dx² to match the pressure_load/gradient units (grad = inv_dx·...).
    let rhs = -dx() * dx() * (divergence_load(idx) - target_divergence) * self_fill
        / max(dt(), 1e-6);
    return vec2<f32>(diag, rhs);
}

fn pressure_apply_search_direction(idx: u32) -> vec2<f32> {
    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();

    if pressure_inv_diag_load(idx) <= 0.0 {
        return vec2<f32>(0.0);
    }

    let cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));
    let d_here = cg[idx].z;
    // q = (A·d)_c; the global reduction wants p_ap = dᵀA d = Σ_c d_c·q_c, so this
    // cell contributes d_c·q_c (full quadratic form; A SPD ⇒ contribution ≥ 0).
    let q = staggered_laplacian_apply_d(cell, d_here);
    let energy = d_here * q;
    return vec2<f32>(q, energy);
}

@compute @workgroup_size(64)
fn pressure_cg_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }
    let idx = active_grid_cell(list_i);

    let kind = cell_kind_load(idx);
    if !pressure_active_cell(idx, kind) {
        pressure_store(idx, 0.0);
        pressure_inv_diag_store(idx, 0.0);
        cg[idx] = vec4<f32>(0.0);
        return;
    }

    let system = pressure_linear_system_cell(idx);
    let diag = system.x;
    let r = system.y;
    if diag <= PRESSURE_MIN_DIAGONAL {
        pressure_store(idx, 0.0);
        pressure_inv_diag_store(idx, 0.0);
        cg[idx] = vec4<f32>(0.0);
        return;
    }

    let inv_diag = 1.0 / diag;
    let z = r * inv_diag;
    // Storable-pressure gate. The weighted operator A = D·Dᵀ folds the per-node
    // liquid weight w_n into both D factors, so a cell's pressure scales like
    // 1/w_n: a sparse free-surface cell (tiny w_n) has a near-zero diagonal and a
    // Jacobi pressure estimate `z = r/diag` that runs past the fixed-point
    // pressure storage ceiling (±pressure_clamp_limit). Storing such a cell would
    // saturate the clamp and feed a CORRUPTED (clamped) pressure back into the
    // staggered gradient — breaking G = -Dᵀ and ejecting the sparse stream. By
    // 1/w_n, an unstorable z means the cell has vanishing continuum support, so
    // drop it from the pressure domain and leave it particle-resolved (ballistic
    // v*), exactly as the active-set design intends for sparse streams. The gate
    // reads only the cell's own (diag, r), so the matvec/gradient — which test
    // membership via the cached inv_diag — stay consistent with classification.
    if abs(z) > pressure_clamp_limit() * PRESSURE_STORABLE_FRACTION {
        pressure_store(idx, 0.0);
        pressure_inv_diag_store(idx, 0.0);
        cg[idx] = vec4<f32>(0.0);
        return;
    }
    pressure_store(idx, 0.0);
    pressure_inv_diag_store(idx, inv_diag);
    cg[idx] = vec4<f32>(r, z, z, 0.0);
    let active_slot = atomicAdd(&metrics[METRIC_PRESSURE_ACTIVE_COUNT_IDX], 1u);
    active_pressure_cell_store(active_slot, idx);
    let rz = max(r * z, 0.0);
    let rz_fixed = u32(clamp(rz, 0.0, f32(0xffffffffu)));
    let observed_rz_fixed = u32(clamp(rz * pressure_residual_fp_scale(), 0.0, f32(0xffffffffu)));
    atomicAdd(&metrics[METRIC_CG_RZ_IDX], rz_fixed);
    atomicAdd(&metrics[METRIC_PRESSURE_INITIAL_RZ_IDX], observed_rz_fixed);
}

@compute @workgroup_size(1)
fn pressure_active_finalize_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u { return; }
    let count = active_pressure_count();
    atomicStore(
        &metrics[METRIC_PRESSURE_ACTIVE_WORKGROUPS_X_IDX],
        max((count + PRESSURE_REDUCE_WORKGROUP_SIZE - 1u) / PRESSURE_REDUCE_WORKGROUP_SIZE, 1u),
    );
    atomicStore(&metrics[METRIC_PRESSURE_ACTIVE_WORKGROUPS_Y_IDX], 1u);
    atomicStore(&metrics[METRIC_PRESSURE_ACTIVE_WORKGROUPS_Z_IDX], 1u);
}

fn pressure_cg_converged(old_rz: f32) -> bool {
    let initial_rz =
        f32(atomicLoad(&metrics[METRIC_PRESSURE_INITIAL_RZ_IDX])) / pressure_residual_fp_scale();
    let converged_rz = max(initial_rz * CG_CONVERGENCE_REL_TOL, CG_RZ_ABS_FLOOR);
    return old_rz <= converged_rz;
}

@compute @workgroup_size(64)
fn pressure_cg_matvec(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Self-terminating solve: once the weighted residual is below the
    // convergence threshold every remaining CG iteration is a no-op. Exit
    // before the workgroup reduction so an over-provisioned iteration budget
    // costs almost nothing. `pressure_cg_converged` reads one uniform metric,
    // so the whole workgroup returns together — no barrier divergence.
    if pressure_cg_converged(f32(atomicLoad(&metrics[METRIC_CG_RZ_IDX]))) {
        return;
    }

    let list_i = gid.x;
    var contribution = 0.0;
    if list_i < active_pressure_count() {
        let idx = active_pressure_cell(list_i);
        let applied = pressure_apply_search_direction(idx);
        cg[idx].w = applied.x;
        contribution = max(applied.y, 0.0);
    }

    pressure_reduce_values[lid.x] = contribution;
    workgroupBarrier();
    for (var stride = PRESSURE_REDUCE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if lid.x < stride {
            pressure_reduce_values[lid.x] += pressure_reduce_values[lid.x + stride];
        }
        workgroupBarrier();
    }
    if lid.x == 0u && pressure_reduce_values[0] > 0.0 {
        atomicAdd(
            &metrics[METRIC_CG_PAP_IDX],
            u32(clamp(pressure_reduce_values[0], 0.0, f32(0xffffffffu))),
        );
    }
}

@compute @workgroup_size(64)
fn pressure_cg_apply_alpha(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Mirror the matvec's self-termination (see pressure_cg_matvec): skip the
    // whole iteration cheaply once converged.
    if pressure_cg_converged(f32(atomicLoad(&metrics[METRIC_CG_RZ_IDX]))) {
        return;
    }

    let list_i = gid.x;
    var contribution = 0.0;
    if list_i < active_pressure_count() {
        let idx = active_pressure_cell(list_i);
        let inv_diag = pressure_inv_diag_load(idx);
        if inv_diag > 0.0 {
            let old_rz = f32(atomicLoad(&metrics[METRIC_CG_RZ_IDX]));
            let p_ap = f32(atomicLoad(&metrics[METRIC_CG_PAP_IDX]));
            if p_ap > max(old_rz * CG_ALPHA_GATE_REL, CG_RZ_ABS_FLOOR) {
                let alpha = old_rz / p_ap;
                // A Jacobi-preconditioned graph Laplacian has O(1) stable CG
                // steps. A much larger alpha means the fixed-point global
                // reductions have reached a numerically singular search
                // direction; stop instead of injecting a late pressure impulse
                // into free-surface particles.
                if alpha <= CG_MAX_ALPHA {
                    let c = cg[idx];
                    let p_new = pressure_load(idx) + alpha * c.z;
                    let r_new = c.x - alpha * c.w;
                    let z_new = r_new * inv_diag;
                    pressure_store(idx, p_new);
                    cg[idx] = vec4<f32>(r_new, z_new, c.z, c.w);
                    contribution = max(r_new * z_new, 0.0);
                }
            }
        }
    }

    pressure_reduce_values[lid.x] = contribution;
    workgroupBarrier();
    for (var stride = PRESSURE_REDUCE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if lid.x < stride {
            pressure_reduce_values[lid.x] += pressure_reduce_values[lid.x + stride];
        }
        workgroupBarrier();
    }
    if lid.x == 0u && pressure_reduce_values[0] > 0.0 {
        atomicAdd(
            &metrics[METRIC_CG_NEW_RZ_IDX],
            u32(clamp(pressure_reduce_values[0], 0.0, f32(0xffffffffu))),
        );
    }
}

@compute @workgroup_size(64)
fn pressure_cg_update_dir(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_pressure_count() { return; }

    let idx = active_pressure_cell(list_i);
    if pressure_inv_diag_load(idx) <= 0.0 {
        return;
    }

    let old_rz = f32(atomicLoad(&metrics[METRIC_CG_RZ_IDX]));
    if pressure_cg_converged(old_rz) {
        return;
    }
    let new_rz = f32(atomicLoad(&metrics[METRIC_CG_NEW_RZ_IDX]));
    let beta_raw = select(0.0, new_rz / old_rz, old_rz > CG_RZ_DIVIDE_EPS);
    // Fixed-point reductions and changing free-surface stencils can lose CG
    // conjugacy. Restart as preconditioned steepest descent when the weighted
    // residual grows; otherwise the next direction can keep feeding the same
    // pressure error back into the liquid head.
    let beta = select(clamp(beta_raw, 0.0, CG_MAX_BETA), 0.0, new_rz > old_rz * CG_BETA_RESTART_RATIO);
    let c = cg[idx];
    cg[idx] = vec4<f32>(c.x, c.y, c.y + beta * c.z, c.w);
}

@compute @workgroup_size(1)
fn pressure_cg_finish_iteration(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u { return; }
    atomicStore(&metrics[METRIC_CG_RZ_IDX], atomicLoad(&metrics[METRIC_CG_NEW_RZ_IDX]));
    atomicStore(&metrics[METRIC_CG_PAP_IDX], 0u);
    atomicStore(&metrics[METRIC_CG_NEW_RZ_IDX], 0u);
}

@compute @workgroup_size(64)
fn pressure_residual_measure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_pressure_count() { return; }

    let idx = active_pressure_cell(list_i);
    if pressure_inv_diag_load(idx) <= 0.0 {
        return;
    }

    let c = cg[idx];
    let rz = max(c.x * c.y, 0.0);
    atomicAdd(
        &metrics[METRIC_PRESSURE_FINAL_RZ_IDX],
        u32(clamp(rz * pressure_residual_fp_scale(), 0.0, f32(0xffffffffu))),
    );
}

// ── project_pressure ──

// project_pressure now iterates NODES (the active-grid list). At each node it
// applies the staggered pressure gradient G = -Dᵀ (built from the 8 surrounding
// cell pressures) to the node velocity: v = grid_vel[node] - dt*grad. Because G
// is the exact transpose of the divergence and A = D·Dᵀ, the projection is
// energy-orthogonal — it removes divergence without injecting kinetic energy at
// the free surface. Bed Darcy/reaction coupling is applied afterwards in the
// separate `project_bed_darcy` pass (it is a per-cell continuum effect, not part
// of the gradient correction).
@compute @workgroup_size(64)
fn project_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }

    let idx = active_grid_cell(list_i);
    let gv = grid_vel[idx];
    if gv.w < 1e-6 { return; }

    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let node = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));

    // Solid nodes are pinned by the boundary projection; the pressure gradient
    // must not move them. (They also carry no fluid mass, but guard explicitly.)
    if sdf_class_is_solid(node) {
        return;
    }

    let grad_p = staggered_node_gradient(node);
    var v = gv.xyz - dt() * grad_p;

    let speed = length(v);
    if speed > vel_cap() {
        v = v * (vel_cap() / speed);
    }
    grid_vel[idx] = vec4<f32>(v, gv.w);
}

// Bed Darcy damping + reaction impulse, split out of project_pressure so the
// gradient correction can iterate nodes. Iterates the active pressure CELLS and
// runs only on CELL_BED_COUPLED cells, reading the just-projected node velocity.
// Behavior matches the old in-line block (it only depended on grid_vel[cell] and
// the bed state, never on the pressure gradient itself).
@compute @workgroup_size(64)
fn project_bed_darcy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_pressure_count() { return; }

    let idx = active_pressure_cell(list_i);
    let gv = grid_vel[idx];
    if gv.w < 1e-6 { return; }
    if pressure_inv_diag_load(idx) <= 0.0 { return; }
    let kind = cell_kind_load(idx);
    if kind != CELL_BED_COUPLED { return; }

    let bed_idx = bed_lookup_load(idx);
    if bed_idx < 0 || u32(bed_idx) >= num_bed() { return; }

    let iz_val = idx / (gx() * gy());
    let rem = idx % (gx() * gy());
    let iy_val = rem / gx();
    let ix_val = rem % gx();
    let bed_cell = vec3<i32>(i32(ix_val), i32(iy_val), i32(iz_val));

    let permeability_m2 = bed_compacted_permeability(u32(bed_idx));
    let darcy_rate = water_kinematic_viscosity_m2_s() / permeability_m2;
    let darcy_damping = 1.0 / (1.0 + max(darcy_rate * dt(), 0.0));
    let bed_v = bed_matrix_velocity_cell(bed_cell);
    var v = bed_v + (gv.xyz - bed_v) * darcy_damping;
    let water_impulse = gv.w * (v - gv.xyz);
    bed_impulse_delta_add_neighborhood(bed_cell, -water_impulse * BED_REACTION_ALPHA);

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
    let kind = current_cell_kind(cell);
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
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }
    let idx = active_grid_cell(list_i);

    let kind = cell_kind_load(idx);
    // This explicit packing pressure is a dense-cell correction. Sparse
    // free-surface pressure remains atmospheric; only floor-supported surface
    // cells get the correction so a settled pool cannot represent its support
    // purely as particle compression.
    if !packing_active_cell(idx, kind) {
        packing_pressure_store(idx, 0.0);
        return;
    }

    let cell_volume = dx() * dx() * dx();
    let packed_fraction = max(rest_volume_load(idx), current_volume_load(idx))
        / max(cell_volume, 1e-8);
    let packing_target = projection_max_rest_volume_fraction();
    let overpack = max(packed_fraction - packing_target, 0.0);
    packing_pressure_store(idx, bulk_K() * overpack);
}

@compute @workgroup_size(64)
fn packing_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }
    let idx = active_grid_cell(list_i);

    let gv = grid_vel[idx];
    if gv.w <= occupancy_mass_threshold() {
        return;
    }

    let kind = cell_kind_load(idx);
    if !packing_active_cell(idx, kind) {
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
    let list_i = gid.x;
    if list_i >= active_grid_count() { return; }
    let idx = active_grid_cell(list_i);

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
    // Pure APIC (IMPM step 5): the particle velocity and the affine matrix C are
    // both interpolated directly from the divergence-free projected grid velocity
    // v^{n+1}. No FLIP blend and no ballistic-preservation cascade — with the
    // projection done correctly as the last grid op, incompressible motion is
    // carried by the affine modes without needing per-region momentum patches.

    // Bed/porous transfer (Darcy drag) — genuine physics, kept as-is. Use the
    // particle's interpolation stencil rather than a single home-cell bed lookup
    // so particles exiting the bed do not toggle abruptly at cell boundaries.
    let porous_overlap = clamp(bed_overlap_weight, 0.0, 1.0);
    var particle_bed_v = vec3<f32>(0.0);
    var particle_bed_permeability = min_bed_permeability_m2();
    if bed_overlap_weight > 1e-6 {
        let inv_bed_overlap = 1.0 / bed_overlap_weight;
        particle_bed_v = bed_velocity_sum * inv_bed_overlap;
        particle_bed_permeability = max(bed_permeability_sum * inv_bed_overlap, min_bed_permeability_m2());
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

const METRICS_SLOT_COUNT: u32 = 18u;

@compute @workgroup_size(8)
fn metrics_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= METRICS_SLOT_COUNT { return; }
    atomicStore(&metrics[idx], 0u);
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
    let id = i32(bid + 1u);

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

    #[test]
    fn apic_columns_are_applied_without_transpose() {
        assert!(MPM_COMPUTE_SHADER.contains("C0 * dpos.x + C1 * dpos.y + C2 * dpos.z"));
        assert!(!MPM_COMPUTE_SHADER.contains("dot(aff_col0, dpos)"));
    }
}

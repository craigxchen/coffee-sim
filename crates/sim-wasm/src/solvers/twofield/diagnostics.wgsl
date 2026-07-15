// State-neutral compact diagnostics. Particle and grid state are scanned in two single-workgroup
// passes so each entry point stays within the WebGPU ten-storage-buffer contract. The passes only
// write diagnostics_output; requesting a snapshot never dispatches a physics kernel.

struct DiagnosticsConfig {
    cup0: vec4<f32>, // (center x, center z, radius, minimum y)
    cup1: vec4<f32>, // (maximum y, enabled, seeded bed top, reserved)
    bed0: vec4<f32>, // (minimum x, maximum x, minimum z, maximum z)
    bed1: vec4<f32>, // (minimum y, maximum y, outlet half-band in cell widths, valid)
};

struct DiagnosticsOutput {
    mobile: vec4<f32>,
    paper: vec4<f32>,
    grain: vec4<f32>,
    cup: vec4<f32>,
    grid: vec4<f32>, // max occupancy, bed-supported pore carrier, ambient enthalpy, reserved
    pressure: vec4<f32>, // pre |r| sum, pre r² sum, post |r| sum, post r² sum
    pressure_ext: vec4<f32>, // pre max, post max, active-cell count, reserved
    even0: vec4<f32>,
    even1: vec4<f32>,
    even2: vec4<f32>,
    even3: vec4<f32>,
    counts: vec4<u32>, // pond particles, held parcels, held volume counts, bad particles
    filter0: vec4<u32>,
    filter1: vec4<u32>, // filter errors, bad grid nodes, reserved, reserved
    snapshot_meta: vec4<u32>,
    reserved: vec4<f32>,
};

// A mapped zero-filled buffer is otherwise indistinguishable from a successfully reduced dry
// scene. Each pass stamps its own lane; the host accepts a sample only when both dispatches ran.
const DIAGNOSTICS_PARTICLE_CANARY: u32 = 0x44504731u; // "DPG1"
const DIAGNOSTICS_GRID_CANARY: u32 = 0x44504732u; // "DPG2"
const DIAGNOSTICS_GRID_PARTIAL_CANARY: u32 = 0x44504750u; // "DPGP"

// Grid diagnostics write one exact partial per dispatched workgroup, then a compact finalizer
// folds those records into the canonical output. This keeps the expensive residual stencil
// parallel across the device instead of serializing the entire grid through one workgroup.
@group(0) @binding(31) var<storage, read_write> diagnostics_grid_partials: array<DiagnosticsOutput>;
@group(0) @binding(32) var<uniform> diagnostics_config: DiagnosticsConfig;
@group(0) @binding(36) var<storage, read_write> diagnostics_output: DiagnosticsOutput;

var<workgroup> diagnostics_f32: array<f32, 256>;
var<workgroup> diagnostics_u32: array<u32, 256>;

fn diagnostics_finite(value: f32) -> bool {
    return value == value && abs(value) <= 3.0e38;
}

fn diagnostics_finite4(value: vec4<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec4<f32>(3.0e38));
}

fn diagnostics_sum_f32(value: f32, lane: u32) -> f32 {
    diagnostics_f32[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) { diagnostics_f32[lane] += diagnostics_f32[lane + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride /= 2u;
    }
    return diagnostics_f32[0];
}

fn diagnostics_max_f32(value: f32, lane: u32) -> f32 {
    diagnostics_f32[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) {
            diagnostics_f32[lane] = max(diagnostics_f32[lane], diagnostics_f32[lane + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride /= 2u;
    }
    return diagnostics_f32[0];
}

fn diagnostics_sum_u32(value: u32, lane: u32) -> u32 {
    diagnostics_u32[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) { diagnostics_u32[lane] += diagnostics_u32[lane + stride]; }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride /= 2u;
    }
    return diagnostics_u32[0];
}

@compute @workgroup_size(256)
fn diagnostics_particles(@builtin(local_invocation_index) lane: u32) {
    var mobile_carrier = 0.0;
    var mobile_solute = 0.0;
    var mobile_capacity = 0.0;
    var mobile_enthalpy = 0.0;
    var paper_carrier = 0.0;
    var paper_solute = 0.0;
    var paper_capacity = 0.0;
    var paper_enthalpy = 0.0;
    var bound_carrier = 0.0;
    var remaining_solute = 0.0;
    var grain_capacity = 0.0;
    var grain_enthalpy = 0.0;
    var cup_carrier = 0.0;
    var cup_solute = 0.0;
    var pond_carrier = 0.0;
    var ambient_total = 0.0;
    var pond_count = 0u;
    var held_count = 0u;
    var held_volume_counts = 0u;
    var bad_particles = 0u;

    var p = lane;
    loop {
        if (p >= params.water_count) { break; }
        let position = pos[p];
        let velocity = vel[p];
        let state = chem[p];
        let c0 = cmat[3u * p + 0u];
        let c1 = cmat[3u * p + 1u];
        let c2 = cmat[3u * p + 2u];
        let ambient = ambient_enthalpy[p];
        let valid = phase[p] == PHASE_WATER
            && diagnostics_finite4(position) && diagnostics_finite4(velocity)
            && diagnostics_finite4(state) && diagnostics_finite4(c0)
            && diagnostics_finite4(c1) && diagnostics_finite4(c2)
            && diagnostics_finite(ambient);
        if (!valid) {
            bad_particles += 1u;
        } else {
            ambient_total += ambient;
            let carrier = water_solvent_mass(position.w);
            let solute_mass = water_solute_mass(position.w, state.x);
            let capacity = carrier * max(params.therm.x, 0.0)
                + solute_mass * max(params.solute.y, 0.0);
            let enthalpy = capacity * state.y;
            let held = state.z >= 0.5 && state.z <= 1.5;
            if (held) {
                paper_carrier += carrier;
                paper_solute += solute_mass;
                paper_capacity += capacity;
                paper_enthalpy += enthalpy;
                held_count += 1u;
                held_volume_counts += filter_volume_encode(clamp(position.w, 0.0, 1.0) * params.wet0.z);
            } else {
                mobile_carrier += carrier;
                mobile_solute += solute_mass;
                mobile_capacity += capacity;
                mobile_enthalpy += enthalpy;
                if (carrier > 0.0 && position.y > diagnostics_config.cup1.z) {
                    pond_count += 1u;
                    pond_carrier += carrier;
                }
                if (diagnostics_config.cup1.y > 0.5) {
                    let dx = position.x - diagnostics_config.cup0.x;
                    let dz = position.z - diagnostics_config.cup0.y;
                    if (dx * dx + dz * dz <= diagnostics_config.cup0.z * diagnostics_config.cup0.z
                        && position.y >= diagnostics_config.cup0.w
                        && position.y <= diagnostics_config.cup1.x) {
                        cup_carrier += carrier;
                        cup_solute += solute_mass;
                    }
                }
            }
        }
        p += 256u;
    }

    let solid_start = params.particle_count - params.solid_count;
    var solid = lane;
    loop {
        if (solid >= params.solid_count) { break; }
        let index = solid_start + solid;
        let position = pos[index];
        let velocity = vel[index];
        let state = chem[index];
        let c0 = cmat[3u * index + 0u];
        let c1 = cmat[3u * index + 1u];
        let c2 = cmat[3u * index + 2u];
        let f0 = fmat[3u * solid + 0u];
        let f1 = fmat[3u * solid + 1u];
        let f2 = fmat[3u * solid + 2u];
        let s0 = sstate[2u * solid + 0u];
        let s1 = sstate[2u * solid + 1u];
        let ambient = ambient_enthalpy[index];
        let valid = phase[index] == PHASE_SOLID
            && diagnostics_finite4(position) && diagnostics_finite4(velocity)
            && diagnostics_finite4(state) && diagnostics_finite4(c0)
            && diagnostics_finite4(c1) && diagnostics_finite4(c2)
            && diagnostics_finite4(f0) && diagnostics_finite4(f1)
            && diagnostics_finite4(f2) && diagnostics_finite4(s0)
            && diagnostics_finite4(s1) && diagnostics_finite(ambient);
        if (!valid) {
            bad_particles += 1u;
        } else {
            ambient_total += ambient;
            let bound = max(position.w, 0.0) * params.particle_mass / max(params.wet0.z, 1.0e-9);
            let remaining = max(state.x, 0.0) + max(state.y, 0.0);
            let capacity = params.solute.x * max(params.therm.y, 0.0)
                + remaining * max(params.solute.y, 0.0)
                + bound * max(params.therm.x, 0.0);
            bound_carrier += bound;
            remaining_solute += remaining;
            grain_capacity += capacity;
            grain_enthalpy += capacity * state.z;
        }
        solid += 256u;
    }

    let mobile = vec4<f32>(
        diagnostics_sum_f32(mobile_carrier, lane), diagnostics_sum_f32(mobile_solute, lane),
        diagnostics_sum_f32(mobile_capacity, lane), diagnostics_sum_f32(mobile_enthalpy, lane));
    let paper = vec4<f32>(
        diagnostics_sum_f32(paper_carrier, lane), diagnostics_sum_f32(paper_solute, lane),
        diagnostics_sum_f32(paper_capacity, lane), diagnostics_sum_f32(paper_enthalpy, lane));
    let grain = vec4<f32>(
        diagnostics_sum_f32(bound_carrier, lane), diagnostics_sum_f32(remaining_solute, lane),
        diagnostics_sum_f32(grain_capacity, lane), diagnostics_sum_f32(grain_enthalpy, lane));
    let cup = vec4<f32>(
        diagnostics_sum_f32(cup_carrier, lane), diagnostics_sum_f32(cup_solute, lane),
        diagnostics_sum_f32(pond_carrier, lane), 0.0);
    let ambient = diagnostics_sum_f32(ambient_total, lane);
    let counts = vec4<u32>(
        diagnostics_sum_u32(pond_count, lane), diagnostics_sum_u32(held_count, lane),
        diagnostics_sum_u32(held_volume_counts, lane), diagnostics_sum_u32(bad_particles, lane));

    if (lane == 0u) {
        diagnostics_output.mobile = mobile;
        diagnostics_output.paper = paper;
        diagnostics_output.grain = grain;
        diagnostics_output.cup = cup;
        diagnostics_output.grid = vec4<f32>(0.0, 0.0, ambient, 0.0);
        diagnostics_output.pressure = vec4<f32>(0.0);
        diagnostics_output.pressure_ext = vec4<f32>(0.0);
        diagnostics_output.even0 = vec4<f32>(0.0);
        diagnostics_output.even1 = vec4<f32>(0.0);
        diagnostics_output.even2 = vec4<f32>(0.0);
        diagnostics_output.even3 = vec4<f32>(0.0);
        diagnostics_output.counts = counts;
        diagnostics_output.filter0 = vec4<u32>(0u);
        diagnostics_output.filter1 = vec4<u32>(0u);
        diagnostics_output.snapshot_meta = vec4<u32>(DIAGNOSTICS_PARTICLE_CANARY, 0u, 0u, 0u);
        diagnostics_output.reserved = vec4<f32>(0.0);
    }
}

fn diagnostics_pressure_residual(c: u32) -> f32 {
    let cm = cell_meta[c];
    if (cm.y <= 0.0) { return 0.0; }
    let h = params.grid_origin.w;
    let cc = vec3<i32>(cell_coords(c));
    var acc = 0.0;
    for (var oz = 0; oz < 2; oz += 1) {
        for (var oy = 0; oy < 2; oy += 1) {
            for (var ox = 0; ox < 2; ox += 1) {
                let o = vec3<i32>(ox, oy, oz);
                let node = cc + o;
                let nidx = node_index(node);
                let m = Minv(nm[2u * nidx], nm[2u * nidx + 1u]);
                var gp = vec3<f32>(0.0);
                for (var qz = 0; qz < 2; qz += 1) {
                    for (var qy = 0; qy < 2; qy += 1) {
                        for (var qx = 0; qx < 2; qx += 1) {
                            let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                            if (!cell_in_range(c2)) { continue; }
                            let ci2 = cell_index(c2);
                            let f2 = cell_meta[ci2].y;
                            if (f2 <= 0.0) { continue; }
                            let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                            gp -= s2 * f2 * pf_src[ci2] / (4.0 * h);
                        }
                    }
                }
                acc += m.b.w * dot(corner_sign(o), minv_apply(m, gp)) / (4.0 * h);
            }
        }
    }
    return cm.x - (-acc * cm.y);
}

fn diagnostics_outlet_bin(node: u32) -> i32 {
    if (diagnostics_config.bed1.w < 0.5) { return -1; }
    let nc = node_coords(node);
    let xp = params.grid_origin.xyz + vec3<f32>(nc) * params.grid_origin.w;
    let band = diagnostics_config.bed1.z * params.grid_origin.w;
    if (abs(xp.y - diagnostics_config.bed1.x) > band
        || xp.x < diagnostics_config.bed0.x || xp.x > diagnostics_config.bed0.y
        || xp.z < diagnostics_config.bed0.z || xp.z > diagnostics_config.bed0.w) { return -1; }
    let ux = clamp((xp.x - diagnostics_config.bed0.x)
        / max(diagnostics_config.bed0.y - diagnostics_config.bed0.x, 1.0e-9), 0.0, 0.999999);
    let uz = clamp((xp.z - diagnostics_config.bed0.z)
        / max(diagnostics_config.bed0.w - diagnostics_config.bed0.z, 1.0e-9), 0.0, 0.999999);
    return i32(floor(ux * 4.0)) + 4 * i32(floor(uz * 4.0));
}

fn diagnostics_grid_partial_count() -> u32 {
    return (max(params.grid_dims.w, num_fine_cells()) + 2047u) / 2048u;
}

@compute @workgroup_size(256)
fn diagnostics_grid(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    var maximum_occupancy = 0.0;
    var bed_pore_carrier = 0.0;
    var bad_grid_nodes = 0u;
    var pre_abs = 0.0;
    var pre_sq = 0.0;
    var pre_max = 0.0;
    var post_abs = 0.0;
    var post_sq = 0.0;
    var post_max = 0.0;
    var active_cells = 0u;
    var bad_cells = 0u;
    var outlet_flux = array<f32, 16>();
    let base = global_id.x * 8u;
    for (var item = 0u; item < 8u; item += 1u) {
        let node = base + item;
        if (node < params.grid_dims.w) {
            let gv = grid_vel[node];
            let n0 = nm[2u * node];
            let n1 = nm[2u * node + 1u];
            let reaction = react[node];
            let sv = grid_svel[node];
            if (!diagnostics_finite4(gv) || !diagnostics_finite4(n0)
                || !diagnostics_finite4(n1) || !diagnostics_finite4(reaction)
                || !diagnostics_finite4(sv)) {
                bad_grid_nodes += 1u;
            } else {
                let carrier = carrier_mass_at(node);
                maximum_occupancy = max(
                    maximum_occupancy, carrier / max(params.extra.x, 1.0e-9));
                if (solid_volume_at(node) > 0.0) { bed_pore_carrier += carrier; }
                let outlet_bin = diagnostics_outlet_bin(node);
                if (outlet_bin >= 0) {
                    let phi = nm[2u * node + 1u].w;
                    if (phi > 0.0 && diagnostics_finite(phi)) {
                        outlet_flux[u32(outlet_bin)] +=
                            max(-phi * intrinsic_mobile_velocity_at(node).y, 0.0);
                    }
                }
            }
        }

        let c = base + item;
        if (c < num_fine_cells()) {
            let cm = cell_meta[c];
            let pressure_value = pf_src[c];
            if (!diagnostics_finite4(cm) || !diagnostics_finite(pressure_value)) {
                bad_cells += 1u;
            } else if (cm.y > 0.0) {
                let pre = abs(cm.x);
                let post_value = diagnostics_pressure_residual(c);
                if (!diagnostics_finite(post_value)) {
                    bad_cells += 1u;
                } else {
                    let post = abs(post_value);
                    pre_abs += pre;
                    pre_sq += pre * pre;
                    pre_max = max(pre_max, pre);
                    post_abs += post;
                    post_sq += post * post;
                    post_max = max(post_max, post);
                    active_cells += 1u;
                }
            }
        }
    }

    let occupancy = diagnostics_max_f32(maximum_occupancy, lane);
    let pore = diagnostics_sum_f32(bed_pore_carrier, lane);
    let pressure_values = vec4<f32>(
        diagnostics_sum_f32(pre_abs, lane), diagnostics_sum_f32(pre_sq, lane),
        diagnostics_sum_f32(post_abs, lane), diagnostics_sum_f32(post_sq, lane));
    let pressure_ext = vec4<f32>(
        diagnostics_max_f32(pre_max, lane), diagnostics_max_f32(post_max, lane),
        f32(diagnostics_sum_u32(active_cells, lane)), 0.0);
    let bad_grid = diagnostics_sum_u32(bad_grid_nodes + bad_cells, lane);

    // A scale-free 4x4 participation ratio is derived on the CPU. Each lane processes a short
    // contiguous chunk before the shared reductions, amortizing WebGPU barrier overhead without
    // changing which nodes or pressure rows enter the exact sums.
    var even = array<f32, 16>();
    for (var bin = 0; bin < 16; bin += 1) {
        even[bin] = diagnostics_sum_f32(outlet_flux[bin], lane);
    }

    if (lane == 0u) {
        let partial = workgroup_id.x;
        diagnostics_grid_partials[partial].mobile = vec4<f32>(0.0);
        diagnostics_grid_partials[partial].paper = vec4<f32>(0.0);
        diagnostics_grid_partials[partial].grain = vec4<f32>(0.0);
        diagnostics_grid_partials[partial].cup = vec4<f32>(0.0);
        diagnostics_grid_partials[partial].grid = vec4<f32>(occupancy, pore, 0.0, 0.0);
        diagnostics_grid_partials[partial].pressure = pressure_values;
        diagnostics_grid_partials[partial].pressure_ext = pressure_ext;
        diagnostics_grid_partials[partial].even0 = vec4<f32>(even[0], even[1], even[2], even[3]);
        diagnostics_grid_partials[partial].even1 = vec4<f32>(even[4], even[5], even[6], even[7]);
        diagnostics_grid_partials[partial].even2 = vec4<f32>(even[8], even[9], even[10], even[11]);
        diagnostics_grid_partials[partial].even3 = vec4<f32>(even[12], even[13], even[14], even[15]);
        diagnostics_grid_partials[partial].counts = vec4<u32>(0u);
        diagnostics_grid_partials[partial].filter0 = select(
            vec4<u32>(0u),
            vec4<u32>(
                atomicLoad(&filter_state[FILTER_OCCUPIED]),
                atomicLoad(&filter_state[FILTER_ADMITTED]),
                atomicLoad(&filter_state[FILTER_RELEASED]),
                atomicLoad(&filter_state[FILTER_OVERFLOW])),
            partial == 0u,
        );
        diagnostics_grid_partials[partial].filter1 = vec4<u32>(
            select(0u, atomicLoad(&filter_state[FILTER_ERRORS]), partial == 0u),
            bad_grid, 0u, 0u);
        diagnostics_grid_partials[partial].snapshot_meta =
            vec4<u32>(DIAGNOSTICS_GRID_PARTIAL_CANARY, 0u, 0u, 0u);
        diagnostics_grid_partials[partial].reserved = vec4<f32>(0.0);
    }
}

@compute @workgroup_size(256)
fn diagnostics_grid_finalize(@builtin(local_invocation_index) lane: u32) {
    var occupancy = 0.0;
    var pore = 0.0;
    var pressure = vec4<f32>(0.0);
    var pressure_ext = vec4<f32>(0.0);
    var filter0 = vec4<u32>(0u);
    var filter_errors = 0u;
    var bad_grid = 0u;
    var completed_partials = 0u;
    var even = array<f32, 16>();
    var partial = lane;
    let partial_count = diagnostics_grid_partial_count();
    loop {
        if (partial >= partial_count) { break; }
        let value = diagnostics_grid_partials[partial];
        occupancy = max(occupancy, value.grid.x);
        pore += value.grid.y;
        pressure += value.pressure;
        pressure_ext.x = max(pressure_ext.x, value.pressure_ext.x);
        pressure_ext.y = max(pressure_ext.y, value.pressure_ext.y);
        pressure_ext.z += value.pressure_ext.z;
        filter0 += value.filter0;
        filter_errors += value.filter1.x;
        bad_grid += value.filter1.y;
        completed_partials += u32(value.snapshot_meta.x == DIAGNOSTICS_GRID_PARTIAL_CANARY);
        even[0] += value.even0.x; even[1] += value.even0.y;
        even[2] += value.even0.z; even[3] += value.even0.w;
        even[4] += value.even1.x; even[5] += value.even1.y;
        even[6] += value.even1.z; even[7] += value.even1.w;
        even[8] += value.even2.x; even[9] += value.even2.y;
        even[10] += value.even2.z; even[11] += value.even2.w;
        even[12] += value.even3.x; even[13] += value.even3.y;
        even[14] += value.even3.z; even[15] += value.even3.w;
        partial += 256u;
    }

    let final_occupancy = diagnostics_max_f32(occupancy, lane);
    let final_pore = diagnostics_sum_f32(pore, lane);
    let final_pressure = vec4<f32>(
        diagnostics_sum_f32(pressure.x, lane), diagnostics_sum_f32(pressure.y, lane),
        diagnostics_sum_f32(pressure.z, lane), diagnostics_sum_f32(pressure.w, lane));
    let final_pressure_ext = vec4<f32>(
        diagnostics_max_f32(pressure_ext.x, lane), diagnostics_max_f32(pressure_ext.y, lane),
        diagnostics_sum_f32(pressure_ext.z, lane), 0.0);
    let final_filter0 = vec4<u32>(
        diagnostics_sum_u32(filter0.x, lane), diagnostics_sum_u32(filter0.y, lane),
        diagnostics_sum_u32(filter0.z, lane), diagnostics_sum_u32(filter0.w, lane));
    let final_filter_errors = diagnostics_sum_u32(filter_errors, lane);
    let final_bad_grid = diagnostics_sum_u32(bad_grid, lane);
    let final_completed_partials = diagnostics_sum_u32(completed_partials, lane);
    for (var bin = 0; bin < 16; bin += 1) {
        even[bin] = diagnostics_sum_f32(even[bin], lane);
    }

    if (lane == 0u) {
        diagnostics_output.grid.x = final_occupancy;
        diagnostics_output.grid.y = final_pore;
        diagnostics_output.pressure = final_pressure;
        diagnostics_output.pressure_ext = final_pressure_ext;
        diagnostics_output.even0 = vec4<f32>(even[0], even[1], even[2], even[3]);
        diagnostics_output.even1 = vec4<f32>(even[4], even[5], even[6], even[7]);
        diagnostics_output.even2 = vec4<f32>(even[8], even[9], even[10], even[11]);
        diagnostics_output.even3 = vec4<f32>(even[12], even[13], even[14], even[15]);
        diagnostics_output.filter0 = final_filter0;
        diagnostics_output.filter1 = vec4<u32>(final_filter_errors, final_bad_grid, 0u, 0u);
        diagnostics_output.snapshot_meta.y = select(
            0u, DIAGNOSTICS_GRID_CANARY, final_completed_partials == partial_count);
    }
}

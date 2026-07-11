// Compact production diagnostics. One 256-thread workgroup strided-scans the live water
// prefix, the solid suffix, and (when water is live) the water grid. It writes one 128-byte
// snapshot for an occasional asynchronous copy; no production getter maps a GPU buffer.

struct DiagnosticsConfig {
    cup0: vec4<f32>, // (center x, center z, radius, minimum y)
    cup1: vec4<f32>, // (maximum y, enabled, seeded bed top, reserved)
};

struct DiagnosticsOutput {
    mobile: vec4<f32>, // carrier mass, solute mass, heat capacity, enthalpy
    paper: vec4<f32>,  // carrier mass, solute mass, heat capacity, enthalpy
    grain: vec4<f32>,  // bound carrier, remaining solute, heat capacity, enthalpy
    cup: vec4<f32>,    // mobile carrier, mobile solute, pond carrier, reserved
    grid: vec4<f32>,   // maximum carrier-volume occupancy, reserved...
    counts: vec4<u32>, // pond particles, held parcels, held volume counts, bad particles
    filter0: vec4<u32>,// occupied counts, admitted, released, overflow
    filter1: vec4<u32>,// invariant flags, bad grid nodes, reserved, reserved
};

@group(0) @binding(31) var<storage, read_write> diagnostics_output: DiagnosticsOutput;
@group(0) @binding(32) var<uniform> diagnostics_config: DiagnosticsConfig;

var<workgroup> diagnostics_f32: array<f32, 256>;
var<workgroup> diagnostics_u32: array<u32, 256>;

fn diagnostics_finite4(value: vec4<f32>) -> bool {
    // WGSL has no portable isfinite builtin. NaN is the only float unequal to itself and every
    // finite simulation value is far below this guard, while either infinity exceeds it.
    return all(value == value) && all(abs(value) <= vec4<f32>(3.0e38));
}

fn diagnostics_sum_f32(value: f32, lane: u32) -> f32 {
    diagnostics_f32[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) {
            diagnostics_f32[lane] += diagnostics_f32[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }
    return diagnostics_f32[0];
}

fn diagnostics_max_f32(value: f32, lane: u32) -> f32 {
    diagnostics_f32[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) {
            diagnostics_f32[lane] = max(
                diagnostics_f32[lane], diagnostics_f32[lane + stride],
            );
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }
    return diagnostics_f32[0];
}

fn diagnostics_sum_u32(value: u32, lane: u32) -> u32 {
    diagnostics_u32[lane] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (lane < stride) {
            diagnostics_u32[lane] += diagnostics_u32[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }
    return diagnostics_u32[0];
}

@compute @workgroup_size(256)
fn diagnostics_reduce(@builtin(local_invocation_index) lane: u32) {
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
    var pond_count = 0u;
    var held_count = 0u;
    var held_volume_counts = 0u;
    var bad_particles = 0u;

    var p = lane;
    loop {
        if (p >= params.water_count) { break; }
        let position = pos[p];
        let state = chem[p];
        if (!diagnostics_finite4(position) || !diagnostics_finite4(state)) {
            bad_particles += 1u;
        } else {
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
                held_volume_counts += filter_volume_encode(
                    clamp(position.w, 0.0, 1.0) * params.wet0.z,
                );
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
        let state = chem[index];
        if (!diagnostics_finite4(position) || !diagnostics_finite4(state)) {
            bad_particles += 1u;
        } else {
            let bound = max(position.w, 0.0) * params.particle_mass
                / max(params.wet0.z, 1.0e-9);
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

    var maximum_occupancy = 0.0;
    var bad_grid_nodes = 0u;
    if (params.water_count > 0u) {
        var node = lane;
        loop {
            if (node >= params.grid_dims.w) { break; }
            let state = grid_vel[node];
            if (!diagnostics_finite4(state)) {
                bad_grid_nodes += 1u;
            } else {
                maximum_occupancy = max(
                    maximum_occupancy,
                    carrier_mass_at(node) / max(params.extra.x, 1.0e-9),
                );
            }
            node += 256u;
        }
    }

    let mobile = vec4<f32>(
        diagnostics_sum_f32(mobile_carrier, lane),
        diagnostics_sum_f32(mobile_solute, lane),
        diagnostics_sum_f32(mobile_capacity, lane),
        diagnostics_sum_f32(mobile_enthalpy, lane),
    );
    let paper = vec4<f32>(
        diagnostics_sum_f32(paper_carrier, lane),
        diagnostics_sum_f32(paper_solute, lane),
        diagnostics_sum_f32(paper_capacity, lane),
        diagnostics_sum_f32(paper_enthalpy, lane),
    );
    let grain = vec4<f32>(
        diagnostics_sum_f32(bound_carrier, lane),
        diagnostics_sum_f32(remaining_solute, lane),
        diagnostics_sum_f32(grain_capacity, lane),
        diagnostics_sum_f32(grain_enthalpy, lane),
    );
    let cup = vec4<f32>(
        diagnostics_sum_f32(cup_carrier, lane),
        diagnostics_sum_f32(cup_solute, lane),
        diagnostics_sum_f32(pond_carrier, lane),
        0.0,
    );
    let grid = vec4<f32>(diagnostics_max_f32(maximum_occupancy, lane), 0.0, 0.0, 0.0);
    let counts = vec4<u32>(
        diagnostics_sum_u32(pond_count, lane),
        diagnostics_sum_u32(held_count, lane),
        diagnostics_sum_u32(held_volume_counts, lane),
        diagnostics_sum_u32(bad_particles, lane),
    );
    let bad_grid = diagnostics_sum_u32(bad_grid_nodes, lane);

    if (lane == 0u) {
        diagnostics_output.mobile = mobile;
        diagnostics_output.paper = paper;
        diagnostics_output.grain = grain;
        diagnostics_output.cup = cup;
        diagnostics_output.grid = grid;
        diagnostics_output.counts = counts;
        diagnostics_output.filter0 = vec4<u32>(
            atomicLoad(&filter_state[FILTER_OCCUPIED]),
            atomicLoad(&filter_state[FILTER_ADMITTED]),
            atomicLoad(&filter_state[FILTER_RELEASED]),
            atomicLoad(&filter_state[FILTER_OVERFLOW]),
        );
        diagnostics_output.filter1 = vec4<u32>(
            atomicLoad(&filter_state[FILTER_ERRORS]), bad_grid, 0u, 0u,
        );
    }
}

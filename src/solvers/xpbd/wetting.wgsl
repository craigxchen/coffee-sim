// Wetting / absorption (XPBD step 4). Concatenated after common/water/bed/coupling.
//
// Dry grains absorb nearby water, conserving volume AND mass: the water volume transferred leaves
// the fluid phase (water's remaining-volume fraction f_w shrinks) and becomes grain volume (V_abs
// grows — the grain swells, applied in a later unit). Atomic-free and conservation-safe via a
// deterministic, water-normalized two-sided allocation read from the frozen pred.w snapshot:
//
//   take_wg = min( f_w·V_w / N_w , demand_g / N_g )
//
// computed identically in the water pass (sums Σ_g take, shrinks its own f_w) and the grain pass
// (sums Σ_w take into its own V_abs + the absorbed momentum). The two-sided cap bounds both per-water
// (Σ_g ≤ f_w·V_w) and per-grain (Σ_w ≤ demand_g ≤ deficit, no V_cap overshoot). Each pass writes only
// its own slot, so no atomics are needed. Runs after finalize; the next predict mirrors the new pos.w.
//
// moisture lane convention (pos.w / its frozen mirror pred.w): water = f_w ∈ [0,1], grain = V_abs ≥ 0.

fn wet_v_cap() -> f32 { return params.r_max * params.rho_ratio * params.grain_volume; }
fn wet_v_water() -> f32 { return params.particle_mass / params.rest_density; }
fn wet_demand(v_abs: f32) -> f32 {
    return max(wet_v_cap() - v_abs, 0.0) * (1.0 - exp(-params.k_abs * params.dt));
}

// Per-particle count of ELIGIBLE opposite-species neighbors in range (from the frozen pred.w):
// for a water this is N_w (# unsaturated grains), for a grain it is N_g (# non-empty waters). Both
// passes read this same buffer, so take_wg is identical on each side.
@compute @workgroup_size(256)
fn wet_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let ph_i = phase[i];
    let xi = pred[i].xyz;
    let v_cap = wet_v_cap();
    var n = 0u;
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (j == i) { continue; }
                    let ph_j = phase[j];
                    if (ph_j == ph_i) { continue; } // opposite species only
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.h) { continue; }
                    if (ph_j == PHASE_WATER) {
                        if (pred[j].w > params.absorb_roundoff) { n = n + 1u; } // non-empty water
                    } else {
                        if (pred[j].w < v_cap) { n = n + 1u; } // unsaturated grain (demand > 0)
                    }
                }
            }
        }
    }
    wet_neighbors[i] = n;
}

@compute @workgroup_size(256)
fn wet_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_WATER) { return; }
    let f_w = pred[i].w;
    let n_w = f32(wet_neighbors[i]);
    if (f_w <= params.absorb_roundoff || n_w <= 0.0) { return; } // inert / no eligible grains

    let v_cap = wet_v_cap();
    let v_w = wet_v_water();
    let xi = pred[i].xyz;
    var total = 0.0;
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (phase[j] != PHASE_GRAIN) { continue; }
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.h) { continue; }
                    let v_abs_g = pred[j].w;
                    if (v_abs_g >= v_cap) { continue; } // saturated grain ineligible
                    let n_g = f32(wet_neighbors[j]);
                    if (n_g <= 0.0) { continue; }
                    total = total + min(f_w * v_w / n_w, wet_demand(v_abs_g) / n_g);
                }
            }
        }
    }
    // Volume only leaves via the capped transfer (no force-dump); the cap guarantees total ≤ f_w·V_w.
    pos[i] = vec4<f32>(pos[i].xyz, max(f_w - total / v_w, 0.0));
}

@compute @workgroup_size(256)
fn wet_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_GRAIN) { return; }
    let v_abs = pred[i].w;
    let v_cap = wet_v_cap();
    let n_g = f32(wet_neighbors[i]);
    let demand_g = wet_demand(v_abs);
    if (demand_g <= 0.0 || n_g <= 0.0) { return; } // saturated / no eligible water

    let v_w = wet_v_water();
    let xi = pred[i].xyz;
    var d_v = 0.0;
    var p_abs = vec3<f32>(0.0);
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (phase[j] != PHASE_WATER) { continue; }
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.h) { continue; }
                    let f_w = pred[j].w;
                    if (f_w <= params.absorb_roundoff) { continue; }
                    let n_w = f32(wet_neighbors[j]);
                    if (n_w <= 0.0) { continue; }
                    // Identical formula + frozen snapshot as wet_water → water-loss == grain-gain.
                    let take = min(f_w * v_w / n_w, demand_g / n_g);
                    d_v = d_v + take;
                    p_abs = p_abs + take * params.rest_density * vel[j].xyz;
                }
            }
        }
    }
    // Exact per-substep inelastic merge of the absorbed mass (m_old is the pre-absorption mass).
    let m_old = params.grain_mass + params.rest_density * v_abs;
    let dm = params.rest_density * d_v;
    vel[i] = vec4<f32>((m_old * vel[i].xyz + p_abs) / (m_old + dm), 0.0);
    pos[i] = vec4<f32>(pos[i].xyz, v_abs + d_v);
}

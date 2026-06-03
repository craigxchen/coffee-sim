// Water/bed coupling (XPBD step 3, mechanical subset). Concatenated after common/water/bed.
//
// Two mechanisms, both active only in mixed (water+grain) scenes:
//  1. compute_fractions — per-particle solid fraction α_s = Σ_grain V_g W (clamped to the packing
//     limit). The water density solve (water.wgsl) reads it and targets ρ₀·(1−α_s), so pore water
//     packs to the pore fraction (drainage-ready) instead of full ρ₀ (which would be expelled).
//  2. grain exclusion (A.2) — a two-way geometric non-penetration between water and grain
//     particles: water can't pass through grain bodies, so it rests on / sits in the bed. Run as a
//     symmetric pair of gathers (exclude_water + exclude_grain) reading the same predicted
//     positions, with opposite-mass weights → momentum-conserving without float atomics.
//  3. implicit drag — symmetric frozen-velocity gathers using the Kozeny-Carman-resolved drag
//     rate in params.drag_gamma. The same pair scale is used by both phases, so each pair impulse
//     is equal-and-opposite to float tolerance.
//  4. buoyancy / pressure-gradient lift — symmetric gathers over the converged water λ field.
//     PBF λ is non-positive under compression in this solver, so pressure is −λ. The grain pass
//     applies +J/m_g and the water pass applies −J/m_w for the same pair, avoiding float atomics.

@compute @workgroup_size(256)
fn compute_fractions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let h = params.h;
    let xi = pred[i].xyz;
    var a_s = 0.0;
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (j == i) { continue; }
                    if (phase[j] != PHASE_GRAIN) { continue; } // solid fraction from grains only
                    let r = length(xi - pred[j].xyz);
                    if (r >= h) { continue; }
                    a_s = a_s + params.grain_volume * w_poly6(r, h);
                }
            }
        }
    }
    alpha_s[i] = min(a_s, params.packing_limit);
}

// Water↔grain contact separation for one pair, opposite-mass weighted. Both passes call this with
// the same (xi, xj) so the pair impulse is identical and conservation holds (to float tolerance).
fn exclusion_push(xi: vec3<f32>, xj: vec3<f32>, w_self: f32) -> vec3<f32> {
    let d_wg = params.grain_diameter; // water–grain contact distance
    let dvec = xi - xj;
    var r = length(dvec);
    if (r >= d_wg) { return vec3<f32>(0.0); }
    var n: vec3<f32>;
    if (r < 1e-6) {
        n = vec3<f32>(0.0, 1.0, 0.0); // coincident → push apart vertically (deterministic)
        r = 1e-6;
    } else {
        n = dvec / r;
    }
    return params.exclusion_relax * (d_wg - r) * w_self * n;
}

@compute @workgroup_size(256)
fn exclude_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    if (phase[i] != PHASE_WATER) { return; }
    let m_w = params.particle_mass;
    let m_g = params.grain_mass;
    let w_self = m_g / (m_w + m_g); // opposite-mass weight for the water side
    let xi = pred[i].xyz;
    var push = vec3<f32>(0.0);
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (phase[j] != PHASE_GRAIN) { continue; }
                    push = push + exclusion_push(xi, pred[j].xyz, w_self);
                }
            }
        }
    }
    // Add onto the water density Δp already in dp (compute_dp wrote it this iteration).
    dp[i] = dp[i] + vec4<f32>(push, 0.0);
}

@compute @workgroup_size(256)
fn exclude_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    if (phase[i] != PHASE_GRAIN) { return; }
    let m_w = params.particle_mass;
    let m_g = params.grain_mass;
    let w_self = m_w / (m_w + m_g); // opposite-mass weight for the grain side
    let xi = pred[i].xyz;
    var push = vec3<f32>(0.0);
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (phase[j] != PHASE_WATER) { continue; }
                    // Mirror of the water-side push for this pair (note the swapped argument order
                    // gives −n) with the grain's opposite-mass weight → exact pair antisymmetry.
                    push = push + exclusion_push(xi, pred[j].xyz, w_self);
                }
            }
        }
    }
    // Grains get no water-density Δp (compute_dp skips them), so SET dp here (the bed contact
    // subcycle that follows will overwrite it again). Always write, so a grain with no water
    // neighbors gets dp=0 rather than a stale value.
    dp[i] = vec4<f32>(push, 0.0);
}

fn drag_pair_beta() -> f32 {
    let x = max(params.drag_gamma * params.dt, 0.0);
    return x / (1.0 + x);
}

fn particle_mass_for_phase(ph: u32) -> f32 {
    if (ph == PHASE_GRAIN) {
        return params.grain_mass;
    }
    return params.particle_mass;
}

@compute @workgroup_size(256)
fn compute_coupling_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let ph_i = phase[i];
    let xi = pred[i].xyz;
    let beta = drag_pair_beta();
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
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (j == i) { continue; }
                    if (phase[j] == ph_i) { continue; }
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.h) { continue; }
                    total = total + beta;
                }
            }
        }
    }

    coupling_scale[i] = min(1.0, params.drag_beta_max / max(total, 1.0e-6));
}

fn drag_delta_for_pair(i: u32, j: u32, self_phase: u32) -> vec3<f32> {
    let beta = drag_pair_beta();
    let s = beta * min(coupling_scale[i], coupling_scale[j]);
    let m_i = particle_mass_for_phase(self_phase);
    let m_j = particle_mass_for_phase(phase[j]);
    return s * (m_j / (m_i + m_j)) * (vel_frozen[j].xyz - vel_frozen[i].xyz);
}

@compute @workgroup_size(256)
fn drag_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_WATER) { return; }
    let xi = pred[i].xyz;
    var dv = vec3<f32>(0.0);

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (phase[j] != PHASE_GRAIN) { continue; }
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.h) { continue; }
                    dv = dv + drag_delta_for_pair(i, j, PHASE_WATER);
                }
            }
        }
    }

    vel[i] = vec4<f32>(vel_frozen[i].xyz + dv, 0.0);
    fluid_impulse[i] = fluid_impulse[i]; // keep drag_water's auto-layout at the shared 8 buffers
}

@compute @workgroup_size(256)
fn drag_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_GRAIN) { return; }
    let xi = pred[i].xyz;
    var dv = vec3<f32>(0.0);

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (phase[j] != PHASE_WATER) { continue; }
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.h) { continue; }
                    dv = dv + drag_delta_for_pair(i, j, PHASE_GRAIN);
                }
            }
        }
    }

    vel[i] = vec4<f32>(vel_frozen[i].xyz + dv, 0.0);
    fluid_impulse[i] = fluid_impulse[i] + params.grain_mass * length(dv);
}

fn pressure_for_water(i: u32) -> f32 {
    return max(-lambda[i], 0.0);
}

@compute @workgroup_size(256)
fn buoyancy_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_GRAIN) { return; }
    let xi = pred[i].xyz;
    var impulse = vec3<f32>(0.0);

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (phase[j] != PHASE_WATER) { continue; }
                    let d = xi - pred[j].xyz;
                    let r = length(d);
                    if (r >= params.h) { continue; }
                    impulse = impulse + params.buoyancy_scale * pressure_for_water(j) *
                        spiky_grad(d, params.h, params.spiky_r_min);
                }
            }
        }
    }

    let dv = impulse / params.grain_mass;
    vel[i] = vec4<f32>(vel_frozen[i].xyz + dv, 0.0);
    fluid_impulse[i] = fluid_impulse[i] + params.grain_mass * length(dv);
}

@compute @workgroup_size(256)
fn buoyancy_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_WATER) { return; }
    let xi = pred[i].xyz;
    let pressure = pressure_for_water(i);
    var impulse = vec3<f32>(0.0);

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (phase[j] != PHASE_GRAIN) { continue; }
                    let d = xi - pred[j].xyz;
                    let r = length(d);
                    if (r >= params.h) { continue; }
                    // Grain pass pair: J_gw = s*p_w*∇W(x_g − x_w). Here d = x_w − x_g, so
                    // ∇W(d) = −∇W(x_g − x_w), making this exactly −J_gw for the same pair.
                    impulse = impulse + params.buoyancy_scale * pressure *
                        spiky_grad(d, params.h, params.spiky_r_min);
                }
            }
        }
    }

    vel[i] = vec4<f32>(vel_frozen[i].xyz + impulse / params.particle_mass, 0.0);
}

@compute @workgroup_size(256)
fn apply_drag_pred(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let dv = vel[i].xyz - vel_frozen[i].xyz;
    pred[i] = vec4<f32>(clamp(pred[i].xyz + params.dt * dv, params.box_min.xyz, params.box_max.xyz), 0.0);
}

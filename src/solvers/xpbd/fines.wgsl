// Fines migration (XPBD Phase 6). Concatenated after common/water/bed/coupling/wetting/extraction.
//
// Fines are a MASSLESS clogging/transport scalar in chem.w (grain = lodged fines, water = suspended
// fines). They erode grain→water where the local Darcy flux is high and deposit water→grain where
// it is low (mirrors models::fines::net_rate), advect for free on the carrying water particle, and
// feed local permeability via the α_s deviation (compute_fractions, U4). They carry NO inertial
// mass, so this transfer moves the chem.w scalar only and writes no velocities — water/grain
// momentum is untouched (the conservation claim is volume-only, with a momentum-non-perturbation
// guard).
//
// Conservation (volume): a signed-net two-sided transfer, atomic-free, mirroring the wetting trio.
// `fines_count` writes the shared `wet_neighbors` buffer = # opposite-species neighbors in range
// (the wetting block has finished for the substep, so the buffer is free to reuse). `fines_grain`
// and `fines_water` read the SAME frozen chem snapshot, the SAME counts, and the SAME per-pair flux,
// and compute the IDENTICAL signed `take`, each writing only its own chem.w slot → grain-loss ==
// water-gain exactly. All passes read live pos/pos.w (post-wetting), NOT pred (KTD-5).

// A water particle can carry suspended fines up to its current active volume (→ 0 as it drains, so a
// near-empty water can't hoard fines — KTD-10). Generous; rarely the binding constraint.
fn fines_susp_cap(f_w: f32) -> f32 {
    let v_w = params.particle_mass / params.rest_density;
    return f_w * v_w;
}
// Lodged fines fill pore space; cap at one grain volume so the α_s deviation stays bounded.
fn fines_lodged_cap() -> f32 { return params.grain_volume; }

// Signed transfer rate constant from local flux (mirror of models::fines::net_rate). Positive ⇒
// erosion (grain→water), negative ⇒ deposition (water→grain), zero at the critical flux.
fn fines_net_rate(flux: f32) -> f32 {
    let f = max(flux, 0.0);
    let c = max(params.fines.z, 1.0e-6);
    return params.fines.x * (f - c) / (f + c);
}
// Bounded per-step transfer fraction from a signed rate: 1 − e^{−|k|·dt} ∈ [0,1).
fn fines_frac(k: f32) -> f32 { return 1.0 - exp(-abs(k) * params.dt); }

// Symmetric roundoff floor: a sub-threshold take is zeroed identically on both sides, so the
// asymptotic tail can't generate a one-signed sub-ulp leak (the wet_sat_cutoff lesson). Applied to
// the per-pair take in both passes, so conservation is preserved.
fn fines_floor(take: f32) -> f32 {
    if (take < 1.0e-7 * params.grain_volume) { return 0.0; }
    return take;
}

@compute @workgroup_size(256)
fn fines_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let ph_i = phase[i];
    let xi = pos[i].xyz;
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
                    if (phase[j] == ph_i) { continue; } // opposite species only
                    let r = length(xi - pos[j].xyz);
                    if (r >= params.h) { continue; }
                    n = n + 1u;
                }
            }
        }
    }
    wet_neighbors[i] = n;
}

@compute @workgroup_size(256)
fn fines_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_GRAIN) { return; }
    let n_i = f32(wet_neighbors[i]);
    if (n_i <= 0.0) { return; }
    let f_grain = chem_frozen[i].w; // lodged fines (frozen)
    let lodged_headroom = max(fines_lodged_cap() - f_grain, 0.0);
    let xi = pos[i].xyz;
    let vi = vel[i].xyz;
    var delta = 0.0; // net change to this grain's lodged fines
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
                    let r = length(xi - pos[j].xyz);
                    if (r >= params.h) { continue; }
                    let n_j = f32(wet_neighbors[j]);
                    if (n_j <= 0.0) { continue; }
                    let k = fines_net_rate(length(vel[j].xyz - vi));
                    let frac = fines_frac(k);
                    if (k >= 0.0) {
                        // erosion grain→water: grain loses. Capped by grain source + water headroom.
                        let erodable = frac * f_grain / n_i;
                        let w_headroom = max(fines_susp_cap(pos[j].w) - chem_frozen[j].w, 0.0) / n_j;
                        delta = delta - fines_floor(min(erodable, w_headroom));
                    } else {
                        // deposition water→grain: grain gains. Capped by water source + grain headroom.
                        let depositable = frac * chem_frozen[j].w / n_j;
                        let g_headroom = lodged_headroom / n_i;
                        delta = delta + fines_floor(min(depositable, g_headroom));
                    }
                }
            }
        }
    }
    chem[i] = vec4<f32>(chem_frozen[i].xyz, max(f_grain + delta, 0.0));
}

@compute @workgroup_size(256)
fn fines_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_WATER) { return; }
    let n_i = f32(wet_neighbors[i]);
    if (n_i <= 0.0) { return; }
    let f_water = chem_frozen[i].w; // suspended fines (frozen)
    let susp_headroom = max(fines_susp_cap(pos[i].w) - f_water, 0.0);
    let xi = pos[i].xyz;
    let vi = vel[i].xyz;
    var delta = 0.0; // net change to this water's suspended fines
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
                    let r = length(xi - pos[j].xyz);
                    if (r >= params.h) { continue; }
                    let n_j = f32(wet_neighbors[j]); // grain's water-neighbor count
                    if (n_j <= 0.0) { continue; }
                    let f_grain_j = chem_frozen[j].w;
                    let k = fines_net_rate(length(vi - vel[j].xyz));
                    let frac = fines_frac(k);
                    if (k >= 0.0) {
                        // erosion grain→water: water gains. IDENTICAL operands to the grain pass.
                        let erodable = frac * f_grain_j / n_j;
                        let w_headroom = susp_headroom / n_i;
                        delta = delta + fines_floor(min(erodable, w_headroom));
                    } else {
                        // deposition water→grain: water loses. IDENTICAL operands to the grain pass.
                        let depositable = frac * f_water / n_i;
                        let g_headroom = max(fines_lodged_cap() - f_grain_j, 0.0) / n_j;
                        delta = delta - fines_floor(min(depositable, g_headroom));
                    }
                }
            }
        }
    }
    chem[i] = vec4<f32>(chem_frozen[i].xyz, max(f_water + delta, 0.0));
}

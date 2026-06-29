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

// Deep-bed filtration: two flow-driven processes (mirror of models::fines). Grains RELEASE lodged
// fines into the flowing water (mobilization) and suspended fines STRAIN back onto grains as the
// water flows past (capture). Straining concentrates fines where the most water funnels through —
// the converging outlet / filter — so the bed clogs there and drawdown slows. Equal coefficients let
// advection carry released fines downstream before they re-strain, building the bottom clog.
const FINES_RELEASE_COEF: f32 = 1.0;
const FINES_STRAIN_COEF: f32 = 1.0;
// Per-step transfer fraction: 1 − e^{−coef·rate·(flux/crit)·dt} ∈ [0,1). Scales with local flux
// normalized by crit_flux (params.fines.z): zero at rest, and a huge crit_flux freezes the transfer.
fn fines_flow_frac(flux: f32, coef: f32) -> f32 {
    let x = coef * params.fines.x * (max(flux, 0.0) / max(params.fines.z, 1.0e-6)) * params.dt;
    return 1.0 - exp(-x);
}

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
                    let flux = length(vel[j].xyz - vi);
                    // Release (grain→water): grain sheds lodged fines, capped by water headroom.
                    let rel_frac = fines_flow_frac(flux, FINES_RELEASE_COEF);
                    let release = fines_floor(min(
                        rel_frac * f_grain / n_i,
                        max(fines_susp_cap(pos[j].w) - chem_frozen[j].w, 0.0) / n_j,
                    ));
                    // Strain (water→grain): suspended fines deposit onto this grain, capped by its
                    // lodged headroom.
                    let str_frac = fines_flow_frac(flux, FINES_STRAIN_COEF);
                    let strain = fines_floor(min(
                        str_frac * chem_frozen[j].w / n_j,
                        lodged_headroom / n_i,
                    ));
                    delta = delta + strain - release; // grain gains strained fines, loses released
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
                    let flux = length(vi - vel[j].xyz);
                    // Release (grain→water): water gains shed fines. IDENTICAL operands to the grain
                    // pass (frozen state, same n's, same flux) ⇒ grain-loss == water-gain exactly.
                    let rel_frac = fines_flow_frac(flux, FINES_RELEASE_COEF);
                    let release = fines_floor(min(
                        rel_frac * f_grain_j / n_j,
                        max(fines_susp_cap(pos[i].w) - f_water, 0.0) / n_i,
                    ));
                    // Strain (water→grain): this water deposits suspended fines onto the grain.
                    let str_frac = fines_flow_frac(flux, FINES_STRAIN_COEF);
                    let strain = fines_floor(min(
                        str_frac * f_water / n_i,
                        max(fines_lodged_cap() - f_grain_j, 0.0) / n_j,
                    ));
                    delta = delta + release - strain; // water gains released fines, loses strained
                }
            }
        }
    }
    chem[i] = vec4<f32>(chem_frozen[i].xyz, max(f_water + delta, 0.0));
}

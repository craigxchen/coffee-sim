// Extraction (XPBD step 5, Phase 1.5). Concatenated after common/water/bed/coupling/wetting.
//
// Wet grains dissolve two-pool solute into overlapping water's concentration `c`, conserving the
// solute inventory exactly without atomics — the same frozen-snapshot, two-sided, count-normalized
// allocation as the wetting transfer:
//
//   take_gw = min( release_g / N_w ,  max(c_sat − c_w, 0)·(f_w·V_w) / N_g )
//
// computed identically in `dissolve_grain` (decrements its pools by Σ_w take, split proportionally
// across the two pools) and `dissolve_water` (adds Σ_g take to its own `c`). Both read the FROZEN
// chem snapshot (`chem_frozen`) + the frozen counts/flux (`diss_neighbors`) and write only the live
// `chem`, so grain-loss == water-gain to float tolerance and there is no read/write race. The grain's
// `release` is a pure function of its frozen state + per-grain aggregate flux, so the water pass
// recomputes the same value. The `(1−c/c_sat)` driving force is realized as the water-headroom cap.
//
// Runs AFTER wetting/finalize, so it reads `pos` for both position (pos.xyz == pred.xyz post-finalize,
// matching the rebuilt grid) and the POST-WETTING moisture (pos.w) — the grain's current saturation
// gates extraction and the water's current volume sizes the headroom. The dissolution passes write
// only `chem`, so reading `pos` is race-free. (Using `pos` for both also keeps both transfer passes
// lean — no separate `pred` binding needed.)
//
// Eligibility (identical on all three passes): opposite species within `h`, and both particles
// "active" — grain wet (pos.w = V_abs > absorb_roundoff) and water mobile (pos.w = f_w > roundoff).
// chem lanes: grain (s_f, s_s, T_g, _), water (c, T_w, _, _). moisture lane pos.w as in wetting.

// Per-grain two-pool release this substep (rf, rs, rf+rs), a pure function of the grain's FROZEN
// state + aggregate flux (so both transfer passes recompute it identically). No driving term here —
// the (1−c/c_sat) driving force lives in the per-pair water-headroom cap.
fn diss_release(g: u32) -> vec3<f32> {
    let cf = chem_frozen[g];        // (s_f, s_s, T_g, _)
    let flux_g = diss_neighbors[g].y;
    let sat = clamp(pos[g].w / wet_v_cap(), 0.0, 1.0); // grain moisture saturation (V_abs / V_cap)
    let kbase = params.extract_rate
        * ex_arrhenius(cf.z, params.ea_over_r, params.t_ref)
        * ex_area(params.grain_diameter, params.d_ref)
        * ex_flux(flux_g, params.u_half)
        * ex_wet_gate(sat, params.s_on);
    let rf = ex_release(cf.x, kbase * params.k0_fast, params.dt);
    let rs = ex_release(cf.y, kbase * params.k0_slow, params.dt);
    return vec3<f32>(rf, rs, rf + rs);
}

// Per-particle eligible-opposite-species count + (for grains) the aggregate relative-velocity flux,
// from the post-wetting positions/moisture + finalized velocity. Both transfer passes read this.
@compute @workgroup_size(256)
fn diss_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let ph_i = phase[i];
    let xi = pos[i].xyz;
    let vi = vel[i].xyz;
    var n = 0u;
    var flux_acc = 0.0;
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
                    if (length(xi - pos[j].xyz) >= params.h) { continue; }
                    if (pos[j].w <= params.absorb_roundoff) { continue; } // inactive neighbor
                    n = n + 1u;
                    if (ph_i == PHASE_GRAIN) { flux_acc = flux_acc + length(vi - vel[j].xyz); }
                }
            }
        }
    }
    var flux_g = 0.0;
    if (ph_i == PHASE_GRAIN && n > 0u) { flux_g = flux_acc / f32(n); }
    diss_neighbors[i] = vec2<f32>(f32(n), flux_g);
}

@compute @workgroup_size(256)
fn dissolve_grain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_GRAIN) { return; }
    if (pos[i].w <= params.absorb_roundoff) { return; } // dry grain doesn't extract
    let rel = diss_release(i);
    let n_w = diss_neighbors[i].x;
    if (rel.z <= 0.0 || n_w <= 0.0) { return; } // no pools/rate or no eligible water

    let v_w = wet_v_water();
    let xi = pos[i].xyz;
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
                    if (phase[j] != PHASE_WATER) { continue; }
                    if (length(xi - pos[j].xyz) >= params.h) { continue; }
                    let f_w = pos[j].w;
                    if (f_w <= params.absorb_roundoff) { continue; }
                    let n_g = diss_neighbors[j].x;
                    if (n_g <= 0.0) { continue; }
                    let headroom = max(params.c_sat - chem_frozen[j].x, 0.0) * (f_w * v_w);
                    total = total + min(rel.z / n_w, headroom / n_g);
                }
            }
        }
    }
    // Deplete the two pools proportionally (Σtake ≤ rel.z ⇒ pools stay ≥ 0). Base off the frozen
    // snapshot; T lane preserved.
    let cf = chem_frozen[i];
    let sf = max(cf.x - total * rel.x / rel.z, 0.0);
    let ss = max(cf.y - total * rel.y / rel.z, 0.0);
    chem[i] = vec4<f32>(sf, ss, cf.z, cf.w);
}

@compute @workgroup_size(256)
fn dissolve_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_WATER) { return; }
    let f_w = pos[i].w;
    if (f_w <= params.absorb_roundoff) { return; } // immobile water
    let n_g = diss_neighbors[i].x;
    if (n_g <= 0.0) { return; }

    let v_w = wet_v_water();
    let cf = chem_frozen[i];
    let headroom = max(params.c_sat - cf.x, 0.0) * (f_w * v_w);
    let xi = pos[i].xyz;
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
                    if (length(xi - pos[j].xyz) >= params.h) { continue; }
                    if (pos[j].w <= params.absorb_roundoff) { continue; } // dry grain
                    let rel = diss_release(j);
                    let n_w = diss_neighbors[j].x;
                    if (rel.z <= 0.0 || n_w <= 0.0) { continue; }
                    // Identical take + frozen snapshot as dissolve_grain → grain-loss == water-gain.
                    total = total + min(rel.z / n_w, headroom / n_g);
                }
            }
        }
    }
    // Concentration rises by the dissolved solute over the remaining water volume (≤ c_sat by the
    // headroom cap). T lane preserved.
    chem[i] = vec4<f32>(cf.x + total / (f_w * v_w), cf.y, cf.z, cf.w);
}

// --- Thermal exchange (U6) -------------------------------------------------------------------------
// Per-particle lumped heat conduction + ambient loss. The temperature lane differs by species
// (grain T = chem.z, water T = chem.y), so read/write go through phase-aware helpers. Capacity
// C = effective_mass · specific_heat (water mass scales with f_w; wet-grain mass includes the
// absorbed water ρ·V_abs). Runs after dissolution on a FRESH chem_frozen snapshot, so it preserves
// the post-dissolution c/pools (it writes only the T lane). Reads `pos` for moisture/position (same
// rationale as the dissolution passes); the grid was rebuilt for diss_count and is still valid.

fn th_temp(ph: u32, cf: vec4<f32>) -> f32 {
    return select(cf.y, cf.z, ph == PHASE_GRAIN); // water T at .y, grain T at .z
}
fn th_capacity(ph: u32, w: f32) -> f32 {
    let cp = select(params.cp_water, params.cp_grain, ph == PHASE_GRAIN);
    return eff_mass(ph, w) * cp;
}

@compute @workgroup_size(256)
fn thermal_exchange(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let ph_i = phase[i];
    let cf_i = chem_frozen[i];
    let t_i = th_temp(ph_i, cf_i);
    let c_i = th_capacity(ph_i, pos[i].w);

    var q_sum = 0.0;
    let xi = pos[i].xyz;
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
                    if (length(xi - pos[j].xyz) >= params.h) { continue; }
                    let ph_j = phase[j];
                    let c_j = th_capacity(ph_j, pos[j].w);
                    if (c_j <= 1e-6) { continue; } // empty neighbor carries no heat
                    let t_j = th_temp(ph_j, chem_frozen[j]);
                    // Antisymmetric pair heat (q_ji = −q_ij) → Σ C·T conserved across the pass.
                    q_sum = q_sum + th_pair_heat(t_i, c_i, t_j, c_j, params.kappa, params.dt);
                }
            }
        }
    }

    var t_new = t_i;
    if (c_i > 1e-6) { t_new = t_i + q_sum / c_i; } // zero-capacity particle exchanges nothing
    t_new = t_new + th_ambient(t_new, params.t_amb, params.h_amb, params.dt);

    // Write only the T lane, preserving the post-dissolution c/pools from the frozen snapshot.
    if (ph_i == PHASE_GRAIN) {
        chem[i] = vec4<f32>(cf_i.x, cf_i.y, t_new, cf_i.w);
    } else {
        chem[i] = vec4<f32>(cf_i.x, t_new, cf_i.z, cf_i.w);
    }
}

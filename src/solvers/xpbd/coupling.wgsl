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
//  3. capped-pair Darcy drag — symmetric frozen-velocity gathers using local Kozeny-Carman
//     porosity rates and the opposite-phase neighbor count. The same pair scale is used by both
//     phases, so each pair impulse is equal-and-opposite to float tolerance.
//  4. buoyancy / pressure-gradient lift — symmetric gathers over the converged water λ field.
//     PBF λ is non-positive under compression in this solver, so pressure is −λ. The grain pass
//     applies +J/m_g and the water pass applies −J/m_w for the same pair, avoiding float atomics.

@compute @workgroup_size(256)
fn compute_fractions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let h = params.coupling_h;
    let xi = pred[i].xyz;
    var a_s = 0.0;
    var fines_dev = 0.0; // Σ_j (lodged fines − uniform baseline)·W from grain neighbors (Phase 6)
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
                    if (phase[j] != PHASE_GRAIN) { continue; } // solid fraction from grains only
                    let r = length(xi - pred[j].xyz);
                    if (r >= h) { continue; }
                    // Swollen grains occupy more space: use the effective volume V_eff = V_dry + V_abs
                    // (grain pred.w = V_abs). Dry grains (V_abs=0) ⇒ params.grain_volume, unchanged.
                    let w = w_poly6(r, h);
                    a_s = a_s + grain_eff_volume(pred[j].w) * w;
                    // Migration-induced fines deviation from the seeded baseline (grain chem.w).
                    fines_dev = fines_dev + (chem[j].w - params.fines.y) * w;
                }
            }
        }
    }
    // Grain-skeleton packing clamp first, then the fines pore-clogging deviation ON TOP (KTD-2):
    // lodged fines fill the pore space the packing limit leaves open, so accumulation can push α_s
    // above packing_limit (toward the φ_f floor) while erosion opens channels. Baseline-relative, so
    // fines_rate=0 leaves α_s exactly min(a_s, packing_limit) — byte-unchanged.
    var result = min(a_s, params.packing_limit);
    if (params.fines.x > 0.0) {
        result = clamp(result + fines_dev, 0.0, 0.95); // 0.95 = 1 − φ_f_min (porosity_drag_factor)
    }
    alpha_s[i] = result;
}

// Water↔grain contact separation for one pair, opposite-mass weighted. Both passes call this with
// the same (xi, xj) so the pair impulse is identical and conservation holds (to float tolerance).
fn exclusion_push(xi: vec3<f32>, xj: vec3<f32>, w_self: f32) -> vec3<f32> {
    let d_wg = params.water_grain_distance; // water–grain contact distance (< grain spacing ⇒ pores)
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
    // Effective masses (swelling): heavier wet grains / lighter shrunk water shift the split. Dry
    // (V_abs=0, f_w=1) ⇒ the original particle_mass/grain_mass split. w_self computed per neighbor.
    let m_w = water_eff_mass(pred[i].w);
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
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (phase[j] != PHASE_GRAIN) { continue; }
                    let m_g = grain_eff_mass(pred[j].w);
                    push = push + exclusion_push(xi, pred[j].xyz, m_g / (m_w + m_g));
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
    // Effective masses (swelling), mirror of exclude_water. Same per-pair m_w/m_g read from pred.w on
    // both sides ⇒ the antisymmetric correction still conserves momentum exactly.
    let m_g = grain_eff_mass(pred[i].w);
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
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (phase[j] != PHASE_WATER) { continue; }
                    // Mirror of the water-side push for this pair (swapped args give −n); grain's
                    // opposite-mass weight m_w/(m_w+m_g) → exact pair antisymmetry.
                    let m_w = water_eff_mass(pred[j].w);
                    push = push + exclusion_push(xi, pred[j].xyz, m_w / (m_w + m_g));
                }
            }
        }
    }
    // Grains get no water-density Δp (compute_dp skips them), so SET dp here (the bed contact
    // subcycle that follows will overwrite it again). Always write, so a grain with no water
    // neighbors gets dp=0 rather than a stale value.
    dp[i] = vec4<f32>(push, 0.0);
}

fn drag_alpha_for_particle_from_alpha(i: u32, ph_i: u32, base_alpha: f32) -> f32 {
    var a = base_alpha;
    // α_s excludes self to keep the water density field a neighbor sum. For a grain's own Darcy
    // probe, include its occupied volume so an isolated or surface grain still carries a finite
    // packed-bed resistance in the symmetric harmonic pair rate.
    if (ph_i == PHASE_GRAIN) {
        a = a + grain_eff_volume(pred[i].w) * w_poly6(0.0, params.coupling_h);
    }
    return min(a, 0.95);
}

fn darcy_beta_from_alpha(a_s: f32) -> f32 {
    let eps = clamp(1.0 - a_s, 0.35, 0.999);
    let solid = 1.0 - eps;
    let d2 = max(params.grain_diameter * params.grain_diameter, 1.0e-8);
    return params.drag_scale * 150.0 * solid * solid / max(eps * eps * eps * d2, 1.0e-12);
}

fn harmonic_pair(a: f32, b: f32) -> f32 {
    return (2.0 * a * b) / max(a + b, 1.0e-12);
}

fn particle_volume(ph: u32, w: f32) -> f32 {
    if (ph == PHASE_GRAIN) { return grain_eff_volume(w); }
    return max(water_eff_mass(w) / max(params.rest_density, 1.0e-12), 0.0);
}

@compute @workgroup_size(256)
fn compute_coupling_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let ph_i = phase[i];
    let xi = pred[i].xyz;
    var a_scan = 0.0;
    var n = 0.0;

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
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.coupling_h) { continue; }
                    if (phase[j] == PHASE_GRAIN) {
                        a_scan = a_scan + grain_eff_volume(pred[j].w) * w_poly6(r, params.coupling_h);
                    }
                    if (phase[j] != ph_i) {
                        n = n + 1.0;
                    }
                }
            }
        }
    }

    let a_from_neighbors = min(a_scan, params.packing_limit);
    let beta_i = darcy_beta_from_alpha(drag_alpha_for_particle_from_alpha(
        i,
        ph_i,
        max(alpha_s[i], a_from_neighbors),
    ));
    coupling_scale[i] = vec2<f32>(beta_i, n);
}

fn drag_delta_for_pair(i: u32, j: u32, self_phase: u32) -> vec3<f32> {
    let cs_i = coupling_scale[i];
    let cs_j = coupling_scale[j];
    let beta_pair = harmonic_pair(cs_i.x, cs_j.x);
    let cap_pair = params.drag_beta_max / max(max(cs_i.y, cs_j.y), 1.0);
    let v_pair = harmonic_pair(
        particle_volume(self_phase, pred[i].w),
        particle_volume(phase[j], pred[j].w),
    );
    let s = min(cap_pair, max(beta_pair, 0.0) * v_pair * params.dt);
    // Effective masses (swelling) keep momentum conserved.
    let m_i = eff_mass(self_phase, pred[i].w);
    let m_j = eff_mass(phase[j], pred[j].w);
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
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (phase[j] != PHASE_GRAIN) { continue; }
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.coupling_h) { continue; }
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
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (phase[j] != PHASE_WATER) { continue; }
                    let r = length(xi - pred[j].xyz);
                    if (r >= params.coupling_h) { continue; }
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
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
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

    let m_g = grain_eff_mass(pred[i].w); // swelling: heavier wet grain accelerates less
    // Density-aware: a denser-than-water grain is buoyed less (sinks); same factor on the water side
    // (per pair) keeps momentum conserved. Uniform over this grain's pairs, so scale the total.
    let dv = grain_buoyancy_factor(pred[i].w) * impulse / m_g;
    vel[i] = vec4<f32>(vel_frozen[i].xyz + dv, 0.0);
    fluid_impulse[i] = fluid_impulse[i] + m_g * length(dv);
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
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (phase[j] != PHASE_GRAIN) { continue; }
                    let d = xi - pred[j].xyz;
                    let r = length(d);
                    if (r >= params.h) { continue; }
                    // Grain pass pair: J_gw = s*p_w*∇W(x_g − x_w). Here d = x_w − x_g, so
                    // ∇W(d) = −∇W(x_g − x_w), making this exactly −J_gw for the same pair. The grain's
                    // density factor matches the grain-pass scaling → equal-and-opposite per pair.
                    impulse = impulse + grain_buoyancy_factor(pred[j].w) * params.buoyancy_scale *
                        pressure * spiky_grad(d, params.h, params.spiky_r_min);
                }
            }
        }
    }

    // Effective water mass, floored so a near-empty (absorbed) particle can't blow up the divide.
    let m_w = max(water_eff_mass(pred[i].w), params.particle_mass * params.pbf_eps);
    vel[i] = vec4<f32>(vel_frozen[i].xyz + impulse / m_w, 0.0);
}

@compute @workgroup_size(256)
fn apply_drag_pred(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let dv = vel[i].xyz - vel_frozen[i].xyz;
    var p = pred[i].xyz + params.dt * dv;
    // Static-solid push-out (push-out + species gate only; friction is owned by apply_dp/finalize, so
    // the drag micro-step needs neither pos nor the floor_mu machinery). No-op when num_solids == 0.
    if (params.num_solids > 0u) {
        let hit = solid_union(p, phase[i]);
        // Water projects to the surface (offset 0, dissipative); grains keep a radius standoff.
        let contact = select(0.0, 0.5 * params.grain_diameter, phase[i] == PHASE_GRAIN);
        if (hit.dist < contact) {
            p = p + (contact - hit.dist) * hit.grad;
        }
    }
    // Preserve the moisture snapshot in pred.w — this runs in the drag/buoyancy subcycle before
    // absorption, so zeroing .w here would wipe the per-frame snapshot the wetting passes read.
    pred[i] = vec4<f32>(clamp(p, params.box_min.xyz, params.box_max.xyz), pred[i].w);
}

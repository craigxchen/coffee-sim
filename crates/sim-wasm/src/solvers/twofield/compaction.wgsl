// Semi-implicit granular split for the production MPM-v2 path.
//
// Physical volumetric response is a cell-centered skeleton pressure q. It uses the exact
// collocated D/G pair from pressure.wgsl with a solid-only node mobility B_s, then applies the
// identical B_s and G in compaction_project:
//
//   A_s = -D B_s G,
//   (I + dt^2 kappa A_s) q = p_bar - dt kappa D v_s*,
//   v_s+ = v_s* - dt B_s G q,                 0 <= q <= p_c.
//
// p_bar comes from the bounded-tangent packing law evaluated at
// phi_e = phi_s exp(-e_p). The cap is fixed during one solve, keeping the projected Jacobi
// problem convex; geometric return + exponential hardening update e_p after the solve and are
// visible on the next macrostep. Physical bulk stiffness never enters the explicit CFL path.
// The only explicitly subcycled stress is a deliberately soft, traceless numerical shear.

const COMPACTION_OMEGA: f32 = 0.6666667;
const COMPACTION_PHI_MAX: f32 = 0.95;

struct PackingResponse { pressure: f32, tangent: f32 };

fn packing_reference_fraction() -> f32 {
    // splas1.y is the declared reference solid fraction on the implicit path. It remains
    // independent of uniform sample-thinning quadrature carried by coupling.z.
    return clamp(params.splas1.y, 1.0e-4, 0.9499);
}

fn packing_response(phi_arg: f32) -> PackingResponse {
    let onset = packing_reference_fraction();
    let full = max(params.granular.z, onset + 1.0e-4);
    let stiffness = max(params.granular.x, 0.0);
    let x = max(phi_arg - onset, 0.0);
    if (x <= 0.0 || stiffness <= 0.0) {
        return PackingResponse(0.0, 0.0);
    }
    let span = full - onset;
    if (x < span) {
        return PackingResponse(
            0.5 * stiffness * x * x / span,
            stiffness * x / span,
        );
    }
    return PackingResponse(stiffness * (x - 0.5 * span), stiffness);
}

fn packing_inverse(pressure_arg: f32) -> f32 {
    let onset = packing_reference_fraction();
    let full = max(params.granular.z, onset + 1.0e-4);
    let stiffness = max(params.granular.x, 0.0);
    let pressure = max(pressure_arg, 0.0);
    if (pressure <= 0.0 || stiffness <= 0.0) {
        return onset;
    }
    let span = full - onset;
    let at_full = 0.5 * stiffness * span;
    if (pressure < at_full) {
        return onset + sqrt(2.0 * pressure * span / stiffness);
    }
    return onset + pressure / stiffness + 0.5 * span;
}

fn consolidation_pressure_for(ep: f32) -> f32 {
    return max(params.granular.y, 0.0) * exp(min(params.splas1.x * max(ep, 0.0), 40.0));
}

// Plastic-compaction moment divided by structural volume at one node.
fn node_plastic_compaction(n: u32) -> f32 {
    let volume = solid_volume_at(n);
    if (volume <= 0.0) {
        return 0.0;
    }
    return max(fp_decode(atomicLoad(&grid_scomp[n])) / volume, 0.0);
}

struct CompactionCellState { phi: f32, ep: f32 };

fn solid_node_visible_volume(xp: vec3<f32>) -> f32 {
    var visible = 0.0;
    let h = params.grid_origin.w;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let sign = corner_sign(vec3<i32>(ox, oy, oz));
                let center = xp + sign * (0.5 * h);
                var allowed = all(center >= params.box_min.xyz) && all(center <= params.box_max.xyz);
                if (allowed && params.num_solids > 0u) {
                    allowed = solid_union(center, PHASE_SOLID).dist >= 0.0;
                }
                if (allowed) { visible += 1.0; }
            }
        }
    }
    return h * h * h * max(visible, 1.0) / 8.0;
}

fn compaction_cell_state(c: vec3<i32>) -> CompactionCellState {
    var solid_volume = 0.0;
    var visible_volume = 0.0;
    var compaction_moment = 0.0;
    let h = params.grid_origin.w;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let node = c + vec3<i32>(ox, oy, oz);
                let n = node_index(node);
                let sv = solid_volume_at(n);
                solid_volume += sv;
                compaction_moment += fp_decode(atomicLoad(&grid_scomp[n]));
            }
        }
    }
    if (solid_volume <= 0.0) {
        return CompactionCellState(0.0, 0.0);
    }
    // The cell denominator includes every geometrically visible corner control volume, not
    // only mass-carrying nodes. Omitting empty surface corners would turn B-spline smear into a
    // falsely packed cell and manufacture a pressure skin.
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let node = c + vec3<i32>(ox, oy, oz);
                let xp = params.grid_origin.xyz + vec3<f32>(node) * h;
                visible_volume += solid_node_visible_volume(xp);
            }
        }
    }
    if (visible_volume <= 0.0) { return CompactionCellState(0.0, 0.0); }
    return CompactionCellState(
        clamp(solid_volume / visible_volume, 0.0, COMPACTION_PHI_MAX),
        max(compaction_moment / solid_volume, 0.0),
    );
}

// Clear only the solid grids. Water mass/momentum and cell occupancy can remain resident while
// the soft shear predictor takes multiple solid-only microsteps.
@compute @workgroup_size(256)
fn solid_grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) { return; }
    let m = n * 4u;
    atomicStore(&grid_sm[m + 0u], 0);
    atomicStore(&grid_sm[m + 1u], 0);
    atomicStore(&grid_sm[m + 2u], 0);
    atomicStore(&grid_sm[m + 3u], 0);
    atomicStore(&grid_sfp[n], 0);
    atomicStore(&grid_scomp[n], 0);
}

// Final macro P2G: no stress. It freezes mass, momentum, structural volume, and the plastic
// compaction moment at the positions produced by the solid shear predictor.
@compute @workgroup_size(256)
fn p2g_solid_macro(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) { return; }
    let p = params.particle_count - params.solid_count + i;
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let v = vel[p].xyz;
    let c0 = cmat[3u * p + 0u].xyz;
    let c1 = cmat[3u * p + 1u].xyz;
    let c2 = cmat[3u * p + 2u].xyz;
    let mass = grain_effective_mass(p);
    let volume = grain_structural_volume(pos[p].w);
    let ep = max(sstate[2u * i + 1u].w, 0.0);
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    let total_structure = fp_encode(volume);
    let total_capacity = fp_encode(max(params.solute.w, 0.0));
    let total_bloom_volume = fp_encode(volume * bloom_factor(chem[p].w));
    var allocated_structure = 0i;
    var allocated_capacity = 0i;
    var allocated_bloom_volume = 0i;
    var cumulative_weight = 0.0;
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var ii = 0; ii < 3; ii = ii + 1) {
                let weight = w[ii].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(ii, j, k);
                let d = (vec3<f32>(node) - xl) * h;
                var va = v + vec3<f32>(dot(c0, d), dot(c1, d), dot(c2, d));
                let speed = length(va);
                if (speed > params.max_speed) { va *= params.max_speed / speed; }
                let n = node_index(node);
                let mi = 4u * n;
                cumulative_weight += weight;
                let is_last = ii == 2 && j == 2 && k == 2;
                let structure_target = extensive_partition_target(
                    total_structure, cumulative_weight, is_last,
                );
                let capacity_target = extensive_partition_target(
                    total_capacity, cumulative_weight, is_last,
                );
                let bloom_target = extensive_partition_target(
                    total_bloom_volume, cumulative_weight, is_last,
                );
                atomicAdd(&grid_sfp[n], structure_target - allocated_structure);
                atomicAdd(
                    &cell_cnt[capillary_capacity_index(n)],
                    u32(max(capacity_target - allocated_capacity, 0)),
                );
                atomicAdd(
                    &cell_cnt[bloom_volume_index(n)],
                    u32(max(bloom_target - allocated_bloom_volume, 0)),
                );
                allocated_structure = structure_target;
                allocated_capacity = capacity_target;
                allocated_bloom_volume = bloom_target;
                atomicAdd(&grid_scomp[n], fp_encode(volume * ep * weight));
                atomicAdd(&grid_sm[mi + 0u], fp_encode(mass * weight));
                atomicAdd(&grid_sm[mi + 1u], fp_encode(mass * weight * va.x));
                atomicAdd(&grid_sm[mi + 2u], fp_encode(mass * weight * va.y));
                atomicAdd(&grid_sm[mi + 3u], fp_encode(mass * weight * va.z));
            }
        }
    }
}

// Soft deviatoric predictor P2G. Mean stress q is deliberately absent: the macro compaction
// solve is its sole owner.
@compute @workgroup_size(256)
fn p2g_solid_shear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) { return; }
    let p = params.particle_count - params.solid_count + i;
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let v = vel[p].xyz;
    let c0 = cmat[3u * p + 0u].xyz;
    let c1 = cmat[3u * p + 1u].xyz;
    let c2 = cmat[3u * p + 2u].xyz;
    let sa = sstate[2u * i + 0u];
    let sb = sstate[2u * i + 1u];
    let t0 = vec3<f32>(sa.x, sa.y, sa.z);
    let t1 = vec3<f32>(sa.y, sa.w, sb.x);
    let t2 = vec3<f32>(sa.z, sb.x, -sa.x - sa.w);
    let mass = grain_effective_mass(p);
    // The calibrated shear modulus is an effective bed stress per represented BULK volume.
    // coupling.z is the quadrature-weighted dry structural volume; dividing by the declared
    // reference solid fraction recovers the corresponding dry bulk control volume. It must not
    // grow with bound-water swelling, or the same modulus would silently stiffen as grains swell.
    let bulk_reference_volume = params.coupling.z / packing_reference_fraction();
    let force_scale = params.granular.w * bulk_reference_volume * 4.0
        / (h * h) / max(mass, 1.0e-9);
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var ii = 0; ii < 3; ii = ii + 1) {
                let weight = w[ii].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(ii, j, k);
                let d = (vec3<f32>(node) - xl) * h;
                var va = v
                    + vec3<f32>(dot(c0, d), dot(c1, d), dot(c2, d))
                    - force_scale * vec3<f32>(dot(t0, d), dot(t1, d), dot(t2, d));
                let speed = length(va);
                if (speed > params.max_speed) { va *= params.max_speed / speed; }
                let n = node_index(node) * 4u;
                atomicAdd(&grid_sm[n + 0u], fp_encode(mass * weight));
                atomicAdd(&grid_sm[n + 1u], fp_encode(mass * weight * va.x));
                atomicAdd(&grid_sm[n + 2u], fp_encode(mass * weight * va.y));
                atomicAdd(&grid_sm[n + 3u], fp_encode(mass * weight * va.z));
            }
        }
    }
}

fn decode_solid_node(n: u32, dt_force: f32) {
    let mc = atomicLoad(&grid_sm[n * 4u]);
    let mass = fp_decode(mc);
    if (mass <= params.extra.z) {
        grid_svel[n] = vec4<f32>(vec3<f32>(0.0), mass);
        return;
    }
    let inv = 1.0 / f32(mc);
    var v = vec3<f32>(
        f32(atomicLoad(&grid_sm[n * 4u + 1u])),
        f32(atomicLoad(&grid_sm[n * 4u + 2u])),
        f32(atomicLoad(&grid_sm[n * 4u + 3u])),
    ) * inv;
    v += params.gravity.xyz * dt_force;
    let c = node_coords(n);
    let xp = params.grid_origin.xyz + vec3<f32>(c) * params.grid_origin.w;
    grid_svel[n] = vec4<f32>(solid_contact_velocity_bc(xp, v), mass);
}

@compute @workgroup_size(256)
fn solid_predict_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.grid_dims.w) { return; }
    decode_solid_node(gid.x, params.granular.w);
}

@compute @workgroup_size(256)
fn solid_macro_decode(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.grid_dims.w) { return; }
    decode_solid_node(gid.x, 0.0);
}

struct SolidGather { v: vec3<f32>, c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32> };

fn gather_solid_velocity(x: vec3<f32>) -> SolidGather {
    let h = params.grid_origin.w;
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    var v = vec3<f32>(0.0);
    var b0 = vec3<f32>(0.0);
    var b1 = vec3<f32>(0.0);
    var b2 = vec3<f32>(0.0);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var ii = 0; ii < 3; ii = ii + 1) {
                let weight = w[ii].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(ii, j, k);
                let d = (vec3<f32>(node) - xl) * h;
                let gv = grid_svel[node_index(node)].xyz;
                v += weight * gv;
                b0 += (weight * gv.x) * d;
                b1 += (weight * gv.y) * d;
                b2 += (weight * gv.z) * d;
            }
        }
    }
    let dinv = 4.0 / (h * h);
    return SolidGather(v, b0 * dinv, b1 * dinv, b2 * dinv);
}

struct ParticleMotion { x: vec3<f32>, v: vec3<f32> };

fn constrain_solid_particle(x_arg: vec3<f32>, v_arg: vec3<f32>) -> ParticleMotion {
    var x = x_arg;
    var v = v_arg;
    if (x.x < params.box_min.x) { x.x = params.box_min.x; if (v.x < 0.0) { v.x = 0.0; } }
    if (x.y < params.box_min.y) { x.y = params.box_min.y; if (v.y < 0.0) { v.y = 0.0; } }
    if (x.z < params.box_min.z) { x.z = params.box_min.z; if (v.z < 0.0) { v.z = 0.0; } }
    if (x.x > params.box_max.x) { x.x = params.box_max.x; if (v.x > 0.0) { v.x = 0.0; } }
    if (x.y > params.box_max.y) { x.y = params.box_max.y; if (v.y > 0.0) { v.y = 0.0; } }
    if (x.z > params.box_max.z) { x.z = params.box_max.z; if (v.z > 0.0) { v.z = 0.0; } }
    if (params.num_solids > 0u) {
        let hit = solid_union(x, PHASE_SOLID);
        if (hit.dist < 0.0) {
            x += (-hit.dist) * hit.grad;
            let vn = dot(v, hit.grad);
            if (vn < 0.0) { v -= vn * hit.grad; }
        }
    }
    return ParticleMotion(x, v);
}

// Gather, advect, and update the isochoric deformation/shear return. `sstate[2i+1].y` is the
// lagged skeleton mean pressure furnished by the preceding macro solve.
@compute @workgroup_size(256)
fn g2p_solid_shear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) { return; }
    let p = params.particle_count - params.solid_count + i;
    let gathered = gather_solid_velocity(pos[p].xyz);
    var v = gathered.v;
    let speed = length(v);
    if (speed > params.max_speed) { v *= params.max_speed / speed; }
    let motion = constrain_solid_particle(pos[p].xyz + v * params.granular.w, v);

    let fr0 = fmat[3u * i + 0u];
    let fr1 = fmat[3u * i + 1u];
    let fr2 = fmat[3u * i + 2u];
    let fm = mat3x3<f32>(
        vec3<f32>(fr0.x, fr1.x, fr2.x),
        vec3<f32>(fr0.y, fr1.y, fr2.y),
        vec3<f32>(fr0.z, fr1.z, fr2.z),
    );
    let cm = mat3x3<f32>(
        vec3<f32>(gathered.c0.x, gathered.c1.x, gathered.c2.x),
        vec3<f32>(gathered.c0.y, gathered.c1.y, gathered.c2.y),
        vec3<f32>(gathered.c0.z, gathered.c1.z, gathered.c2.z),
    );
    let trial = (IDENT3 + params.granular.w * cm) * fm;
    let svd = svd3(trial);
    let sigma = clamp(abs(svd.sigma), vec3<f32>(SIG_MIN), vec3<f32>(SIG_MAX));
    var eps = log(sigma);
    eps -= vec3<f32>((eps.x + eps.y + eps.z) / 3.0);
    let shear = max(params.splas0.y, 0.0);
    var principal = 2.0 * shear * eps;
    let q = max(sstate[2u * i + 1u].y, 0.0);
    let y_coh = cohesion_for_saturation(grain_bound_saturation(pos[p].w));
    let radius = max(y_coh + 3.0 * params.splas0.w * q, 0.0);
    let stress_norm = length(principal);
    if (stress_norm > radius && stress_norm > 1.0e-12) {
        principal *= radius / stress_norm;
    }
    if (shear > 0.0) { eps = principal / (2.0 * shear); } else { eps = vec3<f32>(0.0); }
    eps -= vec3<f32>((eps.x + eps.y + eps.z) / 3.0);
    let sig_new = exp(eps);
    let f_new = svd.u
        * mat3x3<f32>(
            vec3<f32>(sig_new.x, 0.0, 0.0),
            vec3<f32>(0.0, sig_new.y, 0.0),
            vec3<f32>(0.0, 0.0, sig_new.z),
        )
        * transpose(svd.v);
    let tau = svd.u
        * mat3x3<f32>(
            vec3<f32>(principal.x, 0.0, 0.0),
            vec3<f32>(0.0, principal.y, 0.0),
            vec3<f32>(0.0, 0.0, principal.z),
        )
        * transpose(svd.u);

    pos[p] = vec4<f32>(motion.x, pos[p].w);
    vel[p] = vec4<f32>(motion.v, vel[p].w);
    cmat[3u * p + 0u] = vec4<f32>(gathered.c0, 0.0);
    cmat[3u * p + 1u] = vec4<f32>(gathered.c1, 0.0);
    cmat[3u * p + 2u] = vec4<f32>(gathered.c2, 0.0);
    fmat[3u * i + 0u] = vec4<f32>(f_new[0][0], f_new[1][0], f_new[2][0], 0.0);
    fmat[3u * i + 1u] = vec4<f32>(f_new[0][1], f_new[1][1], f_new[2][1], 0.0);
    fmat[3u * i + 2u] = vec4<f32>(f_new[0][2], f_new[1][2], f_new[2][2], 0.0);
    sstate[2u * i + 0u] = vec4<f32>(tau[0][0], tau[1][0], tau[2][0], tau[1][1]);
    let old = sstate[2u * i + 1u];
    sstate[2u * i + 1u] = vec4<f32>(tau[2][1], old.y, old.z, old.w);
}

// Final macro gather stores the compaction/drag/mixture-pressure kick for the next staggered
// solid drift. Positions, Fbar, and constitutive state are not advanced twice.
@compute @workgroup_size(256)
fn g2p_solid_velocity_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) { return; }
    let p = params.particle_count - params.solid_count + i;
    let gathered = gather_solid_velocity(pos[p].xyz);
    var v = gathered.v;
    var c0 = gathered.c0;
    var c1 = gathered.c1;
    var c2 = gathered.c2;
    // The macro grid predates this step's dissolution. Apply the signed mass/momentum transfer
    // produced by chemistry, then consume and clear the transient bridge lanes. As in the legacy
    // path, mass loss may not amplify the gathered affine field.
    let delta_mass = vel[p].w;
    let delta_momentum = vec3<f32>(
        cmat[3u * p + 0u].w,
        cmat[3u * p + 1u].w,
        cmat[3u * p + 2u].w,
    );
    let new_mass = grain_effective_mass(p);
    let old_mass = max(new_mass - delta_mass, 0.0);
    if (new_mass > params.extra.z) {
        v = (old_mass * v + delta_momentum) / new_mass;
        let affine_mass_fraction = clamp(old_mass / new_mass, 0.0, 1.0);
        c0 *= affine_mass_fraction;
        c1 *= affine_mass_fraction;
        c2 *= affine_mass_fraction;
    }
    let speed = length(v);
    if (speed > params.max_speed) { v *= params.max_speed / speed; }
    vel[p] = vec4<f32>(v, 0.0);
    cmat[3u * p + 0u] = vec4<f32>(c0, 0.0);
    cmat[3u * p + 1u] = vec4<f32>(c1, 0.0);
    cmat[3u * p + 2u] = vec4<f32>(c2, 0.0);
}

@compute @workgroup_size(256)
fn compaction_node_setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) { return; }
    let mass = grid_svel[n].w;
    var mobility = minv_zero();
    if (mass > params.extra.z) {
        let c = node_coords(n);
        let h = params.grid_origin.w;
        let xp = params.grid_origin.xyz + vec3<f32>(c) * h;
        let projector = phase_pressure_projector(xp, PHASE_SOLID, false);
        if (projector.b.z > 0.5) {
            mobility = minv_scale(projector, h * h * h / mass);
        }
    }
    nm[2u * n + 0u] = mobility.a;
    nm[2u * n + 1u] = vec4<f32>(mobility.b.x, mobility.b.y, mobility.b.z, 1.0);
}

@compute @workgroup_size(256)
fn compaction_cell_setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= num_fine_cells()) { return; }
    let cc = vec3<i32>(cell_coords(flat));
    let state = compaction_cell_state(cc);
    var div = 0.0;
    let h = params.grid_origin.w;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let o = vec3<i32>(ox, oy, oz);
                div += dot(corner_sign(o), grid_svel[node_index(cc + o)].xyz) / (4.0 * h);
            }
        }
    }
    let phi_e = state.phi * exp(-state.ep);
    let response = packing_response(phi_e);
    let cap = consolidation_pressure_for(state.ep);
    let pressure = min(response.pressure, cap);
    // Keep the elastic tangent at the cap. The projected solve's q <= p_c active set blocks
    // further loading, while a positive dilation is then able to unload q below the cap.
    // Zeroing this tangent would leave a plastified cell permanently pressurized.
    let kappa = phi_e * response.tangent;
    let rhs = pressure - params.dt * kappa * div;
    cell_meta[flat] = vec4<f32>(rhs, kappa, cap, state.phi);
    pf_src[flat] = pressure;
    pf_dst[flat] = pressure;
}

@compute @workgroup_size(256)
fn compaction_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    if (flat >= num_fine_cells()) { return; }
    let cm = cell_meta[flat];
    if (cm.w <= 0.0 || cm.z <= 0.0) { pf_dst[flat] = 0.0; return; }
    if (cm.y <= 0.0) { pf_dst[flat] = clamp(pf_src[flat], 0.0, cm.z); return; }
    let h = params.grid_origin.w;
    let cc = vec3<i32>(cell_coords(flat));
    var acc = 0.0;
    var diag = 0.0;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let o = vec3<i32>(ox, oy, oz);
                let node = cc + o;
                let n = node_index(node);
                let mobility = Minv(nm[2u * n], nm[2u * n + 1u]);
                var gp = vec3<f32>(0.0);
                for (var qz = 0; qz < 2; qz = qz + 1) {
                    for (var qy = 0; qy < 2; qy = qy + 1) {
                        for (var qx = 0; qx < 2; qx = qx + 1) {
                            let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                            if (!cell_in_range(c2)) { continue; }
                            let ci = cell_index(c2);
                            let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                            gp -= s2 * pf_src[ci] / (4.0 * h);
                        }
                    }
                }
                let s = corner_sign(o);
                acc += dot(s, minv_apply(mobility, gp)) / (4.0 * h);
                diag += dot(s, minv_apply(mobility, s)) / (16.0 * h * h);
            }
        }
    }
    let aq = -acc;
    let dt2k = params.dt * params.dt * cm.y;
    let residual = cm.x - pf_src[flat] - dt2k * aq;
    let denominator = 1.0 + dt2k * diag;
    pf_dst[flat] = clamp(
        pf_src[flat] + COMPACTION_OMEGA * residual / max(denominator, 1.0),
        0.0,
        cm.z,
    );
}

@compute @workgroup_size(256)
fn compaction_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w || grid_svel[n].w <= params.extra.z) { return; }
    let h = params.grid_origin.w;
    let node = vec3<i32>(node_coords(n));
    var gp = vec3<f32>(0.0);
    for (var qz = 0; qz < 2; qz = qz + 1) {
        for (var qy = 0; qy < 2; qy = qy + 1) {
            for (var qx = 0; qx < 2; qx = qx + 1) {
                let c = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                if (!cell_in_range(c)) { continue; }
                let s = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                gp -= s * pf_src[cell_index(c)] / (4.0 * h);
            }
        }
    }
    let mobility = Minv(nm[2u * n], nm[2u * n + 1u]);
    grid_svel[n] = vec4<f32>(
        grid_svel[n].xyz - params.dt * minv_apply(mobility, gp),
        grid_svel[n].w,
    );
}

fn compaction_increment_at_cell(c: vec3<i32>) -> f32 {
    let flat = cell_index(c);
    let cm = cell_meta[flat];
    if (cm.w <= 0.0 || cm.z < 0.0) { return 0.0; }
    // Plastic flow obeys the projected solve's active set: a cell that elastically unloaded
    // below its cap has zero cap multiplier and must not accumulate irreversible compaction.
    if (pf_src[flat] < cm.z) { return 0.0; }
    var solid_volume = 0.0;
    var moment = 0.0;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let n = node_index(c + vec3<i32>(ox, oy, oz));
                solid_volume += solid_volume_at(n);
                moment += fp_decode(atomicLoad(&grid_scomp[n]));
            }
        }
    }
    if (solid_volume <= 0.0) { return 0.0; }
    let ep = max(moment / solid_volume, 0.0);
    let phi_e = cm.w * exp(-ep);
    let trial = packing_response(phi_e);
    if (trial.pressure <= cm.z || trial.pressure <= 0.0) { return 0.0; }
    let target_phi = packing_inverse(cm.z);
    if (target_phi <= 0.0) { return 0.0; }
    return max(log(phi_e / target_phi), 0.0);
}

@compute @workgroup_size(256)
fn compaction_state_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) { return; }
    let p = params.particle_count - params.solid_count + i;
    let h = params.grid_origin.w;
    let xc = (pos[p].xyz - params.grid_origin.xyz) / h - vec3<f32>(0.5);
    let base = vec3<i32>(floor(xc));
    let f = xc - vec3<f32>(base);
    var q = 0.0;
    var dep = 0.0;
    var weight_sum = 0.0;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let c = base + vec3<i32>(ox, oy, oz);
                if (!cell_in_range(c)) { continue; }
                let wx = select(1.0 - f.x, f.x, ox == 1);
                let wy = select(1.0 - f.y, f.y, oy == 1);
                let wz = select(1.0 - f.z, f.z, oz == 1);
                let weight = wx * wy * wz;
                let flat = cell_index(c);
                q += weight * pf_src[flat];
                dep += weight * compaction_increment_at_cell(c);
                weight_sum += weight;
            }
        }
    }
    if (weight_sum > 0.0) { q /= weight_sum; dep /= weight_sum; }
    let old = sstate[2u * i + 1u];
    let ep = max(old.w + dep, 0.0);
    sstate[2u * i + 1u] = vec4<f32>(old.x, max(q, 0.0), consolidation_pressure_for(ep), ep);
}

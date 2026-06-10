// APIC water transfers (U2): grid_clear → p2g_water → grid_update → g2p_water, once per frame
// (substeps fixed at 1 for now — a CFL-driven substep policy is a later-unit decision; with the
// default cell h = 2·spacing and the velocity cap, v·dt/h ≈ 50/60/2 ≈ 0.42 worst-case).
//
// Update order (the solver's OWN discrete recurrence — tests/twofield_water.rs derives its
// free-fall reference from exactly this):
//   P2G scatters m and m·(v + C·d) at the OLD positions; grid_update divides to velocity and
//   applies gravity (v_i ← v_i + g·dt), then boundary conditions; G2P gathers the new particle
//   velocity and advects with it. In free air (no BC active) the per-frame closed form is
//     v ← v + g·dt ;  x ← x + v_new·dt            (semi-implicit Euler)
//   because the B-spline weights partition unity (velocity gather is exact for a uniform field)
//   and Σ_k w_k·x_k = x_p (linear consistency keeps C at zero in a uniform field).
//
// Only the WATER range [0, water_count) is touched — the solid field generalizes these passes
// in U5 (KTD-1 phase-range dispatches, no per-particle phase branch needed here).

// Zero the fixed-point water-field lanes (mass + momentum) for this frame's scatter.
@compute @workgroup_size(256)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    atomicStore(&grid_fp[n * 4u + 0u], 0);
    atomicStore(&grid_fp[n * 4u + 1u], 0);
    atomicStore(&grid_fp[n * 4u + 2u], 0);
    atomicStore(&grid_fp[n * 4u + 3u], 0);
}

// Quadratic B-spline scatter of mass + APIC momentum m·(v + C·(x_i − x_p)) via fixed-point
// atomics (order-independent integer sums → bit-deterministic accumulation).
@compute @workgroup_size(256)
fn p2g_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let v = vel[p].xyz;
    let c0 = cmat[3u * p + 0u].xyz;
    let c1 = cmat[3u * p + 1u].xyz;
    let c2 = cmat[3u * p + 2u].xyz;

    let xl = (x - params.grid_origin.xyz) / h;
    // G2P clamps particles into the box, and the grid pads one node layer beyond each face, so
    // base is always in [0, dims−3]; clamp anyway (Tint clamp-and-predicate, never branches).
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    let m = params.particle_mass;

    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                let d = (vec3<f32>(node) - xl) * h; // x_i − x_p (world units)
                var vaff = v + vec3<f32>(dot(c0, d), dot(c1, d), dot(c2, d));
                // Clamp the scattered magnitude to the velocity cap — this is what makes the
                // FP_SCALE momentum-headroom bound in common.wgsl hold by construction.
                let s = length(vaff);
                if (s > params.max_speed) {
                    vaff = vaff * (params.max_speed / s);
                }
                let idx = node_index(node) * 4u;
                atomicAdd(&grid_fp[idx + 0u], fp_encode(m * wijk));
                atomicAdd(&grid_fp[idx + 1u], fp_encode(m * wijk * vaff.x));
                atomicAdd(&grid_fp[idx + 2u], fp_encode(m * wijk * vaff.y));
                atomicAdd(&grid_fp[idx + 3u], fp_encode(m * wijk * vaff.z));
            }
        }
    }
}

// Convert fixed-point mass/momentum to velocity, apply gravity, then the grid-node boundary
// treatment: no-penetration against the domain box faces and the scene SDF solids — the full
// normal component is removed at wall nodes (free slip tangentially), matching the U3
// pressure family's constrained M̃⁻¹ (see the comments at the BC sites; separation stays
// possible at particle resolution). Writes the result for g2p_water; empty nodes get zero
// velocity.
@compute @workgroup_size(256)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let mc = atomicLoad(&grid_fp[n * 4u + 0u]);
    var v = vec3<f32>(0.0);
    // Mass gate (couples to the cap backstop below): nodes below mass_eps (params.extra.z,
    // ~26 fixed-point counts) decode a quantization-noise velocity AND are exactly the nodes
    // the U3 projection cannot correct (M̃⁻¹ = 0 below the same eps in node_setup) — giving
    // them gravity every frame injects uncorrectable free-fall the surface particles gather
    // via G2P. They are treated as empty here (v = 0), consistently with the whole pressure
    // family (cell fractions, M̃⁻¹). The decoded mass is still written for diagnostics.
    if (fp_decode(mc) > params.extra.z) {
        let inv = 1.0 / f32(mc);
        v = vec3<f32>(
            f32(atomicLoad(&grid_fp[n * 4u + 1u])),
            f32(atomicLoad(&grid_fp[n * 4u + 2u])),
            f32(atomicLoad(&grid_fp[n * 4u + 3u]))
        ) * inv;
        v = v + params.gravity.xyz * params.dt;

        let c = node_coords(n);
        let xp = params.grid_origin.xyz + vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) * params.grid_origin.w;
        let eps = 1.0e-4;

        // Nodes strictly OUTSIDE the domain box are inside the walls: any mass there is
        // B-spline smear of wall-clamped particles, so they take the wall's velocity (zero —
        // mirrored by M̃⁻¹ = 0 in node_setup, so the projection family agrees). Leaving them
        // live fed up to ~23% unprojected free-fall gather weight to particles clamped at the
        // box EDGES (per-axis boundary weight 0.125 on the outside node, two axes at an
        // edge), which then sank at ~quarter gravity forever (observed: the four vertical
        // edge columns pumped the settled tank into sloshing).
        if (any(xp < params.box_min.xyz - vec3<f32>(eps))
            || any(xp > params.box_max.xyz + vec3<f32>(eps))) {
            grid_vel[n] = vec4<f32>(vec3<f32>(0.0), fp_decode(mc));
            return;
        }

        // Domain-box faces: face nodes lose the FULL normal component (free slip tangentially).
        // This matches the U3 pressure family, whose constrained M̃⁻¹ removes the normal axis at
        // these nodes entirely — the projection treats v_n as boundary-determined and can
        // neither see nor correct it, so a one-sided ("separating") grid BC lets scatter noise
        // rectify into a sustained inward v_n that the closed-wall flux gate measures as fake
        // volume transport (observed: −5.5 units³/s on a settled tank, tolerance 2). Particle-
        // level separation is untouched: the G2P backstop only clamps INTO-wall motion.
        if (xp.x <= params.box_min.x + eps || xp.x >= params.box_max.x - eps) { v.x = 0.0; }
        if (xp.y <= params.box_min.y + eps || xp.y >= params.box_max.y - eps) { v.y = 0.0; }
        if (xp.z <= params.box_min.z + eps || xp.z >= params.box_max.z - eps) { v.z = 0.0; }

        // Static SDF solids (mirrors utils/sdf.rs conventions): nodes through a wall (signed
        // distance < 0) lose the full normal component; tangential flow is free slip — same
        // M̃⁻¹-consistency argument as the box faces (node_setup subtracts the whole dyad).
        if (params.num_solids > 0u) {
            let hit = solid_union(xp, PHASE_WATER);
            if (hit.dist < 0.0) {
                v = v - dot(v, hit.grad) * hit.grad;
            }
        }

        // Speed-cap backstop: a node with a tiny quantized mass (a few counts) decodes a noisy
        // velocity; the cap bounds it like everywhere else (couples to the FP headroom math).
        let s = length(v);
        if (s > params.max_speed) {
            v = v * (params.max_speed / s);
        }
    }
    grid_vel[n] = vec4<f32>(v, fp_decode(mc));
}

// APIC gather: new particle velocity + new affine C from the grid field, then advection and the
// particle-resolution boundary backstop. C = B·D⁻¹ with B = Σ w·v_i·dᵀ and D = (h²/4)·I for the
// quadratic B-spline (Jiang et al. APIC) — equivalent to a weight-gradient velocity-derivative
// reconstruction, in the cheap D⁻¹ form.
@compute @workgroup_size(256)
fn g2p_water(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    let h = params.grid_origin.w;
    var x = pos[p].xyz;

    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);

    var v = vec3<f32>(0.0);
    var b0 = vec3<f32>(0.0); // rows of B = Σ w·v_i·dᵀ
    var b1 = vec3<f32>(0.0);
    var b2 = vec3<f32>(0.0);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                let d = (vec3<f32>(node) - xl) * h;
                let gv = grid_vel[node_index(node)].xyz;
                v = v + wijk * gv;
                b0 = b0 + (wijk * gv.x) * d;
                b1 = b1 + (wijk * gv.y) * d;
                b2 = b2 + (wijk * gv.z) * d;
            }
        }
    }
    let dinv = 4.0 / (h * h);
    var c0 = b0 * dinv;
    var c1 = b1 * dinv;
    var c2 = b2 * dinv;
    if (params.pic_mode != 0u) {
        // Test-only PIC variant (the APIC-vs-PIC discrimination gate): drop the affine state.
        c0 = vec3<f32>(0.0);
        c1 = vec3<f32>(0.0);
        c2 = vec3<f32>(0.0);
    }

    // Velocity cap backstop (anti-blow-up; the other half of the FP headroom contract).
    let s = length(v);
    if (s > params.max_speed) {
        v = v * (params.max_speed / s);
    }

    x = x + v * params.dt;

    // Particle-resolution boundary backstop (mirrors the xpbd finalize conventions: box clamp +
    // SDF push-out along the cavity gradient; SDF math mirrors utils/sdf.rs). The grid-node BC
    // above is the momentum treatment, but on its own it bounds penetration only at node
    // resolution (h = 2·spacing): a particle between a constrained node layer and a live one
    // keeps a weighted inward velocity fraction and creeps through. The push-out enforces the
    // non-penetration gate at particle resolution; only the into-wall velocity component is
    // removed (separating, free slip).
    if (x.x < params.box_min.x) { x.x = params.box_min.x; if (v.x < 0.0) { v.x = 0.0; } }
    if (x.y < params.box_min.y) { x.y = params.box_min.y; if (v.y < 0.0) { v.y = 0.0; } }
    if (x.z < params.box_min.z) { x.z = params.box_min.z; if (v.z < 0.0) { v.z = 0.0; } }
    if (x.x > params.box_max.x) { x.x = params.box_max.x; if (v.x > 0.0) { v.x = 0.0; } }
    if (x.y > params.box_max.y) { x.y = params.box_max.y; if (v.y > 0.0) { v.y = 0.0; } }
    if (x.z > params.box_max.z) { x.z = params.box_max.z; if (v.z > 0.0) { v.z = 0.0; } }
    if (params.num_solids > 0u) {
        let hit = solid_union(x, PHASE_WATER);
        if (hit.dist < 0.0) {
            x = x + (-hit.dist) * hit.grad; // project back to the cavity surface
            let vn = dot(v, hit.grad);
            if (vn < 0.0) {
                v = v - vn * hit.grad;
            }
        }
    }

    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, vel[p].w);
    cmat[3u * p + 0u] = vec4<f32>(c0, 0.0);
    cmat[3u * p + 1u] = vec4<f32>(c1, 0.0);
    cmat[3u * p + 2u] = vec4<f32>(c2, 0.0);
}

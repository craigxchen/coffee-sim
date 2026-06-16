// APIC water transfers (U2): grid_clear → p2g_water → grid_update → g2p_water, once per frame
// (substeps fixed at 1 for now — a CFL-driven substep policy is a later-unit decision; with the
// default cell h = 2·spacing and the velocity cap, v·dt/h ≈ 50/60/2 ≈ 0.42 worst-case).
//
// Update order (the solver's OWN discrete recurrence — tests/twofield_water.rs derives its
// free-fall reference from exactly this):
//   P2G scatters m and m·(v + C·d) at the OLD positions; grid_update divides to velocity;
//   drag_fold (coupling.wgsl, U6 — it owns the grid forces so the exponential drag fold can
//   integrate the gravity source exactly) applies gravity (v_i ← v_i + g·dt on drag-free
//   nodes), the drag pair update, then the boundary conditions; G2P gathers the new particle
//   velocity and advects with it. In free air (no BC active) the per-frame closed form is
//     v ← v + g·dt ;  x ← x + v_new·dt            (semi-implicit Euler)
//   because the B-spline weights partition unity (velocity gather is exact for a uniform field)
//   and Σ_k w_k·x_k = x_p (linear consistency keeps C at zero in a uniform field).
//
// Only the WATER range [0, water_count) is touched here — the U6 solid-mass P2G and the drag
// fold live in coupling.wgsl (KTD-1 phase-range dispatches, no per-particle phase branch).

// Zero the fixed-point water-field lanes (mass + momentum), the solid-volume lane (U6), and
// the per-cell particle counts for this frame's scatter (cells < nodes, so the node-sized
// dispatch covers both).
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
    atomicStore(&grid_sfp[n], 0);
    // U5 solid momentum lanes (read only in dynamic mode; clearing is mode-independent).
    atomicStore(&grid_sm[n * 4u + 0u], 0);
    atomicStore(&grid_sm[n * 4u + 1u], 0);
    atomicStore(&grid_sm[n * 4u + 2u], 0);
    atomicStore(&grid_sm[n * 4u + 3u], 0);
    // U9 moisture lanes (read only when absorption is on; cleared unconditionally).
    atomicStore(&grid_moist[n * 2u + 0u], 0);
    atomicStore(&grid_moist[n * 2u + 1u], 0);
    if (n < num_fine_cells()) {
        atomicStore(&cell_cnt[n], 0u);
    }
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
    // Open base (U6 drained-flux mode): particles that left through the floor are ballistic
    // and must not scatter — the index clamp would otherwise fold them onto the pad nodes.
    if (params.coupling.w > 0.5 && x.y < params.box_min.y) {
        return;
    }
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
    // U9 effective water mass: a partially-absorbed particle (remaining fraction f_w = pos.w)
    // scatters proportionally less mass/volume (models::wetting::water_eff_mass), so the bed
    // pore volume falls exactly as water is absorbed — the projection sees the real remaining
    // fluid. With absorption off f_w ≡ 1 and this is bit-identical to the U2 constant mass.
    let m = params.particle_mass * clamp(pos[p].w, 0.0, 1.0);

    // Particle-presence census for the surface classification (see cell_cnt in common.wgsl).
    let ci = clamp(
        vec3<i32>(floor(xl)),
        vec3<i32>(0),
        vec3<i32>(params.grid_dims.xyz) - vec3<i32>(2),
    );
    atomicAdd(&cell_cnt[cell_index(ci)], 1u);

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

// Convert fixed-point mass/momentum to velocity. Gravity, the U6 drag fold, the grid-node
// boundary treatment, and the speed-cap backstop all live in drag_fold (coupling.wgsl) —
// the exponential fold must integrate the gravity source itself and the BCs must follow
// every velocity update, so this pass is decode-only. Empty nodes get zero velocity.
@compute @workgroup_size(256)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let mc = atomicLoad(&grid_fp[n * 4u + 0u]);
    var v = vec3<f32>(0.0);
    // Mass gate (couples to drag_fold's cap backstop): nodes below mass_eps (params.extra.z,
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

    // Open base (U6 drained-flux mode): escaped particles fall ballistically — they no
    // longer scatter (p2g guard), so gathering from the empty grid would freeze them.
    let open_base = params.coupling.w > 0.5;
    if (open_base && x.y < params.box_min.y) {
        var v = vel[p].xyz + params.gravity.xyz * params.dt;
        x = x + v * params.dt;
        pos[p] = vec4<f32>(x, pos[p].w);
        vel[p] = vec4<f32>(v, vel[p].w);
        cmat[3u * p + 0u] = vec4<f32>(0.0);
        cmat[3u * p + 1u] = vec4<f32>(0.0);
        cmat[3u * p + 2u] = vec4<f32>(0.0);
        return;
    }

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
    // APIC↔PIC blend knob: scale the affine C by (1 − pic_blend) each transfer (the gathered
    // velocity stays full APIC; the FLIP/PIC distinction lives entirely in C). DEFAULT 0 = pure
    // APIC. This is NOT the settled-pool stirring fix — that was investigated and DEFERRED to the
    // saturated-bed-creep redesign: the stirring is the same open-water agitation that drives the
    // crater slump, so any blend that quiets the pool also freezes the crater (measured: a global
    // blend froze it; a φ_f-gated open-water-only blend froze it too — the slump is pond-driven;
    // and more pressure sweeps inflate the pool by realizing the relief's expansion target). The
    // knob remains for the APIC-vs-PIC discrimination gate (pic_blend = 1 ⇒ pure PIC) and for the
    // redesign, which can re-enable a global blend once the bed slumps via real pore-pressure creep
    // rather than numerical agitation.
    let apic_keep = 1.0 - params.pic_blend;
    var c0 = b0 * (dinv * apic_keep);
    var c1 = b1 * (dinv * apic_keep);
    var c2 = b2 * (dinv * apic_keep);

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
    if (x.y < params.box_min.y && !open_base) { x.y = params.box_min.y; if (v.y < 0.0) { v.y = 0.0; } }
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

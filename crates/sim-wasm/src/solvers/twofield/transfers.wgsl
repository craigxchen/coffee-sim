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

// --- U2 surface-weighted velocity-averaging dissipation consts (g2p_water) --------------------
// Compile-time tuning for the merge-discriminator curve (the runtime lanes are params.flip =
// (c_surface, density_gate, div_scale, water_splash_cap); see common.wgsl). Starting values —
// to be re-tuned by the visual oracle in U4; they have NO effect on the disabled default path
// (c_surface = 1.0 ⇒ c = 1 ⇒ v = v_grid, C unchanged = pure-PIC).
//   APPROACH_SCALE — gain on the directional approach-into-density signal (the robust merge
//     term; post-projection div is weak). approach is a speed (world units/frame); a merging
//     drip approaches the slow pool at O(several) units, so a scale of 0.5 saturates merge≈1
//     by ~2 units of approach.
//   W — width of the density smoothstep above density_gate (in rest-cell units, where a fully
//     packed cell reads m_local/(8·particle_mass) ≈ 1); 0.25 ramps the bulk term over a
//     quarter-cell of fill above the gate.
//   AFFINE_DAMP_K — how much of the affine C is shed at full preserve (c = c_surface): C is
//     multiplied by (1 − K·(1−c)), so c=1 ⇒ ×1 (bulk unchanged), strong preserve ⇒ C shed by
//     ~K. ≈0.75 mirrors the prototype's affine_damp.
const APPROACH_SCALE: f32 = 0.5;
const DENSITY_W: f32 = 0.25;
const AFFINE_DAMP_K: f32 = 0.75;
const GRAD_M_EPS: f32 = 1.0e-6;
// dens_c (bulk-calm term) must damp ONLY the settled pool (dense AND slow) — a dense but
// fast-moving pour stream must NOT be bulk-damped or the free water re-mushes. Gate dens_c off
// as cell speed rises past DENS_SLOW_HI (visual-oracle starting values; tune by eye).
const DENS_SLOW_LO: f32 = 0.5;
const DENS_SLOW_HI: f32 = 3.0;

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

    // U3 separate water-splash cap (flip.w): the crown's peak velocity survives a higher cap on
    // the water path while the solid clamps stay on the global max_speed. Sentinel ≤ 0 ⇒ fall
    // back to the global max_speed (byte-identical default). The FP momentum-headroom bound in
    // common.wgsl is re-derived against this cap (∝ 8192/cap).
    let water_cap = select(params.max_speed, params.flip.w, params.flip.w > 0.0);

    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                let d = (vec3<f32>(node) - xl) * h; // x_i − x_p (world units)
                var vaff = v + vec3<f32>(dot(c0, d), dot(c1, d), dot(c2, d));
                // Clamp the scattered magnitude to the water-splash cap — this is what makes the
                // FP_SCALE momentum-headroom bound in common.wgsl hold by construction.
                let s = length(vaff);
                if (s > water_cap) {
                    vaff = vaff * (water_cap / s);
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
    // U2: analytic weight derivative (1/h-scaled below) for the mass-gradient gather. The full
    // 3D gradient at node (i,j,k) is (dw[i].x·w[j].y·w[k].z, w[i].x·dw[j].y·w[k].z,
    // w[i].x·w[j].y·dw[k].z)/h. NOT reusable from B (which uses the d-offset form).
    var dw = bspline_dw(fx);

    var v = vec3<f32>(0.0);
    var b0 = vec3<f32>(0.0); // rows of B = Σ w·v_i·dᵀ
    var b1 = vec3<f32>(0.0);
    var b2 = vec3<f32>(0.0);
    var m_local = 0.0;             // U2: Σ w·node_mass (local density, grid_vel[].w)
    var grad_m = vec3<f32>(0.0);   // U2: Σ ∇w·node_mass (mass gradient, world units)
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                let d = (vec3<f32>(node) - xl) * h;
                let gvel = grid_vel[node_index(node)];
                let gv = gvel.xyz;
                let gm = gvel.w; // decoded node mass
                v = v + wijk * gv;
                b0 = b0 + (wijk * gv.x) * d;
                b1 = b1 + (wijk * gv.y) * d;
                b2 = b2 + (wijk * gv.z) * d;
                m_local = m_local + wijk * gm;
                let gw = vec3<f32>(
                    dw[i].x * w[j].y * w[k].z,
                    w[i].x * dw[j].y * w[k].z,
                    w[i].x * w[j].y * dw[k].z,
                ) / h;
                grad_m = grad_m + gw * gm;
            }
        }
    }
    let dinv = 4.0 / (h * h);
    // APIC↔PIC blend knob: scale the affine C by (1 − pic_blend) each transfer (the gathered
    // velocity stays full APIC; the FLIP/PIC distinction lives entirely in C). A small blend (the
    // production default) is the SHIPPED settled-pool stirring fix — it bleeds off the spurious
    // affine ringing that re-energizes a settled pool. It is safe to apply globally because the
    // crater gate now asserts the wet bed HOLDS its poured crater (wet-sand plasticity), so the
    // blend "freezing" the crater is the correct outcome, not a regression (see PIC_BLEND_DEFAULT /
    // twofield_full.rs). pic_blend = 1 ⇒ pure PIC (the APIC-vs-PIC discrimination gate).
    let apic_keep = 1.0 - params.pic_blend;
    var c0 = b0 * (dinv * apic_keep);
    var c1 = b1 * (dinv * apic_keep);
    var c2 = b2 * (dinv * apic_keep);

    // U2 surface-weighted velocity-averaging dissipation (KTD1/KTD2). `v` so far is the full
    // local grid average v_grid (today's value = pure-PIC velocity). We blend it toward the
    // particle's RETAINED momentum v_own by a surface-weighted strength c: c→1 (full average =
    // calm bulk) where the particle merges into a dense slow neighborhood, c→c_surface (small =
    // momentum-preserving = splash) for separating/coherent-free surface water. The DISABLED
    // default (flip.x = c_surface = 1.0) makes c≡1 ⇒ v=v_grid, factor=1 ⇒ C unchanged ⇒
    // byte-identical pure-PIC.
    let c_surface = params.flip.x;     // small near-surface smoothing (1.0 ⇒ disabled = pure-PIC)
    let density_gate = params.flip.y;  // bulk density gate (rest-cell units)
    let div_scale = params.flip.z;     // merge-discriminator scale; ≤0 ⇒ whole discriminator off
    let v_grid = v;
    let v_own = vel[p].xyz + params.gravity.xyz * params.dt;
    // Local velocity divergence ∇·v = dinv·trace(B) (B alone is NOT the gradient until ×dinv).
    // Post-projection so weak/noisy — an input, not a proof (the approach term is the robust one).
    let div = dinv * (b0.x + b1.y + b2.z);
    // Directional approach into density: a drip merging INTO the pool moves UP the mass gradient
    // (approach>0 ⇒ dissipate); a crown moving toward air reads approach≈0 ⇒ preserve. Stable
    // normalization (NOT normalize(grad_m+eps), which biases the direction).
    let n = grad_m / max(length(grad_m), GRAD_M_EPS);
    let approach = max(0.0, dot(v_own - v_grid, n));
    // Merge discriminator — the WHOLE block gated on div_scale>0 (≤0 ⇒ merge=0 = the density-only
    // R5 negative control, same shader). Compressive div<0 OR fast approach-into-density ⇒ merge.
    var merge = 0.0;
    if (div_scale > 0.0) {
        merge = clamp(max(-div * div_scale, approach * APPROACH_SCALE), 0.0, 1.0);
    }
    // Density term: calm the dense bulk even without merge-motion — but ONLY when the cell is
    // also SLOW (the settled pool), never a dense fast pour stream (which must stay lively).
    let slow = 1.0 - smoothstep(DENS_SLOW_LO, DENS_SLOW_HI, length(v_grid));
    let dens_c = smoothstep(density_gate, density_gate + DENSITY_W, m_local / (8.0 * params.particle_mass)) * slow;
    // Final smoothing strength, floored at c_surface (never below the preserve setpoint).
    let c = clamp(max(max(c_surface, merge), dens_c), c_surface, 1.0);
    // Apply: blend velocity toward the local average by c; damp the affine C proportional to the
    // preserve amount (c=1 ⇒ ×1 = bulk unchanged; strong preserve ⇒ C shed by ~AFFINE_DAMP_K).
    v = mix(v_own, v_grid, c);
    let affine_keep = 1.0 - AFFINE_DAMP_K * (1.0 - c);
    c0 = c0 * affine_keep;
    c1 = c1 * affine_keep;
    c2 = c2 * affine_keep;

    // Velocity cap backstop (anti-blow-up; the other half of the FP headroom contract). U3: the
    // water path uses the separate splash cap (flip.w) so the crown survives; sentinel ≤ 0 ⇒
    // global max_speed (byte-identical default).
    let water_cap = select(params.max_speed, params.flip.w, params.flip.w > 0.0);
    let s = length(v);
    if (s > water_cap) {
        v = v * (water_cap / s);
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

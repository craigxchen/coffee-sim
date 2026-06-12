// U5 solid-phase elastoplasticity (plan 2026-06-09-001 U5 / KTD-5): the dynamic granular
// skeleton — solid P2G with momentum + APIC + MLS-MPM stress force, grid forces with the
// over-packing solids-pressure guard, solid G2P with the deformation-gradient update and the
// Klar 2016 Drucker-Prager 3-branch return map + compaction cap in log-strain space.
// Gates: tests/twofield_bed.rs. CPU TWIN: every constitutive function here mirrors
// `twofield::plasticity` (characterization-first — the twin and its gates precede this port;
// `gpu_return_map_matches_cpu_twin` pins this file against it).
//
// ============================== STRESS SCATTER (documented choice) ===========================
// The stress divergence enters the next P2G MLS-MPM style, fused into the affine term: with
// quadratic B-splines ∇w_ip = w_ip·d_ip·4/h², so the per-node force −V_p⁰·τ·∇w becomes a
// momentum contribution (m_s·C − Δt·V_p⁰·(4/h²)·τ)·d scattered with the plain weight — zero
// extra passes, no weight-gradient code path, and the same fixed-point headroom math as the
// water field (the scattered velocity m·v + A·d is magnitude-clamped to max_speed before
// encoding). τ is the world-frame Kirchhoff stress computed by the PREVIOUS g2p_solid
// (post-return-map), stored per particle in `sstate`; V_p⁰ = d³ (one grain represents one
// lattice cell of bulk bed; the grain SPHERE volume π/6·d³ still feeds the φ_s field).
//
// ============================== UPDATE ORDER (per frame, dynamic mode) =======================
//   grid_clear → p2g_water → p2g_solid_dyn → grid_update → solid_update → drag_fold
//   → (U3/U4 pressure stack, water only) → project → g2p_water → g2p_solid
// The solid field is NOT in the U6 mixture divergence (v_s enters the projection in U7); the
// drag fold keeps its U6 frozen-skeleton semantics — in the dry L1 scenes there is no water
// mass, so the pair never engages. Frozen mode (Config::solid_dynamics = false) dispatches
// the U6 `p2g_solid` (coupling.wgsl, untouched) instead of `p2g_solid_dyn` and skips
// solid_update/g2p_solid entirely — U6 behavior is preserved bitwise.
//
// ============================== OVER-PACKING GUARD (KTD-5) ===================================
// Grid-level solids pressure P_sp(φ_s) = K_sp·(φ_s − φ_on)²/(φ_max − φ_s) (the TFM/KTGF
// pattern), φ_s hard-clipped below φ_max so the divergence stays finite. Enters the SOLID
// momentum in solid_update as part of the effective/contact stress — NEVER the projection
// (no volumetric mode is resisted twice; the U7 stress-partition audit checks the sum). The
// particle-level cap alone cannot bound the GRID packing, which is why this lives here.
// The per-step velocity kick is clamped (SP_KICK_MAX) to bound ringing of the divergent term.
//
// Tint discipline: pure per-thread math, no barriers anywhere in this family — guard returns
// are uniform-control-flow safe. The SVD is fixed-sweep Jacobi (branch-free-ish: predicated
// rotations, no data-dependent loop bounds).

// --- bindings (continue the global table; 21 = grid_sm lives in common.wgsl) -----------------
// Float solid grid velocity after solid_update: .xyz = velocity, .w = node solid mass.
@group(0) @binding(22) var<storage, read_write> grid_svel: array<vec4<f32>>;
// Per-solid elastic deformation gradient F, 3 vec4 rows per solid (row-major like cmat):
// fmat[3i + r] = (F[r][0], F[r][1], F[r][2], 0). Seeded to identity at build/reset.
// fmat[3i + 0].w carries v_c — the volume-correction debt (apex-discarded expansion still
// owed back; Tampubolon et al. 2017 / plasticity.rs module header). Without it every
// transient expansion permanently rebases the rest volume and the bed ratchets loose.
@group(0) @binding(23) var<storage, read_write> fmat: array<vec4<f32>>;
// Per-solid constitutive state, 2 vec4 per solid:
//   sstate[2i]   = (τ_xx, τ_xy, τ_xz, τ_yy)   — world Kirchhoff stress (symmetric),
//   sstate[2i+1] = (τ_yz, τ_zz, p_c, compaction) — consolidation pressure + accumulated
//                  plastic volumetric compaction (the tamping-memory record).
@group(0) @binding(24) var<storage, read_write> sstate: array<vec4<f32>>;

// --- constants (mirror twofield::plasticity — the CPU twin) -----------------------------------
const SVD_SWEEPS: u32 = 8u;
const SIG_MIN: f32 = 0.05;
const SIG_MAX: f32 = 20.0;
const SP_KICK_MAX: f32 = 5.0;

const IDENT3: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
);

// =================================== 3×3 SVD ===================================================
// Signed SVD F = U·diag(σ)·Vᵀ with det U = det V = +1, σ₀ ≥ σ₁ ≥ |σ₂|, sign(σ₂) = sign(det F):
// fixed-sweep cyclic Jacobi on FᵀF for (V, σ²), then robust U reconstruction with
// cross-product completion for degenerate/rank-deficient F. Mirrors plasticity.rs::svd3.

struct Svd3 { u: mat3x3<f32>, sigma: vec3<f32>, v: mat3x3<f32> };

// One predicated Jacobi rotation in the (p, q) plane (θ = 0 when S_pq ≈ 0 — atan2(0,0) is
// indeterminate in WGSL, so the guard is load-bearing, mirrored in the twin).
fn jacobi_rot(s: mat3x3<f32>, p: i32, q: i32) -> mat3x3<f32> {
    let spq = s[q][p];
    var theta = 0.0;
    if (abs(spq) > 1.0e-12) {
        theta = 0.5 * atan2(2.0 * spq, s[p][p] - s[q][q]);
    }
    let sn = sin(theta);
    let cs = cos(theta);
    // Column p = (c, s), column q = (−s, c) in the (p, q) plane (mirrors the twin).
    var j = IDENT3;
    j[p][p] = cs;
    j[q][q] = cs;
    j[q][p] = -sn;
    j[p][q] = sn;
    return j;
}

fn norm_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l > 1.0e-12) {
        return v / l;
    }
    return fallback;
}

fn svd3(f: mat3x3<f32>) -> Svd3 {
    var s = transpose(f) * f;
    var v = IDENT3;
    for (var sweep = 0u; sweep < SVD_SWEEPS; sweep = sweep + 1u) {
        var j = jacobi_rot(s, 0, 1);
        s = transpose(j) * s * j;
        v = v * j;
        j = jacobi_rot(s, 0, 2);
        s = transpose(j) * s * j;
        v = v * j;
        j = jacobi_rot(s, 1, 2);
        s = transpose(j) * s * j;
        v = v * j;
    }
    var eig = vec3<f32>(s[0][0], s[1][1], s[2][2]);
    var vc0 = v[0];
    var vc1 = v[1];
    var vc2 = v[2];
    // Descending sort: 3 predicated compare-swaps (0,1), (0,2), (1,2).
    if (eig.y > eig.x) {
        let te = eig.x; eig.x = eig.y; eig.y = te;
        let tv = vc0; vc0 = vc1; vc1 = tv;
    }
    if (eig.z > eig.x) {
        let te = eig.x; eig.x = eig.z; eig.z = te;
        let tv = vc0; vc0 = vc2; vc2 = tv;
    }
    if (eig.z > eig.y) {
        let te = eig.y; eig.y = eig.z; eig.z = te;
        let tv = vc1; vc1 = vc2; vc2 = tv;
    }
    // Right-handed V (±v₂ are both eigenvectors).
    vc2 = cross(vc0, vc1);
    let vout = mat3x3<f32>(vc0, vc1, vc2);

    let sig_eig = vec3<f32>(
        sqrt(max(eig.x, 0.0)),
        sqrt(max(eig.y, 0.0)),
        sqrt(max(eig.z, 0.0)),
    );
    let eps = 1.0e-6;
    var u0 = f * vc0;
    if (sig_eig.x > eps) {
        u0 = u0 / sig_eig.x;
    } else {
        u0 = vec3<f32>(1.0, 0.0, 0.0); // F ≈ 0
    }
    u0 = norm_or(u0, vec3<f32>(1.0, 0.0, 0.0));
    var u1 = f * vc1;
    u1 = u1 - dot(u1, u0) * u0;
    let l1 = length(u1);
    if (sig_eig.y > eps && l1 > eps) {
        u1 = u1 / l1;
    } else {
        var pick = vec3<f32>(1.0, 0.0, 0.0);
        if (abs(u0.x) >= 0.9) {
            pick = vec3<f32>(0.0, 1.0, 0.0);
        }
        u1 = norm_or(pick - dot(pick, u0) * u0, vec3<f32>(0.0, 1.0, 0.0));
    }
    let u2 = cross(u0, u1); // det U = +1 by construction
    let uout = mat3x3<f32>(u0, u1, u2);
    let sigma = vec3<f32>(
        max(dot(f * vc0, u0), 0.0),
        max(dot(f * vc1, u1), 0.0),
        dot(f * vc2, u2), // signed: negative iff det F < 0
    );
    return Svd3(uout, sigma, vout);
}

// =================================== return map ================================================
// The Klar 3-branch DP return map + compaction cap in log-strain space (mirror of
// plasticity.rs::return_map; full math in its module header). Returns (ε', p_c', Δcompaction)
// packed as (eps.xyz in a vec3, extras in a vec2).

struct RmOut { eps: vec3<f32>, p_c: f32, d_comp: f32 };

fn return_map(eps_trial: vec3<f32>, p_c: f32) -> RmOut {
    let mu = params.splas0.y;
    let lam = params.splas0.z;
    let alpha = params.splas0.w;
    let y_coh = params.splas1.w;
    let xi = params.splas1.x;
    let k3 = 2.0 * mu + 3.0 * lam;
    let kb = lam + 2.0 * mu / 3.0;
    var eps = eps_trial;
    var tr = eps.x + eps.y + eps.z;

    // CAP: compression past the consolidation pressure compacts plastically; p_c ratchets.
    let tr_cap = -p_c / kb;
    var d_comp = 0.0;
    if (tr < tr_cap) {
        d_comp = tr_cap - tr;
        eps = eps + vec3<f32>(d_comp / 3.0);
        tr = tr_cap;
    }
    let p_c_new = p_c * exp(xi * d_comp);

    // Drucker-Prager (δγ = yield/(2μ)): elastic / tensile apex / yield-surface projection.
    let dev = eps - vec3<f32>(tr / 3.0);
    let dn = length(dev);
    let dg = dn + alpha * k3 / (2.0 * mu) * tr - y_coh / (2.0 * mu);
    if (dg <= 0.0) {
        return RmOut(eps, p_c_new, d_comp);
    }
    let tr_apex = y_coh / (alpha * k3);
    if (tr > tr_apex) {
        return RmOut(vec3<f32>(tr_apex / 3.0), p_c_new, d_comp);
    }
    return RmOut(eps - (dg / max(dn, 1.0e-12)) * dev, p_c_new, d_comp);
}

// Over-packing solids pressure (module header; mirrors plasticity.rs::solids_pressure).
fn solids_pressure(phi_s: f32) -> f32 {
    let phi_max = params.splas1.y;
    let onset = params.splas2.z;
    let x = min(phi_s, phi_max - 1.0e-3);
    if (x <= onset) {
        return 0.0;
    }
    let ksp = params.splas2.y;
    return ksp * (x - onset) * (x - onset) / (phi_max - x);
}

// φ_s at a node (the same wall-truncation-normalized field the U6 family reads).
fn phi_s_at(n: u32) -> f32 {
    let c = node_coords(n);
    let xp = params.grid_origin.xyz
        + vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) * params.grid_origin.w;
    return 1.0 - node_phi_f(n, xp);
}

// Coulomb-friction wall response, SEPARATING-ALLOWED (the Klar-standard granular wall BC):
// only an APPROACHING normal component (v·n < 0, n pointing into the domain) is removed, and
// the tangential remainder is scaled by max(0, 1 − μ·|v_n|/|v_t|) (the removed normal impulse
// bounds the friction impulse). The original unconditional removal was a sticky-normal wall:
// it glued the collapse front to the floor (runout deficit) and suppressed rebound.
fn coulomb(v: vec3<f32>, nrm: vec3<f32>, mu_b: f32) -> vec3<f32> {
    let vn = dot(v, nrm);
    if (vn >= 0.0) {
        return v; // separating or sliding parallel: untouched
    }
    var out = v - vn * nrm;
    let vt = length(out);
    if (vt > 1.0e-9 && mu_b > 0.0) {
        out = out * max(0.0, 1.0 - mu_b * abs(vn) / vt);
    }
    return out;
}

// =================================== p2g_solid_dyn =============================================
// Solid P2G, dynamic mode: grain sphere volume into grid_sfp (byte-identical arithmetic to the
// frozen p2g_solid) + mass/momentum with APIC and the fused MLS-MPM stress force into grid_sm.
@compute @workgroup_size(256)
fn p2g_solid_dyn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) {
        return;
    }
    let p = params.particle_count - params.solid_count + i;
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let v = vel[p].xyz;
    let c0 = cmat[3u * p + 0u].xyz;
    let c1 = cmat[3u * p + 1u].xyz;
    let c2 = cmat[3u * p + 2u].xyz;
    // Stress rows (symmetric τ from sstate).
    let sa = sstate[2u * i + 0u];
    let sb = sstate[2u * i + 1u];
    let t0 = vec3<f32>(sa.x, sa.y, sa.z); // (τxx, τxy, τxz)
    let t1 = vec3<f32>(sa.y, sa.w, sb.x); // (τxy, τyy, τyz)
    let t2 = vec3<f32>(sa.z, sb.x, sb.y); // (τxz, τyz, τzz)
    let m_s = params.splas1.z;
    let d = params.coupling.x;
    let v0 = d * d * d; // V_p⁰: one grain = one lattice cell of bulk bed
    // MLS-MPM fused force factor: contribution (m·C − Δt·V⁰·(4/h²)·τ)·d, scattered as a
    // velocity (divide by m) so the max_speed clamp keeps the FP headroom contract.
    let k = params.dt * v0 * 4.0 / (h * h) / m_s;

    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    let vg = params.coupling.z;
    for (var kk = 0; kk < 3; kk = kk + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i2 = 0; i2 < 3; i2 = i2 + 1) {
                let wijk = w[i2].x * w[j].y * w[kk].z;
                let node = base + vec3<i32>(i2, j, kk);
                let dd = (vec3<f32>(node) - xl) * h;
                var vaff = v
                    + vec3<f32>(dot(c0, dd), dot(c1, dd), dot(c2, dd))
                    - k * vec3<f32>(dot(t0, dd), dot(t1, dd), dot(t2, dd));
                let sp = length(vaff);
                if (sp > params.max_speed) {
                    vaff = vaff * (params.max_speed / sp);
                }
                let ni = node_index(node);
                atomicAdd(&grid_sfp[ni], fp_encode(vg * wijk));
                let mi = ni * 4u;
                atomicAdd(&grid_sm[mi + 0u], fp_encode(m_s * wijk));
                atomicAdd(&grid_sm[mi + 1u], fp_encode(m_s * wijk * vaff.x));
                atomicAdd(&grid_sm[mi + 2u], fp_encode(m_s * wijk * vaff.y));
                atomicAdd(&grid_sm[mi + 3u], fp_encode(m_s * wijk * vaff.z));
            }
        }
    }
}

// =================================== solid_update ==============================================
// Decode the solid field, integrate gravity, apply the over-packing solids-pressure guard
// (contact partition — header), then the grid-node boundary conditions with Coulomb friction.
@compute @workgroup_size(256)
fn solid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let mc = atomicLoad(&grid_sm[n * 4u + 0u]);
    let mass = fp_decode(mc);
    if (mass <= params.extra.z) {
        grid_svel[n] = vec4<f32>(vec3<f32>(0.0), mass);
        return;
    }
    let inv = 1.0 / f32(mc);
    var v = vec3<f32>(
        f32(atomicLoad(&grid_sm[n * 4u + 1u])),
        f32(atomicLoad(&grid_sm[n * 4u + 2u])),
        f32(atomicLoad(&grid_sm[n * 4u + 3u]))
    ) * inv;
    v = v + params.gravity.xyz * params.dt;

    let h = params.grid_origin.w;
    let c = node_coords(n);
    let xp = params.grid_origin.xyz + vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) * h;

    // Over-packing guard: a = −∇P_sp/ρ_s, central differences over the 6 neighbors, kick
    // clamped. Cheap pre-check on the raw scattered volume (actual φ_s ≤ 8·raw under the
    // vis normalization) skips the vis loops away from any packed region.
    let onset = params.splas2.z;
    var raw_max = fp_decode(atomicLoad(&grid_sfp[n])) / (h * h * h);
    let nd = vec3<i32>(params.grid_dims.xyz);
    for (var a = 0; a < 3; a = a + 1) {
        for (var s = -1; s <= 1; s = s + 2) {
            var nb = vec3<i32>(c);
            nb[a] = clamp(nb[a] + s, 0, nd[a] - 1);
            raw_max = max(raw_max, fp_decode(atomicLoad(&grid_sfp[node_index(nb)])) / (h * h * h));
        }
    }
    if (8.0 * raw_max > onset) {
        var grad = vec3<f32>(0.0);
        for (var a = 0; a < 3; a = a + 1) {
            var pp = 0.0;
            var pm = 0.0;
            for (var s = -1; s <= 1; s = s + 2) {
                var nb = vec3<i32>(c);
                nb[a] = clamp(nb[a] + s, 0, nd[a] - 1);
                let pn = solids_pressure(phi_s_at(node_index(nb)));
                if (s > 0) { pp = pn; } else { pm = pn; }
            }
            grad[a] = (pp - pm) / (2.0 * h);
        }
        let rho = max(mass / (h * h * h), 0.1);
        var kick = -(params.dt / rho) * grad;
        let kl = length(kick);
        if (kl > SP_KICK_MAX) {
            kick = kick * (SP_KICK_MAX / kl);
        }
        v = v + kick;
    }

    // Grid-node BC (solid phase): walls + box faces with Coulomb friction (μ_b from
    // Materials::floor_mu), SDF solids with their own friction coefficient.
    let eps = 1.0e-4;
    if (any(xp < params.box_min.xyz - vec3<f32>(eps))
        || any(xp > params.box_max.xyz + vec3<f32>(eps))) {
        grid_svel[n] = vec4<f32>(vec3<f32>(0.0), mass);
        return;
    }
    let mu_b = params.splas2.x;
    if (xp.x <= params.box_min.x + eps) { v = coulomb(v, vec3<f32>(1.0, 0.0, 0.0), mu_b); }
    if (xp.x >= params.box_max.x - eps) { v = coulomb(v, vec3<f32>(-1.0, 0.0, 0.0), mu_b); }
    if (xp.y <= params.box_min.y + eps) { v = coulomb(v, vec3<f32>(0.0, 1.0, 0.0), mu_b); }
    if (xp.y >= params.box_max.y - eps) { v = coulomb(v, vec3<f32>(0.0, -1.0, 0.0), mu_b); }
    if (xp.z <= params.box_min.z + eps) { v = coulomb(v, vec3<f32>(0.0, 0.0, 1.0), mu_b); }
    if (xp.z >= params.box_max.z - eps) { v = coulomb(v, vec3<f32>(0.0, 0.0, -1.0), mu_b); }
    if (params.num_solids > 0u) {
        let hit = solid_union(xp, PHASE_SOLID);
        if (hit.dist < 0.0) {
            v = coulomb(v, hit.grad, hit.friction);
        }
    }
    let sp = length(v);
    if (sp > params.max_speed) {
        v = v * (params.max_speed / sp);
    }
    grid_svel[n] = vec4<f32>(v, mass);
}

// =================================== g2p_solid =================================================
// APIC gather + advection for the solid range, then the per-particle constitutive update:
// F ← (I + Δt·C)·F (C is the APIC velocity-gradient estimate), one SVD, σ clamp, log map,
// return map, rebuild F and the world Kirchhoff stress for the next P2G.
@compute @workgroup_size(256)
fn g2p_solid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) {
        return;
    }
    let p = params.particle_count - params.solid_count + i;
    let h = params.grid_origin.w;
    var x = pos[p].xyz;
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);

    var v = vec3<f32>(0.0);
    var b0 = vec3<f32>(0.0);
    var b1 = vec3<f32>(0.0);
    var b2 = vec3<f32>(0.0);
    for (var kk = 0; kk < 3; kk = kk + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i2 = 0; i2 < 3; i2 = i2 + 1) {
                let wijk = w[i2].x * w[j].y * w[kk].z;
                let node = base + vec3<i32>(i2, j, kk);
                let dd = (vec3<f32>(node) - xl) * h;
                let gv = grid_svel[node_index(node)].xyz;
                v = v + wijk * gv;
                b0 = b0 + (wijk * gv.x) * dd;
                b1 = b1 + (wijk * gv.y) * dd;
                b2 = b2 + (wijk * gv.z) * dd;
            }
        }
    }
    let dinv = 4.0 / (h * h);
    let c0 = b0 * dinv;
    let c1 = b1 * dinv;
    let c2 = b2 * dinv;

    let sp = length(v);
    if (sp > params.max_speed) {
        v = v * (params.max_speed / sp);
    }
    x = x + v * params.dt;

    // Particle-resolution boundary backstop (the g2p_water conventions; the momentum-level
    // friction lives in solid_update).
    if (x.x < params.box_min.x) { x.x = params.box_min.x; if (v.x < 0.0) { v.x = 0.0; } }
    if (x.y < params.box_min.y) { x.y = params.box_min.y; if (v.y < 0.0) { v.y = 0.0; } }
    if (x.z < params.box_min.z) { x.z = params.box_min.z; if (v.z < 0.0) { v.z = 0.0; } }
    if (x.x > params.box_max.x) { x.x = params.box_max.x; if (v.x > 0.0) { v.x = 0.0; } }
    if (x.y > params.box_max.y) { x.y = params.box_max.y; if (v.y > 0.0) { v.y = 0.0; } }
    if (x.z > params.box_max.z) { x.z = params.box_max.z; if (v.z > 0.0) { v.z = 0.0; } }
    if (params.num_solids > 0u) {
        let hit = solid_union(x, PHASE_SOLID);
        if (hit.dist < 0.0) {
            x = x + (-hit.dist) * hit.grad;
            let vn = dot(v, hit.grad);
            if (vn < 0.0) {
                v = v - vn * hit.grad;
            }
        }
    }

    // --- constitutive update (CPU twin: plasticity.rs::solid_step) ---------------------------
    let f0 = fmat[3u * i + 0u].xyz;
    let f1 = fmat[3u * i + 1u].xyz;
    let f2 = fmat[3u * i + 2u].xyz;
    // Rows → columns (storage is row-major; mat3x3 wants columns).
    let fm = mat3x3<f32>(
        vec3<f32>(f0.x, f1.x, f2.x),
        vec3<f32>(f0.y, f1.y, f2.y),
        vec3<f32>(f0.z, f1.z, f2.z),
    );
    let cm = mat3x3<f32>(
        vec3<f32>(c0.x, c1.x, c2.x),
        vec3<f32>(c0.y, c1.y, c2.y),
        vec3<f32>(c0.z, c1.z, c2.z),
    );
    let f_trial = (IDENT3 + params.dt * cm) * fm;
    let svd = svd3(f_trial);
    let sig = clamp(svd.sigma, vec3<f32>(SIG_MIN), vec3<f32>(SIG_MAX));
    let eps_tr = vec3<f32>(log(sig.x), log(sig.y), log(sig.z));
    let sb_old = sstate[2u * i + 1u];
    // Volume correction (twin: plasticity.rs::solid_step): re-inject the remembered dilation
    // before the return map; the apex re-records what it discards, other branches fold the
    // debt into the rebuilt F.
    let vc = fmat[3u * i + 0u].w;
    let eps_eff = eps_tr + vec3<f32>(vc / 3.0);
    let rm = return_map(eps_eff, sb_old.z);
    let tr_eff = eps_eff.x + eps_eff.y + eps_eff.z;
    let vc_new = max(tr_eff + rm.d_comp - (rm.eps.x + rm.eps.y + rm.eps.z), 0.0);
    let sig_new = vec3<f32>(exp(rm.eps.x), exp(rm.eps.y), exp(rm.eps.z));
    let f_new = svd.u
        * mat3x3<f32>(
            vec3<f32>(sig_new.x, 0.0, 0.0),
            vec3<f32>(0.0, sig_new.y, 0.0),
            vec3<f32>(0.0, 0.0, sig_new.z),
        )
        * transpose(svd.v);
    let mu = params.splas0.y;
    let lam = params.splas0.z;
    let tre = rm.eps.x + rm.eps.y + rm.eps.z;
    let tau_p = 2.0 * mu * rm.eps + vec3<f32>(lam * tre);
    let tau = svd.u
        * mat3x3<f32>(
            vec3<f32>(tau_p.x, 0.0, 0.0),
            vec3<f32>(0.0, tau_p.y, 0.0),
            vec3<f32>(0.0, 0.0, tau_p.z),
        )
        * transpose(svd.u);

    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, vel[p].w);
    cmat[3u * p + 0u] = vec4<f32>(c0, 0.0);
    cmat[3u * p + 1u] = vec4<f32>(c1, 0.0);
    cmat[3u * p + 2u] = vec4<f32>(c2, 0.0);
    // F rows (row r = (F[r][0], F[r][1], F[r][2])): tau/f_new are column-major. Row 0's w
    // lane carries the volume-correction debt v_c.
    fmat[3u * i + 0u] = vec4<f32>(f_new[0][0], f_new[1][0], f_new[2][0], vc_new);
    fmat[3u * i + 1u] = vec4<f32>(f_new[0][1], f_new[1][1], f_new[2][1], 0.0);
    fmat[3u * i + 2u] = vec4<f32>(f_new[0][2], f_new[1][2], f_new[2][2], 0.0);
    sstate[2u * i + 0u] = vec4<f32>(tau[0][0], tau[1][0], tau[2][0], tau[1][1]);
    sstate[2u * i + 1u] = vec4<f32>(tau[2][1], tau[2][2], rm.p_c, sb_old.w + rm.d_comp);
}

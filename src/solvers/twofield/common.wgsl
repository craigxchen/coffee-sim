// Two-field solver shared core: the Params mirror, the global binding table, and the helpers
// every pass family shares (fixed-point encoding, B-spline weights, grid indexing, the SDF
// solid mirror). Later pass families (pressure / plasticity / coupling) are concatenated after
// this file and transfers.wgsl into one shader module (WGSL has no imports), so this file
// declares everything they share. Binding indices are GLOBAL and stable across the module:
// each pipeline's auto layout (`layout: None`) keeps only the bindings its entry point
// actually uses, but an index never means two different buffers.
//
// Storage-buffer budget (KTD-7): the device requests 9 storage buffers per stage
// (src/utils/gpu.rs NEEDED_STORAGE_BUFFERS) and U2 does NOT raise it — the widest entry point
// (g2p_water) binds 5 (see MAX_STORAGE_BUFFERS_PER_ENTRY_POINT in mod.rs for the per-pass
// derivation). Grid mass+momentum share ONE array<atomic<i32>> with stride 4 (mass, mom.xyz)
// rather than four arrays — one binding, one clear pass, contiguous per-node lanes.
//
// Tint discipline: any workgroupBarrier() must be reachable from uniform control flow — no
// early returns before a barrier (clamp indices and predicate the work instead). No pass in
// this module uses a barrier, so the per-thread guard returns are safe.

const PHASE_WATER: u32 = 0u;
const PHASE_SOLID: u32 = 1u;

// Byte-identical to the Rust `Params` (160 bytes; vec4-aligned tail).
struct Params {
    box_min: vec4<f32>,     // simulation domain (w unused)
    box_max: vec4<f32>,     // (w unused)
    gravity: vec4<f32>,     // (w unused)
    grid_origin: vec4<f32>, // .xyz = world position of node (0,0,0); .w = cell size h
    grid_dims: vec4<u32>,   // nx, ny, nz, num_nodes (= nx·ny·nz)
    dt: f32,
    particle_mass: f32,  // per water particle (Materials::particle_mass)
    max_speed: f32,      // velocity cap — COUPLED to FP_SCALE (see headroom math below)
    pic_blend: f32,      // APIC↔PIC blend = PIC fraction ∈ [0,1] (G2P scales the affine C by 1−blend)
    water_count: u32,    // particles [0, water_count) are water (KTD-1 range layout)
    solid_count: u32,    // particles [water_count, water_count + solid_count) are solid grains
    particle_count: u32, // = water_count + solid_count (kernel live-set guard)
    num_solids: u32,     // count of static SDF solids in the `solids` buffer (0 = none)
    // U3 pressure stack (pressure.wgsl):
    coarse_dims: vec4<u32>, // coarse CELLS per axis (= ceil(fine_cells/ratio)); .w = ratio
    extra: vec4<f32>,       // (rest_density, rho_floor, mass_eps, U7 wet-cohesion s_peak)
    // U6 coupling (coupling.wgsl): (grain_diameter d, drag_scale, grain_volume π/6·d³,
    // open_base flag — the dev/test drained-column outflow mode).
    coupling: vec4<f32>,
    // U5 plasticity (plasticity.wgsl; mirrors twofield::plasticity constants):
    splas0: vec4<f32>, // (solid_dynamics flag, Lamé μ, Lamé λ, DP α)
    splas1: vec4<f32>, // (cap hardening ξ, φ_max = packing limit, grain mass m_s, cohesion y_c)
    splas2: vec4<f32>, // (floor/wall Coulomb μ_b, guard K_sp, guard onset φ_on, U7 wet c_max)
    // U9 infiltration interface (coupling.wgsl; mirrors models::wetting + the test UNIT MAPPING):
    wet0: vec4<f32>, // (tf_absorb_rate k_abs, V_cap = r_max·ρ_ratio·V_dry, V_w = water vol,
                     //  absorb_roundoff — the f_w/V_abs saturation floor)
    wet1: vec4<f32>, // (suction body-force accel a_suction, bloom_delay seconds, filter_floor
                     //  flag, V_dry = grain sphere volume π/6·d³)
    dbg: vec4<f32>,  // diagnostic toggles (test-only): .x = density-relief enable (1 = on);
                     //  .y = SDF wall-BC mode selector: ≤ 0.5 → binary no-penetration band
                     //  (default, byte-identical to pre-coverage); > 0.5 → graded coverage weight
                     //  (wall_coverage in pressure.wgsl — the embedded-boundary L1 path). (Formerly
                     //  a dropped relief dead-band; the settled-pool stirring fix is the G2P PIC
                     //  blend PIC_BLEND_DEFAULT.) .z reserved. .w = uncapped two-sided
                     //  density-target mode (≤ 0.5 → legacy rate-limited two-sided relief, default
                     //  & byte-identical; > 0.5 → relief uncapped to full strength, driving ρ→ρ₀
                     //  — the over-pack fix; see cell_classify in pressure.wgsl / surface.wgsl).
};

@group(0) @binding(0) var<uniform> params: Params;
// Canonical particle state (ParticleBuffers layout): pos.w is the moisture lane (water =
// remaining fraction, grain = absorbed volume), chem = (concentration, temperature, _, _).
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> phase: array<u32>;
@group(0) @binding(4) var<storage, read_write> chem: array<vec4<f32>>;
// Per-particle APIC affine velocity matrix C (3 rows per particle, row-major):
// cmat[3p + r] = vec4(C[r][0], C[r][1], C[r][2], 0). Rows so that (C·d)[r] = dot(row_r, d).
@group(0) @binding(5) var<storage, read_write> cmat: array<vec4<f32>>;
// WATER grid field, fixed-point: 4 atomic<i32> lanes per node, stride 4 — [mass, mom.x, mom.y,
// mom.z], each value × FP_SCALE. Cleared by grid_clear, scattered by p2g_water, read by
// grid_update. (The SOLID field generalizes this in U5 — not built here.)
@group(0) @binding(6) var<storage, read_write> grid_fp: array<atomic<i32>>;
// WATER grid velocity after grid_update (gravity + boundary conditions applied):
// .xyz = velocity, .w = node mass (decoded float; diagnostics only).
@group(0) @binding(7) var<storage, read_write> grid_vel: array<vec4<f32>>;
// Per-CELL particle count (U4): the particle-presence discriminator for the free-surface
// classification. Node mass alone cannot tell a real surface cell from a particle-free air
// gap narrower than the B-spline smear (≈3h between two water walls): such a gap reads mean
// corner density ≈ 0.3–0.5·ρ_rest, classifies as a constraint row, and its div = 0 row then
// FORBIDS the refilling inflow — observed as a jet-dug chimney standing frozen forever. The
// classical PIC/FLIP rule (a cell is fluid only if it contains particles) is the sharp
// discriminator; mass still provides the fill-fraction taper for cells that ARE fluid.
// Cleared by grid_clear, incremented by p2g_water, read by cell_classify.
@group(0) @binding(18) var<storage, read_write> cell_cnt: array<atomic<u32>>;
// SOLID grid field (U6): per-node solid VOLUME, fixed-point (×FP_SCALE), one lane per node —
// the φ_s carrier (KTD-8: a local evolving field, never a configured scalar). Cleared by
// grid_clear, scattered by p2g_solid (grain sphere volume π/6·d³ per grain), read wherever
// φ_s/φ_f is needed (drag_fold, node_setup, project). Headroom: |V_s| per node ≤ PHI_S-clamped
// h³ ≈ 8 ≪ the 2^13 fixed-point ceiling.
@group(0) @binding(19) var<storage, read_write> grid_sfp: array<atomic<i32>>;
// U6 constraint-reaction ledger: per node, .xyz = the impulse the kinematically frozen
// skeleton absorbs this frame (drag fold + pressure on the solid volume — NEVER silently
// discarded; the buoyant-reaction gate sums it), .w = ς = Δt_eff/Δt, the exponential-
// integrator mobility factor the drag fold hands to node_setup (1.0 where no solid mass).
// Written by drag_fold every frame, accumulated by project; no clear pass needed.
@group(0) @binding(20) var<storage, read_write> react: array<vec4<f32>>;
// U5 SOLID momentum grid field (dynamic skeleton, plasticity.wgsl): fixed-point, 4 lanes per
// node [mass, mom.xyz] — the solid mirror of grid_fp. Cleared by grid_clear, scattered by
// p2g_solid_dyn, decoded by solid_update. Only dispatched-to when Config::solid_dynamics is
// on (params.splas0.x > 0.5); cleared unconditionally (never read in frozen mode).
@group(0) @binding(21) var<storage, read_write> grid_sm: array<atomic<i32>>;
// U9 MOISTURE grid field (coupling.wgsl, GIC-style absorption): per node, 2 fixed-point lanes —
// [0] = water SUPPLY S_n = Σ_w w·f_w·V_w, [1] = grain DEMAND D_n = Σ_g w·demand_g (the
// models::wetting per-step demand). The actual node transfer is T_n = min(S_n, D_n); g2p_absorb
// gathers the node drain/fill fractions T_n/S_n and T_n/D_n so water-loss == grain-gain EXACTLY
// per node (header in coupling.wgsl). Cleared by grid_clear, scattered by p2g_moisture, read by
// g2p_absorb. Only dispatched-to when Config::tf_absorb_rate > 0 (params.wet0.x); the clear runs
// unconditionally (one node-sized dispatch). Headroom: |S_n|,|D_n| per node ≤ a few V_w ≈ a few
// ≪ the 2^13 fixed-point ceiling.
@group(0) @binding(25) var<storage, read_write> grid_moist: array<atomic<i32>>;

// --- fixed-point encoding (KEEP.md §3 pattern, headroom re-validated for U2) -----------------
//
// FP_SCALE = 2^18. i32 range is ±2^31, so an encoded node lane overflows at |value| ≥ 2^13 = 8192.
//   mass lane:      Σ_p m·w per node. Unit-mass water at rest loads an interior node with
//                   Σ m·w ≈ (particles per cell volume) = (h/spacing)³ = 8 → overflow needs
//                   ≈ 1000× rest packing on one node. Unreachable.
//   momentum lanes: every scattered contribution is magnitude-clamped to max_speed (50 by
//                   default — Config::max_speed, the same cap G2P enforces), so |Σ m·w·v_aff| ≤
//                   (Σ m·w)·max_speed → overflow needs Σ m·w ≥ 8192/50 ≈ 164 mass units on one
//                   node ≈ 20× rest compression with EVERY particle at the cap. The overflow
//                   probe gate (tests/twofield_water.rs) exercises a capped, pancaked worst case.
// COUPLED to the velocity cap: raising max_speed shrinks the momentum headroom linearly —
// change one, re-derive the other. fp_encode also saturates (clamp before convert) as a
// backstop so a pathological state degrades bounded instead of wrapping.
const FP_SCALE: f32 = 262144.0; // 2^18
// Largest f32 ≤ i32::MAX (2^31 − 1 = 2147483647 is not representable; nearest-below is 2147483520).
const FP_CLAMP: f32 = 2147483520.0;

fn fp_encode(x: f32) -> i32 {
    return i32(clamp(round(x * FP_SCALE), -FP_CLAMP, FP_CLAMP));
}
fn fp_decode(c: i32) -> f32 {
    return f32(c) / FP_SCALE;
}

// --- quadratic B-spline weights (KEEP.md §3 validated math) -----------------------------------
// Per axis, around base = floor(x/h − 0.5) with fx = x/h − base ∈ [0.5, 1.5]; node offset
// k ∈ {0,1,2} gets w[k]. Weights sum to 1 and Σ_k w[k]·(base+k) = x/h (linear consistency —
// what makes the free-fall recurrence exact and keeps APIC's C zero in a uniform field).
fn bspline_w(fx: vec3<f32>) -> array<vec3<f32>, 3> {
    var w: array<vec3<f32>, 3>;
    w[0] = 0.5 * (1.5 - fx) * (1.5 - fx);
    w[1] = 0.75 - (fx - 1.0) * (fx - 1.0);
    w[2] = 0.5 * (fx - 0.5) * (fx - 0.5);
    return w;
}

// --- grid indexing -----------------------------------------------------------------------------
// Flat node index, x-fastest. Node world position = grid_origin + (i,j,k)·h.
fn node_index(n: vec3<i32>) -> u32 {
    return u32(n.x) + params.grid_dims.x * (u32(n.y) + params.grid_dims.y * u32(n.z));
}
fn node_coords(flat: u32) -> vec3<u32> {
    let nx = params.grid_dims.x;
    let ny = params.grid_dims.y;
    return vec3<u32>(flat % nx, (flat / nx) % ny, flat / (nx * ny));
}

// --- SDF cavity geometry (mirrors utils/sdf.rs; interior positive, gradient toward the cavity) --
// Byte-identical to the Rust `Primitive` (64 bytes). Cone radii in `a` are OUTER wall radii.
struct Primitive {
    kind: u32,          // 0 = cone, 1 = cylinder, 2 = poly-cup
    species_mask: u32,
    friction: f32,
    flags: u32,         // bit0 = apex_open
    a: vec4<f32>,       // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)  poly:(floor_y, rim_y, apothem, sides)
    b: vec4<f32>,       // cone:(thickness, hole_radius, center_x, center_z)  cyl/poly:(center_x, center_z, _, _)
    c: vec4<f32>,       // reserved
};
@group(0) @binding(8) var<storage, read> solids: array<Primitive>;

const SDF_EPS: f32 = 1.0e-6;
const SDF_FREE: f32 = 1.0e30;

struct SolidHit { dist: f32, grad: vec3<f32>, friction: f32 };
struct SegCP { p: vec2<f32>, interior: bool };

fn sdf_normalize2(v: vec2<f32>) -> vec2<f32> {
    let len = length(v);
    if (len > SDF_EPS) { return v / len; }
    return vec2<f32>(0.0, 0.0);
}
fn sdf_normalize3(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len > SDF_EPS) { return v / len; }
    return vec3<f32>(0.0, 0.0, 0.0);
}
fn sdf_closest_on_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> SegCP {
    let ab = b - a;
    let len2 = dot(ab, ab);
    if (len2 < SDF_EPS) { return SegCP(a, false); }
    let t = dot(p - a, ab) / len2;
    let tc = clamp(t, 0.0, 1.0);
    return SegCP(a + ab * tc, t > SDF_EPS && t < 1.0 - SDF_EPS);
}

fn cone_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let apex_y = prim.a.x;
    let apex_r = prim.a.y;
    let top_y = prim.a.z;
    let top_r = prim.a.w;
    let thickness = prim.b.x;
    let hole_radius = prim.b.y;
    let center = vec2<f32>(prim.b.z, prim.b.w);
    let apex_open = (prim.flags & 1u) != 0u;

    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let r = length(rel);
    let y = p.y;
    var rhat = vec2<f32>(0.0, 0.0);
    if (r >= SDF_EPS) { rhat = rel / r; } // axis guard before building the 3D gradient

    if (y > top_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, 1.0, 0.0), prim.friction); }
    if (apex_open && y < apex_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, -1.0, 0.0), prim.friction); }

    let inner_apex = max(apex_r - thickness, hole_radius);
    let inner_top = max(top_r - thickness, hole_radius);
    let a2 = vec2<f32>(inner_apex, apex_y);
    let b2 = vec2<f32>(inner_top, top_y);
    let cp = sdf_closest_on_segment(vec2<f32>(r, y), a2, b2);
    let d2d = length(vec2<f32>(r, y) - cp.p);

    let height = top_y - apex_y;
    var t = 0.0;
    if (height > SDF_EPS) { t = (y - apex_y) / height; }
    let inner = max(mix(apex_r, top_r, t) - thickness, hole_radius);
    let inside = (y >= apex_y) && (r <= inner);
    var dist = -d2d;
    if (inside) { dist = d2d; }

    let dir = vec2<f32>(r, y) - cp.p;
    var n2d = vec2<f32>(0.0, 0.0);
    if (length(dir) > SDF_EPS) {
        if (inside) { n2d = sdf_normalize2(dir); } else { n2d = sdf_normalize2(-dir); }
    } else if (cp.interior) {
        let ab = b2 - a2;
        n2d = sdf_normalize2(vec2<f32>(-ab.y, ab.x)); // smooth-wall analytic normal (never zero here)
    }
    let grad = sdf_normalize3(vec3<f32>(n2d.x * rhat.x, n2d.y, n2d.x * rhat.y));
    return SolidHit(dist, grad, prim.friction);
}

fn cyl_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let floor_y = prim.a.x;
    let rim_y = prim.a.y;
    let radius = prim.a.z;
    let center = vec2<f32>(prim.b.x, prim.b.y);
    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let r = length(rel);
    let y = p.y;
    if (y > rim_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, 1.0, 0.0), prim.friction); }
    let d_side = radius - r;
    let d_floor = y - floor_y;
    if (d_side <= d_floor) {
        var rhat = vec2<f32>(0.0, 0.0);
        if (r >= SDF_EPS) { rhat = rel / r; }
        return SolidHit(d_side, sdf_normalize3(vec3<f32>(-rhat.x, 0.0, -rhat.y)), prim.friction);
    }
    return SolidHit(d_floor, vec3<f32>(0.0, 1.0, 0.0), prim.friction);
}

// DIAGNOSTIC: regular N-gon prism cup (mirror of sdf.rs::poly_cavity). Face normals at 2πk/N
// (k=0 → +x), so sides=4 is an axis-aligned square. Inner side dist = apothem − max_k(rel·n_k).
fn poly_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let floor_y = prim.a.x;
    let rim_y = prim.a.y;
    let apothem = prim.a.z;
    let sides = max(u32(prim.a.w), 3u);
    let center = vec2<f32>(prim.b.x, prim.b.y);
    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let y = p.y;
    if (y > rim_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, 1.0, 0.0), prim.friction); }
    var maxproj = -1.0e30;
    var bestn = vec2<f32>(1.0, 0.0);
    let tau = 6.28318530718;
    for (var k = 0u; k < sides; k = k + 1u) {
        let ang = tau * f32(k) / f32(sides);
        let nk = vec2<f32>(cos(ang), sin(ang));
        let proj = dot(rel, nk);
        if (proj > maxproj) { maxproj = proj; bestn = nk; }
    }
    let d_side = apothem - maxproj;
    let d_floor = y - floor_y;
    if (d_side <= d_floor) {
        return SolidHit(d_side, sdf_normalize3(vec3<f32>(-bestn.x, 0.0, -bestn.y)), prim.friction);
    }
    return SolidHit(d_floor, vec3<f32>(0.0, 1.0, 0.0), prim.friction);
}

fn solid_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    if (prim.kind == 0u) { return cone_cavity(prim, p); }
    if (prim.kind == 2u) { return poly_cavity(prim, p); }
    return cyl_cavity(prim, p);
}

// Species-filtered union: the most-penetrated (min signed-distance) solid that blocks `ph`.
fn solid_union(p: vec3<f32>, ph: u32) -> SolidHit {
    var best = SolidHit(SDF_FREE, vec3<f32>(0.0, 0.0, 0.0), 0.0);
    let n = params.num_solids;
    for (var i = 0u; i < n; i = i + 1u) {
        let prim = solids[i];
        if ((prim.species_mask & (1u << ph)) == 0u) { continue; }
        let hit = solid_cavity(prim, p);
        if (hit.dist < best.dist) { best = hit; }
    }
    return best;
}

// MULTI-NORMAL wall BC (plan 2026-06-17-002): all cavity FACES within `band` of p, generalizing
// solid_union's single most-penetrated pick. At a concave seam (cup floor∩wall) two faces are
// in-band and BOTH must be constrained — solid_union would return only one, leaving the other
// direction free (the corner over-pack). `band` is passed in (= WALL_BAND·h; WALL_BAND lives in
// pressure.wgsl, concatenated after this file). Floor face is pushed FIRST per primitive so the
// rank-aware basis build (build_constraint_basis) never drops it. cone stays single-normal (it is
// one smooth converging surface, not discrete faces). Cap WF_MAX candidates.
const WF_MAX: u32 = 4u;
struct WallFaces { n: array<vec3<f32>, 4>, count: u32 };

fn wf_push(wf: ptr<function, WallFaces>, nrm: vec3<f32>) {
    let c = (*wf).count;
    if (c < WF_MAX && length(nrm) > SDF_EPS) {
        (*wf).n[c] = sdf_normalize3(nrm);
        (*wf).count = c + 1u;
    }
}

// Orthonormal constraint basis Q (≤3 columns) for the MULTI-NORMAL wall BC: the active box-face
// axes plus the in-band SDF wall normals, orthonormalized by modified Gram-Schmidt. The projector
// is P = I − Q·Qᵀ — a TRUE orthogonal projector (PSD for ANY normals, even a non-orthogonal poly
// vertical edge, where a raw Σ n̂n̂ᵀ would be non-PSD and break A = D·M̃⁻¹·G). `node_setup` builds
// M̃⁻¹ = invr·P and `drag_fold` applies v ← P·v from the SAME basis (lockstep). box_mask bits:
// 1 = x axis constrained, 2 = y, 4 = z (the axes node_setup/drag_fold zero on box faces). Rank-aware:
// box axes first, then wall faces residualized + dropped if near-dependent, stop at rank 3 — so a
// floor normal is never crowded out (the floor face is pushed first by wall_binding_faces).
const CB_RESIDUAL_MIN: f32 = 1.0e-3;
struct CBasis { q: array<vec3<f32>, 3>, count: u32 };

fn cb_push_ortho(cb: ptr<function, CBasis>, v: vec3<f32>) {
    if ((*cb).count >= 3u) { return; }
    var r = v;
    for (var j = 0u; j < (*cb).count; j = j + 1u) {
        r = r - (*cb).q[j] * dot((*cb).q[j], r);
    }
    let len = length(r);
    if (len > CB_RESIDUAL_MIN) {
        (*cb).q[(*cb).count] = r / len;
        (*cb).count = (*cb).count + 1u;
    }
}

fn build_constraint_basis(box_mask: u32, faces: WallFaces) -> CBasis {
    var cb: CBasis;
    cb.count = 0u;
    if ((box_mask & 1u) != 0u) { cb_push_ortho(&cb, vec3<f32>(1.0, 0.0, 0.0)); }
    if ((box_mask & 2u) != 0u) { cb_push_ortho(&cb, vec3<f32>(0.0, 1.0, 0.0)); }
    if ((box_mask & 4u) != 0u) { cb_push_ortho(&cb, vec3<f32>(0.0, 0.0, 1.0)); }
    for (var i = 0u; i < faces.count; i = i + 1u) {
        cb_push_ortho(&cb, faces.n[i]);
    }
    return cb;
}

// Apply P = I − Q·Qᵀ to a vector (Q orthonormal ⇒ subtract each column's projection of the
// ORIGINAL v). Used by drag_fold for the velocity BC; node_setup builds the matrix form inline.
fn cbasis_project_vec(cb: CBasis, v: vec3<f32>) -> vec3<f32> {
    var out = v;
    for (var i = 0u; i < cb.count; i = i + 1u) {
        out = out - cb.q[i] * dot(cb.q[i], v);
    }
    return out;
}

fn wall_binding_faces(p: vec3<f32>, ph: u32, band: f32) -> WallFaces {
    var wf: WallFaces;
    wf.count = 0u;
    let ns = params.num_solids;
    for (var i = 0u; i < ns; i = i + 1u) {
        let prim = solids[i];
        if ((prim.species_mask & (1u << ph)) == 0u) { continue; }
        if (prim.kind == 0u) {
            // cone: single smooth surface — one nearest normal, in-band only.
            let hit = cone_cavity(prim, p);
            if (hit.dist < band) { wf_push(&wf, hit.grad); }
        } else if (prim.kind == 2u) {
            // poly cup: floor + every in-band side face (vertical edges yield 2 sides).
            let floor_y = prim.a.x; let rim_y = prim.a.y; let apothem = prim.a.z;
            let sides = max(u32(prim.a.w), 3u);
            let center = vec2<f32>(prim.b.x, prim.b.y);
            let rel = vec2<f32>(p.x - center.x, p.z - center.y);
            if (p.y <= rim_y) {
                if ((p.y - floor_y) < band) { wf_push(&wf, vec3<f32>(0.0, 1.0, 0.0)); }
                let tau = 6.28318530718;
                for (var k = 0u; k < sides; k = k + 1u) {
                    let ang = tau * f32(k) / f32(sides);
                    let nk = vec2<f32>(cos(ang), sin(ang));
                    if ((apothem - dot(rel, nk)) < band) {
                        wf_push(&wf, vec3<f32>(-nk.x, 0.0, -nk.y));
                    }
                }
            }
        } else {
            // cylinder cup: floor + radial side face when in-band.
            let floor_y = prim.a.x; let rim_y = prim.a.y; let radius = prim.a.z;
            let center = vec2<f32>(prim.b.x, prim.b.y);
            let rel = vec2<f32>(p.x - center.x, p.z - center.y);
            let r = length(rel);
            if (p.y <= rim_y) {
                if ((p.y - floor_y) < band) { wf_push(&wf, vec3<f32>(0.0, 1.0, 0.0)); }
                if ((radius - r) < band && r >= SDF_EPS) {
                    let rh = rel / r;
                    wf_push(&wf, vec3<f32>(-rh.x, 0.0, -rh.y));
                }
            }
        }
    }
    return wf;
}

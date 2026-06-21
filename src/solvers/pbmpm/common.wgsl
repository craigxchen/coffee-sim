// PB-MPM (Position-Based MPM) solver shared core: the Params mirror, the global binding table,
// and the fixed-point grid encoding. Later pass families (transfers / constraint) are
// concatenated after this file into one shader module (WGSL has no imports), so this file
// declares everything they share. Binding indices are GLOBAL and stable across the module:
// each pipeline's auto layout (`layout: None`) keeps only the bindings its entry point actually
// uses, but an index never means two different buffers.
//
// U1 scope (KTD2, mirroring twofield's structure): only `grid_clear` (clear the fixed-point
// i32 grid lanes) + `advect` (gravity integration of particle positions) — enough that the
// dropdown shows moving particles. NO P2G/G2P/constraint yet (those are U3/U4). The full
// per-particle PB-MPM state (KTD8: `D`, `F`, `liquid_density`, constraint scalars) is laid out
// in the particle buffer NOW so the ABI is fixed up front; U3/U4 fill it in.
//
// Storage-buffer budget (KTD5): the widest entry point binds 2 storage buffers (`advect`: pos,
// vel — see MAX_STORAGE_BUFFERS_PER_ENTRY_POINT in mod.rs); the device requests 9 per stage
// (src/utils/gpu.rs NEEDED_STORAGE_BUFFERS), well within the 9 grant. Grid mass+momentum share
// ONE array<atomic<i32>> with stride 4 (mass, mom.xyz) per node — one binding, one clear pass.
//
// Tint discipline: any workgroupBarrier() must be reachable from uniform control flow — no
// early returns before a barrier (clamp indices and predicate the work instead). No pass in
// this module uses a barrier, so the per-thread guard returns are safe.

// Byte-identical to the Rust `Params` (128 bytes; vec4-aligned tail).
struct Params {
    box_min: vec4<f32>,     // simulation domain (w unused)
    box_max: vec4<f32>,     // (w unused)
    gravity: vec4<f32>,     // (w unused)
    grid_origin: vec4<f32>, // .xyz = world position of node (0,0,0); .w = cell size h
    grid_dims: vec4<u32>,   // nx, ny, nz, num_nodes (= nx·ny·nz)
    dt: f32,
    particle_mass: f32,   // per water particle (Materials::particle_mass)
    max_speed: f32,       // velocity cap — COUPLED to FP_SCALE (see headroom math below)
    water_count: u32,     // particles [0, water_count) are live water (range layout)
    particle_count: u32,  // allocated pool size (kernel live-set guard)
    // PB-MPM liquid constraint knobs (KTD1; inert in U1 — the constraint lands in U4):
    liquid_density: f32,    // rest liquid density target (1/liquid_density in the alpha term)
    liquid_relaxation: f32, // compliant volume-correction relaxation
    liquid_viscosity: f32,  // deviatoric shear-correction weight
    iter_pad: vec4<u32>,    // .x = iteration_count; .y = num_solids (SDF BC count, 0 = none);
                            // .z = restitution f32 bits (U5, bitcast<f32>); .w = pad
};

@group(0) @binding(0) var<uniform> params: Params;
// Canonical particle state (ParticleBuffers layout): pos.w is the moisture lane (water =
// remaining fraction), chem = (concentration, temperature, _, _).
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> phase: array<u32>;
@group(0) @binding(4) var<storage, read_write> chem: array<vec4<f32>>;
// Per-particle PB-MPM state (KTD8 — lives in the PARTICLE buffer as floats, never on the grid):
//   deform_disp[3p + r] = row r of the deformation displacement D (a mat3x3, vel-grad×dt), zeroed
//     at the start of each substep's iteration loop and accumulated across iterations.
//   deform_grad[3p + r] = row r of the deformation gradient F (a mat3x3), carried across substeps.
// 3 vec4 rows per particle each; .xyz = the matrix row, .w = 0 (constraint scalars reuse .w in U4).
// Allocated + identity-seeded (F = I, D = 0) in U1 so the ABI is fixed; the transfers/constraint
// passes that write them land in U3/U4.
@group(0) @binding(5) var<storage, read_write> deform_disp: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> deform_grad: array<vec4<f32>>;
// LIQUID grid field, fixed-point: 4 atomic<i32> lanes per node, stride 4 — [mass, mom.x, mom.y,
// mom.z], each value × FP_SCALE. The grid carries ONLY these (KTD8); per-particle state stays on
// the particle buffer. Cleared by grid_clear; scattered by p2g; decoded by grid_update.
@group(0) @binding(7) var<storage, read_write> grid_fp: array<atomic<i32>>;
// Grid velocity after grid_update (decode + gravity + domain BC): .xyz = velocity, .w = node mass
// (decoded float, diagnostics only). NOT a fixed-point lane (KTD8 — the grid carries only the
// atomic mass/momentum; this float scratch is the decoded transient the g2p gather reads). Mirrors
// twofield's `grid_vel`.
@group(0) @binding(8) var<storage, read_write> grid_vel: array<vec4<f32>>;
// SDF cavity geometry (U5; byte-identical to the Rust `Primitive`, 64 bytes — mirrors twofield's
// `Primitive` + utils/sdf.rs). Cone radii in `a` are OUTER wall radii. Each solid is a CAVITY:
// interior positive, gradient toward the cavity interior.
struct Primitive {
    kind: u32,          // 0 = cone, 1 = cylinder, 2 = poly-cup
    species_mask: u32,
    friction: f32,
    flags: u32,         // bit0 = apex_open
    a: vec4<f32>,       // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)  poly:(floor_y, rim_y, apothem, sides)
    b: vec4<f32>,       // cone:(thickness, hole_radius, center_x, center_z)  cyl/poly:(center_x, center_z, _, _)
    c: vec4<f32>,       // reserved
};
// Static SDF solids (U5 collider BC; read-only). Each `Primitive` is a CAVITY — `dist > 0` inside
// the free space, `< 0` through the wall material, gradient toward the cavity interior — so the BC
// CONTAINS water inside the cup. Bound only by `grid_update` (node-resolution momentum BC) and
// `particle_integrate` (particle-resolution push-out + restitution). `params.iter_pad.y =
// num_solids`; the union loops `[0, num_solids)` (uniform — Tint-safe, no early return).
@group(0) @binding(9) var<storage, read> solids: array<Primitive>;

// --- fixed-point encoding (KEEP.md §3 pattern; mirrors twofield's FP_SCALE) ------------------
//
// FP_SCALE = 2^18. i32 range is ±2^31, so an encoded node lane overflows at |value| ≥ 2^13 = 8192.
//   mass lane:      Σ_p m·w per node. Unit-mass water at rest loads an interior node with
//                   Σ m·w ≈ (h/spacing)³ = 8 → overflow needs ≈ 1000× rest packing. Unreachable.
//   momentum lanes: every scattered contribution is magnitude-clamped to max_speed, so overflow
//                   needs ≈ 20× rest compression with every particle at the cap (the same headroom
//                   derivation as twofield/common.wgsl — re-validate if a new scattered lane is
//                   added). COUPLED to the velocity cap: raising max_speed shrinks momentum
//                   headroom linearly. fp_encode saturates (clamp before convert) as a backstop.
const FP_SCALE: f32 = 262144.0; // 2^18
// Largest f32 ≤ i32::MAX (2^31 − 1 is not representable; nearest-below is 2147483520).
const FP_CLAMP: f32 = 2147483520.0;

fn fp_encode(x: f32) -> i32 {
    return i32(clamp(round(x * FP_SCALE), -FP_CLAMP, FP_CLAMP));
}
fn fp_decode(c: i32) -> f32 {
    return f32(c) / FP_SCALE;
}

// --- quadratic B-spline weights (mirrors twofield/common.wgsl; KEEP.md §3) --------------------
// Per axis, around base = floor(x/h − 0.5) with fx = x/h − base ∈ [0.5, 1.5]; node offset
// k ∈ {0,1,2} gets w[k]. Weights sum to 1 and Σ_k w[k]·(base+k) = x/h (linear consistency —
// what makes the free-fall recurrence exact and keeps the APIC affine reconstruction zero in a
// uniform field). The 3×3×3 (27-node) stencil per KTD6 — no cubic 4³.
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

const WG: u32 = 256u;

// --- grid_clear ------------------------------------------------------------------------------
// Clear all 4 fixed-point lanes of every node (mass + mom.xyz) before this iteration's scatter.
// The P2G/G2P transfers (transfers.wgsl) replace the U1 `advect` placeholder with the real
// PB-MPM transfer cycle (U3).

// Clear all 4 fixed-point lanes of every node (mass + mom.xyz).
@compute @workgroup_size(WG)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) { return; }
    let base = n * 4u;
    atomicStore(&grid_fp[base + 0u], 0);
    atomicStore(&grid_fp[base + 1u], 0);
    atomicStore(&grid_fp[base + 2u], 0);
    atomicStore(&grid_fp[base + 3u], 0);
}

// --- SDF cavity query (U5; mirrors twofield/common.wgsl + utils/sdf.rs) ------------------------
// Signed distance (interior POSITIVE) + unit gradient (toward the cavity interior) of the nearest
// blocking solid. The union over all primitives is the intersection of cavities (a particle must be
// inside ALL applicable cavities), so the binding constraint is the MIN signed distance. The loop
// runs `[0, num_solids)` (uniform bound, num_solids = 0 → returns FREE) — Tint-safe.
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

// Species-filtered union: the most-penetrated (min signed-distance) solid that blocks `ph`. The
// single-phase prototype only collides WATER (ph = 0), but the mask filter is kept so a future
// grain phase reuses this verbatim. num_solids = 0 → FREE (no constraint), so an empty scene falls
// through to the box clamp untouched.
fn solid_union(p: vec3<f32>, ph: u32) -> SolidHit {
    var best = SolidHit(SDF_FREE, vec3<f32>(0.0, 0.0, 0.0), 0.0);
    let n = params.iter_pad.y;
    for (var i = 0u; i < n; i = i + 1u) {
        let prim = solids[i];
        if ((prim.species_mask & (1u << ph)) == 0u) { continue; }
        let hit = solid_cavity(prim, p);
        if (hit.dist < best.dist) { best = hit; }
    }
    return best;
}

// Water is phase 0 in this single-phase prototype.
const PHASE_WATER: u32 = 0u;

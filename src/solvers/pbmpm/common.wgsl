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
    liquid_density: f32,    // INIT-only rest density (U6: the constraint reads the PER-PARTICLE
                            // accumulated density from deform_grad[3p+0].x, not this lane; the
                            // identity seed inits each particle's lane to 1.0 = this default)
    liquid_relaxation: f32, // compliant volume-correction relaxation
    liquid_viscosity: f32,  // negative-symmetric viscosity weight (EA SEED)
    iter_pad: vec4<u32>,    // .x = iteration_count; .y = num_solids (SDF BC count, 0 = none);
                            // .z = restitution f32 bits (U5, bitcast<f32>); .w = flip_fraction f32
                            // bits (SPLASH knob, bitcast<f32>)
    // U8 coarse-grid pressure pre-pass (coarse.wgsl). Coarse cell = COARSE_FACTOR fine nodes per
    // axis; the CPU derives the dims and the interior rest mass (see mod.rs coarse_spec_for).
    coarse_dims: vec4<u32>, // cx, cy, cz, num_ccells (= cx·cy·cz)
    coarse: vec4<f32>,      // .x = strength κ (0 = pass disabled, CPU skips the dispatches);
                            // .y = interior rest mass per coarse cell (surface classifier);
                            // .z = coarse cell size H (= COARSE_FACTOR · h);
                            // .w = kick cap (max |Δv| the apply pass may inject, velocity units)
    seam: vec4<f32>,        // Seam-blend bed BC (U2, docs/plans/2026-07-09-002): .x = enabled
                            // (1.0/0.0), .y = min node solid fraction counting as "in the bed",
                            // .z = saturation ratio treated as fully saturated (full block),
                            // .w = dry-bed percolation speed cap (v_perc = .w·(1−s))
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
//
// REPURPOSED LANE (U6, EA SEED per-particle liquidDensity): `deform_grad[3p + 0].x` (the F[0][0]
// slot) holds the per-particle accumulated liquid density (EA SEED `particle.liquidDensity` = the
// running product of the per-substep volume Jacobian tr(D)+1). The identity seed already writes
// F[0][0] = 1.0, so the build/reset seed gives the correct initial density = 1 for free; emit also
// sets it to 1.0 for activated slots (defensive). The rest of F (deform_grad) is otherwise unused by
// the single-phase liquid prototype, so this borrows one float lane rather than adding a buffer.
// READ by particle_update (the volume term's 1/liquidDensity); READ-WRITTEN by particle_integrate
// (the per-substep accumulation). Do NOT clobber pos.w (moisture) / vel.w (carried) for this.
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
// SDF thin-wall geometry (byte-identical to the Rust `Primitive`, 64 bytes — mirrors utils/sdf.rs
// packing). Cone radii in `a` are OUTER wall radii. Each solid is a THIN WALL (the vessel material
// only): `dist > 0` everywhere FREE (BOTH the vessel's open interior AND everything OUTSIDE the
// vessel), `dist < 0` only INSIDE the thin wall material, gradient = ∇dist points OUT of the wall
// toward the nearest free space.
struct Primitive {
    kind: u32,          // 0 = cone, 1 = cylinder, 2 = poly-cup
    species_mask: u32,
    friction: f32,
    flags: u32,         // bit0 = apex_open
    a: vec4<f32>,       // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)  poly:(floor_y, rim_y, apothem, sides)
    b: vec4<f32>,       // cone:(thickness, hole_radius, center_x, center_z)  cyl/poly:(center_x, center_z, _, _)
    c: vec4<f32>,       // reserved
};
// Static SDF solids (collider BC; read-only). Each `Primitive` is a THIN WALL — `dist > 0` in free
// space (the open vessel interior AND outside the vessel), `< 0` inside the wall material, gradient
// pointing OUT of the wall toward the nearest free side. Water spilling over a rim therefore falls
// FREELY beside the vessel (the OLD cavity model treated everything-outside as solid and shoved it
// back in — water could never overflow). The box BC (particle_integrate / grid_update) is the real
// container that catches escaped water. Bound only by `grid_update` (node-resolution momentum BC)
// and `particle_integrate` (particle-resolution push-out + restitution). `params.iter_pad.y =
// num_solids`; the union loops `[0, num_solids)` (uniform — Tint-safe, no early return).
@group(0) @binding(9) var<storage, read> solids: array<Primitive>;
// SPLASH (FLIP) snapshot buffers. `grid_vel_old` is the PRE-FORCE pure-transfer grid velocity
// (decode of the substep-start P2G with NO gravity, NO BC, NO constraint), written by
// grid_decode_old once per substep before the iteration loop; `vel_prev` is each particle's
// substep-start velocity, copied host-side via copy_buffer_to_buffer. particle_integrate gathers
// grid_vel_old over the 3×3×3 stencil and forms the FLIP velocity `v_prev + (v_pic − gathered_old)`,
// then blends `v = mix(v_pic, v_flip, flip_fraction)`. The constraint loop never reads these — FLIP
// is applied ONCE per substep so the incompressibility solve stays pure-PIC.
@group(0) @binding(10) var<storage, read_write> grid_vel_old: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> vel_prev: array<vec4<f32>>;
// U8 coarse-grid pressure pre-pass state (coarse.wgsl; only bound by the coarse passes).
// `coarse_fp`: one fixed-point mass lane per coarse cell (restriction target — same FP_SCALE
// encoding as grid_fp; headroom: ≤ 64 fine nodes × ~8m ≈ 512m ≪ the 2^13 ceiling).
// `coarse_list`: [0] = count, [1..] = compacted ACTIVE coarse-cell ids, atomic-appended on
// first-touch during restriction (capacity = num_ccells, so it can never overflow). Every list
// consumer over-dispatches to capacity and early-outs past the count — never indirect dispatch.
// `coarse_src`: .x = rhs (κ·e/dt density-error source), .y = kind (0 = air/surface → Dirichlet
// φ = 0; 1 = interior fluid row). `coarse_phi_a/b`: damped-Jacobi ping-pong potential.
@group(0) @binding(12) var<storage, read_write> coarse_fp: array<atomic<i32>>;
@group(0) @binding(13) var<storage, read_write> coarse_list: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> coarse_src: array<vec2<f32>>;
@group(0) @binding(15) var<storage, read_write> coarse_phi_a: array<f32>;
@group(0) @binding(16) var<storage, read_write> coarse_phi_b: array<f32>;
// Seam-blend bed coupling (U2; only bound by grid_update, only live when params.seam.x > 0).
// `bed_occupancy`: 4 fixed-point (FP_SCALE) lanes per node [solid volume V_eff, absorbed
// V_abs, capacity V_cap, unused], scattered per frame by the SEAM's scatter pass over the bed
// solver's grain particles — pbmpm only READS it (atomicLoad; atomics require read_write).
// `seam_reaction`: the impulse ledger the bed BC accumulates, at its OWN coarser scale
// SEAM_IMPULSE_SCALE = 2^12 (up to iteration_count node-mass × velocity impulses per frame
// would overflow the 2^13 value ceiling at FP_SCALE = 2^18). Consumed by the seam's twofield
// hook (U3); telemetry only — the R2 third-law gate measures momentum deltas directly.
@group(0) @binding(17) var<storage, read_write> bed_occupancy: array<atomic<i32>>;
@group(0) @binding(18) var<storage, read_write> seam_reaction: array<atomic<i32>>;

const SEAM_IMPULSE_SCALE: f32 = 4096.0;

// Occupancy-sampling index with the LATERAL wall clamp (x/z only, never y). The scatter
// kernel has no grains outside the walls to complete its support, so wall node columns read
// artificially low φ_s — a low-occupancy chute water measurably slides down (the M0 wall
// leak). Sampling is clamped ≥2 node columns in from the lateral faces: a wall-hugging
// particle reads the interior column's bed field (the bed material does reach the wall);
// the y axis stays exact so the bed SURFACE is never distorted.
fn seam_occ_index(node: vec3<i32>) -> u32 {
    let hix = max(i32(params.grid_dims.x) - 3, 0);
    let hiz = max(i32(params.grid_dims.z) - 3, 0);
    let c = vec3<i32>(
        clamp(node.x, min(2, hix), hix),
        node.y,
        clamp(node.z, min(2, hiz), hiz),
    );
    return node_index(c);
}

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

// --- SDF thin-wall query (mirrors utils/sdf.rs) ------------------------------------------------
// Signed distance (FREE space POSITIVE — both vessel interior and outside the vessel) + unit
// gradient (∇dist, pointing OUT of the wall toward the nearest free space) of the nearest blocking
// solid. Each vessel is built as a CSG SUBTRACTION: wall material = (outer solid) MINUS (inner
// cavity), so `wall_dist = max(SDF_O, -SDF_C)` is negative ONLY in the thin shell between them. The
// union over all primitives takes the MIN signed distance (the most-penetrated wall). The loop runs
// `[0, num_solids)` (uniform bound, num_solids = 0 → returns FREE) — Tint-safe.
//
// WALL_T — collision wall thickness (anti-tunneling). A thin shell can be crossed in one substep by
// a fast particle (the node BC bounds penetration only at node resolution h; the particle push-out
// fires only when the particle lands INSIDE the shell). The max per-substep displacement is
// max_speed·dt; at the default cap max_speed = 50 and dt = 1/60 that is ≈ 0.83 world units, and the
// web cap (12) gives ≈ 0.2. WALL_T = 1.0 sits above the default-cap bound with margin, so a particle
// at the cap cannot step clean through the shell. The wall is INVISIBLE (only a wireframe is
// rendered) so a collision shell thicker than the geometry's visual thickness is fine — water fills
// to the INNER radius and looks correct. The effective shell thickness is max(geometry_thickness,
// WALL_T): the cup has no geometry thickness (→ WALL_T) and the support cone's 0.05 visual thickness
// is far below the tunneling bound (→ WALL_T).
const WALL_T: f32 = 1.0;
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

// CONE thin wall = (outer cone) MINUS (inner cone). The outer profile is the OUTER radius
// (lerp(apex_r, top_r)); the inner profile is the cavity surface (outer − T, clamped to
// hole_radius). The wall material is the shell between them over y ∈ [apex_y, top_y]. The drain
// (r < hole_radius) and (apex_open ⇒ below apex_y) stay FREE — the funnel passes water through.
// `free_in` = signed distance to the INNER wall, positive on the cavity (small-r) side; `free_out` =
// signed distance to the OUTER wall, positive OUTSIDE the vessel (large-r) side. Free space is
// `free_in > 0 OR free_out > 0`, so `wall_dist = max(free_in, free_out)` is negative only in the
// shell and equals the nearest free face's (negative) distance; the gradient is that face's, pointing
// toward its free side (∇dist out of the wall).
fn cone_wall(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let apex_y = prim.a.x;
    let apex_r = prim.a.y;
    let top_y = prim.a.z;
    let top_r = prim.a.w;
    let thickness = max(prim.b.x, WALL_T); // collision shell ≥ tunneling bound (visual is wireframe)
    let hole_radius = prim.b.y;
    let center = vec2<f32>(prim.b.z, prim.b.w);
    let apex_open = (prim.flags & 1u) != 0u;

    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let r = length(rel);
    let y = p.y;
    var rhat = vec2<f32>(0.0, 0.0);
    if (r >= SDF_EPS) { rhat = rel / r; } // axis guard before building the 3D gradient

    // Free passages: above the open top, below the open apex outlet, and inside the drain hole.
    if (y > top_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, 1.0, 0.0), prim.friction); }
    if (apex_open && y < apex_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, -1.0, 0.0), prim.friction); }

    let p2d = vec2<f32>(r, y);
    let height = top_y - apex_y;
    var t = 0.0;
    if (height > SDF_EPS) { t = (y - apex_y) / height; }
    let outer_r = mix(apex_r, top_r, t);
    let inner_r = max(outer_r - thickness, hole_radius);

    // free_in: distance to the inner wall, positive inside the cavity (r ≤ inner_r). Gradient points
    // toward the cavity interior (smaller r). Mirrors the old cavity surface query exactly.
    let ai = vec2<f32>(max(apex_r - thickness, hole_radius), apex_y);
    let bi = vec2<f32>(max(top_r - thickness, hole_radius), top_y);
    let cpi = sdf_closest_on_segment(p2d, ai, bi);
    let di = length(p2d - cpi.p);
    let inside_cav = (y >= apex_y) && (r <= inner_r);
    var free_in = -di;
    if (inside_cav) { free_in = di; }
    let dir_in = p2d - cpi.p;
    var n_in = vec2<f32>(0.0, 0.0);
    if (length(dir_in) > SDF_EPS) {
        if (inside_cav) { n_in = sdf_normalize2(dir_in); } else { n_in = sdf_normalize2(-dir_in); }
    } else if (cpi.interior) {
        let ab = bi - ai;
        n_in = sdf_normalize2(vec2<f32>(-ab.y, ab.x)); // smooth-wall analytic normal (toward cavity)
    }

    // free_out: distance to the outer wall, positive OUTSIDE the vessel (r ≥ outer_r). Gradient
    // points outward (larger r) — the mirror of free_in with the free side flipped.
    let ao = vec2<f32>(apex_r, apex_y);
    let bo = vec2<f32>(top_r, top_y);
    let cpo = sdf_closest_on_segment(p2d, ao, bo);
    let do2 = length(p2d - cpo.p);
    let inside_outer = (y >= apex_y) && (r <= outer_r);
    var free_out = do2;
    if (inside_outer) { free_out = -do2; }
    let dir_out = p2d - cpo.p;
    var n_out = vec2<f32>(0.0, 0.0);
    if (length(dir_out) > SDF_EPS) {
        if (inside_outer) { n_out = sdf_normalize2(-dir_out); } else { n_out = sdf_normalize2(dir_out); }
    } else if (cpo.interior) {
        let ab = bo - ao;
        n_out = sdf_normalize2(vec2<f32>(ab.y, -ab.x)); // smooth-wall analytic normal (outward)
    }

    // CSG: wall = NOT-in-cavity AND NOT-outside-vessel ⇒ wall_dist = max(free_in, free_out), the
    // nearest free face. The active term's normal is the push-out direction.
    var dist = free_in;
    var n2d = n_in;
    if (free_out > free_in) { dist = free_out; n2d = n_out; }
    let grad = sdf_normalize3(vec3<f32>(n2d.x * rhat.x, n2d.y, n2d.x * rhat.y));
    return SolidHit(dist, grad, prim.friction);
}

// CYLINDER cup thin wall = (outer box) MINUS (inner open-top can), in the (r, y) plane. Outer box:
// r ≤ radius+T, y ∈ [floor_y−T, rim_y] (finite — capped at the rim so water spills OVER the top).
// Inner can: r ≤ radius, y ≥ floor_y (OPEN top, no upper cap). The wall is the side tube
// [radius, radius+T] + the floor disk [floor_y−T, floor_y]. Inside the cup, above the rim, and
// OUTSIDE the cup (r > radius+T) are all FREE. SDFs use the intersection-of-half-spaces form (exact
// sign + face gradients; corner distances are conservative, fine for push-out).
fn cyl_wall(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let floor_y = prim.a.x;
    let rim_y = prim.a.y;
    let radius = prim.a.z;
    let center = vec2<f32>(prim.b.x, prim.b.y);
    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let r = length(rel);
    let y = p.y;
    var rhat = vec2<f32>(0.0, 0.0);
    if (r >= SDF_EPS) { rhat = rel / r; }

    // sdf_O (neg inside the finite outer box): max of the three bounding half-spaces.
    let o_side = r - (radius + WALL_T);     // grad (+rhat): radial out
    let o_floor = (floor_y - WALL_T) - y;   // grad (−y): down
    let o_rim = y - rim_y;                   // grad (+y): up
    var sdf_o = o_side;
    var n_o = vec3<f32>(rhat.x, 0.0, rhat.y);
    if (o_floor > sdf_o) { sdf_o = o_floor; n_o = vec3<f32>(0.0, -1.0, 0.0); }
    if (o_rim > sdf_o) { sdf_o = o_rim; n_o = vec3<f32>(0.0, 1.0, 0.0); }

    // sdf_C (neg inside the open-top can): max(r − radius, floor_y − y). −sdf_C is positive in the
    // cavity; its gradient points into the cavity (toward smaller r / upward off the floor).
    let c_side = r - radius;     // active ⇒ ∇(−sdf_C) = (−rhat): radial in
    let c_floor = floor_y - y;   // active ⇒ ∇(−sdf_C) = (+y): up
    var sdf_c = c_side;
    var n_c = vec3<f32>(-rhat.x, 0.0, -rhat.y);
    if (c_floor > sdf_c) { sdf_c = c_floor; n_c = vec3<f32>(0.0, 1.0, 0.0); }

    // wall = O ∩ (¬C) ⇒ dist = max(sdf_O, −sdf_C); the active term's normal is ∇dist.
    var dist = sdf_o;
    var grad = n_o;
    if (-sdf_c > sdf_o) { dist = -sdf_c; grad = n_c; }
    return SolidHit(dist, sdf_normalize3(grad), prim.friction);
}

// DIAGNOSTIC: regular N-gon prism cup thin wall (mirror of cyl_wall with the polygon apothem in
// place of the radius). Face normals at 2πk/N (k=0 → +x); sides=4 is an axis-aligned square. The
// radial coordinate is replaced by the apothem projection `proj = max_k(rel·n_k)` and its face
// normal `bestn`.
fn poly_wall(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let floor_y = prim.a.x;
    let rim_y = prim.a.y;
    let apothem = prim.a.z;
    let sides = max(u32(prim.a.w), 3u);
    let center = vec2<f32>(prim.b.x, prim.b.y);
    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let y = p.y;
    var maxproj = -1.0e30;
    var bestn = vec2<f32>(1.0, 0.0);
    let tau = 6.28318530718;
    for (var k = 0u; k < sides; k = k + 1u) {
        let ang = tau * f32(k) / f32(sides);
        let nk = vec2<f32>(cos(ang), sin(ang));
        let proj = dot(rel, nk);
        if (proj > maxproj) { maxproj = proj; bestn = nk; }
    }

    let o_side = maxproj - (apothem + WALL_T);   // grad (+bestn): outward face normal
    let o_floor = (floor_y - WALL_T) - y;
    let o_rim = y - rim_y;
    var sdf_o = o_side;
    var n_o = vec3<f32>(bestn.x, 0.0, bestn.y);
    if (o_floor > sdf_o) { sdf_o = o_floor; n_o = vec3<f32>(0.0, -1.0, 0.0); }
    if (o_rim > sdf_o) { sdf_o = o_rim; n_o = vec3<f32>(0.0, 1.0, 0.0); }

    let c_side = maxproj - apothem;
    let c_floor = floor_y - y;
    var sdf_c = c_side;
    var n_c = vec3<f32>(-bestn.x, 0.0, -bestn.y);
    if (c_floor > sdf_c) { sdf_c = c_floor; n_c = vec3<f32>(0.0, 1.0, 0.0); }

    var dist = sdf_o;
    var grad = n_o;
    if (-sdf_c > sdf_o) { dist = -sdf_c; grad = n_c; }
    return SolidHit(dist, sdf_normalize3(grad), prim.friction);
}

fn solid_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    if (prim.kind == 0u) { return cone_wall(prim, p); }
    if (prim.kind == 2u) { return poly_wall(prim, p); }
    return cyl_wall(prim, p);
}

// Species-filtered union: the most-penetrated (min signed-distance) wall that blocks `ph`. The
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

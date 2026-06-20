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
    iter_pad: vec4<u32>,    // .x = iteration_count (constraint→grid rebuild loop); .yzw = pad
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

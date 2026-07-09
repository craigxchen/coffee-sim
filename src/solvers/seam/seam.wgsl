// Seam passes (U2, docs/plans/2026-07-09-002): clear + scatter the bed-occupancy field on
// the SHARED grid (co-registered with both inner solvers by construction; asserted at build).
//
// `bed_occupancy` is pbmpm's binding-17 buffer (the seam writes it, pbmpm's grid_update bed
// BC reads it): 4 fixed-point lanes per node [V_eff, V_abs, V_cap, unused] at FP_SCALE =
// 2^18 — the same encoding pbmpm/twofield use for their grid lanes. The scatter reads the
// bed solver's PERSISTENT grain particle buffer (pos.w = absorbed volume V_abs), never its
// per-substep grid scratch (KTD2: inner scratch is not trusted across the solver boundary).
//
// The field is cleared EVERY frame immediately before the scatter (review r2.2 — an
// accumulated-occupancy leak is silent otherwise). Scatter kernel: quadratic B-spline 3×3×3
// (the same class p2g uses), NOT trilinear — with grain_diameter = 2·spacing the grain seed
// lattice lands EXACTLY on every other grid node, and a trilinear scatter then deposits each
// grain's whole volume onto a single node: the occupancy field becomes a checkerboard with
// φ_s = 0 channels that water threads straight through (the measured M0 leak). The B-spline
// spreads an on-node grain over 27 nodes, so the field has no holes at any lattice alignment.

struct SeamParams {
    grid_origin: vec4<f32>, // .xyz = node (0,0,0) world position; .w = cell size h
    grid_dims: vec4<u32>,   // nx, ny, nz, num_nodes
    solid: vec4<u32>,       // .x = solid range start (element index in bed_pos), .y = count
    wet: vec4<f32>,         // .x = V_dry (π/6·d³), .y = V_cap, .zw = reserved
};

@group(0) @binding(0) var<uniform> params: SeamParams;
// The bed solver's particle positions (read-only; grains at [solid.x, solid.x + solid.y)).
@group(0) @binding(1) var<storage, read> bed_pos: array<vec4<f32>>;
// pbmpm's bed_occupancy (binding 17 there): 4 fixed-point lanes per node.
@group(0) @binding(2) var<storage, read_write> bed_occupancy: array<atomic<i32>>;

const FP_SCALE: f32 = 262144.0; // 2^18 — must match pbmpm/twofield common.wgsl
const WG: u32 = 256u;

@compute @workgroup_size(WG)
fn seam_clear_bed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    atomicStore(&bed_occupancy[n * 4u + 0u], 0);
    atomicStore(&bed_occupancy[n * 4u + 1u], 0);
    atomicStore(&bed_occupancy[n * 4u + 2u], 0);
    atomicStore(&bed_occupancy[n * 4u + 3u], 0);
}

// Quadratic B-spline weights per axis for a particle at fractional offset fx from `base`
// (base = floor(gp − 0.5); support = base..base+2). Local copy — seam.wgsl is a standalone
// module (WGSL has no imports); must match pbmpm common.wgsl's bspline_w.
fn seam_bspline_w(fx: vec3<f32>) -> array<vec3<f32>, 3> {
    var w: array<vec3<f32>, 3>;
    let d0 = fx;                       // distance to base + 0.. offsets
    w[0] = 0.5 * (1.5 - d0) * (1.5 - d0);
    let d1 = fx - vec3<f32>(1.0);
    w[1] = vec3<f32>(0.75) - d1 * d1;
    let d2 = fx - vec3<f32>(2.0);
    w[2] = 0.5 * (1.5 + d2) * (1.5 + d2);
    return w;
}

// Scatter each grain's effective volume (dry + absorbed swelling — the V_eff convention),
// absorbed volume, and capacity over its 3×3×3 node neighborhood (quadratic B-spline).
@compute @workgroup_size(WG)
fn seam_scatter_bed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid.y) {
        return;
    }
    let p = bed_pos[params.solid.x + i];
    let v_abs = max(p.w, 0.0);
    let v_eff = params.wet.x + v_abs;
    let h = params.grid_origin.w;
    let gp = (p.xyz - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(gp - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = gp - vec3<f32>(base);
    let w = seam_bspline_w(fx);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i2 = 0; i2 < 3; i2 = i2 + 1) {
                let node = base + vec3<i32>(i2, j, k);
                let wt = w[i2].x * w[j].y * w[k].z;
                let n = u32(node.x)
                    + u32(node.y) * params.grid_dims.x
                    + u32(node.z) * params.grid_dims.x * params.grid_dims.y;
                atomicAdd(&bed_occupancy[n * 4u + 0u], i32(wt * v_eff * FP_SCALE));
                atomicAdd(&bed_occupancy[n * 4u + 1u], i32(wt * v_abs * FP_SCALE));
                atomicAdd(&bed_occupancy[n * 4u + 2u], i32(wt * params.wet.y * FP_SCALE));
            }
        }
    }
}

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
    solid: vec4<u32>,       // .x = solid range start in bed_pos, .y = count,
                            // .z = live WATER count (host-refreshed per frame, U4),
                            // .w = absorption quantum V_w·FP_SCALE (whole-particle, U4)
    wet: vec4<f32>,         // .x = V_dry (π/6·d³), .y = V_cap,
                            // .z = absorption rate factor 1 − e^{−tf_absorb_rate·dt} (U4),
                            // .w = wet_sat_cutoff = V_cap·(1 − absorb_roundoff) (U4)
};

@group(0) @binding(0) var<uniform> params: SeamParams;
// The bed solver's particle buffer (grains at [solid.x, solid.x + solid.y)). read_write:
// seam_credit writes absorbed volume into grain pos.w (the moisture lane).
@group(0) @binding(1) var<storage, read_write> bed_pos: array<vec4<f32>>;
// pbmpm's bed_occupancy (binding 17 there): 4 fixed-point lanes per node
// [V_eff, V_abs, V_cap, demand(U4)].
@group(0) @binding(2) var<storage, read_write> bed_occupancy: array<atomic<i32>>;
// pbmpm's seam_reaction (binding 18 there): lanes 0-2 = BC impulses (U3);
// lane 3 = consumed absorption volume per node (U4, FP_SCALE — written by seam_mark,
// read by seam_credit next frame, zeroed by the seam after credit).
@group(0) @binding(3) var<storage, read_write> seam_reaction: array<atomic<i32>>;
// Mark list (U4): [0] = atomic count, [1..] = marked pbmpm water particle indices.
@group(0) @binding(4) var<storage, read_write> marks: array<atomic<u32>>;
// pbmpm's water positions (read; live prefix [0, solid.z)).
@group(0) @binding(5) var<storage, read> water_pos: array<vec4<f32>>;
// PERSISTENT per-node demand bank (U4): per-frame demand is ~30× smaller than the
// whole-particle quantum, so instantaneous demand alone can never trigger a transfer —
// it accrues here across frames (soft-capped) and is spent in quanta. Marking still
// requires nonzero INSTANTANEOUS demand (lane 3), so a saturated bed cannot spend a
// stale bank. Never cleared; owned by the seam.
@group(0) @binding(6) var<storage, read_write> seam_bank: array<atomic<i32>>;

const FP_SCALE: f32 = 262144.0; // 2^18 — must match pbmpm/twofield common.wgsl
const WG: u32 = 256u;
// Mark-list capacity (entries after the count slot). Overflow undoes the consumption —
// absorption just proceeds next frame; never silent loss.
const MARK_CAP: u32 = 2048u;

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

// ==================================== U4: absorption ============================================
// Whole-particle infiltration handoff (plan §U4 design). Frame N end: seam_demand scatters
// per-node absorption demand; seam_mark lets bed-contact water consume whole quanta from it
// (atomicSub-with-undo) and records consumption per node. Frame N+1 start: the host removes
// the marked particles, then seam_credit distributes each node's consumed volume to grains
// by demand share — the credited total equals the consumed total exactly by construction.

// Per-grain demand this frame: (V_cap − V_abs)·(1 − e^{−k·dt}), floored to 0 at the
// wet_sat_cutoff (the saturated-tail contract). Scattered per node into lane 3.
@compute @workgroup_size(WG)
fn seam_demand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid.y) {
        return;
    }
    let p = bed_pos[params.solid.x + i];
    let v_abs = max(p.w, 0.0);
    if (v_abs >= params.wet.w) {
        return; // saturated: zero demand, exactly
    }
    let demand = (params.wet.y - v_abs) * params.wet.z;
    if (demand <= 0.0) {
        return;
    }
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
                let d_fp = i32(wt * demand * FP_SCALE);
                atomicAdd(&bed_occupancy[n * 4u + 3u], d_fp);
                // Accrue the bank (soft cap 4 quanta — the check races benignly).
                if (atomicLoad(&seam_bank[n]) < i32(params.solid.w) * 4) {
                    atomicAdd(&seam_bank[n], d_fp);
                }
            }
        }
    }
}

// Bed-contact water tries to consume ONE whole quantum (V_w) of its nearest node's demand.
// Success ⇒ the particle is marked for removal (applied next frame by the host) and the
// quantum is recorded in the consumed lane. Every failure path undoes its subtraction —
// demand never goes silently negative, marks never overflow silently.
@compute @workgroup_size(WG)
fn seam_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid.z) {
        return;
    }
    let h = params.grid_origin.w;
    let gp = (water_pos[i].xyz - params.grid_origin.xyz) / h;
    let nn = clamp(
        vec3<i32>(round(gp)),
        vec3<i32>(0),
        vec3<i32>(params.grid_dims.xyz) - vec3<i32>(1),
    );
    let n = u32(nn.x)
        + u32(nn.y) * params.grid_dims.x
        + u32(nn.z) * params.grid_dims.x * params.grid_dims.y;
    // Bed contact: the particle sits where the bed field is live.
    let phi = fp_decode_i(atomicLoad(&bed_occupancy[n * 4u + 0u])) / (h * h * h);
    if (phi <= 0.15) {
        return;
    }
    // Grains at this node must be thirsty NOW (instantaneous demand — a saturated bed
    // never spends a stale bank), and the bank must hold a whole quantum.
    if (atomicLoad(&bed_occupancy[n * 4u + 3u]) <= 0) {
        return;
    }
    let q = i32(params.solid.w);
    let before = atomicSub(&seam_bank[n], q);
    if (before < q) {
        atomicAdd(&seam_bank[n], q); // bank below one quantum: undo, keep accruing
        return;
    }
    let slot = atomicAdd(&marks[0], 1u);
    if (slot >= MARK_CAP) {
        atomicSub(&marks[0], 1u);
        atomicAdd(&seam_bank[n], q); // list full: undo, absorb next frame
        return;
    }
    atomicStore(&marks[1u + slot], i);
    atomicAdd(&seam_reaction[n * 4u + 3u], q);
}

// Distribute each node's consumed volume to its grains by demand share. Runs at frame N+1
// start, BEFORE seam_clear_bed (lane 3 still holds frame-N REMAINING demand) and BEFORE the
// reaction ledger is zeroed (lane 3 there holds frame-N CONSUMED volume). Grain positions
// and V_abs are unchanged since the demand pass (nothing has stepped yet), so the demand
// recomputation is exact: share = demand_g·w / (remaining_n + consumed_n).
@compute @workgroup_size(WG)
fn seam_credit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid.y) {
        return;
    }
    let pi = params.solid.x + i;
    let p = bed_pos[pi];
    let v_abs = max(p.w, 0.0);
    if (v_abs >= params.wet.w) {
        return;
    }
    let demand = (params.wet.y - v_abs) * params.wet.z;
    if (demand <= 0.0) {
        return;
    }
    let h = params.grid_origin.w;
    let gp = (p.xyz - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(gp - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = gp - vec3<f32>(base);
    let w = seam_bspline_w(fx);
    var gained = 0.0;
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i2 = 0; i2 < 3; i2 = i2 + 1) {
                let node = base + vec3<i32>(i2, j, k);
                let wt = w[i2].x * w[j].y * w[k].z;
                let n = u32(node.x)
                    + u32(node.y) * params.grid_dims.x
                    + u32(node.z) * params.grid_dims.x * params.grid_dims.y;
                let consumed = atomicLoad(&seam_reaction[n * 4u + 3u]);
                if (consumed <= 0) {
                    continue;
                }
                // Denominator: the INTACT instantaneous node demand from frame N's scatter
                // (the mark pass spends the bank, not lane 3; seam_clear_bed runs after
                // credit). Σ over grains of the numerators equals it exactly, so the
                // credited total equals the consumed total by construction.
                let demand_n = atomicLoad(&bed_occupancy[n * 4u + 3u]);
                if (demand_n <= 0) {
                    continue;
                }
                // The numerator MUST replay the scatter's i32 truncation — an untruncated
                // float numerator sums to MORE than the truncated denominator and
                // over-credits ~5% of the transfer (measured). Identical arithmetic ⇒
                // Σ_grains numerators == demand_n exactly ⇒ credited == consumed exactly.
                gained = gained + f32(consumed) * f32(i32(wt * demand * FP_SCALE)) / f32(demand_n);
            }
        }
    }
    if (gained > 0.0) {
        bed_pos[pi].w = min(v_abs + gained / FP_SCALE, params.wet.y);
    }
}

fn fp_decode_i(v: i32) -> f32 {
    return f32(v) / FP_SCALE;
}

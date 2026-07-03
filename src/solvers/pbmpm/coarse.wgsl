// U8 coarse-grid pressure pre-pass — the low-frequency correction the pure-local-Jacobi
// constraint converges too slowly (the measured settled-pool drift was +11.1% vs the +3%
// pre-registered trigger; docs/plans/2026-06-20-001 KTD4).
//
// Shape (once per substep, between the FLIP snapshot decode and the constraint loop):
//   coarse_clear    zero the coarse lanes + the list counter (num_ccells threads — tiny)
//   coarse_restrict per FINE node: fold its snapshot mass into its coarse cell (fixed-point
//                   atomicAdd) and atomic-append the cell to the compacted ACTIVE list on
//                   first-touch — the U8-mandated pattern (never an all-cells reduction, never
//                   indirect dispatch; consumers over-dispatch to capacity and early-out)
//   coarse_source   per ACTIVE cell: density error e = mass/rest − 1, interior/surface
//                   classification (kind), rhs = κ·e/dt
//   coarse_jacobi   ×COARSE_SWEEPS (even, CPU loop): plain Jacobi ping-pong on the 7-point
//                   Poisson ∇²φ = κ·e/dt over ACTIVE cells only. Air/surface rows are
//                   Dirichlet φ = 0 (free surface); out-of-grid neighbors are Neumann mirrors
//                   (domain walls). The active list is a few hundred cells — each sweep is a
//                   trivially parallel guard-return pass, no barriers, no reductions.
//   coarse_apply    per particle: v += −∇φ (trilinear over coarse cell centers), magnitude-
//                   capped by params.coarse.w. κ is already folded into rhs, so φ scales the
//                   whole correction linearly; e < 0 (under-dense) produces the opposite sign —
//                   the correction is TWO-SIDED and self-limiting (shrinks as e → 0), unlike a
//                   one-sided relief source.
// Sign audit (continuity: dρ/dt = −ρ∇·v, so removing OVER-density needs ∇·Δv > 0 = expansion):
// we solve ∇²φ = +κ·e/dt and apply Δv = +∇φ, so ∇·Δv = ∇²φ = +κ·e/dt — expansion where e > 0,
// contraction where e < 0. Pointwise picture: over-dense interior ⇒ rhs > 0 ⇒ φ < 0 inside
// (Dirichlet-0 rim) ⇒ ∇φ points outward (toward the higher rim value) ⇒ the kick pushes the
// pool boundary OUT. (The first cut applied −∇φ — the standard projection reflex — and measured
// +194% drift: it compressed the already-over-dense pool harder every substep.)
//
// Tint discipline: no workgroup barriers anywhere in this family — guard returns only.
// Determinism: restriction is fixed-point integer atomics (order-independent sums); the LIST
// ORDER is nondeterministic but every consumer is order-independent (per-cell writes, gathers).

// Fine nodes per coarse cell per axis. Mirrored in mod.rs (COARSE_FACTOR — dims + rest mass).
const COARSE_FACTOR: u32 = 4u;

fn ccell_coords(c: u32) -> vec3<u32> {
    let cx = params.coarse_dims.x;
    let cy = params.coarse_dims.y;
    return vec3<u32>(c % cx, (c / cx) % cy, c / (cx * cy));
}

fn ccell_index(c: vec3<u32>) -> u32 {
    return c.x + params.coarse_dims.x * (c.y + params.coarse_dims.y * c.z);
}

@compute @workgroup_size(WG)
fn coarse_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.coarse_dims.w) {
        return;
    }
    atomicStore(&coarse_fp[n], 0);
    coarse_src[n] = vec2<f32>(0.0, 0.0);
    coarse_phi_a[n] = 0.0;
    coarse_phi_b[n] = 0.0;
    if (n == 0u) {
        atomicStore(&coarse_list[0], 0u);
    }
}

// One thread per FINE node: restrict the snapshot mass (grid_fp was scattered by the substep's
// FLIP-snapshot p2g just before this) into the node's coarse cell. First-touch (atomicAdd
// returning old == 0) appends the cell to the compacted active list; capacity == num_ccells so
// the append can never overflow.
@compute @workgroup_size(WG)
fn coarse_restrict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let counts = atomicLoad(&grid_fp[n * 4u]);
    if (counts <= 0) {
        return;
    }
    let nx = n % params.grid_dims.x;
    let ny = (n / params.grid_dims.x) % params.grid_dims.y;
    let nz = n / (params.grid_dims.x * params.grid_dims.y);
    let cc = ccell_index(vec3<u32>(nx, ny, nz) / COARSE_FACTOR);
    let old = atomicAdd(&coarse_fp[cc], counts);
    if (old == 0) {
        let slot = atomicAdd(&coarse_list[0], 1u);
        coarse_list[1u + slot] = cc;
    }
}

// Per ACTIVE cell: density error → rhs + interior/surface kind. Interior = mass ≥ half the
// full-cell rest mass (params.coarse.y); partial/surface cells stay Dirichlet φ = 0, which is
// the ghost-fluid-lite free-surface row (mirrors the fine-node fill-fraction idiom in twofield).
@compute @workgroup_size(WG)
fn coarse_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&coarse_list[0])) {
        return;
    }
    let c = coarse_list[1u + i];
    let mass = fp_decode(atomicLoad(&coarse_fp[c]));
    let rest = params.coarse.y;
    if (mass < 0.5 * rest) {
        return; // surface/partial cell: kind stays 0, φ pinned 0 by the clear
    }
    // Bounded error keeps a pathological cell from detonating the rhs (the fine constraint's
    // per-particle clamp is the same posture).
    let e = clamp(mass / rest - 1.0, -0.9, 4.0);
    coarse_src[c] = vec2<f32>(params.coarse.x * e / params.dt, 1.0);
}

// One plain-Jacobi sweep over the ACTIVE list: read binding 15 (phi src), write binding 16
// (phi dst); the CPU swaps the two buffers at the bind-group level per sweep (even total sweep
// count, so the final potential lands back in phi_a — coarse_apply binds phi_a).
@compute @workgroup_size(WG)
fn coarse_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= atomicLoad(&coarse_list[0])) {
        return;
    }
    let c = coarse_list[1u + i];
    let sk = coarse_src[c];
    if (sk.y < 0.5) {
        coarse_phi_b[c] = 0.0; // Dirichlet surface row: keep both parities pinned
        return;
    }
    let cc = vec3<i32>(ccell_coords(c));
    let dims = vec3<i32>(params.coarse_dims.xyz);
    let self_phi = coarse_phi_a[c];
    var sum = 0.0;
    for (var a = 0; a < 3; a = a + 1) {
        for (var s = -1; s <= 1; s = s + 2) {
            var nb = cc;
            nb[a] = nb[a] + s;
            if (nb[a] < 0 || nb[a] >= dims[a]) {
                sum = sum + self_phi; // out-of-grid = domain wall: Neumann mirror
            } else {
                // In-grid neighbor: active interior reads its potential; air/surface cells hold
                // φ = 0 (cleared + never written) = the Dirichlet free-surface contribution.
                sum = sum + coarse_phi_a[ccell_index(vec3<u32>(nb))];
            }
        }
    }
    let hh = params.coarse.z * params.coarse.z;
    coarse_phi_b[c] = (sum - hh * sk.x) / 6.0;
}

// Per particle: gather −∇φ (analytic gradient of the trilinear interpolant over coarse CELL
// CENTERS), cap its magnitude, and kick the particle velocity. Runs BEFORE the constraint loop,
// so the loop's p2g carries the corrected velocity and the iterated local constraint only has
// the high-frequency remainder to fix. The FLIP snapshot was taken before this pass, so the
// kick is part of the "constraint velocity change" FLIP preserves — the splash lever is
// untouched.
@compute @workgroup_size(WG)
fn coarse_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    let hc = params.coarse.z;
    // Cell-center convention: cell c spans [c·H, (c+1)·H) from the grid origin, center at
    // (c + 0.5)·H.
    let xl = (pos[p].xyz - params.grid_origin.xyz) / hc - vec3<f32>(0.5);
    let dims = vec3<i32>(params.coarse_dims.xyz);
    var base = vec3<i32>(floor(xl));
    base = clamp(base, vec3<i32>(0), dims - vec3<i32>(2));
    let f = clamp(xl - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));

    var phi: array<f32, 8>;
    for (var k = 0; k < 2; k = k + 1) {
        for (var j = 0; j < 2; j = j + 1) {
            for (var i = 0; i < 2; i = i + 1) {
                let cc = base + vec3<i32>(i, j, k);
                phi[i + 2 * j + 4 * k] = coarse_phi_a[ccell_index(vec3<u32>(cc))];
            }
        }
    }
    let gx = ((phi[1] - phi[0]) * (1.0 - f.y) + (phi[3] - phi[2]) * f.y) * (1.0 - f.z)
        + ((phi[5] - phi[4]) * (1.0 - f.y) + (phi[7] - phi[6]) * f.y) * f.z;
    let gy = ((phi[2] - phi[0]) * (1.0 - f.x) + (phi[3] - phi[1]) * f.x) * (1.0 - f.z)
        + ((phi[6] - phi[4]) * (1.0 - f.x) + (phi[7] - phi[5]) * f.x) * f.z;
    let gz = ((phi[4] - phi[0]) * (1.0 - f.x) + (phi[5] - phi[1]) * f.x) * (1.0 - f.y)
        + ((phi[6] - phi[2]) * (1.0 - f.x) + (phi[7] - phi[3]) * f.x) * f.y;
    // Δv = +∇φ (see the sign audit in the header — expansion for over-dense regions).
    var dv = vec3<f32>(gx, gy, gz) / hc;
    let s = length(dv);
    let cap = params.coarse.w;
    if (s > cap) {
        dv = dv * (cap / s);
    }
    vel[p] = vec4<f32>(vel[p].xyz + dv, vel[p].w);
}

// PB-MPM APIC fixed-point transfers (U3): p2g → grid_update → g2p, one APIC transfer cycle per
// substep (the iteration_count loop + the compliant density constraint are U4 — NOT here). The
// grid_clear pass lives in common.wgsl. Concatenated after common.wgsl into one shader module.
//
// Per-substep recurrence (the solver's OWN discrete form — tests/pbmpm_transfers.rs derives its
// round-trip reference from exactly this):
//   p2g scatters m and m·(v + D·d) at the OLD positions (fixed-point atomicAdd, order-
//     independent integer sums → bit-deterministic accumulation);
//   grid_update divides momentum/mass → velocity, applies gravity (v ← v + g·dt), and a simple
//     domain BC (zero the into-wall normal so water stays in the box — collider BC/restitution is
//     U5);
//   g2p gathers the new particle velocity, reconstructs the APIC affine matrix into the per-
//     particle D (KTD8: D = B·D⁻¹, the velocity-gradient state U4's tr(D) reads), and advects
//     x ← x + v_new·dt.
// In free air (no BC) the per-substep closed form is v ← v + g·dt ; x ← x + v_new·dt
// (semi-implicit Euler), because the B-spline weights partition unity (the gather is exact for a
// uniform field) and D stays ~0 in a uniform field (linear consistency).
//
// Only the live WATER range [0, water_count) is touched (single-phase prototype). The grid is a
// transient transfer medium rebuilt each substep — there is no standing pressure field.

// Quadratic B-spline scatter of mass + APIC affine momentum m·(v + D·(x_i − x_p)) via fixed-point
// atomics. D is the per-particle affine/velocity-gradient matrix (deform_disp, KTD8); on the
// first U3 cycle D = 0 ⇒ pure mass+velocity scatter. 3×3×3 (27-node) stencil per KTD6.
@compute @workgroup_size(WG)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let v = vel[p].xyz;
    // Per-particle affine matrix rows (D): deform_disp[3p + r].xyz is row r (D·d)[r] = dot(row_r, d).
    let d0 = deform_disp[3u * p + 0u].xyz;
    let d1 = deform_disp[3u * p + 1u].xyz;
    let d2 = deform_disp[3u * p + 2u].xyz;

    let xl = (x - params.grid_origin.xyz) / h;
    // g2p clamps particles into the box and the grid pads one node layer beyond each face, so base
    // is always in [0, dims−3]; clamp anyway (Tint clamp-and-predicate, never branches).
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    let w = bspline_w(fx);
    let m = params.particle_mass;

    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                let d = (vec3<f32>(node) - xl) * h; // x_i − x_p (world units)
                var vaff = v + vec3<f32>(dot(d0, d), dot(d1, d), dot(d2, d));
                // Magnitude-clamp the scattered velocity to max_speed — this is what makes the
                // FP_SCALE momentum-headroom bound in common.wgsl hold by construction (mirrors
                // twofield). COUPLED to the velocity cap.
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

// Decode fixed-point mass/momentum → node velocity, apply gravity, and a simple domain BC. The
// decoded velocity is written to the FLOAT grid_vel scratch (KTD8 — not a fixed-point lane).
// Empty nodes get zero velocity. Collider SDF BC + restitution is U5; here a node sitting on a
// domain face just has its into-wall normal component zeroed so water stays in the box.
@compute @workgroup_size(WG)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let mc = atomicLoad(&grid_fp[n * 4u + 0u]);
    var v = vec3<f32>(0.0);
    let mass = fp_decode(mc);
    if (mass > 0.0) {
        // momentum/mass: both lanes carry the SAME FP_SCALE, so the integer ratio recovers the
        // physical velocity directly (the scale cancels).
        let inv = 1.0 / f32(mc);
        v = vec3<f32>(
            f32(atomicLoad(&grid_fp[n * 4u + 1u])),
            f32(atomicLoad(&grid_fp[n * 4u + 2u])),
            f32(atomicLoad(&grid_fp[n * 4u + 3u]))
        ) * inv;
        // Gravity on every mass-carrying node (the substep's body force; G2P advects with v_new).
        v = v + params.gravity.xyz * params.dt;
    }

    // Simple domain BC (U3 placeholder; collider SDF + restitution is U5): a node within one cell
    // of a domain face zeroes the into-wall normal component so the pool cannot leak out the box.
    let h = params.grid_origin.w;
    let nx = n % params.grid_dims.x;
    let ny = (n / params.grid_dims.x) % params.grid_dims.y;
    let nz = n / (params.grid_dims.x * params.grid_dims.y);
    let wp = params.grid_origin.xyz + vec3<f32>(f32(nx), f32(ny), f32(nz)) * h;
    if (wp.x <= params.box_min.x && v.x < 0.0) { v.x = 0.0; }
    if (wp.y <= params.box_min.y && v.y < 0.0) { v.y = 0.0; }
    if (wp.z <= params.box_min.z && v.z < 0.0) { v.z = 0.0; }
    if (wp.x >= params.box_max.x && v.x > 0.0) { v.x = 0.0; }
    if (wp.y >= params.box_max.y && v.y > 0.0) { v.y = 0.0; }
    if (wp.z >= params.box_max.z && v.z > 0.0) { v.z = 0.0; }

    grid_vel[n] = vec4<f32>(v, mass);
}

// APIC gather: new particle velocity v = Σ w·v_i and the affine matrix B = Σ w·v_i·dᵀ, then
// D = B·D⁻¹ with D⁻¹ = (4/h²)·I for the quadratic B-spline (Jiang et al. APIC). D is written back
// to the per-particle deform_disp (KTD8: the velocity-gradient state U4's constraint reads via
// tr(D)). Then advect x ← x + v·dt and apply a particle-resolution box clamp.
@compute @workgroup_size(WG)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    let h = params.grid_origin.w;
    var x = pos[p].xyz;

    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    let w = bspline_w(fx);

    var v = vec3<f32>(0.0);
    var b0 = vec3<f32>(0.0); // rows of B = Σ w·v_i·dᵀ
    var b1 = vec3<f32>(0.0);
    var b2 = vec3<f32>(0.0);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                let d = (vec3<f32>(node) - xl) * h; // x_i − x_p (world units)
                let gv = grid_vel[node_index(node)].xyz;
                v = v + wijk * gv;
                b0 = b0 + (wijk * gv.x) * d;
                b1 = b1 + (wijk * gv.y) * d;
                b2 = b2 + (wijk * gv.z) * d;
            }
        }
    }
    let dinv = 4.0 / (h * h);
    // APIC affine reconstruction → the per-particle D (velocity gradient; D·d in p2g next cycle).
    let nd0 = b0 * dinv;
    let nd1 = b1 * dinv;
    let nd2 = b2 * dinv;

    // Velocity cap backstop (anti-blow-up; the other half of the FP headroom contract).
    let s = length(v);
    if (s > params.max_speed) {
        v = v * (params.max_speed / s);
    }

    x = x + v * params.dt;

    // Particle-resolution box clamp (the grid BC bounds penetration only at node resolution).
    // Only the into-wall component is removed (separating, free slip). Collider SDF push-out is U5.
    if (x.x < params.box_min.x) { x.x = params.box_min.x; if (v.x < 0.0) { v.x = 0.0; } }
    if (x.y < params.box_min.y) { x.y = params.box_min.y; if (v.y < 0.0) { v.y = 0.0; } }
    if (x.z < params.box_min.z) { x.z = params.box_min.z; if (v.z < 0.0) { v.z = 0.0; } }
    if (x.x > params.box_max.x) { x.x = params.box_max.x; if (v.x > 0.0) { v.x = 0.0; } }
    if (x.y > params.box_max.y) { x.y = params.box_max.y; if (v.y > 0.0) { v.y = 0.0; } }
    if (x.z > params.box_max.z) { x.z = params.box_max.z; if (v.z > 0.0) { v.z = 0.0; } }

    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, vel[p].w);
    deform_disp[3u * p + 0u] = vec4<f32>(nd0, 0.0);
    deform_disp[3u * p + 1u] = vec4<f32>(nd1, 0.0);
    deform_disp[3u * p + 2u] = vec4<f32>(nd2, 0.0);
}

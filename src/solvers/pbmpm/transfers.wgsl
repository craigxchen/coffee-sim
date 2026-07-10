// PB-MPM APIC fixed-point transfers (U3): p2g → grid_update → g2p, one APIC transfer cycle. U4
// wraps `particle_update → grid_clear → p2g → grid_update → g2p` into an iteration_count loop per
// substep (step() in mod.rs) and runs `particle_integrate` once after it; the compliant density
// constraint + advection live in constraint.wgsl. The grid_clear pass lives in common.wgsl.
// Concatenated after common.wgsl into one shader module.
//
// Per-substep recurrence (the solver's OWN discrete form — tests/pbmpm_transfers.rs derives its
// round-trip reference from exactly this):
//   p2g scatters m and m·(v + D·d) at the OLD positions (fixed-point atomicAdd, order-
//     independent integer sums → bit-deterministic accumulation);
//   grid_update divides momentum/mass → velocity, applies gravity (v ← v + g·dt), and a simple
//     domain BC (zero the into-wall normal so water stays in the box — collider BC/restitution is
//     U5);
//   g2p gathers the new particle velocity and reconstructs the APIC affine matrix into the per-
//     particle D (KTD8: D = B·D⁻¹, the velocity-gradient state U4's tr(D) reads). Advection was
//     MOVED OUT of g2p into `particle_integrate` (constraint.wgsl) in U4: it runs ONCE per substep
//     after the iteration_count loop, on the converged velocity.
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
//
// grid_update runs once per ITERATION (the U4 iteration_count loop), but the substep's body force
// is g·dt TOTAL — so gravity is amortized as g·dt/iteration_count per iteration. The gathered
// particle velocity carries across the iterations (it is re-scattered each iteration), so over the
// iteration_count grid_update calls the velocity accumulates exactly g·dt of gravity, regardless of
// the iteration count (the constraint loop does not over-inject the body force).
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
        // Gravity on every mass-carrying node, amortized over the iteration loop (g·dt total).
        let iters = f32(max(params.iter_pad.x, 1u));
        v = v + params.gravity.xyz * (params.dt / iters);
    }

    // Domain BC: a node within one cell of a domain face zeroes the into-wall normal component so
    // the pool cannot leak out the box (the outer backstop the collider BC sits in front of).
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

    // Collider node BC (U5; mirrors twofield's coupling wall BC). A mass-carrying node within one
    // cell (`dist < h`) of a wall surface — inside the wall material (dist < 0) OR on the adjacent
    // free layer (0 ≤ dist < h, which is where the supporting column actually sits, since an SDF
    // surface like the cup floor rarely coincides with a node) — has its INTO-wall normal velocity
    // removed: `v -= min(0, dot(v, n))·n`. This is the node-resolution momentum BC that makes the
    // water collide with the vessel walls/floor; the gradient points OUT of the wall, so removing the
    // into-wall (negative dot) component keeps water off the wall on the free side. Restitution is
    // applied at particle resolution in `particle_integrate` (the node BC is the free-slip stop).
    if (params.iter_pad.y > 0u && mass > 0.0) {
        let hit = solid_union(wp, PHASE_WATER);
        if (hit.dist < h) {
            let vn = dot(v, hit.grad);
            if (vn < 0.0) {
                v = v - vn * hit.grad;
            }
        }
    }

    // Seam bed BC (U2, docs/plans/2026-07-09-002): a mass-carrying node inside the seam's
    // scattered bed (node solid fraction above params.seam.y) has its INTO-BED velocity
    // component removed, scaled by node saturation — a fully saturated bed (s ≥ params.seam.z,
    // the wet_sat_cutoff contract) accepts NO entry flux; below that the linear-in-s ramp is
    // the M0 placeholder (the true Darcy β(φ_s, s) is M1 scope). The into-bed normal is the
    // occupancy gradient (central difference, edge-guarded), falling back to straight down
    // where the gradient degenerates (deep interior). Runs every iteration so the constraint
    // loop sees the bed as a boundary, exactly like the collider BC above. The removed
    // momentum accumulates into `seam_reaction` (SEAM_IMPULSE_SCALE ledger — telemetry for
    // the seam's bed hook; the R2 third-law gate measures momentum deltas directly).
    if (params.seam.x > 0.0 && mass > 0.0) {
        let cell_vol = h * h * h;
        // All occupancy reads go through seam_occ_index (the lateral wall clamp — see
        // common.wgsl): wall node columns otherwise read a kernel-deficit φ_s and become
        // the chute water slides down.
        let nodev = vec3<i32>(i32(nx), i32(ny), i32(nz));
        let ni = seam_occ_index(nodev) * 4u;
        let occ = fp_decode(atomicLoad(&bed_occupancy[ni + 0u]));
        let phi_s = occ / cell_vol;
        if (phi_s > params.seam.y) {
            let vabs = fp_decode(atomicLoad(&bed_occupancy[ni + 1u]));
            let vcap = fp_decode(atomicLoad(&bed_occupancy[ni + 2u]));
            var s = 0.0;
            if (vcap > 0.0) {
                s = clamp(vabs / vcap, 0.0, 1.0);
            }
            // Percolation-speed cap (the corrected M0 placeholder; true Darcy K(φ_s, s) is
            // M1): into-bed velocity is capped at v_perc = vmax·(1−s). Saturated (s ≥
            // cutoff) ⇒ cap 0 — identical to the original full block, so every saturated
            // M0 gate is unchanged. Dry/partial ⇒ water ENTERS at percolation speed while
            // the EXCESS momentum is removed and ledgered — the stream decelerates at the
            // bed and the bed feels the impact, instead of the old β=s fraction ramp that
            // gave a dry bed zero resistance (measured: Center Pour fell straight through).
            var v_perc = params.seam.w * (1.0 - s);
            if (s >= params.seam.z) {
                v_perc = 0.0;
            }
            var grad = vec3<f32>(0.0);
            grad.x = fp_decode(atomicLoad(&bed_occupancy[seam_occ_index(nodev + vec3<i32>(1, 0, 0)) * 4u]))
                - fp_decode(atomicLoad(&bed_occupancy[seam_occ_index(nodev - vec3<i32>(1, 0, 0)) * 4u]));
            if (ny > 0u && ny + 1u < params.grid_dims.y) {
                grad.y = fp_decode(atomicLoad(&bed_occupancy[seam_occ_index(nodev + vec3<i32>(0, 1, 0)) * 4u]))
                    - fp_decode(atomicLoad(&bed_occupancy[seam_occ_index(nodev - vec3<i32>(0, 1, 0)) * 4u]));
            }
            grad.z = fp_decode(atomicLoad(&bed_occupancy[seam_occ_index(nodev + vec3<i32>(0, 0, 1)) * 4u]))
                - fp_decode(atomicLoad(&bed_occupancy[seam_occ_index(nodev - vec3<i32>(0, 0, 1)) * 4u]));
            var n_into = vec3<f32>(0.0, -1.0, 0.0);
            let glen = length(grad);
            // Degeneracy epsilon is PHYSICAL (5% of a cell volume), not machine-small: the
            // fixed-point occupancy lanes carry ~4e-6 truncation noise per frame, and a
            // machine epsilon here turns interior nodes' noise into junk normals — the BC
            // then blocks a random direction and water walks through (the measured interior
            // leak). A real surface gradient is ~φ·h³ ≈ 10⁻²; interior noise is 10⁻⁶.
            if (glen > 0.05 * cell_vol) {
                n_into = grad / glen;
            }
            let vn_bed = dot(v, n_into);
            if (vn_bed > v_perc) {
                let dv = (vn_bed - v_perc) * n_into;
                v = v - dv;
                let imp = dv * mass * SEAM_IMPULSE_SCALE;
                atomicAdd(&seam_reaction[n * 4u + 0u], i32(imp.x));
                atomicAdd(&seam_reaction[n * 4u + 1u], i32(imp.y));
                atomicAdd(&seam_reaction[n * 4u + 2u], i32(imp.z));
            }
        }
    }

    grid_vel[n] = vec4<f32>(v, mass);
}

// SPLASH (FLIP) snapshot decode: the PRE-FORCE pure-transfer grid velocity. Run ONCE per substep
// BEFORE the iteration loop, after a grid_clear + p2g that scattered the substep-start velocity.
// Mirrors grid_update's momentum/mass → velocity decode but writes `grid_vel_old` and applies NO
// gravity, NO domain BC, and NO collider BC — so `v_pic − gathered(grid_vel_old)` in
// particle_integrate is exactly the velocity CHANGE due to gravity + the pressure/constraint over
// the substep, which is what FLIP preserves.
@compute @workgroup_size(WG)
fn grid_decode_old(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let mc = atomicLoad(&grid_fp[n * 4u + 0u]);
    var v = vec3<f32>(0.0);
    let mass = fp_decode(mc);
    if (mass > 0.0) {
        let inv = 1.0 / f32(mc);
        v = vec3<f32>(
            f32(atomicLoad(&grid_fp[n * 4u + 1u])),
            f32(atomicLoad(&grid_fp[n * 4u + 2u])),
            f32(atomicLoad(&grid_fp[n * 4u + 3u]))
        ) * inv;
    }
    grid_vel_old[n] = vec4<f32>(v, mass);
}

// APIC gather: new particle velocity v = Σ w·v_i and the affine matrix B = Σ w·v_i·dᵀ, then
// D = B·D⁻¹ with D⁻¹ = (4/h²)·I for the quadratic B-spline (Jiang et al. APIC). D is written back
// to the per-particle deform_disp (KTD8: the velocity-gradient state U4's constraint reads via
// tr(D)). Velocity is written but NOT advected here — U4 moved advection into `particle_integrate`
// (constraint.wgsl), which runs ONCE per substep after the iteration loop, so the position only
// moves on the converged velocity. g2p now only gathers velocity + writes D.
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

    // No advection here — `particle_integrate` (constraint.wgsl) advects once per substep after the
    // iteration loop. Only the gathered velocity + the reconstructed affine D are written back.
    vel[p] = vec4<f32>(v, vel[p].w);
    deform_disp[3u * p + 0u] = vec4<f32>(nd0, 0.0);
    deform_disp[3u * p + 1u] = vec4<f32>(nd1, 0.0);
    deform_disp[3u * p + 2u] = vec4<f32>(nd2, 0.0);
}

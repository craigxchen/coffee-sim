// Dry granular bed (position-based; Macklin "Unified Particle Physics" 2014, with XPBD friction
// from Macklin et al. 2020). Grains solve direct position corrections — non-penetration + Coulomb
// friction + light cohesion — over their grain neighbors, written into `dp[i]` (Jacobi) and
// applied by the shared `apply_dp` (which also adds grain–boundary friction). Concatenated after
// `common.wgsl` + `water.wgsl`.
//
// Friction budget is μ · (accumulated normal impulse this frame), NOT μ · overlap. The
// non-penetration solve drives overlap → 0 at rest, so an overlap-based budget would vanish
// exactly when the pile needs to hold its slope (→ continuous creep). The accumulated normal
// correction is load-scaled (a deep grain is pushed harder, every iteration) and stays non-zero
// at static rest, so the pile holds a true repose without any freeze hack.
//
// To stay within the WebGPU 8-storage-buffer limit this kernel does NOT bind `status`: the
// convergence early-exit is enforced by `apply_dp`/`residual_reduce`, so a wasted projection
// after convergence is harmless (its dp is never applied).

@compute @workgroup_size(256)
fn bed_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (phase[i] != PHASE_GRAIN) { return; }
    let d = params.grain_diameter;
    let xi = pred[i].xyz;
    let prev_i = pos[i].xyz;

    var separation = vec3<f32>(0.0); // non-penetration (+ cohesion) push
    var tangential = vec3<f32>(0.0); // unclamped tangential-relative-motion removal
    var normal_mag = 0.0;            // total normal-correction magnitude this iteration
    var max_pen = 0.0;

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (j == i) { continue; }
                    if (phase[j] != PHASE_GRAIN) { continue; } // grain contacts only
                    let dvec = xi - pred[j].xyz;
                    var r = length(dvec);
                    if (r >= params.cohesion_range) { continue; }
                    var n: vec3<f32>;
                    if (r < 1e-6) {
                        n = vec3<f32>(0.0, 1.0, 0.0); // deterministic separation for coincident grains
                        r = 1e-6;
                    } else {
                        n = dvec / r;
                    }
                    if (r < d) {
                        // --- non-penetration (½/½ split between the two mobile grains) ---
                        let overlap = d - r;
                        max_pen = max(max_pen, overlap);
                        let push = overlap * 0.5;
                        separation = separation + n * push;
                        normal_mag = normal_mag + push;
                        // --- tangential relative displacement this frame (this grain's ½ share) ---
                        let rel = (xi - prev_i) - (pred[j].xyz - pos[j].xyz);
                        tangential = tangential - (rel - dot(rel, n) * n) * 0.5;
                    } else if (params.dry_cohesion > 0.0) {
                        // --- light dry cohesion just past contact (linear falloff) ---
                        let f = params.dry_cohesion * (1.0 - (r - d) / (params.cohesion_range - d));
                        separation = separation - n * f; // pull i toward j (direction −n)
                    }
                }
            }
        }
    }

    // Accumulate the normal impulse over the frame's iterations; the Coulomb budget is μ times it.
    // (Static if the desired tangential removal is within the cone, else slip to the limit.)
    let impulse = normal_impulse[i] + normal_mag;
    normal_impulse[i] = impulse;
    let limit = params.friction_mu * impulse;
    let tlen = length(tangential);
    var friction = tangential;
    if (tlen > limit) {
        friction = tangential * (limit / tlen);
    }

    dp[i] = vec4<f32>(separation + friction, 0.0);
    c_residual[i] = max_pen / d;
}

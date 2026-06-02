// Dry granular bed (position-based; Macklin "Unified Particle Physics" 2014). Grains solve
// direct position corrections — non-penetration + Coulomb friction + light cohesion — over their
// grain neighbors, written into `dp[i]` (Jacobi) and applied by the shared `apply_dp` (which also
// adds grain–boundary friction). Concatenated after `common.wgsl` + `water.wgsl`.
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
    let frozen_i = is_frozen(i);

    var sum = vec3<f32>(0.0);
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
                        // --- non-penetration ---
                        let overlap = d - r;
                        max_pen = max(max_pen, overlap);
                        if (!frozen_i) {
                            // Split the separation ½/½ with a mobile neighbor; take it all off a
                            // frozen neighbor (zero inverse mass — it doesn't move).
                            let w = select(0.5, 1.0, is_frozen(j));
                            sum = sum + n * (overlap * w);
                            // --- grain–grain Coulomb friction ---
                            // Remove the tangential relative displacement this frame, clamped to
                            // μ·overlap (static if within the cone, else slip to the limit).
                            let rel = (xi - prev_i) - (pred[j].xyz - pos[j].xyz);
                            let tang = rel - dot(rel, n) * n;
                            let tlen = length(tang);
                            if (tlen > 1e-8) {
                                let corr = min(tlen, params.friction_mu * overlap);
                                sum = sum - (tang / tlen) * (corr * w);
                            }
                        }
                    } else if (!frozen_i && params.dry_cohesion > 0.0) {
                        // --- light dry cohesion just past contact (linear falloff) ---
                        let f = params.dry_cohesion * (1.0 - (r - d) / (params.cohesion_range - d));
                        sum = sum - n * f; // pull i toward j (direction −n)
                    }
                }
            }
        }
    }

    // Frozen grains report their penetration (so finalize can thaw them) but do not move.
    if (frozen_i) {
        dp[i] = vec4<f32>(0.0);
    } else {
        dp[i] = vec4<f32>(sum, 0.0);
    }
    c_residual[i] = max_pen / d;
}

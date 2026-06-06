// Water incompressibility (Position Based Fluids — Macklin & Müller 2013): the constant-density
// constraint + λ, the s_corr artificial-pressure Δp, the convergence reduction, and XSPH
// viscosity. Concatenated after `common.wgsl`, which declares Params/Status/bindings + the SPH
// and grid helpers these kernels use.
//
// Every density sum is phase-guarded: a grain neighbor has no fluid density and must not enter
// the water constraint (and grain particles skip the water solve entirely). For a single-species
// water scene the guards are pass-throughs, so water behaviour is unchanged.

// Wall density compensation, precomputed into c_residual for compute_lambda to add (see the note
// there). Its own pass because compute_lambda already binds 8 storage buffers (the WebGPU limit)
// and cannot also bind `solids`; this pass binds `solids` but few others. Dispatched only when the
// scene has SDF solids, so AABB-only scenes never run it and c_residual stays the residual buffer.
@compute @workgroup_size(256)
fn compute_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    c_residual[i] = 0.0;
    if (phase[i] != PHASE_WATER) { return; }
    let f_i = pred[i].w;
    if (f_i <= params.pbf_eps) { return; }
    let hit = solid_union(pred[i].xyz, PHASE_WATER);
    if (hit.dist < params.h) {
        c_residual[i] = params.rest_density * f_i * boundary_psi(hit.dist, params.h);
    }
}

@compute @workgroup_size(256)
fn compute_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    if (phase[i] != PHASE_WATER) { return; } // grains have no density constraint
    let h = params.h;
    let m = params.particle_mass;
    let xi = pred[i].xyz;
    // Wetting: water carries a remaining-volume fraction f_w in pred.w. A near-empty particle is
    // skipped (inert), and every water contributes its f_w-weighted mass — so the density solve
    // matches the actual fluid at the wet front. f_w≡1 (no absorption) ⇒ identical to before.
    let f_i = pred[i].w;
    if (f_i <= params.pbf_eps) {
        lambda[i] = 0.0;
        c_residual[i] = 0.0;
        return;
    }

    var rho = m * f_i * w_poly6(0.0, h);
    var sum_g = vec3<f32>(0.0);
    var sum_g2 = 0.0;

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (j == i) { continue; }
                    if (phase[j] != PHASE_WATER) { continue; } // skip grain neighbors
                    let fj = pred[j].w;
                    if (fj <= params.pbf_eps) { continue; } // skip near-empty (absorbed) water
                    let d = xi - pred[j].xyz;
                    let r = length(d);
                    if (r >= h) { continue; }
                    rho = rho + m * fj * w_poly6(r, h);
                    let gj = m * fj * spiky_grad(d, h, params.spiky_r_min);
                    sum_g = sum_g + gj;
                    sum_g2 = sum_g2 + dot(gj, gj);
                }
            }
        }
    }

    // Boundary (wall) density compensation: a near-wall particle's kernel is partly cut off by the
    // solid, so the fluid-only sum above under-reads — strongest at the cup's floor/side corner,
    // where the deficit hid genuine over-packing and let it accumulate into squeeze-out eruptions.
    // The cut-off fraction ρ₀·f_i·ψ(d) was precomputed by `compute_boundary` into c_residual (a
    // scratch handoff — compute_lambda needs `solids`, but is already at the 8-storage-buffer limit,
    // so a separate pass that CAN bind `solids` stages the value here). Overwritten with the real
    // residual at the end of this kernel. Solid scenes only; AABB-only scenes never dispatch the
    // boundary pass, so c_residual is untouched and this term is skipped (byte-unchanged).
    if (params.num_solids > 0u) {
        rho = rho + c_residual[i];
    }

    // Pore-modulated rest density: where grains are present the water target is ρ₀·(1−α_s), so
    // pore water packs to fill only the pore fraction (α_s+α_f=1) instead of full ρ₀ — which would
    // over-fill and get expelled. α_s=0 (single-species water) → unmodulated ρ₀ (identical).
    let pore = max(1.0 - alpha_s[i], 0.05);
    let inv_rho0 = 1.0 / (params.rest_density * pore);
    let c = rho * inv_rho0 - 1.0;
    let denom = (dot(sum_g, sum_g) + sum_g2) * inv_rho0 * inv_rho0 + params.relaxation_eps;
    var lam = -c / denom;
    if (params.lambda_noncohesive != 0u && lam > 0.0) {
        lam = 0.0;
    }
    lambda[i] = lam;
    // Convergence is measured on COMPRESSION only: free-surface particles always have
    // c ≈ −1 (density deficit), which incompressibility neither can nor should fix, so
    // including them would prevent early-exit forever. max(0, c) = over-density violation.
    c_residual[i] = max(0.0, c);
}

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn residual_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (status.converged != 0u) { return; }
    let tid = lid.x;
    var local_max = 0.0;
    var i = tid;
    loop {
        if (i >= params.particle_count) { break; }
        local_max = max(local_max, c_residual[i]);
        i = i + 256u;
    }
    sdata[tid] = local_max;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        let gmax = sdata[0];
        status.residual_bits = bitcast<u32>(gmax);
        status.iters_done = status.iters_done + 1u;
        if (status.iters_done >= params.min_iters && gmax < params.residual_tolerance) {
            status.converged = 1u;
            status.effective_iters = status.iters_done;
        }
    }
}

@compute @workgroup_size(256)
fn compute_dp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    if (phase[i] != PHASE_WATER) { return; } // grains are projected by bed_project
    let h = params.h;
    let m = params.particle_mass;
    let xi = pred[i].xyz;
    let lam_i = lambda[i];
    // Skip near-empty (absorbed) water; weight neighbor contributions by their f_w (pred.w) to match
    // compute_lambda. f_w≡1 (no absorption) ⇒ identical to before.
    if (pred[i].w <= params.pbf_eps) {
        dp[i] = vec4<f32>(0.0);
        return;
    }

    var sum = vec3<f32>(0.0);
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (j == i) { continue; }
                    if (phase[j] != PHASE_WATER) { continue; } // skip grain neighbors
                    let fj = pred[j].w;
                    if (fj <= params.pbf_eps) { continue; } // skip near-empty (absorbed) water
                    let d = xi - pred[j].xyz;
                    let r = length(d);
                    if (r >= h) { continue; }
                    let wr = w_poly6(r, h);
                    let ratio = wr / params.s_corr_wq;
                    let scorr = -params.s_corr_k * pow(ratio, params.s_corr_n);
                    let gj = m * fj * spiky_grad(d, h, params.spiky_r_min);
                    sum = sum + (lam_i + lambda[j] + scorr) * gj;
                }
            }
        }
    }
    let pore = max(1.0 - alpha_s[i], 0.05);
    let inv_rho0 = 1.0 / (params.rest_density * pore);
    dp[i] = vec4<f32>(sum * inv_rho0, 0.0);
}

@compute @workgroup_size(256)
fn xsph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    // Viscosity is a fluid term; grains pass through unchanged (so the vel_smoothed→vel copy in a
    // mixed scene doesn't clobber grain velocities).
    if (phase[i] != PHASE_WATER) {
        vel_smoothed[i] = vel[i];
        return;
    }
    let h = params.h;
    let xi = pos[i].xyz;
    let vi = vel[i].xyz;
    var dv = vec3<f32>(0.0);
    // Per-particle volume V_w = m/ρ₀ normalizes the kernel sum (standard XSPH: Δv = c·Σ (m_j/ρ_j)
    // (v_j−v_i) W). Without it, Σ W_poly6 ∝ 1/h³ grows as the resolution refines, so a fixed xsph_c
    // overshoots (amplifies velocity instead of smoothing it) at fine spacing — a resolution-dependent
    // eruption. With it, Σ V_w·W ≈ 1 (dimensionless), so xsph_c behaves consistently at any spacing.
    let v_w = params.particle_mass / params.rest_density;

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let lo = cell_start[cid];
                let hi = cell_start[cid + 1u];
                for (var s = lo; s < hi; s = s + 1u) {
                    let j = sorted_indices[s];
                    if (j == i) { continue; }
                    let d = xi - pos[j].xyz;
                    let r = length(d);
                    if (r >= h) { continue; }
                    dv = dv + (vel[j].xyz - vi) * (w_poly6(r, h) * v_w);
                }
            }
        }
    }
    vel_smoothed[i] = vec4<f32>(vi + params.xsph_c * dv, 0.0);
}

// PB-MPM compliant density constraint (U4 — the bounce mechanism). Concatenated after
// common.wgsl + transfers.wgsl into one shader module. Two passes:
//
//   particle_update : per live liquid particle, read the per-particle deformation displacement D
//     (KTD8: a float mat3x3, the velocity-gradient state, zeroed at the start of each substep's
//     iteration loop in step() and accumulated across the iterations), apply EA SEED's compliant
//     LIQUID constraint, write D back. This runs BEFORE p2g each iteration so the corrected D
//     propagates through the scatter (p2g uses D as the affine matrix for `vaff = v + D·d`).
//
//   particle_integrate : per live particle, advect the position by the final gathered velocity·dt.
//     Advection was MOVED OUT of g2p into here so it happens ONCE per substep after the iteration
//     loop converges (g2p now only gathers velocity + writes D, no advect).
//
// EA SEED liquid constraint (KTD1; Lewin, "A Position Based Material Point Method", SIGGRAPH 2024;
// verified against electronicarts/pbmpm siggraph2024). Applied in EA SEED's EXACT order:
//   1. Viscosity FIRST: deviatoric = −(D + Dᵀ); D += liquid_viscosity·0.5·deviatoric  (the NEGATIVE
//      SYMMETRIC part — NOT the trace-removed deviatoric).
//   2. Volume SECOND:  alpha = 0.5·(1/liquidDensity − tr(D) − 1); D += liquid_relaxation·alpha·I
//      where liquidDensity is the PER-PARTICLE accumulated value (not a constant params lane).
// `liquid_relaxation ∈ (0,1]` controls compliance: 1 is the stiffest single-iteration push toward
// rest, smaller is softer/more damped. The per-particle liquidDensity (EA SEED's running product of
// det(F), updated each substep in particle_integrate) is the VOLUME MEMORY: a static over-dense blob
// has tr(D) ≈ 0 but liquidDensity > 1, so alpha < 0 ⇒ an expanding correction recovers it toward rest
// (a CONSTANT density=1 gave alpha ≈ 0 here — the collapse bug). That restoring push, rebuilt and
// re-gathered across the iteration loop, is the stiff incompressibility that makes the water BOUNCE.
// The grid rebuilds each iteration, so the per-particle correction propagates spatially through the
// scatter → grid → gather cycle.
//
// D is the per-particle velocity-gradient matrix (deform_disp rows; p2g scatters `vaff = v + D·d`).
// At rest with liquidDensity = 1.0 the target is tr(D) = 1/liquidDensity − 1 = 0 (zero divergence),
// so a settled uniform pool reads alpha ≈ 0 and the correction vanishes (a bounded fixed point); an
// impact spikes a converging flow (tr(D) < 0) and the constraint pushes back. Sustained compression
// (tr(D) < 0 over several substeps) drives liquidDensity above 1 in particle_integrate, so even after
// the flow stalls the volume term keeps expanding the over-dense pool back toward rest.

// Zero the per-particle deformation displacement D for the live water range. Run ONCE at the start
// of each substep's iteration loop (KTD8: D is zeroed per substep and accumulated across the
// iterations, while F carries across substeps). On a GPU pass (not a host write) so the per-frame
// cost stays on-device at 200k.
@compute @workgroup_size(WG)
fn deform_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    deform_disp[3u * p + 0u] = vec4<f32>(0.0);
    deform_disp[3u * p + 1u] = vec4<f32>(0.0);
    deform_disp[3u * p + 2u] = vec4<f32>(0.0);
}

@compute @workgroup_size(WG)
fn particle_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    // Read the accumulated deformation displacement D (rows in deform_disp; .xyz = the matrix row).
    var d0 = deform_disp[3u * p + 0u].xyz;
    var d1 = deform_disp[3u * p + 1u].xyz;
    var d2 = deform_disp[3u * p + 2u].xyz;

    // Per-particle accumulated liquid density (EA SEED `particle.liquidDensity` = the running
    // product of det(F) ≈ the volume Jacobian; updated each substep in particle_integrate). Stored
    // in the repurposed deform_grad lane [3p+0].x (see common.wgsl). A value > 1 means the particle
    // has accumulated COMPRESSION (it is over-dense); the volume term must then expand it back toward
    // rest even when the INSTANTANEOUS flow is divergence-free (tr(D) ≈ 0) — without this memory a
    // static over-dense blob reads alpha ≈ 0 and never recovers (the collapse bug U6 fixes).
    let liquid_density = deform_grad[3u * p + 0u].x;

    // EA SEED order: VISCOSITY FIRST, then VOLUME.
    // 1. Viscosity: nudge D by the NEGATIVE symmetric part. deviatoric = −(D + Dᵀ);
    //    D += liquid_viscosity·0.5·deviatoric. (This is EA SEED's exact form — the symmetric part,
    //    not the trace-removed deviatoric.)
    let visc = params.liquid_viscosity * 0.5;
    let s00 = -2.0 * d0.x;
    let s11 = -2.0 * d1.y;
    let s22 = -2.0 * d2.z;
    let s01 = -(d0.y + d1.x);
    let s02 = -(d0.z + d2.x);
    let s12 = -(d1.z + d2.y);
    d0.x = d0.x + visc * s00;
    d0.y = d0.y + visc * s01;
    d0.z = d0.z + visc * s02;
    d1.x = d1.x + visc * s01;
    d1.y = d1.y + visc * s11;
    d1.z = d1.z + visc * s12;
    d2.x = d2.x + visc * s02;
    d2.y = d2.y + visc * s12;
    d2.z = d2.z + visc * s22;

    // 2. Volume: compliant isotropic push toward the rest volume, using the PER-PARTICLE accumulated
    //    density. alpha = 0.5·(1/liquid_density − tr(D) − 1); D += liquid_relaxation·alpha·I.
    let tr = d0.x + d1.y + d2.z;
    let alpha = 0.5 * (1.0 / liquid_density - tr - 1.0);
    let vol = params.liquid_relaxation * alpha;
    d0.x = d0.x + vol;
    d1.y = d1.y + vol;
    d2.z = d2.z + vol;

    deform_disp[3u * p + 0u] = vec4<f32>(d0, 0.0);
    deform_disp[3u * p + 1u] = vec4<f32>(d1, 0.0);
    deform_disp[3u * p + 2u] = vec4<f32>(d2, 0.0);
}

// Advect each live particle by its final gathered velocity (semi-implicit Euler), then apply the
// particle-resolution boundary backstops: the box clamp AND the U5 collider SDF push-out +
// restitution (the grid node BC bounds penetration only at node resolution h, so a particle between
// a constrained node and a live one creeps through). Runs ONCE per substep after the iteration loop.
@compute @workgroup_size(WG)
fn particle_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    var x = pos[p].xyz;

    // SPLASH (FLIP) blend, applied ONCE per substep on the converged output velocity. `vel[p]` is
    // the pure-PIC result of the constraint iteration loop (g2p stays pure-PIC so the
    // incompressibility solve is clean). Gather the PRE-FORCE pure-transfer grid velocity over the
    // particle's 3×3×3 stencil → `gathered_old`; then v_flip = vel_prev + (v_pic − gathered_old)
    // adds the substep's grid-velocity CHANGE (gravity + pressure/constraint) onto the substep-start
    // velocity, preserving the impact-generated crown velocity APIC would smooth away. flip_fraction
    // = 0 ⇒ v = v_pic (the byte-identical pure-APIC off-switch).
    let v_pic = vel[p].xyz;
    let h = params.grid_origin.w;
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    let w = bspline_w(fx);
    var gathered_old = vec3<f32>(0.0);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                gathered_old = gathered_old + wijk * grid_vel_old[node_index(node)].xyz;
            }
        }
    }
    let flip_fraction = bitcast<f32>(params.iter_pad.w);
    let v_flip = vel_prev[p].xyz + (v_pic - gathered_old);
    var v = mix(v_pic, v_flip, flip_fraction);
    // Velocity cap (anti-blow-up): FLIP can amplify the kept change, so clamp the blended result.
    let s = length(v);
    if (s > params.max_speed) {
        v = v * (params.max_speed / s);
    }

    x = x + v * params.dt;

    // Box clamp (outer backstop): only the into-wall component is removed (separating, free slip).
    if (x.x < params.box_min.x) { x.x = params.box_min.x; if (v.x < 0.0) { v.x = 0.0; } }
    if (x.y < params.box_min.y) { x.y = params.box_min.y; if (v.y < 0.0) { v.y = 0.0; } }
    if (x.z < params.box_min.z) { x.z = params.box_min.z; if (v.z < 0.0) { v.z = 0.0; } }
    if (x.x > params.box_max.x) { x.x = params.box_max.x; if (v.x > 0.0) { v.x = 0.0; } }
    if (x.y > params.box_max.y) { x.y = params.box_max.y; if (v.y > 0.0) { v.y = 0.0; } }
    if (x.z > params.box_max.z) { x.z = params.box_max.z; if (v.z > 0.0) { v.z = 0.0; } }

    // Collider SDF push-out + restitution (U5; mirrors twofield's particle BC). The cavity sign
    // convention is interior-POSITIVE: `hit.dist < 0` means the particle is INSIDE the wall material
    // (it penetrated the cup wall/floor). Push it back ALONG the gradient (which points into the
    // cavity) to the surface — `x += (−dist)·grad`, where −dist > 0 — so water ends up in the cup
    // CAVITY, never expelled from it. Then reflect the into-solid normal velocity by the restitution:
    // `v_n_out = −restitution·v_n_in` (restitution 0 → free-slip stop, the constraint-only arm; >0 →
    // a rebound). The tangential velocity is untouched (free slip).
    if (params.iter_pad.y > 0u) {
        let hit = solid_union(x, PHASE_WATER);
        if (hit.dist < 0.0) {
            x = x + (-hit.dist) * hit.grad; // project back to the cavity surface
            let vn = dot(v, hit.grad);      // < 0 when moving INTO the solid (against the inward grad)
            if (vn < 0.0) {
                let restitution = bitcast<f32>(params.iter_pad.z);
                // Remove the into-solid normal (−vn·grad), then add the reflected rebound
                // (−restitution·vn·grad): net `v − (1 + restitution)·vn·grad`.
                v = v - (1.0 + restitution) * vn * hit.grad;
            }
        }
    }

    pos[p] = vec4<f32>(x, pos[p].w);
    vel[p] = vec4<f32>(v, vel[p].w);

    // EA SEED `particleIntegrate` liquid accumulation (runs ONCE per substep, on the converged D):
    // liquidDensity *= (tr(D) + 1.0); clamped to ≥ 0.1. tr(D)+1 is the per-substep volume Jacobian
    // (a converging/compressing flow has tr(D) < 0 ⇒ factor < 1 ⇒ density rises toward over-dense;
    // an expanding flow relaxes it back). The running product is the volume memory the constraint's
    // alpha term reads next substep — the fix for the density-blind collapse. Read the FINAL
    // deform_disp (the converged D from the last iteration of this substep).
    let tr_final = deform_disp[3u * p + 0u].x + deform_disp[3u * p + 1u].y + deform_disp[3u * p + 2u].z;
    var liquid_density = deform_grad[3u * p + 0u].x;
    liquid_density = liquid_density * (tr_final + 1.0);
    liquid_density = max(liquid_density, 0.1);
    deform_grad[3u * p + 0u].x = liquid_density;
}

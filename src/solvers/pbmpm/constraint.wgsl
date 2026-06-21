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
// EA SEED liquid constraint (KTD1; Lewin, "A Position Based Material Point Method", SIGGRAPH 2024):
//   alpha = 0.5 * (1/liquid_density − tr(D) − 1)          // signed volume error vs rest
//   D    += liquid_relaxation * alpha * Identity          // compliant VOLUME correction
//   D    += liquid_viscosity  * 0.5 * deviatoric(D)       // viscous SHEAR term
// where deviatoric(D) = D − (tr(D)/3)·I. `liquid_relaxation ∈ (0,1]` controls compliance: 1 is the
// stiffest single-iteration push toward rest, smaller is softer/more damped. Under compression
// (tr(D) < 0, density too high) alpha > 0 ⇒ a positive (expanding) volume correction restores the
// pool toward rest — that restoring push, rebuilt and re-gathered across the iteration loop, is the
// stiff incompressibility that makes the water BOUNCE. The grid rebuilds each iteration, so the
// per-particle correction propagates spatially through the scatter → grid → gather cycle.
//
// D is the per-particle velocity-gradient matrix (deform_disp rows; p2g scatters `vaff = v + D·d`).
// For the default liquid_density = 1.0 the rest target is tr(D) = 1/liquid_density − 1 = 0 (zero
// divergence), so a settled uniform pool reads alpha ≈ 0 and the correction vanishes (a bounded
// fixed point); an impact spikes a converging flow (tr(D) < 0) and the constraint pushes back.

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

    let tr = d0.x + d1.y + d2.z;

    // Compliant VOLUME correction: a relaxation-scaled isotropic push toward the rest volume.
    let alpha = 0.5 * (1.0 / params.liquid_density - tr - 1.0);
    let vol = params.liquid_relaxation * alpha;
    d0.x = d0.x + vol;
    d1.y = d1.y + vol;
    d2.z = d2.z + vol;

    // Viscous SHEAR correction: nudge D toward its deviatoric part (removes shear off-diagonals /
    // the anisotropic trace split). deviatoric(D) = D − (tr(D)/3)·I; uses the post-volume trace.
    let tr2 = d0.x + d1.y + d2.z;
    let third = tr2 / 3.0;
    let shear = params.liquid_viscosity * 0.5;
    // dev(D) diagonal = d_ii − tr/3 ; off-diagonal = d_ij unchanged.
    d0.x = d0.x + shear * (d0.x - third);
    d0.y = d0.y + shear * d0.y;
    d0.z = d0.z + shear * d0.z;
    d1.x = d1.x + shear * d1.x;
    d1.y = d1.y + shear * (d1.y - third);
    d1.z = d1.z + shear * d1.z;
    d2.x = d2.x + shear * d2.x;
    d2.y = d2.y + shear * d2.y;
    d2.z = d2.z + shear * (d2.z - third);

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
    var v = vel[p].xyz;
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
}

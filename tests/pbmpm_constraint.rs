//! PB-MPM U4 compliant density constraint — pure-Rust formula mirror (no GPU), in the spirit of
//! `tests/twofield_dissipation.rs`. It re-derives the `particle_update` correction on the CPU and
//! pins the three load-bearing properties (KTD3 — the real bounce verification is the webapp):
//!   (a) under COMPRESSION (tr(D) < 0, density too high) alpha > 0 ⇒ a positive (expanding) volume
//!       correction that restores the trace toward rest (the restoring push that bounces);
//!   (b) at REST (tr(D) = 0, density = 1) the correction is ~0 (a neutral fixed point);
//!   (c) the iterate is STABLE — repeatedly applying the correction to a compressed state converges
//!       to a bounded fixed point (no blow-up) for a representative liquid_relaxation.
//!
//! IMPORTANT — this Rust mirror MUST stay byte-for-byte equivalent to the WGSL `particle_update` in
//! `src/solvers/pbmpm/constraint.wgsl`. If the shader formula changes, change this mirror too.

/// A 3×3 matrix as row vectors, mirroring the WGSL `deform_disp` rows (`d0`, `d1`, `d2`).
type Mat3 = [[f32; 3]; 3];

fn trace(d: &Mat3) -> f32 {
    d[0][0] + d[1][1] + d[2][2]
}

/// The exact `particle_update` correction from `constraint.wgsl`: the compliant VOLUME term
/// (`liquid_relaxation · alpha · I`, `alpha = 0.5·(1/liquid_density − tr(D) − 1)`) followed by the
/// viscous SHEAR term (`liquid_viscosity · 0.5 · deviatoric(D)`, using the post-volume trace).
fn particle_update(
    d: &Mat3,
    liquid_density: f32,
    liquid_relaxation: f32,
    liquid_viscosity: f32,
) -> Mat3 {
    let mut o = *d;

    let tr = trace(&o);
    let alpha = 0.5 * (1.0 / liquid_density - tr - 1.0);
    let vol = liquid_relaxation * alpha;
    o[0][0] += vol;
    o[1][1] += vol;
    o[2][2] += vol;

    let tr2 = trace(&o);
    let third = tr2 / 3.0;
    let shear = liquid_viscosity * 0.5;
    for (r, row) in o.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            let dev = if r == c { *x - third } else { *x };
            *x += shear * dev;
        }
    }
    o
}

const DENSITY: f32 = 1.0; // rest target tr(D) = 1/density − 1 = 0
const RELAX: f32 = 0.5; // representative compliant relaxation
const VISC: f32 = 0.01; // small viscous shear damp (the default)

/// (a) Under compression (a converging flow, tr(D) < 0) the volume correction pushes the trace
/// back UP toward rest — alpha > 0 and the corrected trace is strictly closer to 0.
#[test]
fn compression_pushes_back_toward_rest() {
    // Isotropic compression: tr(D) = −0.3 (each diagonal −0.1).
    let d: Mat3 = [[-0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]];
    let tr0 = trace(&d);
    assert!(tr0 < 0.0, "test setup is a compressive state");

    let alpha = 0.5 * (1.0 / DENSITY - tr0 - 1.0);
    assert!(
        alpha > 0.0,
        "compression ⇒ alpha > 0 (expanding correction), got {alpha}"
    );

    let o = particle_update(&d, DENSITY, RELAX, VISC);
    let tr1 = trace(&o);
    // The correction moves the trace toward rest (0) without overshooting past it.
    assert!(
        tr1 > tr0 && tr1 <= 0.0,
        "volume correction restores toward rest: tr {tr0} → {tr1} (target 0)"
    );
    // It does not overshoot to expansion at relaxation ≤ 1 (a bounded, stable single step).
    assert!(tr1 >= tr0, "no backward step");
}

/// (b) At rest (tr(D) = 0, density = 1) the correction is ~0 — a neutral fixed point (no spontaneous
/// expansion or compression of a settled uniform pool).
#[test]
fn rest_state_is_a_neutral_fixed_point() {
    let d: Mat3 = [[0.0; 3]; 3];
    let o = particle_update(&d, DENSITY, RELAX, VISC);
    assert!(
        o.iter().flatten().all(|x| x.abs() < 1e-6),
        "rest state must stay ~0, got {o:?}"
    );
    assert!(trace(&o).abs() < 1e-6, "rest trace stays ~0");
}

/// (c) The iterate is a STABLE, BOUNDED fixed point: repeatedly applying the correction to a
/// compressed state converges monotonically toward rest and never diverges, for the representative
/// relaxation. Mirrors the GPU iteration loop's per-particle convergence (the grid coupling is
/// separate; this isolates the per-particle constraint's own contraction).
#[test]
fn iterate_converges_to_a_bounded_fixed_point() {
    let mut d: Mat3 = [[-0.4, 0.05, 0.0], [0.05, -0.4, 0.0], [0.0, 0.0, -0.4]];
    let mut prev_abs_tr = trace(&d).abs();
    for it in 0..32 {
        d = particle_update(&d, DENSITY, RELAX, VISC);
        let abs_tr = trace(&d).abs();
        assert!(
            d.iter().all(|row| row.iter().all(|x| x.is_finite())),
            "iterate {it} blew up (non-finite)"
        );
        // |tr(D)| never grows (monotone contraction toward the rest trace 0).
        assert!(
            abs_tr <= prev_abs_tr + 1e-6,
            "iterate {it} diverged: |tr| {prev_abs_tr} → {abs_tr}"
        );
        prev_abs_tr = abs_tr;
    }
    // Converged near rest (trace ≈ 0) and bounded.
    assert!(
        trace(&d).abs() < 1e-3,
        "iterate did not converge to rest: tr {} after 32 steps",
        trace(&d)
    );
    assert!(
        d.iter().all(|row| row.iter().all(|x| x.abs() < 1.0)),
        "fixed point is bounded"
    );
}

/// The disabled arm (relaxation = 0, viscosity = 0) — the pure-transfer round-trip configuration —
/// leaves D bit-unchanged for ANY state (the iteration loop becomes a pure APIC transfer).
#[test]
fn disabled_relaxation_leaves_d_unchanged() {
    let d: Mat3 = [[-0.2, 0.1, 0.3], [0.4, -0.5, 0.6], [0.7, 0.8, -0.9]];
    let o = particle_update(&d, DENSITY, 0.0, 0.0);
    assert_eq!(
        o, d,
        "relaxation=0, viscosity=0 ⇒ D unchanged (pure transfer)"
    );
}

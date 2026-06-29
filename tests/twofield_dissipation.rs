//! Twofield U2 gate: the surface-weighted velocity-averaging dissipation curve in `g2p_water`
//! (`src/solvers/twofield/transfers.wgsl`).
//!
//! This is a PURE-RUST unit test (no GPU): it re-derives the merge-discriminator strength `c`
//! on the CPU from constructed (`v_own`, `v_grid`, `m_local`, `grad_m`, `div`) states and asserts
//! the R5 mechanism — merging-drip / compressive / dense states read HIGH `c` (dissipate → calm
//! pool, NO drip-stir), while separating/coherent-free sparse states read LOW `c` (= `c_surface`,
//! momentum preserved → splash). The byte-identical-disabled property (the full GPU suite under
//! the default flip) is gated by `tests/twofield_settled.rs` + `tests/twofield_water.rs`.
//!
//! IMPORTANT — this Rust mirror MUST stay byte-for-byte equivalent to the WGSL formula in
//! `g2p_water` (the consts below mirror the `transfers.wgsl` module-scope consts; the curve
//! mirrors the U2 block). If the shader curve changes, change this mirror too.

// --- mirror of the transfers.wgsl module-scope consts (U2) ---
const APPROACH_SCALE: f32 = 0.5;
const DENSITY_W: f32 = 0.25;
const AFFINE_DAMP_K: f32 = 0.75;
const GRAD_M_EPS: f32 = 1.0e-6;

/// WGSL `smoothstep(edge0, edge1, x)` (the std GLSL/WGSL Hermite form). Rust std has no
/// equivalent, so it is hand-rolled here to match the shader exactly.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The exact `g2p_water` U2 strength curve, given the gathered per-particle state and the
/// `flip` lanes `(c_surface, density_gate, div_scale, _)`. `particle_mass` matches the shader's
/// `params.particle_mass`. Returns `(c, affine_keep)`.
fn dissipation_c(
    v_own: [f32; 3],
    v_grid: [f32; 3],
    grad_m: [f32; 3],
    m_local: f32,
    div: f32,
    particle_mass: f32,
    c_surface: f32,
    density_gate: f32,
    div_scale: f32,
) -> (f32, f32) {
    let len = (grad_m[0] * grad_m[0] + grad_m[1] * grad_m[1] + grad_m[2] * grad_m[2]).sqrt();
    let inv = 1.0 / len.max(GRAD_M_EPS);
    let n = [grad_m[0] * inv, grad_m[1] * inv, grad_m[2] * inv];
    let rel = [
        v_own[0] - v_grid[0],
        v_own[1] - v_grid[1],
        v_own[2] - v_grid[2],
    ];
    let approach = (rel[0] * n[0] + rel[1] * n[1] + rel[2] * n[2]).max(0.0);

    let merge = if div_scale > 0.0 {
        (-div * div_scale)
            .max(approach * APPROACH_SCALE)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dens_c = smoothstep(
        density_gate,
        density_gate + DENSITY_W,
        m_local / (8.0 * particle_mass),
    );
    let c = c_surface.max(merge).max(dens_c).clamp(c_surface, 1.0);
    let affine_keep = 1.0 - AFFINE_DAMP_K * (1.0 - c);
    (c, affine_keep)
}

const MASS: f32 = 1.0; // Materials::default().particle_mass

// A representative ENABLED tuning (matches the spirit of the U4 defaults-to-be; the exact
// production values are set in U4). c_surface low = preserve setpoint; merge discriminator ON.
const C_SURFACE: f32 = 0.1;
const DENSITY_GATE: f32 = 0.5;
const DIV_SCALE: f32 = 1.0;

/// (a) A particle whose `v_own` points UP the mass gradient into a dense, slow neighborhood
/// (merging) AND with compressive `div < 0` ⇒ `c → high` (≈1). Also checks each sub-signal
/// alone fires.
#[test]
fn merging_into_dense_slow_neighborhood_reads_high_c() {
    // grad_m points +y (density increases upward = into the pool below is -y...); make the
    // particle move toward higher density: v_own − v_grid aligned with grad_m ⇒ approach>0.
    let grad_m = [0.0, 5.0, 0.0];
    let v_own = [0.0, 4.0, 0.0]; // 4 units/frame up the gradient
    let v_grid = [0.0, 0.0, 0.0]; // slow neighborhood
    let m_local = 4.0 * MASS; // sparse-to-moderate (below the dense bulk gate of 8·mass)
    let div = 0.0; // post-projection, weak: the approach term must carry it

    let (c, keep) = dissipation_c(
        v_own,
        v_grid,
        grad_m,
        m_local,
        div,
        MASS,
        C_SURFACE,
        DENSITY_GATE,
        DIV_SCALE,
    );
    // approach = 4, ·APPROACH_SCALE(0.5) = 2 → clamps to 1 ⇒ merge=1 ⇒ c=1.
    assert!(
        c > 0.95,
        "merging-into-density (approach-only, div=0) must read high c, got {c}"
    );
    // C is fully shed at c=1 → affine_keep returns to 1 (bulk-unchanged behavior at full merge).
    assert!((keep - 1.0).abs() < 1e-6, "c=1 ⇒ affine_keep=1, got {keep}");

    // Compressive div alone (no approach) also fires.
    let (c_div, _) = dissipation_c(
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        2.0 * MASS,
        -0.8,
        MASS,
        C_SURFACE,
        DENSITY_GATE,
        DIV_SCALE,
    );
    assert!(
        c_div > 0.7,
        "compressive div<0 (−0.8·1.0=0.8) must lift c well above c_surface, got {c_div}"
    );
}

/// (b) A free / coherent particle (`v_own ≈ v_grid`, sparse, `grad_m ≈ 0`, no compression) ⇒
/// `c → c_surface` (low, momentum preserved → splash). Affine C is shed by ~AFFINE_DAMP_K.
#[test]
fn free_coherent_sparse_particle_preserves_c_surface() {
    let v_own = [3.0, -2.0, 1.0];
    let v_grid = [3.0, -2.0, 1.0]; // coherent: no relative approach
    let grad_m = [0.0, 0.0, 0.0]; // sparse: no gradient
    let m_local = 0.5 * MASS; // sparse: below density_gate·8·mass = 4
    let div = 0.0;

    let (c, keep) = dissipation_c(
        v_own,
        v_grid,
        grad_m,
        m_local,
        div,
        MASS,
        C_SURFACE,
        DENSITY_GATE,
        DIV_SCALE,
    );
    assert!(
        (c - C_SURFACE).abs() < 1e-6,
        "free/coherent/sparse must preserve c_surface={C_SURFACE}, got {c}"
    );
    let expect_keep = 1.0 - AFFINE_DAMP_K * (1.0 - C_SURFACE);
    assert!(
        (keep - expect_keep).abs() < 1e-6,
        "strong preserve sheds C by ~K: keep={expect_keep} expected, got {keep}"
    );
}

/// (c) The DENSITY-ONLY negative control (`div_scale ≤ 0` ⇒ merge discriminator OFF): the SAME
/// merging-drip state from (a) — fast approach into a slow pool but locally SPARSE — now reads
/// `c → c_surface` (preserved), reproducing main's drip-stir. This asserts the control is
/// meaningfully DIFFERENT from the enabled guard (the merge term is what prevents the stir).
#[test]
fn density_only_control_lets_a_sparse_drip_through() {
    let grad_m = [0.0, 5.0, 0.0];
    let v_own = [0.0, 4.0, 0.0]; // same fast approach as (a)
    let v_grid = [0.0, 0.0, 0.0];
    let m_local = 4.0 * MASS; // sparse drip: m_local/(8·mass)=0.5, the density gate's lower edge
    let div = 0.0;

    // Control: div_scale = 0 ⇒ merge=0 ⇒ c = max(c_surface, dens_c).
    let (c_ctrl, _) = dissipation_c(
        v_own,
        v_grid,
        grad_m,
        m_local,
        div,
        MASS,
        C_SURFACE,
        DENSITY_GATE,
        /*div_scale*/ 0.0,
    );
    // dens_c = smoothstep(0.5, 0.75, 0.5) = 0 at the lower edge ⇒ c = c_surface (preserved!).
    assert!(
        (c_ctrl - C_SURFACE).abs() < 1e-6,
        "density-only control: a sparse drip is PRESERVED (c→c_surface), reproducing main's \
         stir; got {c_ctrl}"
    );

    // Same state WITH the merge discriminator on ((a)) dissipates it — proving the difference.
    let (c_enabled, _) = dissipation_c(
        v_own,
        v_grid,
        grad_m,
        m_local,
        div,
        MASS,
        C_SURFACE,
        DENSITY_GATE,
        DIV_SCALE,
    );
    assert!(
        c_enabled > 0.95 && c_enabled - c_ctrl > 0.8,
        "enabled merge guard must dissipate the SAME drip the control preserves: \
         enabled={c_enabled} vs control={c_ctrl}"
    );
}

/// The disabled sentinel (`c_surface = 1.0`) collapses the curve to pure-PIC for ANY state:
/// `c = 1` (⇒ `v = v_grid`) and `affine_keep = 1` (⇒ C unchanged). This mirrors the WGSL
/// disabled path that the GPU suites gate as byte-identical.
#[test]
fn disabled_sentinel_collapses_to_pure_pic() {
    // A state that would otherwise drive merge AND dens_c high — must still read c=1, keep=1.
    let states = [
        (
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            9.0 * MASS,
            -0.9,
        ),
        (
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            0.1 * MASS,
            0.0,
        ),
    ];
    for (v_own, v_grid, grad_m, m_local, div) in states {
        let (c, keep) = dissipation_c(
            v_own,
            v_grid,
            grad_m,
            m_local,
            div,
            MASS,
            /*c_surface*/ 1.0,
            DENSITY_GATE,
            DIV_SCALE,
        );
        assert!((c - 1.0).abs() < 1e-7, "disabled ⇒ c=1 (v=v_grid), got {c}");
        assert!(
            (keep - 1.0).abs() < 1e-7,
            "disabled ⇒ affine_keep=1 (C unchanged), got {keep}"
        );
    }
}

//! U5 CPU twin of `plasticity.wgsl` — the 3×3 SVD and the Klar et al. 2016 Drucker-Prager
//! return map in log-strain space, plus the compaction-cap (tamping-memory) extension
//! (plan 2026-06-09-001 U5 / KTD-5).
//!
//! CHARACTERIZATION-FIRST: this twin and its gates (`tests/twofield_bed.rs`) precede the WGSL
//! port; the GPU kernel is then pinned against this module on manufactured deformation
//! gradients. Every function here mirrors the WGSL arithmetic in f32 (same sweep counts, same
//! clamps, same branch structure), so twin-vs-GPU comparisons are tight.
//!
//! ============================== CONSTITUTIVE MODEL (KTD-5) ===================================
//! Per solid particle: elastic deformation gradient F (the plastic part is discarded by the
//! return map — sand keeps no memory of plastic *shape*; the compaction memory lives in the
//! per-particle (p_c, compaction) state). One SVD per particle per step: F = U·Σ·Vᵀ,
//! ε = log Σ (Hencky strain, principal frame). Kirchhoff stress in the principal frame:
//!     τ = 2μ·ε + λ·tr(ε)·1,    tr τ = (2μ + 3λ)·tr ε,    p = −tr τ / 3 = −(λ + 2μ/3)·tr ε.
//!
//! Return map (Klar's three branches + the volumetric cap, applied in this order):
//!   CAP    — if the trial pressure exceeds the consolidation pressure p_c (tr ε < −p_c/K_b),
//!            the excess compression is converted to plastic compaction: tr ε is clamped at
//!            the cap, the discarded volumetric strain accumulates in `compaction`, and p_c
//!            RATCHETS by exp(ξ·Δ) — the tamping memory (an explicit-flow hardening update:
//!            the ratchet applies after the clamp, so within one step the state returns to the
//!            OLD cap; the new cap holds from the next step). The cap only changes tr ε, so
//!            the DP branches below cannot re-violate it.
//!   I      — elastic: δγ = ‖ε̂‖ + α·(2μ+3λ)/(2μ)·tr ε − y_c/(2μ) ≤ 0 (the stress-space DP
//!            yield ‖dev τ‖ + α·tr τ − y_c scaled by 1/(2μ)) → state untouched.
//!   II     — tensile apex: tr ε > tr_apex = y_c/(α·(2μ+3λ)) (dry cohesion y_c = 0 ⇒ any
//!            net expansion) → project to the cone tip, ε = (tr_apex/3)·1. This is what lets
//!            separated grains fall as cohesionless dust and crater walls stand-then-slump.
//!   III    — yield-surface projection: ε ← ε − δγ·ε̂/‖ε̂‖ (deviatoric only; tr ε preserved,
//!            so case III generates no plastic volume change — compaction memory is the cap's
//!            job, and the apex discards only dilation, which deliberately does NOT un-ratchet
//!            the packing state).
//!
//! Volume correction (Tampubolon et al. 2017; the warp-mpm `sand_return_mapping` treatment):
//! the apex branch rebases F at the dilated configuration, so without bookkeeping every
//! transient expansion permanently rebases the particle's rest volume outward — expansion is
//! free and permanent while recompression is resisted elastically from the NEW reference, and
//! the bed ratchets itself loose (measured: a settled heap's interior φ_s fell from the seeded
//! π/6 ≈ 0.52 to 0.27). Fix: per-particle v_c accumulates the apex-DISCARDED volumetric strain
//! and is re-injected into the next trial (ε_eff = ε_trial + (v_c/3)·1), so a dilated particle
//! recompresses stress-free until it has paid the debt back, then resists from the true rest
//! volume. The cap's discarded COMPRESSION stays out of v_c — that is the permanent tamping
//! memory (un-recording it would spring the compaction back).
//!
//! Friction-angle mapping: α = √(2/3)·6·sin φ/(9 − sin² φ) with φ = atan
//! (`Materials::friction_mu`) — the MEAN of the Mohr-Coulomb triaxial-compression fit
//! α_tx = √(2/3)·2·sin φ/(3 − sin φ) and triaxial-extension fit
//! α_te = √(2/3)·2·sin φ/(3 + sin φ). RE-REGISTERED from Klar's α_tx with two-sided probe
//! evidence — one DP cone cannot match MC at every Lode angle, and the U5 gates measure BOTH
//! sides of the hexagon:
//!   * α_tx (compression-circumscribed) over-resists extension/plane-strain states: a square
//!     ziggurat flank at φ = 38.7° stuck at 49.7° (the predicted plane-strain overshoot
//!     atan(tan φ·α_tx/α_ps) ≈ 51°), and the spreading collapse sheet (extension side) read
//!     runout 0.76 against a 1.04 band floor.
//!   * The plane-strain match α_ps = √2·tan φ/√(9 + 12·tan² φ) (Chen & Mizuno, ≈ α_te at
//!     coffee's φ) fixes faces (37° measured) and runout (1.32) but under-holds the
//!     near-compression axisymmetric flank a real rounded heap is made of: the round
//!     ziggurat at φ = 38.7° relaxed to ~24°.
//!   * The mean cone keeps the round-heap repose above the 30° floor (the compression side
//!     borrows from its measured 42.7° headroom at α_mid = √2 tan φ/3, probed between the
//!     two) while releasing the extension side enough for plausible runout.
//!
//! Cohesion y_c comes from `models::cohesion` (dry() = 0 in this rung; the Bishop
//! saturation-dependent mapping is U7).
//!
//! The elastic moduli and cap/hardening constants below are FIXED structural constants of the
//! U5 unit (documented per KTD-9 in the `tests/twofield_bed.rs` knob grid), not per-scene
//! tunables: only the friction angle (and later saturation) comes from `Materials`.

use glam::{Mat3, Vec3};

/// Young's modulus E (reduced units). Sized so the bed is rigid-plastic at heap scale: the
/// base strain of a 10-unit column is ρ·g·H/K_b ≈ 0.3% (≪ 1 — the limit the DP repose angle
/// assumes). The original one-substep choice E = 3000 (base strain ~7%) was measured to cost
/// most of the macro friction: collapse heaps settled at 17–19° at φ = 38.7° regardless of
/// resolution/dt, and the tilt-slab creep threshold sat near 25°; sweeping E = 3000/30000/1e5
/// on one scene moved the deposit angle 16.6/21.3/26.7° and at 1e5 a tilted slab holds 30°
/// but flows at 40° — bracketing φ = 38.7°. Costs CFL substeps: see [`solid_substeps`].
pub const YOUNG_E: f32 = 100000.0;
/// Poisson ratio ν (sand-like; Klar uses 0.3). Probed ν = 0.45 during the U5 repose
/// investigation: no effect on the macro repose/creep thresholds (the post-flow lateral
/// stress is set by the return map, not elastic confinement), so the Klar value stands.
pub const POISSON_NU: f32 = 0.3;

/// Explicit-elasticity CFL target for the dynamic-solid update (c·Δt_sub/h ≤ this).
pub const SOLID_CFL: f32 = 0.4;

/// Elastic sound speed c = √((λ+2μ)/ρ_p) at the structural moduli.
pub fn sound_speed(rho_p: f32) -> f32 {
    ((lame_lambda(YOUNG_E, POISSON_NU) + 2.0 * lame_mu(YOUNG_E, POISSON_NU)) / rho_p).sqrt()
}

/// CFL-driven substep count for the dynamic-solid frame (the sanctioned stiff-elasticity
/// lever; dispatch growth is recorded by the `tests/twofield_bed.rs` budget gate).
/// ρ_p = grain_mass/d³, the bulk density of the bed cell each grain represents. At the
/// defaults (dt = 1/60, h = 2, ρ_p = 1.5): c ≈ 300 → 7 substeps.
pub fn solid_substeps(dt: f32, h: f32, rho_p: f32) -> u32 {
    (sound_speed(rho_p) * dt / (SOLID_CFL * h)).ceil().max(1.0) as u32
}
/// Initial consolidation pressure p_c0: ABOVE the static self-weight MEAN PRESSURE of the
/// gate columns, so a resting heap is elastic (no creep-ratcheting at rest), while a tamp
/// pulse (several g) exceeds it and compacts. RE-REGISTERED 500 → 250 with probe evidence:
/// the original rationale compared p_c to the VERTICAL stress ρ_bulk·g·H ≈ 300, but the cap
/// criterion is on the mean pressure, and a laterally confined (K₀) bed carries only
/// p = (1 + 2K₀)/3·σ_v ≈ 0.62·σ_v at ν = 0.3 — static bottom p ≈ 150, and a quasi-static 4×
/// tamp reaches just ~595, so against p_c0 = 500 the documented 4× tamp probe measured mean
/// p_c 500 → 502.7 (no engagement; the gate's compaction came from slam-wave rectification
/// at ~1/10 the floor). At 250 the static heap keeps a 1.7× elastic margin and the 4× press
/// engages the bottom ~2/3 of the bed honestly.
pub const PC0: f32 = 250.0;
/// Cap hardening exponent ξ: p_c ← p_c·exp(ξ·Δcompaction). Gives the diminishing-returns
/// press cycles the tamp-memory gate asserts (each equal-load cycle compacts less).
/// Probed ξ = 1.5 during the U5 tamp investigation: per-cycle φ̄ gains did NOT follow the
/// 1/ξ cap-headroom budget (0.0012 at ξ = 1.5 vs 0.0015 at ξ = 5 on the same 4× pulse) —
/// per-press consolidation is duration-limited, not headroom-limited — so the registered
/// value stands and the tamp probe presses longer instead (see the gate's knob grid).
pub const HARDEN_XI: f32 = 5.0;
/// Singular-value clamp before the log map (inversion guard): an inverted/degenerate F
/// (σ ≤ 0 after the signed SVD) reads as strong compression and the cap/apex branches absorb
/// it. Mirrors SIG_MIN/SIG_MAX in plasticity.wgsl.
pub const SIG_MIN: f32 = 0.05;
pub const SIG_MAX: f32 = 20.0;
/// Jacobi eigensolver sweep count (3 rotations per sweep, fixed — branch-free-ish; cubic
/// convergence makes 8 sweeps ≫ enough for f32). Mirrors SVD_SWEEPS in plasticity.wgsl.
pub const SVD_SWEEPS: u32 = 8;

/// Over-packing solids-pressure guard (the TFM/KTGF pattern, Fluent §16.5): a grid-level
/// pressure P_sp(φ_s) = K_SP·(φ_s − φ_on)²/(φ_max − φ_s) entering the SOLID momentum as part
/// of the effective/contact stress (KTD-5: NEVER the projection). Diverges as φ_s → φ_max
/// (`Config::packing_limit`), zero below the onset.
pub const SP_STIFF: f32 = 8000.0;
pub const SP_ONSET: f32 = 0.58;
/// Per-step velocity-kick clamp on the solids-pressure acceleration (units/s): bounds the
/// ringing of the divergent term near φ_max without changing where it activates.
pub const SP_KICK_MAX: f32 = 5.0;

/// Lamé μ from (E, ν).
pub fn lame_mu(e: f32, nu: f32) -> f32 {
    e / (2.0 * (1.0 + nu))
}
/// Lamé λ from (E, ν).
pub fn lame_lambda(e: f32, nu: f32) -> f32 {
    e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
}

/// Drucker-Prager friction coefficient α from the Coulomb friction coefficient
/// (μ_f = tan φ): the MEAN of the Mohr-Coulomb triaxial-compression and triaxial-extension
/// cone fits, α = √(2/3)·6·sin φ/(9 − sin² φ). Module header for the two-sided probe
/// evidence that re-registered this from Klar's compression-only match: the U5 gates probe
/// both sides of the MC hexagon (heap flanks load near compression, the spreading collapse
/// sheet near extension), and the compression-circumscribed cone over-resists extension
/// states exactly where the runout gate measures.
pub fn dp_alpha(friction_mu: f32) -> f32 {
    let s = friction_mu.atan().sin();
    (2.0f32 / 3.0).sqrt() * 6.0 * s / (9.0 - s * s)
}

/// Constitutive constants handed to the twin (host-derived; mirrored into Params lanes).
#[derive(Clone, Copy, Debug)]
pub struct SolidConsts {
    pub mu: f32,
    pub lambda: f32,
    pub alpha: f32,
    /// Cohesion in stress units (the y_c apex shift). Dry rung: `models::cohesion::dry()`.
    pub y_coh: f32,
    pub xi: f32,
}

impl SolidConsts {
    /// The U5 defaults at a given Coulomb friction coefficient (dry cohesion).
    pub fn dry(friction_mu: f32) -> Self {
        SolidConsts {
            mu: lame_mu(YOUNG_E, POISSON_NU),
            lambda: lame_lambda(YOUNG_E, POISSON_NU),
            alpha: dp_alpha(friction_mu),
            y_coh: crate::models::cohesion::dry(),
            xi: HARDEN_XI,
        }
    }
}

/// Signed SVD of a 3×3 matrix: F = U·diag(σ)·Vᵀ with det U = det V = +1, σ₀ ≥ σ₁ ≥ |σ₂| and
/// sign(σ₂) = sign(det F) (the rotation-variant SVD MPM plasticity needs).
#[derive(Clone, Copy, Debug)]
pub struct Svd3 {
    pub u: Mat3,
    pub sigma: Vec3,
    pub v: Mat3,
}

fn sym_elem(m: &Mat3, r: usize, c: usize) -> f32 {
    m.col(c)[r]
}

/// One predicated Jacobi rotation in the (p, q) plane: returns the rotation J that zeroes
/// S_pq of the symmetric S (θ = ½·atan2(2·S_pq, S_pp − S_qq); θ = 0 when S_pq ≈ 0 — the
/// atan2(0,0) guard, mirrored in WGSL where atan2(0,0) is indeterminate).
fn jacobi_rot(s: &Mat3, p: usize, q: usize) -> Mat3 {
    let spq = sym_elem(s, p, q);
    let theta = if spq.abs() > 1.0e-12 {
        0.5 * (2.0 * spq).atan2(sym_elem(s, p, p) - sym_elem(s, q, q))
    } else {
        0.0
    };
    let (sn, cs) = theta.sin_cos();
    let mut j = Mat3::IDENTITY.to_cols_array_2d();
    // Column-major [col][row]: column p = (c, s), column q = (−s, c) in the (p, q) plane —
    // the orientation for which (JᵀSJ)_pq = S_pq·cos2θ − (S_pp − S_qq)/2·sin2θ vanishes at
    // the θ above.
    j[p][p] = cs;
    j[q][q] = cs;
    j[q][p] = -sn;
    j[p][q] = sn;
    Mat3::from_cols_array_2d(&j)
}

/// 3×3 SVD via fixed-sweep Jacobi eigenanalysis of FᵀF (V, σ²) + robust U reconstruction
/// (degenerate/rank-deficient F falls back to cross-product completion; reflections land in
/// the sign of σ₂). Pure per-thread math — the WGSL port has no barriers (Tint-safe).
pub fn svd3(f: Mat3) -> Svd3 {
    // V, σ²: cyclic Jacobi on the Gram matrix.
    let mut s = f.transpose() * f;
    let mut v = Mat3::IDENTITY;
    for _ in 0..SVD_SWEEPS {
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let j = jacobi_rot(&s, p, q);
            s = j.transpose() * s * j;
            v *= j;
        }
    }
    let mut eig = Vec3::new(sym_elem(&s, 0, 0), sym_elem(&s, 1, 1), sym_elem(&s, 2, 2));
    let mut vc = [v.col(0), v.col(1), v.col(2)];
    // Descending sort (3 predicated compare-swaps).
    for &(a, b) in &[(0usize, 1usize), (0, 2), (1, 2)] {
        if eig[b] > eig[a] {
            eig = swap_lane(eig, a, b);
            vc.swap(a, b);
        }
    }
    // Right-handed V: the third eigenvector is replaced by the cross completion (±v₂ are both
    // eigenvectors), so det V = +1 exactly.
    vc[2] = vc[0].cross(vc[1]);
    let v = Mat3::from_cols(vc[0], vc[1], vc[2]);

    let sig_eig = Vec3::new(
        eig.x.max(0.0).sqrt(),
        eig.y.max(0.0).sqrt(),
        eig.z.max(0.0).sqrt(),
    );
    // U columns: u_i = F·v_i/σ_i where resolvable, Gram-Schmidt + cross completion otherwise.
    const EPS: f32 = 1.0e-6;
    let mut u0 = f * vc[0];
    if sig_eig.x > EPS {
        u0 /= sig_eig.x;
    } else {
        u0 = Vec3::X; // F ≈ 0
    }
    u0 = u0.normalize_or(Vec3::X);
    let mut u1 = f * vc[1];
    u1 -= u1.dot(u0) * u0;
    let l1 = u1.length();
    if sig_eig.y > EPS && l1 > EPS {
        u1 /= l1;
    } else {
        // Rank ≤ 1: any unit vector orthogonal to u0.
        let pick = if u0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        u1 = (pick - pick.dot(u0) * u0).normalize_or(Vec3::Y);
    }
    let u2 = u0.cross(u1); // det U = +1 by construction
    let u = Mat3::from_cols(u0, u1, u2);
    // σ from projections (best reconstruction; σ₂ signed — negative iff det F < 0).
    let sigma = Vec3::new(
        (f * vc[0]).dot(u0).max(0.0),
        (f * vc[1]).dot(u1).max(0.0),
        (f * vc[2]).dot(u2),
    );
    Svd3 { u, sigma, v }
}

fn swap_lane(v: Vec3, a: usize, b: usize) -> Vec3 {
    let mut arr = v.to_array();
    arr.swap(a, b);
    Vec3::from_array(arr)
}

/// Which branch the return map took (test observability).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    Elastic,
    Apex,
    Shear,
}

/// Return-map result: the projected elastic log-strain, the ratcheted consolidation pressure,
/// the plastic-compaction increment (≥ 0; the cap's discarded volumetric strain), and the
/// branch taken.
#[derive(Clone, Copy, Debug)]
pub struct ReturnMap {
    pub eps: Vec3,
    pub p_c: f32,
    pub d_compaction: f32,
    pub branch: Branch,
}

/// The Klar 3-branch Drucker-Prager return map + compaction cap in log-strain space
/// (module header for the math). Input: trial Hencky strain (principal), current p_c.
pub fn return_map(eps_trial: Vec3, p_c: f32, m: &SolidConsts) -> ReturnMap {
    let k3 = 2.0 * m.mu + 3.0 * m.lambda; // tr τ = k3·tr ε
    let kb = m.lambda + 2.0 * m.mu / 3.0; // p = −kb·tr ε
    let mut eps = eps_trial;
    let mut tr = eps.x + eps.y + eps.z;

    // CAP: compression past the consolidation pressure compacts plastically and ratchets p_c.
    let tr_cap = -p_c / kb;
    let mut d_comp = 0.0;
    if tr < tr_cap {
        d_comp = tr_cap - tr;
        eps += Vec3::splat(d_comp / 3.0);
        tr = tr_cap;
    }
    let p_c_new = p_c * (m.xi * d_comp).exp();

    // Drucker-Prager: δγ = yield/(2μ).
    let dev = eps - Vec3::splat(tr / 3.0);
    let dn = dev.length();
    let dg = dn + m.alpha * k3 / (2.0 * m.mu) * tr - m.y_coh / (2.0 * m.mu);
    if dg <= 0.0 {
        return ReturnMap {
            eps,
            p_c: p_c_new,
            d_compaction: d_comp,
            branch: Branch::Elastic,
        };
    }
    let tr_apex = m.y_coh / (m.alpha * k3);
    if tr > tr_apex {
        return ReturnMap {
            eps: Vec3::splat(tr_apex / 3.0),
            p_c: p_c_new,
            d_compaction: d_comp,
            branch: Branch::Apex,
        };
    }
    let eps = eps - (dg / dn.max(1.0e-12)) * dev;
    ReturnMap {
        eps,
        p_c: p_c_new,
        d_compaction: d_comp,
        branch: Branch::Shear,
    }
}

/// Principal Kirchhoff stress τ = 2μ·ε + λ·tr(ε)·1.
pub fn kirchhoff_principal(eps: Vec3, m: &SolidConsts) -> Vec3 {
    let tr = eps.x + eps.y + eps.z;
    2.0 * m.mu * eps + Vec3::splat(m.lambda * tr)
}

/// DP yield value f(τ(ε)) = ‖dev τ‖ + α·tr τ − y_c (≤ 0 admissible) — the KKT observable.
pub fn yield_value(eps: Vec3, m: &SolidConsts) -> f32 {
    let tau = kirchhoff_principal(eps, m);
    let trt = tau.x + tau.y + tau.z;
    let dev = tau - Vec3::splat(trt / 3.0);
    dev.length() + m.alpha * trt - m.y_coh
}

/// Cap admissibility: trial pressure vs consolidation pressure (≤ 0 admissible).
pub fn cap_value(eps: Vec3, p_c: f32, m: &SolidConsts) -> f32 {
    let kb = m.lambda + 2.0 * m.mu / 3.0;
    -kb * (eps.x + eps.y + eps.z) - p_c
}

/// Per-particle solid state the twin carries (mirrors the `sstate`/`fmat.w` GPU lanes).
#[derive(Clone, Copy, Debug)]
pub struct SolidStep {
    pub f: Mat3,
    /// Kirchhoff stress in the world frame (post-return) — what the next P2G scatters.
    pub tau: Mat3,
    pub p_c: f32,
    pub compaction: f32,
    /// Volume-correction debt: apex-discarded expansion still owed back (module header).
    pub vc: f32,
    pub branch: Branch,
}

/// Full per-particle update twin (mirrors `g2p_solid`'s tail): trial F from the velocity
/// gradient (APIC C), SVD, σ clamp, log map, volume-correction shift, return map, rebuild F
/// and the world Kirchhoff stress.
pub fn solid_step(f: Mat3, c_grad: Mat3, dt: f32, p_c: f32, vc: f32, m: &SolidConsts) -> SolidStep {
    let f_trial = (Mat3::IDENTITY + dt * c_grad) * f;
    let svd = svd3(f_trial);
    let sig = Vec3::new(
        svd.sigma.x.clamp(SIG_MIN, SIG_MAX),
        svd.sigma.y.clamp(SIG_MIN, SIG_MAX),
        svd.sigma.z.clamp(SIG_MIN, SIG_MAX),
    );
    let eps_tr = Vec3::new(sig.x.ln(), sig.y.ln(), sig.z.ln());
    // Volume correction (module header): re-inject the remembered dilation before the return
    // map; whatever the apex discards again becomes the new debt. The elastic/shear branches
    // fold the debt into the rebuilt F (rm.eps already carries the shift), so v_c zeroes there.
    let eps_eff = eps_tr + Vec3::splat(vc / 3.0);
    let rm = return_map(eps_eff, p_c, m);
    let tr_eff = eps_eff.x + eps_eff.y + eps_eff.z;
    let tr_new = rm.eps.x + rm.eps.y + rm.eps.z;
    let vc_new = (tr_eff + rm.d_compaction - tr_new).max(0.0);
    let sig_new = Vec3::new(rm.eps.x.exp(), rm.eps.y.exp(), rm.eps.z.exp());
    let f_new = svd.u * Mat3::from_diagonal(sig_new) * svd.v.transpose();
    let tau_p = kirchhoff_principal(rm.eps, m);
    let tau = svd.u * Mat3::from_diagonal(tau_p) * svd.u.transpose();
    SolidStep {
        f: f_new,
        tau,
        p_c: rm.p_c,
        compaction: rm.d_compaction,
        vc: vc_new,
        branch: rm.branch,
    }
}

/// U7 Bishop saturation-weighted wet cohesion fed into the DP yield (CPU twin of
/// `cohesion_for_saturation` in plasticity.wgsl): `y_c(s) = dry + χ·c_max·bump(s)` with the
/// Bishop factor χ = s (degree of saturation) and `bump(s) = models::cohesion::for_saturation`.
/// `c_max = 0` returns `dry` exactly (the dry-rung passthrough). This is where saturation enters
/// the U5 return map's cohesion argument (the effective-stress coupling KTD-4/KTD-5).
pub fn cohesion_for_saturation(sat: f32, dry: f32, c_max: f32, s_peak: f32) -> f32 {
    if c_max <= 0.0 {
        return dry;
    }
    let s = sat.clamp(0.0, 1.0);
    let bump = crate::models::cohesion::for_saturation(s, s_peak, c_max);
    dry + s * bump
}

/// Over-packing solids pressure P_sp(φ_s) (module constants; mirrors `solids_pressure` in
/// plasticity.wgsl): zero below the onset, diverging as φ_s → φ_max.
pub fn solids_pressure(phi_s: f32, phi_max: f32) -> f32 {
    let x = phi_s.min(phi_max - 1.0e-3); // hard-clip: the divergence stays finite
    if x <= SP_ONSET {
        return 0.0;
    }
    SP_STIFF * (x - SP_ONSET) * (x - SP_ONSET) / (phi_max - x)
}

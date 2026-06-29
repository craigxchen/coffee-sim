//! Extraction kinetics shared by solvers (CPU reference) — the **two-pool** dissolution model.
//!
//! Coffee solute lives in a fast pool (surfaces, fines, broken cells) and a slow pool (intact
//! interiors). A wet grain dissolves solute into the surrounding water at a rate
//! `k_eff = k0 · k_T(T) · area(d) · driving(c) · flux(|u_rel|) · wet(s)`, and per substep releases a
//! **bounded** fraction `pool·(1 − e^{−k_eff·dt})` (never overshoots the pool, at any `dt`). These pure
//! functions are the CPU mirror of the GPU dissolution pass, so the kinetics are unit-tested here first.
//! Released solute moves to overlapping water particles' concentration `c`; the transfer is
//! conservation-exact (the GPU pass mirrors the Phase-1.4 atomic-free two-sided allocation). Reference
//! constants + the yield/TDS target band: `KEEP.md` §1/§5 (re-validate, not gospel).

/// Below this a denominator is treated as degenerate (guards divides for degenerate params).
const EPS: f32 = 1.0e-6;

/// Arrhenius temperature factor, **normalized so it returns 1 at `t_ref`**:
/// `exp(−(Ea/R)·(1/T − 1/T_ref))`. Increases with temperature (hotter extracts faster). The exponent
/// is clamped so an extreme `T` returns a finite value rather than overflowing. A non-positive `t`
/// reads as the reference (factor 1).
#[inline]
pub fn arrhenius(t: f32, ea_over_r: f32, t_ref: f32) -> f32 {
    if t <= EPS || t_ref <= EPS {
        return 1.0;
    }
    let arg = -ea_over_r * (1.0 / t - 1.0 / t_ref);
    arg.clamp(-30.0, 30.0).exp()
}

/// Specific-surface-area factor from grind size: surface ∝ 1/diameter, normalized to `d_ref`
/// (`area(d_ref) = 1`). Finer grind (smaller `d_p`) → more surface → faster extraction.
#[inline]
pub fn area_factor(d_p: f32, d_ref: f32) -> f32 {
    if d_p <= EPS {
        return 1.0;
    }
    d_ref / d_p
}

/// Saturating flow→extraction bridge: `u / (u + u_half)` ∈ [0, 1). Zero at no flow, rises with the
/// water↔grain relative velocity, saturates (advective surface renewal can't exceed the surface limit).
#[inline]
pub fn flux_factor(u_rel: f32, u_half: f32) -> f32 {
    let u = u_rel.max(0.0);
    let denom = u + u_half.max(EPS);
    (u / denom).clamp(0.0, 1.0)
}

/// Moisture gate: `smoothstep(0, s_on, saturation)`. A dry grain (`saturation ≈ 0`) doesn't extract;
/// the gate ramps to 1 as the grain wets past `s_on`.
#[inline]
pub fn wet_gate(saturation: f32, s_on: f32) -> f32 {
    if s_on <= EPS {
        return if saturation > 0.0 { 1.0 } else { 0.0 };
    }
    let t = (saturation / s_on).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Concentration-gradient driving force `max(1 − c/c_sat, 0)`: extraction slows as the local water
/// approaches saturation `c_sat`, and stops at it (a slow pour over-saturates a region).
#[inline]
pub fn driving(c: f32, c_sat: f32) -> f32 {
    if c_sat <= EPS {
        return 0.0;
    }
    (1.0 - c / c_sat).clamp(0.0, 1.0)
}

/// Bounded per-step release from a pool: `pool·(1 − e^{−k_eff·dt})`, clamped to `[0, pool]`. The
/// `(1 − e^{−k·dt})` factor is in `[0, 1)` for any non-negative `k_eff·dt`, so the release never
/// exceeds the remaining pool — no overshoot even at a large `dt`.
#[inline]
pub fn release(pool: f32, k_eff: f32, dt: f32) -> f32 {
    let k = k_eff.max(0.0);
    let frac = 1.0 - (-k * dt.max(0.0)).exp();
    (pool * frac).clamp(0.0, pool.max(0.0))
}

/// Split a grain's extractable solute into the fast and slow pools by `fast_fraction ∈ [0,1]`.
/// Conserves: `s_f + s_s == extractable`.
#[inline]
pub fn split(extractable: f32, fast_fraction: f32) -> (f32, f32) {
    let f = fast_fraction.clamp(0.0, 1.0);
    (extractable * f, extractable * (1.0 - f))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-5;

    #[test]
    fn release_is_bounded_and_never_exceeds_the_pool() {
        let pool = 2.0;
        // huge dt → approaches but never exceeds the pool
        let huge = release(pool, 5.0, 1.0e6);
        assert!(huge <= pool + TOL && huge > pool - 0.01, "huge {huge}");
        // dt = 0 → nothing
        assert_eq!(release(pool, 5.0, 0.0), 0.0);
        // depleted pool → nothing
        assert_eq!(release(0.0, 5.0, 1.0 / 60.0), 0.0);
        // normal step → strictly between 0 and the pool
        let n = release(pool, 0.18, 1.0 / 60.0);
        assert!(n > 0.0 && n < pool);
    }

    #[test]
    fn arrhenius_is_one_at_ref_and_monotone_in_t() {
        let (ea, tr) = (4.0, 1.0);
        assert!((arrhenius(tr, ea, tr) - 1.0).abs() < TOL, "1 at t_ref");
        assert!(arrhenius(1.2, ea, tr) > 1.0, "hotter extracts faster");
        assert!(arrhenius(0.8, ea, tr) < 1.0, "cooler extracts slower");
        // monotone increasing in T
        assert!(arrhenius(1.3, ea, tr) > arrhenius(1.1, ea, tr));
    }

    #[test]
    fn flux_is_zero_at_rest_monotone_and_saturating() {
        let uh = 0.5;
        assert_eq!(flux_factor(0.0, uh), 0.0);
        assert!(flux_factor(1.0, uh) > flux_factor(0.2, uh)); // monotone
        assert!(flux_factor(1.0e6, uh) < 1.0 && flux_factor(1.0e6, uh) > 0.99); // saturates → 1
    }

    #[test]
    fn wet_gate_dry_is_zero_wet_is_one() {
        let s_on = 0.1;
        assert_eq!(wet_gate(0.0, s_on), 0.0); // dry grain: no extraction
        assert!((wet_gate(0.2, s_on) - 1.0).abs() < TOL); // past s_on: full
        assert!(wet_gate(0.05, s_on) > 0.0 && wet_gate(0.05, s_on) < 1.0); // ramp
    }

    #[test]
    fn driving_is_one_at_zero_and_zero_at_saturation() {
        let c_sat = 0.08;
        assert!((driving(0.0, c_sat) - 1.0).abs() < TOL);
        assert_eq!(driving(c_sat, c_sat), 0.0);
        assert_eq!(driving(2.0 * c_sat, c_sat), 0.0); // clamped ≥0 above saturation
    }

    #[test]
    fn split_conserves_and_respects_fast_fraction() {
        let (sf, ss) = split(1.0, 0.3);
        assert!((sf + ss - 1.0).abs() < TOL);
        assert!((sf - 0.3).abs() < TOL && (ss - 0.7).abs() < TOL);
    }

    #[test]
    fn degenerate_params_stay_finite() {
        // extreme temperature: clamped exponent, finite
        assert!(arrhenius(1.0e6, 50.0, 1.0).is_finite());
        assert!(arrhenius(1.0e-9, 50.0, 1.0).is_finite());
        // u_half = 0: guarded divide
        assert!(flux_factor(1.0, 0.0).is_finite());
        // c_sat = 0: no driving (avoid divide)
        assert_eq!(driving(0.5, 0.0), 0.0);
        // d_p = 0: guarded
        assert!(area_factor(0.0, 0.45).is_finite());
    }
}

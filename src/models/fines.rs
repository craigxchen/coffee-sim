//! Fines migration model (shared physics, Phase 6).
//!
//! Coffee *fines* (broken-cell fragments, dust) are a small, mobile sub-grain population. They are
//! **released** from grains into the flowing water (mobilization), ride the draining water
//! (Lagrangian — advected for free on the carrying water particle), and **strain** back onto the
//! grain matrix as the water flows past (deep-bed filtration). We model them as a conserved,
//! **massless**
//! transported scalar: a volume that occupies pore space (so it shows up in the local solid
//! fraction `α_s` and thus permeability) but carries no inertial mass — so the transfer moves a
//! conserved scalar only, writes no velocities, and perturbs no momentum. They are NOT a separate
//! particle species and NOT a grain-dispersion force (`solver_xpbd.md` interphase force #4 is a
//! complementary mechanical effect, out of this scope).
//!
//! These are the pure CPU twins of the GPU transfer kernels (`solvers/xpbd/fines.wgsl`); tests pin
//! their shape and the solver mirrors them exactly.

const EPS: f32 = 1.0e-6;

/// Default Darcy-flux normalization (reduced sim units) for the release/strain transfer fractions
/// ([`flow_fraction`]); larger slows transfer, a huge value freezes it. Calibration-pending; only
/// consulted when fines are enabled (`Materials::fines_fraction > 0` and `Config::fines_rate > 0`).
pub const CRIT_FLUX_DEFAULT: f32 = 0.5;

/// Per-grain seeded fines inventory (volume units): `fines_fraction · grain_volume`. Seeded
/// uniformly onto every grain, so the migration-induced **deviation** from this baseline — not the
/// baseline itself — is what changes local permeability (the calibrated bed permeability already
/// bakes the uniform baseline in). `fines_fraction = 0` ⇒ no fines (the default; feature off).
#[inline]
pub fn fines_seed(grain_volume: f32, fines_fraction: f32) -> f32 {
    grain_volume * fines_fraction.max(0.0)
}

/// Deep-bed filtration coefficients (relative to `Config::fines_rate`). Two flow-driven processes
/// run together: grains **release** lodged fines into the flowing water (mobilization), and
/// suspended fines **strain** back onto grains as the water flows past them (capture). Straining
/// concentrates fines where the most water funnels through — the converging outlet / filter — so the
/// bed clogs there and drawdown slows (real pour-over: fines clog the paper). Equal coefficients let
/// advection carry released fines downstream before they re-strain, building the bottom clog.
pub const RELEASE_COEF: f32 = 1.0;
pub const STRAIN_COEF: f32 = 1.0;

/// Per-step transfer fraction for a flow-driven fines process: `1 − e^{−coef·rate·(flux/crit)·dt}`
/// ∈ `[0,1)`. Both release and straining scale with the local Darcy flux normalized by `crit_flux`,
/// so a stagnant bed neither releases nor strains (no transfer at zero flux), and a very large
/// `crit_flux` effectively freezes the transfer (used by tests to isolate the permeability response).
/// Monotone increasing in flux, bounded below 1.
#[inline]
pub fn flow_fraction(flux: f32, crit_flux: f32, rate: f32, dt: f32, coef: f32) -> f32 {
    let x = coef.max(0.0) * rate.max(0.0) * (flux.max(0.0) / crit_flux.max(EPS)) * dt.max(0.0);
    1.0 - (-x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_scales_with_fraction_and_volume() {
        assert!((fines_seed(8.0, 0.1) - 0.8).abs() < 1e-6);
        assert_eq!(fines_seed(8.0, 0.0), 0.0, "fraction 0 ⇒ no fines");
        assert_eq!(fines_seed(8.0, -1.0), 0.0, "negative fraction floored to 0");
    }

    #[test]
    fn flow_fraction_is_zero_at_rest_and_grows_with_flux() {
        let (crit, rate, dt) = (0.5, 2.0, 1.0 / 60.0);
        assert_eq!(
            flow_fraction(0.0, crit, rate, dt, 1.0),
            0.0,
            "stagnant ⇒ no transfer"
        );
        let mut prev = -1.0;
        for &f in &[0.0, 0.1, 0.5, 1.0, 5.0, 100.0] {
            let frac = flow_fraction(f, crit, rate, dt, 1.0);
            assert!(frac >= prev, "monotone increasing in flux (at flux={f})");
            assert!((0.0..1.0).contains(&frac), "bounded in [0,1): {frac}");
            prev = frac;
        }
    }

    #[test]
    fn large_crit_flux_freezes_the_transfer() {
        // A huge crit_flux normalizes the flux to ~0, so neither process moves fines — lets a test
        // hold a manual clog fixed and isolate the permeability response.
        let frac = flow_fraction(5.0, 1.0e6, 2.0, 1.0 / 60.0, 1.0);
        assert!(
            frac < 1e-4,
            "huge crit_flux should freeze transfer, got {frac}"
        );
    }
}

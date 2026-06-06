//! Fines migration model (shared physics, Phase 6).
//!
//! Coffee *fines* (broken-cell fragments, dust) are a small, mobile sub-grain population that
//! detaches from grains under flow, rides the draining water (Lagrangian — advected for free on the
//! carrying water particle), and re-deposits where the flow slackens — locally clogging the pore
//! space, lowering permeability, and slowing drawdown. We model them as a conserved, **massless**
//! transported scalar: a volume that occupies pore space (so it shows up in the local solid
//! fraction `α_s` and thus permeability) but carries no inertial mass — so the transfer moves a
//! conserved scalar only, writes no velocities, and perturbs no momentum. They are NOT a separate
//! particle species and NOT a grain-dispersion force (`solver_xpbd.md` interphase force #4 is a
//! complementary mechanical effect, out of this scope).
//!
//! These are the pure CPU twins of the GPU transfer kernels (`solvers/xpbd/fines.wgsl`); tests pin
//! their shape and the solver mirrors them exactly.

const EPS: f32 = 1.0e-6;

/// Default critical Darcy flux (reduced sim units) at which erosion and deposition balance —
/// the zero-crossing of [`net_rate`]. Calibration-pending; only consulted when fines are enabled
/// (`Materials::fines_fraction > 0` and `Config::fines_rate > 0`).
pub const CRIT_FLUX_DEFAULT: f32 = 0.5;

/// Per-grain seeded fines inventory (volume units): `fines_fraction · grain_volume`. Seeded
/// uniformly onto every grain, so the migration-induced **deviation** from this baseline — not the
/// baseline itself — is what changes local permeability (the calibrated bed permeability already
/// bakes the uniform baseline in). `fines_fraction = 0` ⇒ no fines (the default; feature off).
#[inline]
pub fn fines_seed(grain_volume: f32, fines_fraction: f32) -> f32 {
    grain_volume * fines_fraction.max(0.0)
}

/// Signed fines transfer rate constant (1/s) as a function of local Darcy flux
/// `|v_water − v_grain|`. Positive ⇒ erosion (grain → suspended), negative ⇒ deposition
/// (suspended → grain), zero at the critical flux. Bounded in `(−rate, rate)`, monotone increasing
/// in flux, single zero-crossing at `crit_flux`: fast flow scours fines loose, slack flow lets them
/// settle. `rate` is the overall scale (`Config::fines_rate`).
#[inline]
pub fn net_rate(flux: f32, crit_flux: f32, rate: f32) -> f32 {
    let f = flux.max(0.0);
    let c = crit_flux.max(EPS);
    rate * (f - c) / (f + c)
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
    fn net_rate_crosses_zero_once_at_crit_flux() {
        let (crit, rate) = (0.5, 2.0);
        assert!(net_rate(crit, crit, rate).abs() < 1e-6, "zero at crit flux");
        assert!(net_rate(0.0, crit, rate) < 0.0, "no flow ⇒ deposition");
        assert!(net_rate(5.0, crit, rate) > 0.0, "fast flow ⇒ erosion");
    }

    #[test]
    fn net_rate_is_monotone_and_bounded() {
        let (crit, rate) = (0.5, 2.0);
        let fluxes = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0];
        let mut prev = f32::NEG_INFINITY;
        for &f in &fluxes {
            let r = net_rate(f, crit, rate);
            assert!(r > prev, "monotone increasing in flux (at flux={f})");
            assert!(r.abs() <= rate, "bounded by |rate| (at flux={f})");
            prev = r;
        }
        // No-flow deposition saturates at exactly −rate; erosion approaches +rate.
        assert!((net_rate(0.0, crit, rate) + rate).abs() < 1e-6);
        assert!(net_rate(1.0e6, crit, rate) > rate - 1e-3);
    }
}

//! Wetting / swelling / absorption math shared by solvers (CPU reference).
//!
//! When a dry grain absorbs water, the water **leaves the fluid phase and becomes grain volume**:
//! the solid swells by exactly the absorbed volume, so total volume (fluid + solid) is conserved.
//! Mass moves with it (the absorbed water's mass joins the grain). These pure functions are the
//! CPU mirror of what the GPU absorption passes compute, so the conservation/swelling math is
//! unit-tested here first. Reference values + ranges: the wetting plan + `KEEP.md` §1.

/// Per-grain absorption capacity, as a **volume of water** the grain can hold.
///
/// `r_max` is the gravimetric moisture ratio at saturation (mass water / mass dry solid; coffee
/// ≈ 1.5), and `rho_ratio = ρ_s/ρ_w` converts that mass ratio into a volume of water relative to
/// the dry grain volume `v_dry`. So `V_cap = r_max · (ρ_s/ρ_w) · V_dry`.
#[inline]
pub fn capacity(v_dry: f32, r_max: f32, rho_ratio: f32) -> f32 {
    r_max * rho_ratio * v_dry
}

/// Saturation fraction `s ∈ [0, 1]` = absorbed volume over capacity. Drives the cohesion curve.
/// A grain with no capacity (degenerate) reads as dry.
#[inline]
pub fn saturation(v_abs: f32, v_cap: f32) -> f32 {
    if v_cap <= 0.0 {
        0.0
    } else {
        (v_abs / v_cap).clamp(0.0, 1.0)
    }
}

/// Swollen grain volume. Swelling **equals** the absorbed volume by construction (`V_dry + V_abs`),
/// which is what makes absorption volume-conserving: fluid lost == solid gained.
#[inline]
pub fn effective_volume(v_dry: f32, v_abs: f32) -> f32 {
    v_dry + v_abs
}

/// Swollen contact diameter from the swollen volume (a sphere grows as the cube root of volume).
/// Guards a degenerate dry volume.
#[inline]
pub fn effective_diameter(d_dry: f32, v_eff: f32, v_dry: f32) -> f32 {
    if v_dry > 0.0 {
        d_dry * (v_eff / v_dry).cbrt()
    } else {
        d_dry
    }
}

/// Effective grain mass once it has absorbed `v_abs` of water: the dry mass plus the absorbed
/// water's mass `ρ_w · V_abs`. Used wherever a velocity/momentum/XPBD weight depends on grain mass.
#[inline]
pub fn grain_eff_mass(m_dry: f32, v_abs: f32, rho_w: f32) -> f32 {
    m_dry + rho_w * v_abs
}

/// Effective water-particle mass as it shrinks: a partially-absorbed particle (remaining-volume
/// fraction `f_w`) carries proportionally less mass.
#[inline]
pub fn water_eff_mass(particle_mass: f32, f_w: f32) -> f32 {
    particle_mass * f_w
}

/// Volume a grain wants to absorb this step — a **bounded** first-order law toward capacity:
/// `(V_cap − V_abs) · (1 − e^{−k·dt})`. The `(1 − e^{−k·dt})` factor is in `[0, 1)` for any `dt`,
/// so the demand can never exceed the remaining deficit (no overshoot of `V_cap`, even at large
/// `dt`). `k_abs` sets how fast a wet grain saturates.
#[inline]
pub fn absorb_demand(v_abs: f32, v_cap: f32, k_abs: f32, dt: f32) -> f32 {
    let deficit = (v_cap - v_abs).max(0.0);
    deficit * (1.0 - (-k_abs * dt).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn capacity_is_r_max_times_rho_ratio_times_volume() {
        assert!((capacity(2.0, 1.5, 1.3) - 1.5 * 1.3 * 2.0).abs() < EPS);
    }

    #[test]
    fn effective_volume_grows_linearly_and_is_dry_at_zero() {
        assert!((effective_volume(3.0, 0.0) - 3.0).abs() < EPS);
        // doubling absorbed volume doubles the swelling increment
        let inc1 = effective_volume(3.0, 1.0) - 3.0;
        let inc2 = effective_volume(3.0, 2.0) - 3.0;
        assert!((inc2 - 2.0 * inc1).abs() < EPS);
    }

    #[test]
    fn effective_diameter_is_cube_root_scaling() {
        // V_eff = 8·V_dry ⇒ d_eff = 2·d_dry
        let v_dry = 1.5;
        assert!((effective_diameter(2.0, 8.0 * v_dry, v_dry) - 4.0).abs() < EPS);
    }

    #[test]
    fn saturation_clamps_to_unit_range() {
        assert!((saturation(0.5, 2.0) - 0.25).abs() < EPS);
        assert_eq!(saturation(5.0, 2.0), 1.0); // over capacity → clamped
        assert_eq!(saturation(-1.0, 2.0), 0.0); // negative → clamped
        assert_eq!(saturation(1.0, 0.0), 0.0); // no capacity → dry
    }

    #[test]
    fn absorb_demand_is_rate_limited_and_never_exceeds_deficit() {
        let (v_abs, v_cap, k) = (0.2, 1.0, 0.5);
        let deficit = v_cap - v_abs;

        // huge dt → approaches but never exceeds the deficit
        let huge = absorb_demand(v_abs, v_cap, k, 1.0e6);
        assert!(huge <= deficit + EPS, "demand {huge} deficit {deficit}");

        // normal dt → strictly between 0 and the deficit
        let normal = absorb_demand(v_abs, v_cap, k, 1.0 / 60.0);
        assert!(normal > 0.0 && normal < deficit);

        // saturated grain wants nothing
        assert_eq!(absorb_demand(v_cap, v_cap, k, 1.0 / 60.0), 0.0);
    }

    #[test]
    fn grain_mass_rises_by_exactly_absorbed_water_mass() {
        let (m_dry, v_abs, rho_w) = (1.5, 0.4, 1.0);
        assert!((grain_eff_mass(m_dry, v_abs, rho_w) - (m_dry + rho_w * v_abs)).abs() < EPS);
        // water-side mass scales with the remaining fraction
        assert!((water_eff_mass(1.0, 0.25) - 0.25).abs() < EPS);
    }

    #[test]
    fn volume_identity_holds() {
        let (v_dry, v_abs) = (1.2, 0.7);
        assert!((v_dry + v_abs - effective_volume(v_dry, v_abs)).abs() < EPS);
    }
}

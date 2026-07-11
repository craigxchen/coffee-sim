//! Wetting / swelling / absorption math shared by solvers (CPU reference).
//!
//! Bound water and structural grain volume are separate inventories. Absorption transfers an
//! exact water volume (and its corresponding mass) from the free-water phase into `V_bound`.
//! Structural swelling is a saturation-driven constitutive response with its own maximum
//! diameter fraction; it is **not** `V_dry + V_bound`. This distinction prevents a retention
//! target from silently forcing an equal geometric-volume increase. These pure functions mirror
//! the GPU formulas and keep the independent conservation and swelling contracts testable.

/// Per-grain absorption capacity, as a **volume of water** the grain can hold.
///
/// `r_max` is the gravimetric moisture ratio at saturation (mass water / mass dry solid; coffee
/// ≈ 1.5), and `rho_ratio = ρ_s/ρ_w` converts that mass ratio into a volume of water relative to
/// the dry grain volume `v_dry`. So `V_cap = r_max · (ρ_s/ρ_w) · V_dry`.
#[inline]
pub fn capacity(v_dry: f32, r_max: f32, rho_ratio: f32) -> f32 {
    r_max * rho_ratio * v_dry
}

/// Bound-water capacity derived from the actual dry material-point mass.
///
/// `particle_mass / v_water` is the reduced water density, so dividing the gravimetric bound
/// mass `r_max * m_dry` by it gives a water volume in solver units. Production recipe solvers
/// should use this form: it remains invariant when the number or spacing of numerical solid
/// samples changes and cannot disagree with the mass used by P2G.
#[inline]
pub fn capacity_from_dry_mass(m_dry: f32, r_max: f32, v_water: f32, particle_mass: f32) -> f32 {
    if particle_mass <= 0.0 {
        0.0
    } else {
        m_dry.max(0.0) * r_max.max(0.0) * v_water.max(0.0) / particle_mass
    }
}

/// Saturation fraction `s ∈ [0, 1]` = bound-water volume over capacity. Drives swelling and
/// cohesion. A grain with no capacity (degenerate) reads as dry.
#[inline]
pub fn saturation(v_bound: f32, v_cap: f32) -> f32 {
    if v_cap <= 0.0 {
        0.0
    } else {
        (v_bound / v_cap).clamp(0.0, 1.0)
    }
}

/// Structural diameter at the current saturation.
///
/// `max_diameter_swelling` is the fractional diameter increase at saturation. Negative inputs
/// are treated as zero, so the diameter remains in
/// `[d_dry, d_dry * (1 + max_diameter_swelling)]` for physical non-negative inputs.
#[inline]
pub fn structural_diameter(
    d_dry: f32,
    v_bound: f32,
    v_cap: f32,
    max_diameter_swelling: f32,
) -> f32 {
    let stretch = 1.0 + saturation(v_bound, v_cap) * max_diameter_swelling.max(0.0);
    d_dry * stretch
}

/// Structural volume at the current saturation:
/// `V_struct = V_dry * (1 + s * max_diameter_swelling)^3`.
///
/// This depends on bound water only through saturation. It does not consume, create, or alias
/// bound-water volume.
#[inline]
pub fn structural_volume(v_dry: f32, v_bound: f32, v_cap: f32, max_diameter_swelling: f32) -> f32 {
    let stretch = 1.0 + saturation(v_bound, v_cap) * max_diameter_swelling.max(0.0);
    v_dry * stretch * stretch * stretch
}

/// Reduced-unit mass carried by a bound-water volume. One full water-particle volume `v_water`
/// carries `particle_mass`; the same ratio is used on the free-water and bound-water sides.
#[inline]
pub fn bound_water_mass(v_bound: f32, v_water: f32, particle_mass: f32) -> f32 {
    if v_water <= 0.0 {
        0.0
    } else {
        v_bound.max(0.0) / v_water * particle_mass
    }
}

/// Effective grain mass: immutable dry recipe mass plus the independently-accounted bound-water
/// mass. Structural swelling does not enter this equation.
#[inline]
pub fn grain_effective_mass(m_dry: f32, v_bound: f32, v_water: f32, particle_mass: f32) -> f32 {
    m_dry + bound_water_mass(v_bound, v_water, particle_mass)
}

/// Effective water-particle mass as it shrinks: a partially-absorbed particle (remaining-volume
/// fraction `f_w`) carries proportionally less mass.
#[inline]
pub fn water_eff_mass(particle_mass: f32, f_w: f32) -> f32 {
    particle_mass * f_w
}

/// Volume a grain wants to absorb this step — a **bounded** first-order law toward capacity:
/// `(V_cap − V_bound) · (1 − e^{−k·dt})`. The exponential factor is in `[0, 1)` for any `dt`,
/// so the demand can never exceed the remaining deficit (no overshoot of `V_cap`, even at large
/// `dt`). `k_abs` sets how fast a wet grain saturates.
#[inline]
pub fn absorb_demand(v_bound: f32, v_cap: f32, k_abs: f32, dt: f32) -> f32 {
    let deficit = (v_cap - v_bound).max(0.0);
    deficit * (1.0 - (-k_abs * dt).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn capacity_is_r_max_times_rho_ratio_times_volume() {
        assert!((capacity(2.0, 1.5, 1.3) - 1.5 * 1.3 * 2.0).abs() < EPS);
        let (m_dry, r_max, v_water, particle_mass) = (7.0, 0.4, 0.25, 2.0);
        let volume = capacity_from_dry_mass(m_dry, r_max, v_water, particle_mass);
        assert!((bound_water_mass(volume, v_water, particle_mass) - r_max * m_dry).abs() < EPS);
    }

    #[test]
    fn zero_swelling_keeps_structural_volume_independent_of_bound_water() {
        let v_dry = 3.0;
        for v_bound in [0.0, 0.25, 2.0, 20.0] {
            assert!((structural_volume(v_dry, v_bound, 2.0, 0.0) - v_dry).abs() < EPS);
        }
    }

    #[test]
    fn structural_swelling_is_saturation_bounded_and_volume_matches_diameter() {
        let (d_dry, v_dry, v_cap, max_swelling) = (2.0, 1.5, 4.0, 0.25);
        for v_bound in [-1.0, 0.0, 1.0, 4.0, 40.0] {
            let diameter = structural_diameter(d_dry, v_bound, v_cap, max_swelling);
            let volume = structural_volume(v_dry, v_bound, v_cap, max_swelling);
            assert!(diameter >= d_dry && diameter <= d_dry * (1.0 + max_swelling));
            assert!(volume >= v_dry && volume <= v_dry * (1.0 + max_swelling).powi(3));
            assert!((volume / v_dry - (diameter / d_dry).powi(3)).abs() < EPS);
        }
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
    fn transferred_water_mass_and_momentum_are_identical_on_both_sides() {
        let (particle_mass, v_water, transferred) = (3.0, 2.0, 0.4);
        let free_mass_loss = water_eff_mass(particle_mass, transferred / v_water);
        let bound_mass_gain = bound_water_mass(transferred, v_water, particle_mass);
        assert!((free_mass_loss - bound_mass_gain).abs() < EPS);

        let m_dry = 7.0;
        assert!(
            (grain_effective_mass(m_dry, transferred, v_water, particle_mass)
                - (m_dry + bound_mass_gain))
                .abs()
                < EPS
        );

        // One-node algebra of the GPU S/D/Q ledger. Donors lose shares T/S at their own
        // velocities; receivers gain shares T/D at Q/S. Both sums equal rho*T*Q/S.
        let donor_volume = [0.2_f32, 0.5, 0.8];
        let donor_velocity = [-1.2_f32, 0.4, 2.1];
        let supply: f32 = donor_volume.iter().sum();
        let volume_momentum: f32 = donor_volume
            .iter()
            .zip(donor_velocity)
            .map(|(volume, velocity)| volume * velocity)
            .sum();
        let demand_shares = [0.3_f32, 0.7];
        let demand: f32 = demand_shares.iter().sum();
        let transfer = 0.6_f32.min(supply).min(demand);
        let water_mass_per_volume = particle_mass / v_water;

        let donor_lost_momentum: f32 = donor_volume
            .iter()
            .zip(donor_velocity)
            .map(|(volume, velocity)| {
                water_mass_per_volume * volume * (transfer / supply) * velocity
            })
            .sum();
        let ledger_momentum = water_mass_per_volume * transfer * volume_momentum / supply;
        let receiver_gained_momentum: f32 = demand_shares
            .iter()
            .map(|share| {
                let gained_volume = share * transfer / demand;
                water_mass_per_volume * gained_volume * volume_momentum / supply
            })
            .sum();
        assert!((donor_lost_momentum - ledger_momentum).abs() < EPS);
        assert!((receiver_gained_momentum - ledger_momentum).abs() < EPS);

        // The receiver's inelastic merge preserves its old momentum plus its ledger share.
        let old_mass = 4.0_f32;
        let old_velocity = -0.6_f32;
        let receiver_gain = demand_shares[0] * transfer / demand;
        let gained_mass = water_mass_per_volume * receiver_gain;
        let incoming_momentum = gained_mass * volume_momentum / supply;
        let new_mass = old_mass + gained_mass;
        let new_velocity = (old_mass * old_velocity + incoming_momentum) / new_mass;
        assert!(
            (new_mass * new_velocity - (old_mass * old_velocity + incoming_momentum)).abs() < EPS
        );
    }

    #[test]
    fn equal_saturation_can_have_equal_structure_but_different_bound_mass() {
        let max_swelling = 0.3;
        let v_a = structural_volume(1.0, 0.5, 1.0, max_swelling);
        let v_b = structural_volume(1.0, 1.0, 2.0, max_swelling);
        assert!((v_a - v_b).abs() < EPS);
        assert!(bound_water_mass(1.0, 1.0, 1.0) > bound_water_mass(0.5, 1.0, 1.0));
    }
}

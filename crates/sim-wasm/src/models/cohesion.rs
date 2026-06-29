//! Granular cohesion + repose calibration for the dry bed.
//!
//! Dry coffee grounds hold a steep pile mostly through friction + interlocking, with a little
//! cohesion. For the dry bed a single constant suffices; wet cohesion is a saturation-dependent
//! bump (cohesion rises as the bed wets, then falls as it saturates). Reference values: `KEEP.md`
//! §1 (porosity 0.40, grind ≈ 450 µm) / §5 (dry-bed settle bands).

/// Dry inter-grain cohesion strength, in position-correction units per contact.
///
/// Kept weak on purpose: strong cohesion balls the grains into clumps — the granular analogue of
/// the PBF surface-tension *beading* failure. Tuned alongside `friction_mu`/`rolling_damping`
/// against the standing-heap invariant (a believable repose without clumping), not to a target
/// angle. Starts at zero (friction-only); raise if the sphere pile slumps too shallow.
pub fn dry() -> f32 {
    0.0
}

/// Cohesion (and the dry contact search) reaches out to this multiple of the grain diameter;
/// beyond it grains don't interact. Must stay below `support_radius / grain_diameter` so the
/// neighbor grid covers it.
pub const COHESION_RANGE_RATIO: f32 = 1.3;

/// Saturation-dependent cohesion for wet grains.
///
/// Capillary bridges strengthen contacts at intermediate saturation, peaking at `s_peak`, then
/// collapse near full saturation. The bump is piecewise-linear, zero at dry/flooded endpoints,
/// and scaled by `c_max`.
pub fn for_saturation(saturation: f32, s_peak: f32, c_max: f32) -> f32 {
    let saturation = saturation.clamp(0.0, 1.0);
    let s_peak = s_peak.clamp(f32::EPSILON, 1.0 - f32::EPSILON);

    if saturation <= s_peak {
        c_max * saturation / s_peak
    } else {
        c_max * (1.0 - saturation) / (1.0 - s_peak)
    }
}

#[cfg(test)]
mod tests {
    use super::for_saturation;

    #[test]
    fn cohesion_is_zero_at_dry_and_flooded_endpoints() {
        let s_peak = 0.4;
        let c_max = 2.0;

        assert_eq!(for_saturation(0.0, s_peak, c_max), 0.0);
        assert_eq!(for_saturation(1.0, s_peak, c_max), 0.0);
    }

    #[test]
    fn peak_reaches_c_max_and_is_sweep_maximum() {
        let s_peak = 0.4;
        let c_max = 2.0;
        let peak = for_saturation(s_peak, s_peak, c_max);

        assert_eq!(peak, c_max);

        for i in 0..=100 {
            let saturation = i as f32 / 100.0;
            let cohesion = for_saturation(saturation, s_peak, c_max);
            assert!(
                cohesion <= peak,
                "saturation {saturation} cohesion {cohesion} peak {peak}"
            );
        }
    }

    #[test]
    fn rises_until_peak_and_falls_after_peak() {
        let s_peak = 0.4;
        let c_max = 2.0;
        let mut previous = for_saturation(0.0, s_peak, c_max);

        for i in 1..=40 {
            let saturation = i as f32 / 100.0;
            let cohesion = for_saturation(saturation, s_peak, c_max);
            assert!(
                cohesion >= previous,
                "saturation {saturation} cohesion {cohesion} previous {previous}"
            );
            previous = cohesion;
        }

        previous = for_saturation(s_peak, s_peak, c_max);
        for i in 41..=100 {
            let saturation = i as f32 / 100.0;
            let cohesion = for_saturation(saturation, s_peak, c_max);
            assert!(
                cohesion <= previous,
                "saturation {saturation} cohesion {cohesion} previous {previous}"
            );
            previous = cohesion;
        }
    }

    #[test]
    fn cohesion_scales_linearly_with_c_max() {
        let s_peak = 0.4;
        let saturation = 0.25;
        let cohesion = for_saturation(saturation, s_peak, 2.0);

        assert_eq!(for_saturation(saturation, s_peak, 4.0), cohesion * 2.0);
    }

    #[test]
    fn saturation_is_clamped_to_valid_range() {
        let s_peak = 0.4;
        let c_max = 2.0;

        assert_eq!(for_saturation(-0.5, s_peak, c_max), 0.0);
        assert_eq!(for_saturation(1.5, s_peak, c_max), 0.0);
    }
}

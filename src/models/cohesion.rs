//! Granular cohesion + repose calibration for the dry bed.
//!
//! Dry coffee grounds hold a steep pile mostly through friction + interlocking, with a little
//! cohesion. For the dry bed a single constant suffices; the saturation-dependent cohesion
//! *curve* (cohesion rises as the bed wets, then falls as it saturates) is the wetting phase's
//! job and is left as a documented stub below. Reference values: `KEEP.md` §1 (porosity 0.40,
//! grind ≈ 450 µm) / §5 (dry-bed settle bands).

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

/// Saturation-dependent cohesion (wetting phase) — **stub**. Returns the dry constant until the
/// water/bed coupling step adds moisture; capillary bridges then raise cohesion at intermediate
/// saturation and collapse it near full saturation.
pub fn for_saturation(_saturation: f32) -> f32 {
    dry()
}

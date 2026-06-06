//! Pure control-mapping helpers for the web UI (compiled on every target so they are unit-tested
//! natively; the wasm handle in `web.rs` calls them). These translate the original v1 frontend's
//! controls — water-velocity slider, 2D spout pad, scene buttons — into the rewrite's canonical
//! `EmissionInput`/`Scene` inputs. Keeping them here (not in the wasm-only module) means the
//! conversions are exercised by `cargo test` on native, where the GPU/browser parts can't run.

use crate::engine::Scene;

/// Sim length units per physical metre (KEEP.md §2: V60 paper ≈ 10 cm ≈ 5.77 sim units → ≈ 27.7).
pub const SIM_UNITS_PER_METER: f32 = 27.7;

/// Convert the v1 water-velocity slider (metres/second) to the rewrite's volumetric
/// `EmissionInput.flow_rate` (sim-units³/s). The emitter derives the stream exit speed as
/// `flow / A_eff` with `A_eff = π·nozzle_radius²·discharge_coeff`, so this is the inverse: a slider
/// of `v` m/s reproduces an exit speed of `v·SIM_UNITS_PER_METER` sim-units/s. `ML_PER_SIM_UNIT3` is
/// **not** involved (that scale is for mL display/capping only).
#[inline]
pub fn flow_rate_for_velocity(speed_m_s: f32, nozzle_radius: f32, discharge_coeff: f32) -> f32 {
    let a_eff = std::f32::consts::PI * nozzle_radius * nozzle_radius * discharge_coeff;
    a_eff * (speed_m_s.max(0.0) * SIM_UNITS_PER_METER)
}

/// Map the v1 spout pad (normalized `(u, v) ∈ [-1, 1]`, with a separate height) to a world kettle
/// position. `half_extent` is the in-plane half-width the pad spans (the scene's x/z bound). Out-of-
/// range pad values clamp to the pad edge.
#[inline]
pub fn kettle_pos_for_pad(u: f32, v: f32, height: f32, half_extent: f32) -> [f32; 3] {
    [
        u.clamp(-1.0, 1.0) * half_extent,
        height,
        v.clamp(-1.0, 1.0) * half_extent,
    ]
}

/// The scenes the web UI exposes, mapped from the original's main scene buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebScene {
    /// "Center Pour" — the V60 brew (continuous pour; the water-velocity + spout controls drive the
    /// live inflow, as in v1).
    CenterPour,
    /// "Water Only" — a water dam with no bed (the original's free-stream).
    WaterOnly,
}

impl WebScene {
    /// Parse the frontend's scene id.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "center-pour" => Some(Self::CenterPour),
            "water-only" => Some(Self::WaterOnly),
            _ => None,
        }
    }

    /// The `Scene` to build. CenterPour declares a pour (the live velocity/spout controls feed
    /// `EmissionInput`); WaterOnly is a bed-less dam where the pour controls have no effect.
    pub fn build(self) -> Scene {
        match self {
            Self::CenterPour => Scene::v60_pour(),
            Self::WaterOnly => Scene::dam_break(),
        }
    }

    /// Whether this scene accepts the live pour controls (Center Pour does; Water Only doesn't).
    pub fn accepts_pour(self) -> bool {
        matches!(self, Self::CenterPour)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_to_flow_round_trips_through_exit_speed() {
        let (r, c) = (0.25_f32, 1.0_f32);
        let a_eff = std::f32::consts::PI * r * r * c;
        for &v in &[0.0_f32, 0.05, 0.12, 0.3, 0.6] {
            let flow = flow_rate_for_velocity(v, r, c);
            // The emitter recovers exit_speed = flow / A_eff; it must equal v·SIM_UNITS_PER_METER.
            let exit_speed = flow / a_eff;
            assert!(
                (exit_speed - v * SIM_UNITS_PER_METER).abs() < 1e-3,
                "v={v}: exit_speed {exit_speed} != {}",
                v * SIM_UNITS_PER_METER
            );
        }
    }

    #[test]
    fn velocity_to_flow_is_monotone_zero_at_rest_and_excludes_ml_scale() {
        let (r, c) = (0.5_f32, 0.9_f32);
        assert_eq!(flow_rate_for_velocity(0.0, r, c), 0.0);
        assert_eq!(
            flow_rate_for_velocity(-1.0, r, c),
            0.0,
            "negative clamps to 0"
        );
        assert!(flow_rate_for_velocity(0.2, r, c) < flow_rate_for_velocity(0.4, r, c));
        // Sanity: the value is A_eff·v·27.7 — no 5.20 mL factor folded in.
        let expect = std::f32::consts::PI * r * r * c * 0.3 * SIM_UNITS_PER_METER;
        assert!((flow_rate_for_velocity(0.3, r, c) - expect).abs() < 1e-4);
    }

    #[test]
    fn spout_pad_maps_and_clamps() {
        let p = kettle_pos_for_pad(0.5, -0.5, 2.5, 6.0);
        assert!(
            (p[0] - 3.0).abs() < 1e-6 && (p[1] - 2.5).abs() < 1e-6 && (p[2] + 3.0).abs() < 1e-6
        );
        // Out-of-range clamps to the pad edge.
        let q = kettle_pos_for_pad(5.0, -9.0, 1.0, 6.0);
        assert!((q[0] - 6.0).abs() < 1e-6 && (q[2] + 6.0).abs() < 1e-6);
    }

    #[test]
    fn scene_ids_resolve() {
        assert_eq!(WebScene::from_id("center-pour"), Some(WebScene::CenterPour));
        assert_eq!(WebScene::from_id("water-only"), Some(WebScene::WaterOnly));
        assert_eq!(WebScene::from_id("nope"), None);
        // CenterPour accepts the live pour controls; WaterOnly does not.
        assert!(WebScene::CenterPour.accepts_pour());
        assert!(!WebScene::WaterOnly.accepts_pour());
        // Both build a scene.
        let _ = WebScene::CenterPour.build();
        let _ = WebScene::WaterOnly.build();
    }
}

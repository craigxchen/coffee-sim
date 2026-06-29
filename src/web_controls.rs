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

/// The scenes the web UI exposes. The first two are the main "Scenes" tab; the rest are the
/// "Debug Scenes" catalog, ported by intent from the `main` MPM branch's `DebugScene` enum (plus the
/// new `SandWall`). Kebab ids match `main`'s debug-scene ids (with `sand-wall` added).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebScene {
    /// "Center Pour" — the V60 brew (continuous pour; the water-velocity + spout controls drive the
    /// live inflow, as in v1).
    CenterPour,
    /// "Water Only" — a water dam with no bed (the original's free-stream).
    WaterOnly,

    // --- Debug catalog ---
    /// `filter-water-block` — a still water block in the filter cone over the bed, no pour.
    FilterWaterBlock,
    /// `off-center-filter-wall-pour` — V60 pour with the spout parked toward the filter wall.
    OffCenterFilterWallPour,
    /// `seeded-paper-wall-sheet` — a thin water sheet clinging to the filter wall, no pour.
    SeededPaperWallSheet,
    /// `filter-apex-drain` — water low in the filter cone, no bed, draining through the apex.
    FilterApexDrain,
    /// `cup-wall-floor-corner-contact` — cup-only water in a wall/floor corner wedge.
    CupWallFloorCornerContact,
    /// `asymmetric-cup-mound-settle` — cup-only off-center water mound settling level.
    AsymmetricCupMoundSettle,
    /// `hydrostatic-column` — cup-only tall narrow on-axis water column.
    HydrostaticColumn,
    /// `dam-break-slosh` — cup-only half-fill released to slosh.
    DamBreakSlosh,
    /// `sparse-free-jet` — a thin slow pour into the empty cup.
    SparseFreeJet,
    /// `high-velocity-jet-impact` — a fast pour plunging onto a shallow cup pool.
    HighVelocityJetImpact,
    /// `uniform-bed-saturation` — the V60 bed pre-saturated with water, no pour.
    UniformBedSaturation,
    /// `permeability-comparison` — V60 pour with a tighter (lower-permeability) bed.
    PermeabilityComparison,
    /// `particle-capacity-stress` — V60 pour driven at a high flow rate.
    ParticleCapacityStress,
    /// `sand-wall` (NEW) — a grain wall on one side, a water block released against it.
    SandWall,
}

impl WebScene {
    /// Parse the frontend's scene id (kebab-case).
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "center-pour" => Some(Self::CenterPour),
            "water-only" => Some(Self::WaterOnly),
            "filter-water-block" => Some(Self::FilterWaterBlock),
            "off-center-filter-wall-pour" => Some(Self::OffCenterFilterWallPour),
            "seeded-paper-wall-sheet" => Some(Self::SeededPaperWallSheet),
            "filter-apex-drain" => Some(Self::FilterApexDrain),
            "cup-wall-floor-corner-contact" => Some(Self::CupWallFloorCornerContact),
            "asymmetric-cup-mound-settle" => Some(Self::AsymmetricCupMoundSettle),
            "hydrostatic-column" => Some(Self::HydrostaticColumn),
            "dam-break-slosh" => Some(Self::DamBreakSlosh),
            "sparse-free-jet" => Some(Self::SparseFreeJet),
            "high-velocity-jet-impact" => Some(Self::HighVelocityJetImpact),
            "uniform-bed-saturation" => Some(Self::UniformBedSaturation),
            "permeability-comparison" => Some(Self::PermeabilityComparison),
            "particle-capacity-stress" => Some(Self::ParticleCapacityStress),
            "sand-wall" => Some(Self::SandWall),
            _ => None,
        }
    }

    /// The `Scene` to build for this UI scene.
    pub fn build(self) -> Scene {
        match self {
            Self::CenterPour => Scene::v60_pour(),
            Self::WaterOnly => Scene::v60_pour_water_only(),
            Self::FilterWaterBlock => Scene::debug_filter_water_block(),
            Self::OffCenterFilterWallPour => Scene::debug_off_center_filter_wall_pour(),
            Self::SeededPaperWallSheet => Scene::debug_seeded_paper_wall_sheet(),
            Self::FilterApexDrain => Scene::debug_filter_apex_drain(),
            Self::CupWallFloorCornerContact => Scene::debug_cup_wall_floor_corner_contact(),
            Self::AsymmetricCupMoundSettle => Scene::debug_asymmetric_cup_mound_settle(),
            Self::HydrostaticColumn => Scene::debug_hydrostatic_column(),
            Self::DamBreakSlosh => Scene::debug_dam_break_slosh(),
            Self::SparseFreeJet => Scene::debug_sparse_free_jet(),
            Self::HighVelocityJetImpact => Scene::debug_high_velocity_jet_impact(),
            Self::UniformBedSaturation => Scene::debug_uniform_bed_saturation(),
            Self::PermeabilityComparison => Scene::debug_permeability_comparison(),
            Self::ParticleCapacityStress => Scene::debug_particle_capacity_stress(),
            Self::SandWall => Scene::sand_wall(),
        }
    }

    /// Whether this scene accepts the live pour controls. The pour-driven scenes (the V60 pours +
    /// the free-jet / jet-impact / permeability / capacity debug scenes) feed `EmissionInput`; the
    /// seeded-block / cup-static / settle scenes are released-only.
    pub fn accepts_pour(self) -> bool {
        matches!(
            self,
            Self::CenterPour
                | Self::WaterOnly
                | Self::OffCenterFilterWallPour
                | Self::SparseFreeJet
                | Self::HighVelocityJetImpact
                | Self::PermeabilityComparison
                | Self::ParticleCapacityStress
        )
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

    /// Every scene id the UI can emit, paired with its variant — also the source of truth this test
    /// round-trips through `from_id`/`build`/`accepts_pour`.
    const SCENES: &[(&str, WebScene)] = &[
        ("center-pour", WebScene::CenterPour),
        ("water-only", WebScene::WaterOnly),
        ("filter-water-block", WebScene::FilterWaterBlock),
        (
            "off-center-filter-wall-pour",
            WebScene::OffCenterFilterWallPour,
        ),
        ("seeded-paper-wall-sheet", WebScene::SeededPaperWallSheet),
        ("filter-apex-drain", WebScene::FilterApexDrain),
        (
            "cup-wall-floor-corner-contact",
            WebScene::CupWallFloorCornerContact,
        ),
        (
            "asymmetric-cup-mound-settle",
            WebScene::AsymmetricCupMoundSettle,
        ),
        ("hydrostatic-column", WebScene::HydrostaticColumn),
        ("dam-break-slosh", WebScene::DamBreakSlosh),
        ("sparse-free-jet", WebScene::SparseFreeJet),
        ("high-velocity-jet-impact", WebScene::HighVelocityJetImpact),
        ("uniform-bed-saturation", WebScene::UniformBedSaturation),
        ("permeability-comparison", WebScene::PermeabilityComparison),
        ("particle-capacity-stress", WebScene::ParticleCapacityStress),
        ("sand-wall", WebScene::SandWall),
    ];

    #[test]
    fn scene_ids_resolve_and_build() {
        for &(id, scene) in SCENES {
            assert_eq!(WebScene::from_id(id), Some(scene), "id {id} resolves");
            // Every scene builds without panicking.
            let _ = scene.build();
        }
        assert_eq!(WebScene::from_id("nope"), None);
    }

    #[test]
    fn pour_scenes_accept_pour_and_seeded_scenes_do_not() {
        // Pour-driven scenes feed EmissionInput.
        for s in [
            WebScene::CenterPour,
            WebScene::WaterOnly,
            WebScene::OffCenterFilterWallPour,
            WebScene::SparseFreeJet,
            WebScene::HighVelocityJetImpact,
            WebScene::PermeabilityComparison,
            WebScene::ParticleCapacityStress,
        ] {
            assert!(s.accepts_pour(), "{s:?} should accept pour");
        }
        // Seeded-block / cup-static / settle scenes are released-only.
        for s in [
            WebScene::FilterWaterBlock,
            WebScene::SeededPaperWallSheet,
            WebScene::FilterApexDrain,
            WebScene::CupWallFloorCornerContact,
            WebScene::AsymmetricCupMoundSettle,
            WebScene::HydrostaticColumn,
            WebScene::DamBreakSlosh,
            WebScene::UniformBedSaturation,
            WebScene::SandWall,
        ] {
            assert!(!s.accepts_pour(), "{s:?} should not accept pour");
        }
    }
}

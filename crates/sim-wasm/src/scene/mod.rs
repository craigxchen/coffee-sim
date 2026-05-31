pub(crate) mod seeds;
pub(crate) mod settings;

pub(crate) use settings::{sim_speed_from_meters_per_second, sim_speed_to_meters_per_second};
pub(crate) use settings::{BedSpec, SimSettings, SpoutSettings, METERS_PER_SIM_UNIT};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DebugScene {
    FilterWaterBlock,
    OffCenterFilterWallPour,
    SeededPaperWallSheet,
    FilterApexDrain,
    CupWallFloorCornerContact,
    AsymmetricCupMoundSettle,
    HydrostaticColumn,
    DamBreakSlosh,
    SparseFreeJet,
    HighVelocityJetImpact,
    UniformBedSaturation,
    PermeabilityComparison,
    ParticleCapacityStress,
}

impl DebugScene {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 13] = [
        Self::FilterWaterBlock,
        Self::OffCenterFilterWallPour,
        Self::SeededPaperWallSheet,
        Self::FilterApexDrain,
        Self::CupWallFloorCornerContact,
        Self::AsymmetricCupMoundSettle,
        Self::HydrostaticColumn,
        Self::DamBreakSlosh,
        Self::SparseFreeJet,
        Self::HighVelocityJetImpact,
        Self::UniformBedSaturation,
        Self::PermeabilityComparison,
        Self::ParticleCapacityStress,
    ];

    pub(crate) fn from_id(id: &str) -> Option<Self> {
        match id {
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
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::FilterWaterBlock => "filter-water-block",
            Self::OffCenterFilterWallPour => "off-center-filter-wall-pour",
            Self::SeededPaperWallSheet => "seeded-paper-wall-sheet",
            Self::FilterApexDrain => "filter-apex-drain",
            Self::CupWallFloorCornerContact => "cup-wall-floor-corner-contact",
            Self::AsymmetricCupMoundSettle => "asymmetric-cup-mound-settle",
            Self::HydrostaticColumn => "hydrostatic-column",
            Self::DamBreakSlosh => "dam-break-slosh",
            Self::SparseFreeJet => "sparse-free-jet",
            Self::HighVelocityJetImpact => "high-velocity-jet-impact",
            Self::UniformBedSaturation => "uniform-bed-saturation",
            Self::PermeabilityComparison => "permeability-comparison",
            Self::ParticleCapacityStress => "particle-capacity-stress",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum SceneSpec {
    DefaultV60,
    BenchmarkFreeStream,
    BenchmarkCenterPour,
    Debug(DebugScene),
}

impl SceneSpec {
    pub(crate) fn settings(&self) -> SimSettings {
        match self {
            Self::DefaultV60 | Self::BenchmarkCenterPour => {
                let mut s = SimSettings::default_v60();
                s.spout.origin = coffee_sim_core::Vec3::new(0.0, 7.1, 0.0);
                s
            }
            Self::BenchmarkFreeStream => {
                let mut s = SimSettings::default_v60();
                s.bed = None;
                s.spout.origin = coffee_sim_core::Vec3::new(0.0, 6.8, 0.0);
                s
            }
            Self::Debug(scene) => debug_settings(*scene),
        }
    }
}

fn debug_settings(scene: DebugScene) -> SimSettings {
    use coffee_sim_core::Vec3;
    let mut s = SimSettings::default_v60();
    match scene {
        DebugScene::FilterWaterBlock | DebugScene::SeededPaperWallSheet => {
            s.initial_water_speed_m_s = 0.0;
        }
        DebugScene::OffCenterFilterWallPour => {
            s.spout.origin = Vec3::new(2.4, 7.1, 0.0);
            s.initial_water_speed_m_s = 0.28;
            s.spout.nozzle_radius = 0.15;
            s.spout.max_flow_rate_ml_s = 10.0;
        }
        DebugScene::FilterApexDrain => {
            s.bed = None;
            s.initial_water_speed_m_s = 0.0;
        }
        DebugScene::CupWallFloorCornerContact
        | DebugScene::AsymmetricCupMoundSettle
        | DebugScene::HydrostaticColumn
        | DebugScene::DamBreakSlosh
        | DebugScene::SparseFreeJet
        | DebugScene::HighVelocityJetImpact => {
            s.filter = None;
            s.bed = None;
            s.initial_water_speed_m_s = 0.0;
        }
        DebugScene::UniformBedSaturation => {
            s.initial_water_speed_m_s = 0.0;
            if let Some(bed) = s.bed.as_mut() {
                bed.initially_saturated = true;
            }
        }
        DebugScene::PermeabilityComparison => {
            if let Some(bed) = s.bed.as_mut() {
                bed.grind_diameter_um = 320.0;
            }
            s.initial_water_speed_m_s = 0.18;
        }
        DebugScene::ParticleCapacityStress => {
            s.max_particles = 32_000;
            s.spout.nozzle_radius = 0.20;
            s.spout.max_flow_rate_ml_s = 18.0;
            s.initial_water_speed_m_s = 0.42;
        }
    }
    if matches!(scene, DebugScene::SparseFreeJet) {
        s.spout.origin = Vec3::new(0.0, 7.4, 0.0);
        s.spout.nozzle_radius = 0.07;
        s.spout.max_flow_rate_ml_s = 2.0;
        s.initial_water_speed_m_s = 0.18;
    }
    if matches!(scene, DebugScene::HighVelocityJetImpact) {
        s.spout.origin = Vec3::new(0.0, 6.9, 0.0);
        s.spout.nozzle_radius = 0.14;
        s.spout.max_flow_rate_ml_s = 14.0;
        s.initial_water_speed_m_s = 0.48;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_debug_scene_ids_resolve() {
        for scene in DebugScene::ALL {
            assert_eq!(DebugScene::from_id(scene.id()), Some(scene));
        }
    }

    #[test]
    fn default_scene_has_bed_capacity_and_water_headroom() {
        let settings = SceneSpec::DefaultV60.settings();
        let bed_particles = settings.bed.as_ref().unwrap().num_particles;
        assert!(bed_particles > 1_000);
        assert!(settings.max_particles > bed_particles + 2_000);
    }

    #[test]
    fn water_only_scene_has_no_bed() {
        let settings = SceneSpec::BenchmarkFreeStream.settings();
        assert!(settings.bed.is_none());
    }
}

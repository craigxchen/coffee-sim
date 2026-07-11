//! Solver-neutral, user-calibrated coffee physics controls.
//!
//! These values are deliberately separate from solver discretization knobs. A calibration
//! profile describes the physical behavior an owner wants to reproduce; individual solvers are
//! responsible for converting it into their own uniforms and state. Defaults are provisional
//! starting points, not empirical claims or regression targets.

use std::fmt;

use serde::Serialize;

pub const CALIBRATION_SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationControl {
    BedRetention,
    BoundWaterCapacity,
    MaxDiameterSwelling,
    Log10Permeability,
    DragMultiplier,
    AbsorptionRate,
    SuctionHead,
    BloomDelay,
    DpFrictionAngle,
    DpCohesion,
    BedCompactionTangent,
    InitialConsolidationPressure,
    NumericalShearModulus,
    FullCompactionSolidFraction,
    ExtractionFastRate,
    ExtractionSlowRate,
    InletTemperature,
    AmbientTemperature,
    ThermalExchangeRate,
    ReferenceBedPorosity,
    CapillaryRelativePermeabilityExponent,
    CapillaryMobilityFloor,
    Log10FilterPaperPermeability,
    FilterWaterCapacity,
}

impl CalibrationControl {
    pub const ALL: [Self; 24] = [
        Self::BedRetention,
        Self::BoundWaterCapacity,
        Self::MaxDiameterSwelling,
        Self::Log10Permeability,
        Self::DragMultiplier,
        Self::AbsorptionRate,
        Self::SuctionHead,
        Self::BloomDelay,
        Self::DpFrictionAngle,
        Self::DpCohesion,
        Self::BedCompactionTangent,
        Self::InitialConsolidationPressure,
        Self::NumericalShearModulus,
        Self::FullCompactionSolidFraction,
        Self::ExtractionFastRate,
        Self::ExtractionSlowRate,
        Self::InletTemperature,
        Self::AmbientTemperature,
        Self::ThermalExchangeRate,
        Self::ReferenceBedPorosity,
        Self::CapillaryRelativePermeabilityExponent,
        Self::CapillaryMobilityFloor,
        Self::Log10FilterPaperPermeability,
        Self::FilterWaterCapacity,
    ];

    pub fn from_id(id: &str) -> Option<Self> {
        CALIBRATION_SCHEMA
            .iter()
            .find(|metadata| metadata.id == id)
            .map(|metadata| metadata.control)
    }

    pub fn metadata(self) -> &'static ControlMetadata {
        &CALIBRATION_SCHEMA[self as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlMetadata {
    pub control: CalibrationControl,
    pub id: &'static str,
    pub label: &'static str,
    pub units: &'static str,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    /// Use logarithmic slider positioning. This is false for the `log10_*_permeability_m2`
    /// controls because their stored values are already logarithmic.
    pub logarithmic: bool,
    /// The solver must be rebuilt/reset before the new value is physically coherent.
    pub requires_rebuild: bool,
}

impl ControlMetadata {
    fn clamp(self, value: f32) -> f32 {
        value.clamp(self.min, self.max)
    }
}

pub const CALIBRATION_SCHEMA: [ControlMetadata; 24] = [
    ControlMetadata {
        control: CalibrationControl::BedRetention,
        id: "bed_retention_g_per_g",
        label: "Whole-bed retention",
        units: "g/g dry coffee",
        min: 0.0,
        max: 5.0,
        step: 0.05,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::BoundWaterCapacity,
        id: "bound_water_capacity_g_per_g",
        label: "Bound-water capacity",
        units: "g/g dry coffee",
        min: 0.0,
        max: 3.0,
        step: 0.05,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::MaxDiameterSwelling,
        id: "max_diameter_swelling_fraction",
        label: "Maximum diameter swelling",
        units: "fraction",
        min: 0.0,
        max: 0.15,
        step: 0.005,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::Log10Permeability,
        id: "log10_permeability_m2",
        label: "Intrinsic permeability",
        units: "log10(m²)",
        min: -16.0,
        max: -8.0,
        step: 0.1,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::DragMultiplier,
        id: "drag_multiplier",
        label: "Interphase drag multiplier",
        units: "×",
        min: 0.01,
        max: 100.0,
        step: 0.01,
        logarithmic: true,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::AbsorptionRate,
        id: "absorption_rate_per_s",
        label: "Absorption rate",
        units: "s⁻¹",
        min: 0.001,
        max: 10.0,
        step: 0.001,
        logarithmic: true,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::SuctionHead,
        id: "suction_head_m",
        label: "Capillary suction head",
        units: "m water",
        min: 0.0,
        max: 0.25,
        step: 0.005,
        logarithmic: false,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::BloomDelay,
        id: "bloom_delay_s",
        label: "Bloom wetting delay",
        units: "s",
        min: 0.0,
        max: 30.0,
        step: 0.1,
        logarithmic: false,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::DpFrictionAngle,
        id: "dp_friction_angle_deg",
        label: "Granular friction angle",
        units: "°",
        min: 0.0,
        max: 60.0,
        step: 0.5,
        logarithmic: false,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::DpCohesion,
        id: "dp_cohesion_pa",
        label: "Wet cohesion",
        units: "Pa",
        min: 0.0,
        max: 5_000.0,
        step: 10.0,
        logarithmic: false,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::BedCompactionTangent,
        id: "bed_compaction_tangent_pa",
        label: "Bed compaction stiffness",
        units: "Pa",
        min: 100.0,
        max: 200_000.0,
        step: 10.0,
        logarithmic: true,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::InitialConsolidationPressure,
        id: "initial_consolidation_pressure_pa",
        label: "Initial consolidation pressure",
        units: "Pa",
        min: 10.0,
        max: 50_000.0,
        step: 10.0,
        logarithmic: true,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::NumericalShearModulus,
        id: "numerical_shear_modulus_pa",
        label: "Numerical shear compliance",
        units: "Pa",
        min: 1.0,
        max: 2_000.0,
        step: 1.0,
        logarithmic: true,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::FullCompactionSolidFraction,
        id: "full_compaction_solid_fraction",
        label: "Full-compaction solid fraction",
        units: "fraction",
        min: 0.50,
        max: 0.90,
        step: 0.005,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::ExtractionFastRate,
        id: "extraction_fast_rate_per_s",
        label: "Fast extraction rate",
        units: "s⁻¹",
        min: 0.0001,
        max: 5.0,
        step: 0.0001,
        logarithmic: true,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::ExtractionSlowRate,
        id: "extraction_slow_rate_per_s",
        label: "Slow extraction rate",
        units: "s⁻¹",
        min: 0.0001,
        max: 5.0,
        step: 0.0001,
        logarithmic: true,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::InletTemperature,
        id: "inlet_temperature_c",
        label: "Inlet water temperature",
        units: "°C",
        min: 0.0,
        max: 100.0,
        step: 0.5,
        logarithmic: false,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::AmbientTemperature,
        id: "ambient_temperature_c",
        label: "Ambient temperature",
        units: "°C",
        min: -10.0,
        max: 50.0,
        step: 0.5,
        logarithmic: false,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::ThermalExchangeRate,
        id: "thermal_exchange_rate_per_s",
        label: "Thermal exchange rate",
        units: "s⁻¹",
        min: 0.0001,
        max: 10.0,
        step: 0.0001,
        logarithmic: true,
        requires_rebuild: false,
    },
    ControlMetadata {
        control: CalibrationControl::ReferenceBedPorosity,
        id: "reference_bed_porosity_fraction",
        label: "Reference bed porosity",
        units: "fraction",
        min: 0.10,
        max: 0.90,
        step: 0.01,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::CapillaryRelativePermeabilityExponent,
        id: "capillary_relative_permeability_exponent",
        label: "Capillary permeability exponent",
        units: "dimensionless",
        min: 0.25,
        max: 12.0,
        step: 0.05,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::CapillaryMobilityFloor,
        id: "capillary_mobility_floor_fraction",
        label: "Capillary mobility floor",
        units: "fraction",
        min: 0.0001,
        max: 1.0,
        step: 0.0001,
        logarithmic: true,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::Log10FilterPaperPermeability,
        id: "log10_filter_paper_permeability_m2",
        label: "Filter-paper permeability",
        units: "log10(m²)",
        min: -18.0,
        max: -8.0,
        step: 0.1,
        logarithmic: false,
        requires_rebuild: true,
    },
    ControlMetadata {
        control: CalibrationControl::FilterWaterCapacity,
        id: "filter_water_capacity_g",
        label: "Filter water capacity",
        units: "g",
        min: 0.0,
        max: 25.0,
        step: 0.1,
        logarithmic: false,
        requires_rebuild: true,
    },
];

pub fn calibration_schema() -> &'static [ControlMetadata] {
    &CALIBRATION_SCHEMA
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CalibrationProfile {
    schema_version: u32,
    bed_retention_g_per_g: f32,
    bound_water_capacity_g_per_g: f32,
    max_diameter_swelling_fraction: f32,
    log10_permeability_m2: f32,
    drag_multiplier: f32,
    absorption_rate_per_s: f32,
    suction_head_m: f32,
    bloom_delay_s: f32,
    dp_friction_angle_deg: f32,
    dp_cohesion_pa: f32,
    bed_compaction_tangent_pa: f32,
    initial_consolidation_pressure_pa: f32,
    numerical_shear_modulus_pa: f32,
    full_compaction_solid_fraction: f32,
    extraction_fast_rate_per_s: f32,
    extraction_slow_rate_per_s: f32,
    inlet_temperature_c: f32,
    ambient_temperature_c: f32,
    thermal_exchange_rate_per_s: f32,
    reference_bed_porosity_fraction: f32,
    capillary_relative_permeability_exponent: f32,
    capillary_mobility_floor_fraction: f32,
    log10_filter_paper_permeability_m2: f32,
    filter_water_capacity_g: f32,
}

impl Default for CalibrationProfile {
    fn default() -> Self {
        Self {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            // Starting points only. They are intentionally adjustable and are not test targets.
            bed_retention_g_per_g: 2.1,
            // The bound-water prior is deliberately much smaller than whole-bed retention:
            // retained brew also occupies inter-particle pores and the filter. Published
            // particle-swelling results conflict, so the midpoint below is only a visible
            // starting point inside the full 0..15% control range.
            bound_water_capacity_g_per_g: 0.15,
            max_diameter_swelling_fraction: 0.08,
            // Consolidated-bed measurements are commonly around 1e-14..1e-13 m² while a
            // loose pour-over bed can be more permeable. Start broad rather than silently
            // inheriting the legacy Kozeny estimate near 1e-10 m².
            log10_permeability_m2: -12.0,
            drag_multiplier: 1.0,
            // A first-order prior near 0.04 /s reaches roughly 60–80% of its asymptote after
            // 30 s (before local supply/bloom effects), matching the published kinetic band
            // without treating that study's scattered particle measurements as a fixed target.
            absorption_rate_per_s: 0.04,
            suction_head_m: 0.08,
            bloom_delay_s: 4.0,
            dp_friction_angle_deg: 38.0,
            dp_cohesion_pa: 100.0,
            // These are visible calibration starting points, not asserted coffee constants.
            // Physical compaction is handled by the implicit solve; the much softer shear value
            // is explicitly labeled numerical compliance and only controls the predictor CFL.
            bed_compaction_tangent_pa: 10_000.0,
            initial_consolidation_pressure_pa: 1_000.0,
            numerical_shear_modulus_pa: 50.0,
            full_compaction_solid_fraction: 0.72,
            extraction_fast_rate_per_s: 0.18,
            extraction_slow_rate_per_s: 0.018,
            inlet_temperature_c: 93.0,
            ambient_temperature_c: 22.0,
            thermal_exchange_rate_per_s: 0.05,
            // Provisional retention/filter starting priors only. These remain broad UI controls
            // until an interactive calibration is accepted; tests intentionally do not pin them.
            reference_bed_porosity_fraction: 0.40,
            capillary_relative_permeability_exponent: 3.0,
            capillary_mobility_floor_fraction: 0.01,
            log10_filter_paper_permeability_m2: -13.0,
            filter_water_capacity_g: 1.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SetControlResult {
    pub control: CalibrationControl,
    pub requested: f32,
    pub applied: f32,
    pub clamped: bool,
    pub requires_rebuild: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationError {
    UnknownControl {
        id: String,
    },
    NonFiniteValue {
        id: &'static str,
    },
    UnsupportedSchemaVersion {
        found: u32,
        supported: u32,
    },
    InvalidProfile {
        id: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownControl { id } => write!(f, "unknown calibration control '{id}'"),
            Self::NonFiniteValue { id } => {
                write!(f, "calibration control '{id}' requires a finite value")
            }
            Self::UnsupportedSchemaVersion { found, supported } => write!(
                f,
                "unsupported calibration schema version {found}; supported version is {supported}"
            ),
            Self::InvalidProfile { id, reason } => {
                write!(f, "invalid calibration control '{id}': {reason}")
            }
        }
    }
}

impl std::error::Error for CalibrationError {}

impl CalibrationProfile {
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn value_by_id(&self, id: &str) -> Result<f32, CalibrationError> {
        let control = CalibrationControl::from_id(id)
            .ok_or_else(|| CalibrationError::UnknownControl { id: id.to_owned() })?;
        Ok(self.value(control))
    }

    pub fn set_by_id(
        &mut self,
        id: &str,
        requested: f32,
    ) -> Result<SetControlResult, CalibrationError> {
        let control = CalibrationControl::from_id(id)
            .ok_or_else(|| CalibrationError::UnknownControl { id: id.to_owned() })?;
        self.set(control, requested)
    }

    pub fn set(
        &mut self,
        control: CalibrationControl,
        requested: f32,
    ) -> Result<SetControlResult, CalibrationError> {
        let metadata = control.metadata();
        if !requested.is_finite() {
            return Err(CalibrationError::NonFiniteValue { id: metadata.id });
        }

        let mut applied = metadata.clamp(requested);
        // Whole-bed retention includes bound water, so keep the two independently adjustable
        // capacities in a physically meaningful order without silently changing the other knob.
        match control {
            CalibrationControl::BedRetention => {
                applied = applied.max(self.bound_water_capacity_g_per_g);
            }
            CalibrationControl::BoundWaterCapacity => {
                applied = applied.min(self.bed_retention_g_per_g);
            }
            _ => {}
        }

        self.assign(control, applied);
        Ok(SetControlResult {
            control,
            requested,
            applied,
            clamped: applied != requested,
            requires_rebuild: metadata.requires_rebuild,
        })
    }

    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION {
            return Err(CalibrationError::UnsupportedSchemaVersion {
                found: self.schema_version,
                supported: CALIBRATION_SCHEMA_VERSION,
            });
        }
        for control in CalibrationControl::ALL {
            let metadata = control.metadata();
            let value = self.value(control);
            if !value.is_finite() {
                return Err(CalibrationError::NonFiniteValue { id: metadata.id });
            }
            if !(metadata.min..=metadata.max).contains(&value) {
                return Err(CalibrationError::InvalidProfile {
                    id: metadata.id,
                    reason: "value is outside the declared schema range",
                });
            }
        }
        if self.bound_water_capacity_g_per_g > self.bed_retention_g_per_g {
            return Err(CalibrationError::InvalidProfile {
                id: CalibrationControl::BoundWaterCapacity.metadata().id,
                reason: "bound-water capacity exceeds whole-bed retention",
            });
        }
        Ok(())
    }

    fn value(&self, control: CalibrationControl) -> f32 {
        match control {
            CalibrationControl::BedRetention => self.bed_retention_g_per_g,
            CalibrationControl::BoundWaterCapacity => self.bound_water_capacity_g_per_g,
            CalibrationControl::MaxDiameterSwelling => self.max_diameter_swelling_fraction,
            CalibrationControl::Log10Permeability => self.log10_permeability_m2,
            CalibrationControl::DragMultiplier => self.drag_multiplier,
            CalibrationControl::AbsorptionRate => self.absorption_rate_per_s,
            CalibrationControl::SuctionHead => self.suction_head_m,
            CalibrationControl::BloomDelay => self.bloom_delay_s,
            CalibrationControl::DpFrictionAngle => self.dp_friction_angle_deg,
            CalibrationControl::DpCohesion => self.dp_cohesion_pa,
            CalibrationControl::BedCompactionTangent => self.bed_compaction_tangent_pa,
            CalibrationControl::InitialConsolidationPressure => {
                self.initial_consolidation_pressure_pa
            }
            CalibrationControl::NumericalShearModulus => self.numerical_shear_modulus_pa,
            CalibrationControl::FullCompactionSolidFraction => self.full_compaction_solid_fraction,
            CalibrationControl::ExtractionFastRate => self.extraction_fast_rate_per_s,
            CalibrationControl::ExtractionSlowRate => self.extraction_slow_rate_per_s,
            CalibrationControl::InletTemperature => self.inlet_temperature_c,
            CalibrationControl::AmbientTemperature => self.ambient_temperature_c,
            CalibrationControl::ThermalExchangeRate => self.thermal_exchange_rate_per_s,
            CalibrationControl::ReferenceBedPorosity => self.reference_bed_porosity_fraction,
            CalibrationControl::CapillaryRelativePermeabilityExponent => {
                self.capillary_relative_permeability_exponent
            }
            CalibrationControl::CapillaryMobilityFloor => self.capillary_mobility_floor_fraction,
            CalibrationControl::Log10FilterPaperPermeability => {
                self.log10_filter_paper_permeability_m2
            }
            CalibrationControl::FilterWaterCapacity => self.filter_water_capacity_g,
        }
    }

    fn assign(&mut self, control: CalibrationControl, value: f32) {
        match control {
            CalibrationControl::BedRetention => self.bed_retention_g_per_g = value,
            CalibrationControl::BoundWaterCapacity => self.bound_water_capacity_g_per_g = value,
            CalibrationControl::MaxDiameterSwelling => {
                self.max_diameter_swelling_fraction = value;
            }
            CalibrationControl::Log10Permeability => self.log10_permeability_m2 = value,
            CalibrationControl::DragMultiplier => self.drag_multiplier = value,
            CalibrationControl::AbsorptionRate => self.absorption_rate_per_s = value,
            CalibrationControl::SuctionHead => self.suction_head_m = value,
            CalibrationControl::BloomDelay => self.bloom_delay_s = value,
            CalibrationControl::DpFrictionAngle => self.dp_friction_angle_deg = value,
            CalibrationControl::DpCohesion => self.dp_cohesion_pa = value,
            CalibrationControl::BedCompactionTangent => {
                self.bed_compaction_tangent_pa = value;
            }
            CalibrationControl::InitialConsolidationPressure => {
                self.initial_consolidation_pressure_pa = value;
            }
            CalibrationControl::NumericalShearModulus => {
                self.numerical_shear_modulus_pa = value;
            }
            CalibrationControl::FullCompactionSolidFraction => {
                self.full_compaction_solid_fraction = value;
            }
            CalibrationControl::ExtractionFastRate => self.extraction_fast_rate_per_s = value,
            CalibrationControl::ExtractionSlowRate => self.extraction_slow_rate_per_s = value,
            CalibrationControl::InletTemperature => self.inlet_temperature_c = value,
            CalibrationControl::AmbientTemperature => self.ambient_temperature_c = value,
            CalibrationControl::ThermalExchangeRate => self.thermal_exchange_rate_per_s = value,
            CalibrationControl::ReferenceBedPorosity => {
                self.reference_bed_porosity_fraction = value;
            }
            CalibrationControl::CapillaryRelativePermeabilityExponent => {
                self.capillary_relative_permeability_exponent = value;
            }
            CalibrationControl::CapillaryMobilityFloor => {
                self.capillary_mobility_floor_fraction = value;
            }
            CalibrationControl::Log10FilterPaperPermeability => {
                self.log10_filter_paper_permeability_m2 = value;
            }
            CalibrationControl::FilterWaterCapacity => self.filter_water_capacity_g = value,
        }
    }

    pub fn bed_retention_g_per_g(&self) -> f32 {
        self.bed_retention_g_per_g
    }

    pub fn bound_water_capacity_g_per_g(&self) -> f32 {
        self.bound_water_capacity_g_per_g
    }

    pub fn capillary_water_capacity_g_per_g(&self) -> f32 {
        self.bed_retention_g_per_g - self.bound_water_capacity_g_per_g
    }

    pub fn max_diameter_swelling_fraction(&self) -> f32 {
        self.max_diameter_swelling_fraction
    }

    pub fn permeability_m2(&self) -> f32 {
        10.0_f32.powf(self.log10_permeability_m2)
    }

    pub fn drag_multiplier(&self) -> f32 {
        self.drag_multiplier
    }

    pub fn absorption_rate_per_s(&self) -> f32 {
        self.absorption_rate_per_s
    }

    pub fn suction_head_m(&self) -> f32 {
        self.suction_head_m
    }

    pub fn bloom_delay_s(&self) -> f32 {
        self.bloom_delay_s
    }

    pub fn dp_friction_angle_deg(&self) -> f32 {
        self.dp_friction_angle_deg
    }

    pub fn dp_friction_mu(&self) -> f32 {
        self.dp_friction_angle_deg.to_radians().tan()
    }

    pub fn dp_cohesion_pa(&self) -> f32 {
        self.dp_cohesion_pa
    }

    pub fn bed_compaction_tangent_pa(&self) -> f32 {
        self.bed_compaction_tangent_pa
    }

    pub fn initial_consolidation_pressure_pa(&self) -> f32 {
        self.initial_consolidation_pressure_pa
    }

    pub fn numerical_shear_modulus_pa(&self) -> f32 {
        self.numerical_shear_modulus_pa
    }

    pub fn full_compaction_solid_fraction(&self) -> f32 {
        self.full_compaction_solid_fraction
    }

    pub fn extraction_fast_rate_per_s(&self) -> f32 {
        self.extraction_fast_rate_per_s
    }

    pub fn extraction_slow_rate_per_s(&self) -> f32 {
        self.extraction_slow_rate_per_s
    }

    pub fn inlet_temperature_c(&self) -> f32 {
        self.inlet_temperature_c
    }

    pub fn inlet_temperature_k(&self) -> f32 {
        self.inlet_temperature_c + 273.15
    }

    pub fn ambient_temperature_c(&self) -> f32 {
        self.ambient_temperature_c
    }

    pub fn ambient_temperature_k(&self) -> f32 {
        self.ambient_temperature_c + 273.15
    }

    pub fn thermal_exchange_rate_per_s(&self) -> f32 {
        self.thermal_exchange_rate_per_s
    }

    pub fn reference_bed_porosity_fraction(&self) -> f32 {
        self.reference_bed_porosity_fraction
    }

    pub fn capillary_relative_permeability_exponent(&self) -> f32 {
        self.capillary_relative_permeability_exponent
    }

    pub fn capillary_mobility_floor_fraction(&self) -> f32 {
        self.capillary_mobility_floor_fraction
    }

    pub fn filter_paper_permeability_m2(&self) -> f32 {
        10.0_f32.powf(self.log10_filter_paper_permeability_m2)
    }

    pub fn filter_water_capacity_g(&self) -> f32 {
        self.filter_water_capacity_g
    }

    pub fn retained_water_g(&self, dry_coffee_g: f32) -> f32 {
        self.bed_retention_g_per_g * dry_coffee_g.max(0.0)
    }

    pub fn bound_water_capacity_g(&self, dry_coffee_g: f32) -> f32 {
        self.bound_water_capacity_g_per_g * dry_coffee_g.max(0.0)
    }

    pub fn swollen_diameter(&self, dry_diameter: f32) -> f32 {
        dry_diameter.max(0.0) * (1.0 + self.max_diameter_swelling_fraction)
    }

    pub fn absorption_fraction_after(&self, wet_contact_s: f32) -> f32 {
        first_order_fraction(self.absorption_rate_per_s, wet_contact_s)
    }

    pub fn fast_extraction_fraction_after(&self, wet_contact_s: f32) -> f32 {
        first_order_fraction(self.extraction_fast_rate_per_s, wet_contact_s)
    }

    pub fn slow_extraction_fraction_after(&self, wet_contact_s: f32) -> f32 {
        first_order_fraction(self.extraction_slow_rate_per_s, wet_contact_s)
    }

    pub fn thermal_exchange_fraction_after(&self, elapsed_s: f32) -> f32 {
        first_order_fraction(self.thermal_exchange_rate_per_s, elapsed_s)
    }
}

fn first_order_fraction(rate_per_s: f32, elapsed_s: f32) -> f32 {
    if !elapsed_s.is_finite() || elapsed_s <= 0.0 {
        return 0.0;
    }
    (1.0 - (-rate_per_s * elapsed_s).exp()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn schema_is_unique_finite_and_internally_consistent() {
        assert_eq!(CALIBRATION_SCHEMA.len(), CalibrationControl::ALL.len());
        let mut ids = HashSet::new();
        for (index, metadata) in CALIBRATION_SCHEMA.iter().enumerate() {
            assert_eq!(metadata.control as usize, index);
            assert!(ids.insert(metadata.id));
            assert!(!metadata.id.is_empty());
            assert!(!metadata.label.is_empty());
            assert!(!metadata.units.is_empty());
            assert!(metadata.min.is_finite());
            assert!(metadata.max.is_finite());
            assert!(metadata.step.is_finite() && metadata.step > 0.0);
            assert!(metadata.min < metadata.max);
            if metadata.logarithmic {
                assert!(metadata.min > 0.0);
            }
            assert_eq!(
                CalibrationControl::from_id(metadata.id),
                Some(metadata.control)
            );
        }
    }

    #[test]
    fn defaults_are_valid_without_pinning_empirical_values() {
        let profile = CalibrationProfile::default();
        profile.validate().unwrap();
        for control in CalibrationControl::ALL {
            let metadata = control.metadata();
            let value = profile.value(control);
            assert!((metadata.min..=metadata.max).contains(&value));
        }
    }

    #[test]
    fn retention_and_filter_controls_are_declared_as_rebuilds() {
        for control in [
            CalibrationControl::BedRetention,
            CalibrationControl::BoundWaterCapacity,
            CalibrationControl::MaxDiameterSwelling,
            CalibrationControl::Log10Permeability,
            CalibrationControl::ReferenceBedPorosity,
            CalibrationControl::CapillaryRelativePermeabilityExponent,
            CalibrationControl::CapillaryMobilityFloor,
            CalibrationControl::Log10FilterPaperPermeability,
            CalibrationControl::FilterWaterCapacity,
            CalibrationControl::BedCompactionTangent,
            CalibrationControl::InitialConsolidationPressure,
            CalibrationControl::NumericalShearModulus,
            CalibrationControl::FullCompactionSolidFraction,
        ] {
            assert!(control.metadata().requires_rebuild);
        }
    }

    #[test]
    fn setter_rejects_non_finite_and_unknown_values_without_mutating() {
        let mut profile = CalibrationProfile::default();
        let before = profile.clone();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                profile.set_by_id("drag_multiplier", value),
                Err(CalibrationError::NonFiniteValue { .. })
            ));
            assert_eq!(profile, before);
        }
        assert!(matches!(
            profile.set_by_id("not-a-control", 1.0),
            Err(CalibrationError::UnknownControl { .. })
        ));
        assert_eq!(profile, before);
    }

    #[test]
    fn setter_clamps_to_declared_and_cross_field_ranges() {
        let mut profile = CalibrationProfile::default();
        for control in CalibrationControl::ALL {
            let metadata = *control.metadata();
            let low = profile.set(control, metadata.min - metadata.step).unwrap();
            assert!(low.clamped);
            assert!(profile.value(control) >= metadata.min);
            let high = profile.set(control, metadata.max + metadata.step).unwrap();
            assert!(high.clamped);
            assert!(profile.value(control) <= metadata.max);
        }
        profile.validate().unwrap();

        profile
            .set(CalibrationControl::BoundWaterCapacity, 0.5)
            .unwrap();
        profile.set(CalibrationControl::BedRetention, 1.0).unwrap();
        let total = profile.bed_retention_g_per_g();
        let bound = profile
            .set(CalibrationControl::BoundWaterCapacity, f32::MAX)
            .unwrap();
        assert_eq!(bound.applied, total);
        assert!(profile.capillary_water_capacity_g_per_g() >= 0.0);
    }

    #[test]
    fn physical_transforms_are_monotonic() {
        let mut profile = CalibrationProfile::default();

        profile
            .set(CalibrationControl::Log10Permeability, -14.0)
            .unwrap();
        let low_permeability = profile.permeability_m2();
        profile
            .set(CalibrationControl::Log10Permeability, -10.0)
            .unwrap();
        assert!(profile.permeability_m2() > low_permeability);

        profile
            .set(CalibrationControl::Log10FilterPaperPermeability, -16.0)
            .unwrap();
        let low_filter_permeability = profile.filter_paper_permeability_m2();
        profile
            .set(CalibrationControl::Log10FilterPaperPermeability, -10.0)
            .unwrap();
        assert!(profile.filter_paper_permeability_m2() > low_filter_permeability);

        profile
            .set(CalibrationControl::DpFrictionAngle, 10.0)
            .unwrap();
        let low_friction = profile.dp_friction_mu();
        profile
            .set(CalibrationControl::DpFrictionAngle, 45.0)
            .unwrap();
        assert!(profile.dp_friction_mu() > low_friction);

        profile
            .set(CalibrationControl::MaxDiameterSwelling, 0.01)
            .unwrap();
        let low_swelling = profile.swollen_diameter(1.0);
        profile
            .set(CalibrationControl::MaxDiameterSwelling, 0.1)
            .unwrap();
        assert!(profile.swollen_diameter(1.0) > low_swelling);

        let early = profile.absorption_fraction_after(0.1);
        let late = profile.absorption_fraction_after(10.0);
        assert!(late > early);
        assert!((0.0..=1.0).contains(&late));

        assert!(profile.retained_water_g(20.0) > profile.retained_water_g(10.0));
        assert!(profile.inlet_temperature_k() > profile.inlet_temperature_c());
    }

    #[test]
    fn retention_and_filter_getters_follow_validated_setters() {
        let mut profile = CalibrationProfile::default();

        profile
            .set(CalibrationControl::ReferenceBedPorosity, 0.25)
            .unwrap();
        let low_porosity = profile.reference_bed_porosity_fraction();
        profile
            .set(CalibrationControl::ReferenceBedPorosity, 0.75)
            .unwrap();
        assert!(profile.reference_bed_porosity_fraction() > low_porosity);

        profile
            .set(
                CalibrationControl::CapillaryRelativePermeabilityExponent,
                1.0,
            )
            .unwrap();
        let low_exponent = profile.capillary_relative_permeability_exponent();
        profile
            .set(
                CalibrationControl::CapillaryRelativePermeabilityExponent,
                6.0,
            )
            .unwrap();
        assert!(profile.capillary_relative_permeability_exponent() > low_exponent);

        profile
            .set(CalibrationControl::CapillaryMobilityFloor, 0.001)
            .unwrap();
        let low_mobility = profile.capillary_mobility_floor_fraction();
        profile
            .set(CalibrationControl::CapillaryMobilityFloor, 0.1)
            .unwrap();
        assert!(profile.capillary_mobility_floor_fraction() > low_mobility);

        profile
            .set(CalibrationControl::FilterWaterCapacity, 0.5)
            .unwrap();
        let low_capacity = profile.filter_water_capacity_g();
        profile
            .set(CalibrationControl::FilterWaterCapacity, 5.0)
            .unwrap();
        assert!(profile.filter_water_capacity_g() > low_capacity);
    }
}

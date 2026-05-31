#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConservationLedger {
    pub emitted_water_mass: f32,
    pub active_water_mass: f32,
    pub cup_water_mass: f32,
    pub coffee_retained_water_mass: f32,
    pub active_solute_mass: f32,
    pub cup_solute_mass: f32,
    pub coffee_fast_solute: f32,
    pub coffee_slow_solute: f32,
}

impl ConservationLedger {
    pub(crate) fn total_water_mass(self) -> f32 {
        self.active_water_mass + self.cup_water_mass + self.coffee_retained_water_mass
    }

    pub(crate) fn total_solute_mass(self) -> f32 {
        self.active_solute_mass
            + self.cup_solute_mass
            + self.coffee_fast_solute
            + self.coffee_slow_solute
    }
}

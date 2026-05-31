#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SoluteLedger {
    pub coffee_fast: f32,
    pub coffee_slow: f32,
    pub water: f32,
    pub cup: f32,
}

impl SoluteLedger {
    pub(crate) fn total(self) -> f32 {
        self.coffee_fast + self.coffee_slow + self.water + self.cup
    }
}

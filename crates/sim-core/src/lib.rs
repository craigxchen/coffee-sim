pub mod constants;
pub mod grid;
pub mod transport;
pub mod bed;
pub mod pour;
pub mod fluid;
pub mod extraction;
pub mod thermal;
pub mod co2;

// Re-export key types
pub use grid::Grid;
pub use constants::*;

//! coffee_sim_core: PyO3 module entry point.
//!
//! Exposes the Rust computational core to Python via PyO3.

mod utils;
mod grid;
mod particles;
mod fluid;
mod extraction;
mod thermodynamics;

use pyo3::prelude::*;

/// Coffee Simulation Core — Rust computational engine.
#[pymodule]
fn coffee_sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<grid::SimulationGrid>()?;
    m.add_class::<fluid::FluidSolver>()?;
    m.add_class::<extraction::ExtractionSolver>()?;
    m.add_class::<thermodynamics::ThermalSolver>()?;
    m.add_class::<particles::BedGenerator>()?;
    Ok(())
}

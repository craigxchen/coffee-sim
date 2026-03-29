use pyo3::prelude::*;

/// Coffee Simulation Python bindings.
#[pymodule]
fn coffee_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Will be populated after sim-core integration
    Ok(())
}

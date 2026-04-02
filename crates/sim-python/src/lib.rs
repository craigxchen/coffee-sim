use coffee_sim_core::ParticleSim;
use pyo3::prelude::*;

const SPOUT_Y: f32 = 5.9;
const PARTICLES_PER_ML: f32 = 80.0;

#[pyclass]
struct PySim {
    inner: ParticleSim,
}

#[pymethods]
impl PySim {
    #[new]
    fn new() -> Self {
        Self {
            inner: ParticleSim::new(Default::default()),
        }
    }

    fn step(&mut self, frame_time: f32, pour_x: f32, pour_rate: f32) -> PyResult<()> {
        let emit_count = if pour_rate > 0.0 {
            (pour_rate * PARTICLES_PER_ML * frame_time).ceil() as usize
        } else {
            0
        };
        self.inner
            .step_frame(frame_time, pour_x, SPOUT_Y, emit_count);
        Ok(())
    }

    fn particle_count(&self) -> usize {
        self.inner.particle_count()
    }
}

#[pymodule]
fn coffee_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySim>()?;
    Ok(())
}

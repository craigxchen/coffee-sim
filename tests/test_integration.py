"""Integration tests running the full simulation pipeline."""

from python.config import PouroverParams, EspressoParams, GrindProfile
from python.sim import Simulation


def test_pourover_simulation():
    """Full pourover simulation runs without error and produces reasonable results."""
    params = PouroverParams(
        coffee_mass_g=15.0,
        water_mass_g=240.0,
        water_temp_c=93.0,
        bloom_water_g=40.0,
        bloom_time_s=30.0,
        pour_rate_ml_s=4.0,
        pour_pattern="center",
        geometry="v60",
        grind=GrindProfile(d_main_um=700, fines_fraction=0.15),
        grid_nx=10,
        grid_ny=10,
        grid_nz=8,
        grid_dx=0.002,
    )

    sim = Simulation(params, seed=42)
    result = sim.run(max_time=5.0, dt=0.1)

    assert len(result.time) > 0, "Should have time steps"
    assert result.extraction_yield[-1] >= 0, "EY should be non-negative"
    assert result.concentration_field is not None
    assert result.temperature_field is not None
    assert result.extraction_yield_field is not None
    assert result.porosity_field is not None


def test_espresso_simulation():
    """Full espresso simulation runs without error."""
    params = EspressoParams(
        coffee_mass_g=18.0,
        target_yield_g=36.0,
        water_temp_c=93.0,
        pressure_bar=9.0,
        preinfusion_bar=2.0,
        preinfusion_time_s=3.0,
        pressure_profile="flat",
        basket_diameter_mm=58.0,
        grind=GrindProfile(d_main_um=300, fines_fraction=0.20),
        grid_nx=10,
        grid_ny=10,
        grid_nz=8,
        grid_dx=0.001,
    )

    sim = Simulation(params, seed=42)
    result = sim.run(max_time=5.0, dt=0.05)

    assert len(result.time) > 0
    assert len(result.pressure_drop_bar) > 0
    assert result.extraction_yield[-1] >= 0


def test_declining_pressure_profile():
    """Espresso with declining pressure profile runs."""
    params = EspressoParams(
        pressure_profile="declining",
        grind=GrindProfile(d_main_um=300, fines_fraction=0.20),
        grid_nx=8,
        grid_ny=8,
        grid_nz=6,
        grid_dx=0.001,
    )

    sim = Simulation(params, seed=42)
    result = sim.run(max_time=3.0, dt=0.05)
    assert len(result.time) > 0

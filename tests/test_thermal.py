"""Tests for thermal solver."""

import numpy as np
import coffee_sim_core


def _make_grid_and_bed():
    grid = coffee_sim_core.SimulationGrid(10, 10, 10, 0.001)
    grind = {"d_main_um": 700.0, "fines_fraction": 0.15}
    coffee_sim_core.BedGenerator.generate("kalita", grind, grid, 42)
    return grid


def test_thermal_solver_creation():
    """ThermalSolver initializes with specified temperature."""
    grid = _make_grid_and_bed()
    thermal = coffee_sim_core.ThermalSolver(grid, 93.0)
    avg_t = thermal.avg_temperature_celsius(grid)
    assert abs(avg_t - 93.0) < 1.0, f"Initial temp should be ~93°C, got {avg_t}"


def test_thermal_cooling():
    """Temperature should decrease over time with no fresh hot water."""
    grid = _make_grid_and_bed()
    thermal = coffee_sim_core.ThermalSolver(grid, 93.0)

    vx = np.zeros((10, 10, 10))
    vy = np.zeros((10, 10, 10))
    vz = np.zeros((10, 10, 10))

    # Step with no flow, inlet at lower temp
    for _ in range(100):
        thermal.step(0.1, vx, vy, vz, 80.0, grid)

    avg_t = thermal.avg_temperature_celsius(grid)
    assert avg_t < 93.0, f"Temperature should decrease, got {avg_t}°C"


def test_temperature_field_shape():
    """Temperature field should have correct shape."""
    grid = _make_grid_and_bed()
    thermal = coffee_sim_core.ThermalSolver(grid, 93.0)
    temp = np.array(thermal.temperature_field())
    assert temp.shape == (10, 10, 10)
    # Values should be in Kelvin (~366 K)
    assert temp.mean() > 300

    temp_c = np.array(thermal.temperature_field_celsius())
    assert temp_c.shape == (10, 10, 10)
    assert temp_c.mean() > 50


def test_thermal_with_flow():
    """Hot inlet water should maintain temperature with flow."""
    grid = _make_grid_and_bed()
    thermal = coffee_sim_core.ThermalSolver(grid, 93.0)

    fluid = coffee_sim_core.FluidSolver(grid)
    fluid.solve_pressure(grid, {"p_top": 500.0, "p_bottom": 0.0})
    vx, vy, vz = fluid.get_velocity()

    initial_t = thermal.avg_temperature_celsius(grid)

    for _ in range(20):
        thermal.step(0.1, vx, vy, vz, 93.0, grid)

    final_t = thermal.avg_temperature_celsius(grid)
    # With hot inlet and flow, temperature should stay relatively warm
    assert final_t > 80.0, f"Temperature dropped too much: {final_t}°C"

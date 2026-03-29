"""Tests for extraction kinetics."""

import numpy as np
import coffee_sim_core


def _make_grid_and_bed():
    grid = coffee_sim_core.SimulationGrid(10, 10, 10, 0.001)
    grind = {"d_main_um": 700.0, "fines_fraction": 0.15}
    coffee_sim_core.BedGenerator.generate("kalita", grind, grid, 42)
    return grid


def test_extraction_solver_creation():
    """ExtractionSolver can be created."""
    grid = _make_grid_and_bed()
    params = {"coffee_mass_g": 20.0}
    solver = coffee_sim_core.ExtractionSolver(grid, params)
    tds = solver.outflow_tds(grid)
    assert tds == 0.0, "Initial TDS should be 0"


def test_extraction_increases_tds():
    """After stepping, TDS should increase from zero."""
    grid = _make_grid_and_bed()
    params = {"coffee_mass_g": 20.0}
    ext = coffee_sim_core.ExtractionSolver(grid, params)

    # Create velocity field (downward flow)
    fluid = coffee_sim_core.FluidSolver(grid)
    bc = {"p_top": 500.0, "p_bottom": 0.0}
    fluid.solve_pressure(grid, bc)
    vx, vy, vz = fluid.get_velocity()

    # Temperature at 93°C = 366.15 K
    temp = np.full((10, 10, 10), 366.15)
    temp_arr = temp.astype(np.float64)

    # Step extraction multiple times
    for _ in range(10):
        ext.step(0.1, vx, vy, vz, temp_arr, grid)

    tds = ext.outflow_tds(grid)
    ey = ext.total_extraction_yield()
    assert ey > 0, f"Extraction yield should be > 0 after 10 steps, got {ey}"


def test_extraction_yield_field():
    """Extraction yield field should have non-negative values."""
    grid = _make_grid_and_bed()
    params = {"coffee_mass_g": 20.0}
    ext = coffee_sim_core.ExtractionSolver(grid, params)

    fluid = coffee_sim_core.FluidSolver(grid)
    fluid.solve_pressure(grid, {"p_top": 500.0, "p_bottom": 0.0})
    vx, vy, vz = fluid.get_velocity()
    temp = np.full((10, 10, 10), 366.15)

    ext.step(0.1, vx, vy, vz, temp, grid)

    ey_field = np.array(ext.extraction_yield_field())
    assert ey_field.shape == (10, 10, 10)
    assert (ey_field >= 0).all()
    assert (ey_field <= 1).all()


def test_co2_decays():
    """CO₂ field should decay over time."""
    grid = _make_grid_and_bed()
    params = {"coffee_mass_g": 20.0, "co2_content": 0.01}
    ext = coffee_sim_core.ExtractionSolver(grid, params)

    co2_initial = np.array(ext.co2_field()).sum()

    fluid = coffee_sim_core.FluidSolver(grid)
    fluid.solve_pressure(grid, {"p_top": 500.0, "p_bottom": 0.0})
    vx, vy, vz = fluid.get_velocity()
    temp = np.full((10, 10, 10), 366.15)

    # Step many times
    for _ in range(50):
        ext.step(0.1, vx, vy, vz, temp, grid)

    co2_final = np.array(ext.co2_field()).sum()
    assert co2_final < co2_initial, "CO₂ should decay"

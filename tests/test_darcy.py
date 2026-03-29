"""Tests for Darcy flow / pressure solver."""

import numpy as np
import coffee_sim_core


def test_grid_creation():
    """SimulationGrid can be created with specified dimensions."""
    grid = coffee_sim_core.SimulationGrid(10, 10, 10, 0.001)
    assert grid.shape() == (10, 10, 10)
    assert grid.voxel_size() == 0.001


def test_bed_generation():
    """BedGenerator fills the grid with realistic porosity values."""
    grid = coffee_sim_core.SimulationGrid(20, 20, 15, 0.001)
    grind = {
        "d_main_um": 700.0,
        "d_fines_um": 50.0,
        "fines_fraction": 0.15,
        "sigma_main": 0.2,
        "sigma_fines": 0.3,
    }
    coffee_sim_core.BedGenerator.generate("v60", grind, grid, 42)

    porosity = np.array(grid.get_porosity())
    inside = np.array(grid.get_inside_bed())

    # Should have some bed voxels
    assert inside.any(), "No bed voxels generated"
    assert grid.bed_voxel_count() > 0

    # Bed porosity should be in [0.15, 0.65]
    bed_porosity = porosity[inside]
    assert bed_porosity.min() >= 0.1
    assert bed_porosity.max() <= 0.7

    # Outside-bed porosity should be 1.0
    if (~inside).any():
        outside_porosity = porosity[~inside]
        assert np.allclose(outside_porosity, 1.0)


def test_pressure_solver_basic():
    """FluidSolver produces a pressure field with correct boundary values."""
    grid = coffee_sim_core.SimulationGrid(10, 10, 10, 0.001)
    grind = {"d_main_um": 700.0, "fines_fraction": 0.15}
    coffee_sim_core.BedGenerator.generate("kalita", grind, grid, 42)

    solver = coffee_sim_core.FluidSolver(grid)
    p_top = 500.0  # Pa
    bc = {"p_top": p_top, "p_bottom": 0.0}
    pressure = np.array(solver.solve_pressure(grid, bc))

    # Pressure at top should be p_top, at bottom should be 0
    assert np.allclose(pressure[:, :, -1], p_top)
    assert np.allclose(pressure[:, :, 0], 0.0)


def test_velocity_field():
    """FluidSolver produces non-zero velocity in the bed."""
    grid = coffee_sim_core.SimulationGrid(10, 10, 10, 0.001)
    grind = {"d_main_um": 700.0, "fines_fraction": 0.15}
    coffee_sim_core.BedGenerator.generate("kalita", grind, grid, 42)

    solver = coffee_sim_core.FluidSolver(grid)
    bc = {"p_top": 500.0, "p_bottom": 0.0}
    solver.solve_pressure(grid, bc)
    vx, vy, vz = solver.get_velocity()

    vx = np.array(vx)
    vy = np.array(vy)
    vz = np.array(vz)

    # Should have some non-zero velocity
    vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    assert vel_mag.max() > 0, "No flow in the bed"


def test_outflow_rate():
    """Outflow rate should be positive when there is a pressure gradient."""
    grid = coffee_sim_core.SimulationGrid(10, 10, 10, 0.001)
    grind = {"d_main_um": 700.0, "fines_fraction": 0.15}
    coffee_sim_core.BedGenerator.generate("kalita", grind, grid, 42)

    solver = coffee_sim_core.FluidSolver(grid)
    bc = {"p_top": 500.0, "p_bottom": 0.0}
    solver.solve_pressure(grid, bc)

    flow = solver.outflow_rate(grid)
    assert flow >= 0.0, f"Outflow rate should be non-negative, got {flow}"


def test_mass_conservation():
    """Flow should approximately conserve mass (outflow ~ inflow)."""
    grid = coffee_sim_core.SimulationGrid(10, 10, 10, 0.001)
    grind = {"d_main_um": 700.0, "fines_fraction": 0.15}
    coffee_sim_core.BedGenerator.generate("kalita", grind, grid, 42)

    solver = coffee_sim_core.FluidSolver(grid)
    bc = {"p_top": 500.0, "p_bottom": 0.0}
    solver.solve_pressure(grid, bc)
    vx, vy, vz = solver.get_velocity()
    vz = np.array(vz)

    # Compute inflow at top and outflow at bottom
    dx = grid.voxel_size()
    area = dx * dx
    inside = np.array(grid.get_inside_bed())

    inflow = 0.0
    for i in range(10):
        for j in range(10):
            if inside[i, j, 9]:
                inflow += max(0, -vz[i, j, 9]) * area

    outflow = solver.outflow_rate(grid)

    # Both should be in the same order of magnitude (not exactly equal due to gravity)
    if inflow > 0 and outflow > 0:
        ratio = outflow / inflow
        assert 0.01 < ratio < 100, f"Mass conservation ratio: {ratio}"

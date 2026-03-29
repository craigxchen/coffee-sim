"""Tests for physics features: Ergun equation, bed compression, fines migration,
CO2 impedance, cup TDS tracking, and pour patterns."""

import numpy as np
import coffee_sim_core
from python.config import PouroverParams, EspressoParams, GrindProfile
from python.sim import Simulation, _make_pour_mask


def _make_grid():
    grid = coffee_sim_core.SimulationGrid(10, 10, 8, 0.001)
    grind = {"d_main_um": 700.0, "fines_fraction": 0.15}
    coffee_sim_core.BedGenerator.generate("kalita", grind, grid, 42)
    return grid


def test_ergun_higher_pressure_drop():
    """Ergun equation adds inertial resistance, reducing max velocity vs pure Darcy."""
    grid = _make_grid()
    bc = {"p_top": 500.0, "p_bottom": 0.0}

    # Solve with Darcy only (ergun=False)
    solver_darcy = coffee_sim_core.FluidSolver(grid)
    solver_darcy.set_ergun(False)
    solver_darcy.solve_pressure(grid, bc)
    vx_d, vy_d, vz_d = solver_darcy.get_velocity()
    vel_darcy = np.sqrt(np.array(vx_d)**2 + np.array(vy_d)**2 + np.array(vz_d)**2)
    max_vel_darcy = vel_darcy.max()

    # Solve with Ergun (adds inertial term)
    solver_ergun = coffee_sim_core.FluidSolver(grid)
    solver_ergun.set_ergun(True)
    solver_ergun.solve_pressure(grid, bc)
    vx_e, vy_e, vz_e = solver_ergun.get_velocity()
    vel_ergun = np.sqrt(np.array(vx_e)**2 + np.array(vy_e)**2 + np.array(vz_e)**2)
    max_vel_ergun = vel_ergun.max()

    assert max_vel_darcy > 0, "Darcy solver should produce nonzero velocity"
    assert max_vel_ergun > 0, "Ergun solver should produce nonzero velocity"
    assert max_vel_ergun < max_vel_darcy, (
        f"Ergun max velocity ({max_vel_ergun:.6e}) should be lower than "
        f"Darcy max velocity ({max_vel_darcy:.6e}) due to added inertial resistance"
    )


def test_compression_reduces_porosity():
    """Applying bed compression with high pressure should decrease porosity."""
    grid = _make_grid()
    porosity_before = np.array(grid.get_porosity()).copy()
    inside = np.array(grid.get_inside_bed())

    # Apply a high uniform pressure field
    pressure_field = np.full((10, 10, 8), 9e5, dtype=np.float64)  # 9 bar in Pa
    alpha = 1e-7  # Pa^-1, compressibility coefficient
    grid.apply_compression(pressure_field, alpha)

    porosity_after = np.array(grid.get_porosity())

    # Within the bed, porosity should have decreased
    bed_before = porosity_before[inside]
    bed_after = porosity_after[inside]

    assert bed_after.mean() < bed_before.mean(), (
        f"Mean bed porosity should decrease after compression: "
        f"before={bed_before.mean():.4f}, after={bed_after.mean():.4f}"
    )
    # Porosity should still be physically reasonable (> 0)
    assert bed_after.min() > 0.0, "Porosity should remain positive after compression"


def test_fines_migration():
    """Fines migration should keep porosity in a reasonable range."""
    grid = _make_grid()
    inside = np.array(grid.get_inside_bed())
    porosity_before = np.array(grid.get_porosity()).copy()
    initial_bed_porosity_sum = porosity_before[inside].sum()

    # Create a simple downward velocity field
    shape = (10, 10, 8)
    vx = np.zeros(shape, dtype=np.float64)
    vy = np.zeros(shape, dtype=np.float64)
    vz = np.full(shape, -0.01, dtype=np.float64)  # downward flow

    coffee_sim_core.BedGenerator.migrate_fines(grid, vx, vy, vz, 0.1)

    porosity_after = np.array(grid.get_porosity())
    bed_porosity_after = porosity_after[inside]

    # Porosity values should still be physically reasonable
    assert bed_porosity_after.min() >= 0.05, (
        f"Porosity should remain reasonable after fines migration, "
        f"got min={bed_porosity_after.min():.4f}"
    )
    assert bed_porosity_after.max() <= 0.95, (
        f"Porosity should not exceed 0.95 after fines migration, "
        f"got max={bed_porosity_after.max():.4f}"
    )


def test_co2_impedance():
    """CO2 gas fraction should reduce effective porosity via extraction solver sync."""
    grid = _make_grid()
    inside = np.array(grid.get_inside_bed())

    # Create extraction solver with high CO2 content
    params = {"coffee_mass_g": 20.0, "co2_content": 0.05}
    ext = coffee_sim_core.ExtractionSolver(grid, params)

    porosity_before = np.array(grid.get_porosity()).copy()

    # Sync CO2 from extraction solver to grid, then apply impedance
    ext.sync_co2_to_grid(grid)
    grid.apply_co2_impedance()

    porosity_after = np.array(grid.get_porosity())

    bed_before = porosity_before[inside]
    bed_after = porosity_after[inside]

    assert bed_after.mean() < bed_before.mean(), (
        f"CO2 impedance should reduce effective porosity: "
        f"before={bed_before.mean():.4f}, after={bed_after.mean():.4f}"
    )


def test_cup_tds_tracking():
    """A short pourover simulation should produce non-negative cup TDS values."""
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

    assert len(result.cup_tds) > 0, "Should have cup TDS measurements"
    for i, tds_val in enumerate(result.cup_tds):
        assert tds_val >= 0.0, (
            f"Cup TDS at step {i} should be >= 0, got {tds_val}"
        )


def test_pour_patterns():
    """Pour pattern masks should be properly shaped and normalized."""
    nx, ny = 20, 20
    dx = 0.001

    for pattern in ("center", "spiral", "pulse"):
        mask = _make_pour_mask(pattern, nx, ny, t=1.0, dx=dx)

        assert mask.shape == (nx, ny), (
            f"Mask for '{pattern}' should have shape ({nx}, {ny}), got {mask.shape}"
        )
        assert mask.min() >= 0.0, (
            f"Mask for '{pattern}' should have no negative values"
        )
        # Non-zero masks should sum to ~1 (normalized)
        if mask.sum() > 0:
            assert abs(mask.sum() - 1.0) < 1e-10, (
                f"Mask for '{pattern}' should sum to 1.0, got {mask.sum()}"
            )

    # Center mask should peak in the middle
    center_mask = _make_pour_mask("center", nx, ny, t=0.0, dx=dx)
    mid = nx // 2
    assert center_mask[mid, mid] > center_mask[0, 0], (
        "Center pour mask should peak near the center"
    )

    # Pulse mask during pause phase (t in [3,5) of cycle) should be all zeros
    pause_mask = _make_pour_mask("pulse", nx, ny, t=3.5, dx=dx)
    assert pause_mask.sum() == 0.0, (
        "Pulse pour mask should be zero during the pause phase"
    )

    # Spiral mask should have its peak offset from center (at t>0)
    spiral_mask = _make_pour_mask("spiral", nx, ny, t=2.0, dx=dx)
    peak_idx = np.unravel_index(spiral_mask.argmax(), spiral_mask.shape)
    center_idx = (nx // 2, ny // 2)
    assert peak_idx != center_idx, (
        "Spiral pour mask peak should be offset from center at t>0"
    )

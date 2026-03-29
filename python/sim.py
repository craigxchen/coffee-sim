"""High-level Simulation class that orchestrates Rust solvers."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import coffee_sim_core

from .config import PouroverParams, EspressoParams


@dataclass
class SimulationResult:
    """Results from a completed simulation."""
    time: list[float] = field(default_factory=list)
    tds: list[float] = field(default_factory=list)
    cup_tds: list[float] = field(default_factory=list)
    extraction_yield: list[float] = field(default_factory=list)
    flow_rate_ml_s: list[float] = field(default_factory=list)
    avg_temperature_c: list[float] = field(default_factory=list)
    pressure_drop_bar: list[float] = field(default_factory=list)
    total_water_collected_g: list[float] = field(default_factory=list)

    # Final state fields (3D numpy arrays)
    concentration_field: Optional[np.ndarray] = None
    temperature_field: Optional[np.ndarray] = None
    extraction_yield_field: Optional[np.ndarray] = None
    porosity_field: Optional[np.ndarray] = None

    # Optional snapshots for animation
    snapshots: list[np.ndarray] = field(default_factory=list)


def _make_pour_mask(pattern: str, nx: int, ny: int, t: float, dx: float) -> np.ndarray:
    """Generate a 2D pour-pattern mask for the top surface.

    Returns a (nx, ny) array normalized so that sum = 1 (within nonzero region).
    """
    cx, cy = nx / 2.0, ny / 2.0
    mask = np.zeros((nx, ny), dtype=np.float64)

    if pattern == "center":
        sigma = max(2.0, min(nx, ny) * 0.1)
        for i in range(nx):
            for j in range(ny):
                r2 = (i - cx + 0.5) ** 2 + (j - cy + 0.5) ** 2
                mask[i, j] = np.exp(-r2 / (2 * sigma**2))

    elif pattern == "spiral":
        period = 4.0  # seconds per full rotation
        max_radius = min(nx, ny) * 0.35
        angle = (t / period) * 2 * np.pi
        # Radius oscillates from center to edge
        radius = max_radius * (0.5 + 0.5 * np.sin(t / period * np.pi))
        px = cx + radius * np.cos(angle)
        py = cy + radius * np.sin(angle)
        sigma = max(2.0, min(nx, ny) * 0.12)
        for i in range(nx):
            for j in range(ny):
                r2 = (i - px + 0.5) ** 2 + (j - py + 0.5) ** 2
                mask[i, j] = np.exp(-r2 / (2 * sigma**2))

    elif pattern == "pulse":
        # 3s pour, 2s pause cycle
        cycle_time = t % 5.0
        if cycle_time < 3.0:
            # Pour phase: center pattern
            sigma = max(2.0, min(nx, ny) * 0.15)
            for i in range(nx):
                for j in range(ny):
                    r2 = (i - cx + 0.5) ** 2 + (j - cy + 0.5) ** 2
                    mask[i, j] = np.exp(-r2 / (2 * sigma**2))
        # else: mask stays zero (pause)

    else:
        # Default: uniform
        mask[:, :] = 1.0

    total = mask.sum()
    if total > 0:
        mask /= total
    return mask


class Simulation:
    """Run a coffee extraction simulation."""

    def __init__(self, params: PouroverParams | EspressoParams, seed: int = 42):
        self.params = params
        self.seed = seed
        self.is_espresso = isinstance(params, EspressoParams)

        # Create grid
        self.grid = coffee_sim_core.SimulationGrid(
            params.grid_nx, params.grid_ny, params.grid_nz, params.grid_dx
        )

        # Generate bed
        geometry = "espresso" if self.is_espresso else getattr(params, "geometry", "v60")
        coffee_sim_core.BedGenerator.generate(
            geometry, params.grind.to_dict(), self.grid, seed
        )

        # Create solvers
        extraction_params = {
            "coffee_mass_g": params.coffee_mass_g,
            "fast_fraction": 0.21,
            "slow_fraction": 0.09,
            "co2_content": 0.01,
        }
        self.fluid = coffee_sim_core.FluidSolver(self.grid)
        self.extraction = coffee_sim_core.ExtractionSolver(self.grid, extraction_params)
        self.thermal = coffee_sim_core.ThermalSolver(self.grid, params.water_temp_c)

        if self.is_espresso:
            self.fluid.set_ergun(True)

    def run(self, max_time: Optional[float] = None, dt: Optional[float] = None,
            callback=None, snapshot_interval: int = 0) -> SimulationResult:
        """Run the simulation to completion.

        Args:
            max_time: Maximum simulation time in seconds.
            dt: Timestep in seconds.
            callback: Optional callable(time, result) called each timestep.
            snapshot_interval: Save concentration snapshot every N steps (0 = disabled).

        Returns:
            SimulationResult with time series and final state.
        """
        result = SimulationResult()

        if self.is_espresso:
            return self._run_espresso(result, max_time, dt, callback, snapshot_interval)
        else:
            return self._run_pourover(result, max_time, dt, callback, snapshot_interval)

    def _run_pourover(self, result, max_time, dt, callback, snapshot_interval):
        params: PouroverParams = self.params
        max_time = max_time or 240.0
        dt_max = dt or 0.05
        dt = dt_max

        water_density = 971.8
        gravity = 9.81
        nx, ny = params.grid_nx, params.grid_ny

        total_water_poured_g = 0.0
        cum_water_out_g = 0.0
        cum_extracted_g = 0.0
        t = 0.0
        step_count = 0

        while t < max_time and total_water_poured_g < params.water_mass_g:
            # 1. CFL timestep
            max_v = self.fluid.max_velocity(self.grid)
            if max_v > 0:
                dt = min(dt_max, 0.4 * params.grid_dx / max_v)
            else:
                dt = dt_max

            # 2. Pour state
            if t < params.bloom_time_s:
                if total_water_poured_g < params.bloom_water_g:
                    pour = min(params.pour_rate_ml_s * 2.0 * dt,
                               params.bloom_water_g - total_water_poured_g)
                    total_water_poured_g += pour
            else:
                remaining = params.water_mass_g - total_water_poured_g
                total_water_poured_g += min(params.pour_rate_ml_s * dt, remaining)

            # 3. CO₂ impedance (bloom)
            self.extraction.sync_co2_to_grid(self.grid)
            self.grid.apply_co2_impedance()

            # 4. Compute top pressure with pour pattern
            bed_area_m2 = (nx * params.grid_dx) ** 2 * 0.5
            water_above_g = max(0.0, total_water_poured_g - params.coffee_mass_g * 2.0)
            water_height_m = min((water_above_g * 1e-3) / (water_density * bed_area_m2), 0.10)
            p_top = water_density * gravity * water_height_m

            mask = _make_pour_mask(params.pour_pattern, nx, ny, t, params.grid_dx)
            # Scale: voxels under the pour stream get full pressure, others get reduced
            # Blend between uniform and focused: 50% uniform + 50% patterned
            uniform = np.ones((nx, ny)) / (nx * ny)
            blended = 0.5 * uniform + 0.5 * mask
            blended /= blended.max()  # normalize so max = 1
            # Average p_top weighted by pattern (all voxels still get some pressure)
            effective_p_top = p_top * blended.mean() / uniform.mean()

            bc = {"p_top": float(effective_p_top), "p_bottom": 0.0}

            # 5. Solve pressure/velocity
            self.fluid.solve_pressure(self.grid, bc)
            vel_x, vel_y, vel_z = self.fluid.get_velocity()

            # 6. Step thermal
            temp_field = self.thermal.temperature_field()
            self.thermal.step(dt, vel_x, vel_y, vel_z, params.water_temp_c, self.grid)

            # 7. Step extraction
            self.extraction.step(dt, vel_x, vel_y, vel_z, temp_field, self.grid)

            # 8. Record metrics
            t += dt
            step_count += 1
            inst_tds = self.extraction.outflow_tds(self.grid)
            flow_m3_s = self.fluid.outflow_rate(self.grid)
            flow_ml_s = flow_m3_s * 1e6

            # Cup TDS tracking
            water_this_step_g = flow_ml_s * dt * 0.971
            cum_water_out_g += water_this_step_g
            cum_extracted_g += (inst_tds / 100.0) * water_this_step_g
            cup_tds = (cum_extracted_g / cum_water_out_g * 100.0) if cum_water_out_g > 0 else 0.0

            result.time.append(t)
            result.tds.append(inst_tds)
            result.cup_tds.append(cup_tds)
            result.extraction_yield.append(self.extraction.total_extraction_yield())
            result.flow_rate_ml_s.append(flow_ml_s)
            result.avg_temperature_c.append(self.thermal.avg_temperature_celsius(self.grid))
            result.total_water_collected_g.append(cum_water_out_g)

            if snapshot_interval > 0 and step_count % snapshot_interval == 0:
                result.snapshots.append(np.array(self.extraction.concentration_field()))

            if callback:
                callback(t, result)

        # Final fields
        result.concentration_field = np.array(self.extraction.concentration_field())
        result.temperature_field = np.array(self.thermal.temperature_field_celsius())
        result.extraction_yield_field = np.array(self.extraction.extraction_yield_field())
        result.porosity_field = np.array(self.grid.get_porosity())

        return result

    def _run_espresso(self, result, max_time, dt, callback, snapshot_interval):
        params: EspressoParams = self.params
        max_time = max_time or 30.0
        dt_max = dt or 0.01
        dt = dt_max

        total_yield_g = 0.0
        cum_water_out_g = 0.0
        cum_extracted_g = 0.0
        t = 0.0
        step_count = 0
        alpha = getattr(params, "compressibility_alpha", 1e-7)

        while t < max_time and total_yield_g < params.target_yield_g:
            # 1. CFL
            max_v = self.fluid.max_velocity(self.grid)
            if max_v > 0:
                dt = min(dt_max, 0.4 * params.grid_dx / max_v)
            else:
                dt = dt_max

            # 2. Pressure profile
            if t < params.preinfusion_time_s:
                pressure_pa = params.preinfusion_bar * 1e5
            else:
                if params.pressure_profile == "declining":
                    elapsed = t - params.preinfusion_time_s
                    pressure_pa = params.pressure_bar * 1e5 * max(0.5, 1.0 - 0.02 * elapsed)
                elif params.pressure_profile == "ramp":
                    elapsed = t - params.preinfusion_time_s
                    pressure_pa = params.pressure_bar * 1e5 * min(1.0, 0.5 + 0.1 * elapsed)
                else:  # "flat"
                    pressure_pa = params.pressure_bar * 1e5

            # 3. Solve pressure with Ergun + bed compression
            bc = {"p_top": pressure_pa, "p_bottom": 0.0}
            self.fluid.solve_pressure(self.grid, bc)
            vel_x, vel_y, vel_z = self.fluid.get_velocity()

            # Bed compression feedback (solve → compress → re-solve)
            if alpha > 0:
                pressure_field = np.full(
                    (params.grid_nx, params.grid_ny, params.grid_nz),
                    pressure_pa, dtype=np.float64
                )
                self.grid.apply_compression(pressure_field, alpha)
                self.fluid.solve_pressure(self.grid, bc)
                vel_x, vel_y, vel_z = self.fluid.get_velocity()

            # 4. Step thermal
            temp_field = self.thermal.temperature_field()
            self.thermal.step(dt, vel_x, vel_y, vel_z, params.water_temp_c, self.grid)

            # 5. Step extraction
            self.extraction.step(dt, vel_x, vel_y, vel_z, temp_field, self.grid)

            # 6. Fines migration (every 20 steps)
            if step_count > 0 and step_count % 20 == 0:
                coffee_sim_core.BedGenerator.migrate_fines(
                    self.grid, vel_x, vel_y, vel_z, dt * 20
                )

            # 7. Track yield and cup TDS
            flow_m3_s = self.fluid.outflow_rate(self.grid)
            flow_ml_s = flow_m3_s * 1e6
            water_this_step_g = flow_ml_s * dt * 0.971
            total_yield_g += water_this_step_g
            cum_water_out_g += water_this_step_g

            inst_tds = self.extraction.outflow_tds(self.grid)
            cum_extracted_g += (inst_tds / 100.0) * water_this_step_g
            cup_tds = (cum_extracted_g / cum_water_out_g * 100.0) if cum_water_out_g > 0 else 0.0

            t += dt
            step_count += 1
            result.time.append(t)
            result.tds.append(inst_tds)
            result.cup_tds.append(cup_tds)
            result.extraction_yield.append(self.extraction.total_extraction_yield())
            result.flow_rate_ml_s.append(flow_ml_s)
            result.avg_temperature_c.append(self.thermal.avg_temperature_celsius(self.grid))
            result.pressure_drop_bar.append(pressure_pa / 1e5)
            result.total_water_collected_g.append(cum_water_out_g)

            if snapshot_interval > 0 and step_count % snapshot_interval == 0:
                result.snapshots.append(np.array(self.extraction.concentration_field()))

            if callback:
                callback(t, result)

        # Final fields
        result.concentration_field = np.array(self.extraction.concentration_field())
        result.temperature_field = np.array(self.thermal.temperature_field_celsius())
        result.extraction_yield_field = np.array(self.extraction.extraction_yield_field())
        result.porosity_field = np.array(self.grid.get_porosity())

        return result

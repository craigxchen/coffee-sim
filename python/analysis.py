"""Post-processing and analysis of simulation results."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from .sim import Simulation, SimulationResult

if TYPE_CHECKING:
    from .config import PouroverParams, EspressoParams


def cumulative_extraction_yield(result: SimulationResult) -> np.ndarray:
    """Return cumulative extraction yield over time as a numpy array."""
    return np.array(result.extraction_yield)


def instantaneous_tds(result: SimulationResult) -> np.ndarray:
    """Return instantaneous TDS over time."""
    return np.array(result.tds)


def average_tds(result: SimulationResult) -> float:
    """Return time-averaged TDS."""
    tds = np.array(result.tds)
    return float(np.mean(tds[tds > 0])) if np.any(tds > 0) else 0.0


def identify_channeling(
    result: SimulationResult,
    vel_x: np.ndarray,
    vel_y: np.ndarray,
    vel_z: np.ndarray,
    threshold_factor: float = 3.0,
) -> np.ndarray:
    """Identify channeling: voxels with velocity > threshold_factor × median.

    Returns a boolean mask of channeling voxels.
    """
    vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    active = vel_mag > 0
    if not np.any(active):
        return np.zeros_like(vel_mag, dtype=bool)
    median_vel = np.median(vel_mag[active])
    return vel_mag > threshold_factor * median_vel


def flavor_balance(result: SimulationResult) -> float:
    """Compute flavor balance proxy.

    Returns ratio of extraction progress. Values closer to 1.0 indicate
    more balanced extraction. High values suggest over-extraction of
    slow-pool compounds (bitterness).
    """
    ey_field = result.extraction_yield_field
    if ey_field is None:
        return 0.0
    active = ey_field > 0
    if not np.any(active):
        return 0.0
    mean_ey = np.mean(ey_field[active])
    std_ey = np.std(ey_field[active])
    if mean_ey == 0:
        return 0.0
    # Uniformity: lower std/mean ratio = better balance
    return 1.0 - min(std_ey / mean_ey, 1.0)


def extraction_uniformity(result: SimulationResult) -> float:
    """Standard deviation of per-voxel extraction yield (%)."""
    ey_field = result.extraction_yield_field
    if ey_field is None:
        return 0.0
    active = ey_field > 0
    if not np.any(active):
        return 0.0
    return float(np.std(ey_field[active]) * 100)


def export_timeseries_csv(result: SimulationResult, path: str | Path) -> None:
    """Export time-series data to CSV."""
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "tds_percent", "extraction_yield_percent",
            "flow_rate_ml_s", "avg_temperature_c", "pressure_drop_bar",
        ])
        for i in range(len(result.time)):
            writer.writerow([
                f"{result.time[i]:.4f}",
                f"{result.tds[i]:.4f}",
                f"{result.extraction_yield[i]:.4f}",
                f"{result.flow_rate_ml_s[i]:.4f}",
                f"{result.avg_temperature_c[i]:.2f}",
                f"{result.pressure_drop_bar[i]:.4f}" if i < len(result.pressure_drop_bar) else "0.0",
            ])


def _set_nested_attr(obj, dotted_path: str, value):
    """Set an attribute on an object using a dot-separated path.

    For example, _set_nested_attr(params, "grind.d_main_um", 500) sets
    params.grind.d_main_um = 500.
    """
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def run_sensitivity_sweep(
    base_params: PouroverParams | EspressoParams,
    param_name: str,
    values: list[float],
    max_time: float = 60.0,
    dt: float = 0.1,
    grid_override: dict | None = None,
    seed: int = 42,
) -> dict[float, SimulationResult]:
    """Run a parameter sweep, varying one parameter while keeping others fixed.

    Args:
        base_params: Base configuration (PouroverParams or EspressoParams).
        param_name: Dot-separated parameter path, e.g. "grind.d_main_um"
            or "water_temp_c".
        values: List of values to sweep over for the parameter.
        max_time: Maximum simulation time in seconds.
        dt: Timestep in seconds.
        grid_override: Optional dict with keys "grid_nx", "grid_ny", "grid_nz"
            to override grid resolution for faster sweeps.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping each parameter value to its SimulationResult.
    """
    results: dict[float, SimulationResult] = {}

    for val in values:
        params = copy.deepcopy(base_params)
        _set_nested_attr(params, param_name, val)

        if grid_override is not None:
            for key, grid_val in grid_override.items():
                if hasattr(params, key):
                    setattr(params, key, grid_val)

        sim = Simulation(params, seed=seed)
        result = sim.run(max_time=max_time, dt=dt)
        results[val] = result

    return results


def plot_sensitivity(
    sweep_results: dict[float, SimulationResult],
    param_name: str,
    save_path: str | None = None,
):
    """Plot overlaid EY and cup TDS curves from a sensitivity sweep.

    Args:
        sweep_results: Dictionary mapping parameter values to SimulationResults,
            as returned by run_sensitivity_sweep.
        param_name: Name of the swept parameter (used for legend labels).
        save_path: Optional path to save the figure.
    """
    fig, (ax_ey, ax_tds) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Sensitivity: {param_name}", fontsize=13)

    cmap = plt.cm.viridis
    vals = sorted(sweep_results.keys())
    norm = plt.Normalize(vmin=min(vals), vmax=max(vals))

    for val in vals:
        result = sweep_results[val]
        color = cmap(norm(val))
        label = f"{param_name}={val:.4g}"

        ax_ey.plot(result.time, result.extraction_yield, color=color,
                   linewidth=1.3, label=label)

        tds_data = result.cup_tds if result.cup_tds else result.tds
        ax_tds.plot(result.time, tds_data, color=color,
                    linewidth=1.3, label=label)

    ax_ey.set_xlabel("Time (s)")
    ax_ey.set_ylabel("Extraction Yield (%)")
    ax_ey.set_title("Extraction Yield")
    ax_ey.legend(fontsize=7, loc="lower right")
    ax_ey.grid(True, alpha=0.3)

    ax_tds.set_xlabel("Time (s)")
    ax_tds.set_ylabel("Cup TDS (%)")
    ax_tds.set_title("Cup TDS")
    ax_tds.legend(fontsize=7, loc="upper right")
    ax_tds.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

#!/usr/bin/env python3
"""Fit thermal model parameters against Decent basket temperature data.

Compares 1D thermal model prediction against measured basket temperature.
Fits: [h_wall, k_thermal_factor]

Usage:
    python calibration/analysis/fit_thermal.py [--input calibration/curated/all_shots.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from common import (
    BASKET_AREA,
    BASKET_DIAMETER,
    COFFEE_DENSITY,
    BASE_POROSITY,
    T_AMBIENT,
    WATER_DENSITY,
    viscosity,
)

# Thermal constants
WATER_CP = 4186.0           # J/(kg·K)
THERMAL_DIFFUSIVITY = 1.5e-7  # m²/s (reference)


def simulate_thermal(
    params: np.ndarray,
    shot: dict,
    n_cells: int = 20,
) -> np.ndarray:
    """Simulate basket temperature using 1D thermal model.

    Returns predicted temperature array (°C) at bottom of puck.
    """
    h_wall, k_thermal_factor = params

    dose_kg = shot["dose_g"] / 1000.0
    porosity = BASE_POROSITY
    L = dose_kg / (COFFEE_DENSITY * BASKET_AREA * (1.0 - porosity))
    dz = L / n_cells

    time_s = np.array(shot["time_s"])
    flow_ml_s = np.array(shot["flow_ml_s"])
    temp_basket_c = np.array(shot["temp_basket_c"])

    # Initialize cell temperatures from first basket reading
    t_init = temp_basket_c[0] + 273.15 if temp_basket_c[0] > 0 else 366.0
    T_cells = np.full(n_cells, t_init)

    # Perimeter for wall heat loss
    perimeter = np.pi * BASKET_DIAMETER
    wall_area_per_cell = perimeter * dz

    alpha_eff = THERMAL_DIFFUSIVITY * k_thermal_factor
    predicted_temp_c = np.zeros(len(time_s))
    predicted_temp_c[0] = T_cells[-1] - 273.15

    for i in range(1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        if dt <= 0:
            predicted_temp_c[i] = predicted_temp_c[i - 1]
            continue

        Q = flow_ml_s[i] * 1e-6  # m³/s
        v_sup = Q / BASKET_AREA

        # Inlet temperature: approximate from basket temp at start
        # (Decent heats water, so inlet ~ basket + small offset)
        T_inlet = temp_basket_c[i] + 273.15 + 2.0  # assume water is ~2°C above basket

        # Advection: upwind scheme (flow goes top to bottom, cell 0 = top)
        T_new = T_cells.copy()
        for j in range(n_cells):
            cell_volume = BASKET_AREA * dz
            liquid_volume = porosity * cell_volume

            # Advective heat transport
            T_up = T_inlet if j == 0 else T_cells[j - 1]
            advection = WATER_DENSITY * WATER_CP * v_sup * (T_up - T_cells[j]) * dt / dz

            # Diffusion
            T_left = T_inlet if j == 0 else T_cells[j - 1]
            T_right = T_cells[j + 1] if j < n_cells - 1 else T_cells[j]
            diffusion = alpha_eff * (T_left - 2 * T_cells[j] + T_right) / (dz ** 2)

            # Wall cooling (Newton's law)
            wall_cooling = h_wall * wall_area_per_cell * (T_cells[j] - T_AMBIENT) / (
                WATER_DENSITY * WATER_CP * liquid_volume
            )

            T_new[j] += (advection / (WATER_DENSITY * WATER_CP) + diffusion - wall_cooling) * dt

        T_cells[:] = T_new
        # Clamp to physical range
        T_cells = np.clip(T_cells, T_AMBIENT, 400.0)

        predicted_temp_c[i] = T_cells[-1] - 273.15  # bottom cell = basket location

    return predicted_temp_c


def objective(params: np.ndarray, shots: list[dict]) -> float:
    """Mean squared temperature error (°C²) across all shots."""
    total_error = 0.0
    valid = 0

    for shot in shots:
        try:
            pred = simulate_thermal(params, shot)
            meas = np.array(shot["temp_basket_c"])

            # Skip first 3 seconds (transient noise)
            time_s = np.array(shot["time_s"])
            mask = time_s > 3.0
            if np.sum(mask) < 10:
                continue

            error = np.mean((pred[mask] - meas[mask]) ** 2)
            total_error += error
            valid += 1
        except Exception:
            continue

    return total_error / max(valid, 1)


def main(input_path: Path):
    with open(input_path) as f:
        shots = json.load(f)

    # Filter to shots with basket temperature data
    usable = [
        s for s in shots
        if s.get("temp_basket_c")
        and len(s["temp_basket_c"]) >= 20
        and max(s["temp_basket_c"]) > 70
    ]
    print(f"Loaded {len(shots)} shots, {len(usable)} with usable basket temperature")

    if len(usable) < 10:
        print("Not enough shots with temperature data. Need >= 10.")
        sys.exit(1)

    # Use subset for fitting
    np.random.seed(42)
    indices = np.random.permutation(len(usable))
    n_fit = min(150, len(usable) * 3 // 4)
    fit_shots = [usable[i] for i in indices[:n_fit]]
    val_shots = [usable[i] for i in indices[n_fit:]]

    print(f"Fitting on {len(fit_shots)} shots, validating on {len(val_shots)}")

    bounds = [
        (1.0, 200.0),     # h_wall (W/(m²·K))
        (0.1, 10.0),      # k_thermal_factor (multiplier on thermal diffusivity)
    ]

    print("Running differential evolution...")
    result = differential_evolution(
        objective,
        bounds,
        args=(fit_shots,),
        seed=42,
        maxiter=100,
        tol=1e-4,
        disp=True,
    )

    h_wall, k_thermal = result.x
    rmse = np.sqrt(result.fun)

    print(f"\nFitted thermal parameters:")
    print(f"  H_WALL:              {h_wall:.1f} W/(m²·K)")
    print(f"  Thermal factor:      {k_thermal:.2f}")
    print(f"  RMSE (°C):           {rmse:.2f}")

    # Validate
    errors = []
    for shot in val_shots:
        try:
            pred = simulate_thermal(result.x, shot)
            meas = np.array(shot["temp_basket_c"])
            time_s = np.array(shot["time_s"])
            mask = time_s > 3.0
            if np.sum(mask) < 10:
                continue
            errors.append(np.mean(np.abs(pred[mask] - meas[mask])))
        except Exception:
            continue

    if errors:
        print(f"\nValidation ({len(errors)} shots):")
        print(f"  Median |error|: {np.median(errors):.2f} °C")
        print(f"  Mean |error|:   {np.mean(errors):.2f} °C")
        print(f"  90th pctile:    {np.percentile(errors, 90):.2f} °C")

    # Save
    out = {
        "H_WALL_FITTED": float(h_wall),
        "THERMAL_DIFFUSIVITY_FACTOR": float(k_thermal),
        "fit_rmse_c": float(rmse),
        "n_fit_shots": len(fit_shots),
        "n_val_shots": len(val_shots),
        "median_abs_error_c": float(np.median(errors)) if errors else None,
    }
    out_path = input_path.parent / "thermal_fit_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="calibration/curated/all_shots.json")
    args = parser.parse_args()
    main(Path(args.input))

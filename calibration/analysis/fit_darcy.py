#!/usr/bin/env python3
"""Fit puck flow model against Decent pressure<->flow data.

For pressure-controlled phases: impose P(t), predict Q(t).
For flow-controlled phases: impose Q(t), predict P(t).

Fits parameters: [k0_factor, porosity, fines_rate, fines_capture, d_p]
using scipy.optimize.differential_evolution.

Usage:
    python calibration/analysis/fit_darcy.py [--input calibration/curated/all_shots.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from common import PuckModel, classify_phase_control


def evaluate_shot(params: np.ndarray, shot: dict, puck: PuckModel) -> tuple[float, str]:
    """Evaluate Darcy model fit for a single shot.

    Returns (normalized_mse, control_type).
    """
    k0_factor, porosity, fines_rate, fines_capture, d_p = params
    darcy_params = {
        "k0_factor": k0_factor,
        "porosity": porosity,
        "fines_rate": fines_rate,
        "fines_capture": fines_capture,
        "d_p": d_p,
    }

    result = puck.simulate_shot(shot, darcy_params=darcy_params)

    time_s = np.array(shot["time_s"])
    pressure_bar = np.array(shot["pressure_bar"])
    flow_ml_s = np.array(shot["flow_ml_s"])
    control = result["control_modes"]

    # Evaluate per-phase
    total_error = 0.0
    n_points = 0
    dominant_mode = "unknown"

    # Collect points by control mode
    p_ctrl_idx = [i for i in range(len(time_s)) if control[i] == "pressure_controlled"]
    f_ctrl_idx = [i for i in range(len(time_s)) if control[i] == "flow_controlled"]

    if len(p_ctrl_idx) > len(f_ctrl_idx):
        dominant_mode = "pressure_controlled"
    elif len(f_ctrl_idx) > 0:
        dominant_mode = "flow_controlled"

    # Pressure-controlled: compare predicted vs measured flow
    if p_ctrl_idx:
        pred = np.array([result["predicted_flow_ml_s"][i] for i in p_ctrl_idx])
        meas = np.array([flow_ml_s[i] for i in p_ctrl_idx])
        scale = max(np.ptp(meas), 0.1)
        total_error += np.mean(((pred - meas) / scale) ** 2)
        n_points += 1

    # Flow-controlled: compare predicted vs measured pressure
    if f_ctrl_idx:
        pred = np.array([result["predicted_pressure_bar"][i] for i in f_ctrl_idx])
        meas = np.array([pressure_bar[i] for i in f_ctrl_idx])
        scale = max(np.ptp(meas), 0.1)
        total_error += np.mean(((pred - meas) / scale) ** 2)
        n_points += 1

    if n_points == 0:
        return float("inf"), "unknown"

    return total_error / n_points, dominant_mode


def objective(params: np.ndarray, shots: list[dict]) -> float:
    """Global objective: mean normalized MSE across all shots."""
    puck = PuckModel(n_cells=20)
    total = 0.0
    valid = 0

    for shot in shots:
        try:
            err, mode = evaluate_shot(params, shot, puck)
            if not np.isfinite(err):
                continue
            total += err
            valid += 1
        except Exception:
            continue

    return total / max(valid, 1)


def main(input_path: Path):
    with open(input_path) as f:
        shots = json.load(f)

    # Filter to shots with goal data (needed for phase classification)
    usable = [s for s in shots if s.get("pressure_goal_bar") or s.get("flow_goal_ml_s")]
    print(f"Loaded {len(shots)} shots, {len(usable)} with goal data for fitting")

    if len(usable) < 10:
        print("Not enough shots with goal data for fitting. Need >= 10.")
        sys.exit(1)

    # Use a subset for fitting (faster), validate on rest
    np.random.seed(42)
    indices = np.random.permutation(len(usable))
    n_fit = min(200, len(usable) * 3 // 4)
    fit_shots = [usable[i] for i in indices[:n_fit]]
    val_shots = [usable[i] for i in indices[n_fit:]]

    print(f"Fitting on {len(fit_shots)} shots, validating on {len(val_shots)}")

    bounds = [
        (0.1, 10.0),       # k0_factor (multiplier on K-C prediction)
        (0.25, 0.50),      # porosity
        (0.001, 0.1),      # fines_rate
        (0.01, 1.0),       # fines_capture_rate
        (150e-6, 500e-6),  # d_p (meters)
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
        workers=1,  # PuckModel isn't pickle-safe for multiprocessing
    )

    k0_factor, porosity, fines_rate, fines_capture, d_p = result.x
    print(f"\nFitted parameters:")
    print(f"  k0_factor:     {k0_factor:.3f}")
    print(f"  porosity:      {porosity:.3f}")
    print(f"  fines_rate:    {fines_rate:.4f}")
    print(f"  fines_capture: {fines_capture:.3f}")
    print(f"  d_p:           {d_p*1e6:.0f} μm")
    print(f"  Objective:     {result.fun:.4f}")

    # Validate on held-out set
    puck = PuckModel(n_cells=20)
    nrmses = []
    for shot in val_shots:
        try:
            err, mode = evaluate_shot(result.x, shot, puck)
            if np.isfinite(err):
                nrmses.append(np.sqrt(err))
        except Exception:
            continue

    if nrmses:
        median_nrmse = np.median(nrmses)
        print(f"\nValidation ({len(nrmses)} shots):")
        print(f"  Median NRMSE:  {median_nrmse:.3f}")
        print(f"  Mean NRMSE:    {np.mean(nrmses):.3f}")
        print(f"  90th pctile:   {np.percentile(nrmses, 90):.3f}")

    # Save fitted parameters
    out = {
        "k0_factor": float(k0_factor),
        "porosity": float(porosity),
        "fines_rate": float(fines_rate),
        "fines_capture": float(fines_capture),
        "d_p_m": float(d_p),
        "d_p_um": float(d_p * 1e6),
        "objective": float(result.fun),
        "n_fit_shots": len(fit_shots),
        "n_val_shots": len(val_shots),
        "median_nrmse_validation": float(np.median(nrmses)) if nrmses else None,
    }
    out_path = input_path.parent / "darcy_fit_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="calibration/curated/all_shots.json")
    args = parser.parse_args()
    main(Path(args.input))

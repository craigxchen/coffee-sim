#!/usr/bin/env python3
"""Fit two-pool extraction kinetics against Decent shots with measured EY.

Uses 1D coupled extraction model (PuckModel) — NOT a 0D well-mixed model.
Each of 20 puck cells tracks its own m_fast, m_slow, and local concentration.
Flow carries dissolved solubles downstream (upwind advection).

Loads Darcy parameters from darcy_fit_result.json for the flow solution,
so extraction kinetics are isolated from flow model errors.

Fits: [k_fast, k_slow, e_activation, f_fast, f_slow]

Usage:
    python calibration/analysis/fit_extraction.py [--input calibration/curated/shots_with_ey.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from common import PuckModel


def load_darcy_params(curated_dir: Path) -> dict:
    """Load fitted Darcy parameters, or return defaults if not yet fitted."""
    darcy_path = curated_dir / "darcy_fit_result.json"
    if darcy_path.exists():
        with open(darcy_path) as f:
            d = json.load(f)
        return {
            "k0_factor": d["k0_factor"],
            "porosity": d["porosity"],
            "fines_rate": d["fines_rate"],
            "fines_capture": d["fines_capture"],
            "d_p": d["d_p_m"],
        }
    print("Warning: darcy_fit_result.json not found, using defaults")
    return {
        "k0_factor": 1.0,
        "porosity": 0.35,
        "fines_rate": 0.01,
        "fines_capture": 0.1,
        "d_p": 300e-6,
    }


def simulate_ey(params: np.ndarray, shot: dict, darcy_params: dict, puck: PuckModel) -> float:
    """Simulate extraction yield for a shot using 1D puck model."""
    k_fast, k_slow, e_act, f_fast, f_slow = params

    extraction_params = {
        "k_fast": k_fast,
        "k_slow": k_slow,
        "e_act": e_act,
        "f_fast": f_fast,
        "f_slow": f_slow,
    }

    result = puck.simulate_shot(shot, darcy_params=darcy_params, extraction_params=extraction_params)
    return result["predicted_ey_pct"]


def objective(params: np.ndarray, shots: list[dict], darcy_params: dict) -> float:
    """Mean squared EY error across all shots."""
    puck = PuckModel(n_cells=20)
    total_error = 0.0
    valid = 0

    for shot in shots:
        ey_measured = shot.get("ey_pct")
        if ey_measured is None or ey_measured <= 0:
            continue

        try:
            ey_pred = simulate_ey(params, shot, darcy_params, puck)
            total_error += (ey_pred - ey_measured) ** 2
            valid += 1
        except Exception:
            continue

    return total_error / max(valid, 1)


def main(input_path: Path):
    with open(input_path) as f:
        shots = json.load(f)

    # Filter to shots with actual EY data
    shots_ey = [s for s in shots if s.get("ey_pct") and s["ey_pct"] > 0]
    print(f"Loaded {len(shots)} shots, {len(shots_ey)} with EY data")

    if len(shots_ey) < 5:
        print("Not enough shots with EY data for fitting. Need >= 5.")
        sys.exit(1)

    # Load Darcy params
    curated_dir = input_path.parent
    darcy_params = load_darcy_params(curated_dir)
    print(f"Using Darcy params: d_p={darcy_params['d_p']*1e6:.0f}μm, k0={darcy_params['k0_factor']:.2f}")

    bounds = [
        (1e-6, 1e-3),      # k_fast (m/s at T_REF)
        (1e-8, 1e-5),      # k_slow
        (40_000, 90_000),   # E_activation (J/mol)
        (0.15, 0.28),       # f_fast (solubles fraction)
        (0.05, 0.15),       # f_slow
    ]

    print("Running differential evolution...")
    result = differential_evolution(
        objective,
        bounds,
        args=(shots_ey, darcy_params),
        seed=42,
        maxiter=100,
        tol=1e-4,
        disp=True,
        workers=1,
    )

    k_fast, k_slow, e_act, f_fast, f_slow = result.x
    rmse = np.sqrt(result.fun)

    print(f"\nFitted extraction parameters:")
    print(f"  K_FAST_REF:        {k_fast:.2e} m/s")
    print(f"  K_SLOW_REF:        {k_slow:.2e} m/s")
    print(f"  E_ACTIVATION:      {e_act:.0f} J/mol")
    print(f"  SOLUBLES_FAST_FRAC: {f_fast:.3f}")
    print(f"  SOLUBLES_SLOW_FRAC: {f_slow:.3f}")
    print(f"  RMSE (EY %):       {rmse:.2f}")

    # Per-shot residuals
    puck = PuckModel(n_cells=20)
    residuals = []
    for shot in shots_ey:
        try:
            ey_pred = simulate_ey(result.x, shot, darcy_params, puck)
            ey_meas = shot["ey_pct"]
            residuals.append(ey_pred - ey_meas)
        except Exception:
            continue

    residuals = np.array(residuals)
    print(f"\nResidual statistics ({len(residuals)} shots):")
    print(f"  Mean error:   {np.mean(residuals):+.2f}%")
    print(f"  Std error:    {np.std(residuals):.2f}%")
    print(f"  Max |error|:  {np.max(np.abs(residuals)):.2f}%")
    print(f"  Correlation:  {np.corrcoef([s['ey_pct'] for s in shots_ey[:len(residuals)]], [s['ey_pct'] + r for s, r in zip(shots_ey, residuals)])[0,1]:.3f}")

    # Save
    out = {
        "K_FAST_REF": float(k_fast),
        "K_SLOW_REF": float(k_slow),
        "E_ACTIVATION": float(e_act),
        "SOLUBLES_FAST_FRAC": float(f_fast),
        "SOLUBLES_SLOW_FRAC": float(f_slow),
        "fit_rmse_ey_pct": float(rmse),
        "n_shots_with_ey": len(shots_ey),
        "mean_residual_pct": float(np.mean(residuals)),
        "max_abs_residual_pct": float(np.max(np.abs(residuals))),
    }
    out_path = input_path.parent / "extraction_fit_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="calibration/curated/shots_with_ey.json")
    args = parser.parse_args()
    main(Path(args.input))

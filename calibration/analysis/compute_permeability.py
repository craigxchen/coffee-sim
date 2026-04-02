#!/usr/bin/env python3
"""Compute instantaneous puck permeability from Decent shot data.

For each shot, derives k(t) = Q·μ·L / (A·ΔP) at each timestep.

Usage:
    python calibration/analysis/compute_permeability.py [--input calibration/curated/all_shots.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    BASKET_AREA,
    COFFEE_DENSITY,
    BASE_POROSITY,
    viscosity,
)


def compute_permeability_series(shot: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Derive instantaneous permeability k(t) from a shot's pressure and flow data.

    Returns (time_s, k_m2) arrays, or (None, None) if insufficient data.
    """
    dose_g = shot["dose_g"]
    dose_kg = dose_g / 1000.0
    porosity = BASE_POROSITY

    # Puck thickness
    L = dose_kg / (COFFEE_DENSITY * BASKET_AREA * (1.0 - porosity))

    time_s = np.array(shot["time_s"])
    pressure = np.array(shot["pressure_bar"])
    flow = np.array(shot["flow_ml_s"])
    temp = np.array(shot["temp_basket_c"])

    k_series = []
    t_series = []

    for i in range(len(time_s)):
        P = pressure[i] * 1e5       # bar -> Pa
        Q = flow[i] * 1e-6          # mL/s -> m³/s
        T_k = temp[i] + 273.15

        # Skip near-zero values (noise, pre-infusion)
        if P < 0.5e5 or Q < 0.1e-6:
            continue

        mu = viscosity(T_k)
        k = Q * mu * L / (BASKET_AREA * P)
        k_series.append(k)
        t_series.append(time_s[i])

    if len(k_series) < 5:
        return None, None

    return np.array(t_series), np.array(k_series)


def main(input_path: Path):
    with open(input_path) as f:
        shots = json.load(f)

    print(f"Computing permeability for {len(shots)} shots...")

    results = []
    for shot in shots:
        t, k = compute_permeability_series(shot)
        if t is None:
            continue

        n = len(k)
        k_early = float(np.mean(k[: n // 3]))
        k_late = float(np.mean(k[2 * n // 3 :]))
        k_median = float(np.median(k))

        results.append({
            "id": shot["id"],
            "k_median_m2": k_median,
            "k_early_m2": k_early,
            "k_late_m2": k_late,
            "k_ratio_late_early": k_late / k_early if k_early > 0 else None,
            "n_points": n,
        })

    # Summary
    k_medians = [r["k_median_m2"] for r in results]
    ratios = [r["k_ratio_late_early"] for r in results if r["k_ratio_late_early"] is not None]
    decreasing = sum(1 for r in ratios if r < 1.0)

    print(f"\nResults ({len(results)} shots with valid permeability):")
    print(f"  Median k: {np.median(k_medians):.2e} m²")
    print(f"  Range: {np.min(k_medians):.2e} to {np.max(k_medians):.2e} m²")
    print(f"  Permeability decreasing: {decreasing}/{len(ratios)} ({100*decreasing/max(len(ratios),1):.0f}%)")

    # Save results
    out_path = input_path.parent / "permeability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"  Wrote {out_path}")


# Also make compute_permeability_series importable from common context
# (plot_dataset.py imports it)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="calibration/curated/all_shots.json")
    args = parser.parse_args()
    main(Path(args.input))

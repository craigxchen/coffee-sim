#!/usr/bin/env python3
"""Assemble fitted parameters from all fitting scripts into a single JSON.

Reads darcy_fit_result.json, extraction_fit_result.json, and thermal_fit_result.json
from the curated/ directory, and writes curated/fitted_params.json.

Usage:
    python calibration/analysis/assemble_params.py [--input calibration/curated/]
"""

import argparse
import json
from datetime import date
from pathlib import Path


def load_if_exists(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main(curated_dir: Path):
    darcy = load_if_exists(curated_dir / "darcy_fit_result.json")
    extraction = load_if_exists(curated_dir / "extraction_fit_result.json")
    thermal = load_if_exists(curated_dir / "thermal_fit_result.json")
    metadata = load_if_exists(curated_dir / "metadata.json")

    missing = []
    if darcy is None:
        missing.append("darcy_fit_result.json")
    if extraction is None:
        missing.append("extraction_fit_result.json")
    if thermal is None:
        missing.append("thermal_fit_result.json")

    if missing:
        print(f"Warning: missing fit results: {', '.join(missing)}")
        print("Run the corresponding fit scripts first.")

    output = {
        "extraction": {
            "K_FAST_REF": extraction["K_FAST_REF"] if extraction else None,
            "K_SLOW_REF": extraction["K_SLOW_REF"] if extraction else None,
            "E_ACTIVATION": extraction["E_ACTIVATION"] if extraction else None,
            "SOLUBLES_FAST_FRAC": extraction["SOLUBLES_FAST_FRAC"] if extraction else None,
            "SOLUBLES_SLOW_FRAC": extraction["SOLUBLES_SLOW_FRAC"] if extraction else None,
            "fit_rmse_ey_pct": extraction["fit_rmse_ey_pct"] if extraction else None,
            "n_shots_with_ey": extraction["n_shots_with_ey"] if extraction else None,
        },
        "darcy": {
            "KC_CORRECTION_FACTOR": darcy["k0_factor"] if darcy else None,
            "POROSITY_FITTED": darcy["porosity"] if darcy else None,
            "FINES_DETACH_RATE": darcy["fines_rate"] if darcy else None,
            "FINES_CAPTURE_RATE": darcy["fines_capture"] if darcy else None,
            "D_P_FITTED_M": darcy["d_p_m"] if darcy else None,
            "D_P_FITTED_UM": darcy["d_p_um"] if darcy else None,
            "median_flow_nrmse": darcy["median_nrmse_validation"] if darcy else None,
            "n_shots_used": darcy["n_fit_shots"] if darcy else None,
        },
        "thermal": {
            "H_WALL_FITTED": thermal["H_WALL_FITTED"] if thermal else None,
            "THERMAL_DIFFUSIVITY_FACTOR": thermal["THERMAL_DIFFUSIVITY_FACTOR"] if thermal else None,
            "median_temp_error_c": thermal["median_abs_error_c"] if thermal else None,
            "n_shots_used": thermal["n_fit_shots"] if thermal else None,
        },
        "meta": {
            "date_fitted": str(date.today()),
            "source": "Visualizer.coffee public API",
            "total_shots_fetched": metadata["total_raw_files"] if metadata else None,
            "total_shots_after_filter": metadata["total_after_filter"] if metadata else None,
        },
    }

    out_path = curated_dir / "fitted_params.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="calibration/curated")
    args = parser.parse_args()
    main(Path(args.input))

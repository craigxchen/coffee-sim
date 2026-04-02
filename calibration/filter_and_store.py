#!/usr/bin/env python3
"""Filter and normalize raw Visualizer.coffee shot data into curated datasets.

Reads raw JSONs from calibration/dataset/, applies quality filters, splits shots
into phases via espresso_state_change boundaries, and outputs:
  - curated/all_shots.json        (shots passing quality filters)
  - curated/shots_with_ey.json    (subset with user-entered TDS/EY)
  - curated/metadata.json         (dataset summary statistics)

Usage:
    python calibration/filter_and_store.py [--input DIR] [--output DIR]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def safe_float(val, default=None):
    """Convert a value to float, handling strings and None."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def parse_timeframe(raw_timeframe: list) -> np.ndarray | None:
    """Convert timeframe strings to float seconds array."""
    if not raw_timeframe:
        return None
    try:
        times = np.array([float(t) for t in raw_timeframe])
        return times
    except (ValueError, TypeError):
        return None


def extract_series(data: dict, key: str, n: int) -> list[float] | None:
    """Extract a time-series array from the shot data dict, padded/truncated to length n."""
    raw = data.get(key)
    if not raw or not isinstance(raw, list):
        return None
    try:
        arr = [float(v) if v is not None else 0.0 for v in raw[:n]]
        # Pad with last value if shorter
        while len(arr) < n:
            arr.append(arr[-1] if arr else 0.0)
        return arr
    except (ValueError, TypeError):
        return None


def find_phase_boundaries(data: dict, n: int) -> list[dict]:
    """Split a shot into phases using espresso_state_change boundaries.

    Each phase has:
      - start_idx, end_idx: indices into the time-series arrays
      - state: the espresso state label for this phase
    """
    state_changes = data.get("espresso_state_change")
    if not state_changes or not isinstance(state_changes, list):
        # No state change data -- treat entire shot as one phase
        return [{"start_idx": 0, "end_idx": n - 1, "state": "unknown"}]

    phases = []
    current_state = None
    current_start = 0

    for i, state in enumerate(state_changes[:n]):
        if state is not None and state != current_state:
            if current_state is not None:
                phases.append({
                    "start_idx": current_start,
                    "end_idx": i - 1,
                    "state": str(current_state),
                })
            current_state = state
            current_start = i

    # Close final phase
    if current_state is not None:
        phases.append({
            "start_idx": current_start,
            "end_idx": min(n - 1, len(state_changes) - 1),
            "state": str(current_state),
        })

    return phases if phases else [{"start_idx": 0, "end_idx": n - 1, "state": "unknown"}]


def normalize_shot(raw: dict) -> dict | None:
    """Convert a raw Visualizer.coffee shot to a normalized record.

    Returns None if the shot fails quality filters.
    """
    # Required metadata
    dose_g = safe_float(raw.get("bean_weight"))
    yield_g = safe_float(raw.get("drink_weight"))
    if dose_g is None or yield_g is None:
        return None

    # Sanity filters on metadata
    if not (10.0 <= dose_g <= 25.0):
        return None
    if not (20.0 <= yield_g <= 80.0):
        return None

    # Time-series data lives under "data" key
    data = raw.get("data")
    if not data or not isinstance(data, dict):
        return None

    # Parse timeframe
    raw_tf = raw.get("timeframe") or data.get("timeframe")
    time_s = parse_timeframe(raw_tf)
    if time_s is None or len(time_s) < 20:
        return None

    n = len(time_s)
    duration = float(time_s[-1] - time_s[0])

    # Duration filter
    if not (15.0 <= duration <= 60.0):
        return None

    # Extract required series
    pressure = extract_series(data, "espresso_pressure", n)
    flow = extract_series(data, "espresso_flow", n)
    weight = extract_series(data, "espresso_weight", n)
    temp_basket = extract_series(data, "espresso_temperature_basket", n)

    if any(s is None for s in [pressure, flow, weight, temp_basket]):
        return None

    # Pressure sanity (max between 3 and 12 bar)
    max_pressure = max(pressure)
    if not (3.0 <= max_pressure <= 12.0):
        return None

    # Temperature sanity
    temps_valid = [t for t in temp_basket if t > 0]
    if not temps_valid:
        return None
    avg_temp = sum(temps_valid) / len(temps_valid)
    if not (70.0 <= avg_temp <= 100.0):
        return None

    # Optional series
    temp_mix = extract_series(data, "espresso_temperature_mix", n)
    water_dispensed = extract_series(data, "espresso_water_dispensed", n)
    pressure_goal = extract_series(data, "espresso_pressure_goal", n)
    flow_goal = extract_series(data, "espresso_flow_goal", n)

    # Optional metadata
    tds = safe_float(raw.get("drink_tds"))
    ey = safe_float(raw.get("drink_ey"))

    # Phase boundaries
    phases = find_phase_boundaries(data, n)

    record = {
        "id": raw.get("id", ""),
        "dose_g": dose_g,
        "yield_g": yield_g,
        "ratio": round(yield_g / dose_g, 2),
        "duration_s": round(duration, 2),
        "tds_pct": round(tds, 2) if tds else None,
        "ey_pct": round(ey, 2) if ey else None,
        "grinder": raw.get("grinder_model"),
        "grind_setting": safe_float(raw.get("grinder_setting")),
        "bean": raw.get("bean_brand"),
        "roast_date": raw.get("roast_date"),
        "profile_title": raw.get("profile_title"),
        "time_s": [round(float(t), 3) for t in time_s],
        "pressure_bar": [round(v, 3) for v in pressure],
        "flow_ml_s": [round(v, 3) for v in flow],
        "weight_g": [round(v, 2) for v in weight],
        "temp_basket_c": [round(v, 2) for v in temp_basket],
        "temp_mix_c": [round(v, 2) for v in temp_mix] if temp_mix else None,
        "water_dispensed_ml": [round(v, 2) for v in water_dispensed] if water_dispensed else None,
        "pressure_goal_bar": [round(v, 3) for v in pressure_goal] if pressure_goal else None,
        "flow_goal_ml_s": [round(v, 3) for v in flow_goal] if flow_goal else None,
        "phases": phases,
    }

    return record


def compute_metadata(all_shots: list[dict], shots_with_ey: list[dict], total_raw: int) -> dict:
    """Compute summary statistics for the curated dataset."""
    doses = [s["dose_g"] for s in all_shots]
    yields = [s["yield_g"] for s in all_shots]
    durations = [s["duration_s"] for s in all_shots]
    ratios = [s["ratio"] for s in all_shots]

    ey_values = [s["ey_pct"] for s in shots_with_ey if s["ey_pct"] is not None]
    tds_values = [s["tds_pct"] for s in shots_with_ey if s["tds_pct"] is not None]

    def stats(arr):
        if not arr:
            return {"count": 0}
        a = np.array(arr)
        return {
            "count": len(a),
            "mean": round(float(np.mean(a)), 2),
            "std": round(float(np.std(a)), 2),
            "min": round(float(np.min(a)), 2),
            "max": round(float(np.max(a)), 2),
            "median": round(float(np.median(a)), 2),
        }

    # Count shots with goal data (for Darcy fitting)
    with_pressure_goal = sum(1 for s in all_shots if s.get("pressure_goal_bar") is not None)
    with_flow_goal = sum(1 for s in all_shots if s.get("flow_goal_ml_s") is not None)

    return {
        "total_raw_files": total_raw,
        "total_after_filter": len(all_shots),
        "total_with_ey": len(shots_with_ey),
        "total_with_tds": sum(1 for s in shots_with_ey if s["tds_pct"] is not None),
        "total_with_pressure_goal": with_pressure_goal,
        "total_with_flow_goal": with_flow_goal,
        "dose_g": stats(doses),
        "yield_g": stats(yields),
        "duration_s": stats(durations),
        "ratio": stats(ratios),
        "ey_pct": stats(ey_values),
        "tds_pct": stats(tds_values),
    }


def main(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all raw JSONs
    raw_files = sorted(input_dir.glob("*.json"))
    print(f"Found {len(raw_files)} raw shot files")

    all_shots = []
    shots_with_ey = []

    for i, fpath in enumerate(raw_files):
        try:
            with open(fpath) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        record = normalize_shot(raw)
        if record is None:
            continue

        all_shots.append(record)
        if record["ey_pct"] is not None or record["tds_pct"] is not None:
            shots_with_ey.append(record)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(raw_files)} files...")

    print(f"\nResults:")
    print(f"  Passing quality filters: {len(all_shots)}")
    print(f"  With TDS/EY data: {len(shots_with_ey)}")

    # Write outputs
    with open(output_dir / "all_shots.json", "w") as f:
        json.dump(all_shots, f, indent=1)
    print(f"  Wrote {output_dir / 'all_shots.json'}")

    with open(output_dir / "shots_with_ey.json", "w") as f:
        json.dump(shots_with_ey, f, indent=1)
    print(f"  Wrote {output_dir / 'shots_with_ey.json'}")

    metadata = compute_metadata(all_shots, shots_with_ey, len(raw_files))
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and normalize Visualizer.coffee shots")
    parser.add_argument("--input", type=str, default="calibration/dataset", help="Raw shot directory")
    parser.add_argument("--output", type=str, default="calibration/curated", help="Output directory")
    args = parser.parse_args()

    main(Path(args.input), Path(args.output))

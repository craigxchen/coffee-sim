#!/usr/bin/env python3
"""Exploratory plots of the curated Visualizer.coffee dataset.

Generates:
  - Pressure vs time overlay (50 random shots)
  - Flow vs time overlay
  - Temperature decay overlay
  - EY distribution histogram
  - TDS distribution histogram
  - Dose vs EY scatter
  - Duration vs EY scatter
  - Derived permeability vs time (subset)

Saves all plots to calibration/analysis/plots/.

Usage:
    python calibration/analysis/plot_dataset.py [--input calibration/curated/]
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compute_permeability import compute_permeability_series


def load_shots(curated_dir: Path) -> tuple[list[dict], list[dict]]:
    with open(curated_dir / "all_shots.json") as f:
        all_shots = json.load(f)
    ey_path = curated_dir / "shots_with_ey.json"
    shots_with_ey = []
    if ey_path.exists():
        with open(ey_path) as f:
            shots_with_ey = json.load(f)
    return all_shots, shots_with_ey


def plot_overlay(shots: list[dict], key: str, ylabel: str, title: str, out_path: Path, n: int = 50):
    """Overlay time-series from n random shots."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = random.sample(shots, min(n, len(shots)))
    for shot in sample:
        t = np.array(shot["time_s"])
        y = np.array(shot[key])
        ax.plot(t, y, alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_histogram(values: list[float], xlabel: str, title: str, out_path: Path, bins: int = 30):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(np.median(values), color="red", linestyle="--", label=f"Median: {np.median(values):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_scatter(x: list[float], y: list[float], xlabel: str, ylabel: str, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.5, s=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Fit line
    if len(x) > 5:
        z = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r--", label=f"r={r:.2f}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main(curated_dir: Path):
    plots_dir = Path("calibration/analysis/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_shots, shots_with_ey = load_shots(curated_dir)
    print(f"Loaded {len(all_shots)} shots ({len(shots_with_ey)} with EY)")

    if not all_shots:
        print("No shots to plot!")
        return

    random.seed(42)

    # Time-series overlays
    plot_overlay(all_shots, "pressure_bar", "Pressure (bar)", "Pressure Profiles (50 shots)", plots_dir / "pressure_overlay.png")
    plot_overlay(all_shots, "flow_ml_s", "Flow (mL/s)", "Flow Profiles (50 shots)", plots_dir / "flow_overlay.png")
    plot_overlay(all_shots, "temp_basket_c", "Basket Temperature (°C)", "Temperature Decay (50 shots)", plots_dir / "temperature_overlay.png")

    # EY and TDS histograms
    ey_vals = [s["ey_pct"] for s in shots_with_ey if s.get("ey_pct")]
    tds_vals = [s["tds_pct"] for s in shots_with_ey if s.get("tds_pct")]

    if ey_vals:
        plot_histogram(ey_vals, "Extraction Yield (%)", f"EY Distribution (n={len(ey_vals)})", plots_dir / "ey_histogram.png")
    if tds_vals:
        plot_histogram(tds_vals, "TDS (%)", f"TDS Distribution (n={len(tds_vals)})", plots_dir / "tds_histogram.png")

    # Scatter plots
    if ey_vals:
        doses = [s["dose_g"] for s in shots_with_ey if s.get("ey_pct")]
        durations = [s["duration_s"] for s in shots_with_ey if s.get("ey_pct")]
        plot_scatter(doses, ey_vals, "Dose (g)", "EY (%)", "Dose vs Extraction Yield", plots_dir / "dose_vs_ey.png")
        plot_scatter(durations, ey_vals, "Duration (s)", "EY (%)", "Duration vs Extraction Yield", plots_dir / "duration_vs_ey.png")

    # Permeability evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    sample = random.sample(all_shots, min(30, len(all_shots)))
    for shot in sample:
        t, k = compute_permeability_series(shot)
        if t is not None and len(t) > 0:
            ax.semilogy(t, k, alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Permeability (m²)")
    ax.set_title("Derived Puck Permeability Over Time (30 shots)")
    fig.tight_layout()
    fig.savefig(plots_dir / "permeability_evolution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {plots_dir / 'permeability_evolution.png'}")

    print("\nAll plots saved to calibration/analysis/plots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="calibration/curated")
    args = parser.parse_args()
    main(Path(args.input))

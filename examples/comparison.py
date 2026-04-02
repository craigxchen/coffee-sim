#!/usr/bin/env python3
"""Example: Compare pourover vs espresso and run grind size sensitivity sweep."""

import os
import sys
sys.path.insert(0, ".")

import matplotlib.pyplot as plt

from python.config import PouroverParams, EspressoParams, GrindProfile
from python.sim import Simulation


def compare_methods():
    """Compare pourover vs espresso extraction curves."""
    po_params = PouroverParams(
        coffee_mass_g=20.0,
        water_mass_g=320.0,
        water_temp_c=93.0,
        grind=GrindProfile(d_main_um=700, fines_fraction=0.15),
        grid_nx=12, grid_ny=12, grid_nz=8, grid_dx=0.0015,
    )

    es_params = EspressoParams(
        coffee_mass_g=18.0,
        target_yield_g=36.0,
        pressure_bar=9.0,
        grind=GrindProfile(d_main_um=300, fines_fraction=0.20),
        grid_nx=12, grid_ny=12, grid_nz=8, grid_dx=0.001,
    )

    print("Running pourover simulation...")
    po_sim = Simulation(po_params, seed=42)
    po_result = po_sim.run(max_time=30.0, dt=0.05)

    print("Running espresso simulation...")
    es_sim = Simulation(es_params, seed=42)
    es_result = es_sim.run(max_time=10.0, dt=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(po_result.time, po_result.extraction_yield, "b-", label="Pourover")
    axes[0].plot(es_result.time, es_result.extraction_yield, "r-", label="Espresso")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Extraction Yield (%)")
    axes[0].set_title("Extraction Yield")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(po_result.time, po_result.cup_tds, "b-", label="Pourover")
    axes[1].plot(es_result.time, es_result.cup_tds, "r-", label="Espresso")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Cup TDS (%)")
    axes[1].set_title("Cup TDS (Cumulative)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(po_result.time, po_result.flow_rate_ml_s, "b-", label="Pourover")
    axes[2].plot(es_result.time, es_result.flow_rate_ml_s, "r-", label="Espresso")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Flow Rate (mL/s)")
    axes[2].set_title("Flow Rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Pourover vs Espresso Comparison", fontsize=14)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/comparison.png", dpi=150, bbox_inches="tight")
    print("Saved figures/comparison.png")


def grind_sensitivity():
    """Sweep grind size and compare extraction yield."""
    grind_sizes = [500, 600, 700, 800, 900]
    results = {}

    for d in grind_sizes:
        print(f"Running grind size {d}um...")
        params = PouroverParams(
            coffee_mass_g=20.0,
            water_mass_g=320.0,
            grind=GrindProfile(d_main_um=d, fines_fraction=0.15),
            grid_nx=10, grid_ny=10, grid_nz=8, grid_dx=0.002,
        )
        sim = Simulation(params, seed=42)
        results[d] = sim.run(max_time=20.0, dt=0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for d, r in results.items():
        ax1.plot(r.time, r.extraction_yield, label=f"{d}um")
        ax2.plot(r.time, r.cup_tds, label=f"{d}um")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Extraction Yield (%)")
    ax1.set_title("EY vs Grind Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cup TDS (%)")
    ax2.set_title("Cup TDS vs Grind Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Grind Size Sensitivity Analysis", fontsize=14)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/grind_sensitivity.png", dpi=150, bbox_inches="tight")
    print("Saved figures/grind_sensitivity.png")


if __name__ == "__main__":
    compare_methods()
    grind_sensitivity()

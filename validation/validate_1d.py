#!/usr/bin/env python3
"""1D column validation: saturation front, outflow, TDS, EY, temperature, mass conservation."""

import os

import coffee_sim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    # 1D column: 3x3x40 voxels, dx=1.5mm → 6cm tall, 4.5mm wide
    config = {
        "nx": 3, "ny": 3, "nz": 40, "dx": 0.0015,  # 1.5mm voxels → realistic bed height
        "geometry": "kalita",
        "d_main_um": 700.0,
        "d_fines_um": 50.0,
        "fines_fraction": 0.15,
        "coffee_mass_g": 20.0,
        "water_temp_c": 93.0,
        "co2_kg_per_kg": 0.01,
        "seed": 42,
    }

    sim = coffee_sim.PySim(config)
    print(f"Grid: {sim.grid_dims()}")

    dt = 0.01  # 10ms timestep
    total_time = 120.0  # 2 minutes
    pour_rate = 3.0  # mL/s constant center pour

    times = []
    tds_inst = []
    tds_cum = []
    ey_list = []
    flow_list = []
    temp_list = []
    mass_err = []
    water_in = []
    water_out = []
    sat_profiles = []

    t = 0.0
    step_count = 0
    while t < total_time:
        # Pour for first 90s, then drawdown
        rate = pour_rate if t < 90.0 else 0.0
        result = sim.step(dt, 0.0, 0.0, rate)

        t += result["actual_dt"]
        step_count += 1

        if step_count % 10 == 0:
            times.append(t)
            tds_inst.append(result["tds_instant"])
            tds_cum.append(result["tds_cumulative"])
            ey_list.append(result["ey"])
            flow_list.append(result["flow_rate_ml_s"])
            temp_list.append(result["avg_bed_temp_c"])
            mass_err.append(result["mass_error_pct"])
            water_in.append(result["water_in_ml"])
            water_out.append(result["water_out_ml"])

        if step_count % 500 == 0:
            sat_profiles.append(np.array(sim.saturation()))

        if step_count % 1000 == 0:
            print(f"  t={t:.1f}s  EY={result['ey']:.2f}%  TDS_cum={result['tds_cumulative']:.4f}%  "
                  f"flow={result['flow_rate_ml_s']:.3f}mL/s  mass_err={result['mass_error_pct']:.4f}%")

    print(f"\nDone: {step_count} steps, {t:.1f}s")
    print(f"Final EY: {ey_list[-1]:.2f}%")
    print(f"Final cup TDS: {tds_cum[-1]:.4f}%")
    print(f"Max mass error: {max(mass_err):.4f}%")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("1D Column Validation", fontsize=14)

    ax = axes[0, 0]
    ax.plot(times, tds_inst, "b-", alpha=0.5, label="Instantaneous")
    ax.plot(times, tds_cum, "b-", linewidth=2, label="Cumulative (cup)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("TDS (%)")
    ax.set_title("TDS vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, ey_list, "r-", linewidth=2)
    ax.axhline(18, color="gray", linestyle="--", alpha=0.5, label="SCA min (18%)")
    ax.axhline(22, color="gray", linestyle="--", alpha=0.5, label="SCA max (22%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EY (%)")
    ax.set_title("Extraction Yield vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(times, flow_list, "g-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow (mL/s)")
    ax.set_title("Outflow Rate vs Time")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(times, temp_list, "orange", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temp (C)")
    ax.set_title("Avg Bed Temperature vs Time")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(times, mass_err, "k-", linewidth=1)
    ax.axhline(0.1, color="red", linestyle="--", label="0.1% threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (%)")
    ax.set_title("Mass Conservation Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if sat_profiles:
        for i, prof in enumerate(sat_profiles[::max(1, len(sat_profiles)//5)]):
            ax.plot(prof, label=f"t={i*500*dt:.0f}s")
        ax.set_xlabel("Voxel index (z)")
        ax.set_ylabel("Saturation")
        ax.set_title("Saturation Front Profiles")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/validate_1d.png", dpi=150, bbox_inches="tight")
    print("Saved figures/validate_1d.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example: Hario V60 pourover simulation."""

import sys
sys.path.insert(0, ".")

from python.config import PouroverParams, GrindProfile
from python.sim import Simulation
from python.viz import plot_timeseries, plot_cross_section, plot_extraction_uniformity, plot_brewing_chart
from python.analysis import average_tds, extraction_uniformity


def main():
    params = PouroverParams(
        coffee_mass_g=20.0,
        water_mass_g=320.0,
        water_temp_c=93.0,
        bloom_water_g=50.0,
        bloom_time_s=35.0,
        pour_rate_ml_s=4.0,
        pour_pattern="spiral",
        geometry="v60",
        grind=GrindProfile(d_main_um=700, d_fines_um=50, fines_fraction=0.15),
        grid_nx=20,
        grid_ny=20,
        grid_nz=15,
        grid_dx=0.001,
    )

    print("Starting V60 pourover simulation...")
    print(f"  Coffee: {params.coffee_mass_g}g")
    print(f"  Water: {params.water_mass_g}g at {params.water_temp_c}C")
    print(f"  Grind: {params.grind.d_main_um}um median")
    print(f"  Pour pattern: {params.pour_pattern}")
    print(f"  Grid: {params.grid_nx}x{params.grid_ny}x{params.grid_nz}")

    sim = Simulation(params, seed=42)

    def progress(t, result):
        if len(result.time) % 100 == 0:
            ey = result.extraction_yield[-1]
            cup = result.cup_tds[-1]
            print(f"  t={t:.1f}s  EY={ey:.2f}%  Cup TDS={cup:.4f}%")

    result = sim.run(max_time=60.0, dt=0.05, callback=progress, snapshot_interval=200)

    print(f"\nSimulation complete: {len(result.time)} timesteps")
    print(f"  Final EY: {result.extraction_yield[-1]:.2f}%")
    print(f"  Cup TDS: {result.cup_tds[-1]:.4f}%")
    print(f"  Average TDS (instantaneous): {average_tds(result):.4f}%")
    print(f"  Extraction uniformity (std): {extraction_uniformity(result):.2f}%")

    plot_timeseries(result, save_path="figures/v60_timeseries.png")
    print("Saved figures/v60_timeseries.png")

    if result.concentration_field is not None:
        plot_cross_section(
            result.concentration_field,
            title="Concentration Field (Y-Z slice)",
            slice_axis=0,
            cmap="hot",
            save_path="figures/v60_concentration.png",
        )
        print("Saved figures/v60_concentration.png")

    plot_extraction_uniformity(result, save_path="figures/v60_uniformity.png")
    print("Saved figures/v60_uniformity.png")

    if result.cup_tds:
        plot_brewing_chart(
            tds=result.cup_tds[-1],
            ey=result.extraction_yield[-1],
            save_path="figures/v60_brewing_chart.png",
        )
        print("Saved figures/v60_brewing_chart.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example: 9-bar espresso simulation with Ergun correction and bed compression."""

import sys
sys.path.insert(0, ".")

from python.config import EspressoParams, GrindProfile
from python.sim import Simulation
from python.viz import plot_timeseries, plot_cross_section
from python.analysis import average_tds


def main():
    params = EspressoParams(
        coffee_mass_g=18.0,
        target_yield_g=36.0,
        water_temp_c=93.0,
        pressure_bar=9.0,
        preinfusion_bar=2.0,
        preinfusion_time_s=5.0,
        pressure_profile="flat",
        basket_diameter_mm=58.0,
        compressibility_alpha=1e-7,
        grind=GrindProfile(d_main_um=300, d_fines_um=30, fines_fraction=0.20),
        grid_nx=20,
        grid_ny=20,
        grid_nz=15,
        grid_dx=0.0005,
    )

    print("Starting espresso simulation (9 bar)...")
    print(f"  Coffee: {params.coffee_mass_g}g")
    print(f"  Target yield: {params.target_yield_g}g")
    print(f"  Pressure: {params.pressure_bar} bar ({params.pressure_profile})")
    print(f"  Grind: {params.grind.d_main_um}um median")
    print(f"  Bed compression alpha: {params.compressibility_alpha}")

    sim = Simulation(params, seed=42)

    def progress(t, result):
        if len(result.time) % 20 == 0:
            ey = result.extraction_yield[-1]
            cup = result.cup_tds[-1]
            flow = result.flow_rate_ml_s[-1]
            print(f"  t={t:.2f}s  EY={ey:.2f}%  Cup TDS={cup:.4f}%  Flow={flow:.2f}mL/s")

    result = sim.run(max_time=15.0, dt=0.02, callback=progress)

    print(f"\nShot complete: {len(result.time)} timesteps, {result.time[-1]:.1f}s")
    print(f"  Final EY: {result.extraction_yield[-1]:.2f}%")
    print(f"  Cup TDS: {result.cup_tds[-1]:.4f}%")
    print(f"  Water collected: {result.total_water_collected_g[-1]:.1f}g")

    plot_timeseries(result, save_path="espresso_timeseries.png")
    print("Saved espresso_timeseries.png")

    if result.concentration_field is not None:
        plot_cross_section(
            result.concentration_field,
            title="Espresso Concentration (Y-Z slice)",
            slice_axis=0,
            cmap="hot",
            save_path="espresso_concentration.png",
        )
        print("Saved espresso_concentration.png")


if __name__ == "__main__":
    main()

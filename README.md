# Coffee Extraction Simulator

A physics-based coffee extraction simulator that models fluid dynamics and chemical
extraction for **pourover** and **espresso** brewing methods. The computational core
is written in Rust (exposed via PyO3) and the orchestration, visualization, and UI
layer is in Python.

## Physics Overview

The simulator implements a full multiphysics model of coffee brewing:

### Packed Bed Geometry
The coffee bed is modeled as a packed bed of polydisperse spherical particles. Grind
size follows a **bimodal log-normal distribution** (fines peak + main peak), and
particles are placed via random sequential deposition into the filter geometry (V60
cone, Kalita flat-bottom, or espresso basket). Each voxel stores local porosity,
permeability (via Kozeny-Carman), and specific surface area.

### Fluid Dynamics
- **Pourover (gravity-driven):** Darcy's law with a preconditioned conjugate gradient
  pressure solver. Boundary conditions track the time-varying water column above the
  bed.
- **Espresso (pump-driven, 6-15 bar):** Ergun equation accounting for inertial
  effects at higher Reynolds numbers, plus bed compression feedback where increased
  pressure reduces porosity, which increases pressure drop.

### Two-Pool Extraction Kinetics
Coffee extraction is modeled as dissolution from two pools per voxel:
- **Fast surface wash** -- readily accessible solubles on broken cell surfaces.
- **Slow diffusion-limited extraction** -- solubles inside intact cells that must
  diffuse through the cell wall.

Dissolved solubles are transported by advection-diffusion with temperature-dependent
rate constants (Arrhenius).

### Heat Transfer
The advection-diffusion equation for temperature is solved on the grid, with
temperature-dependent viscosity and extraction rate constants. Boundary losses through
the filter walls and top surface are included.

### CO2 Bloom (Pourover)
Fresh coffee contains trapped CO2. During the bloom phase, CO2 escapes as gas bubbles
that impede water flow. The gas volume fraction decays over bloom time (~30-45 s),
reducing effective porosity.

### Fines Migration (Espresso)
Small particles can detach and re-lodge downstream under flow, reducing local porosity
over time. This is a key mechanism for channeling in espresso.

### Bed Compression (Espresso)
Under high pressure the bed compacts according to a stress-strain relationship, creating
a feedback loop: compression reduces permeability which increases pressure drop which
causes more compression.

## Key Equations

**Kozeny-Carman** (local permeability from porosity and particle size):
```
k = (e^3 * d_p^2) / (180 * (1 - e)^2)
```

**Darcy's law** (flow through porous media):
```
u = -(k / mu) * grad(P)
```

**Ergun equation** (viscous + inertial pressure drop, used for espresso):
```
-dP/dz = (150 * mu * (1-e)^2 * u) / (e^3 * d_p^2)
       + (1.75 * rho * (1-e) * u^2) / (e^3 * d_p)
```

**Two-pool extraction kinetics:**
```
dm_fast/dt = -k_fast * (C_sat - C_local) * a_s * (m_fast / m_fast_0)
dm_slow/dt = -k_slow * (C_sat - C_local) * a_s * (m_slow / m_slow_0)
```

**Arrhenius temperature dependence:**
```
k(T) = k_ref * exp(-E_a / R * (1/T - 1/T_ref))
```

## Build Instructions

### Prerequisites

- Rust toolchain (install via [rustup](https://rustup.rs/))
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [maturin](https://www.maturin.rs/) build tool

### Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install Python dependencies
uv pip install numpy matplotlib pyvista maturin pytest

# Build the Rust extension in development mode
maturin develop --release

# Verify installation
python -c "import coffee_sim_core; print('Rust core loaded')"
```

### Testing

```bash
# Rust unit tests
cargo test

# Python integration tests
pytest tests/ -v
```

## Quick Start

### V60 Pourover Simulation

```python
from python.presets import v60_default
from python.sim import Simulation
from python.viz import plot_timeseries, plot_brewing_chart, plot_cross_section
from python.analysis import export_timeseries_csv

# Load a V60 preset (20g coffee, 320g water, 93C, spiral pour)
params = v60_default()

# Run the simulation
sim = Simulation(params, seed=42)
result = sim.run(max_time=240.0, dt=0.05, snapshot_interval=100)

# Plot time series (TDS, EY, flow rate, temperature)
plot_timeseries(result, save_path="v60_timeseries.png")

# Plot the final brew on an SCA brewing control chart
final_tds = result.cup_tds[-1] if result.cup_tds else result.tds[-1]
final_ey = result.extraction_yield[-1]
plot_brewing_chart(final_tds, final_ey, save_path="v60_chart.png")

# Visualize the concentration field cross-section
if result.concentration_field is not None:
    plot_cross_section(
        result.concentration_field,
        title="Concentration (vertical slice)",
        slice_axis=1,
        cmap="hot",
        save_path="v60_concentration.png",
    )

# Export data to CSV
export_timeseries_csv(result, "v60_data.csv")
```

### Espresso Simulation

```python
from python.presets import espresso_default
from python.sim import Simulation

params = espresso_default()
sim = Simulation(params, seed=42)
result = sim.run(max_time=30.0, dt=0.01)

print(f"Final cup TDS: {result.cup_tds[-1]:.2f}%")
print(f"Extraction yield: {result.extraction_yield[-1]:.1f}%")
print(f"Shot time: {result.time[-1]:.1f}s")
```

## Configuration Reference

### GrindProfile

| Field | Type | Default | Description |
|---|---|---|---|
| `d_main_um` | float | 700.0 | Median main grind size in micrometers |
| `d_fines_um` | float | 50.0 | Median fines size in micrometers |
| `fines_fraction` | float | 0.15 | Mass fraction of fines (0.0-1.0) |
| `sigma_main` | float | 0.2 | Log-normal spread of main peak |
| `sigma_fines` | float | 0.3 | Log-normal spread of fines peak |

### PouroverParams

| Field | Type | Default | Description |
|---|---|---|---|
| `coffee_mass_g` | float | 20.0 | Dry coffee mass in grams |
| `water_mass_g` | float | 320.0 | Total water mass (1:16 ratio) |
| `water_temp_c` | float | 93.0 | Brew water temperature in Celsius |
| `bloom_water_g` | float | 50.0 | Water used for bloom phase |
| `bloom_time_s` | float | 35.0 | Bloom duration in seconds |
| `pour_rate_ml_s` | float | 4.0 | Main pour flow rate |
| `pour_pattern` | str | "spiral" | Pour pattern: "center", "spiral", or "pulse" |
| `geometry` | str | "v60" | Filter geometry: "v60" or "kalita" |
| `grind` | GrindProfile | (see above) | Grind size distribution parameters |
| `grid_nx` | int | 40 | Grid resolution in X |
| `grid_ny` | int | 40 | Grid resolution in Y |
| `grid_nz` | int | 30 | Grid resolution in Z |
| `grid_dx` | float | 0.001 | Voxel size in meters (1 mm) |

### EspressoParams

| Field | Type | Default | Description |
|---|---|---|---|
| `coffee_mass_g` | float | 18.0 | Dry coffee dose in grams |
| `target_yield_g` | float | 36.0 | Target liquid yield (1:2 ratio) |
| `water_temp_c` | float | 93.0 | Brew water temperature in Celsius |
| `pressure_bar` | float | 9.0 | Main extraction pressure in bar |
| `preinfusion_bar` | float | 2.0 | Pre-infusion pressure in bar |
| `preinfusion_time_s` | float | 5.0 | Pre-infusion duration in seconds |
| `pressure_profile` | str | "flat" | Profile: "flat", "declining", or "ramp" |
| `basket_diameter_mm` | float | 58.0 | Portafilter basket diameter |
| `compressibility_alpha` | float | 1e-7 | Bed compressibility coefficient (1/Pa) |
| `grind` | GrindProfile | d_main=300, fines=0.20 | Grind parameters |
| `grid_nx` | int | 60 | Grid resolution in X |
| `grid_ny` | int | 60 | Grid resolution in Y |
| `grid_nz` | int | 40 | Grid resolution in Z |
| `grid_dx` | float | 0.0005 | Voxel size in meters (0.5 mm) |

## Validation Targets

The simulator is calibrated against published experimental data:

| Brew Method | TDS | Extraction Yield |
|---|---|---|
| Pourover | 1.2 - 1.5% | 18 - 22% |
| Espresso | 8 - 12% | 18 - 22% |

## Academic References

- **Moroney et al. (2015)** -- "Modelling of coffee extraction during brewing using
  multiscale methods." Foundational multiscale extraction model.
- **Cameron et al. (2020)** -- "Systematically Improving Espresso." Experimental data
  on espresso extraction vs. grind size.
- **Corrochano et al. (2015)** -- Experimental TDS and EY data for drip/pourover
  brewing methods.
- Kuhn et al. (2017) -- Packed bed flow and extraction for espresso.
- Melrose et al. (2018) -- Effect of grind size distribution on extraction kinetics.

## Architecture

The project is split between Rust (hot inner loops) and Python (orchestration and
visualization):

```
coffee-sim/
├── rust/src/         # Rust computational core (PyO3)
│   ├── lib.rs        # PyO3 module entry point
│   ├── grid.rs       # 3D voxel grid
│   ├── fluid.rs      # Darcy / Ergun flow solver
│   ├── extraction.rs # Two-pool extraction kinetics
│   ├── thermodynamics.rs  # Heat equation solver
│   ├── particles.rs  # Grind size distribution & bed packing
│   └── utils.rs      # Math helpers
├── python/           # Python orchestration layer
│   ├── sim.py        # Simulation class
│   ├── config.py     # Brew parameter dataclasses
│   ├── viz.py        # Visualization (matplotlib, pyvista)
│   ├── analysis.py   # Post-processing (TDS, EY, channeling)
│   └── presets.py    # V60, Kalita, espresso presets
├── tests/            # Python integration tests
└── examples/         # Usage examples
```

## License

See LICENSE file for details.

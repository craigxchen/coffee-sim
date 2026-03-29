# Coffee Extraction Simulator — Claude Code Implementation Guide

## Project Overview

Build a physics-based coffee extraction simulator that models fluid dynamics and chemical extraction for **pourover** and **espresso** brewing methods. The computational core is written in **Rust** (exposed via PyO3) and the orchestration, visualization, and UI layer is in **Python**.

---

## Architecture

```
coffee-sim/
├── Cargo.toml
├── pyproject.toml
├── rust/
│   └── src/
│       ├── lib.rs                  # PyO3 module entry point
│       ├── grid.rs                 # 3D voxel grid for the coffee bed
│       ├── fluid.rs                # Navier-Stokes / Darcy flow solver
│       ├── extraction.rs           # Mass-transfer & dissolution kinetics
│       ├── thermodynamics.rs       # Heat equation solver
│       ├── particles.rs            # Grind size distribution & bed geometry
│       └── utils.rs                # Common math helpers (interpolation, RNG)
├── python/
│   ├── __init__.py
│   ├── sim.py                      # High-level Simulation class (calls Rust)
│   ├── config.py                   # Brew parameter dataclasses
│   ├── viz.py                      # 3D + 2D visualization (matplotlib / pyvista)
│   ├── analysis.py                 # TDS, EY, flow-rate post-processing
│   └── presets.py                  # V60, Kalita, espresso machine presets
├── tests/
│   ├── test_darcy.py
│   ├── test_extraction.py
│   ├── test_thermal.py
│   └── test_integration.py
├── examples/
│   ├── v60_pourover.py
│   ├── espresso_9bar.py
│   └── comparison.py
└── README.md
```

### Language Split Rationale

| Layer | Language | Why |
|---|---|---|
| PDE solvers, per-voxel extraction, particle packing | **Rust** | Hot inner loops over millions of voxels per timestep; memory safety without GC pauses |
| Simulation orchestration, parameter sweeps, plotting | **Python** | Rapid iteration, scientific ecosystem (numpy, matplotlib, pyvista) |
| Binding | **PyO3 + maturin** | Zero-copy numpy interop via `numpy` crate in Rust |

---

## Physics Models to Implement

### 1. Coffee Bed Geometry (`grid.rs`, `particles.rs`)

The coffee bed is a **packed bed of polydisperse spherical particles** inside a known filter geometry.

#### Grind Size Distribution
- Model grind size as a **bimodal log-normal distribution** (fines peak + main peak).
- Parameters: `d_main` (median main grind, μm), `d_fines` (median fines, μm), `fines_fraction`, `sigma_main`, `sigma_fines`.
- Typical values:
  - Pourover: d_main ≈ 600–900 μm, fines_fraction ≈ 0.10–0.20
  - Espresso: d_main ≈ 200–400 μm, fines_fraction ≈ 0.15–0.25

#### Bed Packing
- Use a **random sequential deposition** algorithm to pack particles into the filter cone / basket geometry.
- Each voxel in the 3D grid stores: local **porosity** ε, local **permeability** k (via Kozeny-Carman), and local **specific surface area** a_s.
- Kozeny-Carman equation for local permeability:
  ```
  k = (ε³ · d_p²) / (180 · (1 - ε)²)
  ```
  where `d_p` is the local effective particle diameter.
- Allow **fines migration**: during flow, small particles can detach and re-lodge downstream, reducing local porosity over time (important for espresso channeling).

#### Filter Geometries
- **Pourover cone** (e.g., V60): truncated cone, 60° included angle, single large hole at bottom.
- **Flat-bottom** (e.g., Kalita Wave): cylinder with small drain holes.
- **Espresso basket**: cylinder, 58 mm diameter, ~20 mm bed depth, perforated metal screen at bottom.

### 2. Fluid Dynamics (`fluid.rs`)

Model water flow through the packed coffee bed. Two regimes:

#### 2a. Pourover (Low Pressure, Gravity-Driven)

Use **Darcy's law** for flow through porous media:

```
u = -(k / μ) · ∇P
```

where:
- `u` = superficial velocity vector (m/s)
- `k` = local permeability (m²) from Kozeny-Carman
- `μ` = dynamic viscosity of water (temperature-dependent, ~0.001 Pa·s at 90°C → use Arrhenius fit)
- `∇P` = pressure gradient (gravity head + any water column above bed)

Combined with **continuity** (incompressible):
```
∇ · u = 0   (within the bed)
```

This gives a **Laplace equation for pressure** with variable coefficients:
```
∇ · ((k/μ) ∇P) = 0
```

Solve on the 3D voxel grid using an **iterative solver** (conjugate gradient or multigrid).

**Boundary conditions:**
- Top of bed: pressure = ρgH where H = water column height above bed (time-varying as you pour)
- Bottom of filter: P = 0 (atmospheric, free draining) or small resistance for restricted-flow drippers
- Sides: no-flux (filter paper is impermeable laterally, or for V60 with ribs, partial lateral flow)

**Pour pattern modeling:**
- Model the pour as a time-varying source term on the top surface.
- Support: center pour, spiral pour, pulse pour (bloom + multiple pours).
- Each pour event adds water to the slurry above the bed; track water level H(t).

#### 2b. Espresso (High Pressure, 6–15 bar)

Still Darcy's law, but:
- Inlet pressure is pump-driven: P_inlet = 9 bar (typical), can model pressure profiles (pre-infusion ramp, declining pressure).
- **Ergun equation** may be needed if Re_pore > 1 (likely for espresso):
  ```
  -dP/dz = (150·μ·(1-ε)²·u) / (ε³·d_p²) + (1.75·ρ·(1-ε)·u²) / (ε³·d_p)
  ```
  (first term = viscous / Darcy, second term = inertial / Forchheimer)
- Model **bed compression**: under high pressure, the bed compacts, reducing porosity. Use a simple stress-strain relationship:
  ```
  ε(P) = ε_0 · (1 - α·P)
  ```
  where α is a compressibility coefficient. This creates a feedback loop: compression → lower permeability → higher pressure drop → more compression.
- **Channeling instability**: if local permeability varies, flow concentrates in high-k paths. Track this; it's key to espresso quality. Optionally implement a simple perturbation analysis or let it emerge from the heterogeneous bed.

### 3. Heat Transfer (`thermodynamics.rs`)

Temperature affects viscosity, extraction rate, and solubility. Solve the **advection-diffusion equation for temperature**:

```
∂T/∂t + u · ∇T = α_th · ∇²T + Q_dissolution
```

where:
- `α_th` = thermal diffusivity of the water-coffee mixture
- `Q_dissolution` = small exothermic/endothermic contribution from dissolution (can be neglected initially)
- Boundary: top surface loses heat to air (Newton cooling), sides lose heat through filter/dripper walls.

**Key temperature-dependent properties:**
- Water viscosity: `μ(T) = A · exp(B / T)` — use standard fit (A ≈ 2.414e-5 Pa·s, B ≈ 247.8 K, reference form)
- Extraction rate constant: Arrhenius dependence (see below)
- CO₂ solubility (for bloom modeling): decreases with temperature

### 4. Extraction Kinetics (`extraction.rs`)

This is the core chemistry. Coffee extraction is **dissolution of soluble compounds from ground coffee into water**.

#### Two-Phase Extraction Model

Coffee extraction happens in two distinct phases per particle:

1. **Fast surface wash** — readily accessible solubles on broken cell surfaces dissolve almost instantly on contact.
2. **Slow diffusion-limited extraction** — solubles inside intact cells must diffuse through the cell wall / porous matrix to reach the bulk liquid.

Model each voxel's extractable mass as two pools:

```
dm_fast/dt = -k_fast · (C_sat - C_local) · a_s · (m_fast / m_fast_0)
dm_slow/dt = -k_slow · (C_sat - C_local) · a_s · (m_slow / m_slow_0)
```

where:
- `k_fast`, `k_slow` = mass transfer coefficients (m/s). k_fast >> k_slow.
- `C_sat` = saturation concentration of coffee solubles (~0.25 g/mL at 93°C, temperature-dependent)
- `C_local` = current local concentration in the liquid phase of the voxel
- `a_s` = specific surface area (m²/m³), depends on grind size
- `m_fast_0` ≈ 20–22% of bean mass (easily extractable), `m_slow_0` ≈ 8–10% (hard to extract)

#### Concentration Transport

Dissolved solubles are transported by the flow. Solve **advection-diffusion for concentration**:

```
∂C/∂t + u · ∇C = D · ∇²C + S_extraction
```

where:
- `D` = effective diffusivity of coffee solubles in water (~5e-10 m²/s, corrected for tortuosity)
- `S_extraction` = source term from the extraction kinetics above (mass dissolved per unit volume per second)

#### Arrhenius Temperature Dependence

```
k(T) = k_ref · exp(-E_a / R · (1/T - 1/T_ref))
```

- `E_a` ≈ 50–80 kJ/mol for coffee extraction
- `T_ref` = 93°C (typical brew temp)

#### CO₂ and Bloom (Pourover)

Fresh coffee contains trapped CO₂ (especially if recently roasted). During the bloom phase:
- CO₂ escapes, creating gas bubbles that impede water flow.
- Model CO₂ as a separate species with its own release kinetics and gas-phase volume fraction.
- `ε_effective = ε - φ_gas` where φ_gas decays over bloom time (~30–45 s).

### 5. Output Quantities to Track

At each timestep, compute and store:

| Quantity | Definition | Unit |
|---|---|---|
| **TDS** (Total Dissolved Solids) | Concentration of solubles in the outflow liquid | % (g/100mL) |
| **Extraction Yield (EY)** | Total mass extracted / dry coffee mass | % |
| **Flow rate** | Volume of liquid exiting the bed per unit time | mL/s |
| **Brew time** | Total elapsed time | s |
| **Pressure drop** | (espresso) Pressure difference across the bed | bar |
| **Temperature profile** | T field in the bed | °C |
| **Concentration field** | C field in the bed (for visualization of extraction uniformity) | g/L |
| **Extraction uniformity** | Std dev of per-voxel extraction yield | % |

---

## Numerical Methods

### Spatial Discretization
- Use a **structured 3D Cartesian grid** with voxel sizes ~0.5–1.0 mm (pourover) or ~0.1–0.25 mm (espresso).
- Typical grid: pourover V60 ≈ 80×80×60 voxels; espresso ≈ 240×240×80 voxels.
- Use **finite volume method** for conservation laws (mass, energy, species).

### Time Integration
- **Operator splitting**: each timestep, solve in order:
  1. Pressure (elliptic) → velocity field
  2. Advect temperature (explicit or semi-implicit)
  3. Advect concentration (explicit or semi-implicit)
  4. Update extraction source terms
  5. (Optional) Update bed geometry (fines migration, compression)
- Use **adaptive timestep** based on CFL condition: `Δt < Δx / max(|u|)`
- Typical timesteps: ~0.01–0.1 s for pourover, ~0.001–0.01 s for espresso.

### Linear Solver
- For the pressure Poisson equation: implement **preconditioned conjugate gradient (PCG)** with a Jacobi or incomplete Cholesky preconditioner.
- For advection: **upwind finite volume** or **TVD scheme** (e.g., van Leer limiter) to avoid spurious oscillations.

---

## Rust Implementation Details

### PyO3 Bindings

Expose the following Python-callable interface:

```python
# What the Python side should see after `import coffee_sim_core`

class SimulationGrid:
    """3D voxel grid with bed properties."""
    def __init__(self, nx: int, ny: int, nz: int, dx: float) -> None: ...
    def porosity(self) -> np.ndarray: ...       # returns 3D numpy array (f64)
    def permeability(self) -> np.ndarray: ...    # returns 3D numpy array (f64)

class FluidSolver:
    """Solves pressure & velocity on the grid."""
    def __init__(self, grid: SimulationGrid) -> None: ...
    def solve_pressure(self, boundary_conditions: dict) -> np.ndarray: ...
    def get_velocity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class ExtractionSolver:
    """Advances extraction kinetics and concentration transport."""
    def __init__(self, grid: SimulationGrid, params: dict) -> None: ...
    def step(self, dt: float, velocity: tuple, temperature: np.ndarray) -> None: ...
    def concentration_field(self) -> np.ndarray: ...
    def extraction_yield_field(self) -> np.ndarray: ...
    def outflow_tds(self) -> float: ...

class ThermalSolver:
    """Solves heat equation on the grid."""
    def __init__(self, grid: SimulationGrid) -> None: ...
    def step(self, dt: float, velocity: tuple, T_inlet: float) -> None: ...
    def temperature_field(self) -> np.ndarray: ...

class BedGenerator:
    """Generates packed particle beds."""
    @staticmethod
    def generate(geometry: str, grind_params: dict, grid: SimulationGrid, seed: int = 42) -> None: ...
```

### Rust Crate Dependencies

```toml
[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"                    # PyO3 numpy interop
ndarray = "0.16"                  # N-dimensional arrays
ndarray-rand = "0.15"             # Random sampling into arrays
rand = "0.8"
rand_distr = "0.4"                # LogNormal, Normal distributions
rayon = "1.10"                    # Parallel iterators for voxel loops
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[lib]
name = "coffee_sim_core"
crate-type = ["cdylib"]           # Required for PyO3

[build]
# Use maturin for building: `maturin develop` for dev, `maturin build --release` for prod
```

### Performance Targets

- Pourover simulation (80³ grid, 4 min brew): < 30 seconds wall time in release mode.
- Espresso simulation (240×240×80 grid, 25 s shot): < 60 seconds wall time.
- Use `rayon` par_iter for all per-voxel computations (extraction update, advection).
- Use SIMD-friendly data layout: struct-of-arrays, not array-of-structs.

---

## Python Implementation Details

### `config.py` — Parameter Dataclasses

```python
@dataclass
class GrindProfile:
    d_main_um: float = 700.0          # median main grind size (μm)
    d_fines_um: float = 50.0          # median fines size (μm)
    fines_fraction: float = 0.15      # mass fraction of fines
    sigma_main: float = 0.2           # log-normal spread
    sigma_fines: float = 0.3

@dataclass
class PouroverParams:
    coffee_mass_g: float = 20.0
    water_mass_g: float = 320.0       # 1:16 ratio
    water_temp_c: float = 93.0
    bloom_water_g: float = 50.0
    bloom_time_s: float = 35.0
    pour_rate_ml_s: float = 4.0       # main pour rate
    pour_pattern: str = "spiral"      # "center", "spiral", "pulse"
    geometry: str = "v60"
    grind: GrindProfile = field(default_factory=GrindProfile)

@dataclass
class EspressoParams:
    coffee_mass_g: float = 18.0
    target_yield_g: float = 36.0      # 1:2 ratio
    water_temp_c: float = 93.0
    pressure_bar: float = 9.0
    preinfusion_bar: float = 2.0
    preinfusion_time_s: float = 5.0
    pressure_profile: str = "flat"    # "flat", "declining", "ramp"
    basket_diameter_mm: float = 58.0
    grind: GrindProfile = field(default_factory=lambda: GrindProfile(d_main_um=300, fines_fraction=0.20))
```

### `viz.py` — Visualization

Implement the following visualizations:

1. **3D volume render** of the concentration field at a given timestep (pyvista or matplotlib voxels).
2. **2D cross-section** slices (vertical and horizontal) showing porosity, velocity magnitude, temperature, and concentration.
3. **Time-series plots**: TDS vs. time, EY vs. time, flow rate vs. time, temperature vs. time.
4. **Extraction uniformity heatmap**: top-down view showing how evenly different regions of the bed were extracted.
5. **Brewing chart**: plot the final brew on an SCA-style brewing control chart (TDS vs. EY with ideal zone).
6. **Animation**: optional export of concentration field evolution as a gif or mp4.

### `analysis.py` — Post-Processing

- Compute cumulative extraction yield over time.
- Compute instantaneous and average TDS.
- Identify channeling: flag voxels with velocity > 3× median velocity.
- Compute flavor balance proxy: ratio of fast-pool extraction to slow-pool extraction (over-extraction correlates with high slow-pool extraction).
- Export time-series data to CSV.

---

## Build & Run

### Development Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install Python dependencies
pip install numpy matplotlib pyvista maturin pytest

# Build Rust extension in development mode
maturin develop --release

# Run example
python examples/v60_pourover.py
```

### Testing

```bash
# Rust unit tests
cargo test

# Python integration tests
pytest tests/ -v
```

---

## Implementation Order

Follow this sequence to build incrementally, testing at each stage:

### Phase 1: Grid & Bed Generation
1. Implement `grid.rs` — 3D voxel grid with porosity/permeability storage.
2. Implement `particles.rs` — bimodal log-normal grind distribution, random packing, Kozeny-Carman.
3. Test: generate a bed, export porosity field to numpy, visualize a cross-section in Python.

### Phase 2: Pressure & Flow
4. Implement `fluid.rs` — Darcy's law pressure solver (PCG), velocity computation.
5. Test: apply gravity head to the packed bed, solve for steady-state flow, verify mass conservation.
6. Add Ergun correction for espresso regime.

### Phase 3: Extraction
7. Implement `extraction.rs` — two-pool kinetics, concentration advection-diffusion.
8. Test: run extraction on a uniform 1D column, compare TDS curve against known analytical solutions and published experimental data.
9. Wire up extraction to flow solver in the time-stepping loop.

### Phase 4: Thermal
10. Implement `thermodynamics.rs` — heat advection-diffusion, temperature-dependent viscosity.
11. Test: verify heat loss during pourover drawdown matches expected ~5–8°C drop.

### Phase 5: Full Integration
12. Build `sim.py` — full simulation loop calling Rust solvers in sequence.
13. Implement pourover with bloom (CO₂ model).
14. Implement espresso with pressure profile and bed compression.
15. Build visualization suite.

### Phase 6: Validation & Polish
16. Compare simulated TDS/EY against published experimental data (e.g., Cameron et al. 2020, Corrochano et al. 2015).
17. Parameter sensitivity analysis: grind size, water temp, brew ratio, pour rate.
18. Write README with usage examples and physics documentation.

---

## Physical Constants & Reference Values

```rust
// Water properties at standard brewing conditions
const WATER_DENSITY: f64 = 971.8;            // kg/m³ at 93°C
const WATER_VISCOSITY_93C: f64 = 0.000306;   // Pa·s at 93°C
const WATER_THERMAL_DIFFUSIVITY: f64 = 1.67e-7; // m²/s
const GRAVITY: f64 = 9.81;                   // m/s²

// Coffee extraction parameters
const C_SAT_93C: f64 = 0.25;                 // g/mL saturation concentration at 93°C
const SOLUBLES_FAST_FRACTION: f64 = 0.21;    // fraction of dry mass easily extractable
const SOLUBLES_SLOW_FRACTION: f64 = 0.09;    // fraction of dry mass hard to extract
const K_FAST_REF: f64 = 5e-5;                // m/s fast extraction rate at T_ref
const K_SLOW_REF: f64 = 2e-7;                // m/s slow extraction rate at T_ref
const E_ACTIVATION: f64 = 65000.0;           // J/mol activation energy
const DIFFUSIVITY_SOLUBLES: f64 = 5e-10;     // m²/s effective diffusivity

// CO₂ (bloom)
const CO2_CONTENT_FRESH: f64 = 0.01;         // kg CO₂ / kg coffee (fresh roast)
const CO2_RELEASE_RATE: f64 = 0.05;          // 1/s first-order release constant

// Bed properties
const BED_TORTUOSITY: f64 = 1.5;             // tortuosity factor for effective diffusivity
const COFFEE_PARTICLE_DENSITY: f64 = 1100.0; // kg/m³ solid coffee particle density
const COFFEE_SPECIFIC_HEAT: f64 = 1670.0;    // J/(kg·K) specific heat of dry coffee
```

---

## Key Academic References for Validation

- Moroney et al. (2015) — "Modelling of coffee extraction during brewing using multiscale methods" — foundational multiscale extraction model
- Cameron et al. (2020) — "Systematically Improving Espresso" — experimental data on espresso extraction vs. grind size
- Corrochano et al. (2015) — experimental TDS and EY data for drip/pourover
- Kuhn et al. (2017) — packed bed flow and extraction for espresso
- Melrose et al. (2018) — effect of grind size distribution on extraction kinetics

Use these to validate that your simulated TDS and EY curves are in the right ballpark (pourover: TDS ~1.2–1.5%, EY ~18–22%; espresso: TDS ~8–12%, EY ~18–22%).

---

## Notes for Claude Code

- **Start with 1D simplification**: Before implementing full 3D, prototype each solver (flow, extraction, heat) in 1D (vertical column). This is faster to iterate on and easier to validate analytically.
- **Use `ndarray` in Rust**: It maps cleanly to numpy via the `numpy` PyO3 crate. Store all fields as `Array3<f64>`.
- **Boundary condition handling**: Use ghost cells (one extra layer around the domain) for clean finite-difference stencils.
- **Don't over-resolve**: Start with coarse grids (40³) for development. Fine grids only for final validation and visualization.
- **Profiling**: Use `cargo flamegraph` to find bottlenecks. The pressure solver and advection loops will dominate.
- **Dimensional analysis**: Work in SI units internally (meters, seconds, Pascals, kg). Convert to user-friendly units (grams, mL, bar, μm, °C) only at the Python boundary.
- **Reproducibility**: Always accept a random seed for bed generation so simulations are deterministic.

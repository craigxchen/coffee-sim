# Coffee Extraction Simulator — Claude Code Implementation Guide

## Vision

Build a **physically accurate pourover coffee extraction simulator** in Rust. The long-term goal is a browser-based interactive 3D experience (like a video game) where a player controls a gooseneck kettle and pours water over coffee grounds in real time. But that is far away.

**The immediate goal is getting the physics right.** The Rust simulation core must produce correct, validated fluid dynamics, extraction kinetics, and heat transfer before any rendering work begins. Python is used only as a validation and plotting harness (via PyO3) — it is not part of the final product.

The architecture is designed so the same Rust physics crate compiles to:
- **Native** (for development, testing, and Python-binding validation)
- **WebAssembly** (for the eventual browser game)

No rendering engine decision needs to be made now. The physics core has zero rendering dependencies.

---

## Architecture

```
coffee-sim/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── sim-core/                       # Pure physics simulation (no I/O, no rendering)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  #   Public API: CoffeeSim struct + step()
│   │       ├── grid.rs                 #   3D voxel grid data structure
│   │       ├── bed.rs                  #   Grind distribution, bed packing, geometry mask
│   │       ├── fluid.rs               #   Darcy pressure solver, velocity, saturation
│   │       ├── extraction.rs           #   Two-pool kinetics, concentration transport
│   │       ├── thermal.rs              #   Heat advection-diffusion
│   │       ├── co2.rs                  #   CO₂ bloom model
│   │       ├── constants.rs            #   Physical constants
│   │       └── pour.rs                 #   Pour script representation (time-series input)
│   │
│   ├── sim-python/                     # PyO3 bindings (for validation only)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                  #   #[pyclass] wrappers around sim-core
│   │
│   └── sim-wasm/                       # wasm-bindgen bindings (for future browser use)
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                  #   #[wasm_bindgen] wrappers around sim-core
│
├── validation/                         # Python scripts for testing & plotting
│   ├── requirements.txt                #   numpy, matplotlib, maturin, pytest
│   ├── validate_1d.py                  #   1D column validation
│   ├── validate_3d_flow.py             #   3D flow cross-sections
│   ├── validate_extraction.py          #   TDS/EY time-series for multiple pour scripts
│   ├── compare_techniques.py           #   Side-by-side pour technique comparison
│   ├── test_mass_conservation.py       #   Automated mass balance checks
│   └── plot_utils.py                   #   Shared matplotlib helpers
│
├── tests/                              # Rust unit + integration tests
│   ├── test_pressure_solver.rs
│   ├── test_extraction.rs
│   ├── test_thermal.rs
│   └── test_conservation.rs
│
└── README.md
```

### Key Design Principle: `sim-core` Is Pure Computation

`sim-core` has **no dependencies** on PyO3, wasm-bindgen, or any rendering/IO library. It depends only on:
- `rand` / `rand_distr` (RNG for bed generation)
- `rayon` (parallelism — feature-gated, disabled for WASM)

This means:
- `sim-python` wraps `sim-core` in `#[pyclass]` for Python validation.
- `sim-wasm` wraps `sim-core` in `#[wasm_bindgen]` for browser deployment.
- A future Rust-native renderer (Bevy, wgpu, etc.) imports `sim-core` directly as a crate.
- All three targets share **identical physics code** — no duplication, no divergence.

```toml
# Cargo.toml (workspace)
[workspace]
members = ["crates/sim-core", "crates/sim-python", "crates/sim-wasm"]

# crates/sim-core/Cargo.toml
[package]
name = "coffee-sim-core"
version = "0.1.0"
edition = "2021"

[features]
default = ["parallel"]
parallel = ["rayon"]        # Disabled for WASM target

[dependencies]
rand = "0.8"
rand_distr = "0.4"
rayon = { version = "1.10", optional = true }

# crates/sim-python/Cargo.toml
[package]
name = "coffee-sim-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "coffee_sim"
crate-type = ["cdylib"]

[dependencies]
coffee-sim-core = { path = "../sim-core" }
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"

# crates/sim-wasm/Cargo.toml
[package]
name = "coffee-sim-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
coffee-sim-core = { path = "../sim-core", default-features = false }  # no rayon
wasm-bindgen = "0.2"
js-sys = "0.3"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
```

---

## Scripted Pour Input (`pour.rs`)

Since there is no user interaction yet, the simulation is driven by a **pour script** — a sequence of timed pour commands.

```rust
pub struct PourCommand {
    pub t_start: f64,       // seconds
    pub t_end: f64,
    pub flow_rate: f64,     // mL/s
    pub pattern: PourPattern,
}

pub enum PourPattern {
    Center,
    Spiral { freq_hz: f64, r_min: f64, r_max: f64 },
    Ring { radius: f64 },
    Point { x: f64, y: f64 },  // normalized [-1, 1]
}

pub struct PourScript {
    pub commands: Vec<PourCommand>,
}

impl PourScript {
    /// Returns (pour_x, pour_y, flow_rate_ml_s) at time t.
    /// (0, 0) = bed center. Coordinates normalized to [-1, 1].
    pub fn sample(&self, t: f64) -> (f64, f64, f64) { ... }
}
```

### Built-In Recipes (for validation)

**1. Classic Spiral (V60 standard)**
```
 0–10 s:   Bloom center pour, 5 mL/s → 50 mL
10–40 s:   Wait
40–130 s:  Spiral pour (0.5 Hz, r 0.1–0.8), 3.5 mL/s → 270 mL
130+ s:    Drawdown
```

**2. Center-Only**
```
 0–10 s:   Bloom center, 5 mL/s → 50 mL
10–40 s:   Wait
40–130 s:  Center pour, 3 mL/s → 270 mL
130+ s:    Drawdown
```

**3. Pulse Pour**
```
 0–8 s:    Bloom center, 6 mL/s → 50 mL
 8–35 s:   Wait
35–45 s:   Pulse 1 spiral, 6 mL/s → 60 mL
45–60 s:   Wait
60–70 s:   Pulse 2 spiral, 6 mL/s → 60 mL
70–85 s:   Wait
85–95 s:   Pulse 3 spiral, 6 mL/s → 60 mL
95–110 s:  Wait
110–117 s: Pulse 4 spiral, 6 mL/s → 40 mL
117+ s:    Drawdown
```

**4. Edge-Heavy Pour (intentionally bad)**
```
 0–10 s:   Bloom center, 5 mL/s → 50 mL
10–40 s:   Wait
40–130 s:  Ring pour at r=0.85, 3 mL/s → 270 mL
130+ s:    Drawdown
```

The spiral pour should produce the highest extraction uniformity. The center pour should over-extract the middle and under-extract the edges. The edge pour should do the opposite. If the simulation doesn't show these differences, the physics is wrong.

---

## Physics Simulation (`sim-core`)

### Public API (`lib.rs`)

```rust
pub struct CoffeeSim {
    // Grid
    grid: Grid,
    mask: Vec<bool>,            // geometry mask

    // State fields (flat Vec<f64>, size nx*ny*nz)
    porosity: Vec<f64>,
    permeability: Vec<f64>,
    saturation: Vec<f64>,
    pressure: Vec<f64>,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    velocity_z: Vec<f64>,
    concentration: Vec<f64>,    // dissolved solubles (g/L)
    temperature: Vec<f64>,      // Kelvin
    m_fast: Vec<f64>,           // remaining fast-pool extractable mass (kg)
    m_slow: Vec<f64>,           // remaining slow-pool extractable mass (kg)
    m_fast_0: Vec<f64>,         // initial fast-pool mass (kg)
    m_slow_0: Vec<f64>,         // initial slow-pool mass (kg)

    // Water level heightfield (nx * ny)
    water_level: Vec<f64>,      // meters above bed top

    // CO₂ state
    co2_mass: Vec<f64>,
    gas_fraction: Vec<f64>,

    // Cumulative tracking
    total_water_in: f64,        // mL
    total_water_out: f64,       // mL
    total_solute_out: f64,      // grams
    brew_time: f64,             // seconds

    // Config
    config: SimConfig,
}

pub struct SimConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,                // voxel size (meters)
    pub geometry: Geometry,     // V60, Kalita
    pub grind: GrindConfig,
    pub coffee_mass_kg: f64,
    pub water_temp_k: f64,
    pub co2_kg_per_kg: f64,
    pub seed: u64,
}

pub struct GrindConfig {
    pub d_main_m: f64,          // main grind size (meters)
    pub d_fines_m: f64,
    pub fines_fraction: f64,
    pub sigma_main: f64,
    pub sigma_fines: f64,
}

pub struct StepResult {
    pub tds_instant: f64,       // outflow TDS (%)
    pub tds_cumulative: f64,    // carafe TDS (%)
    pub ey: f64,                // extraction yield (%)
    pub flow_rate_ml_s: f64,
    pub water_in_ml: f64,
    pub water_out_ml: f64,
    pub water_in_bed_ml: f64,
    pub avg_bed_temp_c: f64,
    pub uniformity: f64,        // 0–1
    pub is_done: bool,
    pub mass_error_pct: f64,
}

impl CoffeeSim {
    pub fn new(config: SimConfig) -> Self { ... }

    pub fn step(&mut self, dt: f64, pour_x: f64, pour_y: f64, pour_rate_ml_s: f64) -> StepResult { ... }

    // Field accessors (return slices for zero-copy)
    pub fn saturation(&self) -> &[f64] { &self.saturation }
    pub fn extraction_yield_field(&self) -> Vec<f64> { /* computed from m_fast, m_slow */ }
    pub fn concentration(&self) -> &[f64] { &self.concentration }
    pub fn temperature(&self) -> &[f64] { &self.temperature }
    pub fn water_level(&self) -> &[f64] { &self.water_level }
    pub fn bed_mask(&self) -> &[bool] { &self.mask }
    pub fn grid_dims(&self) -> (usize, usize, usize, f64) { ... }
}
```

### 1. Grid & Bed Generation (`grid.rs`, `bed.rs`)

#### Grid

```rust
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,    // uniform voxel spacing (meters)
}

impl Grid {
    #[inline]
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix * self.ny * self.nz + iy * self.nz + iz
    }

    pub fn total(&self) -> usize { self.nx * self.ny * self.nz }

    /// World-space center of voxel (ix, iy, iz), with (0,0,0) at bed center-bottom.
    pub fn position(&self, ix: usize, iy: usize, iz: usize) -> (f64, f64, f64) { ... }
}
```

#### Grind Size Distribution

Sample particle diameters from a bimodal log-normal:
```
P(d) = (1 - f) · LogNormal(ln(d_main), σ_main) + f · LogNormal(ln(d_fines), σ_fines)
```
where f = fines_fraction.

For each voxel: sample N particles (N ≈ 10–20), compute the effective diameter as the Sauter mean:
```
d_p = 1 / (Σ (1/d_i) / N)
```

#### Porosity & Permeability

Each voxel's porosity:
```
ε = ε_base + uniform_random(-δε, +δε)
```
Fines-rich voxels (smaller d_p) get a porosity correction: `ε -= 0.05 · (d_fines / d_p)`.

Permeability via Kozeny-Carman:
```
k = (ε³ · d_p²) / (180 · (1 - ε)²)
```

Specific surface area:
```
a_s = 6 · (1 - ε) / d_p
```

#### Geometry Mask

**V60 cone**: truncated cone centered on the grid.
```rust
fn v60_mask(grid: &Grid) -> Vec<bool> {
    // V60: 60° included angle → half-angle = 30° from vertical
    // Top radius ≈ 45 mm, bottom radius ≈ 10 mm, height ≈ 60 mm
    // For each voxel, check if radial distance from center axis < cone radius at that height
    let r_bottom = 0.010; // m
    let r_top = 0.045;    // m
    let height = grid.nz as f64 * grid.dx;
    let mut mask = vec![false; grid.total()];
    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let (x, y, z) = grid.position(ix, iy, iz);
                let r = (x * x + y * y).sqrt();
                let frac = z / height; // 0 at bottom, 1 at top
                let r_cone = r_bottom + (r_top - r_bottom) * frac;
                mask[grid.idx(ix, iy, iz)] = r <= r_cone;
            }
        }
    }
    mask
}
```

**Kalita flat-bottom**: cylinder, radius ≈ 40 mm, height ≈ 25 mm. Mask is simply `r <= 0.040`.

#### Extractable Mass Initialization

For each active voxel:
```
coffee_mass_voxel = (1 - ε) · ρ_coffee · V_voxel
m_fast_0 = SOLUBLES_FAST_FRAC · coffee_mass_voxel
m_slow_0 = SOLUBLES_SLOW_FRAC · coffee_mass_voxel
```

Normalize total coffee mass across all voxels to match the configured dose (e.g., 20 g).

### 2. Fluid Dynamics (`fluid.rs`)

#### Water Level Update

Each timestep, update the 2D heightfield H[ix][iy]:

```rust
fn update_water_level(
    water_level: &mut [f64],  // nx * ny
    pour_x: f64, pour_y: f64, pour_rate: f64,  // from script
    velocity_z_top: &[f64],   // uz at top face of bed, nx * ny
    grid: &Grid, dt: f64,
) {
    let sigma = 3.0 * grid.dx; // pour stream spread
    let a_voxel = grid.dx * grid.dx;

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            let (x, y, _) = grid.position(ix, iy, grid.nz - 1);
            // Gaussian distribution of incoming water
            let dx = x - pour_x * bed_radius;
            let dy = y - pour_y * bed_radius;
            let r2 = dx * dx + dy * dy;
            let q_in = pour_rate * 1e-6 // mL/s → m³/s
                * (-r2 / (2.0 * sigma * sigma)).exp()
                / (2.0 * PI * sigma * sigma);

            let idx = ix * grid.ny + iy;
            let q_drain = velocity_z_top[idx].max(0.0); // water entering bed
            water_level[idx] = (water_level[idx] + dt * (q_in - q_drain) / a_voxel).max(0.0);
        }
    }
}
```

#### Pressure Equation

For saturated voxels (s > s_threshold) inside the geometry mask, solve:

```
Σ_faces [ (k_face / μ) · A_face · (P_neighbor - P_center) / Δx ] = 0
```

This is a symmetric positive-definite sparse linear system: **A · p = b**.

**Face permeability**: harmonic mean of adjacent voxels:
```
k_face = 2 · k_i · k_j / (k_i + k_j)
```

**Boundary conditions encoded in b:**
- Top: where H > 0 and s = 1, fix `P = ρ · g · H`. These become Dirichlet conditions → move to RHS.
- Bottom: `P = 0` (V60) at all active bottom voxels. For Kalita: `P = 0` only at drain holes.
- No-flux: omit the face flux term for boundary faces (cone wall, grid edge, inactive neighbors).

**PCG Solver:**

```rust
pub fn solve_pressure(
    pressure: &mut [f64],       // in/out, initial guess + solution
    permeability: &[f64],
    saturation: &[f64],
    water_level: &[f64],
    mask: &[bool],
    viscosity: f64,             // uniform for now; per-voxel later
    grid: &Grid,
    config: &PressureSolverConfig,
) -> usize /* iterations */ {
    // Implement PCG with Jacobi preconditioner.
    // Matrix-vector product is stencil-based (never assembled explicitly).
    // Parallelize with rayon when "parallel" feature is enabled.
    // Converge when ||r|| / ||b|| < 1e-6.
    // Return iteration count for diagnostics.
}
```

**Target**: < 100 iterations, < 5 ms at 60×60×40 in release mode.

**If PCG is too slow or hard to converge**, fall back to **red-black SOR** (simpler, often competitive for Laplace on structured grids):

```rust
fn sor_step(pressure: &mut [f64], ..., omega: f64) {
    // Red sweep: update voxels where (ix + iy + iz) % 2 == 0
    // Black sweep: update voxels where (ix + iy + iz) % 2 == 1
    // P_new = (1 - ω) · P_old + ω · (Σ neighbor terms + b) / a_diag
    // ω ≈ 1.6–1.8 for near-optimal convergence
}
```

#### Velocity Field

After solving pressure, compute face velocities:
```
u_x[i+½, j, k] = -(k_face / μ) · (P[i+1,j,k] - P[i,j,k]) / Δx
u_y[i, j+½, k] = -(k_face / μ) · (P[i,j+1,k] - P[i,j,k]) / Δx
u_z[i, j, k+½] = -(k_face / μ) · (P[i,j,k+1] - P[i,j,k]) / Δx
```

Store as cell-centered velocities (average of the two adjacent face velocities) for the advection step.

#### Saturation Front

Each voxel tracks saturation `s ∈ [0, 1]`:

```
ds/dt = (flux_in - flux_out) / (ε · V_voxel)
```

where flux_in/out are the volume flow rates through each face of the voxel (computed from face velocities). Only active where water is present above (H > 0) or where an adjacent voxel has s > 0.

A voxel "activates" for the pressure solve when s > s_threshold (0.2). Below that, it's in the unsaturated zone and excluded from the linear system.

**Capillary spreading** (simplified): add a diffusion term on saturation:
```
ds/dt += D_cap · ∇²s
```
with D_cap ≈ 1e-6 m²/s. This spreads the wetting front laterally, preventing sharp staircases.

#### Mass Conservation

Every timestep, compute:
```
water_in_heightfield = Σ H[ix][iy] · A_voxel  (over all columns)
water_in_bed = Σ s[i] · ε[i] · V_voxel       (over all active voxels)
error = total_water_in - (water_in_heightfield + water_in_bed + total_water_out)
error_pct = |error| / total_water_in * 100
```

**Assert `error_pct < 0.1%`.** If violated, log a warning with the timestep and fields for debugging. This is the single most important diagnostic.

### 3. Extraction Kinetics (`extraction.rs`)

#### Two-Pool Dissolution

For each voxel where s > 0:

```rust
fn extraction_step(
    m_fast: &mut [f64], m_slow: &mut [f64],
    m_fast_0: &[f64], m_slow_0: &[f64],
    concentration: &mut [f64],
    saturation: &[f64], temperature: &[f64],
    surface_area: &[f64], mask: &[bool],
    porosity: &[f64],
    grid: &Grid, dt: f64,
) {
    for i in 0..grid.total() {
        if !mask[i] || saturation[i] < 1e-10 { continue; }

        let t_k = temperature[i];
        let k_fast = arrhenius(K_FAST_REF, t_k);
        let k_slow = arrhenius(K_SLOW_REF, t_k);
        let c_sat = c_saturation(t_k);
        let c = concentration[i];
        let s = saturation[i];
        let a_s = surface_area[i];
        let v_liquid = porosity[i] * s * grid.dx.powi(3);

        let driving = (c_sat - c).max(0.0);

        let dm_fast = -k_fast * driving * a_s * (m_fast[i] / m_fast_0[i].max(1e-30)) * s * dt;
        let dm_slow = -k_slow * driving * a_s * (m_slow[i] / m_slow_0[i].max(1e-30)) * s * dt;

        // Clamp: can't extract more than what's left
        let dm_fast = dm_fast.max(-m_fast[i]);
        let dm_slow = dm_slow.max(-m_slow[i]);

        m_fast[i] += dm_fast;
        m_slow[i] += dm_slow;

        // Add dissolved mass to local concentration
        if v_liquid > 1e-30 {
            concentration[i] += (-dm_fast - dm_slow) / v_liquid; // kg/m³
        }
    }
}

fn arrhenius(k_ref: f64, t_k: f64) -> f64 {
    k_ref * ((E_ACTIVATION / R_GAS) * (1.0 / T_REF - 1.0 / t_k)).exp()
}

fn c_saturation(t_k: f64) -> f64 {
    C_SAT_REF * (1.0 + 0.01 * (t_k - T_REF)) // ~1% increase per K
}
```

Parallelize the loop body with `rayon::par_chunks_mut` when the `parallel` feature is enabled.

#### Concentration Transport

Advect dissolved solubles with the flow:

```
C_new[i] = C[i] + dt · (-advection + diffusion + source)
```

**Advection** (first-order upwind, finite volume):
```
advection = (1 / V) · Σ_faces [ u_face · C_upwind · A_face ]
```
where C_upwind = the upstream voxel's concentration.

**Diffusion** (central difference):
```
diffusion = D_eff · Σ_faces [ (C_neighbor - C_center) · A_face / Δx ] / V
```
with D_eff = DIFFUSIVITY_SOLUBLES / TORTUOSITY.

**Outflow tracking**: for bottom-face outflow voxels:
```
solute_out_this_step += u_z_bottom · C[i] · A_face · dt   // kg
volume_out_this_step += u_z_bottom · A_face · dt           // m³
```

### 4. Heat Transfer (`thermal.rs`)

Same advection-diffusion structure as concentration transport, but for temperature:

```
T_new[i] = T[i] + dt · (-advection + diffusion)
```

- D_thermal = THERMAL_DIFFUSIVITY (1.5e-7 m²/s)
- Incoming water temperature: `T_pour` (from config, typically 366 K / 93°C)
- Top surface cooling: `T[i] -= dt · h_air · (T[i] - T_ambient) / (ρ · c_p · Δz)` for top-face voxels
- Side cooling: similar with h_wall

**After updating T**, recompute dynamic viscosity for the pressure solver:
```
μ(T) = VISC_A · exp(VISC_B / (T - VISC_C))
```

### 5. CO₂ Bloom (`co2.rs`)

For each wetted voxel:
```
dm_co2/dt = -CO2_RELEASE_RATE · m_co2 · s
gas_volume = m_co2_released · R · T / (P · M_CO2)
gas_fraction = gas_volume / V_voxel
effective_porosity = (ε · s - gas_fraction).max(0.0)
```

Gas fraction reduces effective permeability → flow resistance increases during bloom → flow rate drops temporarily. As CO₂ escapes, gas_fraction → 0 and flow resumes.

### 6. Timestep Loop

```rust
impl CoffeeSim {
    pub fn step(&mut self, dt: f64, pour_x: f64, pour_y: f64, pour_rate: f64) -> StepResult {
        // 1. Update water heightfield from pour input
        update_water_level(&mut self.water_level, pour_x, pour_y, pour_rate, ...);

        // 2. Update saturation front
        update_saturation(&mut self.saturation, &self.velocity_z, &self.water_level, ...);

        // 3. Compute viscosity from temperature
        let mu = compute_viscosity(&self.temperature);

        // 4. Solve pressure → velocity
        solve_pressure(&mut self.pressure, &self.permeability, &self.saturation, &mu, ...);
        compute_velocity(&self.pressure, &self.permeability, &mu, &mut self.velocity_x, ...);

        // 5. Extraction kinetics (update m_fast, m_slow, add solubles to concentration)
        extraction_step(&mut self.m_fast, &mut self.m_slow, &mut self.concentration, ...);

        // 6. Advect-diffuse concentration
        advect_diffuse_concentration(&mut self.concentration, &self.velocity_x, ...);

        // 7. Advect-diffuse temperature
        advect_diffuse_temperature(&mut self.temperature, &self.velocity_x, ...);

        // 8. CO₂ update (optional)
        update_co2(&mut self.co2_mass, &mut self.gas_fraction, &self.saturation, ...);

        // 9. Track outflow
        let (vol_out, sol_out) = compute_outflow(&self.velocity_z, &self.concentration, ...);
        self.total_water_out += vol_out;
        self.total_solute_out += sol_out;
        self.total_water_in += pour_rate * 1e-6 * dt; // mL → m³
        self.brew_time += dt;

        // 10. Mass conservation check
        let mass_error = self.check_mass_conservation();

        // 11. Compute metrics
        StepResult {
            tds_instant: if vol_out > 0.0 { sol_out / vol_out * 100.0 } else { 0.0 },
            tds_cumulative: self.total_solute_out / self.total_water_out.max(1e-30) * 100.0,
            ey: self.total_solute_out / self.config.coffee_mass_kg * 100.0,
            flow_rate_ml_s: vol_out / dt * 1e6,
            water_in_ml: self.total_water_in * 1e6,
            water_out_ml: self.total_water_out * 1e6,
            avg_bed_temp_c: self.avg_bed_temp() - 273.15,
            uniformity: self.extraction_uniformity(),
            is_done: self.is_bed_drained(),
            mass_error_pct: mass_error,
            ..
        }
    }
}
```

---

## Physical Constants (`constants.rs`)

```rust
// Water
pub const WATER_DENSITY: f64 = 971.8;              // kg/m³ at 93°C
pub const GRAVITY: f64 = 9.81;                     // m/s²
pub const VISC_A: f64 = 2.414e-5;                  // Pa·s
pub const VISC_B: f64 = 247.8;                     // K
pub const VISC_C: f64 = 140.0;                     // K

// Thermodynamics
pub const R_GAS: f64 = 8.314;                      // J/(mol·K)
pub const T_REF: f64 = 366.15;                     // K (93°C)
pub const T_AMBIENT: f64 = 295.0;                  // K (22°C)
pub const THERMAL_DIFFUSIVITY: f64 = 1.5e-7;       // m²/s
pub const H_AIR: f64 = 10.0;                       // W/(m²·K) convective coefficient, top
pub const H_WALL: f64 = 5.0;                       // W/(m²·K) convective coefficient, sides

// Extraction
pub const C_SAT_REF: f64 = 250.0;                  // kg/m³ (= 0.25 g/mL) at T_REF
pub const SOLUBLES_FAST_FRAC: f64 = 0.21;
pub const SOLUBLES_SLOW_FRAC: f64 = 0.09;
pub const K_FAST_REF: f64 = 5e-5;                  // m/s at T_REF
pub const K_SLOW_REF: f64 = 2e-7;                  // m/s at T_REF
pub const E_ACTIVATION: f64 = 65_000.0;            // J/mol

// Transport
pub const DIFFUSIVITY_SOLUBLES: f64 = 5e-10;       // m²/s
pub const TORTUOSITY: f64 = 1.5;

// CO₂
pub const CO2_RELEASE_RATE: f64 = 0.05;            // 1/s
pub const CO2_MOLAR_MASS: f64 = 0.044;             // kg/mol

// Bed
pub const COFFEE_DENSITY: f64 = 1100.0;            // kg/m³ particle density
pub const BASE_POROSITY: f64 = 0.40;
pub const POROSITY_VARIATION: f64 = 0.05;
pub const SATURATION_THRESHOLD: f64 = 0.2;
pub const CAPILLARY_DIFFUSIVITY: f64 = 1e-6;       // m²/s (saturation spreading)
```

---

## Validation Scripts (Python)

### `validate_1d.py`

The most important validation. Runs a 1D column (nx=1, ny=1, nz=40) with constant pour rate and plots:

1. **Saturation front position vs. time** — should descend linearly at first, then slow as bed fills.
2. **Outflow rate vs. time** — should be zero until the front reaches the bottom, then ramp up to steady state.
3. **TDS vs. time** — should spike early (fast pool), then decay as the bed depletes. Peak TDS ≈ 5–15% (concentrated first drips), settling to ≈ 1–2%.
4. **Cumulative EY vs. time** — should follow a two-phase curve: fast initial rise, then slow asymptotic approach toward 20–22%.
5. **Mass conservation error vs. time** — should stay < 0.1% throughout.

Compare TDS and EY against published ranges (SCA gold cup: TDS 1.15–1.45%, EY 18–22%).

### `validate_3d_flow.py`

Runs a 3D sim (40×40×30) with center pour for 20 seconds, then exports cross-section slices:

1. **Vertical slice** (y = center) of saturation field — should show wetting front descending from center, forming a cone shape.
2. **Horizontal slices** at z = top, mid, bottom — should show radial saturation pattern.
3. **Velocity magnitude** on the same slices — should be highest in the center where water column is tallest.

Use matplotlib `imshow` for all plots.

### `validate_extraction.py`

Runs the full physics on a 60×60×40 grid with the spiral pour script. Plots:

1. TDS vs. time (instantaneous and cumulative)
2. EY vs. time
3. Flow rate vs. time
4. Average bed temperature vs. time
5. Extraction uniformity vs. time

Checks that final values are in expected ranges.

### `compare_techniques.py`

Runs all four built-in pour scripts on the same bed configuration. Produces:

1. Overlaid TDS-vs-time for all four techniques
2. Overlaid EY-vs-time
3. Final uniformity comparison (bar chart)
4. 2D top-down extraction heatmaps for each technique

**Expected results:**
- Spiral: highest uniformity, EY ≈ 20%, TDS ≈ 1.3%
- Center: low uniformity, center over-extracted, edges under-extracted
- Pulse: similar EY to spiral but with stepped TDS profile
- Edge: low uniformity, edges over-extracted, center under-extracted

---

## Build & Run

```bash
# Set up Python validation environment
python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib maturin pytest

# Build the PyO3 bindings (always release mode)
cd crates/sim-python
maturin develop --release
cd ../..

# Run Rust unit tests
cargo test --workspace

# Run 1D validation
python validation/validate_1d.py

# Run 3D flow validation
python validation/validate_3d_flow.py

# Run full extraction validation
python validation/validate_extraction.py

# Compare pour techniques
python validation/compare_techniques.py

# (Future) Build WASM
cd crates/sim-wasm
wasm-pack build --target web --release
```

---

## Implementation Phases

### Phase 1: Grid + Bed + 1D Flow
**Goal**: Water flows through a 1D coffee column. Mass is conserved.

1. Implement `grid.rs` — Grid struct, indexing, position mapping.
2. Implement `bed.rs` — grind distribution, porosity, Kozeny-Carman permeability, geometry mask (1D: just a column).
3. Implement `fluid.rs` — 1D pressure solve (trivial: linear pressure drop), water level tracking, saturation front.
4. Implement `constants.rs`.
5. Implement `lib.rs` — CoffeeSim::new() and CoffeeSim::step() with flow-only physics.
6. Implement `sim-python` PyO3 bindings.
7. Write `validate_1d.py` (flow only, no extraction yet): saturation front and outflow rate.
8. **Milestone**: Saturation front descends the column. Outflow starts after front reaches bottom. Mass is conserved to < 0.1%.

### Phase 2: 1D Extraction + Thermal
**Goal**: TDS and EY curves from a 1D column match expected ranges.

9. Implement `extraction.rs` — two-pool kinetics, concentration source term.
10. Implement concentration advection-diffusion (1D).
11. Implement `thermal.rs` — 1D heat advection-diffusion, viscosity coupling.
12. Wire extraction + thermal into `step()`.
13. Extend `validate_1d.py` with TDS, EY, temperature plots.
14. **Milestone**: TDS curve shows early spike then decay. Final EY ≈ 18–22%. Temperature drops ~5°C across the bed.

### Phase 3: 3D Pressure Solver
**Goal**: PCG solves the 3D variable-coefficient Laplace equation correctly.

15. Extend `fluid.rs` to 3D: 7-point stencil, PCG with Jacobi preconditioner, boundary condition handling.
16. Write Rust tests: known analytical solutions (uniform permeability → linear pressure), convergence rate.
17. Implement V60 geometry mask.
18. Write `validate_3d_flow.py`: cross-section saturation plots from center pour.
19. **Milestone**: Wetting front forms a cone shape descending from center. Converges in < 100 PCG iterations. Mass conserved.

### Phase 4: Full 3D Extraction
**Goal**: Complete physics in 3D. Pour technique matters.

20. Extend extraction and concentration transport to 3D (upwind advection, 3D diffusion).
21. Extend thermal solver to 3D.
22. Implement `pour.rs` — PourScript, PourPattern, built-in recipes.
23. Write `validate_extraction.py` (spiral pour on 60×60×40).
24. Write `compare_techniques.py` — run all four scripts, verify differentiation.
25. **Milestone**: Spiral pour produces measurably higher uniformity than center pour. All metrics in expected ranges.

### Phase 5: CO₂ Bloom
**Goal**: Bloom phase affects flow dynamics and extraction timing.

26. Implement `co2.rs` — CO₂ release, gas fraction, effective permeability reduction.
27. Add a validation: run with and without CO₂ on the same pour script. With CO₂, flow rate should dip during the bloom wait, then recover. Without CO₂, flow is monotonic.
28. **Milestone**: CO₂ creates a visible flow-rate dip in the first 30–45 s. Extraction curves differ with/without bloom.

### Phase 6: Performance + WASM Shell
**Goal**: Sim runs at real-time speed. WASM compiles and runs in a browser.

29. Profile with `cargo flamegraph`. Optimize the pressure solver and advection loops.
30. Benchmark: 60×60×40 grid, full physics step in < 10 ms (release mode, native).
31. Implement `sim-wasm` bindings — expose `CoffeeSim` via `wasm-bindgen`.
32. Build with `wasm-pack`. Write a minimal HTML page that imports the WASM module, runs a simulation, and prints metrics to the console.
33. **Milestone**: `coffee_sim_wasm.js` loads in a browser. A scripted pour runs to completion and logs TDS/EY to the console. No rendering yet — just proof that the physics runs in WASM.

---

## Future: Rendering (Decision Deferred)

Once the physics is validated and compiles to WASM, the rendering layer needs to be chosen. Here are the options, ranked by pragmatism:

### Option A: Three.js / TypeScript frontend + WASM physics
- **Pros**: Mature web 3D ecosystem. Huge community, tons of examples. Easy to build a polished UI around it. The physics WASM module is a "black box" that the JS frontend calls each frame.
- **Cons**: Two-language boundary (Rust WASM ↔ JS). Serializing 3D field data across WASM boundary has overhead.
- **Best if**: You want a polished browser experience with minimal Rust rendering code.

### Option B: Bevy (Rust game engine) compiled to WASM
- **Pros**: All-Rust. Bevy has a full ECS, renderer, input system, and WASM support. `sim-core` imports directly — no serialization boundary.
- **Cons**: Bevy's WASM support is usable but less mature than Three.js. Bevy is a heavy dependency. Build times are long.
- **Best if**: You want everything in Rust and don't mind Bevy's opinionated ECS architecture.

### Option C: wgpu + custom renderer compiled to WASM
- **Pros**: Full control. wgpu is the standard Rust GPU abstraction and targets WebGPU (with WebGL2 fallback via wgpu's web backend). No engine overhead.
- **Cons**: You build everything from scratch — camera, scene graph, particle systems, UI. Significant work.
- **Best if**: You want maximal control and minimal dependencies, and are comfortable writing shaders.

### Option D: Macroquad (lightweight Rust game lib)
- **Pros**: Simple, minimal, compiles to WASM easily. Good for prototyping.
- **Cons**: Limited rendering features. Not suitable for production-quality visuals.
- **Best if**: You want to quickly visualize the sim in-browser during development, before committing to a full renderer.

**Recommendation**: Start with **Option D (Macroquad)** for rapid in-browser visualization during development, then migrate to **Option A (Three.js) or Option B (Bevy)** for the final product. But this decision is not blocking — the physics comes first.

---

## Notes for Claude Code

- **Phase 1–2 are the foundation.** Do not rush past 1D validation. If TDS, EY, or mass conservation are wrong in 1D, they will be wrong in 3D and much harder to debug.
- **`sim-core` must never depend on PyO3 or wasm-bindgen.** Those are in separate crates. If you find yourself adding `#[pyclass]` to `sim-core`, stop — put the wrapper in `sim-python` instead.
- **Always build Rust in release mode** (`maturin develop --release`, `cargo test --release`). Debug mode is 10–50× slower and will give misleading performance numbers.
- **SI units everywhere in Rust.** Meters, seconds, Pascals, kilograms, Kelvin. Convert to human units (mL, g, μm, bar, °C) only in the Python bindings or display layer.
- **Log mass conservation error** every timestep. Print a warning if it exceeds 0.1%. This is the cheapest, most powerful debug tool.
- **Seed the RNG** in bed generation for reproducibility.
- **Start with coarse grids** (1×1×40 for 1D, 40×40×30 for 3D development). Use 60×60×40 only for final validation.
- **Feature-gate `rayon`** so the same core compiles to WASM (which doesn't support threads in most browsers yet). Use `#[cfg(feature = "parallel")]` around parallel iterators, with a serial fallback.
- **Don't optimize prematurely.** Get correctness first (Phases 1–4). Optimize in Phase 6 after profiling shows where time is actually spent.

# Plan: Calibration Dataset from Visualizer.coffee

## Context

The sim-core physics modules use hardcoded constants for extraction kinetics (`K_FAST_REF`, `K_SLOW_REF`, `E_ACTIVATION`, `SOLUBLES_FAST_FRAC`, `SOLUBLES_SLOW_FRAC`), permeability (Kozeny-Carman coefficients), and thermal transfer. These values come from the Moroney et al. papers and general porous media literature, but they haven't been validated against real-world brewing data.

The [Visualizer.coffee](https://visualizer.coffee) platform hosts 8,200+ users' espresso shot data from Decent espresso machines and other devices. Each shot includes high-resolution time-series (pressure, flow, weight, temperature at ~0.1s intervals) plus metadata (dose, yield, grind setting). Some shots include user-entered TDS and EY from refractometer readings.

This plan covers fetching a calibration dataset, building analysis tools, and writing tests that validate the simulator against real measurements. The work is independent of the sim-core rewrite and can run in a **parallel worktree agent** in the `coffee-sim/` project folder.

### Espresso vs. Pour-over

Decent data is espresso (6–9 bar pump pressure, fine grind ~200–400μm, 25–35s shots). Our sim targets pour-over (gravity ~0.01–0.05 bar, medium grind ~600–900μm, 3–4 min brews). The physics are identical — Darcy flow, two-pool extraction, heat transfer — but the operating regime differs. What transfers cleanly:

- **Extraction kinetics** — the two-pool dissolution chemistry is grind-geometry and temperature dependent, not pressure dependent. Rate constants calibrated from espresso apply directly to pour-over.
- **Temperature dynamics** — heat loss through the basket/dripper, temperature-dependent viscosity.
- **Permeability evolution** — fines migration and puck/bed restructuring over time.

What does NOT transfer directly:

- **Absolute permeability values** — espresso pucks are compressed at high pressure, pour-over beds are loose. K-C coefficients need separate validation.
- **Flow rate magnitudes** — driven by pump vs. gravity. The Darcy relationship k = Q·μ·L/(A·ΔP) still holds, but you can't directly compare Q values.

---

## Worktree Scope

This worktree creates the `calibration/` directory and everything in it. It does NOT modify any sim-core, sim-python, or validation code. It depends only on Python (requests, numpy, scipy, matplotlib) and the eventual sim-python bindings for the fitting step.

### File Structure

```
calibration/
├── README.md                    # Methodology, API usage notes, data license
├── requirements.txt             # requests, aiohttp, numpy, scipy, matplotlib
├── fetch_shots.py               # Download public shots from Visualizer API
├── filter_and_store.py          # Filter for shots with TDS/EY, normalize, store
├── dataset/                     # Downloaded shot JSON files (gitignored)
│   └── .gitkeep
├── curated/                     # Filtered + normalized dataset (checked in)
│   ├── shots_with_ey.json       # Curated shots with extraction yield data
│   ├── all_shots.json           # All shots passing quality filters (for flow validation)
│   └── metadata.json            # Summary statistics of the dataset
├── analysis/
│   ├── compute_permeability.py  # Derive k(t) from pressure + flow data
│   ├── fit_extraction.py        # Fit two-pool kinetic parameters vs. measured EY
│   ├── fit_darcy.py             # Fit puck model against pressure→flow (or flow→pressure)
│   ├── fit_thermal.py           # Compare basket temp to thermal model
│   └── plot_dataset.py          # Exploratory plots of the raw dataset
└── tests/
    ├── test_kinetics_vs_decent.py    # Sim extraction matches Decent EY data
    ├── test_pressure_flow.py         # Sim flow prediction matches Decent measured flow
    ├── test_permeability_range.py    # K-C predicted k within range of Decent-derived k
    └── test_thermal_decay.py         # Sim temp profile matches basket sensor data
```

---

## Step 1: Fetch Public Shots

### `fetch_shots.py`

```
Usage: python calibration/fetch_shots.py [--pages N] [--output calibration/dataset/]
```

Queries the Visualizer.coffee public API:

1. `GET https://visualizer.coffee/api/shots?items=100&page={n}` — paginate through public shots
2. For each shot in the listing, check if it has non-null/non-zero `espresso_enjoyment`, `bean_weight`, or `drink_weight` (indicators of a complete record)
3. For promising shots, fetch full data: `GET https://visualizer.coffee/api/shots/{id}`
4. Save raw JSON to `calibration/dataset/{id}.json`

Rate limiting: insert 1.2s delay between requests (50 req/min limit). Use `aiohttp` with a semaphore of 1 concurrent request to stay well under limits.

Headers:
```python
headers = {
    "User-Agent": "coffee-sim-calibration/0.1 (research; github.com/your-repo)",
    "Accept": "application/json"
}
```

Target: ~2,000–5,000 raw shots (20–50 pages). This is a one-time fetch; re-run only to refresh.

### Gitignore

Add to `.gitignore`:
```
calibration/dataset/*.json
!calibration/dataset/.gitkeep
```

Raw downloaded shots are not checked in (large, and we don't redistribute others' data). The curated/filtered subset is checked in as a small summary.

---

## Step 2: Filter and Normalize

### `filter_and_store.py`

Reads all raw JSONs from `calibration/dataset/` and applies filters:

**Required fields (must be non-null and non-zero):**
- `bean_weight` (dose in grams)
- `drink_weight` (yield in grams)
- Time-series arrays: `espresso_pressure`, `espresso_flow`, `espresso_weight`, `espresso_temperature_basket`
- All time-series arrays must have ≥ 20 data points

**Desired fields (nice to have, used for kinetics fitting):**
- `espresso_tds` > 0 (user-entered TDS from refractometer)
- `espresso_ey` > 0 (user-entered extraction yield)

**Sanity filters:**
- `bean_weight` between 10g and 25g
- `drink_weight` between 20g and 80g
- Shot duration between 15s and 60s
- Max pressure between 3 and 12 bar
- Basket temperature between 70°C and 100°C

**Normalization:**
For each passing shot, extract a compact record:

```python
{
    "id": str,
    "dose_g": float,
    "yield_g": float,
    "ratio": float,
    "duration_s": float,
    "tds_pct": float | None,       # user-entered, may be null
    "ey_pct": float | None,        # user-entered, may be null
    "grinder": str | None,
    "grind_setting": float | None,
    "bean": str | None,
    "roast_date": str | None,
    "time_s": [float],             # elapsed time array
    "pressure_bar": [float],       # measured pressure
    "flow_ml_s": [float],          # measured flow rate
    "weight_g": [float],           # cumulative output weight
    "temp_basket_c": [float],      # basket temperature
    "temp_mix_c": [float],         # mix temperature (if available)
    "water_dispensed_ml": [float], # cumulative water in (if available)
    "pressure_goal_bar": [float],  # target pressure profile
    "flow_goal_ml_s": [float],     # target flow profile
}
```

Output two files:
- `curated/shots_with_ey.json` — only shots that have TDS and/or EY entered (the gold dataset for kinetics fitting)
- `curated/metadata.json` — summary: total shots fetched, total passing filters, total with TDS/EY, distributions of dose, yield, duration, grind settings

---

## Step 3: Exploratory Analysis

### `analysis/plot_dataset.py`

Before fitting, visualize the dataset to sanity-check it:

1. **Pressure vs. time overlay** — plot 50 random shots' pressure curves on one axis. Should see typical espresso profiles (ramp to 6–9 bar, hold, decline).
2. **Flow vs. time overlay** — same treatment. Should see initial ramp, plateau, possible decline.
3. **Temperature decay** — basket temperature over shot duration. Expect 1–5°C drop.
4. **EY distribution** — histogram of user-entered EY values. Expect peak around 18–22%.
5. **TDS distribution** — histogram. Expect peak around 7–12% for espresso.
6. **Dose vs. EY scatter** — any correlation?
7. **Duration vs. EY scatter** — longer shots should trend toward higher EY.
8. **Derived permeability vs. time** — compute k(t) = Q·μ·L/(A·ΔP) for a subset of shots, plot evolution. Expect decreasing k over time (fines migration, puck compression).

Save all plots to `calibration/analysis/plots/`.

### `analysis/compute_permeability.py`

For each shot, compute instantaneous puck permeability at each timestep:

```python
def compute_permeability(shot):
    # Espresso basket geometry
    A = pi * (0.029)**2              # basket area, ~58mm diameter
    L = shot["dose_g"] / (COFFEE_DENSITY * A * (1 - POROSITY))  # puck thickness from dose

    k = []
    for i in range(len(shot["time_s"])):
        P = shot["pressure_bar"][i] * 1e5   # bar -> Pa
        Q = shot["flow_ml_s"][i] * 1e-6     # mL/s -> m³/s
        T = shot["temp_basket_c"][i] + 273.15
        mu = viscosity(T)

        if P > 0.5e5 and Q > 0.1e-6:  # ignore near-zero values
            k_i = Q * mu * L / (A * P)
            k.append(k_i)
        else:
            k.append(None)

    return k
```

This gives us the real-world permeability range for compressed espresso pucks, which we can compare to Kozeny-Carman predictions for the same grind size.

---

## Step 4: Pressure–Flow Validation (Darcy Fitting)

### Why this is the most important calibration step

Every Decent shot gives you paired time-series of pressure P(t) and flow Q(t). For pressure-profiled shots, the machine imposes P(t) and the puck's resistance determines Q(t). For flow-profiled shots, the machine imposes Q(t) and the puck's resistance determines P(t). Either way, the relationship is governed by Darcy's law:

```
Q(t) = k_eff(t) · A · ΔP(t) / (μ(T) · L)
```

This is a direct, timestep-by-timestep test of the fluid dynamics model. If you feed the machine's imposed variable (pressure or flow) as a boundary condition and your sim predicts the other variable correctly, the Darcy solver, permeability model, and viscosity coupling are all validated simultaneously.

No other signal in the Decent data tests the flow physics this directly. EY only validates extraction kinetics. Temperature only validates heat transfer. But the pressure–flow relationship validates the core of the simulator.

### Identifying shot profile type

The Decent data includes both `espresso_pressure` (measured) and `espresso_pressure_goal` (target), plus `espresso_flow` (measured) and `espresso_flow_goal` (target). To determine which variable the machine is controlling:

```python
def classify_shot_control(shot):
    """Determine if the machine is controlling pressure or flow."""
    p_measured = np.array(shot["pressure_bar"])
    p_goal = np.array(shot["pressure_goal_bar"])
    f_measured = np.array(shot["flow_ml_s"])
    f_goal = np.array(shot["flow_goal_ml_s"])

    # During active extraction (skip pre-infusion), how closely
    # does measured track goal?
    mid = len(p_measured) // 3  # skip ramp-up
    end = len(p_measured) - 5   # skip tail

    p_tracking = np.corrcoef(p_measured[mid:end], p_goal[mid:end])[0, 1]
    f_tracking = np.corrcoef(f_measured[mid:end], f_goal[mid:end])[0, 1]

    # The controlled variable tracks its goal closely
    if p_tracking > f_tracking and p_tracking > 0.8:
        return "pressure_controlled"  # pressure imposed, flow is response
    elif f_tracking > p_tracking and f_tracking > 0.8:
        return "flow_controlled"      # flow imposed, pressure is response
    else:
        return "unknown"              # hybrid or complex profile
```

**Pressure-controlled shots** (most common: classic 9-bar profiles, declining profiles) are the most useful — we impose P(t) and predict Q(t).

**Flow-controlled shots** are also valuable — we impose Q(t) and predict P(t), which tests the same Darcy relationship in reverse.

### `analysis/fit_darcy.py`

**1D puck model** (pure Python, no sim-core dependency):

The espresso puck is modeled as a 1D column of N cells (N=20 is sufficient). Each cell has:
- Porosity ε (from Kozeny-Carman, initialized from grind size estimate)
- Permeability k (from K-C, evolves due to fines migration)
- Temperature T (from basket sensor interpolation)
- Mobile fines fraction (transported downstream by flow)

```python
def simulate_puck_flow(params, shot):
    """
    Given a shot's imposed pressure (or flow) profile, predict flow (or pressure).

    params: [k0_factor, porosity, fines_rate, fines_capture_rate]
      - k0_factor: multiplier on Kozeny-Carman base permeability
      - porosity: initial bed porosity
      - fines_rate: rate at which fines detach and migrate
      - fines_capture_rate: rate at which fines re-lodge downstream
    """
    k0_factor, porosity, fines_rate, fines_capture = params

    # Estimate grind size from grinder + setting if available, else default
    d_p = estimate_grind_size(shot)  # ~200-400μm for espresso

    # Puck geometry
    A = pi * (0.029)**2  # basket area, 58mm diameter
    N = 20
    dose_kg = shot["dose_g"] / 1000
    L = dose_kg / (COFFEE_DENSITY * A * (1 - porosity))
    dz = L / N

    # Initialize per-cell state
    k_base = kozeny_carman(d_p, porosity) * k0_factor
    k_cells = np.full(N, k_base)
    fines_mobile = np.zeros(N)
    fines_deposited = np.zeros(N)
    # Initial fines: more in upper cells (where grinding produces them)
    fines_mobile[:N//3] = 0.02

    control = classify_shot_control(shot)
    predicted = []

    for i in range(1, len(shot["time_s"])):
        dt = shot["time_s"][i] - shot["time_s"][i-1]
        T = shot["temp_basket_c"][i] + 273.15
        mu = viscosity(T)

        if control == "pressure_controlled":
            # Imposed: total ΔP across puck
            P_total = shot["pressure_bar"][i] * 1e5  # Pa

            # Total resistance = sum of cell resistances in series
            R_total = sum(dz / (k_cells[j] * A) for j in range(N)) * mu
            Q = P_total / R_total if R_total > 0 else 0
            predicted.append(Q * 1e6)  # m³/s -> mL/s

        elif control == "flow_controlled":
            # Imposed: flow rate through puck
            Q = shot["flow_ml_s"][i] * 1e-6  # m³/s

            # Total pressure = Q × total resistance
            R_total = sum(dz / (k_cells[j] * A) for j in range(N)) * mu
            P_total = Q * R_total
            predicted.append(P_total / 1e5)  # Pa -> bar

        # Fines migration: advect mobile fines downstream
        v_superficial = Q / A if Q > 0 else 0
        for j in range(N - 1, 0, -1):  # sweep bottom-up for upwind stability
            transport = fines_mobile[j-1] * v_superficial * dt / dz
            capture = fines_mobile[j] * fines_capture * dt
            fines_mobile[j] += transport - capture
            fines_mobile[j-1] -= transport
            fines_deposited[j] += capture

        # Update permeability: deposited fines reduce porosity
        for j in range(N):
            eps_j = porosity - fines_deposited[j]
            eps_j = max(eps_j, 0.05)  # can't fully clog
            k_cells[j] = kozeny_carman(d_p, eps_j) * k0_factor

    return np.array(predicted)
```

**Fitting against the dataset** (all shots, not just those with EY):

```python
def objective(params, shots):
    total_error = 0
    for shot in shots:
        control = classify_shot_control(shot)
        predicted = simulate_puck_flow(params, shot)

        if control == "pressure_controlled":
            measured = np.array(shot["flow_ml_s"][1:])
        elif control == "flow_controlled":
            measured = np.array(shot["pressure_bar"][1:])
        else:
            continue

        # Normalize by measured range to weight all shots equally
        scale = np.ptp(measured) if np.ptp(measured) > 0.1 else 1.0
        total_error += np.mean(((predicted - measured) / scale)**2)

    return total_error / len(shots)

bounds = [
    (0.1, 10.0),      # k0_factor (multiplier on K-C prediction)
    (0.25, 0.50),     # porosity
    (0.001, 0.1),     # fines_rate
    (0.01, 1.0),      # fines_capture_rate
]

result = differential_evolution(objective, bounds, args=(all_shots,), seed=42)
```

**Output**:
- Fitted `k0_factor` tells us how far off the base Kozeny-Carman prediction is from reality
- Fitted fines migration rates can transfer to the pour-over sim (scaled for lower velocities)
- Per-shot comparison plots: measured vs. predicted flow (or pressure)
- Residual statistics: NRMSE per shot, distribution across the dataset

### Why this works for pour-over calibration

The Darcy relationship is linear in ΔP, so the physics that governs flow at 9 bar also governs flow at 0.03 bar — just at different magnitudes. Specifically:

- The **k0_factor** correction to Kozeny-Carman applies to any grind size. If K-C systematically over-predicts permeability by 3x for espresso, it likely does the same for pour-over (the error is in the sphericity assumption, not the pressure).
- The **fines migration dynamics** (detachment rate, capture rate) depend on superficial velocity, not pressure. The rates will be slower in pour-over (lower velocity), but the same model structure applies.
- The **viscosity-temperature coupling** is validated identically — μ(T) doesn't depend on pressure in this regime.

---

## Step 5: Kinetics Fitting

### `analysis/fit_extraction.py`

This is the core calibration step. It fits the two-pool extraction model against shots that have measured EY.

**Approach**: For each shot with known EY, simulate extraction using the shot's own measured flow rate and temperature as driving inputs (not Darcy-solved — we impose the flow). This isolates the extraction kinetics from the flow model.

**Simplified 0D extraction model** (no spatial resolution — the puck is treated as well-mixed, consistent with Moroney 2016):

```python
def simulate_extraction(params, shot):
    k_fast, k_slow, E_act, f_fast, f_slow = params

    dose_kg = shot["dose_g"] / 1000
    m_fast = f_fast * dose_kg
    m_slow = f_slow * dose_kg
    m_fast_0 = m_fast
    m_slow_0 = m_slow
    total_extracted = 0.0

    for i in range(1, len(shot["time_s"])):
        dt = shot["time_s"][i] - shot["time_s"][i-1]
        T = shot["temp_basket_c"][i] + 273.15
        Q = shot["flow_ml_s"][i] * 1e-6  # m³/s

        # Arrhenius rate constants
        kf = k_fast * exp((E_act / R) * (1/T_REF - 1/T))
        ks = k_slow * exp((E_act / R) * (1/T_REF - 1/T))

        # Extraction driving force (simplified: assume dilute, C << C_sat)
        dm_fast = kf * (m_fast / max(m_fast_0, 1e-30)) * dt
        dm_slow = ks * (m_slow / max(m_slow_0, 1e-30)) * dt

        dm_fast = min(dm_fast, m_fast)
        dm_slow = min(dm_slow, m_slow)

        m_fast -= dm_fast
        m_slow -= dm_slow
        total_extracted += dm_fast + dm_slow

    ey_pct = (total_extracted / dose_kg) * 100
    return ey_pct
```

**Fitting**:

```python
from scipy.optimize import differential_evolution

def objective(params, shots):
    residuals = []
    for shot in shots:
        ey_pred = simulate_extraction(params, shot)
        ey_meas = shot["ey_pct"]
        residuals.append((ey_pred - ey_meas)**2)
    return np.mean(residuals)

bounds = [
    (1e-6, 1e-3),    # k_fast
    (1e-8, 1e-5),    # k_slow
    (40000, 90000),   # E_activation (J/mol)
    (0.15, 0.28),     # f_fast (solubles fraction)
    (0.05, 0.15),     # f_slow
]

result = differential_evolution(objective, bounds, args=(gold_shots,), seed=42)
```

**Output**: fitted parameter values, residual statistics, and comparison plots (predicted vs. measured EY for each shot).

---

## Step 6: Thermal Fitting

### `analysis/fit_thermal.py`

Compare the sim's thermal model against basket temperature decay:

1. For each shot, the basket sensor gives T_basket(t) — temperature at the bottom of the puck.
2. Run the sim's 1D thermal model with the shot's flow rate and inlet temperature as inputs.
3. Compare predicted T(z=bottom, t) against measured T_basket(t).
4. Fit `H_WALL` (convective coefficient for basket walls) and thermal conductivity of the puck.

This is simpler than the kinetics fit since temperature is directly measured (not derived like EY).

---

## Step 7: Calibration Tests

These are pytest tests that run against the curated dataset and verify the sim produces physically consistent results. They live in `calibration/tests/` and are run separately from the main `validation/` scripts.

### `tests/test_kinetics_vs_decent.py`

```python
def test_ey_within_tolerance():
    """Sim EY prediction matches Decent shots within ±3% absolute."""
    shots = load_curated_shots_with_ey()
    params = load_fitted_params()  # from fit_extraction.py output

    errors = []
    for shot in shots:
        ey_pred = simulate_extraction(params, shot)
        ey_meas = shot["ey_pct"]
        errors.append(abs(ey_pred - ey_meas))

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    assert mean_error < 2.0, f"Mean EY error {mean_error:.1f}% exceeds 2% threshold"
    assert max_error < 5.0, f"Max EY error {max_error:.1f}% exceeds 5% threshold"

def test_ey_correlation():
    """Predicted and measured EY should be positively correlated (r > 0.5)."""
    # If the model captures real variation, shots the model predicts
    # as high-EY should also measure as high-EY.
    ...

def test_two_pool_depletion_ordering():
    """Fast pool should deplete before slow pool for all shots."""
    ...
```

### `tests/test_pressure_flow.py`

```python
def test_predicted_flow_nrmse():
    """For pressure-controlled shots, predicted flow NRMSE < 25%."""
    shots = load_curated_shots(control_type="pressure_controlled")
    params = load_fitted_darcy_params()

    nrmses = []
    for shot in shots:
        predicted = simulate_puck_flow(params, shot)
        measured = np.array(shot["flow_ml_s"][1:])
        nrmse = np.sqrt(np.mean((predicted - measured)**2)) / np.ptp(measured)
        nrmses.append(nrmse)

    median_nrmse = np.median(nrmses)
    assert median_nrmse < 0.25, f"Median flow NRMSE {median_nrmse:.2f} exceeds 25%"

def test_predicted_pressure_nrmse():
    """For flow-controlled shots, predicted pressure NRMSE < 25%."""
    shots = load_curated_shots(control_type="flow_controlled")
    params = load_fitted_darcy_params()

    nrmses = []
    for shot in shots:
        predicted = simulate_puck_flow(params, shot)
        measured = np.array(shot["pressure_bar"][1:])
        nrmse = np.sqrt(np.mean((predicted - measured)**2)) / np.ptp(measured)
        nrmses.append(nrmse)

    median_nrmse = np.median(nrmses)
    assert median_nrmse < 0.25, f"Median pressure NRMSE {median_nrmse:.2f} exceeds 25%"

def test_flow_shape_correlation():
    """Predicted flow curve shape should correlate with measured (r > 0.7).

    Even if absolute magnitudes are off, the temporal dynamics should match:
    ramp-up timing, plateau shape, decline pattern.
    """
    shots = load_curated_shots(control_type="pressure_controlled")
    params = load_fitted_darcy_params()

    correlations = []
    for shot in shots:
        predicted = simulate_puck_flow(params, shot)
        measured = np.array(shot["flow_ml_s"][1:])
        r = np.corrcoef(predicted, measured)[0, 1]
        correlations.append(r)

    median_r = np.median(correlations)
    assert median_r > 0.7, f"Median flow correlation {median_r:.2f} below 0.7"

def test_permeability_evolution_direction():
    """Derived permeability should decrease over the shot for >60% of shots.

    This validates the fines migration model: mobile fines travel downstream
    and clog pore throats, progressively reducing effective permeability.
    """
    shots = load_curated_shots()
    decreasing_count = 0

    for shot in shots:
        k_series = compute_permeability_from_shot(shot)
        k_clean = [k for k in k_series if k is not None]
        if len(k_clean) < 10:
            continue

        # Compare first-third average to last-third average
        n = len(k_clean)
        k_early = np.mean(k_clean[:n//3])
        k_late = np.mean(k_clean[2*n//3:])

        if k_late < k_early:
            decreasing_count += 1

    frac_decreasing = decreasing_count / len(shots)
    assert frac_decreasing > 0.6, (
        f"Only {frac_decreasing:.0%} of shots show decreasing permeability"
    )
```

### `tests/test_permeability_range.py`

```python
def test_kozeny_carman_within_order_of_magnitude():
    """K-C predicted permeability should be within 10x of Decent-derived values."""
    shots = load_curated_shots()

    for shot in shots[:50]:  # spot check
        k_decent = compute_permeability_from_shot(shot)  # from pressure+flow
        k_kc = kozeny_carman(d_p=300e-6, porosity=0.35)  # espresso grind

        k_decent_median = np.nanmedian(k_decent)
        ratio = k_decent_median / k_kc

        # Espresso pucks are compressed, so Decent k should be LOWER than
        # uncompressed K-C prediction. But within an order of magnitude.
        assert 0.1 < ratio < 10, f"K-C ratio {ratio:.1f} outside expected range"

def test_permeability_decreases_over_shot():
    """Effective permeability should generally decrease during a shot (fines migration)."""
    ...
```

### `tests/test_thermal_decay.py`

```python
def test_temperature_drop_range():
    """Basket temperature should drop 1-8°C over a typical shot."""
    shots = load_curated_shots()

    for shot in shots[:50]:
        t_start = shot["temp_basket_c"][5]   # skip first few noisy points
        t_end = shot["temp_basket_c"][-1]
        drop = t_start - t_end

        assert 0 < drop < 15, f"Temperature drop {drop:.1f}°C outside expected range"

def test_sim_thermal_matches_basket():
    """Sim temperature prediction within ±2°C of basket sensor."""
    ...
```

---

## Step 8: Output Fitted Constants

After the fitting runs succeed, write the fitted values to a JSON file that can be referenced when updating `constants.rs`:

```
calibration/curated/fitted_params.json
{
    "extraction": {
        "K_FAST_REF": 3.2e-5,
        "K_SLOW_REF": 1.8e-7,
        "E_ACTIVATION": 62000,
        "SOLUBLES_FAST_FRAC": 0.21,
        "SOLUBLES_SLOW_FRAC": 0.09,
        "fit_rmse_ey_pct": 1.7,
        "n_shots_with_ey": 83
    },
    "darcy": {
        "KC_CORRECTION_FACTOR": 0.45,
        "POROSITY_FITTED": 0.35,
        "FINES_DETACH_RATE": 0.02,
        "FINES_CAPTURE_RATE": 0.15,
        "median_flow_nrmse": 0.18,
        "median_pressure_nrmse": 0.20,
        "n_shots_pressure_controlled": 312,
        "n_shots_flow_controlled": 87
    },
    "thermal": {
        "H_WALL_FITTED": 45.0,
        "median_temp_error_c": 1.2,
        "n_shots_used": 400
    },
    "meta": {
        "date_fitted": "2026-03-29",
        "source": "Visualizer.coffee public API",
        "total_shots_fetched": 3200,
        "total_shots_after_filter": 1450,
        "notes": "Fitted against Decent espresso machine public shots"
    }
}
```

The sim-core `constants.rs` values are then updated manually (not automatically) based on these fits, with a comment referencing the calibration source. The `KC_CORRECTION_FACTOR` is particularly important — it tells us how far the theoretical Kozeny-Carman prediction is from reality for real coffee grounds (which are irregular, not spherical).

---

## Worktree Execution Notes

This plan runs as a worktree agent in `coffee-sim/`. It:

- Creates the `calibration/` directory and all files within it
- Adds `calibration/dataset/*.json` to `.gitignore`
- Adds `requests`, `aiohttp`, and `scipy` to `calibration/requirements.txt` (separate from the main project's pyproject.toml)
- Does NOT modify any files outside `calibration/` and `.gitignore`
- Does NOT depend on sim-core being built — the fitting scripts use a pure-Python reimplementation of the 0D extraction model, not the Rust bindings
- CAN run in parallel with the sim-core rewrite worktrees

Once sim-core is built and the PyO3 bindings work, a follow-up step replaces the pure-Python 0D model in the fitting scripts with calls to the actual sim via `import coffee_sim`, which validates that the Rust implementation matches the fitted Python model.

---

## Dependencies

```
# calibration/requirements.txt
requests>=2.31
aiohttp>=3.9
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
```

---

## Acceptance Criteria

1. `fetch_shots.py` downloads ≥1,000 public shots without hitting rate limits
2. `filter_and_store.py` produces a curated dataset with ≥30 shots with TDS/EY data and ≥200 shots with pressure+flow data
3. `plot_dataset.py` generates exploratory plots showing sensible distributions
4. `fit_darcy.py` converges — median flow NRMSE < 25% across pressure-controlled shots
5. `fit_extraction.py` converges — RMSE < 3% EY across shots with refractometer data
6. `fit_thermal.py` converges — median temperature error < 2°C
7. All tests in `calibration/tests/` pass, including pressure-flow shape correlation (r > 0.7) and permeability decrease in >60% of shots
8. `fitted_params.json` is written with calibrated constants for extraction, Darcy, and thermal models
9. No files outside `calibration/` and `.gitignore` are modified

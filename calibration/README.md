# Calibration Dataset from Visualizer.coffee

Fetches public espresso shot data from [Visualizer.coffee](https://visualizer.coffee) to calibrate
the sim-core physics constants against real-world brewing measurements.

## What transfers from espresso to pour-over

- **Extraction kinetics** -- two-pool dissolution chemistry is grind-geometry and temperature
  dependent, not pressure dependent. Rate constants calibrated from espresso apply directly.
- **Temperature dynamics** -- heat loss, temperature-dependent viscosity.
- **Permeability evolution** -- fines migration and bed restructuring.

What does NOT transfer directly: absolute permeability values (espresso pucks are compressed),
flow rate magnitudes (pump vs gravity). The Kozeny-Carman correction factor bridges this gap.

## Usage

```bash
# Install dependencies
pip install -r calibration/requirements.txt

# Fetch public shots (2-3 concurrent, ~35 min for 2000 shots)
python calibration/fetch_shots.py --pages 20

# Resume after interruption
python calibration/fetch_shots.py --pages 20 --resume

# Filter and normalize
python calibration/filter_and_store.py

# Exploratory plots
python calibration/analysis/plot_dataset.py

# Fit models
python calibration/analysis/fit_darcy.py
python calibration/analysis/fit_extraction.py
python calibration/analysis/fit_thermal.py

# Run validation tests
pytest calibration/tests/ -v
```

## Data

- `dataset/` -- raw downloaded shot JSONs (gitignored, ~2000 files)
- `curated/` -- filtered and normalized subset (checked in)
  - `all_shots.json` -- shots passing quality filters
  - `shots_with_ey.json` -- subset with user-entered TDS/EY
  - `metadata.json` -- dataset summary statistics
  - `fitted_params.json` -- calibrated constants (output of fitting scripts)

## API

Data sourced from the Visualizer.coffee public API. Rate limited to ~50 req/min.
Raw shot data is not redistributed -- run `fetch_shots.py` to download your own copy.

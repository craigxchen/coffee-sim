# KEEP — salvaged reference values from the v1 MPM codebase

The v1 implementation (old `crates/sim-core` + `crates/sim-wasm/src/mpm_3d/`) is being
**superseded** by `coffee-sim`, built fresh from `docs/ARCHITECTURE.md` + `docs/plans/`.
We do **not** port v1's architecture. This file preserves the durable, hard-won **values**
that would be expensive to re-derive, each tagged with its new home. The full v1 source
remains recoverable on `origin/*` branches and in git history.

> These are **reference values to re-validate**, not gospel. Some were hand-tuned to v1's
> MPM numerics; re-confirm against the new solvers' own gates before trusting them.

---

## 1. Recipe / calibration constants  → `models/`, `materials/`, `assets/calibration/`
_Source: `crates/sim-wasm/src/mpm_3d/brew_config.rs`_
- Recipe defaults: coffee dose **15.0 g**, brew water **250.0 ml**, grind diameter **450 µm** (medium-fine).
- Bed: porosity **0.40**, min permeability **1.0e-12 m²**.
- Kozeny-Carman permeability: `k = d_m² · φ³ / (180 · (1 − φ)²)` (grind diameter is a **squared** lever).
- Extraction kinetics: extractable yield **0.28**, fast pool rate **0.18 s⁻¹**, slow pool rate **0.018 s⁻¹**
  (≈10× split), max solute concentration **0.08** (8% by mass), fast-pool fraction **0.30**.
- Water retention: target bed retention **42 ml**, pore→water transfer rate **4.0 s⁻¹**.
- Bed dynamics: compaction rate **5.5**, impact rate **8.0**.

## 2. Unit calibration (scene ↔ SI)  → `utils/config.rs`
_Source: `crates/sim-wasm/src/mpm_3d/units.rs`_
- Length: **SIM_UNITS_PER_METER ≈ 27.7** (from V60 paper height ≈ 10 cm → ~5.77 scene units).
- Volume: **≈ 5.20 ml per sim-unit³** (derived; intentionally near the legacy hand-tuned value).
- Gravity: standard **9.80665 m/s²** → **≈ 270 scene-units/s²**.
- Speed cap: **MAX_WATER_SPEED 1.5 m/s** → **≈ 41.55 sim-units/s** (also sets fixed-point headroom).
- Pour speeds: gentle **0.12 m/s**, high **0.45 m/s**.
- ⚠ The length scale is baked into permeability, compaction, and extraction timescales — changing it
  requires re-tuning all of them.

## 3. Validated math / WGSL  → `utils/{kernels,sdf,gpu}.rs`
_Source: `crates/sim-wasm/src/mpm_3d/{shader.rs,state.rs}`_
- Quadratic B-spline weights: `w[0]=0.5(1.5−fx)²`, `w[1]=0.75−(fx−1.0)²`, `w[2]=0.5(fx−0.5)²` (per axis). Textbook MPM; matches literature.
- Fixed-point scale **FP_SCALE = 262144.0 (2^18)** for i32 mass/momentum atomics; sized from worst-case
  mass loading (≈50 particles × 0.5625 max weight) + the velocity cap. **Coupled to the velocity cap** —
  change one, re-check the other for overflow.
- Velocity cap clamps grid momentum to keep the fixed-point encoding from overflowing.
- Safe-normalize: return ZERO when `len ≤ EPSILON` (guards NaNs in gradients / contact normals).
- SDF: cone + cylinder interior eval (V60 filter + cup). The rewrite evaluates these **analytically**
  per particle (`utils/sdf.rs`) with analytic gradients — NOT v1's baked 3D-texture + trilinear sampling,
  which aliased a circular cone on the Cartesian grid and trapped water on the wall.

## 4. V60 dripper / filter SDF geometry  → `utils/geometry/`
_Source: `crates/sim-wasm/src/mpm_3d/{filter.rs,bed.rs,shader.rs}`_
- Filter center `(0, -0.35, 0)`; cone: `top_y 2.75`, `bot_y -3.02`, `top_radius 4.10`, `bot_radius 0.0` (apex), `thickness 0.08`.
- `radius_at_y(y) = bot_r + (top_r − bot_r)·t`, `t = (y − bot_y)/height`.
- `inner_radius_at_y(y) = max(radius_at_y(y) − thickness, hole_radius)` (hole_radius ≈ 0 for V60).
- Bed seating: clamp pre-settled bed inside the filter with **0.4–0.6 unit** safety margins (prevents collapse/wall-piling).
- Filter floor (bed apex rest y): `bot_y + (contact_offset + thickness)/slope`.

## 5. Physics target bands (regression envelope)  → test suite + `models/` calibration
_Source: `crates/sim-wasm/src/mpm_3d/physics_tests.rs`_
- Dry bed settle: compression **J ∈ [0.55, 1.25]**; post-settle creep < **0.35 units / 240 frames (4 s)**.
- First water impact: `y_extent > 0.65×settled`, `min_J > 0.45` (bed not flattened).
- Mass conservation (pour-off): `bed_gain = emitted − active_water_gain ± 10%`.
- Rest volume: post-pour drift < **2% / 120 frames**.
- Extraction: yield ~**18–22%** brewed (28% of dose soluble), TDS ~**1.2–1.4%**, conc cap **0.08**.
- Filter contact: outside-fraction < **1%**, penetration < **4 mm**, side-jet < **1.5%**, outward speed < **0.18 m/s**.
- Grind ordering: fine (350 µm) pools more than coarse (1100 µm); permeability ∝ d².
- Packing: grid max packed fraction < **1.75** (capped ~120% of settled).

## 6. Known v1 physics issues (carry into the XPBD solver gates, not Phase 0)
_Source: `docs/ISSUES.md`_
1. Spontaneous velocity blow-up in the cup when new particles drip in.
2. Drawdown too slow to be realistic (possibly grind-size tuning).
3. Emitter water penetrates too deep into the pooled water above the bed.
4. No channeling — water gets stuck above the coffee instead of channeling through it.

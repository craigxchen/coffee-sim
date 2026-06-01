# coffeesim — `vis/` Plan (rendering + UI)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/vis/` · **Status:** plan
**Depends on:** `engine/` (`BrewStateView`, `BrewMetrics`, `PourInput`), `profiler/` (for the HUD), `utils/` (GpuContext). Solver-agnostic — consumes only the canonical state.

## Purpose
Everything the user sees and touches, driven entirely off the canonical `BrewStateView` so it works identically across solvers.

## Scope
**In:** fluid/bed rendering, the brew scorecard, interaction controls, debug overlays, the runtime solver-switch UI. **Out:** simulation (`engine`/`solvers`), profiling internals (`profiler.md`).

## Components

### `render.rs`
- **Screen-space fluid** from water particles (depth → smoothed surface → shaded), the standard particle-fluid surface technique.
- **Concentration → color** mapping (extraction made visible: pale fresh water vs dark extracted).
- **Cross-section view** (clip plane through the bed) to expose channeling and the wet/dry front — central to a *training* sim.
- Bed rendering from grain particles; dripper from the `geometry` SDF.

### `scorecard.rs`
Brew-outcome HUD from `BrewMetrics`: **extraction yield, TDS, drawdown time, evenness**, plus an **A/B replay** to compare two runs (or two solvers) side by side.

### `controls.rs`
Interaction → `PourInput` and scene edits: kettle position / flow rate / pour angle, grind size, water temperature, dose / ratio, dripper selection, bloom timing, play / pause / reset / scrub.

### `debug.rs`
- **Field overlays:** velocity, `α_s`, pressure, wet/dry — toggled per field.
- **Profiler HUD:** per-pass GPU timings **and dispatches-per-frame** (from `SolverProfile`).
- **Runtime solver-switch** dropdown (re-builds the active solver via `engine`) — see the cost/visual delta live.
- Parameter sliders bound to `options`/`models` constants for calibration.

## Build phases & gates
1. Particle point render from `BrewStateView`. *Gate:* particles appear; updates each frame.
2. Controls → `PourInput`; play/pause/reset. *Gate:* pouring is interactive.
3. Screen-space fluid + concentration color + cross-section. *Gate:* looks like coffee brewing; channeling visible in cross-section.
4. Scorecard + debug overlays + solver-switch + sliders. *Gate:* metrics readable; fields inspectable; calibration tunable live.

## Open questions
- Render abstraction shared with the eventual web build (same WGSL render pipeline via `wgpu` → reuse) vs a dev-only renderer now. Leaning: shared from the start (it's `wgpu` either way).
- Immediate-mode debug UI (egui) vs custom — egui for speed of iteration on the dev tool.

# coffeesim — `models/` Plan (shared physics)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/models/` · **Status:** plan (high priority — co-developed with `solver_xpbd`)
**Depends on:** `materials/` (params), `assets/calibration/` (tables). Depended on by **every** solver.

## Purpose
The shared physical models + calibration that **all** solvers import, so cross-solver comparison is **fair**: same physics, different numerics. This is the layer that replaces a generic coupler. Models are written as **pure functions / lookup tables over local state** (`φ`, `T`, `|u_rel|`, `c`, moisture `r`) so any solver — particle or grid — can call them.

## Scope
**In:** permeability, extraction kinetics, cohesion, wetting/absorption, thermal exchange, and the calibration tables that land outcomes in realistic ranges. **Out:** how each solver *applies* these (forces vs constraints — that's per-solver); rendering.

---

## `permeability.rs`
Kozeny-Carman: `k(φ_f) = (d_p²/180) · φ_f³ / (1 − φ_f)²`, with `φ_f` = local pore (fluid) fraction and `d_p` = effective grain diameter (**grind size — the primary user lever**). As the bed compacts (`φ_s ↑`), `k ↓`. `φ_s` clamped to the packing limit (~0.64, rising toward `0.74·(1+r_max)` with absorbed moisture). Outside the bed `φ_s=0 ⇒ k→∞ ⇒` no drag. Consumed by every solver's drag term.

## `extraction.rs`
Phenomenological **two-pool** kinetics (calibrated, not first-principles): a fast pool `s_f` (surfaces, fines, broken cells) and slow pool `s_s` (intact interiors). Per grain, per pool (fast shown):
```
ds_f/dt = k_f(T) · A · s_f · (1 − c_local/c_sat) · g(|u_rel|) · wet(r)
k(T)    = k0 · exp(−Ea / R·T)          # Arrhenius
A       ∝ 1/d_p                         # surface area from grind
g(·)    = saturating fn of local Darcy flux   # advective surface renewal
wet(r)  = moisture gate (dry grain doesn't extract)
```
- `(1 − c/c_sat)` — concentration-gradient driving force (slow pours over-saturate a region).
- `g(|u_rel|)` — the single bridge from flow to extraction; uses the drag term's relative velocity, **independent of incompressibility-solve accuracy**.
- Released solute → overlapping water particles' `c`; **advected on water** (Lagrangian). Drain accumulation → **extraction yield** (target ~18–22%) and **TDS** (~1.2–1.4%).

## `cohesion.rs`
Saturation→cohesion curve (Tampubolon / Robert-Soga): cohesion **rises with saturation to ~40%, then falls to zero at full saturation**. Feeds each solver's bed cohesion (XPBD cohesion compliance; MPM Drucker-Prager cohesion). Also sets dry-bed angle-of-repose calibration.

## `wetting.rs`
Per-grain **moisture ratio** `r` (capped at `r_max`); absorption modeled as a phase change (fluid volume → grain moisture, momentum-conserving). **Grid-deficit method** (project moisture deficit to grid, remove that many fluid particles per cell, interpolate back) to avoid order-dependent clumpy absorption. Moisture raises grain effective volume (feeds `permeability`/exclusion), drives `cohesion`, and gates `extraction`. Enables the dry→wet/bloom behavior the MPM mixture omits.

## `thermal.rs`
Lumped bed↔water heat exchange + loss to dripper/ambient. Water carries `T`; kinetics are `T`-dependent, so the natural pour-temperature drop reduces late-brew extraction. Feeds `extraction`'s Arrhenius term.

## Calibration tables (`assets/calibration/`)
Owns the targets that make the sim believable: drawdown curves vs grind, kinetics constants (`k0`, `Ea`, `c_sat`, pool split), the cohesion curve, `r_max`. **The fairness contract:** every solver reads these, so `compare_solvers` measures the numerical method, not differing parameters.

## Build phases & gates
1. `permeability` + `cohesion` (needed by the primary's coupling/bed). *Gate:* drawdown-vs-grind and angle-of-repose calibrations land in band (with the XPBD solver).
2. `wetting`. *Gate:* dry bed wets/retains; packing clamp rises with moisture.
3. `extraction` + `thermal`. *Gate:* causal levers correct; yield/TDS in band; temperature drop lowers extraction.

## Open questions
- Calibration data sources (published V60 drawdown curves; extraction-yield/TDS references) — `assets/calibration/` provenance.
- `c_sat` and pool-split values per roast; expose as `materials/` presets.
- Fines: separate species with higher `s_f` share — model here or in the solver? Leaning: pool init here, transport in the solver.

## References
Ellero & Navarini, *Mesoscopic modelling of espresso extraction*, J. Food Eng. 2019; Tampubolon et al. 2017 (cohesion); standard brewing references for EY/TDS bands.

# coffeesim — `solver_xpbd` Plan (PRIMARY, shipped)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/solvers/xpbd/` · **Status:** plan
**Depends on:** `models/` (permeability, extraction, cohesion, wetting, thermal), `utils/` (GpuContext, spatial hash, SDF, kernels), `solvers/base.rs` (the seam)

## Purpose
The shipped solver. A single **unified XPBD** (position-based) particle method handling water + bed + their coupling + extraction in one constraint-projection loop. Chosen because XPBD is the only family that satisfies **real-time + long-horizon stability** simultaneously (unconditional stability, large timesteps) while keeping material parameters physically meaningful via compliance. `SolverDescriptor`: `PositionBased`, `owns_grid: false`, `Unconditional`.

## Scope
**In:** water incompressibility, bed mechanics, internal coupling (grain exclusion + interphase forces), the wetting→extraction pipeline, the substep loop. **Out:** the physical *models* it calls (those live in `models.md`); the seam (`solvers.md`).

---

## Internal structure
Two particle species — **water** and **bed** (grounds + fines) — sharing one spatial hash (`utils/hash`), advanced by one XPBD substep loop. Per-particle channels: water carries `{x, v, c, T}`; bed carries `{x, v, x_rest, plastic_state, moisture r, s_fast, s_slow, T, fines_frac}`.

### 1. Water incompressibility — the anti-collapse guarantee (Patch A.1)
This is **internal** to the solver and **always on**; it is what prevents water condensing to a point. Two parts:
- **Constant-density constraint** over water neighbors only: `C_i = ρ_i/ρ₀ − 1`, with `ρ_i = Σ_{j∈water} m_j W_ij`. Target is always `ρ₀`, never modulated by sand.
- **Anti-clustering term** (Monaghan `s_corr = −k (W(Δp_ij)/W(Δq))^n`, e.g. `k≈0.1, n≈4, Δq≈0.2h`) added into the position-correction sum `Δp_i = (1/ρ₀) Σ_j (λ_i + λ_j + s_corr) ∇W_ij`. This injects repulsion that holds spacing even when local density drops below rest — the deficient-neighborhood/free-surface condition where clumping begins.
- **Warm-started** multipliers across frames; **residual-adaptive** iteration count (capped by frame budget; KE watchdog as safety net).

### 2. Bed mechanics
- **Rest-shape + friction + cohesion** compliant constraints. Rest-shape holds the packed configuration (cures "bed collapses at rest"); friction/cohesion compliances **calibrated to angle-of-repose** (via `models/cohesion`).
- **Freeze-when-static**: deformation-rate gate; a quiescent bed skips its constraint solve (stability + the dominant real-time saving over a mostly-quiescent brew).

### 3. Internal coupling
- **Grain exclusion (Patch A.2)** — a **unilateral** constraint that only pushes water *out of* grain-occupied space, layered on top of A.1 (never replaces it): `C_excl = (1 − α_s) − α_f`, projected **only when `α_f > 1 − α_s`**. `α_s = Σ_{j∈bed} V_j^eff W_ij`, clamped to the packing limit (~0.64, rising with absorbed moisture). Pore water stays at `ρ₀` (A.1 guarantees it); exclusion only displaces. This + drag are the two complementary pooling mechanisms (volume exclusion + momentum resistance).
- **Interphase forces** (momentum-conserving pairs), parameterized via `models/`:
  1. **Drag** — ∝ relative velocity, scaled by Kozeny-Carman permeability (`models/permeability`). The **stiff** term → **compliant/implicit** (unconditionally stable for any grind). Its relative velocity is the **extraction-flux signal**.
  2. **Pressure-gradient / buoyancy** — grain pushed by fluid pressure gradient → bed lift/fluidization under high head.
  3. **Virtual (added) mass** — effective added inertia for grains accelerating in water (folded into grain effective mass); damps light-grain jitter.
  4. **Concentration-gradient force** — drives grain dispersion down the `α_s` gradient (realistic fines migration).

### 4. Wetting → extraction pipeline (calls `models/`)
`dry grain → absorb water (moisture r) → cohesion + extraction readiness → two-pool dissolution → solute transport`.
- **Wetting/absorption** (`models/wetting`): per-grain moisture ratio via the grid-deficit method (avoids order-dependent clumpy absorption); raises grain effective volume and the packing clamp.
- **Cohesion** (`models/cohesion`): saturation→cohesion curve feeds the bed cohesion compliance.
- **Extraction** (`models/extraction`): two-pool kinetics gated by moisture; rate ∝ `k(T)·A·(1−c/c_sat)·g(flux)`. Solute added to overlapping water particles' `c`, **advected on water particles** (Lagrangian, no numerical diffusion). Yield/TDS accumulated at the drain.
- **Channeling** (emergent), **fines migration → transient permeability**, **bloom** (CO₂ source + freshness) — all ride on the above.

---

## Substep loop (concrete)
```
per frame (dt):
  1. utils::hash.rebuild(water ∪ bed)                       # one shared hash
  2. predict: gravity + interphase forces (incl. IMPLICIT drag); accumulate grain reaction
  3. bed subcycle ×n (frozen fast path if quiescent):
        rest-shape + friction + cohesion projection
  4. water projection (warm-started, residual-adaptive):
        density constraint + s_corr           ← A.1 (always)
  5. grain-exclusion projection (unilateral)   ← A.2 (on top)
  6. finalize velocities
  7. models: wetting → cohesion update → extraction → solute advection → thermal
```
Steps 4–5 are the position-projection iterations; keep them **batched into as few dispatches as possible** (dispatch count predicts the eventual browser/mobile cost — see `profiler.md`).

## Build phases & gates
1. **Water core** — A.1 (density + `s_corr`), warm-start, shared hash, `α_s=0`. *Gate:* incompressible dam-break; **3-min run, no clumping/collapse, volume-stable**.
2. **Bed (dry)** — rest-shape + friction + cohesion. *Gate:* angle-of-repose matches grounds; **static 3 min, zero drift**.
3. **Coupling** — grain exclusion (A.2) + four interphase forces. *Gate:* drawdown responds to grind; volume conserved under displacement; momentum conserved; bed floods under over-fine grind.
4. **Wetting + cohesion** (`models`). *Gate:* dry bed wets and retains moisture; cohesion tracks saturation.
5. **Extraction + thermal** (`models`). *Gate:* causal levers correct; yield ~18–22%, TDS ~1.2–1.4%; solute conserved over full brew.
6. **Channeling + fines + bloom.** *Gate:* channeling emerges from flow non-uniformity; fines slow drawdown; fresh coffee blooms.

## Open questions
- Coupling-split convergence when fines mass ≈ local water mass (1–2 sub-iterations; implicit drag keeps it stable if under-converged).
- Iteration-count policy for the density constraint vs frame budget (residual-driven; multigrid acceleration only if a deep bed under-resolves head).
- Whether grain exclusion needs its own iteration or can share the density-projection sweep.

## References
Macklin et al. *XPBD* (MIG 2016); Macklin & Müller *Position Based Fluids* (TOG 2013); Tang et al. *Granule-in-Cell* (2025, volume-fraction + interphase forces); Tampubolon et al. 2017 (saturation→cohesion). Physics details in `models.md`.

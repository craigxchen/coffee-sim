---
title: "feat: Phase 1.4 — Wetting + Cohesion (volume-conserving absorption + swelling)"
type: feat
status: completed
date: 2026-06-03
origin: docs/plans/solver_xpbd.md (step 4), docs/plans/models.md (phase 2)
deepened: 2026-06-03
---

# feat: Phase 1.4 — Wetting + Cohesion

## Summary

Add saturation-driven **wetting** and **cohesion** to the XPBD coffee bed as two shared `models/`
modules applied by the GPU solver. A dry grain bed in contact with water **absorbs** it, **retains**
the moisture, **swells** by the absorbed volume, and its **cohesion follows a saturation curve**
(clumps when damp, weakens toward full saturation). The defining constraint is **volume
conservation**: water volume that leaves the fluid phase becomes grain volume 1:1 (the solid swells),
so total system volume is invariant — verified by an explicit conservation test that gates the phase.

This is the wetting+cohesion milestone only. Extraction kinetics, thermal, bloom (CO₂), fines, and
solute transport are explicitly deferred to Phases 1.5/1.6; the V60 cone geometry is a separate phase.

---

## Problem Frame

The bed (Phase 1.2) and the water/bed mechanical coupling (Phase 1.3) are committed and stable. The
bed's cohesion term is currently a **constant** `dry_cohesion = 0` (`src/solvers/xpbd/bed.wgsl:65-69`),
and grains have no moisture state — so the bed cannot wet, cannot retain water, and its mechanical
strength cannot respond to saturation. Real coffee grounds absorb ~their own mass in water, swell,
clump when damp, and slump when flooded; that behavior is the substrate every later phase
(extraction, bloom) rides on.

The hard problem is **doing this conservatively on the GPU**: absorbed water cannot simply vanish
into a scalar (that destroys volume and momentum), and the WebGPU baseline gives us **8 storage
buffers/stage** and **no portable float atomics**. The bed contact kernel is already at the 8-buffer
limit, so the moisture state must be threaded without new bindings there.

**Source of truth:** `docs/plans/solver_xpbd.md` step 4 + substep step 7; `docs/plans/models.md`
phase 2 (`wetting.rs`, `cohesion.rs`, the `0.74·(1+r_max)` packing clamp). Physics grounded in Tang
et al. 2025 (Granule-In-Cell, already cited) and Tampubolon et al. 2017.

---

## Requirements

- **R1 — Wetting.** A dry grain in contact with water gains moisture over time (rate-limited, not
  instantaneous) up to a per-grain capacity; moisture is retained (does not evaporate this phase).
- **R2 — Volume conservation (HARD).** Every unit of water volume removed from the fluid phase is
  accounted for as grain swelling. Total volume (fluid + solid) is invariant across absorption to a
  tight tolerance. This is a gating test, not a diagnostic.
- **R3 — Swelling.** A grain's effective volume and contact radius grow with its absorbed volume, and
  that swollen size feeds **α_s/exclusion** (`compute_fractions`), **permeability** (pores shrink →
  drag rises), the **packing clamp**, and the **bed contact** non-penetration distance.
- **R4 — Saturation→cohesion.** Bed cohesion follows a curve that rises with saturation to a peak
  (~40%) then falls toward zero at full saturation, replacing the constant `dry_cohesion`.
- **R5 — Conservation of momentum.** Absorbed water's momentum is transferred to the absorbing grain
  via an exact per-substep inelastic merge (no spurious force, no lag/leak).
- **R6 — No regressions.** The water-only, dry-bed, and coupling test suites stay green; single-species
  scenes remain byte-unchanged (all wetting passes are no-ops without both species present).
- **R7 — Portability.** All kernels stay ≤ 8 storage buffers/stage; no float atomics; `Params` stays
  byte-matched Rust↔WGSL; targets `wgpu::Limits::default()`.

**Success gate (from the docs):** dry bed wets and retains moisture; cohesion tracks saturation;
packing clamp rises with moisture; **volume conserved across absorption**.

---

## Key Technical Decisions

### KTD-1 — Conservation by tracking *volume* AND *mass*, not a moisture scalar

Each **water** particle carries a remaining-volume fraction `f_w ∈ [0,1]` (1.0 = full); each **grain**
carries an absorbed-water volume `V_abs ≥ 0`. Absorption transfers volume from water to grain 1:1, so
grain effective volume is `V_eff = V_dry + V_abs` **by construction** — swelling *equals* absorbed
volume, total volume invariant with no separate balancing step (R2, R3). Saturation for the cohesion
curve is *derived* from `V_abs`:

```
V_cap   = r_max · (ρ_s/ρ_w) · V_dry         # per-grain capacity (volume of water it can hold)
s       = clamp(V_abs / V_cap, 0, 1)         # saturation fraction (drives cohesion)
V_eff   = V_dry + V_abs                       # swelling = absorbed volume, exactly
d_eff   = grain_diameter · (V_eff/V_dry)^(1/3)
```

**Single water-particle volume** is fixed by the PBF density convention: `V_w = particle_mass /
rest_density` (Codex R2 — use this same density everywhere volume and mass appear, so they agree).

**Effective mass is tracked alongside volume (Codex R1.3).** Volume conservation alone does not give
momentum conservation — mass must move too. Absorbed water mass is `m_abs = ρ_w · V_abs`; so:

```
m_grain_eff(i) = m_dry + ρ_w · V_abs(i)      # grain gains the absorbed water's mass
m_water_eff(j) = particle_mass · f_w(j)      # water loses mass as it shrinks
```

These effective masses (not the constant `params.grain_mass` / full `particle_mass`) must be used
**everywhere a velocity/momentum/XPBD weight depends on mass** — the Phase 1.3 drag, buoyancy, and
exclusion opposite-mass weights included (see U6). **Momentum transfer is an exact per-substep
inelastic merge (Codex R2.4), not a Tang-style gradual fraction:** for the mass `dm = ρ_w·(volume
taken this substep)`, `v_g' = (m_g_old·v_g + dm·v_w)/(m_g_old + dm)`. The bounded rate law already
spreads absorption across substeps, so each substep merges exactly the mass it removed — no momentum
lag/leak. Under full conservation the swelling is *pinned* (α effectively = ρ_s/ρ_w); `ρ_s/ρ_w`
(roasted-coffee particle density ≈ 1.3) is a tunable `Param`. Directional guidance — exact field
names resolved at implementation.

### KTD-2 — Moisture/volume state on `pos.w`, mirrored to `pred.w` each frame

`pos.w` is persistent and currently unread (every kernel writes `vec4(xyz, 0.0)`; none reads `.w`),
and `pos` is bound to `bed_project`, `predict`, `apply_dp`, `finalize`, `xsph`. Repurpose `pos.w`
**phase-dependently**: grain → `V_abs`, water → `f_w`. `predict` **copies `pos.w → pred.w`** for every
particle, so the predicted-state kernels that bind `pred` (binding 2) — `compute_lambda`,
`compute_dp`, `compute_fractions`, `drag_*`, `buoyancy_*`, `exclude_*` — read the moisture/volume
state from `pred.w` with **zero new bindings**, and `bed_project` reads it from `pos.w` (also already
bound). This is what keeps every full (8/8) kernel within budget.

Strict invariants (Codex R1.6, R2.3): **(a)** state is seeded at particle *creation* and on `reset()`
(water `f_w=1`, grain `V_abs=0`), never (re)initialized in `predict` — `predict` only mirrors
`pos.w→pred.w`; **(b)** *every* writer of `pos` OR `pred` **preserves `.w`** — the known writers are
`predict`, `apply_dp`, **`apply_drag_pred`** (Codex R2.3 — it currently zeroes `pred.w` and runs in the
drag/buoyancy subcycles *before* absorption, so it would wipe the mirror), `finalize`, and
`bed_project` if it ever writes pos; **(c)** the inactive (absorbed) water sentinel cannot be confused
with a valid `f_w∈[0,1]` or with `phase`; **(d)** tests cover `predict`, `apply_dp`, `apply_drag_pred`,
`finalize`, the grid sort, deactivation, and the render/readback path. A dedicated per-grain buffer is
rejected: `bed_project` is at 8/8.

### KTD-3 — Continuous-shrink absorption with water-normalized allocation (conservation-safe, atomic-free)

**Decision (Codex R1.2): continuous shrink, with the PBF density solve weighted by `f_w`.** Binary
water quantizes volume loss and makes noisy wet fronts; "binary for density, shrink for bookkeeping"
is physically inconsistent (over-counts density at the wet front) and is **removed**. `f_w` decreases
smoothly; a water particle deactivates (sentinel) only at `f_w ≤ ε`. `compute_lambda`/`compute_dp`
**weight each water particle's density contribution and λ gradient/denominator by `f_w`** (read from
`pred.w`, already bound) so a half-absorbed particle counts as half — consistent effective volume/mass
throughout.

**Over-subscription is the real conservation hazard, on BOTH sides (Codex R1.1 + R2.1):** independent
gathers let multiple grains claim the same water (per-water overfill) AND a single grain claim
`demand_g` from each of several waters (per-grain overfill → overshoots `V_cap`). Fix with a
**two-sided** deterministic per-pair allocation computed identically on both sides, off a **frozen
snapshot**:

```
demand_g  = (V_cap_g − V_abs_g) · (1 − exp(−k_abs·dt))      # bounded rate law (Codex R1.9), ≥ 0
N_w       = # ELIGIBLE grains in range of water w           # eligible = active ∧ in-range ∧ demand_g>0
N_g       = # ELIGIBLE waters in range of grain g           # eligible = active ∧ in-range ∧ f_w>ε
take_wg   = min( (f_w·V_w)/N_w , demand_g/N_g )             # two-sided cap (Codex R2.1)
```

Then `Σ_g take_wg ≤ f_w·V_w` (water-safe) AND `Σ_w take_wg ≤ demand_g ≤ deficit` (grain-safe: no
`V_cap` overshoot, even mid-step). **Frozen-snapshot semantics (Codex R2.2):** both passes read
`f_w`, `V_abs`, the active mask, `N_w`, `N_g`, and eligibility from the immutable `pred.w` snapshot
(set in `predict`); they write only their own `pos.w`. So the **water pass** (sums `Σ_g take_wg`,
writes its own new `f_w` to `pos.w`) and the **grain pass** (sums `Σ_w take_wg` into its own `pos.w`
`V_abs` + does the inelastic momentum merge) compute the *identical* per-pair number → water-loss ==
grain-gain to float tolerance, **no atomics, no over/under-subscription**. The eligibility predicate
is byte-identical in both passes. `N_w`/`N_g` come from a cheap `wet_count` pre-pass (new kernel,
ample headroom).

**Deactivation never discards or force-dumps (Codex R3.2):** volume moves *only* via the mirrored,
capped `take_wg` — so water-loss == grain-gain by construction. The earlier "give the entire
remainder" override is **removed**: it let water remove more than grains added (e.g.
`A_w=1.0, demand=0.99 → water −1.0, grain +0.99`, 0.01 leak) and could exceed a grain's `demand` cap.

**Two distinct thresholds, to avoid stranding a residual (Codex R4):** `absorb_roundoff ≪ pbf_eps`.
- **Absorption eligibility / count:** a water is eligible (and counted in `N_w`/`N_g`) iff active and
  `f_w > absorb_roundoff`.
- **Deactivation:** only when a capped transfer drives `new f_w ≤ absorb_roundoff` (natural roundoff;
  never forced). At `absorb_roundoff` ~machine scale the discarded residual is negligible → effectively
  exact conservation.
- **PBF skip:** `compute_lambda`/`compute_dp` skip water with `f_w ≤ pbf_eps` (perf/stability), but
  such water is **still absorption-eligible** (since `pbf_eps > absorb_roundoff`) — so a water left at
  `absorb_roundoff < f_w ≤ pbf_eps` keeps getting absorbed until it deactivates; it is **not stranded**.
  Its residual volume stays on the books in `pos.w` (still in the conservation sum) the whole time.

**`wet_count` is part of the frozen-snapshot contract (Codex R3.3):** it reads `pred.w` (not `pos.w`)
and runs before any absorption write, so `N_w`/`N_g`/eligibility are snapshot-consistent with the
`f_w`/`V_abs`/`demand` the two transfer passes use.

**PBF small-`f_w` guard (Codex R3.5):** beyond the `pbf_eps` skip above, the λ denominator is clamped
so a vanishing effective volume can't collapse the constraint; `dp` contributions scale smoothly to
zero as `f_w → 0`.

### KTD-4 — Saturation→cohesion curve as a pure `models/cohesion.rs` function

`cohesion(s) = c_max · bump(s, s_peak)` with `bump` zero at `s=0` and `s=1`, peak 1.0 at `s_peak`.
Ship the **piecewise-linear** rise/fall first (cheapest, adequate, tunable); the smooth Beta/Hermite
hump is a drop-in refinement. `c_max` is calibrated to a wet-dome angle-of-repose, not a physical Pa.
Defaults: `s_peak = 0.4`, `r_max = 1.5`, `c_max` tuned. The per-pair cohesion combiner is
**`min(c_i, c_j)`** (Codex R1.5) — mean would let one wet grain glue dry/noncohesive neighbors;
`min` (or `sqrt(c_i·c_j)`) prevents that. Replaces `params.dry_cohesion` in `bed_project`. Cohesion
scales inversely with grain size for free (finer grind → stronger relative cohesion) — a real lever.

### KTD-5 — Swelling drives α_s / permeability / contact; the packing clamp does NOT also rise

`compute_fractions` currently uses uniform `params.grain_volume` (`coupling.wgsl:41`); replace with
per-grain `V_eff` (from the neighbor grain's `pred.w`). `bed_project` uses `d_eff` (from `pos.w`)
instead of the constant `params.grain_diameter` for non-penetration distance and cohesion range.

**Live-porosity drag (corrects an earlier claim).** The drag rate must vary with *local* packing for
swelling to actually slow drainage — it does **not** happen for free. Today the drag rate is a single
global constant `params.drag_gamma`, baked at build from the *static* `Materials.porosity = 0.40`; the
live `α_s` feeds only the water density target, **not** the drag. So as written, swelling would tighten
density + contact but **drainage would not slow** — the bloom/drawdown mechanism wouldn't fire. Fix:
make permeability **local**. `compute_coupling_scale` (the drag-setup pass; binds 5/8, has headroom)
**reads live `α_s` (binding 13)** and computes a per-particle local permeability `k(φ_f = 1 − α_s)`
via Kozeny–Carman → a per-particle local drag rate. The 8/8 `drag_water`/`drag_grain` kernels then use
that per-particle rate (no new binding in the drag kernels — see delivery note). So as the bed wets
and swells, `α_s ↑ → φ_f ↓ → k ↓ → drag ↑` locally — drainage slows where it's wet.

A scoped Codex review (`.deliberate/codex_review_porosity.md`) **confirmed conservation is sound** —
a shared *symmetric* pair scalar `s_ij` with the opposite-mass weights gives exactly
`m_i·dv_i + m_j·dv_j = 0` — and added three refinements:
- **Blend permeability, not the rate (Codex porosity #3).** The per-pair rate must be a **resistance
  blend — harmonic mean of `k_i, k_j`, then convert to a rate** — NOT `min`/mean of the *rates*. Since
  `rate ∝ 1/k`, `min(rate)` would pick the higher-permeability / weaker-drag side, the opposite of the
  series-resistance behavior you want across a porosity gradient. (Conservation is unaffected by the
  choice; only the magnitude — but harmonic-`k` is the physical one.)
- **Stability clamps (Codex porosity #5).** K-C diverges as `φ_f → 0`, and `α_s` lags ~one substep
  (computed earlier in the step). Clamp `φ_f` away from 0 and cap `rate·dt` so a sharp local packing
  jump can't blow up the drag.
- **Don't lossily pack two values into one f32 (Codex porosity #4).** Carrying both the cap `c_i` and
  the local rate in the single `coupling_scale` f32 risks corrupting monotonicity/bounds (NaN/denormal).
  Prefer a dedicated delivery — a spare lane drag reads-but-doesn't-write, or recomputing local `α_s`
  inline in the gather — over bit-packing. Exact mechanism is an Open Q (drag kernels are at 8/8).

`params.drag_gamma` demotes to the γ coefficient feeding the local `k`, not the final rate.

**Swelling is the *sole* solid-fraction mechanism (Codex R1.4): do NOT also raise the packing clamp by
`0.74·(1+r_max)`.** Growing `V_eff` already raises local α_s; raising the clamp too permits impossible
over-packing (`0.74·2.5 = 1.85`). Keep `packing_limit` at the physical close-packing bound (~0.64–0.74)
and clamp α_s below 1.0. (Departs from GIC, which keeps radius fixed and raises the clamp instead —
we chose explicit swelling, so the clamp stays physical.)

### KTD-6 — A new "models" stage in the substep loop, mixed-scene-gated

Insert the wetting/swelling stage (`wet_count` → `wet_water` → `wet_grain`) after the bed contact
loop / `finalize` (`src/solvers/xpbd/mod.rs:1162`+), guarded by `mixed = has_water && has_grain`.
Single-species scenes skip it entirely (R6), so the next `predict` mirrors the updated `pos.w`.
Absorbed momentum is merged **exactly per substep** (the inelastic merge in KTD-1), not as a Tang
gradual fraction — the rate law already spreads absorption over substeps, so a fraction-on-top would
leak momentum (Codex R2.4). The solve within a substep sees the previous substep's moisture
(one-substep lag, like the Phase 1.3 contact lag) — acceptable.

---

## High-Level Technical Design

### Volume + mass conservation flow (one absorption step)

Three atomic-free passes; **all reads come from the frozen `pred.w` snapshot**, each writes only its
own `pos.w` slot (Codex R2.2):

```
wet_count: reads pred.w (frozen). N_w = #eligible grains in range of w ; N_g = #eligible waters
           in range of g   → scratch     # eligibility identical to the transfer passes

wet_water (each water w, f_w > absorb_roundoff):  # eligible grain = in-range ∧ demand>0
    total = 0
    for each eligible grain g in range:                            # all reads from pred.w snapshot
        demand_g = (V_cap_g − V_abs_g)·(1 − exp(−k_abs·dt))
        take_wg  = min( (f_w·V_w)/N_w , demand_g/N_g )             # two-sided cap (Codex R2.1)
        total   += take_wg
    pos.w(w) = f_w − total/V_w                                     # deactivate only if ≤ absorb_roundoff
                                                                   # (no force-dump — Codex R3.2/R4)
wet_grain (each grain g):
    dV = 0 ; p_abs = 0
    for each eligible water w in range:                            # SAME formula, SAME pred.w snapshot
        take_wg = min((f_w·V_w)/N_w , demand_g/N_g)
        dV += take_wg ;  p_abs += take_wg·ρ_w · v_w
    pos.w(g) = V_abs + dV
    m_old = m_dry + ρ_w·V_abs_snapshot ;  dm = ρ_w·dV              # m_old is PRE-absorption (Codex R3.4)
    v_g = (m_old·v_g + p_abs)/(m_old + dm)                         # exact inelastic merge (R2.4)

# INVARIANTS (U9):  Σ_w f_w·V_w + Σ_g V_abs == const (exact)   AND   Σ (m_eff·v) conserved
```

`take_wg` is the identical pure function of the frozen snapshot on both sides ⇒ water-loss ==
grain-gain exactly, no double-spend, no `V_cap` overshoot. The PBF solve already weighted water by
`f_w` during the iteration loop (KTD-3); absorption mutates `pos.w` only here, after `finalize`.

### Storage-buffer budget (the binding constraint)

| Kernel | Current | Change | After |
|---|---|---|---|
| `bed_project` | 8/8 | reads `V_abs`→`d_eff`/saturation via `pos.w` (already bound) | 8/8 ✓ |
| `compute_lambda` / `compute_dp` | 8/8 | weight water by `f_w` via `pred.w` (already bound) | 8/8 ✓ |
| `compute_fractions` | 5/8 | neighbor `V_eff` via `pred.w` (already bound) | 5/8 ✓ |
| `compute_coupling_scale` | 5/8 | **+`α_s` (b13)** → local K-C drag rate into `coupling_scale` | 6/8 ✓ |
| `drag_*` / `buoyancy_*` / `exclude_*` | 6–8/8 | effective mass + **local drag rate** via `pred.w`/`coupling_scale` (already bound) | ≤8 ✓ |
| `predict` | 6/8 | mirror `pos.w→pred.w` (already bound) | 6/8 ✓ |
| `apply_drag_pred` | 3/8 | preserve `pred.w` (Codex R2.3 — currently zeroes it) | 3/8 ✓ |
| `finalize` / `apply_dp` | 6/8 · 5/8 | preserve `pos.w` (no new binding) | ✓ |
| `wet_count`/`wet_water`/`wet_grain` (new) | — | pos, pred, vel, phase, grid, +N_w/N_g scratch | ≤ 8 ✓ |

The `pos.w`/`pred.w` aliasing is precisely what keeps every existing full (8/8) kernel within budget;
all new bindings live in the new absorption kernels, which have headroom. **Every `pred` writer
(`predict`, `apply_dp`, `apply_drag_pred`) must preserve `.w`** or the frozen snapshot is lost.

### Params growth

`Params` 240 B → **256 B** (16-byte aligned; 7 free scalar slots; 3 existing pad + 4 new). New
tunables (≤ 5): `r_max`, `s_peak`, `c_max` (wet cohesion scale), `k_abs` (absorption rate),
`rho_ratio` (ρ_s/ρ_w). **No** moisture-raised packing term (KTD-5). Both literals (`mod.rs` +
`common.wgsl`) stay byte-matched; add a `size_of::<Params>() == 256` assertion + ordered-field
comment.

---

## Output Structure

```
src/models/
  cohesion.rs        # MODIFY — replace for_saturation() stub with the curve
  wetting.rs         # CREATE — moisture/volume/swelling pure functions + capacity/rate laws
src/solvers/xpbd/
  coupling.wgsl      # MODIFY — compute_fractions uses per-grain V_eff
  bed.wgsl           # MODIFY — cohesion from saturation; contact uses d_eff
  common.wgsl        # MODIFY — Params fields; finalize/apply_dp preserve pos.w
  wetting.wgsl       # CREATE — wet_absorb pass(es) (concatenated after coupling.wgsl)
  mod.rs             # MODIFY — pipelines/bind groups, Params, step() stage, read_moisture hook
src/models/mod.rs    # MODIFY — Materials: r_max, rho_ratio, s_peak, c_max, k_abs
src/utils/config.rs  # MODIFY — wetting numerics (absorb rate, eps, packing coeff)
src/utils/buffers.rs # (moisture lane already exists; populate via particles())
```

---

## Implementation Units

### U1. `models/cohesion.rs` — saturation→cohesion curve

- **Goal:** Replace the `for_saturation()` stub with the bump curve (R4).
- **Requirements:** R4.
- **Dependencies:** none.
- **Files:** `src/models/cohesion.rs` (modify, incl. unit tests).
- **Approach:** `for_saturation(s) -> c_max · bump(s, s_peak)`; piecewise-linear bump, zero at 0 and
  1, peak at `s_peak`. Keep `dry()` returning 0 for the dry path. Parameters passed in or read from a
  config/materials struct (not hardcoded).
- **Patterns to follow:** `src/models/permeability.rs` (pure function + monotonicity test).
- **Test scenarios:** `cohesion(0) == 0`; `cohesion(1) == 0`; peak at `s_peak` is the maximum over a
  sweep; monotone rising on `[0, s_peak]`, monotone falling on `[s_peak, 1]`; scales linearly with
  `c_max`.
- **Verification:** curve has the documented shape; existing `models` tests green.

### U2. `models/wetting.rs` — moisture/volume/swelling pure functions

- **Goal:** CPU reference for capacity, saturation, swelling, and absorption rate (R1, R2, R3).
- **Requirements:** R1, R2, R3.
- **Dependencies:** none.
- **Files:** `src/models/wetting.rs` (create), `src/models/mod.rs` (register module).
- **Approach:** pure functions: `capacity(V_dry, r_max, rho_ratio) -> V_cap`; `saturation(V_abs,
  V_cap)`; `effective_volume(V_dry, V_abs)`; `effective_diameter(d_dry, V_eff, V_dry)`;
  `grain_eff_mass(m_dry, V_abs, rho_w)`; `water_eff_mass(particle_mass, f_w)`; `absorb_demand(V_abs,
  V_cap, k_abs, dt) -> demand` using the **bounded** law `demand = (V_cap−V_abs)·(1−exp(−k_abs·dt))`
  (never overshoots, even at large dt — Codex R1.9). These mirror exactly what the GPU passes compute,
  so the conservation/swelling/mass math is unit-tested on the CPU first.
- **Patterns to follow:** `src/models/permeability.rs`.
- **Test scenarios:** capacity is `r_max·rho_ratio·V_dry`; `effective_volume` grows linearly with
  `V_abs` and `V_eff(V_abs=0)==V_dry`; `effective_diameter` is cube-root scaling; `saturation` clamps
  to `[0,1]`; `absorb_demand` is rate-limited and **never exceeds the deficit** for any `dt`;
  **volume identity** `V_dry + V_abs == effective_volume`; **mass identity** `grain_eff_mass` rises by
  exactly `ρ_w·V_abs`.
- **Verification:** `cargo test` for `models::wetting` green.

### U3. Params + Materials + Config plumbing

- **Goal:** Carry the new tunables to the GPU, byte-matched (R7).
- **Requirements:** R7.
- **Dependencies:** U1, U2 (for the field set).
- **Files:** `src/solvers/xpbd/mod.rs` (Params struct + literal), `src/solvers/xpbd/common.wgsl`
  (Params), `src/models/mod.rs` (Materials), `src/utils/config.rs` (Config).
- **Approach:** grow `Params` to 256 B; add `r_max, s_peak, c_max, k_abs, rho_ratio` (5 fields — **no**
  moisture-packing term, KTD-5/Codex R1.4); add an `assert_eq!(size_of::<Params>(), 256)` guard and a
  Rust↔WGSL field-order comment. Materials gets the physical defaults (`r_max=1.5`, `rho_ratio≈1.3`,
  `s_peak=0.4`, `c_max` tuned); Config gets the numeric knobs (`absorb_rate k_abs`, and the **two**
  thresholds `absorb_roundoff ≪ pbf_eps` — Codex R4). Keep `packing_limit` at its physical value (do
  not scale by `r_max`).
- **Patterns to follow:** existing coupling fields added in Phase 1.3 (`grain_mass`…`water_grain_distance`).
- **Test scenarios:** `Test expectation: none — pure plumbing`, but the `size_of::<Params>() == 256`
  assertion is the guard.
- **Verification:** builds; existing suites green (defaults must leave dry/water/coupling behavior
  unchanged — `c_max` only matters once moisture > 0).

### U4. GPU moisture/volume state on `pos.w`, mirrored to `pred.w` + readback hook

- **Goal:** Per-grain `V_abs` / per-water `f_w` on `pos.w`, seeded at creation, preserved across the
  frame, mirrored to `pred.w` for the solve kernels (R2, KTD-2).
- **Requirements:** R2, R6, R7.
- **Dependencies:** U3.
- **Files:** `src/solvers/xpbd/common.wgsl` (`predict` mirrors `pos.w→pred.w`; `finalize` + `apply_dp`
  preserve `pos.w`), `src/solvers/xpbd/coupling.wgsl` (`apply_drag_pred` preserves `pred.w` — Codex
  R2.3), `src/solvers/xpbd/mod.rs` (seed `pos.w` at build in `seed_block`; **`reset()` reseeds** water
  `f_w=1`/grain `V_abs=0`; `read_moisture()` hook; populate `particles().moisture`).
- **Approach:** seed `pos.w` at **build/creation and on `reset()`** (water=1.0, grain=0.0) — **not** in
  `predict` (Codex R1.6); `predict` only copies `pos.w→pred.w`. Change every `pos`/`pred` write
  (`finalize`, `apply_dp`, `apply_drag_pred`) to carry `.w`. Choose an inactive-water sentinel that
  can't collide with a valid `f_w∈[0,1]` or with `phase`. Add `read_moisture()`. Until U5 lands,
  `pos.w` is inert → no behavior change.
- **Patterns to follow:** `read_phases()` / `read_positions()` readback path (`mod.rs:304-336`).
- **Test scenarios:** after seeding, `read_moisture()` returns 1.0 for water, 0.0 for grains; after N
  mixed-scene steps with no absorption wired, values unchanged — **specifically proving
  `apply_drag_pred` (which runs in the drag/buoyancy subcycle) preserves `pred.w`** and finalize/
  apply_dp preserve `pos.w`; sentinel round-trips the grid sort + readback; `reset()` restores seed
  moisture; single-species water + dry-bed suites byte-unchanged.
- **Verification:** water + bed + coupling suites green; `pos.w`/`pred.w` survive a multi-step mixed run.

### U5. GPU absorption passes — water-normalized transfer, atomic-free, conserving volume + mass

- **Goal:** Water→grain volume+mass transfer with bounded rate, momentum carry, deactivation, with no
  over-subscription (R1, R2, R5).
- **Requirements:** R1, R2, R5, R7.
- **Dependencies:** U4. **U9's conservation + multi-grain-competition tests are written first and
  drive this unit (Codex R1.8).**
- **Files:** `src/solvers/xpbd/wetting.wgsl` (create; concatenate after `coupling.wgsl` in `mod.rs`),
  `src/solvers/xpbd/mod.rs` (pipelines + bind groups + step() stage, mixed-gated; an `N_w` scratch
  buffer if used), `src/solvers/xpbd/common.wgsl` (weight water density by `f_w` in
  `compute_lambda`/`compute_dp` via `pred.w`).
- **Approach:** three atomic-free passes per KTD-3, **all reading the frozen `pred.w` snapshot**
  (Codex R2.2) — `wet_count` (reads `pred.w`, runs **before any absorption write**, writes `N_w` per
  water + `N_g` per grain to scratch — part of the snapshot contract, Codex R3.3), `wet_water` (each
  water sums `take_wg = min(f_w·V_w/N_w, demand_g/N_g)`, writes new `f_w` to `pos.w`; **deactivates
  only when the capped transfer drives `f_w ≤ absorb_roundoff` — never force-dumps**, Codex R3.2/R4),
  `wet_grain`
  (each grain recomputes the *same* `take_wg`, adds to own `V_abs`, and does the **exact per-substep
  inelastic merge** `v_g=(m_old·v_g+p_abs)/(m_old+dm)` where `m_old = m_dry+ρ_w·V_abs_snapshot` is the
  **pre-absorption** mass from the frozen snapshot, Codex R3.4). Two-sided cap (R2.1); bounded rate
  `(1−exp(−k_abs·dt))`; cap per-substep `V_abs` growth so `d_eff` doesn't jump (R1.9); identical
  eligibility predicate in all three passes. Weight PBF water density by `f_w` and **skip
  `f_w ≤ pbf_eps` water with a clamped λ denominator**, while absorption eligibility uses the smaller
  `f_w > absorb_roundoff` so a residual is never stranded (Codex R1.2 + R3.5 + R4). **Execution note:
  test-first against U9.**
- **Patterns to follow:** Phase 1.3 `exclude_water`/`exclude_grain` symmetric pair gathers and
  `drag_*` opposite-mass conservation (`coupling.wgsl`).
- **Test scenarios:** see U9. Plus: single water + single grain → `V_abs` rises rate-limited toward
  `V_cap`, `f_w` falls by the matching volume; **one water shared by 3 grains → Σ grain gain ≤ that
  water's `f_w·V_w`**; **one grain near 3 waters → grain gain ≤ `demand_g`, never overshoots `V_cap`**
  (Codex R2.1); momentum conserved with effective masses; saturated grain absorbs nothing.
- **Verification:** both-sided conservation + competition tests pass; finest-grind / high-`r_max` stable.

### U6. Swelling + effective mass into α_s, permeability, contact, and the coupling forces

- **Goal:** Swollen `V_eff`/`d_eff` drive exclusion/permeability/contact, and **effective masses** feed
  the Phase 1.3 coupling forces (R3, R5).
- **Requirements:** R3, R5.
- **Dependencies:** U5.
- **Files:** `src/solvers/xpbd/coupling.wgsl` (`compute_fractions` uses neighbor `V_eff` from
  `pred.w`; **`compute_coupling_scale` reads live `α_s` (binding 13) → per-particle local K-C
  permeability → local drag rate**; `drag_*` use the per-particle rate via `coupling_scale`;
  `drag_*`/`buoyancy_*`/`exclude_*` use effective masses from `pred.w`), `src/solvers/xpbd/bed.wgsl`
  (non-penetration distance + cohesion range use `d_eff`; effective inverse-mass split).
- **Approach:** read neighbor `pred.w` → `V_eff` in `compute_fractions`; clamp α_s at the **physical**
  `packing_limit` (NOT raised by moisture — KTD-5); derive `d_eff` per grain pair in `bed_project`;
  replace the constant `params.grain_mass`/`particle_mass` in the opposite-mass weights of
  `drag_delta_for_pair`, buoyancy `/m`, and exclusion with `m_grain_eff`/`m_water_eff` derived from
  `pred.w`. **`bed_project`'s grain-grain non-penetration + cohesion split must also use effective
  inverse masses** (currently equal ½/½; wet grains have unequal mass so the heavier moves less, or
  grain-grain momentum is wrong outside absorption — Codex R3.7); reads effective mass from `pos.w`
  (already bound). **Live-porosity drag (KTD-5):** `compute_coupling_scale` reads `α_s` (it has
  headroom: 5/8 → 6/8) and computes per-particle local `k(φ_f = 1−α_s)`; the 8/8
  `drag_water`/`drag_grain` consume it (delivery is Open Q2 — not lossy-packed), combining the pair as
  a **harmonic mean of `k_i, k_j` → rate** (resistance blend, not `min` of rates — Codex porosity #3),
  which keeps `m_i·dv_i+m_j·dv_j=0`. **Clamp `φ_f` away from 0 and cap `rate·dt`** (K-C diverges at low
  porosity; `α_s` lags a substep — Codex porosity #5). `params.drag_gamma` becomes the γ coefficient
  feeding local `k`, not the final rate.
- **Patterns to follow:** `compute_fractions` (`coupling.wgsl:18-47`), `compute_coupling_scale` +
  `drag_delta_for_pair` + `bed_project` (`bed.wgsl:17-88`), `models/permeability.rs` (K-C).
- **Test scenarios:** wetted (swollen) grain block raises local α_s vs dry; **drainage through a
  wetted/swollen bed is slower than dry at the same grind because local `k` drops (live-porosity drag
  actually fires, not just density)**; a denser-packed region drains slower than a looser one in the
  same scene (spatially-varying permeability); swollen grains push apart more in contact; drag/buoyancy
  on a heavy wet grain differs from a dry grain (effective mass); a wet (heavy) grain contacting a dry
  (light) grain pushes the light one more — grain-grain momentum conserved with effective masses (Codex
  R3.7); **drag momentum still conserved with the per-particle local rate** (symmetric pair rate);
  α_s stays < 1.0 at high `r_max`; dry-bed + dry-coupling suites byte-unchanged (uniform α_s ⇒ uniform
  rate ⇒ identical to the old global `drag_gamma` when porosity is the build-time 0.40).
- **Verification:** coupling suite green; swelling + local porosity observably slow drainage; drag
  momentum conserved.

### U7. Saturation cohesion into `bed_project`

- **Goal:** Replace the constant `dry_cohesion` with per-grain wet cohesion from U1 (R4).
- **Requirements:** R4.
- **Dependencies:** U1, U4.
- **Files:** `src/solvers/xpbd/bed.wgsl` (cohesion term reads saturation from `pos.w` → curve).
- **Approach:** compute `s` from the grain's `V_abs/V_cap`, evaluate the curve (same math as U1), use
  it in place of `params.dry_cohesion` in the cohesion falloff. **Per-pair combiner = `min(c_i, c_j)`**
  (Codex R1.5) so a single wet grain can't glue dry/noncohesive neighbors.
- **Patterns to follow:** existing cohesion term `bed.wgsl:65-69`.
- **Test scenarios:** dry bed (s=0) behaves exactly as today (cohesion 0); a partially-wet bed
  (s≈s_peak) holds together more (less spread / steeper) than dry; a flooded bed (s→1) loses the
  cohesive hold; a wet grain adjacent to a dry grain does NOT glue it (min combiner); conservation/
  coupling suites green.
- **Verification:** cohesion visibly tracks saturation; dry path byte-unchanged.

### U8. Wetting scene, viewer, and diagnostics

- **Goal:** A way to see and measure wetting/swelling/cohesion + the conservation invariant.
- **Requirements:** R1–R5 (observability).
- **Dependencies:** U5, U6, U7.
- **Files:** `examples/coupling_render.rs` and/or a small wetting example (offscreen frames),
  `src/ui/particles.wgsl` (color grains by saturation — dry brown → wet dark), `examples/water_app.rs`
  (a `SCENE=wet` option, optional).
- **Approach:** reuse the `pour_over`/`dam_through_sand` scenes; add a per-frame **volume-conservation
  readout** (Σ fluid volume + Σ solid volume vs initial) and a moisture-tinted render. Optional
  **wet-repose diagnostic**: does a damp poured pile hold a steeper angle than the dry ~20°?
- **Patterns to follow:** the Phase 1.3 `coupling_render` diagnostics + `set_grain_radius_scale`.
- **Test scenarios:** `Test expectation: none — example/diagnostic` (the numeric checks live in U9).
- **Verification:** frames show a wet front + swelling; conservation readout flat.

### U9. Conservation + behavior test suite

- **Goal:** Lock R1–R6 as executable invariants.
- **Requirements:** R1, R2, R5, R6.
- **Dependencies:** U4 (skeleton written before U5), then U5, U6, U7.
- **Files:** `tests/xpbd_wetting.rs` (create).
- **Approach:** micro-scenes via the two-particle / bed-column patterns from `tests/xpbd_coupling.rs`;
  read back via `read_moisture()` + `read_positions()` + `read_velocities()` + CPU volume/momentum
  reductions (using **effective** masses).
- **Execution note:** the conservation + multi-grain-competition tests are written before U5 and drive
  it (Codex R1.8).
- **Test scenarios:**
  - **Volume conservation (R2, gate):** total volume (Σ f_w·V_w + Σ V_abs + Σ V_dry) constant to
    **float tolerance, exactly** (no-discard deactivation — not merely within `ε·count`).
  - **Per-water competition (R2, Codex R1.7):** one water in range of several grains → Σ grain `V_abs`
    gain in a step never exceeds that water's `f_w·V_w`.
  - **Per-grain overfill (R2, Codex R2.1):** one grain in range of several waters → its `V_abs` gain
    ≤ `demand_g` and it never overshoots `V_cap`.
  - **Frozen-snapshot agreement (Codex R2.2):** total water volume removed == total grain volume gained
    for the step (the two passes agree), independent of dispatch order.
  - **`pred.w` mirror survives the subcycle (Codex R2.3):** in a mixed scene with drag/buoyancy active,
    `pred.w` still equals the frozen `pos.w` after `apply_drag_pred` (regression guard).
  - **No-discard / no-leak deactivation (Codex R3.2):** `A_w=1.0, demand=0.99` case → water loses
    exactly 0.99, grain gains exactly 0.99, residual 0.01 stays active and is absorbed next step; no
    force-dump, no leak.
  - **Small-`f_w` PBF stability (Codex R3.5):** a near-empty water particle is skipped by the density
    solve and does not blow up λ / dp as `f_w → 0`.
  - **No stranded residual (Codex R4):** a water left at `absorb_roundoff < f_w ≤ pbf_eps` (PBF-skipped)
    is still absorption-eligible and continues to be absorbed until it deactivates at `absorb_roundoff`
    — it is never permanently stuck on the books.
  - **Grain-grain effective-mass momentum (Codex R3.7):** a wet (heavy) grain contacting a dry (light)
    grain conserves Σ(m_eff·v) and moves the light grain more (effective inverse-mass split in
    `bed_project`).
  - **Mass + momentum conservation (R5):** Σ(m_eff·v) conserved to tolerance across absorption over
    several substeps (using grain `m_dry+ρ_w·V_abs` and water `particle_mass·f_w`).
  - **Density coupling (Codex R1.7):** a partially-absorbed water particle contributes proportionally
    less to PBF density (not full) — wet-front density is not over-counted.
  - **Deactivation accounting:** a fully-absorbed water particle is skipped by subsequent kernels (no
    ghost density), and the discarded `≤ε` residual is accounted for (conservation holds within
    `ε·count`).
  - **Wets + retains (R1):** a dry grain beside water gains moisture rate-limited toward `V_cap` and
    holds it after the water leaves; never overshoots `V_cap`.
  - **Saturated grain (R1):** a grain at `V_abs==V_cap` absorbs nothing further.
  - **Saturation cohesion (R4):** damp bed spreads less than dry; flooded bed loses cohesion; min
    combiner — a wet grain doesn't glue a dry neighbor.
  - **Swelling (R3):** wetted grain block raises α_s / slows drainage vs dry; α_s stays < 1.0 at high
    `r_max` (packing clamp not over-raised).
  - **Finest-grind / high-`r_max` stability (R-4):** no NaN/blow-up; swelling bounded per step.
  - **No regression (R6):** water-only and dry-bed metrics byte-unchanged.
- **Verification:** `cargo test` green including the new suite; existing suites unchanged.

---

## Implementation Order

Per Codex R1.8 (tests before the conserving kernel; mass/volume convention before its consumers):

```
U1, U2  (pure models: cohesion curve, wetting/mass/volume + bounded rate)
U3      (Params 256B + Materials + Config plumbing)
U4      (pos.w/pred.w state + read_moisture hook)        + U9 skeleton (conservation & competition tests)
U5      (absorption passes — driven to pass the U9 skeleton)
U6      (swelling + effective mass into α_s/permeability/contact/coupling forces)
U7      (saturation cohesion into bed_project)
U8      (scene/viewer/diagnostics)
U9      (full suite)
```

---

## Scope Boundaries

**In scope:** per-grain moisture/volume state, volume-conserving rate-limited absorption with
momentum transfer, grain swelling feeding α_s/permeability/packing/contact, saturation→cohesion curve,
moisture-tinted viewer + conservation diagnostic, the test suite.

### Deferred for later (later phases, per the docs)
- Extraction kinetics, two-pool dissolution, solute transport, TDS/yield (Phase 1.5).
- Thermal exchange / temperature-dependent kinetics (Phase 1.5).
- Bloom as CO₂ degassing source (Phase 1.6) — note the *swelling* part of bloom is in scope; the gas
  source is not.
- Fines as a separate species / fines migration (Phase 1.6).
- Evaporation / drying (moisture only rises this phase).

### Deferred to Follow-Up Work (plan-local)
- Smooth Beta/Hermite cohesion bump (ship piecewise-linear first).
- The V60 cone geometry (separate geometry phase; flat floor / open scenes for now).
- The discrete-removal absorption variant if continuous-shrink density-weighting proves problematic.

### Out of scope
- Raising the device storage-buffer limit (WASM portability — disallowed).
- Any float-atomic-based absorption.

---

## Risks & Mitigations

- **R-1 `pos.w`/`pred.w` aliasing.** Repurposing the `.w` lane is fragile — any kernel that zeroes it
  silently wipes moisture or the frozen snapshot. *Mitigation:* documented invariant at the binding;
  **every `pos`/`pred` writer preserves `.w` (incl. `apply_drag_pred`, Codex R2.3)**; seed at
  creation/reset, never in predict (KTD-2); U4's "survive N mixed steps + sentinel round-trips the
  sort" tests; consider a debug assert.
- **R-2 Density solve vs shrinking water — RESOLVED.** Continuous `f_w` shrink IS reflected: `predict`
  mirrors `pos.w→pred.w`, and `compute_lambda`/`compute_dp` (which bind `pred`) weight water density +
  λ by `f_w` (KTD-3/Codex R1.2). "Binary water for density" removed as inconsistent. Guarded by the U9
  density-coupling test.
- **R-3 Over-subscription (both sides) / drift — RESOLVED.** The **two-sided** `take_wg =
  min(f_w·V_w/N_w, demand_g/N_g)` cap guarantees Σ_g ≤ water availability AND Σ_w ≤ grain demand (no
  `V_cap` overshoot); both passes read a **frozen `pred.w` snapshot** and compute the identical per-pair
  amount → water-loss == grain-gain, no double-spend (KTD-3/Codex R1.1+R2.1+R2.2). No-discard
  deactivation makes it exact. Guarded by the U9 per-water + per-grain + frozen-agreement tests.
- **R-4 Swelling instability.** Large `r_max` (coffee ~1.5) means grains can swell ~2–3×; sudden
  `d_eff` growth could inject contact energy. *Mitigation:* bounded rate law `(1−exp(−k_abs·dt))`;
  cap per-substep `V_abs` growth; finest-grind/high-`r_max` stability test. (Momentum is the exact
  per-substep merge — not a gradual fraction — per R-6.)
- **R-5 Packing over-pack — RESOLVED.** Swelling via `V_eff` is the *sole* solid-fraction mechanism;
  the packing clamp is NOT raised by `0.74·(1+r_max)` (KTD-5/Codex R1.4). `packing_limit` stays
  physical; U6/U9 assert α_s < 1.0 at high `r_max`.
- **R-6 Effective-mass omission.** If kernels keep using constant `params.grain_mass`/full
  `particle_mass`, momentum tests pass only superficially (Codex R1.3). *Mitigation:* U6 threads
  `m_grain_eff`/`m_water_eff` (from `pred.w`) through drag/buoyancy/exclusion; U9 checks Σ(m_eff·v).
- **R-7 Params 256 B drift.** Rust/WGSL desync silently corrupts all uniforms. *Mitigation:* size
  assertion + ordered field comment (U3).
- **R-8 Live-porosity drag — momentum + stability.** A per-particle drag rate (local `k(φ_f)`) risks
  asymmetric pair impulses → momentum drift. *Mitigation (scoped Codex review confirmed sound):* the
  per-pair rate is a **symmetric resistance blend** — harmonic mean of `k_i, k_j` → rate (NOT `min` of
  the rates, which picks the weaker-drag side), giving exactly `m_i·dv_i + m_j·dv_j = 0`; U6 re-asserts
  drag Σ(m_eff·v) conservation. *Stability:* clamp `φ_f` away from 0 and cap `rate·dt` (K-C diverges as
  `φ_f→0`; `α_s` lags ~one substep). (KTD-5.)

---

## Open Questions

Resolved in Codex round 1 (now decisions): continuous-shrink + `pred.w`-weighted density (was Q1);
`min` cohesion combiner (was Q2); packing clamp stays physical (was Q3); water-normalized
`min`-cap allocation for multi-grain competition (was Q4). Remaining:

1. **`N_w`/`N_g` delivery** — the `wet_count` scratch-buffer pre-pass (both counts) vs a 2-ring
   recompute in `wet_water`/`wet_grain` (perf vs an extra binding in the new pass). Implementation-time;
   the new passes have headroom either way.
2. **Local-drag-rate delivery to the 8/8 drag kernels** (KTD-5) — `coupling_scale` already holds the
   cap `c_i`; the local rate needs a separate channel. Codex disfavors lossy bit-packing into one f32;
   candidates are a spare lane drag reads-but-doesn't-write, or recomputing local `α_s` inline in the
   drag gather (drag_water has grain neighbors; drag_grain would need care). Implementation-time;
   resistance blend stays symmetric (R-8). *(Conservation is settled; this is purely delivery.)*
3. **Wet-repose** — keep as a diagnostic, or promote to a gated test? (Default: diagnostic.)
4. **Per-substep swelling cap value** — how much `V_abs` growth/step before `d_eff` jumps inject
   contact energy; tune empirically against the finest-grind stability test (R-4).

---

## Sources & Research

- Tang et al. 2025, *The Granule-In-Cell Method for Sand–Water Mixtures* (already cited in
  `solver_xpbd.md`) — grain mass/absorption (Eq.19), gradual absorbed-momentum (Eq.20), grid-deficit,
  capillary saturation modulation. **Most implementable reference.**
- Tampubolon et al. 2017, *Multi-species simulation of porous sand and water mixtures* — saturation→
  cohesion peaking ~40%, collapse at full saturation; grid-deficit origin.
- Bishop effective stress / Lu & Likos 2010 — rigorous origin of the cohesion hump (apparent cohesion
  from suction), justifies the curve shape.
- Lucas–Washburn / Green–Ampt — wet-front `ℓ ∝ √t` (justifies rate-limited absorption + emergent
  slowdown).
- Coffee-specific: retained water `r_max ≈ 1.27–2.0` (×dry mass); packed-bed porosity ~0.4;
  permeability `k ≈ 1e-14–1e-13 m²`; bloom 15–45 s — sources in research notes.
- Repo: `KEEP.md` §1 (porosity 0.40, retention 42 ml, pore-transfer 4.0 s⁻¹, K-C form), §5 (settle/
  creep bands); `docs/plans/models.md` phase 2; `docs/plans/solver_xpbd.md` step 4.
- Internal note: GIC keeps grain radius fixed (mass-only); **explicit swelling is a deliberate,
  conservation-required departure** from the cited source.

---
status: active
type: feat
date: 2026-06-06
deepened: 2026-06-06
origin: docs/plans/solver_xpbd.md (build phase 6 / step 6), docs/plans/models.md (fines open question)
title: "feat: Phase 6 — fines migration + transient permeability + channeling"
---

# feat: Phase 6 — Fines Migration + Transient Permeability + Channeling

## Summary

Add **fines migration** to the XPBD solver and make **channeling** observable — the first
two of the three Phase 6 (`solver_xpbd.md` step 6) phenomena. Bloom/CO₂ is deferred to a
follow-up plan.

Coffee *fines* (broken-cell fragments, dust) detach from grains under flow, ride the draining
water, and re-deposit where the flow slackens — locally clogging the pore space, lowering
permeability, and **slowing drawdown**. That same redistribution is what makes a non-uniform
bed *channel*: fast paths erode clear (permeability rises) while slow regions silt up
(permeability falls), a self-reinforcing loop. Channeling is therefore **emergent**, not a
scripted mechanism — this plan adds the physics that lets it emerge and the **metrics**
(`evenness`, `drawdown_time`) that let a test see it.

Fines are modeled as a **massless clogging/transport scalar** (KTD-4): they occupy
permeability-relevant pore volume but carry no inertial mass, so the transfer moves a conserved
scalar only — it writes no velocities, perturbs no momentum, and stays well inside the
8-storage-buffer budget. The work rides existing seams rather than inventing machinery:
- Fines live in the free `chem.w` lane (both species) — verified free and untouched by every
  pass; rides the existing chem cell-order reorder (`grid_reorder_b` copies the full `vec4`) for
  free. No new storage buffer.
- Erosion/deposition is **one conservation-safe two-sided scalar transfer** (count + grain +
  water passes over a frozen `chem` snapshot with an identical signed allocation), the shape of
  the wetting (`wetting.wgsl`) and dissolution (`extraction.wgsl`) trios — but volume-only, with
  no velocity writes.
- The permeability feedback is a gated **deviation term** folded into the existing per-particle
  `α_s` (`compute_fractions`), which drives the live Kozeny–Carman drag factor. To make
  channeling actually *amplify*, the drag pair-combiner is sharpened from `min(rate)` to a
  **harmonic-mean-k blend** (KTD-7) so clog barriers aren't leaked at their edges.
- Suspended fines **advect for free** on the carrying water particle (Lagrangian, like solute
  `c`) — no transport pass.

Everything is gated off by default (`Config.fines_rate = 0`), so existing scenes and the test
suite stay byte-unchanged.

---

## Problem Frame

`solver_xpbd.md` lists three Phase 6 phenomena under one gate: *"channeling emerges from flow
non-uniformity; fines slow drawdown; fresh coffee blooms."* The solver today has none of them
(confirmed: no `fines` channel, no concentration-gradient force, no freshness/CO₂ state) — but
it has the **hooks**: drag already reads a live per-particle porosity (`α_s`) and converts it to
a local permeability via Kozeny–Carman (`coupling.wgsl` `porosity_drag_factor`), and the
`Metrics` struct already declares `drawdown_time` and `evenness` (both currently hard-zero,
never computed — `src/engine/state.rs:11-12`).

This plan delivers **fines migration + channeling** (the physically coupled pair — channeling
emergence *depends on* fines redistributing local permeability). Bloom is independent and
deferred.

**Origin divergence (see KTD-3).** `solver_xpbd.md:33` frames fines migration as interphase
force #4, a "concentration-gradient force [that] drives grain dispersion down the α_s gradient."
This plan models fines as a sub-grid transported scalar instead. Per the Codex cross-review,
these are **complementary, not interchangeable**: the scalar model captures pore clogging and
washout (what "fines slow drawdown" needs); force #4 captures mechanical rearrangement of the
*grain skeleton* (granular segregation, solid diffusion down a concentration gradient). This
plan does the former and does **not** claim to deliver force #4 — which, with force #3
(virtual/added mass), remains unbuilt and out of scope (see Scope Boundaries).

---

## Requirements

Grouped by capability. IDs are stable across revisions.

**Fines transport & conservation**
- R1. Grains carry a conserved fines inventory (seeded from a material fines fraction); fines
  erode into overlapping flowing water and re-deposit into grains where flow is slow, modeled
  as a transported scalar — **not** a new particle species and **not** a grain-dispersion force.
- R2. Every fines transfer conserves **total fines volume** to float tolerance — a hard
  constraint, including the long-run asymptotic tail (no one-signed roundoff sink, per the
  saturated-tail leak class). Fines are massless (KTD-4), so the transfer writes no velocities
  and leaves water/grain momentum unperturbed; the conservation gate is volume, plus a
  momentum-non-perturbation guard.
- R3. Suspended fines advect on their carrying water particle (Lagrangian), with no numerical
  diffusion pass; fines on a water particle that is deactivated (absorbed, `f_w→0`) or drained
  are not silently lost (R-handled via KTD-10).

**Transient permeability & channeling**
- R4. Net-accumulated fines locally raise solid fraction `α_s` (lower permeability) and
  net-eroded regions lower it, feeding the drag/Kozeny–Carman factor — so enabling fines
  measurably **increases whole-bed drawdown time** versus fines-off, all else equal.
- R5. The drag pair-combiner resolves a porosity discontinuity by **harmonic-mean-k**, not
  `min(rate)`, so a clog barrier is not leaked at its low-resistance edge — a precondition for
  channeling self-amplifying (KTD-7). The combiner stays symmetric (momentum-conserving).
- R6. Under a non-uniform pour/bed, flow non-uniformity (channeling) is detectable as a
  degraded `evenness` and is **amplified — not created** — by fines redistribution. No
  zone/scene/SDF-gated heuristic produces it.

**Observability**
- R7. The solver computes and exposes `evenness` (volume-weighted downward-flux uniformity) and
  `drawdown_time` in `Metrics`, replacing the current hard-zero placeholders.

**Inertness**
- R8. With `Config.fines_rate = 0` (default), the solver is byte-unchanged versus today: no new
  pass runs, `compute_fractions` takes its existing path, the drag combiner takes its existing
  `min` path, existing scenes and suites are unaffected.

---

## Key Technical Decisions

- KTD-1 — **Fines live in the `chem.w` lane, both species; no new buffer.** Verified in-code
  (Codex cross-review): `seed_chem` sets `.w = 0`; `dissolve_grain`/`dissolve_water`/
  `thermal_exchange` preserve `cf.w`; emitted water seeds `.w = 0`; `grid_reorder_b` copies the
  full `chem` `vec4` through `chem_scratch`. Grain `chem.w` = lodged-fines volume (seeded to
  `params.fines.y`); water `chem.w` = suspended-fines volume (seeded 0). Rejected: a new storage
  binding (the ceiling is tight — `drag_*`/`buoyancy_grain` are at 8/8) and lossy f32
  lane-packing (the porosity review's top correctness risk).
- KTD-2 — **Permeability feedback as a gated baseline *deviation*, applied past the packing
  clamp.** Only when `params.fines.x > 0`, `compute_fractions` adds the summed grain-neighbor
  fines deviation `Σ_j (chem[j].w − params.fines.y)·W` into the local solid fraction. Crucially
  it is added **after** the grain-skeleton packing clamp, not before: today the pass ends
  `alpha_s[i] = min(a_s, packing_limit)` (`coupling.wgsl:49`), and folding fines in before the
  clamp would erase positive clogging in already-packed regions while still letting negative
  (erosion) deviations open channels — an asymmetric artifact. So:
  `alpha_s[i] = clamp( min(a_s_grains, packing_limit) + fines_dev, 0.0, 1.0 − φ_f_min )`.
  Physically right: fines clog the *pore* space the grain packing limit leaves open, so clog can
  push `α_s` above `packing_limit` (toward the `φ_f` floor), while erosion lowers it. The
  calibrated bed permeability (`porosity = 0.40`) bakes in the baseline fines, so only
  migration-induced deviation changes drainage, and the fines-off path is literally the old code
  (R8). Budget-safe: `compute_fractions` storage today is `pred, cell_start, sorted_indices,
  phase, alpha_s` (5) → +`chem` = 6.
- KTD-3 — **Fines as sub-grid scalar transport, complementary to (not a replacement for)
  interphase force #4.** Diverges from `solver_xpbd.md:33` for the *clogging/washout* mechanism;
  does not deliver the *grain-skeleton rearrangement* force #4 would. Both can coexist later;
  force #4 is deferred, not subsumed. (See Alternatives.)
- KTD-4 — **Fines are a massless clogging/transport scalar.** They occupy permeability-relevant
  pore volume (the α_s deviation, KTD-2) but carry **no inertial mass**: the transfer moves
  `chem.w` only, writes no velocities, and leaves water/grain momentum untouched. Rationale:
  fines are a small mass fraction and drawdown-slowing is a *permeability* effect, not an
  inertial one; this also avoids a frozen-velocity snapshot (which would push a momentum-safe
  write pass to ~9 storage buffers, over the limit) and avoids threading `chem.w` into
  `water_eff_mass`/`eff_mass`/`grain_buoyancy_factor`/drag/exclusion/buoyancy/thermal (each of
  which keys off `pos.w` and several of which are already at 8/8). The conservation claim is
  therefore **volume-only**; the momentum check is a *non-perturbation* guard, not an
  effective-mass merge. (Inertial fines is a deferred enhancement.)
- KTD-5 — **One signed-net erosion/deposition transfer, volume-conserving.** A single trio
  (`fines_count`, `fines_grain`, `fines_water`) computes a per-pair signed transfer from a
  frozen `chem` snapshot using an identical water-normalized `min`-capped allocation on both
  sides (grain→water when local Darcy flux is high, water→grain when low). Each pass writes only
  its own `chem.w` slot (atomic-free). Symmetric roundoff floors on both lanes (the
  `wet_sat_cutoff` lesson) prevent a tail sink. Buffer/timing specifics:
  - **Read `pos`/`pos.w`, not `pred`.** The block runs *after* wetting (which writes `pos.w`);
    `pred.w` is the stale start-of-substep snapshot. Like the extraction block, fines reads the
    live post-wetting `pos`/`pos.w` so the deactivation cap (KTD-10) sees the current `f_w`.
  - **`fines_count` is eligibility-only (no `vel`).** It counts opposite-species neighbors in
    range (optionally checking `chem_frozen.w` pool state) — flux affects only the transfer
    magnitude/direction, not eligibility. Count buffers: `pos, phase, cell_start,
    sorted_indices, chem_frozen, wet_neighbors` (≤ 6).
  - **Transfer passes read `vel` for flux.** The flux signal is the local `|v_water − v_grain|`.
    `fines_grain`/`fines_water` = `pos, phase, cell_start, sorted_indices, chem_frozen,
    wet_neighbors, chem (write), vel` (= 8, at the ceiling like `drag_*`).
  - **Reuse the `wet_neighbors` binding/identifier directly** (binding 17, `u32`,
    `common.wgsl:133`) — do **not** declare a second `fines_neighbors` at the same binding (WGSL
    forbids it). Either reference `wet_neighbors` by name in the fines passes, or rename it
    globally to a neutral `neighbor_count`. Lifetimes don't overlap within a substep.
- KTD-6 — **`evenness` precise definition.** `evenness ∈ [0,1]`, from the coefficient of
  variation of **volume-weighted downward water flux** `max(-v_y, 0) · f_w · V_w` accumulated
  into horizontal cross-section bins over the bed region — **excluding** the inlet pour stream
  and the cup/outlet region, **ignoring** inactive water (`f_w ≤ pbf_eps`), with deterministic
  empty-bin handling. 1 = uniform; a channel shows as a high-flux column → low evenness.
  `drawdown_time` = time for the free-water column above the bed to clear a height/count
  threshold. Both are CPU-readback samplers (the `sample_extraction` pattern, `mod.rs:750`),
  refreshed by `sample_diagnostics`. No new reduction shader.
- KTD-7 — **Harmonic-mean-k drag pair-combiner (requires reworking `compute_coupling_scale`).**
  Today the drag pair scale takes `min(coupling_scale[i], coupling_scale[j])` where
  `coupling_scale` is a single `f32` holding a *capped, nonlinear* `β = beta_from_rate(...)`
  (`coupling.wgsl:206,212`). Since rate ~ 1/k, `min` selects the *weaker-resistance* side at an
  interface and leaks clog barriers. The fix is a harmonic mean of the per-pair **permeabilities**
  (dominated by the lower-k / higher-resistance side) — but it is **not** a simple swap on the
  existing `β`: `β` is nonlinear and the cap destroys invertibility, so harmonic-k can't be
  recovered from it. So `compute_coupling_scale` is reworked to store, per particle, the
  **uncapped local resistance/permeability and the cap** — by widening the existing
  `coupling_scale` buffer (binding 16) from `f32` to `vec2<f32>` = `(k_or_resistance, cap)`
  (same binding, no new buffer, drag stays ≤ 8). `drag_delta_for_pair` then forms the symmetric
  harmonic-mean-k, converts to `β`, and applies the (symmetric `min`) cap. Harmonic mean is
  symmetric, so pair antisymmetry — and momentum conservation — is preserved. Gated to the
  fines-active path (or applied generally with the existing drag/coupling suite re-verified) so
  default behavior is unchanged (R8). This is the porosity review's recommendation and is
  **required** for the channeling-amplification gate (R6), not deferred.
- KTD-8 — **One-substep lag on the fines→α_s feedback is acceptable.** The fines block runs at
  the end of a substep; `compute_fractions` reads it at the start of the next. Same lagged
  explicit/Jacobi coupling pattern already used for drag/contact, and the regime the porosity
  review blessed given Kozeny–Carman's low-φ blow-up is tamed (`porosity_drag_factor` clamps
  `φ_f ∈ [0.05, 0.999]`).
- KTD-9 — **New knobs.** `Config.fines_rate` (default 0) gates + scales the feature (mirrors
  `extract_rate`); `Materials.fines_fraction` (default 0.0) seeds the inventory. The WGSL
  `Params` gains a single `fines: vec4<f32> = (rate, seed, crit_flux, _)` — a clean 16-byte
  addition that fixes the accounting (the two existing `_pad_chem*` slots cannot hold the three
  new scalars).
- KTD-10 — **Fine-laden inactive/drained water.** Suspended fines on a water particle that gets
  absorbed to `f_w→0` or drains must not vanish. Mitigations: (a) cap a water particle's
  suspended fines by its **post-wetting active volume** (read live `pos.w`, per the KTD-5 timing
  rule — not the stale frozen `pred.w`) and force deposition to rise as `f_w → 0` (shed before
  deactivation); (b) the closed-system conservation gate runs with absorption **off** so no
  deactivation occurs (clean Σ invariant); (c) brew/draining scenes treat suspended fines on
  drained water as a **measured cup-fines sink** (accounted like yield), not an unaccounted leak.

---

## High-Level Technical Design

### Fines lifecycle (the self-reinforcing loop)

```mermaid
flowchart TB
  seed["grain chem.w = fines_seed S (uniform)"] --> flux{local Darcy flux<br/>|v_water − v_grain|}
  flux -->|high| erode["erosion: grain chem.w → water chem.w (scalar)"]
  flux -->|low| deposit["deposition: water chem.w → grain chem.w (scalar)"]
  erode --> advect["suspended fines advect on the<br/>water particle (free, Lagrangian)"]
  advect --> deposit
  deposit --> dev["grain fines deviation = chem.w − S"]
  dev --> frac["compute_fractions: α_s += deviation (gated)"]
  frac --> perm["porosity_drag_factor + harmonic-k pair blend: k↓ ⇒ drag↑"]
  perm --> slow["local drainage slows"]
  slow --> flux
  slow --> chan["clogged regions slow further,<br/>cleared paths stay fast ⇒ channeling amplifies"]
```

The loop is the point: erosion and deposition are flux-gated, the deviation feeds the
permeability factor, the harmonic-k combiner keeps the clog barrier sharp, and slower local
drainage changes the flux driving the next step. Channeling is the macroscopic signature of this
loop on a non-uniform bed. Fines carry volume into the loop, not momentum (KTD-4).

### Where the new pass block slots into `step()`

The fines transfer is its own gated block, after the extraction/thermal block (`mod.rs` step 9),
where post-finalize velocities give a settled flux signal — mirroring that block's structure but
writing only `chem.w`:

```text
per substep (existing):
  ... predict → water density loop → drag → buoyancy → bed → finalize → xsph
      → wetting (if k_abs>0) → extraction/thermal (if extract_rate>0)
  + NEW fines block (if fines_rate>0):
      grid rebuild → copy chem → chem_frozen
      fines_count → fines_grain → fines_water        # signed-net SCALAR transfer (no vel writes)

per substep (modified):
  water density loop → compute_fractions             # + gated fines deviation term (KTD-2)
  drag subcycle      → drag pair scale               # + harmonic-k combiner (KTD-7)
```

---

## Implementation Units

### U1. Fines state, seeding, gate, and CPU model

- **Goal:** Establish the massless fines scalar (`chem.w` both species), its seed, the opt-in
  knobs, the `Params` fines vec4, and a CPU model twin — no behavioral change yet.
- **Requirements:** R1 (state shape), R8 (inertness).
- **Dependencies:** none.
- **Files:**
  - `src/models/fines.rs` (new) — CPU twins: `fines_seed(grain_volume, fines_fraction)`,
    `net_rate(flux, crit_flux, rate) -> f32` (signed: + erodes / − deposits, single zero-crossing
    at `crit_flux`, bounded), with unit tests.
  - `src/models/mod.rs` — `pub mod fines;`; add `fines_fraction: f32` (default `0.0`) to
    `Materials` under a `// --- fines (Phase 6) ---` block.
  - `src/utils/config.rs` — add `fines_rate: f32` (default `0.0`) to `Config` (opt-in gate +
    rate scale; mirrors `extract_rate`).
  - `src/solvers/xpbd/common.wgsl` — add `fines: vec4<f32>` (rate, seed, crit_flux, _) to the
    WGSL `Params` mirror (do **not** try to reuse the two `_pad_chem*` slots — three scalars
    won't fit; KTD-9).
  - `src/solvers/xpbd/mod.rs` — extend the Rust `Params` mirror with the fines vec4; resolve
    `fines_seed = fines_fraction · grain_volume` at build; seed grain `chem.w = fines_seed`,
    water `chem.w = 0` in `seed_chem` (`mod.rs:476`); add a `read_fines()` helper (or document
    indexing `read_chem().w`).
- **Approach:** Pure plumbing. Re-confirm the cell-order reorder carries `chem.w` (it copies the
  full `chem` vec4 via `chem_scratch` — verify no writer drops `.w`).
- **Patterns to follow:** the `extract_rate` gate + pool-seed wiring; the `pos.w`
  phase-multiplex convention; CPU-twin modules under `src/models/`.
- **Test scenarios:**
  - Covers R8. *Inertness, byte-level:* a representative mixed scene stepped N frames with
    `fines_rate = 0` produces identical positions/velocities/chem to baseline (the new pass must
    not run; `compute_fractions` and the drag combiner take their old paths).
  - *Seed correctness:* after build with `fines_fraction > 0`, every grain `chem.w ==
    fines_seed` (±1e-6), every water `chem.w == 0`; total = `Σ grain · fines_seed`.
  - *CPU model:* `net_rate` is monotone in flux, crosses zero once at `crit_flux`, is bounded;
    `fines_seed == fines_fraction · grain_volume`.
  - Test expectation for the `Config`/`Materials`/`Params` field additions: none (pure
    declarations; covered by inertness + seed tests).

### U2. Erosion + deposition scalar transfer (volume-conserving, massless)

- **Goal:** Move fines grain↔water under a flux-gated signed-net scalar transfer that conserves
  total fines volume exactly and perturbs no momentum.
- **Requirements:** R1, R2, R3.
- **Dependencies:** U1.
- **Files:**
  - `src/solvers/xpbd/fines.wgsl` (new) — `fines_count` (eligibility-only: opposite-species
    neighbors in range over the frozen snapshot; **no `vel`**), `fines_grain`, `fines_water`
    (read the **frozen** `chem` snapshot + `vel` for the per-pair flux `|v_water − v_grain|`,
    compute an identical per-pair signed `take = clamp(net_rate(flux)·dt·…)` two-sided-`min`
    capped so neither lane over-draws, write only their own `chem.w`). All read live `pos`/`pos.w`
    (post-wetting), **not `pred`** (KTD-5). **No velocity writes** (massless, KTD-4).
  - `src/solvers/xpbd/mod.rs` — concat `fines.wgsl`; create the three pipelines + bind groups
    (counts per KTD-5: count ≤ 6, transfers = 8); add the gated fines block to `step()` after the
    extraction/thermal block (grid rebuild → `chem → chem_frozen` copy → `fines_count` →
    `fines_grain` → `fines_water`); reuse `chem_frozen` (binding 22) and the existing
    `wet_neighbors` binding/identifier (binding 17) for the count — no second declaration.
  - `tests/xpbd_fines.rs` — conservation gates below.
- **Approach:** Signed-net so one trio handles both directions: high local flux ⇒ erosion
  (grain→water), low flux ⇒ deposition (water→grain). KTD-10: deposition rate rises as the
  neighbor water's `f_w → 0` (shed before deactivation), and suspended fines are capped by active
  water volume. Symmetric roundoff floors on both lanes (the `wet_sat_cutoff` floor pattern).
  **Grid/timing invariant:** the block lives in the post-`finalize` window where
  `pos.xyz == pred.xyz` and wetting/extraction don't move `xyz`, so the shared (pred-keyed) grid
  build and the fines passes' `pos`-based neighbor reads agree. Keep fines here; if a future pass
  ever moves `pos.xyz` after `finalize`, this invariant must be revisited.
- **Patterns to follow:** the wetting trio `wet_count`/`wet_water`/`wet_grain` (`wetting.wgsl`)
  for the two-sided capped allocation over a frozen snapshot — but **drop** `wet_grain`'s
  inelastic-velocity merge (fines are massless); the dissolution trio (`extraction.wgsl`) for
  the `chem_frozen` copy-then-read sequencing.
- **Test scenarios:**
  - Covers R2. *Volume conservation (closed system, absorption OFF):* a sealed mixed scene,
    `fines_rate > 0`, stepped; `Σ(grain chem.w + water chem.w)` invariant to < 0.1%.
  - Covers R2. *Long-run saturated-tail gate:* ≥ 2000 steps driving strong transfer toward
    equilibrium; post-equilibrium drift slope ≈ 0 (pins the one-signed-ulp leak class). Reuse the
    wetting saturated-tail gate shape.
  - Covers R2. *Momentum non-perturbation:* with the fines block on, total water+grain momentum
    `Σ m_eff·v` (using the existing `pos.w`-based `eff_mass`) is **identical** to a fines-off
    control across the block — the fines passes must not touch `vel`.
  - Covers R1/R3. *Directionality:* a high-flux column drops grain `chem.w` and raises the
    co-located/downstream water `chem.w`; a stagnant region reverses — net scalar moves fast→slow.
  - Covers R3. *Deactivation safety:* a fine-laden water absorbed toward `f_w→0` deposits its
    fines first; `Σ` fines stays invariant (no stranded hidden volume).
  - *No-flux null:* water and grain at rest ⇒ net transfer ≈ 0 (no spurious erosion).
  - *Buffer budget:* each new pass's bind-group list is ≤ 8 storage buffers (documented counts;
    build does not panic).

### U3. Harmonic-mean-k drag pair-combiner

- **Goal:** Sharpen the drag pair scale so a porosity discontinuity is resolved by the
  higher-resistance side, making clog barriers hold and channeling self-amplify (R5/R6).
- **Requirements:** R5 (and a precondition for R6).
- **Dependencies:** none (independent of fines transport; parallelizable with U1/U2).
- **Files:**
  - `src/solvers/xpbd/coupling.wgsl` — rework `compute_coupling_scale` to store, per particle,
    the **uncapped local resistance/permeability and the cap** in a widened `coupling_scale`
    (`vec2<f32>` = `(k_or_resistance, cap)`), instead of a single capped `β`. Change
    `drag_delta_for_pair` to form the symmetric **harmonic-mean-k** from the two particles'
    stored `k`, convert to `β` (`beta_from_rate`), then apply the symmetric (`min`) cap — keeping
    `drag_water`/`drag_grain` equal-and-opposite per pair. Gate the harmonic path to
    `fines_rate > 0` (or apply generally and re-verify), so default behavior is unchanged (R8).
  - `src/solvers/xpbd/common.wgsl` / `src/solvers/xpbd/mod.rs` — widen the `coupling_scale`
    buffer (binding 16) element type to `vec2<f32>` (same binding, no new buffer; drag passes
    stay ≤ 8).
  - `tests/xpbd_coupling.rs` — combiner tests below (extend the existing coupling suite).
- **Approach:** The current path stores a *capped, nonlinear* `β` (`coupling.wgsl:206`) and
  pairs with `min` (`:212`); since rate ~ 1/k, `min` selects the lower resistance at an interface
  and leaks the clog barrier. Harmonic-mean-k can't be recovered from the capped `β`, hence the
  `vec2` widening to carry raw `k` + cap. Harmonic mean of `k` is dominated by the lower-`k`
  (higher-resistance) side — the correct upwinding for a clog — and is symmetric, so momentum is
  preserved. Smallest viable edit that keeps the anti-overshoot cap and the buffer budget.
  **Store-semantics caveat (easy to get wrong):** be consistent — either store `k` and take the
  harmonic mean directly, *or* store resistance `R = 1/k` and form `k_pair = 2/(R_i + R_j)`;
  do **not** harmonic-mean the resistance. Name the vec2 fields explicitly (e.g.
  `(k_or_factor, beta_cap)`) with a unit comment, and resize the `coupling_scale` buffer from
  `N·4` to `N·8` bytes. **Byte identity:** when `fines_rate == 0`, retain the exact old
  `min(capped_beta_i, capped_beta_j)` path (R8).
- **Patterns to follow:** `compute_coupling_scale` (`coupling.wgsl:166`) and
  `drag_delta_for_pair` (`coupling.wgsl:209`); the symmetric-pair-scalar requirement from
  `.deliberate/codex_review_porosity.md`.
- **Test scenarios:**
  - Covers R5. *Symmetry / momentum:* across a water↔grain pair the drag impulse remains exactly
    equal-and-opposite (momentum conserved to float tolerance) under the new combiner.
  - Covers R5. *Clog upwinding:* at a synthetic interface (one side high-α_s/low-k, one side
    low-α_s/high-k), the pair resistance tracks the **high-resistance** side (harmonic), not the
    low — assert versus the old `min` behavior on the same inputs.
  - Covers R8. *Default unchanged:* with `fines_rate = 0`, existing `tests/xpbd_coupling.rs`
    drag/buoyancy/conservation assertions still pass (gated path or re-verified).
  - *Stability:* a strong k-discontinuity does not blow up (β cap + φ_f clamp hold).

### U4. Fines → transient permeability coupling

- **Goal:** Make lodged-fines deviation change local permeability so accumulation slows drainage
  and erosion speeds it — producing R4 and (with U3) channeling amplification.
- **Requirements:** R4, R6 (amplification), R8 (gated).
- **Dependencies:** U2 (fines exist), U3 (sharp combiner for crisp amplification).
- **Files:**
  - `src/solvers/xpbd/coupling.wgsl` — two gated edits, both no-ops when `params.fines.x == 0`:
    (1) in `compute_fractions`, accumulate the summed grain-neighbor fines deviation
    `Σ_j (chem[j].w − params.fines.y)·W` and add it **after** the grain packing clamp
    (`alpha_s[i] = clamp(min(a_s_grains, packing_limit) + fines_dev, 0.0, 1.0 − φ_f_min)`, per
    KTD-2); (2) extend the live-porosity gate in `compute_coupling_scale` from
    `if (params.k_abs > 0.0)` to `if (params.k_abs > 0.0 || params.fines.x > 0.0)` — otherwise
    the fines-driven `α_s` change never reaches drag in a no-wetting scene.
  - `src/solvers/xpbd/mod.rs` — bind `chem` into the `compute_fractions` pipeline's bind group
    (→ 6 storage buffers, safe).
  - `tests/xpbd_fines.rs` — local-permeability test below.
- **Approach:** Reuse `α_s → porosity_drag_factor → drag` (KTD-2/KTD-7/KTD-8). The deviation is
  signed: clogged regions (`chem.w > S`) raise `α_s` (slow) — and, applied past the packing
  clamp, can push above `packing_limit` to model pore clogging — while eroded regions
  (`chem.w < S`) lower it (speed). One-substep lag intentional and safe (KTD-8). The `φ_f` clamp
  plus bounded `net_rate·dt` guard Kozeny–Carman's low-φ sensitivity.
- **Patterns to follow:** `grain_eff_volume(pred[j].w)` usage already in `compute_fractions`
  (`coupling.wgsl:44`); the `if (params.k_abs > 0.0)` gating style.
- **Test scenarios:**
  - Covers R4. *Fines slow drawdown:* identical brew run with `fines_rate = 0` vs `> 0`; the
    fines-on run has a strictly longer `drawdown_time` (uses U5's sampler). Causal lever, not a
    tuned absolute.
  - *Local cause:* a seeded over-clogged patch (`chem.w` written above `S` via a test-write
    helper) shows lower reconstructed downward water velocity through it than a baseline patch —
    isolates the permeability response from the transfer.
  - Covers R8. *Gated:* `fines_rate = 0` ⇒ drainage behavior matches baseline.
  - *Stability (high-clog clamp):* explicitly drive `α_s` past `packing_limit` toward the
    `1 − φ_f_min` ceiling (the numerically stiffest case — pore-water density target → its floor);
    assert no NaN/blow-up and bounded KE over a long run (the `φ_f`/pore-fraction floors and the
    correction cap should hold).

### U5. Brew-evenness & drawdown-time diagnostics

- **Goal:** Compute and expose `evenness` and `drawdown_time` in `Metrics`, replacing the
  hard-zero placeholders, so channeling and drawdown are observable.
- **Requirements:** R7.
- **Dependencies:** none (depends only on the solver; parallelizable with U1–U4).
- **Files:**
  - `src/solvers/xpbd/mod.rs` — add a `sample_channeling` CPU readback (positions + velocities +
    moisture + phases) computing `evenness` per KTD-6 (CoV of volume-weighted downward flux
    `max(-v_y,0)·f_w·V_w` across bed cross-section bins, excluding inlet + cup/outlet, ignoring
    inactive water, deterministic empty bins) and `drawdown_time` (free-water column clears a
    threshold); cache like `cached_yield`/`cached_tds`; call from `sample_diagnostics`
    (`mod.rs:743`); populate the fields in `metrics()` (`mod.rs:2280`).
  - `src/engine/state.rs` — no struct change; align field docs to the KTD-6 definitions.
  - `tests/xpbd_channeling.rs` (new) — diagnostic tests below.
- **Approach:** Pure observation, no physics change. Follow the existing CPU-readback sampler
  (the documented sync point tests already call).
- **Patterns to follow:** `sample_extraction` (`mod.rs:750`) readback + cache; the
  `read_positions`/`read_velocities`/`read_moisture`/`read_phases` helpers; the
  `sample_diagnostics` → `metrics()` flow (`tests/xpbd_extraction.rs:697`).
- **Test scenarios:**
  - Covers R7. *Wiring:* after `sample_diagnostics()`, `evenness ∈ [0,1]` and `drawdown_time ≥ 0`
    are finite and non-default for a real brew.
  - *Synthetic uniform field:* a test-written perfectly-uniform downward-flux field yields
    `evenness` at its analytic upper bound (CoV = 0 ⇒ 1) — a constructed bound, not a tuned
    "near 1".
  - *Synthetic channel field:* a single injected high-flux column yields a strictly lower
    `evenness` than the uniform field (paired inequality) — the metric responds to a channel.
  - *Drawdown ordering:* a coarser grind (faster drain) gives a strictly shorter `drawdown_time`
    than a finer grind — ties the metric to a known causal lever.

### U6. Fines + channeling integration gates

- **Goal:** Prove the gate end-to-end: fines slow drawdown, and channeling emerges from flow
  non-uniformity and is amplified by fines.
- **Requirements:** R4, R6 (and exercises R1–R3, R5, R7 in integration).
- **Dependencies:** U2, U3, U4, U5.
- **Files:** `tests/xpbd_channeling.rs` — integration scenarios; reuse `Scene::v60()` and the
  `mixed_scene` helpers.
- **Approach:** Build on the brew scene; seed non-uniformity (off-center pour, or a low-packing
  stripe) and measure metrics over a brew vs matched uniform controls. Assertions are
  **inequalities / causal directions**, not tuned targets (AGENTS.md doctrine).
- **Patterns to follow:** the brew-scene tests in `tests/xpbd_extraction.rs`; the
  conservation-as-gate idiom for the closed-system checks.
- **Test scenarios:**
  - Covers R4. *Fines slow drawdown (integration):* full brew, `fines_rate` on vs off →
    `drawdown_time` strictly larger with fines.
  - Covers R6. *Channeling emerges:* a non-uniform bed/pour produces lower `evenness` than a
    matched uniform run; the flow field shows a preferential high-flux path (corroborated via
    `read_velocities`).
  - Covers R6. *Fines amplify, don't create:* channeling is present without fines (pure flow
    non-uniformity) and **more pronounced** with fines on; the only knob is `fines_rate` (no
    zone-gating).
  - Covers R3/KTD-10. *Cup-fines accounting:* in a draining brew, fines leaving on drained water
    are accounted (closed-system Σ over the bed + measured drained sink balances) — no
    unaccounted leak.
  - *Regression guard:* the three pre-existing red tests on `rewrite`
    (`xpbd_extraction::no_extraction_without_opt_in`,
    `xpbd_extraction::brew_populates_finite_yield_and_tds`,
    `xpbd_wetting::wetting_with_full_solve_conserves_volume_and_stays_finite`) are no more red;
    this plan adds no new failures. Note whether the wetting-conservation red is the same
    saturated-tail class before extending near it.

---

## Alternatives Considered

- **Fines as interphase force #4 (grain dispersion down ∇α_s).** The `solver_xpbd.md` framing.
  Not the right primary mechanism for "fines slow drawdown by clogging": it moves the grain
  skeleton, not a sub-grid fines population, so it can't represent fines washing out of channels
  and silting up slow zones without distorting bed structure, and it's harder to conserve. Per
  the Codex cross-review the two are **complementary** — force #4 captures granular segregation /
  solid diffusion the scalar model omits — so it is deferred, not subsumed.
- **Inertial (mass-carrying) fines.** Physically more complete (suspended fines add water
  effective mass; transfer conserves momentum via an inelastic merge). Rejected for this plan
  (KTD-4): it needs a frozen-velocity snapshot (pushing a momentum-safe write pass to ~9 storage
  buffers, over the limit) and threading `chem.w` through every effective-mass path
  (`eff_mass`/drag/exclusion/buoyancy/thermal — several at 8/8). The mass is small and the gate
  is permeability-mediated, so massless is the right minimal call; inertial fines is a deferred
  enhancement if a momentum effect is ever needed.
- **Fines as a third particle species.** Physically literal but expensive (more particles,
  another phase in every neighbor loop, new buffers against the ceiling). The scalar model
  delivers the same observable at a fraction of the cost. (`models.md` already leans this way:
  "transport in the solver.")
- **Defer the harmonic-k blend.** Rejected (KTD-7): with the existing `min(rate)` combiner the
  clog barrier leaks at its low-resistance edge, so channeling would not reliably amplify — the
  blend is a precondition for R6, not a polish item.
- **`evenness` = extraction/concentration uniformity** (brew-quality flavor). Richer and more
  user-facing, but needs a full extraction brew and couples the metric to kinetics. Deferred;
  flow-uniformity is the direct, lighter signal for the "flow non-uniformity" gate.

---

## Scope Boundaries

**In scope:** massless fines erosion/advection/deposition (scalar transport), fines→transient-
permeability coupling, the harmonic-mean-k drag combiner, `evenness`/`drawdown_time` metrics, and
the tests proving fines slow drawdown + channeling emerges/amplifies.

### Deferred to Follow-Up Work
- **Bloom / CO₂ outgassing (the third Phase 6 phenomenon).** Its own plan. Decided model (per
  product direction): a **swelling source + CO₂ particle emission** — fresh grounds carry a CO₂
  inventory ≈ **1% of dry bean mass**, released on wetting as (a) a transient effective-volume
  source (reusing the swelling seam that already raises `α_s`/packing in Phase 1.4) and (b)
  emitted gas particles. Note for that plan: after this plan claims `chem.w` for fines on **both
  species**, grain `chem` lanes are **full** (`s_f, s_s, T_g, fines`) and water has only `chem.z`
  free (`c, T_w, free, fines`). So the per-grain CO₂ inventory will need a **new storage buffer**;
  water `chem.z` is the only spare lane.
- **Inertial fines** (momentum-carrying suspended fines) — only if a momentum effect is needed.
- **Interphase force #4 (grain-skeleton dispersion / ∇α_s)** — complementary mechanical effect,
  not delivered here.
- **Extraction-evenness metric** (brew-quality flavor of `evenness`).
- **Fines as a higher-`s_f` extraction source** (`models.md` open question). Not needed for the
  drawdown/channeling gate.

### Outside this phase
- **Interphase force #3 (virtual/added mass)** — still unbuilt in the coupling layer; tracked
  against `solver_xpbd.md` §3, not this gate.
- **Absolute yield/TDS calibration** — a separate calibration track.

---

## Risks & Dependencies

- **Conservation tail leak (high prior).** Two-lane transfers with mixed fraction/absolute units
  have leaked one-signed in the asymptotic tail before (the wetting `wet_sat_cutoff` fix).
  *Mitigation:* symmetric roundoff floors on both lanes (KTD-5) and the mandatory ≥2000-step
  saturated-tail gate (U2).
- **Fine-laden water deactivation/drain leak.** Suspended fines stranded when `f_w→0` or water
  drains. *Mitigation:* KTD-10 (cap by `f_w` + forced deposition before deactivation; closed
  conservation gate with absorption off; measured cup-fines sink for draining brews; U6 cup
  accounting test).
- **8-storage-buffer ceiling.** Resolved for the stated design: massless fines means the
  transfer writes only `chem.w` (no frozen-velocity), so passes are 6/8/8; `compute_fractions`
  gains `chem` → 6; the harmonic-k combiner edits existing drag passes without adding buffers.
  *Mitigation:* documented counts in U2/U3/U4; the build panics on a bind-group mismatch, so any
  regression is caught immediately.
- **Reorder desync.** Fines live in `chem.w`, and `chem` is already reordered (`grid_reorder_b`
  copies the full vec4) — but the reorder area has *uncommitted* working-tree edits
  (`common.wgsl`, `mod.rs`, `water.wgsl`). *Mitigation:* U1 verifies `chem.w` survives the live
  reorder; coordinate with that in-flight change.
- **Channeling still too weak even with harmonic-k.** *Mitigation:* `fines_rate` and
  `fines_crit_flux` are tuning levers; the gate is an inequality (emergence + amplification
  direction), not a magnitude.
- **Harmonic-k combiner perturbs existing calibration.** Changing the drag combiner can shift
  drawdown-vs-grind behavior. *Mitigation:* gate it to `fines_rate > 0` (or apply generally and
  re-verify the existing coupling suite); U3 asserts default-unchanged.
- **Kozeny–Carman low-φ blow-up** where fines clog hard. *Mitigation:* the existing `φ_f ∈
  [0.05, 0.999]` clamp plus bounded `net_rate·dt`.
- **Pre-existing red tests (×3 on `rewrite`).** Do not attribute Phase 6 failures to them, and do
  not let this work add new failures (U6 regression guard).

**Verification gates (AGENTS.md):** `cargo fmt --check`; `cargo clippy --all-targets -- -D
warnings`; `cargo test`; `cargo run --example phase0_noop`. Every new compute stage ≤ 8 storage
buffers; never `#[ignore]`/delete a test to go green; `todo!()` stubs compile inert.

---

## Sources & Research

- `docs/plans/solver_xpbd.md` — build phase 6 gate; the four interphase forces (KTD-3 divergence).
- `docs/plans/models.md` — fines open question ("pool init here, transport in the solver");
  Kozeny–Carman.
- `.deliberate/codex_review_porosity.md` — prior art for the permeability coupling: symmetric
  pair scalar for momentum, harmonic-mean-k over `min` at porosity discontinuities, clamp `φ_f`
  and cap `rate·dt`, avoid lossy lane-packing. Shaped KTD-1, KTD-2, KTD-5, KTD-7, KTD-8.
- `.deliberate/codex_review.md` — the gpt-5.5 cross-review of this plan (REVISE). Folded in:
  massless-fines decision (KTD-4), volume-only conservation + momentum-non-perturbation (R2/U2),
  documented buffer counts and `fines_neighbors` reuse (KTD-5), `chem.w`-freeness verification
  (KTD-1), the deactivation hole (KTD-10), harmonic-k brought in scope (KTD-7/U3), the `Params`
  fines-vec4 accounting fix (KTD-9), the precise `evenness` definition + law-based tests (KTD-6,
  U5), and softening the force-#4 claim to "complementary" (KTD-3).
- `.deliberate/codex_review_r2.md` — round-2 cross-review of the revised plan (REVISE,
  implementability gaps only). Folded in: read live `pos.w` post-wetting not `pred.w` (KTD-5/
  KTD-10), the `fines_count`-has-no-`vel` buffer fix (KTD-5), reworking `compute_coupling_scale`
  to a `vec2` `(k, cap)` so harmonic-k is implementable (KTD-7/U3), extending the
  `porosity_drag_factor` gate to `k_abs>0 || fines_rate>0` (U4), applying the α_s deviation past
  the packing clamp (KTD-2/U4), reusing the `wet_neighbors` binding/identifier rather than a
  second declaration (KTD-5), and fixing the water-`chem.w`-free-for-Bloom contradiction
  (Deferred Work).
- `docs/ideation/2026-06-05-physics-validation-test-suite-ideation.md` — CPU-twin + analytic-gate
  test doctrine; conservation-as-gate template; the fine-spacing/eruption lesson.
- In-repo templates reused: `src/solvers/xpbd/wetting.wgsl` (conservation-safe transfer trio),
  `src/solvers/xpbd/extraction.wgsl` (`chem_frozen` sequencing), `src/solvers/xpbd/coupling.wgsl`
  (`compute_fractions`, `porosity_drag_factor`, `compute_coupling_scale`, `drag_delta_for_pair`),
  `src/solvers/xpbd/mod.rs` (`sample_extraction` readback, `seed_chem`, `grid_reorder_b`),
  `src/engine/state.rs` (`Metrics`).
- External research: not run. Local patterns are strong and the physics (Kozeny–Carman, fines
  clogging) is settled; modeling is constrained by the existing solver architecture.

---
title: "feat: Water→grain dynamic-pressure (momentum-flux) impact coupling"
status: active
date: 2026-06-08
type: feat
plan_id: 2026-06-08-001
related:
  - docs/plans/2026-06-06-006-feat-water-grain-coupling-physics-plan.md  # the coupling this extends
  - docs/plans/2026-06-07-001-fix-pbf-wall-pressure-boundary-plan.md      # structural precedent (Akinci force pattern)
---

# feat: Water→grain dynamic-pressure (momentum-flux) impact coupling

## Summary

A center water pour does not excavate a crater in the coffee bed — it only swells (absorption) while grains drift slightly *up* (buoyancy). The existing water↔grain coupling has three forces — porosity (`α_s`) exclusion, Darcy drag, and a momentum-conserving buoyancy/pressure pair (`buoyancy_grain`/`buoyancy_water`) — but **none of them converts the water's downward velocity into a grain push**. Drag is cap-diluted and isotropic; buoyancy is a static-pressure flotation force. Measured: at first contact on a *fresh dry bed* the jet reaches the grains at ~8.5 (peak 13.4) units/s downward but grains respond at only ~0.34 (a ~30× loss), decaying to ≈0 within ~0.3 s as a thin water cushion self-forms.

This plan adds one new coupling: a **dynamic-pressure (momentum-flux) impact force**. When water *approaches* a grain, the pair exchanges a momentum-conserving impulse scaled by the **square of the normal approach velocity** (dynamic pressure ~ρv²). The v² law makes the fast jet dominate (10² ≫ 0.5²) while slow percolation and hydrostatic water contribute ~nothing — so it produces an impact crater **without** blocking percolation or making a static bed creep. It mirrors the existing `buoyancy_grain`/`buoyancy_water` equal-and-opposite pair and the wall-pressure-force precedent, reads only existing buffers (no new storage buffer), and is opt-in via a config scale defaulting to 0 so all existing scenes stay byte-unchanged.

---

## Problem Frame

**Observed:** No crater under a center pour; grains swell and float, never excavate. Confirmed on both a steady-state (filled) bed and a fresh dry bed.

**Diagnosed root cause (this session, by instrumentation):**
- The jet arrives fast (`-17` a couple units above the bed; `-24` on the water-only pool) and is *not* decelerating in flight.
- At the grain surface the water has already braked to `-0.5 … -1.3`; the braking momentum is absorbed by the **water** (PBF density solve redistributes it), and a cushion self-forms within ~0.3 s even on a dry bed.
- The water→grain forces that exist cannot recover the impact:
  - **Drag** (`drag_grain`, coupling.wgsl) is symmetric and capped at `drag_beta_max / max(n_i, n_j)`; because water is finer than grains, the grain side is diluted by the large water-neighbor count → the fast jet barely registers. Even at the grain surface the water is already slow, so no drag tuning recovers it.
  - **Buoyancy** (`buoyancy_grain`/`buoyancy_water`) is a *static-pressure flotation* force (uses `params.h`, isotropic) — it pushes grains *up*; cranking it makes the saturated bed float/creep.
  - **No collision** — the grain contact solve (`bed.wgsl:49`) is grain-grain only; U2 of the coupling plan deliberately removed the soft water↔grain exclusion to preserve percolation.
- **Parameter sweeps do not fix it:** grain_mass 10→2, drag_beta_max→0.97, drag_subiters→10, substeps→4, buoyancy→0.3 all left grain impact velocity at ~`-0.03`. This is a missing *mechanism*, not a calibration miss.

**The gap:** there is no term that turns water *velocity* into a grain force. The momentum-conserving pressure pair exists for the *static* PBF pressure; the *dynamic* (inertial) pressure of moving water onto the bed is unmodeled.

---

## Goal & Success Criteria

1. A center pour produces a visible **crater**: the grain surface under the impact is pushed **below** the surrounding rim (and below its own pre-pour level once the rim/excavated material is accounted for), and grain impact velocity at the center is clearly **downward** during the pour — on a fresh dry bed and on a wetted bed.
2. **Momentum is conserved** by the new pass: the per-pair impulse on grain and water is equal and opposite; total linear momentum change from the impact pass alone is ~0 (gate test).
3. **Percolation and ponding are not regressed** vs the current build (the v² gate makes slow flow negligible).
4. **No static-bed creep**: a hydrostatic saturated bed with no pour stays asleep / shows no net drift over a long run.
5. **No-bed and no-pour scenes are byte-unchanged** (the pass is gated off when `impact_scale == 0` and/or no grains are present).
6. **No eruption / blow-up**: grain and water speeds stay bounded under the pour; the existing suite stays green.

---

## Key Technical Decisions

> **Codex round 1 (REVISE) and round 2 (REVISE) — both folded in below.** R1: impact must not share the drag freeze/apply (overwrite); threshold upfront; explicit Δv formula; `max_speed` insufficient. R2: the stability cap must clamp the **shared pair scalar `s_pair` before the mass split** (a per-particle post-accumulation clamp re-introduces net momentum); the velocity sampling must be **stated explicitly** (resolved in KTD-9: the subcycle reads the **stored pre-`finalize` velocity** carried from the prior step — `predict` and `apply_dp` write only `pred`, and the density-solve velocity reduction is deferred to `finalize` at mod.rs:2434 — so the impact block *does* see the jet's velocity, though not literally `(pred−pos)/dt` nor the current substep's gravity); U4/U5 must not be **circular**; the `Params` layout choice must be **concrete**; and the helper needs an `r`/NaN guard + a shared water-eff-mass floor.

### KTD-1 — Dynamic pressure (v²) with an upfront smooth threshold (WGSL constants)
The impulse scales with the **square of the normal approach speed** gated through a smooth threshold:
`gate(approach) = approach² · smoothstep(V_IMPACT_MIN, V_IMPACT_FULL, approach)`, where `approach = max(dot(v_water − v_grain, n̂), 0)`. Rationale: physical impact pressure is ~½ρv²; the quadratic gives a ~400× jet/percolation separation; the `smoothstep` floor (above normal drawdown speed) ensures slow percolation and solver jitter contribute *nothing*, not merely *little* (R1 #4, #5). `V_IMPACT_MIN`/`V_IMPACT_FULL` are **WGSL constants** tuned in source during U2/calibration (not `Config` floats) — this keeps the only new `Params` float to `impact_scale` (KTD-6). The threshold lives in U2, **not** deferred. A linear drag term cannot separate these regimes.

### KTD-2 — Explicit per-pair velocity-impulse formula, with the cap on the SHARED scalar (conserving by construction)
Specify the formula exactly, and **cap `s_pair` before the mass split** so the equal-and-opposite property survives capping (R2 #1 — a per-particle post-accumulation clamp re-introduces net momentum and is forbidden):
```
s_raw    = impact_scale · w(r, coupling_h) · gate(approach) · params.dt
# stability caps applied to the SHARED scalar, inside the helper (KTD-5):
s_pair   = min( s_raw,
                approach,                                  # normal-reversal: impact at most STOPS approach
                (k · coupling_h / params.dt) / max(m_w_eff/M, m_g_eff/M) )   # CFL on the larger side
Δv_grain = +s_pair · ( m_w_eff / M ) · n̂        # M = m_w_eff + m_g_eff
Δv_water = −s_pair · ( m_g_eff / M ) · n̂
```
Same momentum-conserving idiom as `drag_delta_for_pair` (eff-mass split). Because `s_pair`, `gate`, and `n̂` are identical on both sides and the cap is on the *shared scalar*, `m_g·Δv_g + m_w·Δv_w = 0` per pair by construction — capping never breaks it. **No post-accumulation per-particle `|Δv|` clamp** (it cannot be made conserving across two independent gather passes).

### KTD-3 — One canonical pair helper; no duplicated sign logic; NaN + eff-mass guards
A single WGSL helper `impact_pair(water_idx, grain_idx) -> {s_pair, n̂}` defines `v_rel = v_water − v_grain`, `approach = max(dot(v_rel, n̂), 0)`, the gate, and the capped `s_pair`. **Both** passes call it with canonical (water, grain) roles — neither re-derives signs (R1 #2). Guards (R2 #5): compute `n̂` with an `r`-floor (`spiky_r_min`-style) so coincident/near-coincident particles don't NaN; use `water_eff_mass(...)` with the **same floor on both sides** (`max(water_eff_mass, particle_mass·pbf_eps)`, as `buoyancy_water` does) so an absorbed/near-empty water particle can't diverge or blow up `M`. Locked by an impact-pass-only conservation gate + a sign-flip test.

### KTD-4 — Its OWN freeze/apply block, once per substep (NOT inside the drag subiter loop)
**[R1 #1, #8.]** Every coupling pass writes `vel = vel_frozen + dv`; passes sharing a `vel_frozen` snapshot + apply point *overwrite* each other for the same species. The code already separates concerns: drag (mod.rs ~2324–2361) loops `drag_subiters` with its own freeze→drag_water→drag_grain→`apply_drag_pred`; buoyancy (~2363–2403) is a **separate** single freeze→passes→`apply_drag_pred`. Impact mirrors **buoyancy exactly**: its own `vel→vel_frozen` copy, then `impact_grain`/`impact_water`, then `apply_drag_pred`, **once per substep** after the buoyancy block. Once-per-substep (not Jacobi-looped) also avoids v²-overshoot around the approach/separation boundary.

### KTD-5 — Stability cap on the shared scalar before `apply_drag_pred` (max_speed is not enough)
**[R1 #7, tightened by R2 #1.]** `apply_drag_pred` moves `pred` by `params.dt·(vel − vel_frozen)` *before* `finalize`'s `max_speed` clamp, so an uncapped quadratic impulse can throw `pred` far in one substep. The cap is therefore applied **at the source, on `s_pair`** (KTD-2): the normal-reversal term `min(s_pair, approach)` ensures impact can at most *stop* relative approach (never reverse past rest), and the CFL term bounds the larger-side `|Δv|` to `k·coupling_h/dt`. Both act on the shared scalar → conserving. **Scope of the bound (R3 #2):** these cap each *pair*; they do **not** mathematically bound a particle's *accumulated* Δv over many neighbors. So the per-particle pre-`finalize` displacement is verified **empirically** by the U5 gate (under a strong pour), not proven — and if that gate trips, the fallback is a conserving neighbor-count normalization (mirror drag's `/max(n_i,n_j)` via `coupling_scale.y`), accepting some force dilution. The eruption test inspects pre-`finalize` `pred` displacement, not just final speed.

### KTD-6 — No new buffer; reuse the reserved `_pad_coupling0` slot; ONE new float; default 0 → byte-unchanged
**[R2 #4 — concrete decision.]** The passes read `pred`, `vel`/`vel_frozen`, `phase`, `cell_start`, `sorted_indices`, `params` and write `vel` — the **same binding set** as drag/buoyancy — so no new storage buffer (within the web 9-buffer limit). Exactly **one** new `Params` float, `impact_scale`, which **replaces the reserved `_pad_coupling0` slot** (common.wgsl:59 + Rust mirror) so `size_of::<Params>()` stays **336**, no field shifts, the assert is unchanged. The two thresholds are WGSL constants (KTD-1), not `Params` fields, so no further layout pressure. Dispatch is **fully skipped** when `impact_scale == 0.0` (gated `mixed && impact_scale > 0.0`), so no-bed / not-opted-in scenes are byte-identical.

### KTD-7 — Calibrate by the crater/no-creep/percolation inequality (no circular dependency)
Consistent with the coupling plan's reduced-unit posture: pick `impact_scale` and tune the `V_IMPACT_*`/`IMPACT_CFL_K` consts (web CenterPour) by the inequality "visible crater under the pour AND no static-bed creep AND percolation/ponding within A/B tolerance of the `impact_scale==0` build," not by matching an absolute pressure. **Sequencing (R2 #3):** U4 does *exploratory* calibration with a throwaway probe; **U5 then codifies** the chosen value with the committed gates. U4 does not depend on U5's committed gates, and U5 asserts at U4's chosen value — no cycle.

### KTD-8 — Why a new term, not a buoyancy/drag refinement (rejected alternatives)
**[Per Codex #6 — make this a tested choice, not an assertion.]** See *Alternatives Considered*. In short: de-diluting the drag cap would raise packed-bed resistance and hurt drawdown (it's a linear, isotropic term keyed on the *already-cushioned* surface velocity); impact-gating the buoyancy/`λ` pressure cannot recover the jet's momentum because the PBF density solve has **already redistributed** that momentum among water particles by the time it appears as `λ` near the bed. Only a term keyed on the water's *approach velocity* (before it's dissipated) recovers the impact — hence an additive dynamic-pressure pass.

### KTD-9 — It reads the stored pre-`finalize` velocity, which still carries the jet
**[R2 #2 / R3 #1 — explicit decision, verified against code.]** The "recover the jet before it's dissipated" rationale holds, but state the sampling precisely: `predict` (common.wgsl) reads `vel` and writes **only `pred`** (`pred = pos + dt·v + dt²·g`) — it does **not** write `vel`, so the current substep's gravity lives in `pred`, not `vel`. The water PBF loop's `apply_dp` (common.wgsl:576) also writes **only `pred`**. The density solve's velocity reduction `v = (pred − pos)/dt` happens in `finalize` (common.wgsl:657), dispatched at mod.rs:2434 — **after** the drag/buoyancy/impact subcycle. Therefore the velocity `impact_*` freezes/reads is the **stored velocity carried from the previous step's `finalize`** (which, for a pour particle, is its emission/fall velocity) — *unaffected by the current step's density solve and not yet cushioned*. It is **not** literally `(pred − pos)/dt` and does **not** include the current substep's gravity. That stored velocity still carries the jet's downward motion, so the impact term sees a fast approach. **Implementation note:** if calibration shows the drag/buoyancy blocks (which run just before) meaningfully pre-attenuate the approach, move the impact block to run **first** in the subcycle (its own grid build + freeze/apply, before drag). Default placement is after buoyancy; this is a calibration-time choice, not a correctness one.

---

## High-Level Technical Design

Where the new force sits in the per-substep mixed-scene pipeline (new passes in **bold**):

```
substep loop (mod.rs):
  predict (gravity → vel, pred)
  PBF water density solve (compute_boundary → compute_lambda → compute_dp → apply_dp)   # water only moves
  grain contact solve (bed.wgsl)                                                        # grain-grain
  DRAG block (mixed && drag_subiters>0):                                                # EXISTING, own block
      compute_coupling_scale            # α_s field + flow/wake signal (coupling-plan KTD-9)
      ×(drag_subiters):  freeze(vel→vel_frozen) → drag_water → drag_grain → apply_drag_pred
  BUOYANCY block (mixed && buoyancy_scale>0):                                           # EXISTING, own block
      freeze(vel→vel_frozen) → buoyancy_grain → buoyancy_water → apply_drag_pred
  **IMPACT block (mixed && impact_scale>0):                                             # NEW, own block (mirrors buoyancy)
      freeze(vel→vel_frozen) → impact_grain → impact_water → apply_drag_pred**          #   once per substep
  finalize (dead-band / wake / friction; max_speed clamp)
```

The impact block is a **separate** freeze/apply block (KTD-4), not interleaved into the drag subiter loop — otherwise its `vel = vel_frozen + dv` write would overwrite the drag deltas for grains.

Per-pair velocity impulse (directional guidance, not implementation spec — see KTD-2/KTD-3/KTD-5):

```
# canonical helper, called with (water, grain) roles by BOTH passes:
d        = x_grain − x_water;  r = max(length(d), spiky_r_min)   # r-floor → no NaN (KTD-3)
n̂        = d / r                                   # line of centers, water→grain
v_rel    = v_water − v_grain
approach = max(dot(v_rel, n̂), 0)                   # 0 unless water moves toward grain
gate     = approach² · smoothstep(V_IMPACT_MIN, V_IMPACT_FULL, approach)   # WGSL consts (KTD-1)
w        = kernel_weight(r, coupling_h)
m_w_eff  = max(water_eff_mass(...), particle_mass·pbf_eps);  m_g_eff = grain_eff_mass(...)   # shared floor
M        = m_w_eff + m_g_eff
s_raw    = impact_scale · w · gate · params.dt
s_pair   = min(s_raw, approach, (k·coupling_h/params.dt)/max(m_w_eff/M, m_g_eff/M))   # cap SHARED scalar (KTD-2/5)

Δv_grain = +s_pair · (m_w_eff / M) · n̂             # equal-and-opposite survives the cap
Δv_water = −s_pair · (m_g_eff / M) · n̂
```

At the stagnation point the jet approaches the central surface grains from above → push **down**; as water spreads it approaches the rim grains laterally → push **out**. Together: a crater with a raised rim.

---

## Implementation Units

### U1. Config + params plumbing for `impact_scale` (one float, reusing the reserved pad slot)

**Goal:** Introduce the single opt-in strength knob, defaulted off, wired to the GPU `params` with **zero layout shift**.
**Requirements:** KTD-6; success criteria 5.
**Dependencies:** none.
**Files:**
- `src/utils/config.rs` (add **one** field `impact_scale: f32` default `0.0`, in the "water/bed coupling (mixed scenes)" block)
- `src/solvers/xpbd/common.wgsl` (rename the reserved `_pad_coupling0: f32` slot → `impact_scale: f32`; add `V_IMPACT_MIN`/`V_IMPACT_FULL`/`IMPACT_CFL_K` as module WGSL `const`s near the other tunables)
- `src/solvers/xpbd/mod.rs` (mirror the WGSL `Params` rename + its construction; the `size_of::<Params>() == 336` assert is **unchanged**)
**Approach:** Reuse the reserved `_pad_coupling0` slot for `impact_scale` so `size_of::<Params>()` stays **336**, no field shifts, assert untouched (R2 #4 — concrete: exactly one float, thresholds are constants not Params fields). Keep WGSL/Rust field order in lockstep (known footgun).
**Patterns to follow:** `buoyancy_scale`/`drag_scale` declarations + plumbing; the `_pad_coupling0` reserved-slot convention; the `size_of::<Params>()` assert at `mod.rs:~112`.
**Test scenarios:** `Test expectation: none` — pure plumbing with default 0. The unchanged `size_of` assert + green suite is the layout-integrity proof.
**Verification:** Compiles native + wasm; full suite unchanged/green; `size_of::<Params>() == 336` holds; a scene with `impact_scale == 0` is byte-identical (spot-check an existing coupling/extraction test).

### U2. `impact_grain` + `impact_water` WGSL passes (canonical helper, threshold, Δv cap)

**Goal:** The dynamic-pressure, threshold-gated, momentum-conserving impact force — with the conservation contract and the stability cap built in from the start.
**Requirements:** KTD-1, KTD-2, KTD-3, KTD-5; success criteria 1, 2, 4, 6.
**Dependencies:** U1.
**Files:**
- `src/solvers/xpbd/coupling.wgsl` (one shared `impact_pair(water_idx, grain_idx)` helper + two `@compute` entry points; reuse `cell_*`, `w_poly6`/`spiky_grad`, `grain_eff_mass`/`water_eff_mass`, `eff_mass`)
**Approach:** Copy the structure of `buoyancy_grain`/`buoyancy_water` (27-cell neighbor loop, opposite-phase filter, `vel_frozen` read, `vel` write) over `coupling_h`. Implement the **one** canonical helper (KTD-3) returning `{s_pair, n̂}`: `r`-floored `n̂`, `approach = max(dot(v_water−v_grain, n̂),0)`, `gate = approach²·smoothstep(V_IMPACT_MIN, V_IMPACT_FULL, approach)`, and **`s_pair` capped on the shared scalar** `= min(impact_scale·w·gate·dt, approach, (k·coupling_h/dt)/max(m_w/M, m_g/M))` (KTD-2/KTD-5). Both passes call it with canonical (water, grain) roles — **neither re-derives signs**. Apply the eff-mass split: grain `+s_pair·(m_w/M)·n̂`, water `−s_pair·(m_g/M)·n̂`, with the shared `water_eff_mass` floor (`max(…, particle_mass·pbf_eps)`). **No** post-accumulation per-particle `|Δv|` clamp (breaks conservation, R2 #1). No barriers in non-uniform control flow (Tint/browser portability). **No wake-signal write** (owned by `compute_coupling_scale`; impact must not wake a hydrostatic bed — already ~0 by the threshold).
**Patterns to follow:** `buoyancy_grain` (coupling.wgsl:232) / `buoyancy_water` (coupling.wgsl:273) incl. the equal-and-opposite per-pair comment (lines 299–301); `drag_delta_for_pair` for the eff-mass split + per-pair cap idioms.
**Test scenarios:**
- *Momentum conservation (gate):* fast water approaching a grain cluster — Σ(mass·Δv) over all particles from the impact pass alone is ~0 (fp tolerance). Covers criterion 2.
- *Canonical-role / sign:* the helper yields the same `s_pair` and `n̂` regardless of which pass calls it; a **sign-flip** mutation makes the conservation and crater tests fail (proves no duplicated sign logic).
- *Directionality:* water approaching a grain from directly above → **downward** Δv on grain, decelerating Δv on water; magnitudes obey the mass split.
- *Approach gate:* separating water (`v_rel·n̂ < 0`) → **zero** impulse both sides.
- *Threshold:* approach below `v_impact_min` → ~zero impulse; above `v_impact_full` → full v² law; smooth in between (no discontinuity).
- *v² scaling:* doubling approach (well above the threshold) quadruples the impulse.
- *Capped + still conserving:* an extreme approach speed yields a bounded `s_pair` (≤ both the CFL and the `approach` reversal caps) AND the impact-pass-only momentum sum stays ~0 (proves capping the shared scalar didn't break the pair).
- *NaN guard:* a coincident water/grain pair (r→0) produces a finite (non-NaN) result.
- *Static water:* near-zero-velocity co-located water/grain → ~zero impulse (no creep seed).
**Verification:** Unit tests pass; pass is a no-op when `impact_scale == 0`.

### U3. Dispatch as its own freeze/apply block (once per substep, after buoyancy)

**Goal:** Run the impact passes in a dedicated block so they don't overwrite drag/buoyancy deltas, once per substep.
**Requirements:** KTD-4, KTD-6; success criteria 5, 6.
**Dependencies:** U2.
**Files:**
- `src/solvers/xpbd/mod.rs` (pipeline + bind-group creation mirroring `buoyancy_grain`; a **new** `mixed && self.params.impact_scale > 0.0` block placed **after** the buoyancy block at ~2403, structured identically: `copy_buffer_to_buffer(vel→vel_frozen)` → `impact_grain` → `impact_water` → `apply_drag_pred`)
**Approach:** Add `impact_grain`/`impact_water` to the pipeline + bind-group structs and `make(...)`/`bg(...)` exactly as `buoyancy_grain` (same binding set → reuse the buoyancy bind-group layout; no new buffer, KTD-6). **Do not** insert into the drag subiter loop — that would clobber the drag deltas (Codex #1). The block runs **once per substep** (not looped), matching buoyancy, which also avoids v² Jacobi overshoot (Codex #8). Its own freeze ensures it composes additively with drag+buoyancy via their already-applied `pred`.
**Patterns to follow:** the buoyancy block at mod.rs ~2363–2403 (its single freeze→two passes→`apply_drag_pred` shape) — copy it verbatim and swap the two pass names + the gate condition.
**Test scenarios:**
- *No-bed byte-unchanged:* water-only / no-grain scene byte-identical before/after (block not dispatched). Covers criterion 5.
- *Opt-in:* `impact_scale == 0` → coffee scene byte-identical; `> 0` → grain motion changes.
- *Overwrite regression:* with drag + buoyancy + impact all enabled, each mechanism still contributes (e.g. disabling impact changes the result vs all-on; disabling buoyancy changes it differently) — proves the impact block didn't clobber the others. Covers Codex #1, #10.
- *Bounded pre-finalize (empirical):* under a strong pour, the max per-substep `pred` displacement (pre-`finalize`) stays within a sane bound (not just final speed). This is an empirical check — the per-pair cap doesn't prove the accumulated bound (R3 #2). Covers criterion 6, R1 #7.
**Verification:** Suite green; no-bed unchanged; the three coupling forces are independently load-bearing; pre-finalize displacement bounded.

### U4. Web CenterPour calibration

**Goal:** Turn the force on for the coffee scene at a value that craters without creep and preserves percolation/ponding.
**Requirements:** KTD-7; success criteria 1, 3, 4.
**Dependencies:** U3. (No dependency on U5 — calibration uses its **own throwaway probe**; U5 then codifies the chosen value. This breaks the R2 #3 circularity.)
**Files:**
- `src/web.rs` (`setup_for` CenterPour `Config`: set `impact_scale`; tune the `V_IMPACT_*`/`IMPACT_CFL_K` WGSL consts in `common.wgsl` if needed)
**Approach:** Exploratory calibration with a temporary in-test harness (not the committed U5 gates): sweep `impact_scale` + thresholds against the inequality (KTD-7) — visible crater AND no static-bed creep AND percolation/ponding within tolerance of the `impact_scale==0` build. Do not regress the shipped ponding/permeability settings (`substeps:2, drag_beta_max:0.92, drag_subiters:6`). Leave WaterOnly untouched (no grains).
**Patterns to follow:** the existing per-scene `Config` overrides in `setup_for`; the session's XSPH/substeps web-only scoping; the diagnostic-probe idiom used this session.
**Test scenarios:** `Test expectation: none` (config calibration) — behavior is **codified** by U5 at the chosen value. Record chosen `impact_scale`/threshold consts, crater depth, creep, percolation, ponding in the commit notes.
**Verification:** In the running web build, a center pour visibly craters; the static bed does not creep; drainage/ponding within tolerance; the chosen values are then locked by U5.

### U5. Verification suite (crater, conservation, creep, no-regression)

**Goal:** Lock the behavior and the invariants with headless GPU-gated tests.
**Requirements:** all success criteria.
**Dependencies:** U2 (conservation), U3 (dispatch), U4 (calibrated value to assert crater).
**Files:**
- `tests/xpbd_coupling.rs` or a new `tests/xpbd_impact.rs` (follow the existing GPU-gated test idiom: `GpuContext::new_headless()` early-return, `pour(...)`/`DT` helpers as in `tests/xpbd_emission.rs`)
**Approach:** Build the coffee scene at the calibrated config and assert the full gate set (Codex #10 — more than crater/no-crater):
**Test scenarios:**
- *Crater forms (fresh dry bed), two-sided + sign:* settle dry bed, pour at the web flow; central grain-surface percentile ends **below** the rim and below the `impact_scale==0` baseline (A/B in one test). Grain center vy clearly negative during early contact. Test must **fail** both when `impact_scale==0` AND when the per-pair `n̂`/sign is flipped. Covers criterion 1.
- *Momentum conservation gate:* (promoted from U2) Σ(mass·Δv) from the impact pass alone ~0.
- *Approach/separation sign + v² ratio:* (promoted from U2) separating water → 0; doubling approach → 4× impulse.
- *No static-bed creep:* saturated bed, **no pour**, long run → net grain displacement and mean speed within a tight dead-band; assert the impact pass alone produces zero-or-deadband motion on this scene (Codex #5). Covers criterion 4.
- *Percolation A/B:* percolation velocity under pour within tolerance of the `impact_scale==0` build (the gate that must pass before U4). Covers criterion 3, Codex #4.
- *Ponding A/B:* water retained above/in the bed not lower than the `impact_scale==0` build.
- *Bounded pre-finalize displacement (empirical):* max per-substep `pred` jump under a strong pour stays within a sane bound (inspect pre-`finalize`, not just final speed). Empirical, not a proof — per-pair caps don't bound the accumulation (R3 #2); neighbor-count normalization is the fallback if it trips. Covers criterion 6, R1 #7.
- *No-bed unchanged:* water-only scene byte-identical with the passes compiled in.
**Verification:** All gates pass; `cargo test` (native) green; the crater test has teeth (fails at `impact_scale==0` and on sign flip); percolation/ponding A/B within tolerance.

---

## Scope Boundaries

**In scope:** one new momentum-conserving water→grain dynamic-pressure coupling, its config knob, dispatch, web calibration, and verification.

**Out of scope / non-goals:**
- Hard water↔grain particle collision (re-adds what U2 removed; breaks percolation).
- Changing the drag cap-dilution or the buoyancy formulation (left as-is; the impact term is additive).
- Making the drag/buoyancy implicit, or changing substep counts beyond what shipped this session.
- The water-on-water splash feel (separate XSPH lever already shipped for WaterOnly).

### Deferred to Implementation
- Exact kernel weight `w` (poly6 vs spiky-derivative vs flat area term) and whether `n̂` uses line-of-centers or a smoothed grain-normal — pick during U2 by which gives the cleanest crater without jitter.
- Exact form/value of the KTD-5 caps (`k` in the CFL term; whether both the `approach` and CFL terms are needed) — pick during U2/U5 against the empirical bounded-displacement gate.
- **Aggregate-displacement fallback (R3 #2):** the per-pair caps don't bound a particle's accumulated Δv. If the U5 empirical displacement gate trips, add a conserving neighbor-count normalization (divide the pair CFL by `max(n_i,n_j)` from `coupling_scale.y`, mirroring the drag cap) — accepting some crater-force dilution. Only add if needed.
- Whether `impact_scale` folds the `coupling_h`/area normalization into a single tuned constant (KTD-7) vs sub-knobs (prefer the single constant; minimal surface).
- **Decided upfront (not deferred):** the smooth approach threshold (`v_impact_min`/`v_impact_full`) is in U2 per Codex #4, not a follow-up.

### Deferred to Follow-Up Work
- If percolation *still* regresses at high drawdown despite the KTD-1 threshold, restrict the term to surface grains (e.g. gate on `α_s` below a packed threshold) or raise `v_impact_min`; scope as a follow-up only if the U5 A/B gate trips.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Velocity-delta overwrite** — impact pass in the drag subcycle clobbers drag/buoyancy deltas (all write `vel = vel_frozen + dv`) | **High if mis-placed** | KTD-4: impact gets its **own** freeze/apply block after buoyancy, once per substep; U3 overwrite-regression test proves all three forces stay load-bearing (Codex #1). |
| Pair impulse not exactly equal-and-opposite (sign/branch divergence) → momentum leak, bed drift | Medium | KTD-3 single canonical `impact_pair` helper called with canonical roles by both passes; U2 conservation + sign-flip gates (Codex #2). |
| Underspecified Δv/mass-split → accidentally non-conserving | Medium | KTD-2 spells out the exact drag-style split + `dt` placement; conservation gate (R1 #3). |
| **Stability cap breaks conservation** if applied per-particle after accumulation | **High if mis-applied** | KTD-2/KTD-5: cap the **shared `s_pair`** inside the helper *before* the mass split (never a post-accumulation per-particle clamp); U2 "capped + still conserving" gate (R2 #1). |
| Impact samples cushioned residual, not the jet → no crater | Low (resolved) | KTD-9: subcycle reads the **stored pre-`finalize`** velocity (carries the jet); `predict`/`apply_dp` write only `pred`, cushioning deferred to `finalize` (after the subcycle). Verified in code (R2 #2 / R3 #1). |
| Per-pair caps don't bound a particle's **accumulated** Δv over many neighbors → pre-`finalize` displacement not mathematically bounded | Medium | Per-pair `min(s_pair, approach, CFL)` bounds each interaction; the U5 bounded-displacement gate is therefore **empirical** (tested under strong pour), not a proof. If it trips, add a conserving neighbor-count normalization (mirror drag's `/max(n_i,n_j)` from `coupling_scale.y`), accepting some force dilution (R3 #2). |
| Coincident water/grain → NaN; absorbed water → divergence | Low | KTD-3: `r`-floored `n̂` + shared `water_eff_mass` floor; U2 NaN-guard test (R2 #5). |
| U4↔U5 circular sequencing | n/a (resolved) | KTD-7: U4 calibrates with a throwaway probe; U5 codifies at the chosen value (R2 #3). |
| v² + approach gate still resists percolation (a downward column approaches grains below) → drawdown regression | Medium | KTD-1 **upfront** smooth threshold above drawdown speed; U5 percolation A/B gate **blocks** U4 calibration (Codex #4). |
| Static saturated bed creeps (impact + jitter/dead-band) | Medium | Threshold floor + impulse soft-knee (KTD-1); U5 impact-pass-alone no-creep gate; no wake-signal write (Codex #5). |
| Eruption from the quadratic force at high jet speed — **`max_speed` is not enough** (`apply_drag_pred` moves `pred` with unclamped `vel−vel_frozen`) | Med | KTD-5 explicit per-particle Δv cap (`k·coupling_h/dt`) **before** `apply_drag_pred`; U5 bounded pre-finalize-displacement gate (Codex #7). |
| v² overshoot/oscillation if sub-iterated | Low–Med | KTD-4 runs impact **once per substep**, not inside the Jacobi `drag_subiters` loop (Codex #8). |
| WGSL/Rust `Params` layout drift | Low | KTD-6 reuses the reserved `_pad_coupling0` slot → `size_of::<Params>()` stays 336; U1 asserts it (Codex #9). |
| Web device 9-buffer limit | None expected | KTD-6: no new buffer (reuses drag/buoyancy binding set); confirm bind-group reuse in U3. |

---

## Alternatives Considered

**[KTD-8 — recorded so the additive dynamic-pressure term is a tested choice, not an assertion (Codex #6).]**

- **De-dilute the water→grain drag cap.** Rejected as the primary fix: drag is linear, isotropic, and keyed on the *current* (already-cushioned) surface velocity — by the time water is among the grains it's slowed to ~-1.3, so even an un-diluted drag yields a shallow crater. Worse, raising the cap raises packed-bed resistance and would hurt drawdown (the very percolation we just calibrated). It cannot capture the jet's momentum *before* dissipation.
- **Strengthen / impact-gate the existing buoyancy (`λ`) pressure.** Rejected: buoyancy is a static pressure-gradient (flotation) force. The jet's momentum is **already redistributed among water particles by the PBF density solve** by the time it manifests as `λ` near the bed — so a stronger/`λ`-gated buoyancy amplifies hydrostatic flotation (and risks bed creep) without recovering the lost impact. Only a term keyed on the water's *approach velocity* (pre-dissipation) recovers it.
- **Hard water↔grain particle collision.** Rejected (out of scope): re-adds what coupling-plan U2 removed and blocks percolation (pores are sub-particle).

Conclusion: an **additive** dynamic-pressure term keyed on approach velocity is the minimal mechanism that recovers the impact while leaving drag, buoyancy, and percolation intact.

---

## Verification Strategy

- Per-unit tests as above; the headline U5 gates (two-sided+sign crater, momentum conserved, no creep, percolation/ponding A/B, bounded pre-finalize displacement, no-bed unchanged) must all pass.
- The crater test is **two-sided and sign-checked**: crater at the calibrated `impact_scale`, ~none at `impact_scale==0`, and failure on a flipped `n̂` — so it proves causation and can't pass with the force disabled or mis-signed.
- The percolation/ponding A/B gate **gates** the web calibration (U4 depends on it).
- Manual check in the running web build (CenterPour): visible crater, no creep, drainage/ponding intact.
- **Codex rounds 1, 2 & 3 (gpt-5.5 high): all REVISE — every finding folded.** R1: overwrite/threshold/formula/stability-guard. R2: cap on the shared scalar (conservation), velocity sampling, U4/U5 de-circularization, concrete `Params` layout, NaN/eff-mass guards. R3 confirmed all R2 fixes and corrected two: (1) KTD-9 wording — impact reads the *stored* pre-`finalize` velocity (not `v_prev+g·dt`); (2) the per-pair cap bounds each pair, not accumulated Δv, so the displacement gate is **empirical** with a neighbor-normalization fallback. The remaining open items are calibration values (U4) and the kernel-weight choice (Deferred), both implementation-time — no open plan-level correctness issues.

---

## Sources & Research

- This session's instrumentation (steady-state and fresh-dry-bed probes) establishing the ~30× transfer loss, the ~0.3 s cushion, and the parameter-sweep null result.
- `src/solvers/xpbd/coupling.wgsl` — existing `drag_*`, `buoyancy_*`, `compute_coupling_scale`, `apply_drag_pred`; the momentum-conserving pair idiom (lines 299–301).
- `src/solvers/xpbd/bed.wgsl:49` — grain-grain-only contact (no water↔grain collision).
- `docs/plans/2026-06-06-006-...coupling-physics-plan.md` — the coupling this extends (porosity + Darcy + buoyancy; reduced-unit calibration posture).
- `docs/plans/2026-06-07-001-fix-pbf-wall-pressure-boundary-plan.md` — structural precedent for adding an Akinci-style coupling force (staging, conservation, 16-buffer note).

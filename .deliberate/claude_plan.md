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

### KTD-1 — Dynamic pressure (v²), not a velocity drag
The impulse scales with the **square of the normal approach speed** (`(v_rel · n̂)₊²`), i.e. dynamic pressure ~½ρv². Rationale: the physical impact pressure of a jet on a surface is ½ρv²; and the quadratic law gives a ~400× separation between the jet (~10) and percolation (~0.5), so a single scale can produce a strong crater while leaving slow flow and hydrostatics untouched. A *linear* drag-style term cannot separate these regimes (it's what `drag_grain` already is, and it's cap-diluted). **Directional guidance, not spec:** `F_pair ≈ impact_scale · ρ₀ · A · (v_rel·n̂)₊² · n̂`, applied over `params.dt` as an impulse, with `n̂` the unit vector from water to grain (line of centers) and `A` a kernel/area weight. Exact kernel weighting (poly6 vs spiky, area normalization) is an implementation-time choice — see Deferred.

### KTD-2 — Approach-gated, so it never blocks percolation or creeps
The force fires **only when water moves toward the grain** (`v_rel · n̂ > 0`, approach). Separating or co-moving water contributes nothing. Combined with the v² law: hydrostatic water (v≈0) → 0; percolating water (v≈0.5 down, approaching grains below) → negligible; the jet (v≈10) → dominant. This is the percolation-safe "soft collision" — it is an *impulse on approach*, not a standing exclusion, so water still flows through the pores. **Risk to verify:** a downward-percolating column does approach the grains beneath it; confirm via the percolation no-regression test that the v² gate keeps this negligible at realistic drawdown speeds.

### KTD-3 — Momentum-conserving equal-and-opposite pair (mirror buoyancy)
Implement as two passes, `impact_grain` and `impact_water`, exactly mirroring `buoyancy_grain`/`buoyancy_water`: each computes its own side from `vel_frozen`, and the per-pair impulse uses a shared symmetric scalar with a reduced-mass / effective-mass split so the grain gets `+J` and the water gets `−J` for the same pair (the buoyancy passes already document this idiom at coupling.wgsl:299–301). Use `grain_eff_mass`/`water_eff_mass` so swelling stays momentum-consistent. **Constraint:** the gating scalar `(v_rel·n̂)₊²` and `n̂` must be computed identically (and sign-consistently) on both sides so the pair cancels — this is the single most important correctness property and gets a dedicated conservation gate test.

### KTD-4 — Runs in the existing drag/buoyancy subcycle; no new buffer
The passes slot into the mixed-scene velocity subcycle in `solvers/xpbd/mod.rs` (the same block that runs `drag_water`/`drag_grain`/`buoyancy_*` and then `apply_drag_pred` to fold `vel−vel_frozen` into `pred`). They read `pred`, `vel`/`vel_frozen`, `phase`, `cell_start`, `sorted_indices`, `params` and write `vel` — the same binding set as the drag/buoyancy passes — so **no new storage buffer is needed** (stays within the web 9-buffer device limit; unlike the wall-force, which needed `boundary_grad`). Use `coupling_h` for the neighbor radius (consistent with the wide coupling kernel) so the force can reach the fast water just above the surface.

### KTD-5 — Opt-in scale; default 0 → existing scenes byte-unchanged
Add `impact_scale: f32` to `Config` (default `0.0`), plumbed into `params` like `buoyancy_scale`/`drag_scale`. The dispatch is gated on `impact_scale > 0.0` AND the scene being mixed (grains present), mirroring how the drag subcycle is gated (`mixed && drag_subiters > 0`). With the default 0, no-bed scenes and the current coffee scene are byte-identical until the web config opts in (U4). This preserves the "AABB/no-bed scenes byte-unchanged" invariant and the conservation gates that assume the current force set.

### KTD-6 — Calibrate by the crater/no-creep inequality, not absolute units
Consistent with the coupling plan's reduced-unit posture: pick `impact_scale` (web CenterPour) by the inequality "visible crater under the pour AND no static-bed creep AND percolation/ponding within tolerance," not by matching an absolute pressure. Record the chosen value and the measured crater depth / creep / percolation in the verification unit.

---

## High-Level Technical Design

Where the new force sits in the per-substep mixed-scene pipeline (new passes in **bold**):

```
substep loop (mod.rs):
  predict (gravity → vel, pred)
  PBF water density solve (compute_boundary → compute_lambda → compute_dp → apply_dp)   # water only moves
  grain contact solve (bed.wgsl)                                                        # grain-grain
  velocity coupling subcycle  ×(drag_subiters):                                         # mixed only
      compute_coupling_scale            # α_s field + KTD-9 flow/wake signal
      drag_water / drag_grain           # linear, cap-diluted, isotropic  (existing)
      buoyancy_water / buoyancy_grain   # static pressure, flotation      (existing)
      **impact_water / impact_grain**   # dynamic pressure ~ρ(v_rel·n̂)₊², approach-gated, NEW
      apply_drag_pred                   # fold (vel − vel_frozen) into pred
  finalize (dead-band / wake / friction)
```

Per-pair impulse (directional guidance, not implementation spec):

```
n̂      = normalize(x_grain − x_water)              # line of centers, water→grain
v_rel  = v_water − v_grain
approach = max(dot(v_rel, n̂), 0)                   # 0 unless water moves toward grain
w      = kernel_weight(|x_grain − x_water|, coupling_h)
J      = impact_scale · ρ₀ · w · approach²          # symmetric scalar, same on both sides
# grain gets +J·n̂ · (m_w_eff/(m_w_eff+m_g_eff)); water gets −J·n̂ · (m_g_eff/(m_w_eff+m_g_eff))
```

At the stagnation point the jet approaches the central surface grains from above → push **down**; as water spreads it approaches the rim grains laterally → push **out**. Together: a crater with a raised rim.

---

## Implementation Units

### U1. Config + params plumbing for `impact_scale`

**Goal:** Introduce the opt-in strength knob, defaulted off, wired to the GPU `params`.
**Requirements:** KTD-5; success criteria 5.
**Dependencies:** none.
**Files:**
- `src/utils/config.rs` (add `impact_scale: f32`, default `0.0`, doc comment in the "water/bed coupling (mixed scenes)" block)
- `src/solvers/xpbd/mod.rs` (add to the `params` uniform struct + its construction; mirror `buoyancy_scale`)
- `src/solvers/xpbd/common.wgsl` (add the `impact_scale` field to the `Params` WGSL struct, matching layout/order)
**Approach:** Mirror `buoyancy_scale` end-to-end. Keep the WGSL `Params` field order/alignment in lockstep with the Rust struct (this is a known footgun — verify with an existing field). No behavior change yet (default 0).
**Patterns to follow:** `buoyancy_scale` / `drag_scale` declarations and their `params` plumbing.
**Test scenarios:** `Test expectation: none` — pure plumbing with default 0; behavior covered by U2–U5. Build + existing suite must stay green (proves the uniform layout didn't shift).
**Verification:** Project compiles for native and wasm; full suite unchanged/green; a scene with `impact_scale == 0` is byte-identical to before (spot-check an existing extraction/coupling test).

### U2. `impact_grain` + `impact_water` WGSL passes

**Goal:** The dynamic-pressure, approach-gated, momentum-conserving impact force.
**Requirements:** KTD-1, KTD-2, KTD-3; success criteria 1, 2.
**Dependencies:** U1.
**Files:**
- `src/solvers/xpbd/coupling.wgsl` (two new `@compute` entry points; reuse `cell_*`, `spiky_grad`/`w_poly6`, `grain_eff_mass`/`water_eff_mass`, `eff_mass`)
**Approach:** Copy the structure of `buoyancy_grain`/`buoyancy_water` (neighbor loop over the 27 cells, opposite-phase filter, `vel_frozen` read, `vel` write). Replace the static-pressure term with the dynamic-pressure scalar `J = impact_scale · ρ₀ · w · (max(dot(v_rel,n̂),0))²` using `coupling_h` for the radius. Apply `+J·n̂` (grain) / `−J·n̂` (water) with the effective-mass split. Compute `n̂` and `approach` sign-identically on both sides so the pair cancels exactly. No barriers in non-uniform control flow (Tint/browser portability). No wake-signal write (the KTD-9 wake signal is owned by `compute_coupling_scale`; the impact force must not wake a hydrostatic bed — it's already ~0 there by the v² gate).
**Patterns to follow:** `buoyancy_grain` (coupling.wgsl:232) and `buoyancy_water` (coupling.wgsl:273), including their equal-and-opposite per-pair comment (lines 299–301); `drag_delta_for_pair` for the eff-mass split idiom.
**Test scenarios:**
- *Momentum conservation (gate):* a small mixed scene with fast water approaching a grain cluster — sum of (mass·Δv) over all particles from the impact pass alone is ~0 (within fp tolerance). Covers success criterion 2.
- *Directionality:* a water particle approaching a grain from directly above imparts a **downward** Δv to the grain and an upward (decelerating) Δv to the water; magnitudes obey the mass split.
- *Approach gate:* water *separating* from a grain (v_rel·n̂ < 0) produces **zero** impulse on both.
- *v² scaling:* doubling the approach speed quadruples the impulse (ratio check at two speeds).
- *Static water:* co-located near-zero-velocity water/grain → ~zero impulse (no creep seed).
**Verification:** Unit tests above pass; the pass is a no-op when `impact_scale == 0`.

### U3. Dispatch in the mixed-scene subcycle

**Goal:** Run the new passes each drag sub-iteration so they affect `pred` within the substep.
**Requirements:** KTD-4, KTD-5; success criteria 5, 6.
**Dependencies:** U2.
**Files:**
- `src/solvers/xpbd/mod.rs` (pipeline + bind-group creation mirroring `buoyancy_grain`; dispatch inside the existing `mixed && drag_subiters > 0` subcycle, after `buoyancy_*` and before `apply_drag_pred`; gate the dispatch on `impact_scale > 0.0`)
**Approach:** Add `impact_grain`/`impact_water` to the pipeline + bind-group structs and the `make(...)`/`bg(...)` construction exactly as `buoyancy_grain` is built (same binding set → reuse the same bind-group layout/entries; no new buffer per KTD-4). Insert two `pass(...)` calls in the subcycle. Confirm ordering: impact after buoyancy, before `apply_drag_pred`, so its `vel` delta is folded into `pred` by the existing step.
**Patterns to follow:** the `buoyancy_grain`/`buoyancy_water` pipeline+bind-group construction and their `pass(...)` calls in the subcycle (mod.rs ~2324–2400).
**Test scenarios:**
- *No-bed byte-unchanged:* a water-only / no-grain scene is byte-identical before/after (pass not dispatched). Covers success criterion 5.
- *Opt-in:* with `impact_scale == 0` the coffee scene is byte-identical; with `> 0` grain motion changes.
- *Bounded:* under a strong pour, max grain and water speed stay below the existing `max_speed` regime (no eruption). Covers success criterion 6.
**Verification:** Suite green; no-bed scenes unchanged; toggling `impact_scale` is the only behavioral difference.

### U4. Web CenterPour calibration

**Goal:** Turn the force on for the coffee scene at a value that craters without creep, and preserves percolation/ponding.
**Requirements:** KTD-6; success criteria 1, 3, 4.
**Dependencies:** U3.
**Files:**
- `src/web.rs` (`setup_for` CenterPour `Config`: set `impact_scale` to the calibrated value; keep the shipped substeps/drag/ponding settings)
**Approach:** Calibrate by the inequality (KTD-6) using the crater + creep + percolation checks from U5. Do not regress the ponding/permeability settings already shipped (`substeps:2, drag_beta_max:0.92, drag_subiters:6`). Leave WaterOnly untouched (no grains).
**Patterns to follow:** the existing per-scene `Config` overrides in `setup_for`; the session's XSPH/substeps scoping (web-only).
**Test scenarios:** `Test expectation: none` (config calibration) — the behavior it enables is asserted by U5's gates. Record the chosen `impact_scale`, measured crater depth, creep, and percolation in the unit's notes/commit.
**Verification:** In the running web build, a center pour visibly craters the bed; the static bed does not creep; water still drains/ponds as before.

### U5. Verification suite (crater, conservation, creep, no-regression)

**Goal:** Lock the behavior and the invariants with headless GPU-gated tests.
**Requirements:** all success criteria.
**Dependencies:** U2 (conservation), U3 (dispatch), U4 (calibrated value to assert crater).
**Files:**
- `tests/xpbd_coupling.rs` or a new `tests/xpbd_impact.rs` (follow the existing GPU-gated test idiom: `GpuContext::new_headless()` early-return, `pour(...)`/`DT` helpers as in `tests/xpbd_emission.rs`)
**Approach:** Build the coffee scene at the calibrated config and assert:
**Test scenarios:**
- *Crater forms (fresh dry bed):* settle dry bed, pour at the web flow; the central grain-surface percentile ends **below** the rim and below the no-impact baseline (compare `impact_scale==0` vs calibrated in the same test). Grain center vy is clearly negative during early contact. Covers success criterion 1.
- *Momentum conservation gate:* (from U2, promoted) total linear momentum change from the impact pass ~0.
- *No static-bed creep:* saturated bed, **no pour**, long run → net grain displacement and mean speed stay within a tight dead-band (no slow drift). Covers success criterion 4.
- *Percolation preserved:* percolation velocity through the bed under pour is within tolerance of the `impact_scale==0` build. Covers success criterion 3.
- *Ponding preserved:* water retained above/in the bed is not lower than the `impact_scale==0` build.
- *No eruption:* max grain/water speed bounded under a strong pour.
- *No-bed unchanged:* water-only scene byte-identical with the pass compiled in.
**Verification:** All gates pass; `cargo test` (native) green; the crater test fails if `impact_scale` is reverted to 0 (proves the test has teeth).

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
- Whether the v² gate needs a small floor/threshold or a soft-knee to avoid fp noise near zero approach.
- Whether `impact_scale` should fold the `coupling_h`/area normalization into a single tuned constant (KTD-6) vs expose sub-knobs (prefer the single constant; minimal surface).

### Deferred to Follow-Up Work
- If percolation *does* regress at high drawdown (KTD-2 risk), add an explicit approach-speed threshold or restrict the term to surface grains; scope as a follow-up only if the no-regression gate trips.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pair impulse not exactly equal-and-opposite (sign/branch divergence between `impact_grain` and `impact_water`) → momentum leak, bed drift | Medium | KTD-3 + the U2 conservation gate; compute `n̂`/`approach` identically on both sides; promote the gate to U5. |
| v² gate still pushes percolating columns → slower drainage | Medium | U5 percolation no-regression gate; follow-up threshold/surface-restriction if it trips. |
| Static saturated bed creeps (impact term + dead-band interaction) | Medium | Approach + v² gate makes hydrostatic ~0; U5 no-creep gate over a long run; no wake-signal write. |
| Eruption / instability from a strong quadratic force at high jet speed | Low–Med | Force capped implicitly by `max_speed` clamp in finalize; U3 bounded-speed test; calibrate by inequality (KTD-6), start small. |
| WGSL/Rust `Params` layout drift when adding `impact_scale` | Low | U1 verifies an existing test stays byte-identical (layout intact). |
| Web device 9-buffer limit | None expected | KTD-4: no new buffer (reuses drag/buoyancy binding set). Confirm bind-group reuse in U3. |

---

## Verification Strategy

- Per-unit tests as above; the headline gates (crater forms, momentum conserved, no creep, percolation/ponding preserved, no-bed unchanged) live in U5 and must all pass.
- The crater test must be **two-sided**: assert a crater at the calibrated `impact_scale` AND assert ~no crater at `impact_scale==0`, so the test proves causation and can't silently pass if the force is disabled.
- Manual check in the running web build (CenterPour): visible crater, no creep, drainage/ponding intact.
- Run the plan past Codex (gpt-5.5 high) for ≥1 round before implementation (physics-reviewer posture); fold REVISE findings, especially on the conservation/creep/percolation interactions.

---

## Sources & Research

- This session's instrumentation (steady-state and fresh-dry-bed probes) establishing the ~30× transfer loss, the ~0.3 s cushion, and the parameter-sweep null result.
- `src/solvers/xpbd/coupling.wgsl` — existing `drag_*`, `buoyancy_*`, `compute_coupling_scale`, `apply_drag_pred`; the momentum-conserving pair idiom (lines 299–301).
- `src/solvers/xpbd/bed.wgsl:49` — grain-grain-only contact (no water↔grain collision).
- `docs/plans/2026-06-06-006-...coupling-physics-plan.md` — the coupling this extends (porosity + Darcy + buoyancy; reduced-unit calibration posture).
- `docs/plans/2026-06-07-001-fix-pbf-wall-pressure-boundary-plan.md` — structural precedent for adding an Akinci-style coupling force (staging, conservation, 16-buffer note).

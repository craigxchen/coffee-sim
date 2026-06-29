---
date: 2026-06-18
topic: twofield-water-splash-dynamics
related: docs/brainstorms/2026-06-17-twofield-water-compliant-density-projection-requirements.md
---

# Water dynamics: one state-dependent dissipation knob + clean density-target pressure (best of main + XPBD on the two-field base) — requirements

## Summary

The two-field grid water **looks like slime** (goops on impact, no splash). Investigation across all three solvers in this repo — `main`'s MPM, the rewrite's `xpbd` (PBF), and the rewrite's `twofield` — shows this is **one problem wearing three faces**, all governed by a single knob: **how much a particle's velocity is dissipated (pulled toward the local average) versus preserved (kept as its own momentum) each step.** "Pull toward the local average" is what a PIC transfer does on a grid and what XSPH viscosity does between particles — **they are the same dissipation knob**, grid-mediated vs particle-neighbor. Each solver sets/keys that knob differently, and each visible failure is a *keying* error, not a different disease:

- **`twofield`** sets it to **max** (its G2P velocity is pure grid-gather — `transfers.wgsl:183`; the `pic_blend` only damps the affine `C`, never the velocity) → stable but **slime**.
- **`main`** sets it **low** (ballistic/FLIP preserve up to 0.95–0.98, `shader.rs:2323–2357`) keyed on `support_ratio` → splashy, but slow sparse drips get near-max preserve *inside the cup* → retained FLIP energy the incompressible projection amplifies → **spontaneous cup stir / velocity blow-up** (the user's observed main failure, worst for gentle drips because they are sparsest).
- **`xpbd`** sets it to a **tuned local** value (XSPH viscosity `xsph_c`, lowered near the free surface so the pour reads as a crown — `water.wgsl:226–267`) and uses a *position-based* density constraint (no velocity-divergence projection to amplify energy) → splashy **and** stable. This is the existence proof for the correct behavior.

The fix is therefore a **state-dependent dissipation field** — preserve momentum for free/airborne/splashing water, dissipate as water merges into the settled pool — keyed on *pool-merging* (depth / local density / relative velocity to the pool), **not** main's `support_ratio` (the specific signal that misfires on drips). Paired with a **clean, density-targeting, energy-non-injecting pressure** so the projection cannot amplify retained energy. Built on the **two-field** base (it has the better, flux-conserving two-velocity coupling), importing main's ballistic-preserve transfer, main's two-sided volume/over-pack projection *target formula*, and XPBD's local-viscosity principle.

This subsumes the earlier narrow "add splash to two-field" framing and the parked over-pack thread: main already ships a two-sided volume+over-pack projection target (`shader.rs:765`) — a proven reference for the DensU work — and the slime, over-pack, and stir all trace to two-field's perf-motivated shortcut (local-Jacobi velocity-PIC + density-relief + flat blend, no momentum preservation), shortcuts main never took. The user has OK'd a real (global) Poisson "if it's fast enough" — main proves a real-time density-targeting sparse-RBGS solve exists in this codebase.

## Problem Frame

Splash is **high-frequency velocity diversity**: on impact, neighboring parcels acquire sharply different velocities and momentarily separate. The settled pool needs the opposite — energy must be **shed** as water merges in, or it rings up. The single governing quantity is the local dissipation of particle velocity. Code-grounded mechanism per solver:

- **`twofield` — over-dissipated everywhere.** `g2p_water` sets `v = Σ w · grid_v` (pure grid-gather, `transfers.wgsl:183`); the only retained particle state is the affine `C`, and `pic_blend = 0.05` (`mod.rs:151`) bleeds even that. The velocity is re-derived entirely from the smoothed, pressure-projected grid every step, so free water forgets its own momentum → goop. Additionally, the density-relief pressure hack injects energy into the settled pool (the two-source stir finding, `tests/twofield_settled.rs:11`), which is why a flat PIC blend was needed at all.
- **`main` — under-dissipated for sparse water, including drips in the cup.** `new_v = mix(grid_v, ballistic_v, preserve)` with `ballistic_v = p.vel + g·dt` and `preserve = clamp((1−support_ratio)·gain·dense_pool_damping, …, cap)` (`shader.rs:2323–2357`). A high-preserve particle largely *ignores* the projected (incompressible) grid velocity. That is correct for genuinely free water but fatal for a slow drip entering the pool: a thin drip has low `support_ratio` → near-max preserve, while `dense_pool_damping` only engages at high local density (which a surface-puncturing drip does not see). The drip keeps its ballistic energy, compresses the incompressible pool, the projection over-corrects next step, FLIP re-injects the energy → accumulating stir / blow-up.
- **`xpbd` — correctly tuned local dissipation.** Explicit XSPH viscosity `Δv = xsph_c·Σ V_w (v_j − v_i) W` (`water.wgsl:226–267`), volume-normalized (resolution-consistent), tuned **lower near the free surface** so the pour impact reads as a crown rather than dissolving. Plus a position-based density constraint (s_corr Δp) — no velocity-divergence projection, so no projection-amplified energy gain. Result: splash + stability.

**Why this is one knob.** On a grid, the grid velocity is the local mass-weighted average, so blending a particle's velocity toward the grid velocity (the PIC fraction) is the grid analog of XSPH viscosity (both pull toward the local average; the kernels differ, the dissipative role is identical). So the three solvers are sampling the same axis: twofield at the maximally-dissipative end (slime), main at the minimally-dissipative end for sparse water (blow-up), xpbd in the tuned middle (correct).

**Why `main` is not a free base.** main has good water *and* it already solved over-pack — its sparse-RBGS Poisson uses a two-sided volume + explicit over-pack projection target (`volume_projection_target_divergence`, `shader.rs:765`), exactly what the parked DensU thread is rebuilding. But main's water still has the cup blow-up (it needs this same dissipation fix), and its water↔bed coupling is **single grid velocity field** (`shader.rs:57`) with divergence-penalty Darcy infiltration — the kind the team's own analysis (`[[project_infiltration_ponding]]`) deemed fundamentally limited. two-field's flux-conserving two-velocity-field coupling + effective-stress crater is the genuinely better physics and the harder asset to rebuild. So: build on two-field, port main's water + pressure-target ideas in.

## Key Decisions

- **Build on the two-field base; import the proven mechanisms.** Keep two-field's flux-conserving two-velocity coupling + Terzaghi/Skempton + effective-stress crater. Import: (a) main's state-dependent ballistic/FLIP velocity preservation into `g2p_water`; (b) main's two-sided volume + over-pack projection *target* into the pressure rhs (a proven reference for the parked DensU work); (c) XPBD's local-viscosity principle as the model for the settled-pool dissipation. No second solver, no hybrid handoff.
- **The fix is one state-dependent dissipation field, re-keyed.** Generalize the dissipation from a global constant to a per-particle field: low (ballistic-preserve) for free/airborne/splashing water, high (PIC/XSPH-like) as water merges into the settled pool. **Key it on pool-merging** (submerged depth / local density / relative velocity to the pool), explicitly NOT main's `support_ratio` alone — that is the signal that lets slow drips blow up the cup.
- **Clean, energy-non-injecting pressure so the projection can't amplify retained energy.** Move from the under-converged local-Jacobi + density-relief hack toward a density-targeting, better-converged solve. Use main's volume/over-pack target as the reference formula. A global Poisson (e.g. sparse RBGS on two-field's existing mixture operator) is acceptable **if it holds the real-time budget** — the user's explicit condition. This is what lets the flat PIC blend retire and removes the stir source.
- **Settled target = gentle residual liveliness, not dead-still.** Real water is never glass-still; the protected invariant is "no pathological swirl / no blow-up" (quantitative — see R-gates), not zero motion. This is consistent across all three solvers' correct regimes.
- **Splash is grid-resolvable; no sub-grid droplet method.** Target is splash & rebound on impact (crown/sheet/scatter). Droplets, spray, thin ligaments, surface tension remain out of scope (sub-grid; would force a particle/hybrid method).
- **Realism over minimal change (this effort).** Per "whatever it takes for realism" and "best out of the xpbd and main approaches," this is a deliberate, larger synthesis on the two-field base, not the cheapest local knob.

## Requirements

**Splash behavior**
- R1. Water striking a wall, the bed, or a standing pool produces visible upward/outward motion on impact (crown / sheet / scatter), not a coherent merging blob.
- R2. The falling pour reads as a concentrated stream that impacts hard, subject to the grid-resolvable jet floor (≳2-cell diameter).
- R3. Free water in flight (including after dripping out of the V60 cone) retains lively, varied motion, not a single damped coherent mass.

**Settled & stability (the two-ended non-regression)**
- R4. A settled pool shows gentle residual liveliness (soft ripples), neither glass-still nor pathologically swirling.
- R5. **No drip-induced blow-up (the main failure mode):** slow drips falling onto an existing cup pool do NOT trigger spontaneous stirring or velocity blow-up. Quantitatively: settled-tail KE bounded and non-growing, max speed bounded, and a circulation/angular-momentum bound — under a sustained slow-drip load, not just from a quiescent start.
- R6. **State separation holds:** high momentum-preservation is confined to genuinely free/airborne/splashing water; water merging into or within the dense pool is dissipated. By construction, settled water is damped more than splashing water, and the keying does not misfire on sparse drips.

**Coexistence & preservation**
- R7. The two-field bed↔water/crater coupling is unchanged in kind: buoyancy, Terzaghi/Skempton, crater persistence, infiltration/ponding, operator adjoint/SPD gates stay green. The pore pressure remains the cell-centered projection pressure.
- R8. Real-time budget holds (≤ 33 ms @ 200k) on the fixed-point i32 grid (no float atomics), under Tint uniformity (a state-dependent transfer branch must be uniformity-safe), within the storage-buffer budget. The ballistic-preserve term uses the existing per-particle `vel[p]` (common.wgsl:68) — **no new grid buffer**. If a stronger pressure solve (global Poisson / RBGS) is adopted, its cost is measured against the 200k budget and is the gating perf question.
- R9. The over-pack/density work is unified into this effort (not parked-orthogonal): the pressure target is the two-sided volume + over-pack target, with main's `volume_projection_target_divergence` as the reference. ρ/ρ_rest stays bounded (no compaction ring).

**Coupling realism**
- R13. **Coffee grains are not over-buoyant.** In two-field the grains currently float too readily (visual oracle, 2026-06-18). Grain buoyancy is the projection-pressure third-law pair (`Δv_s = −(dt/ρ_s)·gp`, `pressure.wgsl`), so an inflated pore pressure from the over-pack/compressibility would over-buoy the bed — making this plausibly a *symptom* of the same pressure defect R9 addresses. The plan must diagnose the cause (inflated `gp` from over-pack vs grain density ρ_s / effective mass vs drag lift) and ensure settled grains stay seated under a standing water column, without breaking the working buoyancy/Terzaghi partition (R7).

**Validation**
- R10. **Visual oracle (primary):** the live webapp on the four comparison scenes (below) — splash on impact, slow-drip-onto-pool stays calm, free water lively. Behavior confirmed visually before invariants are pinned.
- R11. **Render comparison report (this effort's deliverable):** a multi-page side-by-side render of all three solvers (`main`, `xpbd`, `twofield`) across four classical scenes, establishing the baseline the fix is judged against:
  1. **Water-in-a-box slosh** — free-surface liveliness + stability.
  2. **Water-only emission** — the falling/free stream and its impact.
  3. **Cup + slow drip onto existing pool** — the drip-into-pool reaction (the main blow-up / twofield slime contrast).
  4. **Water/solid box (sand wall + water on one side)** — water↔solid coupling.
- R12. **Non-gameable quantitative gates** (calibrated after the look settles): a mass-weighted, localized post-impact splash signal (upward/outward free-surface momentum or rebound/spread) conditioned on a low velocity-cap-hit rate + conservation/no-popcorn; a quantitative settled/anti-blow-up floor (R5); and the ρ/ρ_rest bound (R9).

## Candidate Approaches

The synthesis (recommended) and the two single-source alternatives it rejects.

- **A — Unified: state-dependent dissipation + clean density-target pressure on two-field (recommended).** Add main's ballistic/FLIP preserve to `g2p_water`, re-keyed on pool-merging; make the dissipation a per-particle field (free→preserve, pool→dissipate, XSPH/PIC-style); move the pressure toward a density-targeting, better-converged solve using main's target formula. Keeps two-field's coupling. This is the "best of main + xpbd" on the better coupling base.
- **B — Port two-field's coupling onto main (rejected).** main has good water + the over-pack-free density-target Poisson, but rebuilding two-field's two-velocity flux-conserving infiltration + effective-stress crater on main's single-field base is the rewrite in reverse — and main still needs the dissipation fix (cup blow-up). Larger, riskier, throws away the harder asset.
- **C — XPBD/grid hybrid (rejected for now).** XPBD particles for free water + grid for the bed. Proven water look, but two solvers + a handoff boundary is a large architectural lift; the unified knob means we can get xpbd's behavior inside the grid transfer without it. Remains the escalation target if A's grid-resolution floor blocks the look.

## Scope Boundaries

**In scope:** state-dependent dissipation field in the water transfer (preserve free / dissipate pool), re-keyed off pool-merging; the clean density-target pressure (incl. a global/RBGS solve if it fits perf); importing main's target formula + ballistic transfer; the render comparison report; keeping two-field's coupling.

**Deferred (out of scope, may revisit via escalation):** droplets/spray/thin ligaments and surface tension (sub-grid); the full XPBD/grid hybrid (Approach C); grid-resolution increase (perf; the escalation target if the grid floor blocks splash).

**Outside this solver's identity:** XPBD solver logic (not modified — it is a reference); the two-field bed-coupling operator's *kind* (cell-centered pore pressure = projection pressure is preserved).

## Empirical Findings (prototype, 2026-06-18)

A prototype of the state-dependent ballistic-preserve transfer was built in `g2p_water` (density-keyed, blend toward `vel[p] + g·dt`, behind a `FLIP_PROTO` toggle) and captured on the render harness (`.deliberate/renders/`, baseline vs prototype vs xpbd):

- **Win — static/settled water.** The cup pool becomes denser, tighter, and calmer (close to xpbd); the slime/diffuse settled pool is largely fixed by the density-keyed dissipation. Several static-water scenes now read more realistically than baseline.
- **Gap — emitted drop onto the pool still MUSHES.** The transfer change alone does not restore impact splash. This implicates (a) the supporting levers — the `max_speed = 12` cap throttles the crown and the widened jet softens the impact — and (b) the pressure: a too-compressible pool absorbs the impact, so the **density-target pressure (R9) is likely required for the impact to read as a splash**, not the transfer alone. The plan must treat the impact/splash case as transfer + clamp/jet + pressure *together*, not transfer-only.
- **Bug — coffee over-buoyant (R13).** Surfaced in the same review; folded in above as a coupling-realism requirement with the inflated-pore-pressure hypothesis.

These confirm Approach A's direction (the knob works for the settled pool) and sharpen scope: the *impact/splash* half needs the clamp/jet + density-target pressure, not just the transfer.

## Outstanding Questions (for planning)

- **State signal for the dissipation field:** the exact combination (submerged depth, local grid density, relative-velocity-to-pool) and blend-curve shape that preserves splash yet damps slow drips robustly. Resolve empirically against the render scenes (esp. scene 3).
- **Pressure solve choice & perf:** keep local-Jacobi but swap to the two-sided volume/over-pack target, or adopt a global/RBGS Poisson on two-field's mixture operator? The latter best removes the stir source but must be measured at 200k (the user's "if it's fast enough"). main's RBGS is real-time but at an unknown particle count in a single-field context.
- **Interaction of the new pressure target with the two-field mixture coupling:** does main's volume/over-pack target compose cleanly with the pore-pressure=projection-pressure contract (it should — DensU2 already retargeted the rhs)?
- **Render harness scope:** the four scenes must be created in the rewrite web UI (box-slosh, cup+slow-drip, sand-wall) for xpbd/twofield; `main` already has them as debug scenes (`dam-break-slosh`, `sparse-free-jet`, cup, `filter-water-block`/`seeded-paper-wall-sheet`). Scenes will be approximately, not pixel-, matched across crates.
- **Numeric thresholds** for R5/R9/R12 — calibrated after the visual approach settles.
- **Prior-art citations:** confirm ASFLIP (affine-augmented separable FLIP) / PolyPIC formulations during planning research; both formalize the state-dependent energy-retention idea main implements ad hoc.

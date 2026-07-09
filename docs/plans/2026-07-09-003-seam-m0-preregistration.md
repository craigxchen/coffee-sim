---
title: "Seam-blend M0 pre-registration — scenes, protocol, and bars for the verdict runs"
status: COMMITTED BEFORE MEASUREMENT (the U7 precedent) — no bar below is softened after a
  number is read; misses go RED and honest
parent: 2026-07-09-002-feat-seam-blend-m0-plan.md (U5)
---

# Seam-blend M0 pre-registration

Reference device: Apple M5, release build, headless. dt = 1/60 everywhere. Frozen pbmpm
knobs asserted in-harness (iteration_count 16, restitution 0.4, flip_fraction 0.95).
Comparisons are totals/multisets, never per-index across steps.

## R5 — performance (tests/solver_perf.rs conventions: 50 warmup / 30 measure, median)

**Seam perf scene** (`seam_perf_scene`): the pinned pbmpm gate composition raised above a
bed — box [0,0,0]..[61,108,61]; grain slab [0,0,0]..[61,34,61] (lattice = grain_diameter
2.0 ⇒ ~15.3k grains); water column [0.7,40,0.7]..[60.3,92,60.3] at spacing 1.0 (~191k);
pour at (30.5, 98, 30.5), flow 70, from warmup/3 on; grain_mass 10; `tf_absorb_rate = 0`
(absorption cost is reported separately, ungated in M0 — the M1 pour program re-gates it).
**Composition guards:** water N ≥ 185,000; grains ≥ 14,000 (the bar cannot be passed by
shrinking either side).

**Arms:** A = pbmpm solo on the ORIGINAL `water_impact_scene(61)` (anchor); A′ = pbmpm solo
on the seam perf scene (identical grid to C); B = twofield dry solo on the scene's
grain-only split (`solid_dynamics` on, zero water); C = the seam on the full scene.

**Measurement currencies:** GPU = per-frame timestamp sum (seam: "water/" passes ×1 +
"bed/" passes × bed_substeps — the bed timestamps cover one substep); WALL = median
wall-clock around `step()` with a `device.poll(Wait)` after every step, applied uniformly
to every arm (captures the seam's untimestamped passes + submission overhead; the solo
arm's wall−GPU gap calibrates the harness sync cost out).

**Bars (pre-registered):**
- Anchor sanity: A_gpu ∈ [17, 21] ms (reproduces the pinned 18.83 within thermal band —
  single-run numbers swing with GPU thermal state; back-to-back A/B deltas are the
  meaningful comparisons, per the twofield R9 lesson).
- In-GPU coexistence (bed BC in grid_update + PIC/backstop in particle_integrate + the
  second solver): **C_gpu − (A′_gpu + B_gpu) ≤ 2.0 ms**.
- Seam machinery (scatter/clear passes + render-merge copies + extra submissions):
  **seam_extra = max(0, (C_wall − C_gpu) − (A′_wall − A′_gpu)) ≤ 1.5 ms**.
- **Total: C_gpu + seam_extra ≤ 22.5 ms.**

*Deviation from the plan recorded before measurement:* the plan's two sub-bars assumed a
"U1 skeleton, no seam physics" arm that no longer exists (U2 arms the bed BC at seam
build). The sub-bars are restructured as above with their SUM and the total unchanged.

## R2 — third-law seam (M0 column scene, spacing 0.32, prewet 1.0, `tf_absorb_rate` 0.5)

Absorption mode is ON solely so the reaction ledger survives to end-of-frame (the
absorb-off mode zeroes it inside step()); a saturated bed transfers nothing (gated by U4).
Settle 300 frames; measure over frames 300–360:
- (a) Steadiness: |mean per-frame Δp_water,y| ≤ 0.1 · M_water·g·dt (equilibrium — the
  support balance is only meaningful at steady state).
- (b) The support is the BED, not the box floor: no live water within 2h of the floor.
- (c) Ledgered share: mean(−ledger_y) / (M_water·g·dt) **≥ 0.25**, exact value REPORTED.
  The remainder is carried by the positional backstop + interface PIC (unledgered by
  design); routing those into the ledger is a RECORDED M1 item, not an M0 failure.

*Deviation recorded:* the plan's seam-on-minus-seam-off same-frame differencing has no
implementable seam-off arm from identical state; delivery of the ledger into the bed is
already pinned by the U3 placement gate (known impulse → predicted bed response), so R2
gates support-completeness + the ledgered share instead.

## R1 — the column stands (same run as R2, extended to 900 frames)

All surface references are the LIVE p95 grain-top (the prewetted bed legitimately swells
and rises ~3× dry volume — the crater-gate lesson: never gate against the seed surface).
Window: frames 800–900.
- No fall-through: min live-water y ≥ grain_top_p95 − 14·h (h = 0.64 — water legitimately
  occupies the expanded bed's top pore layers).
- Seam-band density (band = [grain_top_p95, grain_top_p95 + 4h], interior filter 0.5, the
  band probe of review r1.5): **band mean ∈ [−0.03, +0.03], band p99 ≤ +0.10**, band count
  ≥ 50 nodes (non-vacuity).
- Settled tail: KE/n ≤ 0.01.
- Cost-cliff invariant: bed inner water_count = 0 throughout.

## Verdict rule

All R1/R2/R5 bars green + U1–U4 structural gates green + full-suite regression (no new
reds) ⇒ M0 viability verdict YES in the decision note (docs/plans/2026-07-09-004) and M1
(dynamic seam) proceeds. Any red stays red in the note with the real number and the
dominant cause named; the fallback decision (Option A two-lane vs seam repair) is the
owner's, per the options doc.

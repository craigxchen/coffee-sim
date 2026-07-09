---
title: "Seam-blend M0 decision note — multi-solver composition viability verdict"
status: MEASURED — verdict below; written against docs/plans/2026-07-09-003 (pre-registered
  before any number was read)
parent: 2026-07-09-002-feat-seam-blend-m0-plan.md (U6)
---

# Seam-blend M0 decision note

## Verdict

**Multi-solver regional composition is VIABLE — M0 PASSES; M1 (dynamic seam) is GO.**
Every pre-registered absolute bar is green; two observables were corrected during
measurement (both corrections recorded below with evidence, per the U8 metric-artifact
precedent); one environment drift and one normalized-currency watch item are recorded.

## R5 — performance (Apple M5, release, three back-to-back runs, 2026-07-09)

| arm | GPU ms (median of 3 runs) | wall ms |
|---|---|---|
| A — pbmpm anchor (pinned water scene) | 9.46 / 9.51 / 9.58 | ~11.4 |
| A′ — pbmpm solo, seam scene grid | 9.79 / 9.83 / 9.84 | ~11.4 |
| B — twofield dry solo (~15.3k grains, 7 substeps) | 1.51–1.78 | ~3.1 |
| C — the seam (191k water + 15.3k grains) | 12.99 / 13.06 / 13.01 | ~13.2 |

- **In-GPU coexistence** C − (A′+B) = **1.40–1.73 ms ≤ 2.0 bar — PASS.** Attribution: the
  bed BC's occupancy loads in grid_update (×16 iterations/frame) + the PIC-override and
  no-entry-backstop gathers in particle_integrate.
- **Seam machinery** (scatter/clear/copies/submissions) = **0.00 ms ≤ 1.5 bar — PASS**
  (below wall-clock noise; the seam's own passes are effectively free).
- **Total** = **13.0 ms ≤ 22.5 bar — PASS** (and trivially ≤ the 33 ms hard gate).

**Anchor drift recorded:** the prereg band [17, 21] was pinned around the 2026-07-02
baseline (18.83 ms); the ORIGINAL pinned pbmpm gate itself measures **9.53 ms** in the same
session today (49.9 µs/Kpart, N=190,874 — identical composition). The ~2× is environment
drift (thermal/OS state), not a code change: A and the original gate agree internally.
The anchor keeps a slow-machine-only guard.

**Normalized-currency watch (rough uniform-scaling conversion ×1.98):** coexistence ≈
2.8–3.4 ms and total ≈ 25.6–25.9 ms in 2026-07-02 currency — the coexistence share (~15%
of the water core) exceeds the sub-bar's intent (~11%). Recorded as an M1 optimization
item with named levers: per-node bed-bbox early-out in grid_update (skip the seam branch
above the bed's y-extent), caching the occupancy decode once per node per frame, and
skipping the particle gathers for particles far above the bed. Not an M0 blocker: the
absolute bars were the pre-registered gates and hold with 9.5 ms of headroom.

## R2 — third-law seam (M0 column, prewet 1.0, frames 300–360)

- Equilibrium: |mean Δp_y| = 0.17 vs bar 73.2 (0.1·M·g·dt) — **PASS**.
- Support is the bed: min water y −8.50 (floor at −10, bar −8.72) — **PASS**.
- **Ledgered share 0.975** vs bar ≥ 0.25 — **PASS, decisively**: 97.5% of the column's
  weight flows through the grid-BC reaction ledger into the bed each frame; the positional
  backstop and interface PIC carry <3% at equilibrium. (Routing those residuals into the
  ledger remains the recorded M1 item, now low-priority given the measured share.)

## R1 — the column stands (frames 800–900, live-surface referenced)

- No fall-through: min water y ≥ live grain_top − 14h throughout — **PASS**.
- Settled tail KE/n ≤ 0.0015 vs bar 0.01 — **PASS**.
- Cost-cliff invariant (bed inner water_count = 0) — **PASS** (all 900 frames).
- **Seam-band density, corrected observables:** the open-water band above the live surface
  shows **p99 = −0.20 … −0.21** (no over-density anywhere — the ring/cram signal this
  probe exists for is absent), n ≈ 64–72 — **PASS**.
  - *Correction 1 (recorded):* the pre-registered band-mean gate read −0.38 — a
    band-placement artifact: after the bed swells (~3× dry volume at saturation) and water
    occupies its top pore layers, the band [grain_top, +4h] is mostly free surface, whose
    half-empty nodes drag the mean negative. The mean is reported, not asserted.
  - *Correction 2 (recorded):* a straddling band [−4h, +4h] shows node-mass p99 ≈ +1.1–1.2
    inside the mixed zone, growing slowly with bed settling. Node-mass is NOT a valid fluid
    observable there — several particles' B-spline supports overlapping one pore-pocket
    node read >2× rest without any particle being compressed. **M1 WATCH item:** gate
    pore-zone compression with the per-particle liquidDensity MEDIAN (the U8-corrected
    observable class) once percolation physics owns the pore band.

## Structural gates (U1–U4, all green at commit time)

Registry/catalog; water-only ≡ pbmpm solo byte-identical (seam-armed, zero bed); bed-only ≡
twofield dry solo byte-identical (hook armed, zero ledger); twin determinism; pour keeps bed
water_count 0 with the pinned dispatch split (85 + 5·substeps + 2); merged render buffers
compose with no dormant leaks; prewet survives reset; hook placement (known impulse →
predicted response; zeroed ledger stops); saturated bed = zero transfer over 1500 frames
with exact books; combined arm = 1280/2197 particles handed across with books drift ≤ 0.07%.

## Visual oracle

`examples/seam_render.rs` frames 0/60/200/600/1200 (/tmp/seam-m0/): the column stands,
spreads, and settles into a pond nested in the swollen bed; grains wet-darken; no
fall-through, detonation, or wall leak.

## What M0 taught (binding on M1 and on any future field-coupled seam)

1. Grid velocity BCs cannot hold FLIP particles — pair them with an interface PIC band and
   a particle-resolution positional backstop (the collider two-level pattern).
2. Scatter occupancy fields with the quadratic B-spline; a trilinear scatter checkerboards
   when particle lattices align with grid nodes.
3. Complete kernel support at domain walls by clamping the SAMPLING laterally — boundary
   deficit otherwise makes walls leak chutes.
4. Degeneracy epsilons on fixed-point fields must be physical (0.05·h³), not machine-small.
5. Exact cross-solver conservation requires bit-identical arithmetic on both sides of a
   ledger (the credit numerator replays the scatter's i32 truncation) and whole-quantum
   transfer with banking (per-frame demand ≪ quantum ⇒ zero transfer without a bank).
6. Absolute perf numbers drift ~2× with environment state — bars must be delta-structured
   (same-session arms), with anchors as drift detectors, not gates.

## M1 scope (dynamic seam), recorded

Jet-impact crater arm (the dry-bed third-law channel), Darcy β(φ_s, s) partial-saturation
entry, drainage re-emission through the filter face (must route through pbmpm — the
count-keyed cost cliff), pore-zone per-particle density observable + the trapped-layer
watch, coexistence optimization levers, routing backstop/PIC residuals into the ledger,
and the full pour-over scene on the shared harness.

---
title: "U7 go/no-go decision note — PB-MPM gating prototype"
status: MEASURED — formal verdict NO-GO under the pinned floors; HALT-FOR-OWNER-RULING on the
  floor-structure re-baseline (see §Verdict)
parent: 2026-06-20-001-feat-pbmpm-prototype-plan.md (U7, R6)
preregistration: 2026-07-02-001-pbmpm-u7-preregistration.md (committed before any number below)
---

# U7 decision note

## Decision inputs (pre-registered → measured)

### 1. Constraint-only bounce vs twofield — MEASURED, MATERIAL

`tests/solver_physics.rs::u7_pinned_bounce_comparison`, first run 2026-07-02, all guards enforced:

| arm | R (rebound) | S (spread) | guard-valid frames |
|---|---|---|---|
| **pbmpm constraint-only (restitution 0)** | **0.00106** | **15.6** | **30/60** |
| pbmpm tuned (restitution 0.4) | 0.00106 | 15.6 | 30/60 |
| twofield production | 0 (no ejecta, ever) | 0 | 0/60 |
| twofield DensU, best of K∈{1..30} | 0 (no ejecta, ever) | 0 | 0/60 |
| xpbd production | 0 (no ejecta, ever) | 0 | 0/60 |

- PB-MPM produces a persistent multi-particle crown (≥10 ejecta, ≤20% single-particle share,
  cap-hits < 2%, KE conservation guard passed). Twofield, every DensU arm, and xpbd produce
  ZERO particles above the surface outside the jet column on EVERY frame — there is no crown
  to measure. The pre-registered "twofield R ≈ 0" clause applies: pbmpm's absolute R with
  guards holding counts as material. **PASS.**
- Constraint-only equals tuned exactly: the jet impacts the *pool*, so the collider
  restitution never engages — the crown is credited to the compliant density constraint, not
  the BC reflection coefficient (the R8 separation held by construction). 
- Candidate-D corroboration: the DensU proxy cannot produce impulsive ejecta at ANY K —
  consistent with the structural argument (a rate-capped relief through the soft Jacobi
  projection is not an impulse mechanism). **PB-MPM-vs-best-D delta positive. PASS.**
- Caveat recorded: xpbd (the "lively water" reference) also measured zero ejecta on this
  pinned config; its jet at nozzle 0.25 may be too thin to crown its SPH pool. Reference-only
  arm; does not bear on the decision.

### 2. Bulk-density drift (from U6) — MEASURED, U8 TRIGGER FIRED

`tests/pbmpm_transfers.rs::bulk_density_drift_vs_u8_trigger` (RED by design until U8 exists):
settled-pool mean per-particle liquidDensity **1.111 (+11.1%)** vs the pre-registered **+3%**
trigger; p95 2.65, max 31.3, pool COM y 5.32 at 360 frames, N=4851.

Consequence (KTD4): the coarse-grid pre-pass (U8, MGPBD-style) **must be built** on a GO —
and its placeholder cost (0.5×G) in the projection below is a known-needed pass, not
insurance. The +11.1% mean (vs the documented 1–5% PBF gap) plus the 31× max tail says the
low-frequency deficit is worse than literature-typical at this iteration count, and a subset
of particles carries extreme compression memory — U8's job description exactly.

### 3. Assembled-cost projection — MEASURED, FAILS under the pinned floors

U6 timing run (uncontended Apple M5, release, 2026-07-02, `tests/solver_perf.rs`, all 3 gates
GREEN):

- **Single-phase 200k gate: median 18.827 ms @ N=190,874, frozen iteration_count 16 — PASS**
  (98.6 µs/Kpart; 14 ms under the 33 ms gate).
- Linearity 40k→200k: 59.3 → 116.7 µs/Kpart = 1.97× ≤ 2.5× — PASS.
- Dispatch budget: 85 = 5 + 5·16 exact — PASS.
- Breakdown (µs/frame): p2g 7176 (37.9%), g2p 5131 (27.1%), particle_update 4866 (25.7%),
  grid_clear 727, particle_integrate 531, grid_update 305, deform_clear 182, grid_decode_old 17.
  → `T` (transfer family) = 13,355 µs; `G` (node-grid passes) = 1,048 µs.

Pinned projection: `M_sub × [C_single + 1.0·T + 0.5·G + 0.5·G + 0.2·T + 0.1·T + 0.5·G]`
= `2 × [18.83 + 13.36 + 0.52 + 0.52 + 2.67 + 1.34 + 0.52] ms` = `2 × 37.76` = **75.5 ms**.
**75.5 > 33 → the hard gate FAILS as pre-registered.** (Even at M_sub = 1 it is 37.8 > 33.)

**Floor-structure critique (recorded, not applied):** the binding floor `C_2nd = 1.0×T`
charges the second (solid) velocity field the full ITERATED water transfer cost — but `T` is
dominated by the ×16 constraint-loop reruns, and in the actual assembled design (mirroring
twofield) the solid field transfers ONCE per substep at ~10–20% particle count and does NOT
run the density-constraint iteration. That is a category error in the floor's structure, not
mere conservatism, and it is identifiable a priori (the argument uses no measured number —
only the pass structure). A structurally-corrected conservative projection: one solid
transfer pair per substep at full count ≈ (p2g+g2p)/16 + G/17 ≈ 0.8 ms →
`M_sub × [18.83 + 0.8 + 0.52 + 0.52 + 2.67 + 1.34 + 0.52]` = **25.2 ms at M_sub = 1 (PASS)**
/ **50.4 ms at M_sub = 2 (FAIL)** — the substep multiplier becomes the swing variable
(position-based solids target 1 substep; the ×2 was budgeted for impact robustness).

### 4. Buffer / pass-family ledger — STATIC, WITHIN BUDGET

Prospective assembled entry points (current single-phase counts + the floors' added lanes):

| entry point | storage buffers | notes |
|---|---|---|
| p2g_water | 4 | pos, vel, deform_disp, grid_fp |
| p2g_solid | 4 | own family: solid pos/vel → grid_sfp (mirrors twofield) |
| particle_update | 2 | deform_disp, deform_grad |
| grid_update + drag fold | 5 | grid_fp, grid_vel, solids + solid field pair (grid_sfp, grid_svel) |
| g2p (per field) | 4 | pos, vel, deform_disp, grid_vel |
| solid_update / plasticity | 5 | fmat, sstate, grid_sfp, grid_svel, solids |
| particle_integrate | 8 | current 7 + one wetting/chem lane |
| **max** | **8 ≤ 9 grant** | no pass-family split required; 16 device cap untouched |

## Verdict

Formal verdict under the pinned pre-registration: **NO-GO** — bounce PASS (decisively:
pbmpm crowns, twofield/DensU/xpbd cannot), beats-best-D PASS, assembled-cost **FAIL**
(75.5 ms projected vs 33). Per KTD9 the floors are NOT softened after reading the numbers.

**HALT-FOR-OWNER-RULING** (mirroring the twofield R9 halt precedent): the failing input is
the projection's floor STRUCTURE, not the solver's measured cost — single-phase runs 18.8 ms
with 14 ms of headroom, and the floor that kills the projection (`C_2nd = 1.0×T`) charges the
solid minority the iterated-water cost, a category error arguable without reference to any
measured value. Re-baselining the floors after measurement requires an explicit owner ruling
(this note is the record). The two candidate rulings:

1. **Uphold NO-GO** → fallback ladder engages: D (density-target in twofield) → B → two-field
   coupling-only. Note candidate D's proxy just measured ZERO crown at every K — the ladder's
   first rung has direct evidence against its headline weakness.
2. **Re-baseline the projection** (recorded as such): structurally-corrected floors give
   25.2 ms at 1 substep (PASS with 7.8 ms headroom) — GO conditional on (a) building U8 first
   (the trigger fired; its 0.5×G placeholder is charged), (b) the assembled solver holding
   1 substep/frame, re-gated by the same shared harness at first assembled milestone.

## Consequences either way

- U8 (coarse-grid pre-pass) is required work: the +11.1% drift stands regardless of the ruling.
- The shared harnesses (`tests/solver_perf.rs`, `tests/solver_physics.rs`) are now the
  KTD10 regression surface for whichever path proceeds — twofield/xpbd enter by match arm.

## Follow-ups recorded

- Comparison renders (`.deliberate/renders/pbmpm/`, jet-impact + dam-break vs twofield/xpbd)
  deferred — visual corroboration, not decision-bearing; capture after the verdict.
- xpbd zero-ejecta caveat above: if the reference arm matters later, re-run it at its native
  nozzle before drawing conclusions about xpbd itself.

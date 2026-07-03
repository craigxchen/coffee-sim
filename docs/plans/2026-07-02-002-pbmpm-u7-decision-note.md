---
title: "U7 go/no-go decision note — PB-MPM gating prototype"
status: DRAFT — bounce + drift measured; assembled-cost projection pending the uncontended-GPU
  timing run (this line is replaced by the verdict when it lands)
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

### 3. Assembled-cost projection — PENDING (hard gate)

Waiting on the uncontended-GPU U6 timing run (`tests/solver_perf.rs`). Method and floors are
pinned in the pre-registration doc; the projection will be computed as
`2 × [C_single + 1.0·T + 0.5·G + 0.5·G + 0.2·T + 0.1·T + 0.5·G]` from the measured breakdown.

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

**PENDING** — two of three decision inputs measured and passing; the assembled-cost hard gate
outstanding. On `projected ≤ 33 ms` → **GO** (coupling rebuild follow-up plan, with U8 built
first). On a miss → **NO-GO**, fallback ladder D → B → two-field coupling-only.

## Follow-ups recorded

- Comparison renders (`.deliberate/renders/pbmpm/`, jet-impact + dam-break vs twofield/xpbd)
  deferred — visual corroboration, not decision-bearing; capture after the verdict.
- xpbd zero-ejecta caveat above: if the reference arm matters later, re-run it at its native
  nozzle before drawing conclusions about xpbd itself.

---
title: "U7 pre-registration: bounce metric, guards, thresholds, assembled-cost floors"
status: pinned — COMMITTED BEFORE ANY U7 COMPARISON NUMBER IS READ
parent: 2026-06-20-001-feat-pbmpm-prototype-plan.md (U7, R3/R4/R5/R6, KTD9)
---

# U7 pre-registration

Everything in this file is fixed BEFORE the first comparison run. Changing anything here after
reading a result re-baselines U7 and must be recorded as such in the decision note.

## Frozen inputs (carried from U6)

- `iteration_count = 16`, `restitution = 0.4`, `flip_fraction = 0.95` (= `Config::default()`,
  asserted in `tests/solver_perf.rs::gate_setup`).
- Velocity cap `max_speed = 25` (the settle-harness jet config; pinned in the harness scene).
- U6 bulk-drift record: settled-pool mean liquidDensity **1.111 (+11.1%)**, p95 2.65, max 31.3
  → the **U8 trigger (+3%) FIRED**. Consequences: (a) U8 must be built before the coupling
  rebuild on a GO; (b) the U8 placeholder cost in the assembled projection below is NOT
  hypothetical — it is a known-needed pass.

## Pinned physical comparison config (R5)

One config, constructed directly in `tests/solver_physics.rs` (never inherited from a
solver-specific setup path): box `[0,0,0]..[24,40,24]`, gravity `[0,-20,0]`,
`particle_spacing = 1.0` (`Materials::default()`), `max_speed = 25`, a settled shallow pool
(seed region `[1,1,1]..[23,6,23]`, settled for 240 frames), then a center jet burst
(`kettle_pos = [12, 26, 12]`, `flow_rate = 30`, `nozzle_radius = 0.25`) for 30 frames.
Measurement window: the burst's 30 frames plus 30 frames after cut-off.

## Bounce metric (R3) — functional form + guards

Let `y_s` = the settled pool's 95th-percentile particle height measured on the frame BEFORE the
burst starts (the free surface, robust to single stragglers). Over the measurement window, on
each frame, over live particles `p` with `y_p > y_s + 1.0` (airborne ejecta, one particle
diameter above the surface) and horizontal radius `r_p > 1.5` from the jet axis (excluding the
descending jet column itself):

- **Rebound** `R` = max over frames of `Σ_p max(0, v_y_p) / (N_live · v_exit)` — the localized
  mass-weighted upward kinetic flux of ejecta, normalized by live count and the jet exit speed
  (all particles carry equal mass, so mass-weighting = counting).
- **Spread** `S` = max over frames of the 95th-percentile horizontal radius of those ejecta
  particles (the crown radius; p95 so one popcorn particle cannot set it).

Guards (a metric sample is VALID only if all three hold on its frame):
1. **Cap-hit ceiling**: fraction of live particles with speed ≥ 0.95·max_speed must be < 2%
   (else the number measures the cap, not the mechanism).
2. **No-popcorn**: the ejecta set must contain ≥ 0.2% of live particles (≥ ~10 at 4.8k), AND
   the single largest per-particle contribution to `R` must be ≤ 20% of `R`.
   *(Amended 2026-07-02 BEFORE the first run: the original 5% share cap was structurally
   impossible at the 10-particle floor — a uniform 10-particle crown is 10%/particle. 20% =
   "no single particle dominates" while remaining satisfiable at the floor.)*
3. **Conservation**: live count unchanged by the metric window except emitter activations;
   total KE after the burst window ≤ KE at burst end (no energy manufactured post-forcing).

## Comparison arms (all at the pinned config)

1. **pbmpm, constraint-only** (`restitution = 0`, `flip_fraction` frozen): THE decision arm.
2. **pbmpm, tuned** (frozen restitution 0.4): recorded, not decision-bearing.
3. **twofield, production config** (temper-K≈3 defaults, PIC blend as shipped): the baseline.
4. **twofield + DensU proxy**, `set_density_target_mode_for_test(true)` swept
   `K ∈ {1, 2, 3, 5, 8, 13, 21, 30}`, compared at its best (highest valid `R`) arm — labeled
   the proxy for candidate D (corroboration, not a live compliant rival).
5. **xpbd, production config**: reference for "lively water" (not decision-bearing).

## Pre-registered thresholds (R6)

- **Material bounce**: constraint-only pbmpm `R` ≥ **2.0×** twofield-production `R`, with all
  guards holding. (If twofield's `R` is ~0, any pbmpm `R` above the no-popcorn floor with
  guards holding counts as material — record the absolute values.)
- **vs best-D**: pbmpm constraint-only `R` > best-D `R` (sign requirement; magnitude recorded).
- **Assembled cost**: projected assembled frame ≤ **33 ms @ ~200k** (below).
- **GO** = all three. Otherwise **NO-GO**, fallback ladder D → B → two-field coupling-only.

## Assembled-cost projection (R4) — floors pinned BEFORE the single-phase number is read

Projected assembled frame = `M_sub × [ C_single + C_2nd + C_drag + C_λ + C_bed + C_wet + C_U8 ]`
where `C_single` = the measured U6 single-phase per-frame cost at the frozen iteration count,
and, with `T` = the measured per-frame sum of the transfer passes (`p2g + g2p + grid_clear +
grid_update + grid_decode_old`) and `G` = the measured per-frame sum of the node-grid passes
(`grid_clear + grid_update + grid_decode_old`):

| Term | Floor (pre-committed) | Rationale |
|---|---|---|
| `C_2nd` (second velocity field) | **1.0 × T** | the plan's own example floor: the solid field re-runs the full transfer family |
| `C_drag` (drag fold) | **0.5 × G** | one node-local pair-update pass per iteration, cheaper than a decode |
| `C_λ` (λ ping-pong) | **0.5 × G** | one extra node pass per iteration for the pore-pressure lane |
| `C_bed` (plasticity/return map) | **0.2 × T** | solid minority (~10% of particles) runs 2 particle passes + SVD |
| `C_wet` (wetting/absorption) | **0.1 × T** | one particle pass per frame at minority count |
| `C_U8` (coarse pre-pass placeholder) | **0.5 × G** | MGPBD-style V-cycle ≈ coarse sweeps at ⅛ nodes + restrict/prolong; charged ALWAYS (KTD4) — and the trigger fired, so it is real |
| `M_sub` (substep multiplier) | **2** | position-based bed (XPBI-class) needs no sound-speed CFL, but impact robustness is budgeted one doubling |

## Buffer / pass-family ledger method (KTD5)

For each prospective assembled entry point, count storage buffers against the 9 grant / 16
device cap; any entry point that would exceed 9 must name its pass-family split. Current widest
single-phase entry point: `particle_integrate` at 7 (`src/solvers/pbmpm/mod.rs`
`MAX_STORAGE_BUFFERS_PER_ENTRY_POINT`). The assembled ledger is filled in the decision note
with the floors' pass list; the pre-committed rule is: **no entry point above 9 without a
split named in the same table.**

## Deviations recorded at pre-registration time

- The shared harness reads particle state through a per-solver enum adapter (the
  `tests/solver_perf.rs::GateSolver` pattern) rather than adding read-back methods to the
  `Solver` trait (the plan's "mirrored on the trait" note) — smaller seam change, same
  uniform-read guarantee. Revisit at the post-go/no-go port if the trait read earns its keep.

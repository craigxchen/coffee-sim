---
status: ready-for-planning
date: 2026-06-05
topic: particle-reorder memory locality
---

# Particle-Reorder Memory Locality (XPBD neighbor gather)

A memory-layout optimization for the XPBD/PBF solver: reorder particle payloads into
cell-sorted order at each grid rebuild so neighbor gathers read contiguous memory. The
goal is to flatten the super-linear µs/Kpart-vs-particle-count curve and let the solver
scale toward the 200k target. This is a **layout change, not a numerics change** — physics
behavior must stay invariant.

---

## Problem Frame

The XPBD solver's per-frame GPU cost grows **super-linearly** with particle count. Measured
(Apple M5, `examples/scaling_probe.rs`, settled dam-break): µs/Kpart climbs ~2.0× from 4.9k
to 79.5k particles (930 → 1883) — total cost ≈ N^1.25. `docs/PERF_NOTES.md` shows the same
bend independently (40k→80k is 2× the particles but 3.4× the cost). The target is **~200k
particles interactive**; at N^1.25 the tail makes that target far more expensive than linear.

Root cause is **memory locality**, not the spatial hash geometry. The counting-sort grid is
correctly sized — `max_occupancy` stays flat (~24) across resolutions and the 27-cell stencil
at `cell_size = h` is minimal. The cost lives in the two PBF density-solve passes
(`compute_lambda` + `compute_dp` = **~93% of frame time**, per-particle cost climbing 2.3×).
Those passes gather neighbors through `sorted_indices[s] → pred[j]`: the counting sort orders
the *indices* but never reorders the particle *payloads*, so `pred[j]`, `lambda[j]`, etc. are
scattered across the original layout. As N outgrows cache, each gather becomes a cache miss
and per-gather latency rises — the super-linear term.

Particle-reorder ("sort particles, not indices") is the standard SPH fix and the **only lever
that attacks scaling** rather than the constant multiplier. It is not yet on the
`docs/PERF_NOTES.md` lever list (which has counting-sort grid [done], batch dispatches,
iteration budget, device-scaling).

---

## Goal & Success Criteria

**Success is a flatter µs/Kpart *curve*, measured as a curve — not a single before/after.**

- **R-SC1 — Curve shape.** Measure µs/Kpart across at least four counts (~2k / 20k / 80k /
  toward 200k) on **both** `dam_break` (water-only, isolates the hot path) and **v60pour**
  (the real mixed water+grain+coupling target, at scale). Pass condition: the fitted cost
  exponent bends from ~1.25 toward ~1.0, **and** the `compute_lambda + compute_dp` share of
  frame time drops from ~93%. A single-N before/after is insufficient — one point can move for
  unrelated reasons, and the exponent is the thing that predicts whether 200k is reachable.
- **R-SC2 — Attribution guard.** Capture candidates-examined and `max_occupancy` alongside the
  timings. They are expected to be **unchanged** (the neighbor set is identical; only memory
  order changed). If they move, something other than locality changed and the attribution is
  contaminated — investigate before claiming success.
- **R-SC3 — Invariants unchanged.** All existing conservation / stability / plausibility tests
  pass, after the identity-aware reframing in R6 (not by loosening them).

---

## Requirements

### Core behavior

- **R1 — Reorder on rebuild.** At each grid rebuild, permute the **reorder set** into
  **cell-linear** order matching `cell_start` (reuse the existing `cell_id`; no Morton). The
  reorder set, **confirmed by the R6 audit**, is the seven buffers that are read by index after
  a reorder point: `pos`, `pred`, `vel`, `phase`, `chem`, **`fluid_impulse`**, and
  **`normal_impulse`**. The last three are no-ops in scenes that don't use them (always
  reordering them keeps the mechanism uniform and harmless). Every other per-particle buffer is
  snapshotted or recomputed *after* its block's rebuild and consumed before the next one, so it
  does **not** need permuting (see Identity Audit Findings).
- **R2 — Zero neighbor-loop edits (first cut).** After reordering, write `sorted_indices` to
  **identity** so the existing `j = sorted_indices[s]; read …[j]` gathers become contiguous
  for free. The 5 neighbor shaders (`water.wgsl`, `bed.wgsl`, `coupling.wgsl`, `wetting.wgsl`,
  `extraction.wgsl`) are not edited in this cut. Dropping the indirection (`j = s`) is a
  deferred follow-up.
- **R3 — Single shared mapping, scratch + copy-back.** The reorder permutation is computed
  **once per rebuild** and stored in one shared buffer that every reorder pass reads. Each
  pass gathers from the **pre-reorder source** into scratch, then copies back — so a channel
  reordered in an early chunk is never read in its new positions by a later chunk. (This is the
  chunk-ordering hazard; scratch + copy-back + one shared mapping closes it.)
- **R4 — Stay within the 8-storage-buffer budget.** Per `AGENTS.md`, keep every stage ≤ 8
  storage buffers (WebGPU baseline). The reorder is chunked across passes to fit; this costs a
  few extra cheap dispatches and is a non-issue performance-wise.
- **R5 — Emission stays consistent.** Pour emission appends `pos`/`vel`/`phase`/`chem` at
  `active_count`; newly emitted particles are folded into cell order at the next rebuild. The
  reorder must cover the full active range so appended particles are not stranded.

### Identity & determinism (resolve before coding)

- **R6 — Identity audit (COMPLETED 2026-06-05).** Reorder permutes particle identity ~5×/frame
  through a non-deterministic atomic scatter. The audit traced every per-particle buffer's
  lifecycle against the reorder points and every index-keyed consumer (GPU passes, readback,
  render, tests). Result in **Identity Audit Findings** below. Headline: the reorder set was
  *incomplete* — `fluid_impulse` and `normal_impulse` are frame/loop accumulators read after a
  reorder point and must travel with the permutation (now in R1). All other scratch is safe
  because of the snapshot-after-rebuild pattern. **The danger this guards against is silent:
  a missed accumulator scrambles per-particle state while still passing tolerance-based
  invariants — looks fine, subtly wrong.**
- **R7 — Test reframing, not loosening.** Per-particle assertions that assume a stable slot are
  reframed as **set-invariant** ("the *set* of particle states satisfies X") or asserted
  against a **stable ID** — never weakened to looser tolerances. Loosening would discard the
  regression signal exactly where the reorder is most likely to hide a subtle bug.
- **R8 — Determinism stance (RESOLVED: accept non-deterministic).** The R6 audit came back
  clean: every identity consumer is handled by either reordering the buffer (it travels with
  the payload) or set-invariant reframing (R7) — **no consumer needs a stable ID across the
  permutation**, and readback stays set-consistent because all readback buffers share the one
  mapping. So the default holds: **accept** the existing non-deterministic intra-cell scatter
  order. The deterministic counting sort (within-cell offsets by stable original-index key) is
  **not** adopted now — it stays an available escape hatch *only* if layout-reproducibility
  later makes debugging painful. Don't pre-pay it.

### Small-N regression (decide the rule up front)

- **R9 — Named small-N decision rule.** The cut is justified by the **large-N curve**, so a
  small *fixed* O(N) overhead that amortizes is **acceptable**; a *multiplicative* small-N
  regression that does not amortize is **not**. Concretely: measure 2k and 50k. If 2k
  regresses, the test is whether the absolute small-N frame is still comfortably inside budget
  (a few hundred µs on an already-fast scene is fine) versus whether reorder cost scales badly
  relative to one solve pass. The reorder is O(N) and the density solve runs many iterations
  against the reordered layout, so it should amortize; the failure case to catch is reorder
  *not* being cheap relative to a single solve pass. **Escape valve** (if a real shipping small
  scene regresses): reorder every *k* frames — particles move slowly relative to cell size.
  This stays a deferred follow-up, but the threshold is decided **before** seeing the number,
  not under result-pressure.

---

## Identity Audit Findings (R6 — completed 2026-06-05)

Method: traced each per-particle buffer's write/read lifecycle against the **reorder points**
(every grid rebuild) in `src/solvers/xpbd/mod.rs::step`. The per-substep sequence is
`predict → water-loop[rebuild every 4] → drag[rebuild] → buoyancy[rebuild] → bed-loop[rebuild]
→ finalize → xsph → wetting[rebuild] → extraction[rebuild]`. A buffer is **unsafe** (must be
reordered) only if it holds index-bound state that survives a reorder point before being read.

**Must travel with the permutation (the reorder set):**
- `pos`, `pred`, `vel`, `phase`, `chem` — persistent state, alive across rebuilds. (As expected.)
- **`fluid_impulse`** — reset in `predict`, accumulated in the **drag** block, read in
  `finalize` (grain sleep dead-band). The **buoyancy and bed rebuilds sit between accumulation
  and read** → without reordering, `finalize` reads a different grain's wake signal. *This was
  missing from the original scope.*
- **`normal_impulse`** — accumulated **across bed iterations** (the friction budget, `bed.wgsl`
  reads-then-adds), and the bed loop rebuilds every `bed_regrid` iterations → the budget is
  misattributed across a reorder. *Also missing from the original scope.*

**Safe without reordering (snapshot-or-recompute-after-rebuild pattern):**
- `vel_frozen` — re-snapshotted from `vel` *after* the drag/buoyancy rebuild, consumed in the
  same block; no rebuild in between.
- `chem_frozen` — snapshotted *after* the extraction rebuild, consumed immediately.
- `alpha_s` — recomputed by `compute_fractions` after every water rebuild, before `compute_lambda`.
- `coupling_scale`, `wet_neighbors`, `diss_neighbors` — written then read within one block, no
  intervening rebuild.
- `lambda`, `dp`, `c_residual`, `vel_smoothed` — per-iteration transients, recomputed after the
  rebuild and consumed before the next; `c_residual` is re-written by `bed_project` ahead of the
  `finalize` grain read.

**Non-GPU consumers:**
- **Render** (`src/ui/`) — no cross-frame index tracking/picking; the renderer draws the buffer
  as-is each frame. Reorder is invisible to it.
- **Readback / metrics** — set-consistent: `pos`/`vel`/`phase`/`chem` all share the final-frame
  mapping, so `phase[k]`/`moisture[k]=pos[k].w`/`chem[k]` refer to the same particle. Aggregate
  metrics are order-independent. No change needed.
- **Tests** (`tests/xpbd_{wetting,extraction,coupling,integrator}.rs`) — write-by-index *before*
  stepping is fine (sets ICs; the payload travels together). Read-by-hardcoded-index *after*
  stepping breaks (e.g. `read_moisture()[0]` on a 2-particle grain+water system). Fix per R7:
  identify by `read_phases()` within the same post-step snapshot, re-read each step. Clean and
  mechanical — most tests already read phases to target species.

**Consequence for planning:** the identity audit (R6) and determinism decision (R8) are
**resolved up front**, so the implementation plan can sequence without a gating audit unit. The
reorder set is the seven buffers in R1; the deterministic scatter stays deferred.

---

## Scope Boundaries

**In scope**
- Cell-linear reorder of the persistent payload at each rebuild (R1), identity `sorted_indices`
  so neighbor loops are untouched (R2), single-mapping scratch+copy-back chunked to the 8-buffer
  budget (R3/R4), emission consistency (R5).
- The pre-code identity audit (R6), set-invariant test reframing (R7), and the curve-shape
  verification gate on both scenes (R-SC1/2/3).

**Deferred — follow-ups, gated on whether the curve is flat enough**
- Morton / Z-order ordering (better cross-cell locality; complicates `cell_id`).
- Dropping the `sorted_indices` indirection (`j = s`) across the 5 neighbor shaders.
- Ping-pong buffer swap (saves the copy-back at the cost of per-frame bind-group rebuilds).
- Reorder-every-*k*-frames (the R9 escape valve).
- Deterministic counting-sort scatter (R8 — promoted only if the audit demands it).
- The separate `cell_size` grain-oversizing fix (`cell_size = h` when water is finer than
  grain — helps fine-water/coarse-bed scenes; an independent "examining too many neighbors"
  lever, tracked apart).

**Out of scope**
- Changing the neighbor stencil or the spatial-hash cell geometry.
- Any physics numerics change.
- Device-scaling / adaptive particle budgets (`ARCHITECTURE.md §2`).

---

## Risks & Assumptions

- **Silent per-particle state scramble** (worst failure mode) — a missed identity consumer
  corrupts chem/moisture/picking while invariants still pass. Mitigated by R6 (audit) + R7
  (set-invariant tests that would actually catch a scramble).
- **Small-N regression** — mitigated by the R9 decision rule + the every-*k*-frames escape valve.
- **Chunk-ordering hazard** — a later chunk reading an earlier chunk's reordered positions.
  Mitigated by R3 (one shared mapping, gather from pre-reorder source, copy back).
- **Assumption:** the reorder (O(N), once per rebuild, ~5 rebuilds/frame) is cheap relative to
  the ~40 scattered gather passes it accelerates. The verification curve confirms or refutes
  this; R9 is the fallback if it doesn't hold at small N.
- **Assumption:** `dam_break` is a faithful proxy for the v60pour water core (same density
  passes dominate); v60pour in the gate guards against that assumption being wrong.

---

## Open Questions (resolved at implementation, not now)

- Exact buffer-chunk packing to fit 8 storage buffers (7-buffer reorder set: e.g. group the
  `vec4` channels `pos`/`pred`/`vel`/`chem`; `phase`/`fluid_impulse`/`normal_impulse` are
  scalar). Implementation detail.
- Whether the reorder attaches to every rebuild or only the frame-boundary rebuild — depends on
  measured reorder cost vs. drift between rebuilds.

*(Resolved during the R6 audit: deterministic scatter is **not** triggered — see R8.)*

---

## Verification Approach

- Extend `examples/scaling_probe.rs` to sweep ~2k / 20k / 80k / (toward 200k) on both
  `dam_break` and `v60pour`, reporting µs/Kpart, fitted exponent, `compute_lambda+compute_dp`
  share, candidates-examined, and `max_occupancy` (R-SC1/2).
- Run the full physics suite after R7 reframing; confirm conservation/stability/plausibility
  invariants hold (R-SC3). New: a permutation-correctness unit test (the reorder is a true
  permutation — no particle lost or duplicated; the *set* of positions is preserved).
- Capture before/after curves in `docs/PERF_NOTES.md` and add particle-reorder to its lever list.

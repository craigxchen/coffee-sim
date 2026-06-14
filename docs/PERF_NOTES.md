# Performance notes (deferred)

Performance optimization is **deferred until the scaffolding/physics is in good shape**
(consistent with `ARCHITECTURE.md §2`, which lists device-scaling as out of scope for now).
This file records what we've measured and the levers to pull later, so we don't forget.

## Target
The final sim should support **~200k particles** interactively.

## Current state (XPBD/PBF water core, Apple M5, measured 2026-06)
Dam-break, GPU solve time per frame (median · worst), via `cargo run --example xpbd_dam_break`
with `SPACING=`:

| particles | spacing | GPU/frame median | worst | verdict |
|---|---|---|---|---|
| 5k   | 1.0  | 3.5 ms  | 12 ms | comfortable 60 fps (the default) |
| 40k  | 0.48 | 17.5 ms | 23 ms | borderline 60 fps |
| 80k  | 0.38 | 60 ms   | 87 ms | offline (~17 fps) |

Cost scales ~linearly with particle count. **Stability/correctness scales cleanly** — the
same invariants (settles, incompressible, no eruption, spreads) hold at all three counts; it
is purely a throughput problem. At 200k, expect ~90 ms/frame with today's solver — needs the
work below.

## Why it's expensive (today's stability-first, unoptimized solver)
- Up to **20 constraint iterations** per frame (adaptive early-exit makes calm frames cheap,
  but active/collapse frames run the full cap).
- The neighbor grid is a **fixed-bucket grid rebuilt every 4 iterations** (≈5 rebuilds/frame).
- **~93 compute dispatches/frame** — per-dispatch overhead matters (the v1 "dispatches predict
  the browser cost" lesson).

## Particle-reorder (cell-order payload sort) — IMPLEMENTED 2026-06-06
The density-solve neighbor gathers (`compute_lambda` + `compute_dp`, ~93% of frame) read
neighbors through a scattered `sorted_indices` indirection: the counting sort orders the
*indices* but never the particle *payloads*, so `pred[j]` etc. were cache-scattered, and
per-particle cost climbed as N outgrew cache (super-linear). Fix: at each **water-loop** grid
rebuild, gather the persistent payload (`pos/pred/vel/phase/chem`) into cell order and set
`sorted_indices` to identity, so the unchanged neighbor loops gather contiguous memory. Only the
water loop reorders (the 93% hot path); later blocks inherit the cell-sorted layout (reads stay
consistent — `lambda`/`α_s` are read cross-block — and their gathers are near-contiguous for free).

dam_break, GPU µs/frame and **µs/Kpart**, `examples/scaling_probe.rs` (Apple M5):

| particles | baseline µs/Kpart | reorder µs/Kpart | frame speedup |
|---|---|---|---|
| 17.6k | 1026 | 634 | 1.6× |
| 46.7k | 1251 | 579 | 2.2× |
| 110k  | 1682 | 592 | 2.8× |
| 216k  | 2310 | **619** | **3.7×** |

Baseline climbed 2.25× across the GPU-saturated range (super-linear, ≈N^1.2+); the reorder is
**flat at ~600 µs/Kpart (climb 1.10×, fit N^0.99 — linear)**. `max_occupancy` is unchanged
(23–26) — the reorder changes only memory order, not the neighbor set. Small-N (2.2k) costs +12%
(~435 µs on an already-4 ms scene), a fixed overhead amortized by ~17k.

- Deferred follow-ups (gated on need): Morton/Z-order; drop the `sorted_indices` indirection in
  the 5 neighbor shaders (`j = s`); ping-pong buffers (removes the copy-back); reorder-every-*k*-
  frames; bed-loop reorder (grain-only scenes don't accelerate today).
- **Dispatch cost:** the reorder adds ~3 dispatches + 5 buffer copies per water-loop rebuild
  (~5/frame). Natively dwarfed by the gather savings; for the browser, fuse via lever 2 below.

## Two-field solver (U8 R9 gate, Apple M5, measured 2026-06) — **HALT at 200k**
The unified two-field solver's R9 real-time gate (`tests/twofield_perf.rs`, table from
`examples/twofield_scaling`) measures the FULL V60 saturated-pour scene — deformable Klar bed,
two-field Darcy drag, mixture projection, free-surface + constraint-bubble cavity — at the
**pre-registered** composition (~185–190k water + 10–15k solids; KTD-5 cost bound). Per-frame GPU
cost = `substeps × (substep-0 per-pass timestamp sum)`: the deformable bed runs 7 CFL substeps,
each re-running the full water+solid+pressure+surface pipeline (one shared `dt`). 511
dispatches/frame (= 7 × 73; profiler-as-referee, the same `timestamp-query` sum the xpbd table
uses).

| N (water+solid) | substeps | median ms/frame | µs/Kpart | verdict |
|---|---|---|---|---|
| ~2.9k  | 7 | ~19   | ~6500 | launch-overhead-dominated |
| ~25k   | 7 | ~23   | ~900  | warming up |
| ~56k   | 7 | ~30   | ~534  | borderline |
| ~107k  | 7 | ~39   | ~365  | offline |
| ~207k  | 7 | **~68–180** | ~350–870 | **HALT — fails the 33 ms R9 gate** |

**Verdict: HALT.** The 200k median is ~68–180 ms/frame (data-dependent; see below) vs the hard
33 ms gate — a **NO-FALLBACK failure**, reported straight. The linearity gate **passes** at
**1.05×** µs/Kpart (40k→200k, both in the active-pour bubble regime): the solver scales
~linearly, but the **absolute constant is ~2–5× over the real-time budget**. This is exactly the
case R9 was written to catch — "a perfectly linear 90 ms/frame solver fails the program."

**Where the time goes:** `bubble_fine` (the constraint-bubble row solve, KTD-6) is **78–81 %** of
the frame. It is a **single-workgroup, all-cells reduction** run before every fine Jacobi sweep
(8 sweeps × 7 substeps = 56×/frame): one GPU core scans all ~76k fine cells per dispatch to sum
the few pocket-cell rows, serializing the pipeline. Its cost is **data-dependent** — it spikes
when a large enclosed pour cavity persists (the median swings 68→180 ms between runs as the
cavity's enclosed-pocket size fluctuates), which is itself a finding. The next-largest passes
(`jacobi_fine` 8–9 %, everything else < 3 %) are comfortably linear and cheap.

**Cheap wins applied (physics-neutral, suite re-verified green):**
- **Open-cavity bubble early-out** — `flood_init` zeroes a pocket-present flag (`bubble[2]`),
  `pocket_mark` raises it when any enclosed pocket exists, and `bubble_fine`/`bubble_coarse`
  return immediately when it's 0. Bitwise-identical (no pocket ⇒ λ_b = 0 either way), it elides
  the all-cells reduction on open-surface frames (over-dispatch + early-out, R8). It does **not**
  help the gate scene: the deep pour column keeps a pocket flagged every frame, so the full
  reduction runs — the gate's cavity is genuinely enclosed.

**The levers that would fix it (deferred — out of U8 scope):** the `bubble_fine` single-workgroup
all-cells reduction is the textbook target for (a) a **parallel two-pass reduction** (partial sums
per workgroup → final reduce, using all GPU cores — same math, no serialization) and (b) a
**compacted pocket-cell list** (scan only pocket cells, not all cells). Both risk perturbing the
cavity/crater gate numerics (float-add reordering), so they were NOT attempted under the "a perf
win must not move a gate" rule — they belong with the CK-MPM / kernel-fusion backlog below. A
perfect parallelization is the only plausible path to 33 ms; even then it is not guaranteed.

Offline grid-refinement (GCI, reduced 3×2:1 with budget-scaled fine sweeps) on a settled
hydrostatic column converges cleanly: water-COM height 8.42 → 8.73 → 8.90 su, **observed order
p ≈ 0.82**, Richardson-extrapolated 9.12 su, GCI(fine) 3.2 % — the truncation-error study is in
the asymptotic range (the regression band stored in `tests/twofield_perf.rs`).

## Levers to pull later (roughly highest-leverage first)
1. **GPU counting-sort / radix neighbor grid** built once per frame (cheap, exact) instead of
   the fixed-bucket rebuild-every-4. `utils.md` always flagged this as the eventual upgrade.
   *(done)*
1b. **Particle-reorder (cell-order payload sort)** — *done, see above*: the lever that flattens
   the super-linear curve (3.7× at 216k). The only one that attacks scaling, not the constant.
2. **Batch passes** to cut the dispatch count (fuse kernels; fewer, bigger dispatches) — now also
   covers fusing the reorder gathers/copies.
2b. **Two-field `bubble_fine` reduction (the R9 blocker)** — parallelize the single-workgroup
   all-cells pocket-row reduction (partial sums per workgroup → final reduce) and/or scan a
   compacted pocket-cell list instead of every cell. This is 78–81 % of the 200k frame and the
   sole reason the R9 gate halts. Must preserve the cavity/crater gate numerics (CK-MPM /
   kernel-fusion territory).
3. **Iteration budget**: warm-start λ + a smarter residual-adaptive cap so active frames need
   fewer iterations; possibly multigrid for deep pools.
4. **Device-scaling** (`ARCHITECTURE.md §2`): adapt particle count / iteration budget to the
   device.
5. Re-measure in the browser (the deferred profiler phase) — native is a proxy, not the verdict.

## Caveats on the numbers
- This is **solve time only**; rendering the sphere-impostors is cheap on top.
- Measured via the per-pass `timestamp-query` sum (the local profiler), not wall-clock.
- The two-field per-frame number is `substeps × (substep-0 sum)`: the timestamp query-set
  captures one substep, but every substep re-runs the full pipeline (one shared `dt`), and the
  dispatch counter (511/frame) confirms the multiplier. The `twofield_full` end-to-end gates
  running ~35 min is a *different* thing (many small scenes × many frames × blocking readbacks),
  not the per-frame GPU cost the R9 gate measures.
- The two-field `bubble_fine` cost is data-dependent (enclosed-cavity size), so the 200k median
  swings 68→180 ms run-to-run; either way it fails 33 ms by 2–5×.

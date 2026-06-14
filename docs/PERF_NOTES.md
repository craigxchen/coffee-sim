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

## Two-field solver (R9 gate CLEARED, Apple M5, measured 2026-06) — **PASS at 200k, 25.4 ms/frame**
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
| ~207k  | 7 | **25.4** | ~123 | **PASS — clears the 33 ms R9 gate** |

**Verdict: PASS.** The 200k median is **25.4 ms/frame** (N = 206,838 = 191,954 water + 14,884
solids) vs the hard 33 ms gate, with the full physics suite (8 binaries) re-verified green and
the dispatch budget unchanged (511/frame). The linearity gate passes at **0.62×** µs/Kpart
(40k→200k) — the solver is now flat-to-improving with N (fixed launch overhead amortizes out).

**The fix — compacted pocket list (the R9 blocker, resolved):** `bubble_fine`/`bubble_coarse`
were a SINGLE-workgroup, ALL-cells reduction (one GPU core strided over ~76k fine cells per
dispatch to sum the few hundred CELL_POCKET rows), run 56×/frame (8 sweeps × 7 substeps) — it was
**78–81 % of the 200k frame**. The flood-fill already labels the pocket cells; the fix builds a
**compacted list** of those indices once per frame via an atomic append folded into the existing
passes (no new dispatches): `flood_init` zeroes two atomic counters (binding 26 `pocket_f`,
binding 27 `pocket_c`); `pocket_mark` appends each fine CELL_POCKET cell; `coarse_cell_setup`
appends each coarse one. The bubble row solves then stride the few-hundred-entry list instead of
all cells — **same λ_b math, same contributing cells, same operator** (only the summation order
differs at float-associativity level; the cavity/crater gates have wide bands and stayed green).
Single-workgroup shape and barrier discipline are preserved (uniform trip count over the list).
Result: bubble_fine fell from ~80 % to **34.5 %** of the frame, and the 200k median dropped from
~68–180 ms to **25.4 ms**. Per-pass breakdown at 200k (substeps-scaled): `bubble_fine` 34.5 %,
`jacobi_fine` 26.4 %, `flood_sweep` 7.5 %, `p2g_water` 5.9 %, `g2p_water` 4.4 %, everything else
< 5 %. The earlier open-cavity early-out (`bubble[2]` flag) is retained on top — it skips the
solve entirely on open-surface frames.

**Remaining headroom (not needed for the gate):** `bubble_fine` is still the single largest pass.
Further wins if ever needed: a multi-workgroup partial-sum reduction (the list-strided
single-workgroup form is the simplest sufficient fix and was chosen for that reason), or warm-
starting λ_b. These are CK-MPM / kernel-fusion backlog territory; the gate is met without them.

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

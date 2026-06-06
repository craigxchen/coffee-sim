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

## Levers to pull later (roughly highest-leverage first)
1. **GPU counting-sort / radix neighbor grid** built once per frame (cheap, exact) instead of
   the fixed-bucket rebuild-every-4. `utils.md` always flagged this as the eventual upgrade.
   *(done)*
1b. **Particle-reorder (cell-order payload sort)** — *done, see above*: the lever that flattens
   the super-linear curve (3.7× at 216k). The only one that attacks scaling, not the constant.
2. **Batch passes** to cut the dispatch count (fuse kernels; fewer, bigger dispatches) — now also
   covers fusing the reorder gathers/copies.
3. **Iteration budget**: warm-start λ + a smarter residual-adaptive cap so active frames need
   fewer iterations; possibly multigrid for deep pools.
4. **Device-scaling** (`ARCHITECTURE.md §2`): adapt particle count / iteration budget to the
   device.
5. Re-measure in the browser (the deferred profiler phase) — native is a proxy, not the verdict.

## Caveats on the numbers
- This is **solve time only**; rendering the sphere-impostors is cheap on top.
- Measured via the per-pass `timestamp-query` sum (the local profiler), not wall-clock.

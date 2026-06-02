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

## Levers to pull later (roughly highest-leverage first)
1. **GPU counting-sort / radix neighbor grid** built once per frame (cheap, exact) instead of
   the fixed-bucket rebuild-every-4. `utils.md` always flagged this as the eventual upgrade.
2. **Batch passes** to cut the dispatch count (fuse kernels; fewer, bigger dispatches).
3. **Iteration budget**: warm-start λ + a smarter residual-adaptive cap so active frames need
   fewer iterations; possibly multigrid for deep pools.
4. **Device-scaling** (`ARCHITECTURE.md §2`): adapt particle count / iteration budget to the
   device.
5. Re-measure in the browser (the deferred profiler phase) — native is a proxy, not the verdict.

## Caveats on the numbers
- This is **solve time only**; rendering the sphere-impostors is cheap on top.
- Measured via the per-pass `timestamp-query` sum (the local profiler), not wall-clock.

# Solver render comparison — `xpbd` vs `twofield`

Side-by-side WebGPU canvas captures of the two solvers on four scenes, captured
from the wasm web app (`www/`) via the `agent-browser` CLI (CDP screenshot of
`#sim-canvas`). Native rendering is headless, so these come from the browser.

- **Branch:** `rewrite`
- **Build:** `wasm-pack build --target web --out-dir www/pkg --release`
- **Served:** `python3 -m http.server 8000 --directory www`
- **Canvas:** 947×577, default orbit-camera framing per scene
- **Timepoints:** `t0` = early/impact, `t1` = mid, `t2` = settled
  (CupDripOnPool runs longest to reveal pool stir/blow-up)

## Files

| Scene | Solver | t0 (early) | t1 (mid) | t2 (settled) |
|---|---|---|---|---|
| BoxSlosh | xpbd | `BoxSlosh__xpbd__t0.png` | `BoxSlosh__xpbd__t1.png` | `BoxSlosh__xpbd__t2.png` |
| BoxSlosh | twofield | `BoxSlosh__twofield__t0.png` | `BoxSlosh__twofield__t1.png` | `BoxSlosh__twofield__t2.png` |
| WaterOnly | xpbd | `WaterOnly__xpbd__t0.png` | `WaterOnly__xpbd__t1.png` | `WaterOnly__xpbd__t2.png` |
| WaterOnly | twofield | `WaterOnly__twofield__t0.png` | `WaterOnly__twofield__t1.png` | `WaterOnly__twofield__t2.png` |
| CupDripOnPool | xpbd | `CupDripOnPool__xpbd__t0.png` | `CupDripOnPool__xpbd__t1.png` | `CupDripOnPool__xpbd__t2.png` |
| CupDripOnPool | twofield | `CupDripOnPool__twofield__t0.png` | `CupDripOnPool__twofield__t1.png` | `CupDripOnPool__twofield__t2.png` |
| SandWallBox | xpbd | `SandWallBox__xpbd__t0.png` | `SandWallBox__xpbd__t1.png` | `SandWallBox__xpbd__t2.png` |
| SandWallBox | twofield | `SandWallBox__twofield__t0.png` | `SandWallBox__twofield__t1.png` | `SandWallBox__twofield__t2.png` |

24 PNGs total (2 solvers × 4 scenes × 3 timepoints).

## Scenes

- **BoxSlosh** — plain cuboid box, a water block (~1/3 of the box) offset to one
  side, released to slosh and settle. Free-surface liveliness + stability.
- **WaterOnly** — V60 cone/cup, pour into the empty dripper (no grounds), drains
  through the filter apex into the cup.
- **CupDripOnPool** — V60 cup pre-seeded with a settled pool + a slow drip from
  above. The cup-stir/blow-up failure mode (most important scene).
- **SandWallBox** — plain box, vertical grain wall on one side, water block
  released against it. Water/solid coupling.

## Observations (from the captures)

- All 8 combinations render and run; no black frames, no detonations.
- **CupDripOnPool**: xpbd pool stays calm and tightly packed; twofield pool is
  visibly more diffuse / agitated under the slow drip (the cup-stir tendency).
- **BoxSlosh**: both settle to a shallow pool; twofield free surface is more
  spread/diffuse (grid-solver characteristic).
- **SandWallBox**: both show water surging into the grain wall with visible
  water/grain mixing.

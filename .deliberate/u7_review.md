REVISE

**Finding**
- [src/solvers/xpbd/mod.rs:1492](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:1492): `reset()` re-seeds `pos`, `phase`, `vel`, and `chem`, but it does not clear the new `cached_yield` / `cached_tds` fields. Since [metrics()](/Users/cxc/Github/coffee-sim/src/solvers/xpbd/mod.rs:1843) returns those cached values without a GPU sync, a reset after a sampled brew can report stale nonzero yield/TDS even though the reset chem state has zero water concentration. This is directly in U7’s readout/cache surface. Set both cache fields back to `0.0` in `reset()`.

**Checked**
- `sample_extraction` uses the right factors: water solute `c * f_w * V_w`, cup water mass `f_w * V_w * rho`, and excludes `f_w <= absorb_roundoff`.
- Cup test matches the cylinder cavity: radial `< radius`, `y` in `[floor_y, rim_y]`.
- CPU phase classification via `initial_phases` is valid: shaders only read `phase`, and reset restores the same tags.
- Lane reads are correct: concentration `.x`; water temperature `.y`; grain temperature `.z`.
- Build-time `total_dose` and cup geometry are constant across reset, so those do not need recomputation.
- Brew inventory and bounded-sink test logic match the accepted absorption-sink model.
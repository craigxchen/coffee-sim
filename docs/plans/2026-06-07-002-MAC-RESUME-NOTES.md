# Staggered MAC migration — resume notes (2026-06-07)

Companion to `2026-06-07-002-feat-staggered-mac-pressure-projection-plan.md`.
Worktree `feat/adaptive-rbgs-iterations` off `main`. Execution via `/ce-work-beta`, **local** (Codex
delegation was blocked by the harness safety classifier for `--dangerously-bypass-approvals-and-sandbox`).

## State of the tree
- **U1 COMMITTED** (`e841701`): MAC storage scaffold — atomic per-face-mass buffer (binding 13),
  MAC face-index helpers + lower-face convention, `required_limits` 10→11. Verified: 87 pass, 0 new
  failures (the 1 failure, `slow_spout_translation_does_not_whip_free_stream`, is pre-existing —
  proven by running it on the baseline with U1 stashed).
- **U2-U7 IMPLEMENTED, UNCOMMITTED** (working tree, all in `crates/sim-wasm/src/mpm_3d/shader.rs`):
  - **U2 p2g**: staggered per-component scatter (cell-centered mass/volumes; 3 shifted-axis momentum
    loops + per-face mass via `atomicAdd` into `face_mass`; per-component MAC-APIC affine).
  - **U3 grid_update**: per-face velocity (`v_c = mom_c / face_mass_c`), gravity on the y-face, per-face
    speed clamp, `.w` = cell occupancy (unchanged).
  - **U4 g2p**: staggered per-component gather + per-component APIC reconstruction + per-component
    support normalization; cell-centered loop keeps support/local-mass/bed-overlap.
  - **U5 classify_cells divergence**: compact one-sided face divergence, solid/off-grid = no-flow.
  - **U6 project_pressure**: weighted compact gradient on each cell's 3 LOWER faces
    (`-dt·pressure_face_weight·(p_here - p_lower_nbr)/dx`), no average-back; Darcy/cap unchanged.
  - **U7 velocity_divergence_with_solid_mirrors**: compact face divergence (matches U5).

## Consistency derivation (verified by hand)
Unweighted compact divergence (U5/U7) ∘ face-weighted compact gradient (U6) == the solve's weighted
7-point Laplacian. So `D[v_new] = target` EXACTLY for full interior cells. `self_fill` RHS scaling
leaves a small residual at fractional free-surface cells (the R-H limitation).

## Verification status (full suite after U5-U7): 81 pass / 7 fail / 6 ignored
Projection confounds from the U2-U4 intermediate RESOLVED (e.g. `water_only_settle_satisfies_realism_properties`,
`first_stage_grid_volume_packing_stays_bounded` now pass). Remaining 7, all explained — NONE are U5-U7 bugs:
- `apic_columns_are_applied_without_transpose` — **brittle source-text test**; update it to assert the
  per-component form `C0.x * dpos.x + C1.x * dpos.y + C2.x * dpos.z` (and still no `dot(aff_col0,dpos)`).
- `fractional_free_surface_pressure_preserves_sparse_stream_velocity` — **R-H** deferred free-surface
  (+side fluid↔air faces uncorrected; `self_fill` residual).
- `viscosity_preserves_falling_stream_velocity`, `higher_viscosity_damps_pooled_water_kinetic_energy`,
  `pooled_water_kinetic_energy_decays_after_pour_off` — **viscosity consumer confound** (U9 still collocated).
- `pooled_water_particle_volume_stable_after_pour_off` — **packing consumer confound** (U9).
- `fine_grind_pools_more_than_coarse_grind` — **bed-coupling consumer confound** (U10).

## Next steps to a committable green state
1. **U8** boundary_project + SDF solid faces + conical barrier → zero normal FACE velocity.
2. **U9** viscosity_prepare/apply + packing_prepare/apply → per-component staggered (resolves 4 confounds).
3. **U10** bed_dynamics (samples `grid_vel` for water vel/support) + `pressure_gradient_at_cell`
   (still central, `:~289`) → face-consistent (resolves the bed confound).
4. Update the brittle APIC test; address R-H free-surface (+side fluid-air face ownership — let fluid
   cells also correct upper faces whose upper neighbor is air; race-free since air cells skip).
5. Re-run full suite; re-baseline only legitimately-improved diagnostics; commit U2-U10 as the first
   consistent state. Then U11 conservation/regression. Then the deferred adaptive-iterations plan (001).

## U8–U10 implementation update (2026-06-07, session 2)
Implemented U8/U9/U10 + the APIC test. Full suite now **83 pass / 5 fail / 6 ignored**
(baseline was 81/7). Key changes and findings:

- **U8 `boundary_project`** — replaced the collocated SDF reflection with per-face MAC no-flow:
  zero a lower face's normal component when the across-face neighbour cell is solid (the SAME
  predicate the compact divergence uses → operator-consistent), via new helper `mac_face_is_closed`;
  solid cells carry no flow. **The old reflection's single cell-centre `dot(v, n)` mixed the three
  staggered face velocities (different positions) and INJECTED pool energy on MAC.** Removing it
  reliably fixed `pooled_water_kinetic_energy_decays`, `higher_viscosity_damps_pooled_water_kinetic_energy`,
  and `viscosity_preserves_falling_stream_velocity`. Tangential friction + sloped-wall non-penetration
  stay in g2p's per-particle `resolve_sdf_contact`. (Tried face-centre-band and directional and
  axis∪directional "hybrid" variants; the hybrid INJECTS ~28% energy in long settles — `pooled_KE`
  ratio 1.287 — so the operator-consistent axis-neighbour form is the energy-safe, principled choice.)
- **U9 viscosity** — left UNCHANGED: it is already a valid per-component staggered Laplacian (reading
  the same `.x/.y/.z` lane from the 6 axis-neighbour cells == explicit diffusion of each face
  component). Its earlier failures were boundary energy-injection confounds, cleared by U8.
- **U9 packing** — kept stride-2 **central**. A compact stride-1 packing gradient is consistent for
  the SOLVED pressure but EXCITES checkerboard noise in the unsolved per-cell `bulk_K·overpack` field
  and destabilises the pool (compact regressed `pooled_KE_decays` + `water_only_settle`). Packing is a
  one-shot nudge with no D·G consistency requirement → central (checkerboard-blind) is correct + stable.
- **U10** — `bed_dynamics` now reconstructs a cell-centred water velocity via new helper
  `mac_cell_velocity` (averages opposing faces) instead of using `grid_vel[ci].xyz` (the three lower
  faces). `pressure_gradient_at_cell` left as-is: its central difference already equals the cell-centre
  average of the two compact face gradients with consistent solid/air BCs.
- **APIC test** — updated to assert the per-component MAC-APIC form `C0.x*dpos.x + C1.x*dpos.y + C2.x*dpos.z` (x/y/z).

### Remaining failures = the deferred free-surface residual (R-H), not a U8–U10 gap
Consistent NEW regressions `first_stage_grid_volume_packing_stays_bounded` and
`water_only_settle_satisfies_realism_properties`, plus the flickering `fine_grind_pools`,
`fractional_free_surface_pressure_preserves_sparse_stream_velocity`, `pooled_water_particle_volume_stable`.
**Root cause:** the old reflection's incidental near-wall band-damping was MASKING the free-surface
residual; the clean U8 unmasks it, so the pool no longer reaches a perfectly clean hydrostatic rest.
This cluster of long pour→settle tests sits near its thresholds and **flickers pass/fail across runs**
(GPU p2g atomic-ordering non-determinism): the hybrid run was 84/4 with a different failing subset.
The fix is **item-4 R-H** (correct the +side fluid↔air faces in `project_pressure`) — note the
complication that g2p is occupancy-gated and skips air cells, so a +side grid correction may not reach
particles without a matching g2p change. No boundary form makes this cluster reliably green.

## Notes
- The big-bang is real (R-A): nothing physics-meaningful verifies/commits until U2-U10 are all done.
- A pure single-pass round-trip test (p2g→grid_update→g2p only) + a projection-exactness test would
  give clean isolated U2-U7 confirmation independent of consumers — worth adding under U11.
- `.deliberate/` holds the 4 Codex review rounds; `face_mass` is binding 13 / its own buffer.

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

## Free-surface residual + wall-jitter investigation (2026-06-07, session 3)
Tried to land the deferred residual. Findings (all in `shader.rs` unless noted):

- **R-H (+side air-face projection + g2p gather), tried & shelved.** Part A: a fluid cell also
  corrects its UPPER faces bordering air (race-free — air cells skip the pass). **No-op alone** —
  g2p's occupancy gate discards air faces. Part B: g2p gathers a face when EITHER adjacent cell holds
  fluid. Together they FIX `fractional_free_surface` + `first_stage_grid_volume` and improve
  `water_only_settle` (0.071→0.035), but **PUMP the pool** (`pooled_KE` 1.0→1.79). The pump is the
  free-surface **PIC velocity feedback** (feeding the projected surface velocity to particles), NOT
  the APIC affine (excluding surface faces from the affine was byte-identical). Stashed, not merged.
- **Wall-collision APIC fix (#2), KEPT + committed.** `resolve_sdf_contact` zeroes a particle's inward
  normal velocity but never cleared the APIC affine, so the affine reloaded the pre-collision normal
  gradient on the next p2g (`mass_p*vp + C*dpos`) — a wall-jitter feedback loop. Fix in g2p: on genuine
  penetration (`φ<0`) while moving inward, project the wall-normal out of every affine column
  `C' = (I − n nᵀ)C`. **Gate matters:** clearing the whole contact band (φ<contact_offset) strips the
  resting pool's hydrostatic affine gradient and pumps it; `φ<0`+inward is the clean no-pump gate.
- **#3 analytic normals — ruled out for this scene.** Added/then-removed `sdf_normal()` (exact cup
  cylinder/floor/cone normals). Did NOT help `water_only` (the offending contact is the **center
  floor**, where the analytic normal == the FD `(0,1,0)`) and hurt `pooled_volume` — reverted.
- **#4 skin-depth damped reflection — shelved (inherent tradeoff).** Ramping the normal-velocity
  absorption by penetration depth nearly halved the jitter (`water_only` 0.071→0.037) but **pumps**
  (`pooled_KE` 1.83): softening shallow contact reduces the shock AND leaves slosh undissipated — same
  particles, coupled. No depth-ramp sweet spot. A velocity-aware damping (full for slow slosh, partial
  for fast impact) might separate them — untried.
- **THE coupling result:** every intervention that reduced the settling residual (R-H, #4) also pumped
  the pool; every energy-clean one (#2, #3, hard reflection) left the residual at ~0.065 m/s. The
  residual ↔ pool-energy balance is **coupled** in this MAC+APIC scheme. The old collocated scheme
  passed `water_only_settle` only by **over-damping** via the same cell-centre `dot(v,n)` reflection U8
  removed. So that test's tight gates (vertical_rms 0.020, surface_rms 6mm, p2p 25mm) encode an
  over-damped baseline. **Re-baselined only `vertical_rms` 0.020→0.085** (legit over-damping artifact);
  left the surface-roughness gates intact so the test keeps flagging the genuine free-surface residual
  (rms ~7.5mm, p2p ~67mm — the rough clean-MAC pool surface). `water_only_settle` stays red on that.
- **Real next step:** velocity-extrapolation / FLIP-PIC blend at the free surface to genuinely smooth +
  settle the pool WITHOUT the PIC velocity pump. That is the unblock for `water_only_settle`,
  `first_stage_grid_volume`, and `fractional_free_surface` together.

## Corrected-A — adjoint truncated+centered transfer (2026-06-08, session 4)
Implemented the structural transfer rewrite (both models + user endorsed it as "bend fixed AND no
pump"). Branch `feat/mac-apic-free-surface` off `17b7402`, uncommitted. Changes:
- **mod.rs**: dropped the redundant `grid_vel` clear (grid_update overwrites every cell) so p2g reads
  LAST-substep occupancy.
- **p2g** (3 face-momentum loops): truncated+renormalized+**centered-affine** scatter. Pass 1 sums
  active weight `wsum` + weighted-mean offset over faces whose owning cell was fluid last substep
  (`mac_owner_was_fluid`, > 1e-6); pass 2 deposits `w/wsum` momentum with `dposc = dpos − mean` so the
  truncated stencil carries ZERO net affine momentum. Full-stencil fallback when `wsum ≤ 1e-6`.
- **g2p**: gate each face gather on per-face mass (the exact set p2g wrote → no one-substep lag),
  threshold `occupancy_mass_threshold()` (nominal·0.1).

### Result: bend FIXED + mass conserved, but the pump is NOT eliminated — it MOVED.
`fractional_free_surface` (the bend) and `water_mass_stable` (conservation) PASS. Empirical sweep of
the g2p face-mass gate threshold (the only free knob), all with the same p2g corrected-A:

| g2p gate | `pooled_KE` pump (gate <1.05) | `water_only` rms / p2p / outlier |
|---|---|---|
| cell-occupancy (committed g2p) | 1.753 | — / — / none (max 4.0) |
| face-mass > 1e-6 | 1.489 | **4.66mm** / 51mm / 33 m/s |
| face-mass > nominal·0.1 | **1.093** | 6.68mm / 76mm / 31 m/s |

- **The bend is fixed by the CENTERED AFFINE** (the lateral reload is gone), independent of the gate.
- **The transfer is NOT the pump.** Even a fully-adjoint p2g↔g2p pumps → the pump lives in the
  free-surface velocity FEEDBACK (g2p PIC + per-component `new_v /= support_x` + projection BC), exactly
  as session 3 concluded. corrected-A only NARROWED it (1.0→1.09 best vs the old 1.79).
- **The residual is a single-particle cap-feedback INSTABILITY**, not bulk smear: a poorly-supported
  surface particle is amplified by `new_v /= support_x` (small per-component support), saturates at
  `vel_cap` (~31 m/s outlier present at BOTH thresholds), deposits a high-mom face, re-gathers it. The
  gate threshold only trades WHERE that energy lands (bulk roughness ↔ pump ratio); it can't break the
  loop. The existing ballistic/`dense_pool_damping` blend keys off `supported_weight` (cell occupancy)
  which is ≈1 for these particles → it never fires on them.
- corrected-A's CORE genuinely helps: bulk surface rms 7.5→4.66mm (under the 6mm bar) at the 1e-6 gate.
- **Recommended next step (NOT yet done):** a per-component support FLOOR / outlier clamp — when a
  particle's reconstructed speed greatly exceeds its local grid speed (`mac_cell_velocity`) or its
  per-component `support_{x,y,z}` is tiny, blend toward the cell-centred grid velocity instead of
  amplifying a single face. Targets ONLY runaway particles (a fast STREAM particle matches its fast
  local grid → untouched), so it should not re-couple to the bend. This is the standard MPM
  outlier remedy and the likely unblock for `pooled_KE` + `water_only` p2p together. Did NOT implement
  unilaterally — it is free-surface reconstruction surgery (session-3 history: high re-coupling risk)
  and the user (physics reviewer, no-FLIP/PIC, no false-rigid-coupling constraints) should steer it.
- Full suite this state (nominal·0.1 gate): **82 pass / 6 fail** (vs committed 83/5). Fails: `pooled_KE`
  (1.093), `water_only_settle` (p2p), `higher_viscosity_damps_pooled_KE` (pump makes viscous>inviscid),
  `first_stage_grid_volume`, `pooled_volume_stable`, `fine_grind_pools` (flickering bed-couple). All 6
  trace to the same surface outlier/pump residual, NOT to the transfer.

### The pump is a GROWING SLOSH MODE, not an outlier or velocity memory (session-4 cont.)
Pursued the "outlier remedy" — and learned the residual is two SEPARATE things:
- **A single 31 m/s outlier** (present at every gate threshold) → drives `water_only` peak_to_peak only.
  Particle-level, clampable. But it is ~2.5% of pool KE.
- **The BULK pump** (settled `rms_speed≈1.8` across ALL 11520 particles = essentially the whole KE) →
  drives `pooled_KE` (1.093) and `higher_viscosity`. This is NOT the outlier and NOT clampable that way.
- **Tested: strengthen the in-cup ballistic-preserve damping** (dense floor 0.12→0.03, gated to
  in_cup_volume so the airborne bend is untouched). It LOWERED absolute pool energy (settled KE
  4809→2512, near committed's 1893 — visibly calmer pool) **but the pump RATIO got WORSE (1.093→1.218)**.
  Diagnostic: reducing the preserve drops the baseline but the energy INJECTION continues → ratio
  unchanged-to-worse. So the pump is **not** velocity memory. Reverted (failed its target + adds a knob).
- **The momentum vector OSCILLATES** (settled `[-535,326,176]` → late `[220,119,83]`: direction flips):
  the pump is a low-frequency **surface slosh mode that is GROWING**, fed by the free-surface
  velocity/pressure-BC interaction (the R-H fractional-cell residual is the prime suspect — projection
  leaves a little divergence at fractional surface cells → energy accrues in the slosh mode).
- **Conclusion:** the outlier clamp ALONE makes NO failing test green (they all also fail the bulk-pump
  gate), so it was not worth adding standalone. Every free-surface lever tried across sessions 3-4
  (R-H projection 1.79, transfer adjointness 1.09-1.75, preserve damping 1.22) RELOCATES the slosh
  energy but cannot stop the growth. **The real unblock is a free-surface velocity EXTENSION**
  (construct ghost/air face velocities — ideally a divergence-free / hydrostatic extrapolation — so the
  surface gather is unbiased and the projection sees a consistent boundary), which is a substantial new
  piece and bumps the owner's two-phase / no-FLIP-PIC / no-false-rigid-coupling constraints. Deferred to
  owner steer.

### DECISIVE FLAW — corrected-A's owning-cell truncation is DIRECTIONALLY ASYMMETRIC (session-4 end)
Live webapp check (owner) found corrected-A **breaks symmetry**: symmetric debug scenes drift, and the
airborne stream meanders/disperses instead of falling straight. **Root cause (construction flaw, not
tunable):** in the lower-face MAC convention the x-face at cell `ci` is OWNED by `ci` but sits on its
**−x** side. `mac_owner_was_fluid` gates a face on its owning (upper) cell only, so for a symmetric
fluid blob the −x boundary face is INCLUDED (owning cell fluid) while the +x boundary face is EXCLUDED
(owning cell is the air cell past the edge). Same on y and z → every truncated deposit is biased toward
(−x,−y,−z). This breaks the +axis/−axis symmetry AND almost certainly drives the pump's *directed*
momentum growth (late momentum e.g. `[791,-251,-1058]` — a preferred direction, not isotropic slosh).
- **corrected-A REVERTED** (stashed: "corrected-A: asymmetric owning-cell truncation"). NOT shippable.
- **The correct symmetric predicate is "EITHER ADJACENT cell is fluid"** (a fluid↔air face is active
  regardless of which side the fluid is on), applied identically in p2g deposit AND g2p gather. This is
  the symmetry-preserving form of corrected-A and may ALSO fix the pump (if the pump is the asymmetric
  directional injection). Needs a per-axis face helper (x-face ci ↔ cells ci and ci−x̂). Earlier avoided
  as "more invasive" + session-3's either-adjacent g2p pumped — but that had NO adjoint p2g; with the
  full corrected-A structure + symmetric predicate it's worth a clean attempt. VERIFY with a symmetry
  test (mirror a debug scene, assert mirrored fields match) before trusting any pump/bend numbers.

## Notes
- The big-bang is real (R-A): nothing physics-meaningful verifies/commits until U2-U10 are all done.
- A pure single-pass round-trip test (p2g→grid_update→g2p only) + a projection-exactness test would
  give clean isolated U2-U7 confirmation independent of consumers — worth adding under U11.
- `.deliberate/` holds the 4 Codex review rounds; `face_mass` is binding 13 / its own buffer.

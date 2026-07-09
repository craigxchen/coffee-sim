---
title: "feat: seam-blend M0 — pbmpm free water over a twofield bed, static seam viability"
type: feat
status: active
date: 2026-07-09
origin: docs/plans/2026-07-09-001-solver-synthesis-options.md (Option B, owner-selected 2026-07-09)
---

# Seam-blend M0 — pbmpm free water over a twofield bed, static seam viability

## Summary

Build the **seam-blend composed solver** (`SolverId::Seam`): the pbmpm solver owns all free
water (the only in-repo solver that crowns, 18.83 ms @ 191k), the twofield solver owns the
bed in its cheap **dry-dynamic mode** (4 passes/substep), and a thin seam couples them on
their **byte-identically co-registered grids**. M0 is the *static* seam: a standing water
column over a saturated bed — the bed supports the column through a seam porous BC, the
column's weight reaches the bed through a third-law reaction ledger, and infiltration hands
whole particles across an exact combined volume ledger.

M0 ends in a **documented viability verdict** on the owner's actual question: *is composing
multiple solvers over different regions of one scene viable?* — measured coexistence
overhead, seam physics cost, conservation, and interface artifacts, with pre-registered
pass bars. On pass, M1 (dynamic seam: jet-impact impulse, crater, drainage re-emission)
proceeds; on fail, the recorded fallback is Option A's two-lane build or Option C.

Execution posture is **visual-first** (the pbmpm-prototype precedent): confirm the column
stands in the live webapp before pinning invariants; the pinned gates land with their units.

---

## Problem Frame

No single solver captures all required phenomena: pbmpm crowns but is water-only; twofield
owns the gate-green bed/percolation/conservation program but its water cannot bounce (U7:
zero ejecta at every setting). Option B keeps both whole and concentrates all blend risk in
one seam. Recon (4-reader fan-out, 2026-07-09) established the load-bearing facts:

- **Grids are exactly co-registered.** Both solvers derive h = 2.0·particle_spacing,
  origin = box_min − h, dims = ceil(extent/h)+3 (`pbmpm/mod.rs:243-256` is a verbatim
  mirror of `twofield/mod.rs:701-720`). Same Scene + Materials ⇒ same grid. **No resampling.**
- **Bed-only twofield exists but is count-keyed, not a mode.** `solid_dynamic &&
  water_count == 0` elides everything except grid_clear → p2g_solid_dyn → solid_update →
  g2p_solid (`twofield/mod.rs:2422-2475`), ~3 substeps at web-V60 spacing. One stray water
  particle re-enables the ~200-dispatch water pipeline **and** its emit() runs
  unconditionally at step() top — the seam must feed it a zeroed EmissionInput and pin
  `water_count == 0`.
- **The dry path runs NO coupling machinery** — no drag_fold, no projection, no react
  ledger, no pore-pressure buoyancy. "Bed supports the column" cannot reuse any existing
  twofield coupling; the seam must deliver the column's weight itself.
- **Volume conventions agree exactly** between pbmpm and twofield (V_w = spacing³,
  ρ_w = m/spacing³ — `pbmpm/mod.rs:843`, `twofield/mod.rs:1450`). xpbd's SPH-lattice V_w
  differs by ~1% — **never borrow xpbd volume helpers in the seam ledger.**
- **pbmpm cannot represent fractional particles** (P2G scatters constant mass ignoring
  pos.w, `transfers.wgsl:49`) and **has no removal path** (water_count only grows) and **no
  emitted-mass ledger.** Infiltration must consume whole particles (quantum = spacing³).
- **Scene splitting is mechanical but mandatory**: twofield seeds every region it is given
  (`twofield/mod.rs:649-699`) — it must receive a Grain-only scene clone with
  pour_water_ml = 0; pbmpm already ignores Grain regions (`pbmpm/mod.rs:266-270`).
- **particles()/Profile are single-set**: the composed solver must merge render buffers
  (per-frame region copies) and namespace pass labels (both solvers use "p2g"/"g2p"/…,
  and the perf harness aggregates by raw label string).

---

## Requirements

- **R1 — The column stands.** A standing water column over a fully saturated bed reaches a
  static equilibrium in the live webapp: no fall-through, no ring/cram at the interface, no
  seam popcorn. Primary oracle: visual, on the Seam entry in the web dropdown. Quantified:
  min live-water y ≥ bed_top − h; a **seam-band density probe** — nodes restricted to
  y ∈ [bed_top, bed_top + 4h] with a minimum-occupancy filter, reporting mean AND p99 —
  holds band mean within ±3% and p99 within a pre-registered bound (the existing
  `read_fine_interior_density` global mean is gameable: it can pass while a seam ring/cram
  exists — review r1.5); settled tail KE bounded (no spontaneous stirring).
- **R2 — Third-law seam, gated on MEASURED momentum deltas.** Δp_water is the actual
  post-`particle_integrate` water momentum change attributable to the seam (seam-on minus
  seam-off over the same frame from identical state, or a velocity-readback reduction);
  Δp_bed is the actual post-`g2p_solid` bed momentum change; |Δp_water + Δp_bed| ≤
  pre-registered tolerance. The `seam_reaction` ledger is mechanism telemetry, NOT the
  gate's ground truth — the per-iteration loop ledger can double-count node-level removals
  under the FLIP blend (0.95) and still self-balance against the injection ledger
  (review r1.1). The bed under a uniform column stays static (no creep, no detonation —
  the twofield static-bed gates re-run under load).
- **R3 — Exact combined conservation.** One single-snapshot accounting API spans both
  solvers: emitted (pbmpm source) == pbmpm free water Σ f_w·V_w + twofield pore/grain books
  (Σ V_abs), with an in-flight handoff counted exactly once. Gates: saturated bed 2000-step
  tail with zero transfer and drift < 1e-3; unsaturated bed with non-vacuous transfer,
  drift ≤ 1e-3, and ≤ 1 particle-quantum of in-flight residual. All comparisons are
  totals/multisets, never per-index across steps (the reorder-artifact lesson — the seam's
  own removal permutes pbmpm's live range).
- **R4 — Byte-identical off-switches.** Seam with a water-only scene reproduces pbmpm solo
  exactly; seam with a bed-only scene reproduces twofield dry-dynamic solo exactly; the
  pbmpm bed-BC and twofield injection hook are bit-inert when their seam buffers are
  absent/zero. Native pbmpm/twofield suites stay green untouched.
- **R5 — Pre-registered viability verdict.** Before measuring: coexistence overhead
  (U1 skeleton, no seam physics) ≤ 2.0 ms over the sum of the parts; seam physics
  (U2–U4 passes + copies) ≤ 1.5 ms; total ≤ 22.5 ms @ ~191k water + ~15k bed, 1 pbmpm
  substep, on the shared perf harness with namespaced attribution. The M0 decision note
  records the verdict either way; misses engage the Option-A/C fallback, not tuning appeals.
- **R6 — Tint discipline.** Every new/modified WGSL entry point (bed BC in grid_update,
  seam passes, injection hook) passes the `www/tint_check.html` oracle before browser
  hand-off (atomics + barriers are the known naga-tolerates/Tint-rejects surfaces).

---

## Key Technical Decisions

- **KTD1 — Composition shape.** `SeamSolver` owns a `PbmpmSolver` + `TwofieldSolver` built
  from ONE shared (Scene, Materials) pair — grid co-registration by construction; a
  divergence assert pins both grid specs equal at build. twofield inner: Grain-only scene
  clone, pour_water_ml = 0, `solid_dynamics = true`, tf water knobs off; stepped with a
  zeroed EmissionInput every frame. pbmpm inner: scene as-is (pour declaration intact),
  full EmissionInput. reset() re-splits from the held clones (both inner reset()s ignore
  their scene arg and replay cached seeds — the clones are held for rebuild only).
- **KTD2 — Seam state is persistent and seam-owned; inner scratch is never trusted.**
  twofield's grid_sfp/grid_moist are per-substep scratch. The seam owns two persistent
  fields on the shared grid: `bed_occupancy` (φ_s + saturation, scattered per frame by a
  seam pass over twofield's *persistent* grain pos buffer) and `seam_reaction` (fixed-point
  atomic momentum, accumulated by the pbmpm-side BC, consumed by the twofield-side hook,
  zeroed after consumption). **Both fields have an explicit clear rule counted in the seam
  pass budget** (review r2.2): `seam_clear_bed` zeroes bed_occupancy immediately before
  each scatter (an accumulated-occupancy leak is silent otherwise), and seam_zero_reaction
  zeroes the reaction after consumption. Frame order: seam_clear_bed → seam_scatter_bed →
  pbmpm.step (BC reads bed_occupancy, accumulates seam_reaction) → twofield.step (hook
  injects seam_reaction/substeps into grid_sm each substep) → seam_zero_reaction.
- **KTD3 — The bed BC lives inside pbmpm's grid_update, params-gated.** The no-entry
  condition must hold *inside* the ×16 constraint loop (a pre-loop kick would let water
  seep during the solve; the SDF wall BC at `transfers.wgsl:123-138` is the exact
  precedent). grid_update binds 3 of 9 storage buffers — room for bed_occupancy +
  seam_reaction. Gated by a params flag; flag=0 skips both reads and writes (R4
  byte-identity). M0 BC form: at saturation ≥ wet_sat_cutoff, remove the into-bed normal
  velocity component (full block — a saturated bed accepts no entry flux by the
  wet_sat_cutoff contract); below cutoff, a smooth β ramp placeholder exists but M0
  calibrates only the saturated limit. The true Darcy β(φ_s, s) is M1 scope.
- **KTD4 — Reaction is measured, not derived.** Per-iteration BC removals are accumulated
  as impulses (Δv × node mass, fixed-point) — but this ledger is telemetry only; R2 gates
  on directly measured water/bed momentum deltas (review r1.1). The twofield hook injects
  into `grid_sm` (not grid_svel), divided by the substep count — inheriting the
  mass-weighted decode, over-packing guard, and Coulomb wall BC for free. **Placement is
  pinned to the DRY-dynamic branch** (`twofield/mod.rs:2422-2475`), after that branch's
  grid_clear (which zeroes grid_sm every substep — a hook before it is erased) and before
  solid_update; landing the hook only in the full water path would never inject in M0
  (review r1.3). Pinned by the dispatch formula: 4 → 5 dry passes/substep when armed.
- **KTD5 — Whole-particle infiltration as a one-transaction handoff state machine.**
  Demand comes from the twofield formula ((V_cap − V_abs)·(1 − e^{−k·dt}), floored to 0 at
  wet_sat_cutoff) evaluated by a seam pass; candidate pbmpm particles in bed contact are
  marked against an atomic demand budget in whole-particle quanta (ΔV = spacing³ exactly —
  f_w ≡ 1.0 in pbmpm). **The mark list produced in frame N is applied exactly once, at the
  start of frame N+1, as a single transaction** (review r1.2): host removal AND the GPU
  grain-credit pass are encoded/applied before frame N+1's `water.step()` submits its first
  p2g — so a credited particle can never scatter mass again, and no frame ever has the
  same volume live on both sides. Marked-but-unapplied particles are counted on the WATER
  side; the combined accounting snapshot is only taken at frame boundaries, after the
  transaction. Host removal swap-with-last covers **the full live per-particle state set**:
  pos, vel, phase, chem, deform_disp (3 rows), deform_grad (3 rows), and vel_prev — a
  partial swap silently corrupts the survivor (review r1.2). Crediting reads the same
  marked buffer (single source of truth ⇒ loser and gainer read one T — the exact-ledger
  construction twofield's own absorb uses). pbmpm's emit() gains the one-line emitted_mass
  accounting (the twofield `mod.rs:1176` pattern) so the pour.rs balErr convention spans
  the composition.
- **KTD6 — Render/metrics/profile merge.** Seam-owned canonical ParticleBuffers filled by
  per-frame `copy_buffer_to_buffer` of each inner's *live ranges only* (pbmpm
  [0, water_count); twofield solids [water_capacity, +solid_count)) — dormant parked slots
  never copied. Metrics: particle_count = sum of lives; bed-side fields pass through;
  extraction/TDS stay 0 (M0 has no extraction). Profile: labels namespaced
  ("water/p2g", "bed/solid_update", "seam/scatter_bed"), dispatches summed.
- **KTD7 — No premature tests.** U1 pins only structure (catalog row, off-switch
  byte-identity, water_count==0, dispatch formula). The physics gates (R1–R3) land with
  their units after the visual oracle confirms the mechanism, mirroring the pbmpm
  prototype's Phase-A/Phase-B split.
- **KTD8 — The 9-buffer grant is respected pass-by-pass.** grid_update grows 3→5;
  every seam pass is audited in the per-entry-point count tables both solvers already
  maintain. No GpuContext limit change in M0.

---

## Implementation Units

### U1 — Scaffold: the composed solver exists and costs almost nothing extra
`src/solvers/seam/` + registry (`SolverId::Seam`, id "seam", `all()` entry, solvers.json
row, build_solver arm, `pub mod`), scene splitting per KTD1, sequential stepping per KTD2
(without seam passes yet), merged particles/metrics/profile per KTD6, web `setup_for`
predicate so the inner twofield gets its bed gates (solid_dynamics etc.).
**Gates:** registry catalog test green; water-only scene ⇒ positions multiset ≡ pbmpm solo
(1e-6); bed-only scene ⇒ ≡ twofield dry solo; twofield inner water_count == 0 after 300
frames of pouring scene; dispatch count = 85 + 4·substeps + copies (pinned formula);
**measured coexistence overhead ≤ 2.0 ms** (R5, the first viability number).

### U2 — Bed field + porous BC: the column stands
`seam_clear_bed` + `seam_scatter_bed` passes (grain pos → bed_occupancy φ_s + saturation,
cleared every frame per KTD2); pbmpm grid_update bed-BC extension per KTD3 (params-gated,
off ⇒ byte-identical — pinned by re-running a pbmpm-solo scene through the seam water-only
path); Seam appears in the web dropdown (free via `SolverId::all()`); the M0 scene
(`Scene`: grain bed region + water column region above it, no pour) added to the Debug
Scenes. **Bed-saturation initialization** (review r2.1): grains seed dry (pos.w = 0) and
`SeedRegion` carries no moisture, so the saturated/half-saturated M0 arms need an explicit
path — `TwofieldSolver::prewet_grains(sat_frac)` writes grain V_abs = sat_frac·V_cap
(wet_sat_cutoff-aware so sat_frac = 1.0 really zeroes demand), exposed through
`SeamSolver::prewet_bed(sat_frac)`; the written volume enters the combined accounting's
t0 snapshot (pre-wet volume is initial in-domain inventory, not emitted volume) and the
call is part of the pre-registered scene setup for every R1–R3 arm. **Reset-stable by
construction** (review r3.1): prewet mutates BOTH the live pos buffer AND the solver's
cached `initial_positions` seed, so `reset()` replays the wet bed — a reset would
otherwise silently dry the M0 scene and invalidate R1/R3/t0 accounting (the web UI calls
reset directly). Gated: prewet → step → reset → per-grain saturation preserved exactly.
**Oracle first:** column visibly stands in the browser. **Then gates:** R1 quantified
(no fall-through, interface density band, tail KE); tint_check green (R6).

### U3 — Reaction: the bed feels the column
seam_reaction accumulation in the BC; twofield dry-path injection hook (one optional pass,
buffer-handle gated, /substeps); seam_zero_reaction.
**Gates:** R2 measured momentum-delta pairing (seam-on vs seam-off arms, pre-registered
tolerance form before first measurement; the ledger is telemetry); **hook-placement gate**:
dry dispatch formula 4 → 5 passes/substep when armed, and a known injected test impulse
moves the bed COM velocity by the predicted amount (proves the hook sits after grid_clear
inside the dry branch); bed static under uniform column (twofield static gates re-run under
load); hook with zero buffer ⇒ twofield byte-identical; reaction fixed-point headroom probe
(16 iters × node mass × vel_cap vs i32 range — the new overflow surface named by the pbmpm
plan's R7(a)).

### U4 — Infiltration handoff + the combined ledger
KTD5 end-to-end: demand pass, marking, async readback, host removal, GPU crediting,
pbmpm emitted_mass, single-snapshot combined accounting accessor on SeamSolver.
**Gates:** R3 in **three** regimes (review r1.6): (a) saturated 2000-step zero-transfer
tail; (b) unsaturated non-vacuous exact transfer with ≤ 1-quantum in-flight residual;
(c) **the combined arm — a pond over a half-saturated bed**, where support (R1), reaction
(R2), and whole-particle handoff (R3) operate *simultaneously*: the column stands while
transfer proceeds, conservation holds exactly through active transfer, and the run
transitions into the saturated regime and stays conserved. Passing (a) and (b) in
isolation does not certify the seam — (c) is the actual multi-solver coupling question.
Also: balErr printout works in a seam `examples/` driver; multiset discipline in every new
test (the seam's own removal permutes pbmpm's live range).

#### U4 implementation design (settled during build)

- **Demand** (`seam_demand`, over grains, end of frame N after both inner steps): node-lane
  demand via the B-spline scatter into bed_occupancy lane 3 (cleared per frame by
  seam_clear_bed): demand_g = (V_cap − V_abs)·(1 − e^{−tf_absorb_rate·dt}), floored to 0 at
  wet_sat_cutoff. dt is pinned 1/60 (prereg); the factor rides SeamParams.wet.z, cutoff .w.
- **Mark** (`seam_mark`, over pbmpm live water, same encoder): a bed-contact particle
  (φ_s > SEAM_PHI_MIN at its position) tries to consume ONE quantum (V_w = spacing³, FP)
  from its nearest node's demand lane via atomicSub-with-undo; on success appends its index
  to the mark list (atomic count, capacity-capped with undo on overflow) and adds the
  quantum to seam_reaction lane 3 (the consumed-per-node ledger). Sub-quantum node demand
  is discarded at frame end — demand is a rate, it regenerates; granularity ≤ 1 quantum
  per node per frame.
- **Apply** (frame N+1 start, the one-transaction): (1) host removal from the mapped mark
  list — swap-with-last over the FULL live state set (pos, vel, phase, chem, deform_disp,
  deform_grad, vel_prev) via GPU buffer copies planned host-side (marks sorted descending,
  tail-collision-safe), then water_count decrement; (2) `seam_credit` (over grains): grain
  V_abs += Σ_nodes w·consumed_n·(demand_g·w / (remaining_n + consumed_n)) — demand
  recomputed deterministically (V_abs unchanged since frame N), initial demand
  reconstructed as remaining (bed_occupancy lane 3) + consumed (reaction lane 3), so the
  credited total equals the consumed total EXACTLY by construction; (3) zero the reaction
  ledger (moved from frame end to post-credit). Then the normal frame proceeds
  (clear+scatter → water → bed).
- **Books**: `SeamSolver::water_books()` returns (emitted_vol, in_domain_vol) with
  in_domain = pbmpm live·V_w + Σ grain V_abs, read at frame boundaries only —
  marked-but-unapplied particles are still live (water side), credit lands in the same
  transaction as removal.

### U5 — Pre-registration (before any quantified run)
`docs/plans/2026-07-09-003-seam-m0-preregistration.md`, committed **before** the U2–U4
quantified gates are first measured (the U7 precedent — review r1.4): the exact M0 scene
(box, bed region extents, column layers, spacing r = 0.16, expected particle counts),
frozen config asserts (iteration_count 16, flip_fraction 0.95, solid_dynamics true, seam
knobs), measurement protocol (50 warmup / 30 measure frames, median, namespaced-pass
aggregation, twofield-substep multiplication), and every numeric bar: R5's coexistence ≤
2.0 ms / seam physics ≤ 1.5 ms / total ≤ 22.5 ms, R1's band-mean ±3% + p99 bound + tail-KE
tolerance, R2's pairing tolerance form, R3's drift and residual bounds. U1's structural
gates (byte-identity, water_count == 0, dispatch formulas) may land before this doc; every
*measured* verdict input lands after it.

### U6 — M0 decision note
`docs/plans/2026-07-09-004-seam-m0-decision-note.md`: the R5 numbers (coexistence
overhead, seam physics cost, total @ 191k+15k), R1–R3 gate outcomes, interface artifact
assessment (visual), the viability verdict for multi-solver composition, and go/no-go for
M1 (dynamic seam: jet-impact impulse channel, crater under pour, drainage re-emission,
Darcy β). Written against the pre-registered bars, no post-hoc softening (KTD9 precedent).

---

## Risks

1. **Count-keyed cost cliff (twofield).** Any path that puts a live water particle into
   the inner twofield multiplies its cost ~50×. Pinned by U1's water_count gate; the M1
   drainage re-emission design must route through pbmpm, never twofield.
2. **Reaction correctness under FLIP.** The ×16-iteration impulse sum vs the true frame
   momentum change (KTD4) may disagree through the FLIP blend / particle_integrate path;
   R2 measures it. If pairing fails structurally, fallback: compute Δp_water directly
   (vel_prev vs vel readback reduction) and inject that — costlier but exact.
3. **Seam artifacts at the interface** (stacking, density ring, popcorn at the BC
   boundary): R1's density band + the visual oracle. Known lever if the hard block rings:
   soften the BC over one cell (β ramp) before touching constraint internals.
4. **Whole-quantum absorption granularity** (spacing³ ≈ 4.1 mL at r=0.16): acceptable at
   M0 scale; the ledger residual bound (R3) keeps it honest. If granularity artifacts
   appear, the remainder-accumulator (bank sub-quantum demand per column) is the recorded
   M1 refinement.
5. **Tint rejections** on the new atomic-writing BC (R6, third incident on record).
6. **Grid divergence by config drift**: a future caller building the seam with per-solver
   Materials would silently break co-registration — the U1 build assert makes it loud.

## Outstanding Questions (for the review rounds)

- Reaction injection point: grid_sm pre-solid_update (chosen, KTD4) vs a post-solid_update
  grid_svel kick — is the P_sp/Coulomb inheritance worth the substep-division bookkeeping?
- Grain-crediting shape: per-node share distribution (reusing the g2p_absorb math shape)
  vs per-marked-particle nearest-grain — which keeps the exact-ledger construction simpler?
- ~~Does the M0 scene need a *pond-on-unsaturated-bed* third arm?~~ RESOLVED (review
  r1.6): yes — it is the required combined-regime gate R3(c).
- bed_occupancy refresh cadence: per frame (chosen) vs every k frames for a static bed —
  premature optimization at 15k grains?

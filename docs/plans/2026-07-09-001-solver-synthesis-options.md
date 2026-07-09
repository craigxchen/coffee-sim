---
title: "Solver synthesis — three blend options for the unified pour-over solver"
status: PROPOSAL — awaiting owner choice of option (A / B / C)
parent: 2026-07-02-002-pbmpm-u7-decision-note.md (answers the halt-for-owner-ruling),
  2026-06-19-pbmpm-rearchitecture-requirements.md (candidates A–D)
provenance: 4 architect proposals (ship / novel / perf / unify lenses) + 12-judge adversarial
  panel, 2026-07-09; distilled to 3 genuinely distinct bets. Owner directive 2026-07-09:
  "no solver captures all the phenomena — synthesize what we've learned and blend."
---

# Solver synthesis — three blend options

## Shared ground (measured anchors, all Apple M5, release)

| fact | value | source |
|---|---|---|
| pbmpm single-phase water | **18.83 ms @ 190,874**, iter 16, gates green | `tests/solver_perf.rs` 2026-07-02 |
| pbmpm transfer family T / node passes G | 13.36 ms / 1.05 ms | same run |
| U7 pinned bounce | pbmpm crowns (R=0.00106, 30/60 guard-valid); twofield, every DensU K, xpbd: **zero ejecta** | `tests/solver_physics.rs` |
| twofield full coupled program | 25.4 ms @ 207k (7 solid CFL substeps internal; bubble_fine = 34.5% of frame) | `docs/PERF_NOTES.md` |
| xpbd water core | ~90 ms @ 200k extrapolated, levers unpulled; owns proven pairwise coupling + conserved wetting + extraction | `docs/PERF_NOTES.md` |
| corrected assembled floor | **25.2 ms @ 1 substep** (PASS, 7.8 ms headroom) / 50.4 @ 2 (FAIL) | U7 decision note §3 |
| candidate D (DensU proxy) | zero crown at every K∈{1..30} — the fallback ladder's first rung has direct evidence against it | U7 decision note §1 |

**Judge-panel corrections that bind on any option** (each was independently verified against
source by an adversarial judge; treat as constraints, not opinions):

1. **The frozen-skeleton drag gap.** Twofield's Laibe-Price fold has only ever run one-sided
   against a kinematically frozen solid (`coupling.wgsl:3` "frozen solid field (v_s = 0)";
   reaction to a `react` ledger). A live **mutual** two-field drag exchange with mass-weighted
   momentum pairing is *new physics with no in-repo proof* — it must be gated
   (`drag_fold_pair_momentum_exact_per_node`), not claimed as reuse.
2. **The water core does not stay 18.83 ms if you touch its loop.** φ_s/mixture reads inside
   the ×16 constraint iteration cost ~1.3–2.4 ms (trilinear gather in the hottest pass);
   any budget must carry an explicit in-loop growth line and a byte-identical off-switch
   (solid_count=0 reproduces today's water core bit-for-bit).
3. **The bed cannot be twofield's explicit elastic solid at 1 frame-substep.** Twofield runs
   solid_substeps=7 (sound-speed CFL); explicit elastic beds at frame-dt detonate. The bed
   must be XPBI-style position-based, or sub-iterate the solid family only (cheap at ~15k).
4. **Constraint-bubble cavity machinery is not portable as claimed.** λ_b is structurally
   welded to twofield's Jacobi pressure family (labels live in the pressure ping-pong
   buffers); and single-workgroup pocket passes measure ~156 µs/dispatch, not ~6 µs. In a
   density-constraint solver, first bet is *air = absence of particles* (constraint only acts
   where particles are); bubbles stay shelved as the pinned fallback if cavity popcorn appears.
5. **The dry-bed jet-impact channel needs an explicit mechanism.** On a dry bed, saturation-
   keyed drag ≈ 0 — the crater impulse must arrive via the mixture constraint's third-law-
   paired reaction (or a seam contact impulse in option B), and it needs its own gate
   (crater-from-jet on dry bed + momentum pairing), because nothing proven-in-repo covers it.

All four architect lenses (max-reuse, novel-core, perf-first, one-framework) independently
converged on the same skeleton for the deep blend — that convergence is itself evidence, and
the three options below are the three *genuinely distinct* bets, not four flavors of one.

---

## Option A — Deep blend: two-velocity porous PB-MPM on one grid

*SOTA candidate A assembled; the four-architect convergence with judge corrections baked in.*

**One-liner:** keep the measured pbmpm water pipeline as the single water phase; add a second
co-located solid momentum lane transferred **once per substep** (~15k bed particles, the
floor-structure correction); fold Laibe-Price mutual drag at the node; run Klar
Drucker-Prager XPBI-style inside the position-based loop; port conserved wetting
(`water_lost == grain_gained` + swelling→φ_s→K) and xpbd's two-pool extraction; cavity =
particle absence (bubbles shelved); U8 coarse pre-pass stays default OFF (it kills 75% of
the crown).

**Perf arithmetic** (caps, vs 33 ms gate): 18.83 water core + ≤1.0 in-loop mixture growth
(correction 2, gated) + 0.80 solid transfer pair + 0.52 drag fold + 2.67 bed constitutive
(0.2×T floor; fat — twofield runs this class at 15k inside 25.4) + 1.34 wetting + 0.52
pore-pressure/suction + 0.52 U8 slack ≈ **26.2 ms cap / ~22–24 expected @ 1 substep**.
2 substeps = 50.4 = automatic kill.

**Phenomena:** incompressibility + cup-ring (proven: the density constraint, −2.9%/−0.4%
interior), crown (proven, the only solver family that has one), stream integrity (proven +
new pinned gate), crater/bed (literature: Klar DP via XPBI at 1 substep — *not* proven at
frame-dt, correction 3), percolation (novel: mutual drag, correction 1), conservation
(ported ledger + gates re-certified on the new operator — the gates certified
operator+mechanism jointly, they do not transfer for free), extraction (proven on xpbd,
local pass, rebinding only), pore pressure for Bishop/Terzaghi = **explicit hydrostatic +
Green-Ampt suction first** (KD5's pre-named fallback); λ-as-pore-pressure runs only as a
gated diagnostic A/B.

**Milestones** (convergent across all four lenses):
- **M1 — skeleton holds the crown and the cost:** water + solid lanes + once-per-frame solid
  transfers + drag fold + renormalization clamp, no plasticity. Gates: assembled ≤ ~22.3 ms
  @ 200k+15k, 1 substep; U7 bounce re-run on the *assembled* arm R ≥ 0.5× solo with all
  guards; `drag_fold_pair_momentum_exact_per_node` green; byte-identical off-switch;
  interior density ±3% both pool depths; new pre-registered pour-stream gate.
- **M2 — bed + infiltration + wetting:** XPBI Klar DP at 1 substep (solid-only sub-iters
  allowed); crater absolute-pit PERSIST ≥ 0.8; dry-bed jet crater + third-law gate
  (correction 5); entry-flux-no-barrier + Darcy-scales-with-K; conservation incl.
  saturated tail, re-certified.
- **M3 — full pour-over** re-gated by the shared SolverId harness vs the pinned floors;
  extraction inventory conserved; buffer/pass ledger verified ≤ 9 granted (16 cap) in the
  browser, not just headless.

**Kill criteria:** any milestone needs M_sub ≥ 2 (arithmetic 50.4, no appeal); assembled
crown R < 0.0005 with guards holding and unrecoverable via flip_fraction/iteration levers;
perf gate missed > 15% after one bounded optimization pass; conservation gate fails at the
saturated tail. Kill → fallback = **option B** (not D: zero crown on record).

**Judge scores:** perf-lens variant 22/30 (highest), novel 21, unify 21, ship 19 — spread is
small because they are one architecture; the objections above are the real content.

**Why choose A:** only option where every required phenomenon has a mechanism on the same
grid with a coherent momentum ledger; the novel core (real-time two-velocity position-based
percolation) is the publishable contribution; perf path is arithmetic on measured floors.
**Why not:** three genuinely novel joints at once (mutual drag, in-loop mixture, XPBI at
frame-dt) — the highest compound research risk of the three.

---

## Option B — Seam blend: PB-MPM free water ↔ twofield porous bed

*Two gate-green solvers kept whole; all blend risk concentrated in one conservation seam.*

**One-liner:** pbmpm owns all free water (pour stream, splash/crown, pooling, cup) —
byte-identical, 18.83 ms; twofield owns the bed interior (saturation field, Darcy
percolation, swelling, crater plasticity, extraction state) minus its own free-water field;
a single interface layer at the bed surface converts between them: water particles that
infiltrate become twofield saturation (exact ledger, `water_lost == grain_gained`), drained
water at the filter face re-emits as pbmpm particles, and jet impact delivers a paired
contact impulse to the bed.

**Perf arithmetic:** 18.83 (untouched — no in-loop changes at all) + twofield bed-side subset
(bed constitutive + coupling minus water pressure family and minus bubble_fine; est. 2–4 ms
at 15k, *measure at M0 by running twofield with its water field disabled*) + seam pass
(~0.3–0.6 ms, one masked particle pass + emission) ≈ **22–24 ms expected**. No 2-substep
exposure on the water side; the bed keeps its own internal substepping (it already pays it
inside 25.4).

**Phenomena:** everything each solver already passes stays proven *in its own domain* —
crown/incompressibility/stream (pbmpm), crater persistence/Terzaghi/repose/conservation/
extraction (twofield, all gates green today). The seam owns: infiltration handoff
(ledger-exact, absorbing particles into a field), buoyancy/seating of grains under standing
water (needs a paired seam impulse — grains and free water never share a momentum ledger),
dry-bed crater impulse (seam contact force), drainage re-emission (position/velocity
reconstruction at the filter face).

**Milestones:** M0 — static seam: standing column over saturated bed, mass/volume exact
across the handoff incl. saturated tail, no ring/cram at the interface. M1 — dynamic seam:
jet impact craters a dry bed via the paired impulse (third-law gate); grains seat under a
column (no-over-buoyancy gate); drained re-emission conserves mass + momentum plausibility.
M2 — full pour-over on the shared harness.

**Kill criteria:** seam double-counting or conservation failure unresolvable at M0; visible
seam artifacts (pressure discontinuity, particle stacking at the interface, popcorn at
re-emission) that survive one bounded fix pass at M1.

**Why choose B:** fastest credible route to a full pour-over that both crowns *and* holds a
crater — nothing proven is rebuilt, the water core is literally untouched, and it can serve
as **M0 scaffolding for option A** (the seam version degenerates into the two-lane version
by moving the exchange from the surface into every cell). **Why not:** the interesting
physics (buoyancy, Terzaghi at the surface, crater impulse, suspended fines leaving the bed)
lives *exactly at the seam*; a surface-coupled model is physically cruder than a mixture
model and the owner-reviewer will probe the seam first. Not publishable as a core.

---

## Option C — Particle-deep: XPBD coupling core + grid density assist

*Candidate B modernized with the PB-MPM lesson; replaces dead candidate D as
fallback-of-record.*

**One-liner:** keep xpbd — the solver that already owns proven pairwise λ-impulse coupling,
conserved wetting, extraction, and lively water — as the physics core, and attack its one
disqualifier (~90 ms @ 200k) by replacing the O(n·k) PBF neighbor loops for *interior* bulk
water with a pbmpm-style grid density constraint (P2G → grid constraint → G2P), keeping
particle-pair constraints only near interfaces/surfaces where they earn their cost.

**Perf arithmetic (speculative, that is the point of M1):** neighbor iteration dominates
xpbd's frame; if the grid path absorbs the bulk-water majority at pbmpm per-particle rates
(18.8 ms / 191k all-grid is the existence proof), a ≥3× claw-back is plausible → ~30 ms;
**unmeasured**. M1 is a measurement gate, not a build gate.

**Phenomena:** coupling/conservation/extraction (proven, xpbd's own gates); bounce —
existence-proofed on xpbd historically but **zero ejecta on the pinned U7 config** (thin
nozzle caveat recorded in the decision note): must re-run at native nozzle *before* any
build; hybrid grid/particle constraint consistency at the boundary between regimes is the
novel joint (speculative).

**Milestones:** M1 — hybrid density constraint on the dam scene: ≥2.5× frame-time claw-back
measured, density band held across the grid/particle boundary, native-nozzle bounce
re-measured with the U7 harness. M2 — coupling re-validation (the xpbd gates re-run with the
hybrid core). M3 — assembled pour-over.

**Kill criteria:** claw-back < 2.5× at M1 after one optimization pass; native-nozzle bounce
fails the U7 guards; density discontinuity at the hybrid boundary that survives one fix pass.

**Why choose C:** the most physics-conservative option — the coupling machinery the project
trusts most stays primary, and the blend imports only pbmpm's *cheapness*, not its physics.
**Why not:** the perf mountain is real and front-loaded; even the optimistic arithmetic
lands at ~30 ms with no headroom; two novel joints (hybrid constraint seam, perf) stand
between M1 and any coffee physics.

---

## Recommendation

**A as primary, B as the de-risking on-ramp, C as fallback-of-record.** Concretely: build
option B's M0 seam first (1–2 weeks of work, every piece exists) — it produces the first
full pour-over that crowns *and* craters, and its conservation seam, shared harness arms,
and interface gates are exactly the scaffolding option A's M1 needs. Then attempt A's M1
(the two-lane skeleton) with B as the standing fallback. D is removed from the ladder on
the U7 evidence (zero crown at every K); C replaces it.

Choosing A or B **is** the owner ruling the U7 halt asked for: it adopts the structurally-
corrected floor (25.2 ms @ 1 substep) with the decision note's conditions — U8 built
(satisfied, default OFF) and the 1-substep hold re-gated at the first assembled milestone.
This document records that ruling once an option is chosen.

Full architect proposals + all 12 judge verdicts: workflow artifacts
`wf_cb236243-9ca` (session 2026-07-09); the four proposals agree on M1's gate set
verbatim, which is why M1 above is quoted as convergent.

---
type: fix
status: superseded
date: 2026-06-17
depth: deep
origin: docs/brainstorms/2026-06-17-twofield-sdf-wall-embedded-boundary-requirements.md
review: codex-r4-approved
superseded_by: docs/brainstorms/2026-06-17-twofield-cup-corner-seam-bc-requirements.md
superseded_reason: >
  U2a audit (tests/twofield_wall_audit.rs) overturned this plan's premise: the over-pack is
  not a ~25× mid-wall phenomenon fixed by a cut-cell coverage coefficient. The aligned-square
  side-wall M̃⁻¹ exactly matches the box (Δ=0); the "25×" was a tail-sensitive nearest-neighbor
  metric artifact (robust over-pack ~1.8×). The real, visible defect is corner packing at the
  concave floor∩wall SEAM (2.4× vs box 1.4×) — a single-normal multi-face-incidence BC bug.
  Re-brainstormed → the corner-seam requirements doc above. U1 (the dbg.y selector, commit
  3980eb5) stays valid.
---

# Two-field SDF wall: embedded-boundary pressure BC — plan

> Accepted at Codex review round 4 (gpt-5.5, high — APPROVE-WITH-NITS, all NITs folded).
> Revised through rounds 1–3. R3 added the node-metric
> representability constraint (the node `M̃⁻¹` is one symmetric 3×3 shared by 8 cells, not a
> per-incidence aperture table — so the EB coefficient family is bounded by what it can
> express, proven in U2a), split the design into a spec-gate (U2a) before implementation (U2b),
> and added the 2D corner/seam case (a single SDF dyad can't match the box's two-axis corner
> zeroing). R2 corrected a real design error:
> the principled fix is **not** a fractional cell *volume* folded into `f` (that conflates
> geometric coverage / free-surface fill / relief density, and has a sign trap — shrinking the
> fluid volume raises `rho`, saturates `f`, and *increases* relief, making cut cells look
> denser or hiding over-pack). The correct missing quantity is a shared **geometric
> embedded-boundary coefficient (Gibou θ / Batty aperture) in the pressure stencil**, reused
> identically by `D` and `G`, reducing to the box-face coefficients for an aligned planar wall.
> Crucially, that coefficient is **defined by an aligned-square operator audit** (U2), not
> invented up front — the plan is now audit-driven, matching how this bug was diagnosed.

## Problem Frame

Water in the V60 cup over-packs into a dense wall shell — measured (particle-side
neighbor-count ρ/ρ_rest, 1.0 = incompressible) at ~25× at the wall, ~35× under a pour. A
multi-experiment re-diagnosis isolated the cause to the **SDF-solid wall code path**:

- A cup-sized **flat domain box** settles to **0.98×** — the fluid is incompressible enough.
- A **statically seeded full cup** (no pour) collapses to 16.7× / wall 25.6× on its own.
- A geometrically identical **square `PolyCup`** over-packs to 18× vs the box's 0.98× — they
  differ *only in code path*.
- A **node-aligned square `PolyCup` still measures ~2.9× at the wall** — so the over-pack is
  not only sub-node placement; an aligned SDF wall still differs from a box face. **That
  difference is concrete and enumerable in code** and must be traced before any general
  mechanism is designed (see D-AUDIT / U2).

The domain box uses an exact box-**FACE** BC: `node_setup` zeroes whole axes (`pressure.wgsl`
~308-315), `drag_fold` zeroes the normal component and zeroes outside-box nodes
(`coupling.wgsl` ~239-256), and `cell_classify` marks box-face corners "wallish" via
`xp <= box_min+eps` so they are excluded from the density count. Every SDF solid instead uses
a binary `WALL_BAND·h` (1-cell) wall-normal `n̂n̂ᵀ` projection over a fluid *shell* in
`node_setup` + `drag_fold`, and a *different* wallish test in `cell_classify`
(`solid_union(xp).dist < 0.0`, ~393-397). The fix makes the SDF wall coefficient
geometrically correct — reducing to the box-face behavior when aligned — without the
over-constraining binary band. (See origin: `docs/brainstorms/2026-06-17-twofield-sdf-wall-embedded-boundary-requirements.md`.)

## Scope Boundaries

- **Not** a compressibility / density-constraint rewrite (shelved
  `docs/plans/2026-06-16-001-...`; flip to `superseded` when this lands).
- **Not** a fractional-cell-*volume*-into-`f` mechanism (R2: wrong quantity + sign trap).
- **Not** pour-specific, curvature-specific, a snap-to-nodes hack, or Rhie-Chow/MAC staggering.
- **Not** a change to `FLOOD_WALL_BAND` (separate pocket-detection flood band).

## Key Technical Decisions

### D-AUDIT. The aligned-square residual is localized by an operator audit *before* the mechanism is chosen

The ~2.9× aligned-square over-pack proves an aligned SDF wall is not yet equivalent to a box
face. Before designing a coefficient, U2 **enumerates the exact differences** between an
aligned square `PolyCup` node/cell and the geometrically identical domain box, item by item:
the node M̃⁻¹ matrix, the wallish-corner census (`dist < 0` vs `<= box_min+eps`), the
active-cell mask, `f`, `rho`, and the RHS — coefficient-by-coefficient. The fix is defined by
what this audit finds (likely starting from the wallish-classification mismatch at SDF nodes
with `dist ≈ 0`), not by a pre-invented general mechanism. This mirrors the diagnosis method
that produced this plan.

### D1. The fix is a shared geometric embedded-boundary coefficient in the stencil — two forms, naive then geometric

- **L1 — soft scalar coverage weight `w(dist,h) ∈ [0,1]` (Approach C, the cheap first form).**
  `M̃⁻¹ = invρ·(P_box − w·n̂n̂ᵀ)` in `node_setup` and `v -= w·dot(v,n̂)·n̂` in `drag_fold`, same
  `w`. Honestly a *soft normal penalty*: for an off-node wall a graded `w` leaves even the
  nearest fluid node at `w < 1`, so it is not a hard cut-cell boundary. Cheapest lever; stood
  up first behind a selector to prove inertness at `w≡1` and to *characterize* the baseline —
  **not** to size the real fix.
- **L2 — the geometrically-correct embedded-boundary coefficient (defined by U2's audit).** A
  shared geometric measure (Gibou-style wall-distance θ / Batty-style face aperture) on the
  pressure-stencil incidence, applied **identically to `D` and `G`** (so `A_ij = A_ji`), and
  **reducing to the box-face coefficients for an aligned planar wall** (the property L1 lacks,
  per the aligned 2.9×). It is **anisotropic where needed** (the box-face path zeroes specific
  axes; an isotropic scalar cannot reproduce that) — the natural home is therefore the
  per-node M̃⁻¹ (`nm`), already a symmetric 3×3 that carries the wall and that `project`
  already reads, rather than a new cell field. L2 may *subsume* L1 (it is the correct form of
  the same M̃⁻¹ wall coefficient); whether L1's form already reaches target or L2 is required
  is decided by U2's audit + the 1D test below, not assumed.
  - **Representability is a hard constraint, not an assumption.** The node metric is *one*
    symmetric 3×3 per node — a 6-DOF node-local quadratic form contributing `sᵢᵀ·Mₙ·sⱼ` shared
    by all 8 adjacent pressure cells (`pressure.wgsl` ~641-696, ~830-861), **not** an arbitrary
    per-cell-cell-link (per-incidence) aperture table. It can reproduce the box's aligned
    per-axis zeroing and node-local anisotropy — including a **multi-axis** projector at a
    corner *iff the construction detects multi-face incidence* — but it **cannot** express a
    general per-incidence θ field around arbitrary curved geometry. So the L2 coefficient
    family is *constrained to what the node metric can express*: U2a must prove the target
    per-node metric is exactly `S·Mₙ·Sᵀ` under the existing `corner_sign` basis, or stop and
    either restrict L2 to a node-metric EB family (accepting its geometry limits) or declare
    the no-new-buffer path impossible (R-6).
  - **Corners/seams are the binding case.** A single `solid_union` normal yields one `n̂n̂ᵀ`
    dyad, leaving the seam-tangent component free; the box zeroes *both* normal axes at a
    corner. The EB construction must therefore **detect multi-face / multi-axis incidence at
    seams** and build the corresponding multi-axis node metric — validated by a 2D corner test
    (U2b/D3), not only the planar 1D case.
- **Approach A (full Batty variational face fractions) is rejected as a grid mismatch** —
  staggered MAC face velocities vs our collocated grid. L2 takes Batty/Gibou's *coefficient
  idea* onto the collocated M̃⁻¹/stencil without face velocities.
- **Do NOT overload `f` or scale the fluid mass to encode coverage.** Keep three separate
  quantities: geometric wall coverage/θ (new, for the stencil coefficient), the free-surface
  fill fraction `f` (existing, from mass), and the relief density `rho/ρ_rest` (existing). The
  sign of any density/mass change is proven in a **1D planar-wall hydrostatic cut-cell test**
  (U2) before it touches `cell_classify` — shrinking the effective volume raises `rho` and
  `f` and *increases* relief, which is the wrong direction (Codex R2 BLOCKER 2).

### D2. Velocity BC and M̃⁻¹ change in lockstep; consistency is proven by a coefficient audit, not only SPD

Operator consistency `A = D·M̃⁻¹·G` requires `drag_fold`'s pre-projection field and
`node_setup`'s M̃⁻¹ to constrain the same axes at the same strength; the coefficient (L1 `w`
or L2 θ) is applied identically in both. Because `A = D·M̃⁻¹·Dᵀ` is symmetric-PSD for any
symmetric-PSD M̃⁻¹, the existing adjoint/SPD twins **pass any symmetric coefficient, including a
wrong one** — they read back the same `nm`/`cell_meta` on CPU and GPU. So correctness needs a
**coefficient audit** (D3), not just SPD.

### D3. Physical gates: deterministic, normalized, measured before push-out

Algebraic SPD is necessary-not-sufficient (`P_w = I − w·n̂n̂ᵀ` is not a projector for `0<w<1`).
Add gates the algebra can't provide, all **normalized and deterministic**, compared against
the aligned-box reference. **Sampling timing**: the wall-normal flux and inward-displacement
quantities are sampled **after `project` but before `g2p_water`** (the pressure-side result);
the push-out metrics are necessarily instrumented **in/after `g2p_water`** (they only exist
once push-out runs) and report what the pressure solve failed to handle:

- **Coefficient audit (planar)**: assembled operator rows for an aligned square `PolyCup`
  match the box-face operator *away from seams* coefficient-by-coefficient; a planar offset
  wall matches the chosen θ/coverage formula coefficient-by-coefficient.
- **Coefficient audit (seam/corner)** — separate expected-result section, because corners are
  where SDF normals and the box's multi-axis constraints diverge: at an aligned square corner
  the assembled node metric must match the box's two-axis zeroing (the multi-face-incidence
  construction, D1), and the corner's wall-normal flux / inward displacement must match the
  box reference — not just the planar rows.
- **Wall-normal flux**: mass-weighted inward SDF-normal velocity over wall-adjacent nodes, and
  net signed inward flux, bounded vs the box reference (a soft wall that leaks broadly fails).
- **Inward normal displacement**: p95/p99 per step bounded.
- **Push-out activity** (D6): count, p99/max penetration depth, and normal momentum removed as
  a **fraction of the pressure impulse** — must *drop* vs HEAD.
- Gated **together with** the radial-fill (`settled_cup_water_no_edge_ring`) and
  volume-density gates, so a hollow collapse that avoids the wall cannot pass.

### D4. Co-calibrate L1+L2 on a WIDE matrix — L1 isolation is characterization only

L1 and L2 couple nonlinearly (`diag ~ f_c²`, RHS `~ f_c`, density relief, push-out), so
L1-only numbers are **not decision-quality** for sizing L2 (Codex R2 MAJOR 5). U1 records a
baseline + proves inertness; U2a specs + U2b implements the geometric coefficient; **U3
co-calibrates L1+L2 together** across: offsets `{0, 0.25h, 0.5h}`, square, **rotated** square,
octagon, round, poured cup, the **cone dripper** (anti-overfit), and a water-only floor sweep.
Prefer a geometric coverage with a **near-wall plateau** over a naive linear ramp `1−dist/h`
(the linear form ties wall stiffness to sub-cell phase and can *shift* the shell, not remove
it). Thresholds pinned only after the matrix + visual oracle clear (`feedback_no_premature_tests`).

### D5. Storage: the coefficient lives where `project` already reads — no new buffer, no `solids` in `project`

`project` is at the 7-buffer ceiling and **does not bind `solids`**, so a per-incidence
coefficient cannot be recomputed there; and `cell_meta` has **no free lane** (`.x` rhs, `.y`
f, `.z` rho, `.w` cat). Therefore the coefficient is precomputed in a pass that already binds
`solids` (`node_setup` / `cell_classify`) and stored where `project` already reads — the
**per-node M̃⁻¹ (`nm`)** is the natural carrier (anisotropic, wall-aware, already consumed by
`project`'s `minv_apply`). U2a must **prove** the node metric can express the target (the
`S·Mₙ·Sᵀ` representability check, D1) or pick the documented fallback — it does not assume
sufficiency. R6
(R9 ≤ 33 ms @ 200k) verified in the perf gate.

### D6. Push-out is a backstop that can MASK a bad BC — instrument and gate it

`g2p_water` push-out (`transfers.wgsl` ~224-233) confines at particle resolution but can hide
a bad grid BC by repeatedly clipping particles (injecting/removing momentum outside the
operator). Confinement is decoupled from relief, **but a healthy fix must reduce push-out
activity** — gated per D3, not leaned on.

### D7. Floor is gravity-loaded, not "just another wall"

The band was added because `< 0` left the supporting floor node y-free and pancaked the
column to 10–20× (`pressure.wgsl` ~317-327). Weakening the *vertical* coefficient by distance
is riskier than the side wall. Calibration includes an explicit water-only SDF **floor offset
sweep** `{0,0.25h,0.5h}` and may need a normal-direction-aware coefficient (faster `→1` /
plateau for downward normals), verified by floor density + push-out diagnostics, not only
bed/Terzaghi. **U2a/U3 must record** whether the floor (a single axis-aligned vertical normal)
is covered by the same `S·Mₙ·Sᵀ` representability proof as the side wall, or is a documented
special case with its own coefficient — it must not be silently assumed identical.

### D8. Bring up behind the selector; faithful CPU twin; fix the `dbg.y` default

Land L1 behind the `Params.dbg.y` selector, default to binary, prove operator gates
**bitwise-identical at `w≡1`**, then add L2, co-calibrate, flip. The selector requires
updating **both** the WGSL and the Rust default/comment — `dbg.y` is currently initialized as
`RELIEF_DEADBAND` and documented as reserved/dropped (`common.wgsl` ~52-55), so the modes must
be made deterministic and not confused with the dead-band. The CPU operator twin
(`tests/twofield_pressure.rs`, `twin_from_gpu`/`constrain_sym` ~1783-1808) must **mirror the
final L1+L2 coefficient**, or explicitly declare it does not cover the SDF path (so a green
twin can't give false assurance).

## Requirements Traceability

| Req (origin) | Addressed by |
|---|---|
| R1 — full cup holds wall ρ/ρ_rest near rest (< 2×, round + square) | U3 co-calibration + volume-density gate |
| R2 — node-aligned wall reduces to box-face (~0.98× square) | **D-AUDIT/U2** (trace the residual) + D3 coefficient audit; L2 designed to reduce to box-face — verified, not claimed |
| R3 — confinement AND radial distribution preserved | D6 + D3 (radial-fill gated with volume-density) |
| R4 — operator consistency `A = D·M̃⁻¹·G` | D2 (SPD by construction) **+ D3 coefficient audit** (physical) |
| R5 — all L0–L3 gates green; floor/bed support preserved | D7 floor sweep + U4 regression |
| R6 — R9 ≤ 33 ms; fixed-point i32, buffer ceiling, Tint | D5 (no new buffer; M̃⁻¹ carrier) + U4 perf gate |
| R7 — volume-density gate fails on HEAD (~25×), passes after | U3 (probe → calibrate → pin) |
| R8 — `PolyCup` + `v60_cup_static_full` harness | U1/U3 + widened matrix (D4) |

## Implementation Units

### U1 — L1 soft-penalty BC behind selector + diagnostics + inertness proof + baseline characterization

- **Goal**: Apply `w(dist,h)` to the M̃⁻¹ dyad and velocity BC (same `w`) behind the
  `Params.dbg.y` selector (default binary). Stand up the D3 diagnostics. Prove inertness at
  `w≡1`. Record a baseline (characterization only — not L2 sizing data).
- **Files**: `src/solvers/twofield/common.wgsl` (`wall_coverage(dist,h)`),
  `src/solvers/twofield/pressure.wgsl` (`node_setup` ~328-342, subtract `w·invc·n̂n̂ᵀ`),
  `src/solvers/twofield/coupling.wgsl` (`drag_fold` ~266-279, `v -= w·dot(v,n̂)·n̂`),
  `src/solvers/twofield/mod.rs` (selector via `Params.dbg.y` — update Rust default + comment,
  D8). Diagnostic probe harness (not yet gates): post-`project` wall-normal flux / inward
  displacement / push-out count·depth·momentum (D3, D6).
- **Execution note**: Characterization-first. Prove bitwise-identical operator-gate output at
  `w≡1` before any calibration. No pinned threshold here.
- **Test scenarios**: `w≡1` → `gpu_operators_match_twins_and_are_adjoint`,
  `gpu_assembled_a_symmetric_and_positive`, `cpu_assembled_a_spd_and_row_sums`,
  `divergence_decay_static_tank`, `divergence_decay_dam_break_midsplash` bitwise/tolerance
  identical to HEAD; coverage `w` → same gates green (PSD by construction).
- **Verification**: gates green both modes; `w≡1` matches HEAD; diagnostics emit baseline
  numbers; `fmt`/`clippy` clean.

### U2a — Audit + coefficient/storage spec gate (no shader change)

- **Goal**: Produce a reviewable **spec** before any solver edit: localize the ~2.9× aligned
  residual, define the L2 coefficient formula, prove it is representable in the node metric,
  and fix the storage. This is a gate — U2b does not start until U2a's deliverable passes.
- **Deliverable** (all required before implementation commits):
  1. **Residual localization** — the aligned square `PolyCup` node/cell enumerated vs the box
     (node metric matrix, wallish census `dist<0` vs `<=box_min+eps`, active mask, `f`, `rho`,
     RHS), naming the exact code difference(s) that cause 2.9× (incl. corner/seam rows).
  2. **Coefficient formula** — the θ/aperture/coverage family, the geometry class it covers,
     and how it reduces to box-face for an aligned planar wall AND an aligned corner.
  3. **Representability proof** — the target per-node metric is exactly `S·Mₙ·Sᵀ` under the
     `corner_sign` basis for the supported geometry class (planar + detected multi-face corner)
     (D1). State **explicitly what is inside vs outside the compared matrix** — `f`, `φ_f`, the
     `1/(4h)` gather scale, and the row taper factors must be on the same side for the GPU and
     the CPU twin, so the comparison isolates the wall metric `Mₙ` and not the surrounding
     scalars. If a needed target is *not* expressible, U2a must say so and pick the fallback:
     constrain L2 to the node-metric EB family (and document the geometry it cannot capture),
     or escalate the no-new-buffer constraint (R-6) for a decision.
  4. **Storage/encoding** — exact `nm` encoding (no new buffer; `project` reads `nm`, does not
     bind `solids`, `cell_meta` full — D5), and the **multi-face-incidence detection method**
     itself (how a seam node discovers it abuts >1 face when `solid_union` returns one nearest
     normal) — this is part of the spec, and is validated independently by U2b's 2D corner audit.
  5. **Pass/fail criteria** — for the U2b coefficient audit and the sign tests.
- **Files**: audit harness only — a test that dumps the above for the aligned square vs box
  (extend `tests/twofield_pressure.rs` twins / new `tests/twofield_wall_audit.rs`). No
  `node_setup`/`drag_fold`/`cell_classify` changes in U2a.
- **Approach**: D-AUDIT, D1 (L2 + representability + corners), D5. If the residual is purely
  the wallish-classification mismatch, the spec may reduce to a small classification
  reconciliation — let the evidence decide; do not pre-commit a general coefficient.
- **Verification**: the 5-item deliverable is complete and internally consistent; the
  representability result is explicit (expressible, or fallback chosen); reviewer can validate
  the mechanism from the spec without reading future code.

### U2b — Implement the spec'd coefficient + prove the sign (1D + 2D corner)

- **Goal**: Implement exactly what U2a specified and pin its correctness with sign oracles.
- **Files**:
  - Implement the coefficient in `node_setup` (node metric, with multi-face-incidence
    detection) + mirror in `drag_fold`; reconcile the SDF wallish test (`pressure.wgsl`
    ~393-397) with the box-face one per U2a's localization.
  - **1D planar-wall** hydrostatic cut-cell test — pins the sign/value of any density/coverage
    term *before* it touches `cell_classify` (D1 sign trap).
  - **2D corner/seam** cut-cell test — an aligned square corner must match the box operator and
    the box wall-normal-flux / inward-displacement behavior (D1 corners, D3 seam audit); a
    single dyad must NOT leave the seam-tangent free.
- **Approach**: implement to spec; keep coverage/θ separate from `f` and relief density (D1).
- **Execution note**: prove both sign tests before broad calibration; re-run the operator twins
  (must mirror the coefficient, D8).
- **Test scenarios**: 1D hydrostatic sign/value; 2D corner matches box; operator
  adjoint/SPD/divergence-decay + MMS order + volume conservation green with the coefficient
  active; coefficient audit (planar + seam, D3) green for the aligned square.
- **Verification**: coefficient reduces to box-face for the aligned planar wall AND corner; 1D
  + 2D sign proven; operator + conservation gates green.

### U3 — Co-calibrate L1+L2 on the wide matrix, pin volume-density + physical wall gates, flip default

- **Goal**: Co-calibrate the coefficient so the confined cup holds wall ρ/ρ_rest near rest
  (square → ~0.98×, round < 2×) across the wide matrix; confirm visually; pin gates; flip
  default to coverage.
- **Files**: `tests/twofield_cup_density.rs` (or extend `tests/twofield_settled.rs`) —
  neighbor-count ρ/ρ_rest (mean/p90/wall-shell) + the D3 physical wall gates;
  `common.wgsl`/`mod.rs` finalize coefficient + flip default. Add rotated-square + offset
  `PolyCup` variants and a water-only floor fixture.
- **Approach**: D4. Co-calibrate L1+L2 over offsets/rotations/octagon/round/poured/cone/floor.
  Confinement AND radial fill preserved (`settled_cup_water_no_edge_ring` < 2.5 while
  volume-density drops). Pin with margin.
- **Execution note**: Calibrate-then-pin; gates written **after** visual + probe confirmation
  and shown to **fail on HEAD** (record HEAD numbers). Order-invariant ρ reads
  (`reorder_stale_phase_artifact`); p90/p99 not raw-max.
- **Test scenarios**: volume-density gate fails on HEAD (~25×), passes after; box-face audit
  (R2) on aligned square; radial-distribution < 2.5; octagon/round/rotated/poured below target;
  D3 physical wall gates (normalized flux bounded, push-out activity < HEAD).
- **Verification**: new gates red on HEAD / green after; box-face audit passes in flat-wall
  region; visual oracle confirms ring gone before any gate pinned; default flipped.

### U4 — Full regression (L0–L3), floor sweep (R5/D7), perf (R6), retire binary path

- **Goal**: Keep every twofield gate green (esp. the gravity-loaded floor), meet perf, remove
  the binary band + selector.
- **Files**: `pressure.wgsl`/`coupling.wgsl`/`mod.rs` (delete binary branch + selector once
  green); CPU twin mirrors final coefficient (D8); finalize `dbg.y` comments/defaults.
- **Approach**: Run L0–L3 + explicit floor offset sweep `{0,0.25h,0.5h}` (density + push-out,
  not just bed/Terzaghi).
- **Execution note**: regression + cleanup; remove binary only after green everywhere; never
  loosen a gate (`AGENTS.md`); manual in-browser Tint compile check.
- **Test scenarios** (existing gates green): L0 `tests/twofield_pressure.rs`; L1
  `tests/twofield_bed.rs` (`settled_heap_static_rest_no_creep`, `over_packing_rejected_at_phi_max`);
  L1–L2 `tests/twofield_coupling.rs` (extended face-velocity twin, buoyancy, Darcy, slip-decay,
  `settled_saturated_column_stays_settled_long_run`); L2–L3 `tests/twofield_full.rs`
  (Terzaghi+Skempton, no-fluidize, crater persist + slump-without-blend, volume conservation,
  determinism); floor sweep; perf `tests/twofield_perf.rs` R9 ≤ 33 ms @ 200k.
- **Verification**: L0–L3 + perf green; floor holds at all offsets; binary path removed; one
  BC; `clippy` clean; flip `docs/plans/2026-06-16-001-...` `status` → `superseded`.

## Risks & Dependencies

- **R-1: the aligned-square residual may be a small classification mismatch, not a general
  coefficient.** Mitigated by audit-first (U2) — the fix is sized to the evidence, avoiding
  over-engineering.
- **R-2: sign trap in any density/coverage term.** Mitigated by the 1D planar-wall test before
  `cell_classify` changes (D1/U2).
- **R-3: floor pancaking (R5/D7).** Gravity-loaded vertical wall; explicit floor sweep +
  diagnostics + possible normal-aware coefficient.
- **R-4: soft wall masked by push-out (D6).** Push-out activity gate (must drop) + normalized
  wall-flux gate.
- **R-5: anisotropy.** An isotropic scalar can't reproduce the box-face axis-zeroing; L2 lives
  in the anisotropic M̃⁻¹ — verified by the coefficient audit (D3).
- **R-6: storage / representability.** `project` can't see `solids` and `cell_meta` is full;
  coefficient carried in the node metric `nm`. But `nm` is one symmetric 3×3 shared by 8 cells,
  not a per-incidence aperture table (D1) — so U2a must **prove** `S·Mₙ·Sᵀ` representability for
  the supported geometry class, and if it fails, explicitly choose: constrain L2 to the
  node-metric EB family (documenting the geometry it can't capture) or escalate for a
  new-buffer / alternative decision. The plan accepts that "no-new-buffer impossible" is a
  legitimate U2a outcome, not a failure to hide.
- **R-7: Tint uniformity** — coefficient is a uniform scalar from `solid_union`; manual
  in-browser compile check (U4).
- **Dependency**: `g2p_water` push-out unchanged (backstop, D6). Fixtures `SolidKind::PolyCup`,
  `Scene::v60_cup_static_full` in tree; add rotated/offset `PolyCup` + water-only floor fixtures.

## Deferred to Implementation

- The exact L2 coefficient formula (θ / aperture / coverage), its anisotropy, and whether L1's
  form already suffices — all from U2's audit + 1D test.
- Whether the floor needs a normal-direction-aware coefficient (D7 / R-3).
- The exact `nm` encoding of the coefficient (no new buffer; D5).
- Gate thresholds — from U3's calibration matrix.

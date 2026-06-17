---
type: fix
status: active
date: 2026-06-17
depth: standard
origin: docs/brainstorms/2026-06-17-twofield-cup-corner-seam-bc-requirements.md
review: codex-r3-approved
---

# fix: Two-field cup corner-seam multi-normal BC

## Summary

Water packs into the cup's concave **floor∩wall corner** (measured 2.4× ρ/ρ_rest vs the
domain box's 1.4× at the same corner) because the SDF wall boundary condition takes a **single**
nearest normal from `solid_union()` at the seam — so `node_setup`'s M̃⁻¹ and `drag_fold`'s
velocity BC constrain only **one** of the two surfaces meeting there. The fix constrains **all**
binding faces at a seam via an **orthonormalized constraint basis**: collect the cavity faces
within `WALL_BAND·h`, build an orthonormal basis `Q` from the active box-face axes plus the
residualized SDF normals (rank ≤ 3), and apply the **same** projector `P = I − QQᵀ` in both
`node_setup` (M̃⁻¹ = invρ·P) and `drag_fold` (v ← P·v). The orthonormal basis is the load-bearing
choice: a raw dyad sum `Σ n̂ᵢn̂ᵢᵀ` is **not** PSD for non-orthogonal normals (e.g. a poly vertical
edge), which would break `A = D·M̃⁻¹·G`; `I − QQᵀ` is a true orthogonal projector for any normals
and reduces exactly to the box's two-axis corner zeroing when the faces are axis-aligned.

(See origin: `docs/brainstorms/2026-06-17-twofield-cup-corner-seam-bc-requirements.md`.)

---

## Problem Frame

Confirmed this session (audit `tests/twofield_wall_audit.rs` + the live-render oracle):
- The **corner (floor∩wall)** is the worst locus — coarse ρ/ρ_rest **2.4×** vs mid-wall 1.8×,
  floor-center 1.2×, interior 1.3×; with push-out-stacked coincident pairs (min nearest-neighbor
  distance 0.0007).
- It is the **corner BC code path**, not geometry: an aligned **square** SDF cup corner packs to
  **2.40×** while a geometrically identical **flat domain box** corner is **1.40×** — same
  geometry, differing only in the BC. The box zeroes both axes at its corner; the SDF path takes
  one `solid_union` normal.
- `cyl_cavity` (`src/solvers/twofield/common.wgsl` ~245-262) and `poly_cavity` (~266-290) compute
  `d_side`/`d_floor` (and, for poly, every face projection) then return only the nearer; the cup
  cavity is an *intersection* (above-floor AND inside-wall), and the corner is where the side and
  floor distances are both small.

The visible coincident pairs are a **symptom** of the per-particle push-out (`g2p_water`); the
root is the grid BC under-constraint — the box uses the *same* push-out without stacking because
its grid BC constrains the corner upstream. So the fix is in the grid BC, not the push-out.

---

## Requirements Traceability

| Req (origin) | Addressed by |
|---|---|
| R1 — cup corner ρ/ρ_rest drops ~2.4× → ~box (round + square), coincident-pair stacking recovers | U3 calibration + corner-audit parity gate |
| R2 — axis-aligned corner reduces exactly to the box's two-axis zeroing | U2 (`P = I − QQᵀ` with axis normals) + operator twin; U3 square-corner-matches-box |
| R3 — single-face wall/floor support preserved away from seams | U1 (basis = one normal away from seams) + U4 floor check |
| R4 — operator consistency `A = D·M̃⁻¹·G` (SPD/adjoint + divergence-decay), incl. non-orthogonal faces | U2 (orthonormal basis → PSD; identical `P` both kernels) + non-orthogonal gates |
| R5 — all L0–L3 gates green | U4 regression |
| R6 — R9 ≤ 33 ms; fixed-point i32, buffer ceiling (no new buffer), Tint | U1/U2 (inline from `solids`; no barriers) + U4 perf |
| R7 — corner audit fails on HEAD (2.4×), passes after (parity with box) | U3 |
| R8 — `settled_cup_water_no_edge_ring` stays green | U3/U4 |

---

## Key Technical Decisions

### D1. Generalize `solid_union` to a binding-faces-within-band query

Add a query returning **every** cavity face within `WALL_BAND·h` of a point (unit normal +
signed distance), to a small fixed candidate cap (e.g. 6). Away from a seam exactly one face is
in-band → identical to today's single-normal behavior. `cyl_cavity` surfaces side+floor;
`poly_cavity` collects per-face distances deterministically and surfaces every in-band face
(vertical edges included); **the cone stays single-normal** (see D5). The CPU twin
(`src/utils/sdf.rs`) mirrors it.

The **rank cap (≤3) is applied by the basis builder (D2), not by pre-filtering raw faces.** A
"keep the 3 most-penetrated faces" pre-cap is wrong: a high-sided poly near the floor can have 3
nearly-coplanar *side* faces in-band that crowd out the *floor* face, leaving the vertical
direction unconstrained and regressing floor support. The query surfaces all in-band candidates;
D2 adds them by residual independence (tie-break by penetration) and only discards once rank 3 is
reached. **Collection discipline:** the candidate set must never drop the floor face before the
rank-build sees it — for the real cup this is trivially floor + one side (2 faces, well under the
cap); the candidate cap only matters for the high-sided poly stress fixture, where the cap is
raised/reserved so the floor is always a candidate.

### D2. Constrain via an orthonormalized basis `P = I − QQᵀ` — NOT a raw dyad sum

The constrained directions at a node are: the active box-face axes (the directions `node_setup`
zeroes today) **plus** each in-band SDF binding normal. Build an orthonormal basis `Q` by modified
Gram-Schmidt: seed `Q` with the active box axes, then add each SDF candidate normal **after**
residualizing it against the columns already in `Q`, **dropping any residual with `len ≤
SDF_NORMAL_MIN`** (so a candidate that adds no new direction is discarded, not double-counted).
**Rank-aware overflow:** process candidates by descending residual contribution (tie-break by
penetration depth) and stop once rank 3 is reached — so an independent floor normal is never
crowded out by near-coplanar side faces. The projector is `P = I − QQᵀ`.

- **Why orthonormal, not `Σ n̂ᵢn̂ᵢᵀ`:** a raw dyad sum is PSD only when the normals are mutually
  orthogonal. The cylinder floor (+y) ⟂ wall (radial) is orthogonal, but a `PolyCup` vertical edge
  has two side normals `2π/N` apart (octagon: `dot ≈ 0.707`), so `I − Σ n̂ᵢn̂ᵢᵀ` has a **negative
  eigenvalue** → M̃⁻¹ not PSD → `A` not SPD. `I − QQᵀ` is a true orthogonal projector onto the
  unconstrained subspace for **any** normals; orthonormalizing two non-orthogonal normals spans
  their plane and constrains both directions (the physically correct "pin both" at a sharp edge),
  and it is always PSD.
- **Reductions (verifiable):** box axes only → `P = P_box` (today's box-face BC); one SDF normal,
  no box axis → `P = I − n̂n̂ᵀ` (today's single-normal dyad); aligned floor+wall corner → `Q =
  {ê_y, radial}` → `P` zeroes both axes = the box's two-axis corner zeroing → matches the box.
- **Exact scalar (not raw 1/ρ):** the six matrix lanes are `M̃⁻¹ = sig · invr · (I − QQᵀ)`, where
  `invr` is the **existing wall-truncated, floored** inverse density (`1/max(rho, ρ_rest·vis/8)`,
  `pressure.wgsl` ~305-321) and `sig` (= ς) is folded exactly as today (~367-369). The two
  **non-matrix `nm` lanes are preserved unchanged** — `b.z` massy flag, `b.w = φ_f` (~371-372).
  This **unifies** the current separate box-axis zeroing and SDF dyad subtraction into one
  projector, so they cannot double-remove a shared direction or exceed rank 3.

### D3. Lockstep means the IDENTICAL projector in both kernels

`node_setup` and `drag_fold` must build the **same** `Q` — same in-band face set, same box axes,
same Gram-Schmidt order, same drop rule — and apply the **same** `P`: M̃⁻¹ = invρ·P in
`node_setup`, and `v ← P·v` (i.e. `v −= QQᵀv`) in `drag_fold`. NOT "same face set" with
`drag_fold` doing sequential per-normal projection — sequential projection ≠ `I − QQᵀ` for
non-orthogonal normals and would re-break `A = D·M̃⁻¹·G` (the exact defect a prior one-sided wall
BC hit and was reverted for). The shared basis builder is one function both call.

### D4. Repurpose the `dbg.y` selector; revert the U1 graded-coverage placeholder

U1 (commit 3980eb5) shipped `wall_coverage(dist,h)` (binary band vs graded ramp) behind `dbg.y` —
a placeholder for the now-superseded coverage approach. This plan **drops the graded ramp** and
repoints `dbg.y`: **single-normal** (default) vs **multi-normal-seam** (the `P = I − QQᵀ` path).

**Bitwise discipline:** `invr·(1 − n̂n̂ᵀ)` is *not* bit-identical to the current `invr − invr·n̂n̂ᵀ`
(different float rounding), so the single-normal default keeps the **existing branch literally
unchanged** during U2/U3 — the new projector is added as the separate `WALL_BC_MULTI` branch. That
makes the "operator gates bitwise-identical at the single-normal default" claim true *by
construction* (the default code is untouched). Only at U4, when the selector is retired and the
unified `P = I − QQᵀ` becomes the sole path, does the single-normal *case* flow through the
projector — and there it is operator-equivalent **to tolerance**, not bitwise; U4 re-runs the
operator gates at tolerance to confirm.

### D5. Cone stays single-normal; push-out unchanged; CPU twins mirror

`cone_cavity` is a single smooth converging surface (closest-point-to-segment, one gradient —
`common.wgsl` ~220-242 / CPU twin `src/utils/sdf.rs` ~260-277). Do **not** synthesize a second
cone normal from segment endpoints — the cone has no discrete face seam except a (currently
absent) physical apex cap, and the origin defers cone seams. The cone returns its single normal.
`g2p_water`'s single-normal push-out is unchanged (the particle-resolution backstop). The CPU SDF
twin and the operator CPU twin (`tests/twofield_pressure.rs`, `constrain_sym`) must mirror the
binding-faces query AND the orthonormal-basis projector, or the SPD/adjoint gates pass a wrong
operator.

### D6. Target is two-sided parity with the box, not a fixed number

The corner-audit gate asserts a **two-sided band**: `|SDF corner ρ/ρ_rest − box corner| ≤ margin`
(round + square), no residual coincident-pair tail, and **unchanged mid-wall/floor shells**. The
two-sided band matters because over-constraining could carve an artificial **void / under-density**
at the corner — a one-sided `≤ box + margin` would pass that. The box's own corner value is the
live reference (measured in the same run); the gate does not encode "1.4×" as a universal truth.

---

## High-Level Technical Design

Shared orthonormal-basis projector feeding both BC sites (directional, not implementation spec):

```
                 solids[] (already bound)
                        │
        wall_binding_faces(p, ph, h)  →  in-band faces {n̂ᵢ, dᵢ}, candidate cap ~6
          • away from seam → 1 face (== today)   • floor∩wall → 2 (⟂)
          • poly vertical edge → 2 (NON-orthogonal)
                        │
        build_constraint_basis():
          Q ← active box-face axes
          for SDF n̂ᵢ by ↓residual contribution: r = n̂ᵢ − Q(Qᵀn̂ᵢ); if |r|>eps: Q←[Q, r/|r|]
          stop at rank 3   ← rank-aware: independent normals kept, near-coplanar dropped
          P = I − QQᵀ        ← orthogonal projector, PSD for ANY normals
                        │
          ┌─────────────┴─────────────┐
   node_setup                  drag_fold
   M̃⁻¹ = sig·invr · P          v ← P · v   (v −= QQᵀv)
          └─────────────┬─────────────┘
              IDENTICAL Q and P  ⇒  A = D·M̃⁻¹·G holds (incl. non-orthogonal)
              aligned corner ⇒ P zeroes both axes ≡ box (parity)
              raw Σ n̂n̂ᵀ would be NON-PSD here ⇒ rejected
```

---

## Implementation Units

### U1. Binding-faces-within-band SDF query (WGSL + CPU twin)

- **Goal:** A query returning every cavity face within `WALL_BAND·h` (unit normal + signed
  distance), to a candidate cap (~6), generalizing `solid_union`'s single-min. `cyl_cavity`
  surfaces side+floor; `poly_cavity` collects per-face projections and surfaces every in-band
  face; `cone_cavity` returns its single normal (D5). **The rank-≤3 reduction is the basis
  builder's job (U2/D2), not a raw-penetration pre-cap.**
- **Requirements:** R1, R3, R6.
- **Dependencies:** none.
- **Files:**
  - Modify `src/solvers/twofield/common.wgsl` — `wall_binding_faces(p, ph, h)` (candidate cap ~6:
    normals + distances + count); refactor the cup cavity helpers to surface all in-band faces.
  - Modify `src/utils/sdf.rs` — mirror on the CPU twin.
  - Test: `src/utils/sdf.rs` unit tests (or extend `tests/twofield_wall_audit.rs`).
- **Approach:** D1, D5. A face binds iff its signed distance `< WALL_BAND·h`. Normalize each
  normal; the consumer (U2) orthonormalizes, so U1 just surfaces raw in-band normals + distances.
- **Patterns to follow:** `cyl_cavity`/`poly_cavity`/`cone_cavity`/`solid_union` in `common.wgsl`;
  CPU mirror in `src/utils/sdf.rs`.
- **Test scenarios:**
  - Happy: floor∩wall corner (cylinder) → 2 normals (floor `+y`, wall radial), orthogonal;
    mid-wall → 1 (radial); mid-floor → 1 (`+y`); deep interior → 0.
  - `Covers R3.` Away-from-seam parity: the single-face result equals the old `solid_union`
    normal (mid-wall/floor unchanged).
  - Edge: `PolyCup` `N=4` vertical edge → 2 side normals (orthogonal for square); `N=8` vertical
    edge → 2 side normals **non-orthogonal** (`dot ≈ 0.707`) — surfaced as distinct (the U2 basis
    handles them); triple corner (floor + 2 sides) → 3 faces; near the axis (r→0) the radial guard
    fires without NaN.
  - `Covers R3.` **Rank-aware floor retention:** a high-sided poly node near the floor with 3+
    near-coplanar side faces in-band still yields a basis that constrains the vertical (floor)
    direction (the floor normal is not crowded out) — verified after U2's basis build.
  - CPU twin agrees with WGSL on counts/normals at corner/mid/interior, **including near the
    `SDF_NORMAL_MIN` drop threshold** (a candidate just above vs just below the residual-drop
    cutoff resolves identically on CPU and GPU — f32 ordering/comparison mirrored).
- **Verification:** unit tests green; CPU↔WGSL agreement; `fmt`/`clippy` clean.

### U2. Orthonormal-basis projector in node_setup M̃⁻¹ + drag_fold (lockstep), behind `dbg.y`

- **Goal:** Both BC sites build the same orthonormal constraint basis `Q` (box axes +
  residualized SDF normals, rank ≤3) and apply the same `P = I − QQᵀ` — M̃⁻¹ = invρ·P in
  `node_setup`, `v ← P·v` in `drag_fold` — behind the repurposed `dbg.y` selector (single-normal
  default → inert).
- **Requirements:** R2, R4, R6.
- **Dependencies:** U1.
- **Files:**
  - Modify `src/solvers/twofield/common.wgsl` — `build_constraint_basis(box_axes, faces)` →
    orthonormal `Q` (modified Gram-Schmidt; add SDF candidates by descending residual contribution,
    drop residual `len ≤ SDF_NORMAL_MIN`, stop at rank 3 — D2 rank-aware) + `apply` helpers; shared
    by both kernels.
  - Add the `WALL_BC_MULTI` branch in `src/solvers/twofield/pressure.wgsl` `node_setup` (~308-345):
    `M̃⁻¹ = sig·invr·(I − QQᵀ)` for the six matrix lanes, **`nm` flag lanes (`b.z` massy, `b.w`
    φ_f) unchanged**. **Leave the existing single-normal branch literally unchanged** (D4 bitwise
    discipline) — gated by `dbg.y`. Drop the U1 graded ramp from `wall_coverage`.
  - Add the matching `WALL_BC_MULTI` branch in `src/solvers/twofield/coupling.wgsl` `drag_fold`
    (~252-282): `v ← (I − QQᵀ)·v` from the same `Q`; existing box-face + single-normal path
    unchanged under the default.
  - Modify `src/solvers/twofield/mod.rs` — repoint `dbg.y`: `WALL_BC_SINGLE` (default) vs
    `WALL_BC_MULTI`; rename `WALL_BC_BINARY/COVERAGE` + `set_wall_bc_mode_for_test` doc.
  - Modify `tests/twofield_pressure.rs` — operator CPU twin (`constrain_sym`/`twin_from_gpu`)
    mirrors the orthonormal-basis projector for the multi mode (and its f32 ordering / drop-threshold
    comparisons), not a single-dyad.
- **Approach:** D2, D3, D4. The single-normal default path is the **existing code, untouched** (so
  it is bitwise-identical by construction); the multi path builds the full rank-aware basis and
  applies the identical `P` in both kernels (D3).
- **Execution note:** Characterization-first — confirm the operator gates are **bitwise-identical**
  to HEAD with `dbg.y` at the single-normal default (the default branch is untouched) before
  exercising the multi branch. No pinned thresholds here.
- **Patterns to follow:** box-face axis-zeroing in `node_setup` (~308-315); the consistency
  comment in `coupling.wgsl` (~257-265).
- **Test scenarios:**
  - `Covers R4.` Single-normal mode: `gpu_operators_match_twins_and_are_adjoint`,
    `gpu_assembled_a_symmetric_and_positive`, `cpu_assembled_a_spd_and_row_sums`,
    `divergence_decay_static_tank`, `divergence_decay_dam_break_midsplash` — bitwise/tolerance
    identical to HEAD.
  - `Covers R4.` **Non-orthogonal PSD gate (new):** a CPU wedge node with two non-orthogonal
    normals AND a triple floor+wedge corner — assembled M̃⁻¹ is symmetric **PSD** (min eigenvalue
    ≥ −tol) and `A` is SPD; plus a direct per-node eigenvalue check of GPU `nm` on a `PolyCup`
    `sides=8` (the raw-dyad-sum bug would fail this).
  - `Covers R2.` Aligned-corner reduction: at an aligned square corner the assembled node metric
    equals the box's two-axis zeroing to tolerance.
  - `Covers R4.` Lockstep: `drag_fold`'s applied operator equals `node_setup`'s `P` on random
    velocities at a multi-face node (same `Q`), including a non-orthogonal wedge node.
  - Near-threshold determinism: a candidate normal whose residual sits just above vs just below
    `SDF_NORMAL_MIN` produces the same basis on CPU twin and GPU (no order/threshold divergence).
- **Verification:** operator + PSD gates green in both modes; single-normal matches HEAD; CPU twin
  mirrors; `fmt`/`clippy` clean.

### U3. Corner-audit parity gate + calibrate + flip default

- **Goal:** Promote the corner audit to a **two-sided** parity gate (`|cup corner − box corner| ≤
  margin`, round + square; coincident-pair tail gone; mid-wall/floor shells unchanged), flip
  `dbg.y` to multi.
- **Requirements:** R1, R2, R7, R8.
- **Dependencies:** U2.
- **Files:**
  - Modify `tests/twofield_wall_audit.rs` — assert `|SDF cup corner ρ/ρ_rest − box corner| ≤
    margin` (box measured in the same run, D6 — two-sided so an over-constrained **void** also
    fails, not just over-density); no coincident-pair tail (min nn-distance recovers);
    mid-wall/floor shells within tolerance of pre-fix. Failing on HEAD (single-normal), passing in
    multi. Round cup + square `PolyCup`.
  - Modify `src/solvers/twofield/mod.rs` — flip `dbg.y` default to `WALL_BC_MULTI`.
- **Approach:** D6. Calibrate-then-pin: confirm the multi-normal corner reaches box parity and the
  visual ring is gone before pinning the margin (record HEAD corner numbers). Order-invariant
  density reads (`reorder_stale_phase_artifact`); quantiles not raw max.
- **Patterns to follow:** the corner/`dnn` measurement already in `tests/twofield_wall_audit.rs`.
- **Test scenarios:**
  - `Covers R7.` Corner parity gate (two-sided): round + square `|corner − box| ≤ margin` after
    (neither over-dense NOR a void); the **same assertion fails on HEAD (~2.4× vs box 1.4×)** —
    record HEAD numbers.
  - `Covers R8.` `settled_cup_water_no_edge_ring` < 2.5.
  - Mid-wall + floor-center shells stay near pre-fix (fix is corner-local).
- **Verification:** parity gate red on HEAD / green after; square matches box; area-density green;
  default flipped.

### U4. Full L0–L3 regression + floor check + perf + retire the selector

- **Goal:** Keep every twofield gate green (esp. the gravity-loaded floor), meet perf, remove the
  single-normal branch + `dbg.y` selector (multi becomes the only BC).
- **Requirements:** R3, R5, R6.
- **Dependencies:** U3.
- **Files:**
  - Modify `src/solvers/twofield/{pressure,coupling}.wgsl`, `mod.rs` — delete the single-normal
    branch + `dbg.y` selector once green (the basis = one normal away from seams, so mid-wall/floor
    behavior is preserved); finalize docs.
  - Tests: existing L0–L3 suites.
- **Approach:** Run the full ladder; explicitly confirm the cup **floor** does not regress (the
  basis must still constrain the floor face away from the corner — the band that prevents column
  pancaking to 10–20×). Retiring the selector routes the single-normal *case* through the unified
  `P = I − n̂n̂ᵀ`, which is operator-equivalent **to tolerance** (not bitwise) vs the old
  `invr − invr·n̂n̂ᵀ` (D4). **Name the equivalence check before deleting the old branch:** a
  single-face `WALL_BC_SINGLE` vs `WALL_BC_MULTI` characterization of `nm` (per-node) and the
  `drag_fold` velocity result, agreeing to the existing GPU↔twin tolerances, plus the L0 operator
  gates green at tolerance.
- **Execution note:** Regression + cleanup; remove the selector only after green everywhere; never
  loosen a gate (`AGENTS.md`). Manual in-browser Tint compile check.
- **Test scenarios (existing gates must stay green):**
  - L0 operator (`tests/twofield_pressure.rs`); L1 bed/plasticity (`tests/twofield_bed.rs` incl.
    `settled_heap_static_rest_no_creep`, `over_packing_rejected_at_phi_max`, repose).
  - L1–L2 coupling (`tests/twofield_coupling.rs` incl. the **face-velocity twin**, buoyancy,
    Darcy, slip-decay, `settled_saturated_column_stays_settled_long_run`).
  - L2–L3 full (`tests/twofield_full.rs` incl. Terzaghi+Skempton, no-fluidize, crater persist +
    slump-without-blend, volume conservation, determinism).
  - `Covers R3.` Floor: cup floor holds (no pancaking regression) — bed/Terzaghi column + the
    corner audit's floor-shell stay bounded.
  - `Covers R6.` Perf `tests/twofield_perf.rs` R9 ≤ 33 ms @ 200k.
- **Verification:** L0–L3 + perf green; floor holds; single-normal path + selector removed; `clippy`
  clean.

---

## Scope Boundaries

- **In:** the binding-faces query; the orthonormal-basis projector `P = I − QQᵀ` in `node_setup` +
  `drag_fold`; the corner-audit parity gate; bring-up behind `dbg.y` then retirement.
- **Out:** the mid-wall coverage coefficient and broad cut-cell coefficient (superseded); the
  "25×" framing (artifact); vertical pancaking / PIC-blend (separate, crater tradeoff);
  corner-aware push-out; per-face SDF restructure; Rhie-Chow; geometry fillet/inset.

### Deferred to Follow-Up Work

- The **cone** apex∩floor / converging-wall seam — the cone is a single smooth surface; it stays
  single-normal here (D5). If a physical apex cap is added later, give it its own faces + test.
- **Square-cup vertical wall∩wall edges** are exercised by the query + the U2 non-orthogonal PSD
  gate (they prove the basis is correct), but are **not** product-gated (the V60 cup is a round
  cylinder with no vertical edges; PolyCup is a diagnostic fixture).

---

## Risks & Dependencies

- **R-1: non-orthogonal binding normals break a naive projector.** A raw `Σ n̂ᵢn̂ᵢᵀ` is non-PSD
  for non-orthogonal normals (poly vertical edge) → `A` not SPD. **Mitigated by the core design**
  (D2): the orthonormal basis `P = I − QQᵀ` is PSD for any normals; the U2 non-orthogonal PSD gate
  (`PolyCup sides=8` eigenvalue check + CPU wedge/triple-corner) guards it.
- **R-2: node_setup/drag_fold operator drift.** Sequential velocity projection ≠ `I − QQᵀ` for
  non-orthogonal normals. **Mitigated** (D3): both kernels build the identical `Q` and apply the
  identical `P`; the lockstep test pins it; the single-normal bitwise-identity catches accidental
  divergence.
- **R-3: box-axis / SDF double-removal or rank > 3.** **Mitigated** (D2): `Q` is seeded with the
  active box axes and SDF normals are residualized against existing columns, capped at rank 3 — the
  unified projector cannot double-remove. (The V60 cup floor at y=−8 is **not** coincident with the
  domain floor at y=−10, but the query runs on all geometry, so the accumulator handles overlap
  generally.)
- **R-4: floor-support regression (R3/R5).** **Mitigated:** the basis is a strict generalization
  (one floor normal away from the corner); U4 re-checks the floor.
- **R-5: operator twin must mirror the basis** or the SPD/adjoint gates pass a wrong operator.
  **Mitigated:** U2 updates the twin to the orthonormal-basis projector (not single-dyad).
- **Dependency:** U1 → U2 → U3 → U4 strictly sequential. `g2p_water` push-out unchanged. Fixtures
  `SolidKind::PolyCup` + `Scene::v60_cup_static_full` + `tests/twofield_wall_audit.rs` in the tree.

---

## Sources & Research

- This session's audit (`tests/twofield_wall_audit.rs`, commits 14f85ce/716fb90): corner 2.4× vs
  box 1.4×; square-vs-box corner contrast; metric reconciliation (the "25×" was a tail-sensitive
  `(s/d_nn)³` artifact).
- Code: `src/solvers/twofield/common.wgsl` (`cyl_cavity`/`poly_cavity`/`cone_cavity`/`solid_union`),
  `pressure.wgsl` (`node_setup` + `wall_coverage` selector from U1), `coupling.wgsl` (`drag_fold`),
  `transfers.wgsl` (`g2p_water`), `tests/twofield_pressure.rs` (operator twins), `src/utils/sdf.rs`
  (CPU SDF twin).
- Codex review round 1 (gpt-5.5 xhigh): caught that `Σ n̂ᵢn̂ᵢᵀ` is non-PSD for non-orthogonal
  normals and that sequential velocity projection breaks lockstep — folded into the orthonormal
  basis `P = I − QQᵀ` (D2/D3) + the non-orthogonal PSD gate (U2).
- Codex review round 2 (gpt-5.5 xhigh): exact scalar `sig·invr·(I − QQᵀ)` + preserved `nm` flag
  lanes; single-normal default kept literally unchanged for the bitwise claim (D4); rank-aware
  overflow so side faces can't crowd out the floor normal (D2/U1); two-sided parity band so an
  over-constrained void also fails (D6); near-`SDF_NORMAL_MIN` CPU/GPU determinism test (U1/U2).
- Origin: `docs/brainstorms/2026-06-17-twofield-cup-corner-seam-bc-requirements.md`.
- Superseded: `docs/plans/2026-06-17-001-fix-twofield-sdf-wall-embedded-boundary-plan.md`.
- Memory: `project_twofield_cup_compressibility`.

---
date: 2026-06-17
topic: twofield-cup-corner-seam-bc
supersedes: docs/brainstorms/2026-06-17-twofield-sdf-wall-embedded-boundary-requirements.md
---

# Two-field cup corner-seam BC — requirements

## Summary

Water visibly packs into the **concave floor∩wall corner** of the V60 cup. This was
re-diagnosed this session (audit `tests/twofield_wall_audit.rs` + the user's live-render oracle)
and pinned to a specific boundary-condition defect: at the seam where two cavity surfaces meet,
`solid_union()` returns a **single** nearest normal, so the grid BC (`node_setup`'s M̃⁻¹ dyad and
`drag_fold`'s velocity BC) constrains only **one** of the two surfaces — water flows into the
corner along the unconstrained normal and the `g2p_water` push-out stacks particles onto the
surface. The fix is a **multi-normal seam projection**: when a node is within the band of two
cavity faces, project out **both** orthogonal normals — the SDF analog of the domain box's
two-axis corner zeroing.

This supersedes the embedded-boundary requirements doc (`2026-06-17-twofield-sdf-wall-embedded-boundary-requirements.md`),
which was scoped to a mid-wall cut-cell coverage coefficient at the wrong locus and a ~25×
severity that turned out to be a metric artifact.

## Problem Frame

Confirmed, with data:
- **Corner is the worst locus.** Coarse particle-density ρ/ρ_rest: **corner (floor∩wall) 2.4×**
  vs mid-wall 1.8×, floor-center 1.2×, interior 1.3×. The corner also has push-out-stacked
  coincident pairs (min nearest-neighbor distance 0.0007).
- **It is the corner BC code path, not geometry.** An aligned **square** SDF cup corner packs to
  **2.40×** while a geometrically identical **flat domain box** corner is **1.40×** — same
  geometry, differing only in the corner BC. The box zeroes both axes at its corner
  (mutually-orthogonal `P_box`); the SDF path takes one `solid_union` normal.
- **The cavity SDFs already compute both faces.** `cyl_cavity`/`poly_cavity`
  (`src/solvers/twofield/common.wgsl`) compute `d_side` and `d_floor` (and both normals)
  separately, then **return only the nearer** (`if (d_side <= d_floor) …`). The cup cavity is an
  *intersection* (above-floor AND inside-wall); the corner is where `d_side ≈ d_floor`. The fix
  is to **keep both** binding normals when both are within the band.

The visible coincident pairs are a **symptom** of the push-out — the box, with the identical
push-out, shows none, because its grid BC constrains the corner upstream. So the fix belongs in
the grid BC, not the push-out.

## Key Decisions

- **Fix the grid BC, not the push-out** (the push-out stacking is downstream of the BC
  under-constraint).
- **Multi-normal seam projection (chosen approach).** A new SDF query returns the **binding
  faces within band** (each `(dist, n̂)`); `node_setup` subtracts each face's dyad from M̃⁻¹ and
  `drag_fold` removes each face's normal component — the **same** set of faces in both (lockstep).
- **Operator consistency holds by construction.** Two (or more) mutually-orthogonal face normals
  give a rank-k symmetric projector `P = I − Σ n̂ᵢn̂ᵢᵀ`, so `M̃⁻¹ = invρ·P` stays symmetric-PSD and
  `A = D·M̃⁻¹·G` stays SPD. At an axis-aligned corner it reduces **exactly** to the box's
  multi-axis zeroing.
- **Severity is mild (corner ~2.4×, box ~1.4×).** The target is to **match the box (~1.4×)**, not
  rest (1.0×) — the box corner's 1.4× is ordinary hydrostatic settling, not a bug.
- **Gate behind the existing `dbg.y` selector** (shipped this session, commit 3980eb5) during
  bring-up; binary single-normal stays the default until the multi-normal path is verified.

## Requirements

**Behavior**
- R1. The cup floor∩wall corner holds near the box reference: corner ρ/ρ_rest drops from
  **~2.4× to ~1.4×** on both the round cup and the square `PolyCup`; the coincident-pair stacking
  (min nearest-neighbor distance) recovers toward the box's.
- R2. For an axis-aligned corner the multi-normal projector reduces **exactly** to the box's
  two-axis zeroing (the aligned square corner matches the box).
- R3. The existing single-face wall and floor support is preserved away from seams (no
  regression of the floor band that prevents column pancaking to 10–20×).

**Correctness & coexistence**
- R4. Operator consistency `A = D·M̃⁻¹·G` holds (assembled-A SPD/adjointness + divergence-decay
  gates green); `node_setup` M̃⁻¹ and `drag_fold` velocity BC project the **same** face set.
- R5. All twofield L0–L3 gates stay green: operator, Terzaghi/Skempton, volume conservation,
  no-fluidize, crater persistence, buoyancy, the face-velocity twin, no-creep/repose.
- R6. R9 ≤ 33 ms @ 200k; fixed-point i32 grid (no float atomics), per-entry-point storage-buffer
  ceiling (no new buffer — computed inline from the already-bound `solids`), Tint uniformity
  (the per-face loop must keep any downstream `workgroupBarrier` under uniform control flow).

**Validation**
- R7. The corner audit (`tests/twofield_wall_audit.rs`) is the empirical gate: the SDF cup corner
  must drop 2.4× → ~1.4× (the box), with the box-vs-SDF-square corner contrast closing.
- R8. `settled_cup_water_no_edge_ring` (radial area-density) stays green.

## Chosen Approach (and alternatives considered)

**(a) Multi-normal seam projection — CHOSEN.** Keep both binding face normals when both are within
band; subtract both dyads in M̃⁻¹ + the velocity BC, lockstep. Cheap (the faces are already
computed — stop discarding the non-nearest), exact (reduces to the box), operator-consistent by
construction, no new buffer.

Alternatives explored and rejected:
- **(b) Corner-aware push-out** — fixes the *symptom* (push-out stacking) not the root (grid BC);
  injects momentum outside the operator; the box uses the same push-out without stacking.
- **(c) Per-face SDF restructure** (separate floor + wall primitives) — `solid_union` still
  returns a single min, so it *still* needs (a)'s multi-binding query; and a min-union is
  geometrically wrong at a concave (intersection) corner. More machinery, worse geometry.

## Scope Boundaries

- **In:** the floor∩wall seam of the cup (cylinder + poly); the binding-faces-within-band SDF
  query; multi-normal projection in `node_setup` + `drag_fold`; the corner audit gate; bring-up
  behind `dbg.y`.
- **Out:** the mid-wall coverage coefficient and the broad cut-cell coefficient (wrong locus —
  mid-wall matches the box); the "25×" severity framing (metric artifact); vertical pancaking and
  the PIC-blend lever (separate, milder, crater tradeoff); corner-aware push-out; per-face SDF
  restructure; Rhie-Chow / MAC; geometry fillet/inset.
- **Deferred (follow-up):** the **cone**'s converging-wall / apex seam — the cone is a single
  smooth surface (segment closest-point), so its only true seam is apex∩floor; verify whether it
  needs the same treatment after the cup is fixed. Square-cup **vertical** wall∩wall edges (less
  gravity-loaded than floor∩wall) — confirm the per-face query covers them for free.

## Dependencies / Assumptions

- The cavity functions already expose `d_side`/`d_floor` + both normals (`common.wgsl`), so the
  binding-faces query is a refactor, not new geometry. (Verified.)
- The CPU SDF twin (`src/utils/sdf.rs`) and any operator CPU twin
  (`tests/twofield_pressure.rs`) must mirror the multi-normal query so the twins don't give false
  assurance.
- `g2p_water` push-out is unchanged (it remains the particle-resolution backstop).
- Two binding normals at a cup corner are orthogonal (radial ⟂ +y); the projector is rank-2. If a
  primitive ever produced near-parallel binding normals, the projector must guard against
  double-counting (degenerate-normal guard, as the single-normal path already does).

## Outstanding Questions

**Resolve before planning:** none blocking.

**Deferred to planning / Codex review:** the exact "binding faces within band" API shape (return a
small fixed-array of hits vs a fold); the band width for the seam test (reuse `WALL_BAND·h`?);
how to keep the per-face loop Tint-uniform; whether the floor-only band tweak that prevents
pancaking is subsumed by the multi-face query or stays separate; the corner audit's pinned
threshold (after the fix is confirmed to match the box).

## Sources / Research

- This session's audit (`tests/twofield_wall_audit.rs`, commits 14f85ce/716fb90): corner 2.4× vs
  box 1.4×; metric reconciliation (the "25×" was a tail-sensitive `(s/d_nn)³` nearest-neighbor
  metric; robust over-pack 1.8× coarse / ~1.7× nn vs box).
- Code: `src/solvers/twofield/common.wgsl` (`cyl_cavity`/`poly_cavity`/`cone_cavity`/`solid_union`
  — both face distances already computed), `pressure.wgsl` (`node_setup` M̃⁻¹ + `wall_coverage`
  selector), `coupling.wgsl` (`drag_fold`), `transfers.wgsl` (`g2p_water` push-out).
- Memory: `project_twofield_cup_compressibility` (full diagnosis chain, corrected).
- Prior (superseded) framing: `docs/brainstorms/2026-06-17-twofield-sdf-wall-embedded-boundary-requirements.md`,
  `docs/plans/2026-06-17-001-fix-twofield-sdf-wall-embedded-boundary-plan.md`.

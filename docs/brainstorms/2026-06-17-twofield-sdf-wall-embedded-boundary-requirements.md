---
date: 2026-06-17
topic: twofield-sdf-wall-embedded-boundary
---

# Two-field SDF wall: embedded-boundary pressure BC — requirements

## Summary

Replace the two-field water solver's SDF-solid wall boundary condition — the binary `WALL_BAND·h` wall-normal (`n̂n̂ᵀ`) projection that over-constrains a 1-cell fluid shell and crushes water against *any* SDF wall — with a **fractional / cut-cell embedded-boundary** treatment: place the no-penetration constraint at the *true* (sub-node) wall crossing on a *single* fluid layer with a fractional weight. This is the domain box-face BC **generalized** to walls that don't land on grid nodes; it reduces to the box-face BC for aligned walls and stops the wall-shell over-pack the V60 cup exhibits.

## Problem Frame

Water in the V60 cup over-packs into a dense shell at the wall — visible as a ring, measured (particle-side neighbor-count ρ/ρ_rest, 1.0 = incompressible) at ~25× at the wall (poured ~35×). A multi-experiment re-diagnosis this session ruled out every easier explanation and isolated the cause:

- **Not too-compressible fluid.** A cup-sized *flat domain box* at the same spacing/fill (more water at its wall) settles to **0.98×** — the solver holds a full column at the wall at rest density.
- **Not the pour.** A *statically* seeded full cup (no pour) collapses on its own to 16.7× / wall 25.6×.
- **Not curvature or off-grid normals.** A regular N-gon prism cup over-packs at every orientation: square (axis-aligned faces) 18×, octagon 26×, round 26×. The square SDF cup and the flat box are *geometrically identical* (axis-aligned flat walls) yet differ 18× — they differ only in **code path**.
- **It is the SDF-solid wall code path.** The domain box uses the box-FACE BC (exact, node-aligned, single axis, **no band**). Every SDF solid uses the `WALL_BAND·h` banded `n̂n̂ᵀ` projection (`node_setup` M̃⁻¹ + `drag_fold` velocity) over a 1-cell fluid shell. That shell can't relieve radial over-density → over-pack.
- **Two contributing effects** (node-alignment test): snapping a square cup's faces exactly onto nodes drops it to ~2.9× (vs ¼-cell-off 4.3×, vs the earlier non-aligned 18×) — so (1) sub-node wall placement matters, but (2) even perfectly aligned it stays ~2.9× (not 0.98×): the band's 1-cell reach over-constrains an interior layer regardless of alignment.

The fix the field uses for exactly this — accurate fluid–solid coupling at irregular grid boundaries — is a variational / cut-cell pressure BC (Batty, Bertails & Bridson 2007): sub-cell solid *fractions* in the operator instead of a binary projection.

## Key Decisions

- **Embedded-boundary BC fix, not a compressibility fix.** The prior density-constraint plan (`docs/plans/2026-06-16-001`) and its 3 Codex rounds targeted the wrong cause and are shelved — the flat box proves the fluid is incompressible enough.
- **Generalize the box-face BC, don't bolt on a band.** The box-face BC is correct *because* it constrains exactly at the (node-aligned) wall on a single layer. The fix carries that property to sub-node / curved walls via fractional coverage; it must reduce to the box-face BC when the wall is node-aligned and axis-aligned.
- **Operator consistency is sacred.** Any change to the wall BC touches `node_setup`'s M̃⁻¹ AND `drag_fold`'s velocity BC together — `A = D·M̃⁻¹·G` must hold (a prior one-sided wall BC broke this and was reverted).
- **The real symptom is local volume-density; the existing gate is blind.** `settled_cup_water_no_edge_ring` measures radial *area*-density (2.2× baseline = passes) and does not see the 25× local volume crush. A new particle-volume-density gate is required.

## Requirements

**Behavior**

- R1. A full cup (static or poured) holds wall-shell ρ/ρ_rest near rest — target the domain box's ~1.0, and at minimum < 2× even at the wall, on the round cup AND the square `PolyCup`.
- R2. For a node-aligned axis-aligned wall the new BC reduces to the box-face BC (the square `PolyCup` reaches the box's ~0.98×).
- R3. Confinement and radial distribution are both preserved — no penetration/leak AND no hollow-center radial collapse (the failure mode of naively dropping the band).

**Correctness & coexistence**

- R4. Operator consistency `A = D·M̃⁻¹·G` holds (assembled-operator adjointness/SPD + divergence-decay gates green); the `drag_fold` velocity BC and `node_setup` M̃⁻¹ stay mutually consistent.
- R5. All twofield L0–L3 gates stay green: Terzaghi/Skempton, volume conservation, no-fluidize, crater persistence, buoyancy, face-velocity twin. (The band currently also supports the non-grid-aligned bed floor — the fix must keep that support.)
- R6. R9 ≤ 33 ms @ 200k holds; fixed-point i32 grid (no float atomics), per-entry-point storage-buffer ceiling, Tint uniformity respected.

**Validation**

- R7. A new **particle-volume-density** gate (neighbor-count ρ/ρ_rest) on the confined cup fails on HEAD (~25×) and passes after the fix (the existing area-density gate is blind to it).
- R8. The `PolyCup` (square/octagon) + `v60_cup_static_full` fixtures (already added) are the acceptance harness; the box stays the ~0.98 reference.

## Candidate Approaches

- **A — Variational / fractional-coverage solid boundary (Batty et al. 2007).** Sub-cell solid volume/face fractions weight the pressure operator at wall cells; the no-penetration is enforced at the true surface with the right partial weight. The established, accurate method; reduces to box-face for aligned walls. Larger change to the pressure operator; must map onto the fixed-point grid + mixture and hold operator consistency.
- **B — 1D normal-direction cut-cell / ghost-fluid (lighter).** Constrain only the single fluid layer adjacent to the wall, at the fractional normal distance to the true crossing (not a full 1-cell band). Node-alignment alone recovered most of the over-pack (18→2.9), suggesting the dominant fix is placement + single-layer, which B targets with less machinery than full A.
- **C — Fractional band weight.** Keep the band structure but weight each banded node's projection by its fractional distance/coverage instead of a binary apply. Smallest code change; may not fully reach box-face accuracy.

*Recommendation carried into planning:* explore B first (lighter; the alignment result suggests single-layer + fractional placement captures most of it) with A (full variational) as the principled fallback if B can't reach the box's ~1.0 while holding R4/R5; external research should confirm Batty applicability to the fixed-point MPM mixture and the real-time budget.

## Scope Boundaries

- Not a compressibility / density-constraint rewrite (wrong cause — shelved).
- Not pour-specific work (the static cup over-packs identically).
- Not curvature-specific hacks (the axis-aligned square over-packs too).
- Not a "snap geometry to nodes" hack (insufficient — residual band over-reach; impossible for round cups).

## Dependencies / Assumptions

- The push-out (`g2p_water`) already enforces particle-resolution non-penetration independently — the BC change is about *pressure relief*, not confinement.
- The band currently supports the non-grid-aligned bed floor; the fix must preserve that (R5) — verify against the bed/Terzaghi gates.
- External research (Batty variational coupling; cut-cell/ghost-fluid embedded boundaries) is warranted at planning to choose A vs B and confirm the fixed-point/real-time fit.

## Outstanding Questions

**Resolve before planning:** none blocking.

**Deferred to planning / Codex review:** A vs B vs C; the fractional-weight formulation that keeps M̃⁻¹ symmetric-PSD and `A = D·M̃⁻¹·G`; whether the floor (vertical-normal) band can stay while only the side-wall treatment changes; the exact volume-density gate thresholds.

## Sources / Research

- This session's experiments (particle-side ρ/ρ_rest): flat box 0.98; static-full round cup 16.7/wall 25.6; square 18 / octagon 26 / round 26; band-dropped 25.6→4.3 (but radial collapse, fails `settled_cup_water_no_edge_ring`); aligned square 2.9 vs misaligned 4.3.
- Code: `src/solvers/twofield/pressure.wgsl` (`node_setup` M̃⁻¹ wall band, box-face BC, `WALL_BAND`), `src/solvers/twofield/coupling.wgsl` (`drag_fold` wall band + box-face BC), `src/solvers/twofield/transfers.wgsl` (`g2p_water` push-out), `src/solvers/twofield/common.wgsl` (`solid_cavity`), `tests/twofield_settled.rs::settled_cup_water_no_edge_ring` (area-density gate, blind to volume crush), `tests/twofield_cup.rs` footer.
- Fixtures added this session: `SolidKind::PolyCup` (sdf.rs + common.wgsl + both packers + wireframe), `Scene::v60_cup_static_full` (scene.rs).
- External: Batty, Bertails & Bridson, "A Fast Variational Framework for Accurate Solid-Fluid Coupling," SIGGRAPH 2007 (the variational/fractional embedded-boundary pressure method). Memory: `project_twofield_cup_compressibility` (full corrected diagnosis).

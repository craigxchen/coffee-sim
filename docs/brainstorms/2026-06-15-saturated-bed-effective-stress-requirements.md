---
date: 2026-06-15
topic: saturated-bed-effective-stress
---

# Saturated Bed Effective-Stress Yield — Wet-Sand Crater Persistence

## Summary

Redefine the target deformation behavior of the two-field solver's saturated coffee bed, and the constitutive model that produces it. A poured crater should **form while the bed is flooded and then persist** (hold almost fully) after drawdown — wet-sand plasticity, not slump-back-to-flat. Drive this with an **effective-stress-dependent Drucker-Prager yield** (yield strength as a function of total stress − pore pressure), reusing the existing two-field pore-pressure field: the bed is weak/mobile when flooded and strong/holding when drained. Correct the crater gate (which currently asserts the bed *slumps* — physically wrong, and passing today only on numerical agitation), and re-enable the open-water APIC damping that quiets the settled-pool stirring — now compatible, because a held crater is the correct outcome rather than a failure.

This brainstorm is inherently a physics/architecture decision, so the constitutive model is in scope here (not deferred to planning); the *how* (return-map placement, the damping mechanism) is left to `/ce-plan`.

---

## Problem Frame

The preceding investigation (memory `project_twofield_stirring_crater_coupling`) proved that the deformable bed's crater slump rides on **numerical APIC agitation** — the *same* open-water agitation the user sees as settled-pool "stirring." Every damping knob that quiets the pool also freezes the crater; more pressure sweeps inflate the pool ~5×. The crater-collapse gate `center_pour_craters_saturated_deformable_bed_then_collapses` (`tests/twofield_full.rs`) passes only because the agitation shakes the cohesive wet bed loose enough to slump it.

External research into real pour-over physics says the **opposite** of that gate: a poured crater **persists** into the spent bed. The Rao Spin technique exists *only because* an un-agitated bed is not flat — it retains the pour's topography; "gravity alone does not redistribute particles." So the gate asserts the wrong behavior, and "stirring" and "slump" are two faces of one artifact. Correcting the gate's target, grounding the bed's yield in effective stress, and re-enabling open-water damping resolve all three together.

---

## Target Behavior

- **Wet coffee grounds are a cohesive saturated granular material — wet sand.** Capillary cohesion gives the wet bed a higher yield / angle of repose than dry grounds; it holds steep walls. It deforms when carved (the pour) and then *holds the deformed shape*. This is plasticity, not flow.
- **Yield strength is governed by effective stress** = total stress − pore pressure (Terzaghi), so it is a strong function of saturation, not a constant:
  - **Flooded** (active pour, high pore pressure → low effective stress): weak, slurry-like, readily carveable; grains avalanche (the jet digs a hollow, rim grains avalanche back).
  - **Draining** (drawdown, pore pressure falls → effective stress rises): capillary cohesion peaks, the bed stiffens to wet-sand, and whatever topography exists **freezes in**.
- **Observable crater life-cycle:** a center pour carves a crater (depth above the floor) while the bed is flooded; after the pour stops and the bed drains, the crater **persists almost fully** (residual ≈ peak depth) — it does not relax back toward flat. The only thing that re-levels it is active agitation (swirl/stir), which is a user action, not automatic (out of scope here).
- **The open-water pool** (cup, pond) settles quiet — no spontaneous stirring — once the APIC damping is enabled, which is now compatible with the held crater.

---

## Key Decisions

- **Effective-stress-dependent Drucker-Prager yield (chosen fidelity).** Tie the bed's yield to effective stress using the pore-pressure field the solver already computes (the Terzaghi/Skempton machinery in `coupling.wgsl`). Chosen over: *constant wet-cohesion* (already held the crater in experiments but does not reproduce the flooded→drained strength transition) and *creep/viscoplastic flow* (rejected outright — creep is slow flow over time, i.e. exactly the slumping we must avoid; the right physics is plasticity with a saturation-dependent yield).
- **Invert the crater gate.** Replace "crater slumps to ≤ 0.6 × peak" with "crater forms while flooded, then persists after drawdown (residual ≈ peak, minor-settling tolerance only)." The current gate tested an artifact and rewarded it.
- **Re-enable the open-water APIC damping** (the `PIC_BLEND_DEFAULT` knob, kept inert from the investigation) to fix the settled-pool stirring. Now compatible — a held crater is correct, so damping no longer "breaks" the crater.
- **Add a decouple-proof gate.** Prove the crater persists *with the open-water agitation damped*, so persistence is demonstrably physics-driven (effective-stress cohesion), not numerical-agitation-driven.
- **Defer agitation re-leveling.** Swirl/stir flattening the bed is real but the sim has no agitation input today; out of scope for this work.

---

## Requirements

**Bed constitutive model**

- R1. The bed's Drucker-Prager yield is a function of effective stress (total stress − pore pressure), computed from the existing two-field pore-pressure field. Higher pore pressure (flooded) lowers the yield (weaker/mobile); lower pore pressure (drained) raises it (stronger/holding).
- R2. The effective-stress dependence is observable, not nominal: a bed under the same applied load deforms measurably more when flooded than when drained.

**Crater behavior**

- R3. A center pour carves a crater in the saturated bed to a depth at or above the pre-registered crater-depth floor, while the pour runs (forms readily while flooded).
- R4. After the pour stops and the bed drains, the crater persists almost fully — residual depth a large fraction of peak (target ≈ ≥ 0.8 × peak), explicitly NOT slumping to flat. This inverts the current collapse gate.
- R5. Crater persistence (R4) holds with the open-water APIC agitation damped — demonstrating the persistence is effective-stress cohesion, not numerical agitation.

**Settled pool (loop closure)**

- R6. With the open-water APIC damping enabled, the settled cup pool / pond shows bounded KE with no spontaneous re-energization (the original stirring complaint is fixed) AND the crater still persists per R5.

**Preservation (hard constraints)**

- R7. All existing L0–L3 gates remain green: Terzaghi consolidation + Skempton-B, volume conservation, no-fluidize (the bed does not liquefy under sustained pour), the face-velocity-consistency twin, and the R9 real-time gate (≤ 33 ms/frame @ 200k on the reference device).
- R8. Solver discipline preserved: no float atomics (fixed-point i32), no indirect dispatch, per-pipeline storage-buffer ceiling respected, and no changes to `src/solvers/xpbd/`.

---

## Acceptance Examples

- AE1. Center pour on a saturated deformable bed → a crater forms with peak depth ≥ floor while the pour runs. (Covers R3.)
- AE2. Pour stops, bed drains for a long settling window → residual crater depth is a large fraction of peak (≈ ≥ 0.8 ×), NOT ≤ 0.6 × peak. (Covers R4 — the inverted gate.)
- AE3. Repeat AE2 with open-water APIC damping ON → crater still persists (≈ ≥ 0.8 × peak) AND the open pool's settled-tail KE stays bounded with no spikes. (Covers R5, R6 — the decouple gate.)
- AE4. Effective-stress probe: an identical applied load deforms a flooded bed (high pore pressure) measurably more than a drained bed (low pore pressure). (Covers R1, R2.)
- AE5. The existing Terzaghi/Skempton, volume-conservation, no-fluidize, twin, and R9 gates all remain green. (Covers R7.)

---

## Scope Boundaries

**In scope**

- Effective-stress-dependent Drucker-Prager yield for the bed (the constitutive change).
- Crater gate correction (persistence) + the decouple gate (AE3) + the effective-stress probe gate (AE4).
- Re-enabling the open-water APIC damping.

**Deferred to follow-up work**

- The damping mechanism choice (global PIC blend vs φ_f-gated blend) — resolved during planning, verified against the consolidation gates.
- Yield/cohesion calibration constants — execution-time tuning against the gates.

**Outside this work**

- Any creep / viscoplastic (time-dependent flow) bed model — wrong physics for this target (creep slumps).
- Agitation / swirl re-leveling input — the sim has no agitation control yet.
- Re-architecting the two-field core, the pressure solve, or the transfer scheme.

---

## Open Questions

- **Damping mechanism.** Global PIC blend (simplest; also damps bed pore-water, possible Terzaghi interaction) vs φ_f-gated blend (keeps bed pore-water full-APIC; safest for consolidation gates). The faithful effective-stress yield may make a global blend safe (the crater no longer depends on agitation) — to be confirmed in planning against Terzaghi/Skempton.
- **Pore-pressure → yield mapping.** Linear effective-stress reduction vs a saturation-indexed softening curve, and where in the Klar return map it applies. Planning/research decision.
- **Fate of `tf_wet_cohesion`.** Whether the existing wet-cohesion term stays, is replaced by, or is subsumed into the effective-stress term.

---

## Dependencies / Assumptions

- The two-field solver already computes a pore-pressure field and applies it to the solid (the Terzaghi/Skempton gates pass), so the effective-stress signal needed for R1 is available — to verify during planning.
- The integration point is the existing Klar Drucker-Prager return map (`src/solvers/twofield/plasticity.wgsl`) and the pore-pressure coupling (`src/solvers/twofield/coupling.wgsl`).
- The bounded-KE settled gates (`tests/twofield_settled.rs`) and the inert `PIC_BLEND_DEFAULT` knob are already in place from the stirring investigation.

---

## Sources & Research

Pour-over bed physics research (external; full findings folded into Target Behavior and Key Decisions):

- **Crater persistence (the crux) — confirmed.** Pour-induced craters/channels persist into the spent bed unless actively agitated; the Rao Spin exists precisely because the un-spun bed is not flat. Sources: Scott Rao ("Some Observations on Hand Pours"), Barista Hustle ("The Rao Spin"), Coffee ad Astra (Gagné — "Four Rules of Optimal Coffee Percolation", "Why Spin the Slurry", "Extraction Uniformity and Channeling").
- **Yield strength = f(effective stress), saturation-dependent.** Crater scaling controlled by compressive yield stress in wet granular media; yield rises with capillary cohesion then drops at full saturation (pore pressure → low effective stress). Sources: Phys. Rev. E 92, 042205 (2015) "Scaling of liquid-drop impact craters in wet granular media"; arXiv 2404.01631 (2024) "Drop impact on wet granular beds"; arXiv cond-mat/0106572 (wet pile angle of repose).
- **Carve-then-freeze / avalanche during the jet.** The jet digs a hollow; grains avalanche back during the active pour (granular solid, not uniform slurry); the bed swells and stiffens as it saturates/drains. Sources: Penn/UPenn Physics of Fluids 2025 "Pour-over coffee: Mixing by a water jet impinging on a granular bed with avalanche dynamics"; arXiv 2512.21528 (2025) poroelastic espresso (bed swelling/stiffening, cohesive spent puck).
- **Wet-sand / soil-consolidation analogy is load-bearing.** Effective stress = total − pore pressure; as the bed dewaters during drawdown it consolidates and gains yield strength, freezing topography in. This is the physical justification for R1.

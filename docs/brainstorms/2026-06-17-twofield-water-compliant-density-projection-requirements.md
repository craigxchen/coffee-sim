---
date: 2026-06-17
topic: twofield-water-compliant-density-projection
supersedes: docs/brainstorms/2026-06-16-twofield-water-incompressibility-requirements.md
---

# Two-field water: compliant predicted-density-error projection (PB-MPM, in spirit) — requirements

## Summary

Replace the two-field **water** phase's under-converged `∇·v = 0` projection + one-sided density-relief with a **two-sided predicted-density-error, compliant pressure projection** solved on the **existing** D / G = −Dᵀ / M̃⁻¹ operator and the existing cell-centered `pf_*` pressure field. This is **PB-MPM in spirit, not in letter**: it borrows PB-MPM's *compliant density constraint* (stable at any stiffness, no sound-speed CFL) as a diagonal regularization term, but does **not** port PB-MPM's particle position-based integration — because this solver's bed coupling is built around a cell-centered grid pressure, and a particle method produces position deltas, not the grid `∇p` the bed consumes. The pressure unknown stays `pf_*` and the operator is unchanged, so the bed↔water/crater coupling (buoyancy, Terzaghi/Skempton) is **structurally preserved** — the one physics check is that the partition still closes once the pressure is made slightly compliant (see Key Decisions).

This resolves the mechanism pick that `2026-06-16-twofield-water-incompressibility-requirements.md` left deferred (its candidates A/B/C), narrowed by this session's validation that the over-pack is **model-bound**, and by an independent Codex (gpt-5.5, high) source review.

## Problem Frame

Water poured into the V60 cup over-packs and (in the worst historical reading) collapses into a dense wall ring with a hollow center. Two prior framings were ruled out: it is not a wall/collision geometry artifact (two geometry fixes only *relocated* the ring), and not a collocated-grid checkerboard mode (Rhie–Chow / MAC staggering was tried and rejected). The fluid is simply **too compressible**, and the cause is in the incompressibility *model*, not its convergence:

- The projection enforces `∇·v = 0`, which **does not constrain density** — a static, fully-compacted pool measures zero velocity divergence. The projection is structurally blind to compaction (`src/solvers/twofield/pressure.wgsl`, header note "a fully-compacted tank measures ZERO divergence").
- The only restoring force is a **one-sided** density-relief source folded into the rhs: `rhs_c = (s_target − Dv)/dt`, `s_target = max(ρ̄_c/ρ_rest − 1, 0)/(DENSITY_RELAX·dt)` (`pressure.wgsl:494`). It relieves over-density only; an under-density (suction) half *does* exist as a separate path in `surface.wgsl`, gated **away from** air-adjacent cells (`!near_air`) and folding its deficit into the **same** solved rhs lane (`cell_meta.x`) — but it is a separate one-pass feedback, not part of a single two-sided **compliant** density target.
- **Validation (this session):** cranking fine sweeps 8 → 64 barely moved coarse over-pack (1.80× → 1.77× ρ/ρ_rest); turning relief OFF → 2.86× (relief is load-bearing but structurally insufficient). "Converge harder" does **not** fix it — the constraint *target* is wrong. *Caveat (what this does and does not discriminate):* the sweep rules out **under-convergence**, but not the corner-seam BC or the APIC/PIC blend (neither is convergence-sensitive either), and this same session attributed part of the residual ~1.8× to blend-reducible pancaking and the corner BC. So the load-bearing claim is that the constraint **target** is structurally wrong (a necessary fix), not that the density model is the *sole* contributor to the measured 1.8×. A clean isolation — a two-sided density target on a flat, node-aligned box at fixed blend — should precede any model-only attribution.

**The hard constraint — the bed coupling contract.** The two-field solver's value is the working bed↔water/crater coupling, and it depends on the projection pressure: "**the pore pressure IS the projection pressure**" (`pressure.wgsl:903–906`). `project` gathers `gp = G·p` from `pf_*`, applies water `v −= dt·M̃⁻¹·Gp`, and applies solid buoyancy `Δv_s = −(dt/ρ_s)·gp` from the **same** `gp` (`pressure.wgsl:853–928`). Whatever replaces the water incompressibility must keep producing that exact cell-centered pressure / `∇p` field, or the coupling breaks.

The over-pack reproduces in **WaterOnly** (single-phase, φ_s = 0), localizing it to the base water incompressibility; the mixture coupling is inert there and is a non-breaking constraint, not the cause.

## Key Decisions

- **Constrain density, two-sided, on the existing operator.** Retarget the rhs from `∇·v = 0` + one-sided `max(…,0)` relief to a **two-sided predicted-density-error** target (negative target = suction). Reuse the current consistent D / G / M̃⁻¹ operator, `pf_src`/`pf_dst`, the local-Jacobi sweep, the free-surface masks, and `project`. "Converge" now *means* "ρ → ρ₀".
- **PB-MPM in spirit = the compliance term.** Add PB-MPM's compliant-constraint stability as a **diagonal regularization** in the Jacobi denominator and the `A·p` residual — *not* a new buffer, *not* a particle method. Conceptual form: `A·p + (compliance_diag)·p = density_error_target/dt² − D(Φv)/dt`. This is the stability mechanism that lets a stiff density target hold without the explicit-EOS detonation (sound-speed CFL) the project already measured.
- **Bed coupling structurally preserved — pending a finite-compliance check.** The buffer, G, and the operator `A = D·M̃⁻¹·G` are unchanged, and the bed reads the same `gp = G·p`. But the compliance term changes the *solved* system from `A·p = b` to `(A + C)·p = b`, so the converged `p` is a **compliant** (slightly compressible) pressure, recovering the strictly-incompressible projection pressure only as `C → 0`. Whether the Terzaghi partition `σ = σ' + u` still closes at the chosen finite compliance is a check to run, not a given (see Outstanding Questions / R7). No lockstep change to `node_setup` M̃⁻¹ / `drag_fold` BC is required by this fix (the operator-consistency invariant is not in scope to modify).
- **Sign/units are pressure, not PBF λ.** The multiplier *is* the existing cell-centered pressure unknown, with **positive** pore pressure stored in `pf_*`. Do **not** import the XPBD/PBF particle λ convention (per-particle, compression-negative, `max(−lambda,0)`); that sign would corrupt the bed. PB-MPM/PBF are cited as the existence proof and the compliance idea, not as code to port.
- **Two-sided only on interior fluid rows.** Apply suction to interior fluid rows; keep air-adjacent / free-surface rows gated as `surface.wgsl` does today. Leaking suction onto surface rows would pull the free surface up or re-crush pockets — the chief risk.
- **This is the base water solve, single-phase first.** Because the ring reproduces in WaterOnly, the work targets single-phase water incompressibility; mixture coexistence is a non-breaking constraint validated by the existing grain gates.
- **No per-particle J-tracking needed.** The grid density census (`ρ̄_c`, already computed in `cell_classify`) supplies the density error directly — unlike the MLS-MPM J route, which is unnecessary on a grid that already measures node/cell density.

## Requirements

**Incompressibility behavior**
- R1. A standing water pool holds ρ/ρ_rest near rest and does **not** drift upward over prolonged settling (no monotonic compaction). Working bar: ρ/ρ_rest ≲ 1.2× even after long standing (revisable — see Outstanding Questions).
- R2. The density correction is **two-sided as a general interior constraint**: over-dense regions expand AND under-dense (jet-evacuated) interior regions draw fluid back, so an evacuated center refills rather than persisting as a hole. The structural change is to **unify** the existing interior suction (`surface.wgsl`, gated away from air-adjacent cells, already folded into the solved rhs lane) with the one-sided over-density relief into a single two-sided **compliant** density target — not to add a suction path, since one already exists.
- R3. Water poured into the cup fills as a coherent pool (interior carries mass; the free surface relaxes toward hydrostatic), not a thin wall ring or a packed corner.

**Stability & real-time budget**
- R4. Stable at the stiffness required for R1–R3 — no detonation / runaway inflation when stiffness is raised to incompressible levels (the explicit-relief failure mode must not recur). The compliance term must deliver this without a sound-speed timestep limit.
- R5. Fits the real-time budget: the real-time gate (≤ 33 ms @ 200k) holds, on the fixed-point i32 grid (no float atomics), **within the existing per-entry-point 7-buffer ceiling** (no new storage buffer — the compliance term and two-sided target reuse existing lanes), and under Tint uniformity.

**Coexistence & preservation**
- R6. All existing twofield L0–L3 gates stay green: operator adjoint/SPD/divergence-decay, Terzaghi/Skempton consolidation & load-sharing, volume conservation, no-fluidize, crater persistence, buoyancy, face-velocity twin.
- R7. The pore pressure fed to the mixture is unchanged in kind — positive cell-centered `pf_*`, same `gp = G·p` — so buoyancy / Terzaghi coupling keeps action–reaction consistency. The stress partition `σ_total = σ' + u` must still close **at the chosen finite compliance** — verified on BOTH the static hydrostatic audit (`tests/twofield_full.rs:899–929`) AND the dynamic Skempton / load-sharing coverage (`tests/twofield_full.rs:946–1036`), not only in the `C → 0` limit.

**Validation**
- R8. A success gate detects the ring/corner-pack failure **directly** — concretely, the fraction of settled water mass within an interior core (`r < R_interior`) versus the wall annulus, where a hollow center or a wall-packed corner FAILS an interior-mass-fraction floor (the numeric floor is a planning calibration). The current cup gate is blind to a centered ring: it passes `radial_rms ≥ 0.75·initial`, `centroid_r ≤ 0.4·R`.
- R9. The density excess is measurable and bounded as a regression signal (ρ/ρ_rest distribution: mean and high-percentile), not just a visual.
- R10. A gate proves the **suction does not leak to the surface/pockets**: free-surface height and pocket λ stay bounded (no surface pull-up, no pocket re-crush) when the two-sided target is on.
- R11. **Model-isolation gate** — a new test in `tests/twofield_wall_audit.rs` (alongside `aligned_square_localize_overpack`): with `pic_blend = PIC_BLEND_DEFAULT` held fixed across arms, measure settled `ρ/ρ_rest` via the existing `mean_nb` neighbor-count probe (box interior calibrated to 1.0) on a **flat node-aligned box** (`box_scene`) vs the **SDF cup** (`sdf_floor_scene`), each with the density fix ON. The flat box has no SDF seam and the blend is held constant, so its over-pack reduction is the **model** contribution; the residual flat-box→SDF gap is the BC contribution. *Pass:* the flat-box over-pack drops to ≤ the R1 bar (~1.2×). Model-only attribution of the residual ~1.8× AND retirement of the `WALL_BC_MULTI` selector require this gate green **and** the corner audit (`corner_parity_multi_drops_to_box`) at box parity with the fix on. (The exact numeric threshold is a planning calibration.)

## Candidate Approaches

The three adoption scopes posed at brainstorm open, with the resolved recommendation. All are two-field-grid-native; the XPBD particle PBF is cited only as the stability existence proof.

**Recommended — Compliant predicted-density-error projection on the existing operator** (the "hybrid"; this resolves the *lead recommendation* deferred in 2026-06-16 — an implicit, two-sided density-corrected projection — now carrying PB-MPM's compliance diagonal for stability. It is **not** that doc's Option A, which was weakly-compressible EOS).
Retarget the existing solve to a two-sided density-error target with a compliance diagonal, reusing `pf_*`, D/G/M̃⁻¹, the Jacobi sweep, surface masks, pocket rows, and `project`.
- *Pros:* G and `A = D·M̃⁻¹·G` unchanged so the bed coupling is structurally preserved (finite-compliance partition check pending — see Key Decisions); no new buffer; reuses the consistent operator; two-sided + compliant kills both the divergence loophole and the detonation; closest to the existing code, so the validation gates transfer directly.
- *Cons / risk:* the two-sided target can fight free-surface/pocket classification (the chief risk, R10); the converged pressure is compliant (slightly compressible), so the stress-partition closure must be checked; compliance value and target form need calibration vs. the real-time gate and the stress-partition audit.
- *Best when:* (always, here) — the bed contract makes a cell-centered pressure mandatory.

**Rejected — Full PB-MPM particle step (original Option B).**
Per-particle deformation-gradient (J) volume constraint with PB-MPM's semi-implicit particle integration, reshaping the water P2G/G2P toward the EA reference.
- *Why rejected:* it produces particle position/F corrections, **not** the grid `gp = G·p` the bed consumes. Recovered pressure would be a diagnostic approximation, not the force actually moving the water, breaking buoyancy/Terzaghi/Skempton action–reaction consistency. Largest change, highest coupling risk, for a method whose headline (stability at any dt) the compliance term already buys on the existing operator.
- *Kept on the table only if:* matching the published method's exact guarantees ever outweighs the bed-coupling cost — not foreseen.

**Rejected — Augment, don't replace (original Option C).**
Keep the `∇·v` projection for the bed `∇p` and add a separate PB-MPM-style density constraint purely for water stiffness.
- *Why rejected:* if the old projection remains the bed pressure, the bed reads the *wrong* pore pressure (water stiffness and bed pressure disagree); if the bed reads the sum, it collapses into the recommended approach with double-counting hazards. Two mechanisms also add cost against an already-tight pass/storage budget.

## Acceptance Examples

- AE1. **Poured cup fills, not rings/corners.** Pour into the V60 cup (web config), stop, settle. **Covers R1, R2, R3, R8.** Interior annuli carry mass, no hollow center, no packed corner; ρ/ρ_rest ≲ the R1 bar. A centered hollow ring or packed corner fails this example.
- AE2. **Standing pool does not creep.** A statically seeded cup pool run 1000+ steps holds ρ/ρ_rest near rest with no upward drift. **Covers R1, R4.**
- AE3. **Mixture still correct.** A saturated deformable bed under a center pour still craters and holds, with Terzaghi/Skempton, volume-conservation, no-fluidize, and buoyancy gates green, and `σ_total = σ' + u` closing. **Covers R6, R7.**
- AE4. **No detonation at target stiffness.** Raising stiffness/compliance to the level that satisfies R1–R3 keeps kinetic energy and density bounded. **Covers R4.**
- AE5. **Suction stays interior.** With the two-sided target on, the free surface does not rise and pocket λ does not re-crush a sub-floor air disc. **Covers R10.**

## Success Criteria

- Cup water ρ/ρ_rest: mean ≲ 1.2× and high-percentile bounded, **standing**, not just during pour (the historical ~5× mean / ~23× tail eliminated; the validated ~1.8× coarse over-pack pulled toward 1.0×).
- The interior-fill / radial-mass gate passes for a filled pool and **fails** for a centered hollow ring and a packed corner.
- The real-time gate (≤ 33 ms @ 200k) holds; all existing L0–L3 gates green; `σ_total = σ' + u` closes.
- Rigorous enough for multi-round Codex review to act on without re-deriving the diagnosis.

## Scope Boundaries

- **Corner-seam multi-normal BC (`WALL_BC_MULTI`) + the velocity-gated pour-pocket patch** — built this session behind the `dbg.y` selector (off by default). This fix targets the **bulk over-pack root**; the expectation is that fixing the density model relieves the corner-pack locus too, **retiring** those patches. Whether the corner audit (`tests/twofield_wall_audit.rs`, 2.4× → ~1.4× box parity) passes from the density fix alone is an **open question to verify**, not a guaranteed outcome — do not retire the selector until BOTH the corner gate (`twofield_wall_audit.rs`) and the isolation gate (R11) confirm it.
- **Settled-pool "stirring"** — convergence-insensitive in this session's validation, i.e. a **separate APIC/PIC particle-dynamics root**, not the over-pack model. Tracked as a follow-up; out of scope here beyond "must not regress."
- **Literal particle PB-MPM / per-particle J-tracking** — rejected (see Approaches). Out.
- **XPBD solver** — the referee and stability existence proof; unaffected. Out.
- **Two-field mixture-coupling redesign** — inert in the WaterOnly repro; a non-breaking constraint, not part of this fix. Out (beyond "must not regress").
- **Operator-consistency / wall-BC lockstep changes** — not required by this fix and not in scope (the pressure unknown and `gp` are unchanged).

## Dependencies / Assumptions

- Fixed-point i32 grid (no float atomics, FP_SCALE = 2^18), the per-entry-point 7-buffer ceiling (`cell_classify` and `project` already at 7), and Tint uniformity bound the implementation.
- **Confirmed (this review):** the compliance term + two-sided target fit in existing lanes with no new binding — `jacobi_fine` binds only 5 of the 7 allowed storage buffers and already binds `cell_meta`/`pf_*`/`nm`, so the `+α` diagonal is local scalar math, and the density error reuses the already-computed `ρ̄_c` census (`cell_classify`). The `+α` term touches the `A·p` evaluation in **all three** places — `residual`, `jacobi_fine`, AND `jacobi_coarse` — consistently, still with no new binding.
- **Assumption to confirm:** the existing interior suction path in `surface.wgsl` (gated away from air-adjacent cells) can be unified with the over-density relief into a single two-sided compliant target without leaking onto surface rows (R10).
- **Assumption (carry from 2026-06-16, to verify):** fixing the base water incompressibility may also relieve some settled-pool stirring; if not, stirring remains the separate follow-up above.

## Outstanding Questions

**Resolve before planning**
- None blocking. The mechanism is chosen (compliant predicted-density-error projection on the existing operator).

**Gate before committing the compliance value / merging behavior**
- **Finite-compliance stress-partition check.** Confirm the converged compliant pressure (solving `(A+C)·p = b`) still closes `σ_total = σ' + u` and preserves Terzaghi/Skempton action–reaction within gate tolerance at the chosen `C` — not only in the `C → 0` limit. Rerun BOTH the static hydrostatic partition audit (`twofield_full.rs:899–929`) AND the dynamic Skempton / load-sharing coverage (`twofield_full.rs:946–1036`) at the chosen `C`. This does **not** block writing the plan; it gates selecting/defaulting `C` and merging the behavior.

**Deferred to planning / Codex review**
- Exact two-sided density-error target form (and any dead-band), and how negative target = suction is expressed without destabilizing free-surface rows.
- The compliance value (α / diagonal weight) and its calibration vs. the real-time gate, R4, and the stress-partition audit.
- The interior-vs-surface row gate mechanism — extend `surface.wgsl`'s gating vs. a new interior-row predicate from existing lanes.
- The ρ/ρ_rest acceptance bar (working 1.2×) and the R8/R10 gate thresholds.
- Whether the corner audit passes from the density fix alone (→ retire `WALL_BC_MULTI` + pour-pocket patches) or needs a residual BC touch.

## Prior Art / Standard Approaches

Carried from `2026-06-16-twofield-water-incompressibility-requirements.md` and re-pointed at the chosen mechanism. The consistent finding: explicit stiffness is the wrong lever; the stable-stiff answers are **implicit or compliant**.

- **IISPH / DFSPH (implicit incompressible / divergence-free SPH)** — solve pressure from the **predicted density change** so the accelerations remove the density deviation. This is the chosen mechanism's lineage on the velocity/pressure side: the two-sided predicted-density-error target, on a grid instead of particles.
- **PB-MPM — Position-Based MPM (EA SEED, SIGGRAPH 2024, open WebGPU/WGSL, electronicarts/pbmpm)** — a semi-implicit **compliant-constraint** MPM, "stable at any time-step." The chosen mechanism's lineage on the stability side: the compliance diagonal. **Adopted in spirit (the compliant constraint), not in letter (the particle integration)** — see Approaches for why the literal port breaks the bed coupling.
- **Weakly-compressible EOS MPM/SPH (WCMPM/WCSPH)** — artificial sound speed ~10× max velocity to hold ~1% compression ⇒ tiny explicit timestep / detonation. This *is* the project's measured detonation, and the reason for compliance over explicit stiffness.
- **Incompressible MPM (operator-splitting projection)** — constrains `∇·v`, not density — the exact loophole the current projection has.
- **MLS-MPM volume control via J** — per-particle deformation-gradient volume restore. Not needed: the grid density census already measures density (decision above).
- **Position-Based Fluids (PBF, Macklin & Müller 2013)** — the XPBD referee's particle density constraint; the in-repo stability existence proof. Particle-side (compression-negative λ), so cited as proof, **not** portable to the grid pressure (sign trap, see Key Decisions).

## Sources / Research

- `src/solvers/twofield/pressure.wgsl` — the consistent corner-trilinear D / G = −Dᵀ operator (`:7–30`), `node_setup` M̃⁻¹ (`:269–384`), the one-sided density-relief rhs (`:438–495`, formula `:494`), the local-Jacobi sweep (`:735`), and the bed-coupling block where "the pore pressure IS the projection pressure" (`:853–928`, comment `:903–906`).
- `src/solvers/twofield/surface.wgsl` — the deferred air-neighbor-gated under-density suction path (`:175–201`).
- `src/solvers/twofield/coupling.wgsl` — `drag_fold` wall-projection basis (`:252–298`) that mirrors `node_setup` (the lockstep this fix does **not** touch).
- `src/solvers/twofield/mod.rs` — bind-group layouts confirming `cell_classify` and `project` at the 7-buffer ceiling (`:87–115`, `:1909`, `:2023`).
- `tests/twofield_full.rs:899–929` — the `σ_total = σ' + u` stress-partition audit (R7).
- `tests/twofield_cup.rs` — the existing cup gates (blind to a centered ring) and over-compression footer.
- `tests/twofield_wall_audit.rs` — the corner-pack audit (2.4× SDF cup vs 1.4× box) referenced in Scope Boundaries.
- `src/solvers/xpbd/coupling.wgsl:228`, `src/solvers/xpbd/water.wgsl:37–222` — PBF existence proof and the particle-side compression-negative λ convention (sign trap).
- This session: validation that the over-pack is model-bound (sweeps 8→64: 1.80→1.77×; relief-off: 2.86×); independent Codex (gpt-5.5, high) source review concurring (rank: hybrid > A-as-same-operator > C > B).
- `docs/brainstorms/2026-06-16-twofield-water-incompressibility-requirements.md` — superseded; this doc resolves its deferred A/B/C pick.
- `docs/plans/2026-06-09-001-feat-unified-twofield-solver-plan.md` — the active two-field program this work sits inside.
- External: PB-MPM paper https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-siggraph2024-pbmpm-paper.pdf , code https://github.com/electronicarts/pbmpm ; IISPH/DFSPH https://interactivecomputergraphics.github.io/physics-simulation/examples/iisph.html .

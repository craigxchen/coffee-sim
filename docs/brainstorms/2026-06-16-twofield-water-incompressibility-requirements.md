---
date: 2026-06-16
topic: twofield-water-incompressibility
---

# Two-field water incompressibility — requirements

## Summary

Give the two-field solver's **water** phase a stiff-but-stable, **density-based** incompressibility to replace or augment the velocity-divergence pressure projection, which structurally cannot constrain density. This doc captures three candidate mechanisms (grid EOS pressure, density-targeting projection, compliant grid constraint) with their trade-offs; the mechanism is selected at planning + Codex-review time, not here.

## Problem Frame

Water poured into the V60 cup collapses into a thin dense ring at the wall with a hollow center, instead of filling as a pool. It was first read as a wall/collision artifact, but two geometry fixes (a wall-tangent corner fillet, then an interior cavity inset) only **relocated the ring inward** — proving the wall is not the cause. It is also not a collocated-grid checkerboard mode (Rhie–Chow / MAC staggering was tried previously and rejected). The fluid is simply **too compressible**.

A throwaway density probe quantified it (cup water, ρ/ρ_rest where 1.0 = incompressible; rest = one particle per `particle_spacing³`):

- **During pour:** mean 1.9×, wall annulus 2.2×.
- **Settled (pour off + 200 steps):** mean **5.0×**, median 1.4×, p90 **23×**, max 24×, with **~27% of cup particles jammed into a thin shell at r ≥ cup radius**.
- It **worsens monotonically with time** (1.9× → 5.0× as it stands), not a transient.

Root mechanism (from `src/solvers/twofield/pressure.wgsl`): the pressure projection enforces **∇·v = 0**, which does not constrain density — a compacted, hydrostatically-wrong, *static* pool has zero velocity divergence ("a fully-compacted tank measures ZERO divergence"). The only thing resisting compaction is a **one-sided, sweep-coupled density-relief source** folded into the rhs (relieves over-density, gives no suction to under-dense regions). Its stiffness is capped by stability: raising it detonates — pushing the solve to 32 sweeps inflated the pool ~5× (Δh 1.29 vs 0.27 predicted), so the volume balance currently leans on under-convergence as accidental damping. Net effective bulk modulus is far below incompressible, and the jet + self-weight ratchet water into the wall with no restoring force.

The symptom reproduces in **WaterOnly** (single-phase, φ_s = 0), which localizes it to the **base water incompressibility** of the grid solve — the two-field mixture coupling (drag, pore pressure, solid volume) is inert here and is therefore a constraint the fix must not break, not the cause.

## Key Decisions

- **Constrain density, not velocity divergence.** The lever is *what* is constrained, not how well the divergence Poisson converges or whether the grid is staggered. A velocity-divergence constraint cannot see compaction; the fix must target ρ → ρ₀ directly.
- **Stiff-but-stable is a hard requirement.** Explicit stiffening already detonated. The winning mechanism must remain stable at incompressible stiffness (unconditionally stable, or with a clearly-bounded stability regime that fits the timestep), not merely "stiffer relief."
- **Capture three mechanisms; defer the pick.** The stiffness/stability/real-time/coexistence trade-offs need the deeper planning pass and multi-round Codex review to resolve. This doc scopes the *what* and the candidate space; ce-plan + review choose the *how*.
- **The fix is the base water solve.** Because the ring reproduces in WaterOnly, the work targets single-phase water incompressibility first; mixture coexistence is a non-breaking constraint validated by the existing grain gates.
- **New water pressure inherits the mixture role.** Whatever produces the bulk stiffness must also serve as the water pressure that feeds pore-pressure / buoyancy and coexists with two-field drag and the Drucker–Prager solid coupling.
- **Explicit stiffness is off the table; the field's answer is implicit/compliant.** Approaches A (explicit EOS) and C (compliant constraint) are the *same* density constraint applied explicitly vs. implicitly. Explicit EOS hits a sound-speed CFL limit (literature: artificial sound speed must be ~10× max fluid velocity to hold compression under ~1% ⇒ a very small timestep) — that *is* our measured detonation. The standard stable-stiff routes are implicit density-invariant pressure (IISPH/DFSPH) or compliant-constraint MPM (PB-MPM). See Prior Art below.

## Requirements

**Incompressibility behavior**

- R1. A standing water pool holds ρ/ρ_rest near rest and does **not** drift upward over prolonged settling (no monotonic compaction). Working acceptance bar: ρ/ρ_rest ≲ 1.2× even after long standing (revisable — see Outstanding Questions).
- R2. The density correction is **two-sided**: over-dense regions expand AND under-dense (jet-evacuated) regions draw fluid back, so an evacuated center refills rather than persisting as a hole.
- R3. Water poured into the cup fills as a coherent pool (interior carries mass; the free surface relaxes toward hydrostatic), not a thin wall ring.

**Stability & real-time budget**

- R4. The mechanism is stable at the stiffness required for R1–R3 — no detonation / runaway inflation when stiffness is raised to incompressible levels (the explicit-relief failure mode must not recur).
- R5. Fits the real-time budget: the R9 ≤ 33 ms @ 200k gate holds, on the fixed-point i32 grid (no float atomics), within the per-entry-point storage-buffer ceiling and Tint uniformity rules.

**Coexistence & preservation**

- R6. All existing twofield L0–L3 gates stay green: Terzaghi/Skempton consolidation & load-sharing, volume conservation, no-fluidize, crater persistence, face-velocity twin.
- R7. The new water pressure fills the existing mixture role — feeds pore-pressure / buoyancy and coexists with two-field drag and the Drucker–Prager solid phase; grain scenes (infiltration, bed, crater) remain correct even though the cup repro is single-phase.

**Validation**

- R8. A new success gate detects the ring failure **directly** — an interior-fill / radial-mass-profile metric on which a centered hollow ring FAILS. The current cup gate is blind (a centered ring passes `radial_rms ≥ 0.75·initial` and `centroid_r ≤ 0.4·R`) and must be replaced or supplemented for this symptom.
- R9. The density excess is measurable and bounded as a regression signal (e.g., the probe's ρ/ρ_rest distribution: mean and high-percentile), not just a visual.

## Candidate Approaches

All three are two-field-grid-native (not a port of the XPBD particle PBF, which would bypass the grid mixture). XPBD's PBF is cited only as the existence proof that a stiff density constraint *can* be stable.

**Approach A — Weakly-compressible EOS pressure in the grid update (PB-MPM style).**
Replace the iterative relief with a local per-node equation-of-state pressure, e.g. `p = k·((ρ_n/ρ₀)^γ − 1)` (Tait/Cole; directional, not final), applied as −∇p in the grid velocity update. `k` is the bulk modulus, set directly; two-sided by construction; no global solve. This pressure *is* the water pressure that feeds the mixture.
- *Pros:* most MPM-native; cheapest in the real-time budget; naturally becomes the mixture pressure; two-sided for free.
- *Cons / risk:* explicit EOS stiffness couples to the timestep via sound speed — too stiff detonates (the failure already observed). Likely needs an implicit EOS or substepping, plus `k`-vs-R9 calibration.
- *Best when:* an implicit/substepped EOS can hit incompressible stiffness inside the frame budget.

**Approach B — Density-targeting projection (close the ∇·v loophole).**
Keep the projection structure but retarget its constraint to the **accumulated density/volume error** rather than instantaneous ∇·v — track per-particle volume J (the solver already carries a deformation field for the solid phase; extend or mirror for water) and drive the solve toward ρ → ρ₀, two-sided.
- *Pros:* least-invasive relative to current code; "converge more" finally means "incompressible"; reuses the existing consistent D/G operator.
- *Cons / risk:* stays in the iterative-solve cost regime the design left the global Poisson to escape; J-tracking + drift correction is fiddly; convergence-vs-budget tension returns.
- *Best when:* the existing solver scaffold is worth preserving and J can be tracked cheaply for water.

**Approach C — Compliant (XPBD-style) density constraint on the grid (PB-MPM proper).**
Solve a grid density constraint as a compliant constraint with compliance α — unconditionally stable stiffness with **no sound-speed timestep limit** — and gather corrections back to velocity/position.
- *Pros:* guarantees R4 (stable at any stiffness); decouples stiffness from dt.
- *Cons / risk:* largest architectural change; mapping the compliant solve onto the fixed-point i32 grid and the mixture is the open work.
- *Best when:* explicit EOS (A) can't be made both stiff enough and stable in budget — the stability-guaranteed fallback.

*Recommendation carried into planning (sharpened by the Prior Art research below):* A and C are the same density constraint applied explicitly vs. implicitly, and explicit EOS hits the sound-speed CFL wall that *is* our detonation — so the viable routes are the implicit/compliant ones. **Lead recommendation: an implicit, two-sided density-corrected projection** — retarget the existing solve from `∇·v = 0` to driving the *predicted* node density → ρ₀ (drop the one-sided `max()`; allow negative target = suction). This is IISPH/DFSPH's density-invariant idea on our existing consistent operator, reuses the current Jacobi/multigrid solve (so it fits R5, unlike "converge harder"), and converging it now *means* ρ→ρ₀ (the prior "32 sweeps → 5× inflation" was converging a one-sided expansion source, not a two-sided restoring constraint). **PB-MPM** (compliant-constraint MPM, *same WebGPU/WGSL stack, open reference code*) is the strongest packaged alternative if the implicit-projection retarget proves unstable under the two-field mixture weighting. Final pick deferred to planning + Codex review.

## Acceptance Examples

- AE1. **Poured cup fills, not rings.** Pour into the V60 cup (web config), stop, settle. **Covers R1, R2, R3, R8.** Cup water forms a filled pool: interior annuli (r < ~2) carry mass, no hollow center, and ρ/ρ_rest ≲ the R1 bar. A centered hollow ring fails this example.
- AE2. **Standing pool does not creep.** A statically seeded cup pool run 1000+ steps holds ρ/ρ_rest near rest with no upward drift. **Covers R1, R4.**
- AE3. **Mixture still correct.** A saturated deformable bed under a center pour still craters and holds, with Terzaghi/Skempton, volume-conservation, and no-fluidize gates green. **Covers R6, R7.**
- AE4. **No detonation at target stiffness.** Raising stiffness to the level that satisfies R1–R3 keeps kinetic energy and density bounded (no runaway inflation). **Covers R4.**

## Success Criteria

- Cup water ρ/ρ_rest: mean ≲ 1.2× and high-percentile bounded, **standing**, not just during pour (the current ~5× mean / ~23× tail eliminated).
- The new interior-fill / radial-mass gate passes for a filled pool and **fails** for a centered hollow ring.
- R9 ≤ 33 ms @ 200k holds; all existing L0–L3 gates green.
- The result is rigorous enough for multi-round Codex review to act on without re-deriving the diagnosis.

## Scope Boundaries

- **Cup corner geometry / fillet / inset** — reverted; not the cause. Out.
- **XPBD solver** — the referee; unaffected. Out.
- **Two-field mixture-coupling redesign** — inert in the WaterOnly repro; a non-breaking constraint, not part of this fix. Out (beyond "must not regress").
- **Selecting THE mechanism** — deferred to planning + Codex review; this doc scopes the candidate space only.

## Dependencies / Assumptions

- Fixed-point i32 grid (no float atomics), per-entry-point storage-buffer ceiling, and Tint uniformity (barriers reachable from uniform control flow) bound all three approaches.
- Approach B assumes a per-particle volume / deformation measure (J) can be tracked for water (the solid phase already has a deformation field to mirror or extend).
- The new water pressure is assumed to be able to assume the current projection's mixture role (pore-pressure / buoyancy feed) — to be confirmed against `coupling.wgsl` during planning.
- **Assumption to verify:** fixing the base water incompressibility also relieves the settled-pool "stirring" (same one-sided-relief + under-convergence family); if not, stirring remains a separate item.

## Outstanding Questions

**Resolve before planning**
- None blocking. The mechanism choice is intentionally deferred (below).

**Deferred to planning / Codex review**
- Which mechanism (A / B / C), and whether it **replaces** the local-Jacobi projection (A, C) or **augments** it (B).
- Exact target stiffness and the ρ/ρ_rest acceptance bar (working value 1.2×) and gate thresholds.
- For A: implicit EOS vs substepping, and `k`/γ vs sound-speed/R9 calibration.
- For B: whether water J can be tracked cheaply, and the drift-correction form.
- For C: the compliant-solve mapping onto the fixed-point grid and the mixture.

## Prior Art / Standard Approaches

Web research (2026-06-16) on how the field keeps an MPM/particle fluid incompressible without a detonating stiffness. The families map onto the candidates above; the consistent finding is that **explicit stiffness is the wrong lever — the stable-stiff answers are implicit or compliant.**

- **Weakly-compressible EOS MPM/SPH (WCMPM/WCSPH)** — pressure from an EOS on density. To hold density fluctuation under ~1% the artificial sound speed must be ~10× the max fluid velocity, forcing a very small explicit timestep and causing pressure oscillation. This is Approach A and exactly our measured detonation (the CFL/sound-speed wall). Cheap, GPU-friendly, but cannot reach incompressible at a 1/60 s step.
- **Incompressible MPM (iMPM), operator-splitting projection** — Chorin splitting: predict velocity, then a pressure-Poisson correction to a divergence-free field. Large timesteps but an elliptic solve, and it constrains **∇·v, not density** — the exact loophole our current projection has. The literature confirms divergence-projection alone doesn't pin density.
- **IISPH / DFSPH (implicit incompressible / divergence-free SPH)** — solve pressure from the **predicted density change** so the resulting accelerations remove the density deviation (IISPH: density-invariant pressure-Poisson; DFSPH: density-invariant to ~0.01% *and* divergence-free). Implicit ⇒ large stable timesteps. This is precisely the implicit density-invariant mechanism in our lead recommendation, on a grid instead of particles.
- **PB-MPM — Position-Based MPM (EA SEED, SIGGRAPH 2024)** — a semi-implicit **compliant-constraint** MPM, "stable at any time-step while as easy to implement as an explicit integrator," demonstrated on fluid + elastic + rigid. **Open-source, WebGPU/WGSL — this project's exact stack** — so Approach C comes with studyable reference code (electronicarts/pbmpm, references nialltl's incremental_mpm).
- **MLS-MPM volume control via J** — track the deformation-gradient determinant J (mass conservation ρJ = ρ₀) and apply a volume-preserving correction that restores J→1 when the scheme drifts. This is Approach B's J-tracking with an established form.
- **Position-Based Fluids (PBF, Macklin & Müller 2013)** — particle-side ∑W→ρ₀ density constraint via position deltas + s_corr anti-clustering. The XPBD referee's method and our stability existence proof; particle-side, so it bypasses the grid mixture (cited as proof, not directly portable).

**Takeaway for the pick:** the two real-time-credible, stable-stiff routes are (a) **compliant-constraint MPM (PB-MPM)** — same WebGPU stack, open reference code — and (b) **implicit density-invariant pressure (IISPH/DFSPH-style)** folded into our existing projection (the lead recommendation). Both confirm the reframe: do **not** chase explicit EOS stiffness.

## Sources / Research

- `src/solvers/twofield/pressure.wgsl` — the consistent corner-trilinear D / G = −Dᵀ operator (post-projection divergence = solver residual), the one-sided density-relief rhs source, and the "fully-compacted tank measures ZERO divergence" note that defines the loophole.
- `tests/twofield_cup.rs` — the existing cup gates (blind to a centered ring: `radial_rms ≥ 0.75·initial`, `centroid_r ≤ 0.4·R`) and the documented over-compression footer.
- `src/solvers/xpbd/coupling.wgsl` — PBF existence proof (density-constraint `lambda`, poly6/spiky kernels, `rest_density`, `s_corr`): a stiff density constraint that fills the cup.
- `docs/plans/2026-06-09-001-feat-unified-twofield-solver-plan.md` — the active two-field program this work sits inside.
- Density-probe measurements (this session): pour mean 1.9× / wall 2.2×; settled mean 5.0× / median 1.4× / p90 23× / max 24×; ~27% at r ≥ cup radius; monotonic worsening.
- External prior art (web, 2026-06-16):
  - PB-MPM, "A Position Based Material Point Method", Lewin / EA SEED, SIGGRAPH 2024 — paper https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-siggraph2024-pbmpm-paper.pdf , overview https://www.ea.com/seed/news/siggraph2024-pbmpm , open-source WebGPU code https://github.com/electronicarts/pbmpm (Approach C; same stack).
  - IISPH "Implicit Incompressible SPH" (Ihmsen et al. 2014) and DFSPH "Divergence-Free SPH" (Bender & Koschier 2015) — implicit density-invariant pressure (the lead recommendation's mechanism). https://interactivecomputergraphics.github.io/physics-simulation/examples/iisph.html
  - Incompressible MPM for free-surface flow (operator-splitting projection) — https://www.sciencedirect.com/science/article/abs/pii/S0021999116305721 (the divergence-projection family = our current approach + its loophole).
  - Weakly-compressible EOS MPM/SPH sound-speed/CFL tradeoff (Approach A's detonation, quantified) — variable-sound-speed WCSPH https://arxiv.org/pdf/2310.04139 .
  - MLS-MPM volume control via J (Approach B's J-tracking); Position-Based Fluids, Macklin & Müller 2013 (PBF; the XPBD referee's method, stability existence proof).

---
date: 2026-06-05
topic: physics-validation-test-suite
focus: each test isolates ONE phenomenon vs a math-derived/closed-form expectation; no test bloat
mode: repo-grounded
---

# Ideation: Physics-Validation Test Suite

Goal: a focused suite where every test isolates one physical phenomenon and compares the simulation's output to a math-derived / closed-form expectation with a quantitative acceptance criterion. No test bloat — every test earns its place. 45 raw candidates across 6 ideation frames → deduped → adversarially filtered to 8 survivors. Convergence was very high: free-fall, the extraction exponential, and Darcy-d² were each proposed independently by all 6 frames.

## Grounding Context (Codebase)

- Active solver: single unified XPBD/PBF GPU solver at `src/solvers/xpbd/` (mod.rs + 6 WGSL). Old MPM crate is superseded.
- Integration: symplectic-Euler predict `pred = pos + dt·v + dt²·g` (full `dt²·g`, not ½), iterative constraint projection, velocity recovery `v=(pred−pos)/dt·damping`.
- Observables readable on CPU (pub methods on `XpbdSolver`, mod.rs): `read_positions/velocities/phases/moisture/concentration/temperature/chem`, `metrics()` {extraction_yield, tds}, `diagnostics()` {overflow, residual, effective_iters}, `total_emitted_water_mass()`, `water_particle_volume()`. Seeding helpers: `write_velocities/chem/moisture/lambdas_for_test`.
- NOT exposed: per-particle density ρ, λ (write-only), α_s, dp. Pressure/density must be reconstructed on CPU from positions (`density_in_band` in `tests/xpbd_water.rs:49` sums m·W_poly6).
- Harness: `GpuContext::new_headless()` (skips if no adapter) → build Scene+Materials+Config → optional `write_*_for_test` IC → loop `solver.step(1/60, &EmissionInput::default())` → `sample_diagnostics()` → read. CPU math mirrors in `utils/kernels.rs` and `models/*.rs` are exact twins of the GPU kernels — use them to compute expectations.
- Config gates default OFF (a test must enable the phenomenon it isolates): `absorb_rate`, `extract_rate`, `drag_subiters`, `buoyancy_scale`. Scene gravity is a reduced sim value (default `[0,-20,0]`), not SI.
- Existing tests lock invariants/bounds (density-in-band 80%, settles, no clump, momentum conserved, enthalpy conserved, ordinal "coarse drains faster") — none compare against a math-derived value.

Grounding corrections verified in-repo (4 of 6 frames):
- Permeability wired is `kozeny_carman` (mod.rs:1004), NOT `li_li` (stale note); d² test is unambiguous.
- predict applies full `dt²·g` → free-fall reference is the discrete recurrence, not continuous ½gt².
- λ is write-only; direct hydrostatic pressure needs a new `read_lambdas()`; the g-ratio variant avoids it.
- Thermal/extraction only run in a MIXED scene with `extract_rate>0` → cooling test needs a non-interacting far-corner grain.

Doctrine (AGENTS.md): assert invariants/LAWS, not tuned target values; never `#[ignore]` to go green; structural fixes over tuning patches. Conservation-as-gate template exists (`tests/xpbd_extraction.rs`, `xpbd_wetting.rs`): isolate the pass, effective-mass momentum, invariance to float tolerance, failing-first. Fine-spacing eruption lesson (93eb9ba): parametrize by spacing.

## Topic Axes
A1. Water-phase fluid mechanics — hydrostatics, incompressibility, free-surface, viscous profiles, settling
A2. Granular bed & porous-media coupling — Darcy/Kozeny-Carman, terminal velocity/drag, buoyancy, repose
A3. Transport & reaction — extraction kinetics, thermal mixing, cooling, absorption/swelling, TDS/yield
A4. Conservation laws (cross-cutting) — mass/volume, momentum, solute, energy/enthalpy
A5. Numerical correctness & methodology — spacing-invariance/convergence, observability gaps, determinism, acceptance-criterion design

## Foundational (not a ranked test — the anti-bloat spine)
A small `tests/support/analytic.rs`: one tolerance table keyed by phenomenon (external benchmark numbers) + a reusable `reconstruct_density(pos,h,m)` lifted from `density_in_band`. The first hydrostatic/rest-density test pays for it; every later analytic test becomes ~10 lines. Recommended before writing tests #2/#3.

## Ranked Ideas

### 1. Free-fall ballistic recurrence (integrator exactness)
**Description:** One isolated particle (no neighbors → every constraint pass is a no-op); step N frames; compare to the solver's own discrete semi-implicit-Euler recurrence.
**Axis:** A5
**Basis:** `direct:` predict `np = p + dt·v + dt²·g` (`common.wgsl:410`), recovery `v=(pred−pos)/dt·damping` (damping=1.0 default `config.rs:146`), `read_positions/velocities` (`mod.rs:549,554`). [ALL 6 frames]
**Equation/criterion:** `v_n=v₀+n·dt·g`, `x_n=x₀+n·dt·v₀+dt²·g·n(n+1)/2`. Pass: `|Δx|<1e-4`, `|Δv|<1e-5` over 120 steps; explicitly reject the continuous ½gt² form; invariant across substeps∈{1,2,4}.
**Rationale:** Every downstream physics test rides on the integrator being exact for the unconstrained case; a stray ½ factor or damping leak silently biases everything. Cheapest, sharpest, impossible-to-fake anchor.
**Downsides:** Tests numerics, not "coffee."
**Confidence:** 95%  **Complexity:** Low  **Status:** Explored

### 2. Rest-lattice null + no spurious drift (spacing-parametrized)
**Description:** Seed water on the exact lattice that defines ρ₀, g=0; reconstruct density on CPU; `C=ρ/ρ₀−1` must vanish by construction and the block must not twitch.
**Axis:** A5 (cross-cuts A1)
**Basis:** `direct:` `rest_density(spacing,h,m)` (`kernels.rs:46`), `density_in_band` (`xpbd_water.rs:49`). `external:` still-water `|v_spurious|<1e-3·√(gH)`, drift<0.5%/1000 steps. [4 frames]
**Equation/criterion:** interior `|C|<2%`; `max|v|<1e-3√(gH)`, centroid drift<0.5%·H. Run at spacing {0.8,1.0,1.2} (ρ₀ recomputed) → worst-case residual must not grow with refinement.
**Rationale:** Encodes the eruption-bug lesson (93eb9ba): a default-spacing-only test missed a vmax-10⁵ kernel-normalization bug. Closes the gap that water has no equilibrium test while the bed does.
**Downsides:** Needs CPU density reconstruction (shared with #3) + spacing parametrization.
**Confidence:** 88%  **Complexity:** Medium  **Status:** Unexplored

### 3. Hydrostatic pressure–depth, g-ratio law
**Description:** Settle a column at g∈{0.5,1,2}×; the reconstructed over-density gradient `dρ/dz` must scale linearly with g — a ratio law that cancels the unknown PBF stiffness constant.
**Axis:** A1
**Basis:** `direct:` CPU density reconstruction; `Scene.gravity` free field (`scene.rs:58`). `external:` hydrostatic L2<1%. g-ratio is the constraint-flip refinement that sidesteps the observability gap. [5 frames]
**Equation/criterion:** `dP/dz=−ρg` ⇒ slope ratio `m(2g)/m(g)=2±5%`, `m(0.5g)/m(g)=0.5±5%`, all negative, R²>0.9; settled `|v|<1e-2√(gH)`.
**Rationale:** "Show me P=ρgh" is a reviewer's first question for any incompressible solver — the suite can't answer it today. Optional sharper variant: expose `read_lambdas()` for a direct PBF-pressure profile (also unlocks buoyancy checks).
**Downsides:** Free-surface bins excluded; absolute P=ρgh needs the λ readback.
**Confidence:** 85%  **Complexity:** Medium  **Status:** Unexplored

### 4. First-order extraction exponential
**Description:** Mixed grain+water pair, `extract_rate>0`, hold concentration ≪ c_sat so driving≈1 and k_eff is a known constant; fit the grain pool decay.
**Axis:** A3
**Basis:** `direct:` `release(pool,k,dt)=pool·(1−e^{−k·dt})` (`extraction.rs:72`, exact GPU twin); k_eff decomposition in `xpbd_extraction.rs:488`; `read_chem()`. `external:` first-order within 5% RMSE. [ALL 6 frames]
**Equation/criterion:** `pool(t)=pool₀·e^{−k_eff·t}`. Fit log(s_f) vs t: slope within 5% of CPU-twin k_eff, R²>0.99, RMSE<5% of pool₀.
**Rationale:** Conservation is tested; the kinetic shape (which makes early vs late brew differ and underlies the open yield/TDS calibration) is not. A mis-scaled k_eff conserves mass yet gives the wrong curve.
**Downsides:** Must pin T/flux/saturation so k_eff stays constant.
**Confidence:** 90%  **Complexity:** Medium  **Status:** Unexplored

### 5. Capacity-weighted mixing equilibrium VALUE
**Description:** Hot water + cold grain with unequal heat capacities, ambient off; assert the system settles at the capacity-weighted mean, not just that enthalpy is conserved.
**Axis:** A3 (cross-cuts A4)
**Basis:** `direct:` `pair_heat` capacity-weighted cap (`thermal.rs:20`), equilibrium twin `t_eq=(cᵢtᵢ+cⱼtⱼ)/(cᵢ+cⱼ)` in its unit test; `read_temperature()` (`mod.rs:937`). `external:` mixing within 1°C. [5 frames]
**Equation/criterion:** `T_eq=(ΣCᵢTᵢ)/(ΣCᵢ)`, `Cᵢ=mᵢcₚᵢ`. Pass: `|T−T_eq|<1°C-equiv`, enthalpy drift<0.5%, no overshoot. Extreme-ratio variant (C_g=50·C_w) pins T_eq to the reservoir.
**Rationale:** A capacity-blind average conserves energy only when C_w=C_g and lands at the arithmetic mean for real unequal capacities — silently mis-driving Arrhenius extraction. Conservation alone misses this.
**Downsides:** Needs co-located two-species seeding via `write_chem_for_test` in a mixed scene.
**Confidence:** 90%  **Complexity:** Low-Medium  **Status:** Unexplored

### 6. Drag relaxation + Kozeny-Carman d² scaling
**Description:** The grind→flow knob, three complementary forms: (a) CPU-pure resolved `γ(d)∝d^−2` (ratio 4.000±0.5% for 2× grind, no GPU); (b) single-grain velocity relaxation `v(t)=v₀(1−β)^n` (most surgical); (c) full-bed `q∝d²` (macroscopic Darcy).
**Axis:** A2
**Basis:** `direct:` `kozeny_carman(d,φ)=d²φ³/180(1−φ)²` wired at `mod.rs:1004`; implicit β=γdt/(1+γdt) (`coupling.wgsl:143`). `external:` Darcy R²>0.99, K-C within 10% for Re_p<1. [ALL 6 frames]
**Equation/criterion:** k∝d² ⇒ γ∝d^−2 / q∝d². Pass: log-log slope 2.0±0.1–0.3, R²>0.99. Caveat: drag is a velocity-blend, not Stokes force — keep γdt≪1 so β≈γdt; verify Re_p<1.
**Rationale:** Upgrades the only-ordinal "coarse drains faster" to a falsifiable exponent and locks the (now-unambiguous) permeability wiring against a silent model swap.
**Downsides:** Full-bed form (c) has a β-saturation caveat; (a)+(b) are the safe, sharp core.
**Confidence:** 82%  **Complexity:** Medium-High  **Status:** Unexplored

### 7. Volume-conserving absorption ledger + swelling
**Description:** `absorb_rate>0`; assert the project's flagged hard constraint (water volume lost == solid volume gained) plus cube-root swelling geometry and the saturation curve.
**Axis:** A4
**Basis:** `direct:` `effective_volume=V_dry+V_abs` (`wetting.rs:33`), `d_eff=d_dry·(V_eff/V_dry)^{1/3}` (`wetting.rs:40`), `absorb_demand=(V_cap−V_abs)(1−e^{−k·dt})` (`wetting.rs:67`); `read_moisture()` (`mod.rs:560`). Conservation-gate template exists. [3 frames]
**Equation/criterion:** total volume invariant<0.1%; `V_eff−V_dry==V_abs` (1e-5); `d_eff` cube-root (1e-4); `V_abs(t)=V_cap(1−e^{−k·t})` within 5%.
**Rationale:** Encodes the explicitly-critical volume-conservation invariant (a leak shows as drift; impossible to fake) and reuses the existing gate template, so it's near-free.
**Downsides:** Overlaps the existing monotonic-wetting test — this adds the exact ledger + geometry it lacks.
**Confidence:** 88%  **Complexity:** Low-Medium  **Status:** Unexplored

### 8. Newton cooling geometric decay
**Description:** Isolated water particle, ambient-loss only, must follow the exact geometric decay toward ambient. Gating wrinkle: thermal runs only in a mixed scene → seed a far-corner grain >h away so pair-heat is provably zero.
**Axis:** A3
**Basis:** `direct:` `ambient_delta=−clamp(h·dt,0,1)·(T−T_amb)` (`thermal.rs:34`); `read_temperature()`. `external:` Newton cooling curve. [4 frames]
**Equation/criterion:** `T_n=T_amb+(T₀−T_amb)(1−h·dt)^n`. Pass: pointwise <1e-4, recovered rate within 5%, no overshoot below T_amb.
**Rationale:** Cooling rate sets late-brew temperature (→ Arrhenius → yield); the existing test only checks "cools toward ambient" (an inequality).
**Downsides:** Simpler operator, less subtle failure than #5; far-corner-grain workaround. Lowest-value of the eight — folds into the thermal cluster if trimming to 7.
**Confidence:** 78%  **Complexity:** Low  **Status:** Unexplored

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Analytic-reference harness | Not a phenomenon test — surfaced as the foundational spine instead |
| 2 | Spacing-sweep macro | Folded into #2 + cross-cutting practice on #3/#6/#7 |
| 3 | Incompressibility residual vs iteration count | Convergence-monotonicity, not a closed-form-value comparison; weak fit to the "compare to an equation" bar; noted as future methodology |
| 4 | Zero-g two-particle momentum (Σmv=0) | Overlaps existing coupling momentum tests; #2 catches self-propulsion; equation is trivially zero |
| 5 | extract_rate→0 byte-exact gate | Gate-leak regression (partially exists `xpbd_extraction.rs:177`), not phenomenon-vs-equation |

Axis coverage: A1 (#3), A2 (#6), A3 (#4,#5,#8), A4 (#5,#7), A5 (#1,#2) — all five axes covered, no gaps.

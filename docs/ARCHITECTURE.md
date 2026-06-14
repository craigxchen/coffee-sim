# coffeesim — Main Architecture

**Status:** authoritative top-level reference. Stack, repo structure, core abstractions, and cross-cutting decisions live here and change rarely. **Sub-module internals — solver physics, UI, shared models — live in their own plans** (indexed in §7); those change often. The detailed physics worked out in the v2–v8 design history (incompressibility constraint, interphase forces, extraction kinetics, etc.) has moved *out* of this document into the relevant sub-module plans.

---

## 1. What we're building
An interactive, real-time pour-over coffee simulator: water poured through a deformable coffee bed in a dripper, with believable flow, bed deformation, and a coffee-extraction model whose causal levers (grind, temperature, ratio, pour technique, freshness, bed evenness) drive the right outcomes (yield, strength, drawdown, channeling). The bar is **perceptual + causal fidelity within tolerance — no glaring artifacts** — not quantitative CFD. The repo is organized as a **modular library of complete coffee+water solvers** so methods can be implemented and compared through one seam.

---

## 2. Stack & platform

**Rust + `wgpu` + WGSL.** One WGSL codebase compiles to:
- **native desktop** (Metal / Vulkan / DX12) — the development + local-profiling target now,
- **native mobile** (iOS Metal / Android Vulkan) — full performance + device reach,
- **web** (WASM + WebGPU) — instant access on modern browsers.

Rationale: it is the only single-codebase path that reaches desktop, mobile, and web for GPU compute; no framework lock-in; and Firefox's WebGPU *is* `wgpu`, so native and web behavior align closely. Frameworks we rejected for this (Taichi/MLX/Warp/Genesis) have no mobile-deployment story.

**Deployment (deferred).** Eventually: a web build for instant reach on modern devices plus native mobile apps for the long tail and full performance. **Device-scaling** (adaptive particle count / grid resolution / solver-iteration budget, detected from device limits) is a known design property to add later — explicitly out of scope until the sim is in good shape.

---

## 3. Cross-cutting decisions (settled; the things sub-module plans must respect)

1. **Complete-solver modularity.** Each entry in `solvers/` is a *complete* coffee+water method (XPBD, MPM, SPH+MPM) that owns its **internal** coupling. There is **no generic cross-solver coupler** — coupling is method-intrinsic (XPBD couples via constraints, MPM via the shared grid, SPH+MPM via drag forces), so a generic coupler would be a leaky abstraction.
2. **Shared physics models.** A `models/` library (permeability, extraction kinetics, cohesion, wetting, thermal) is imported by *every* solver, so cross-solver comparison is **fair**: same physics, different numerics. This is the shared layer that replaces the coupler.
3. **One active solver per run.** The simulator drives a single selected solver. "Multiple solvers" means a **swappable library + comparison harness**, not multiple solvers running at once. This is **build-time modularity with a single shipped primary (XPBD)** — not runtime failover or redundant hedging.
4. **Profiler as referee.** Method comparisons (e.g. XPBD vs MPM cost) are settled by measurement, not priors.
5. **Profiling scope = local only, for now.** `wgpu` native + `timestamp-query` wrapped per pass, read in our own loop, plus a **dispatches-per-frame counter** (the variable that predicts the eventual browser gap). Instruments only for deep single-kernel work. Browser profiling is deferred until the sim is in good shape — local profiling is a sound proxy for kernel cost and method ranking, which is what this phase needs.
6. **Fidelity bar.** Believable + causally faithful, outcomes in realistic ranges. Not a validated CFD/mass-transfer model.

---

## 4. Repo structure

```
coffee-sim/                         # repo root = workspace + package "coffee-sim"
├── Cargo.toml                      # workspace + package manifest
├── KEEP.md                         # salvaged reference values from the v1 MPM codebase
├── assets/                         # data ONLY: calibration tables, presets (dripper SDFs are analytic, in geometry/)
├── examples/                       # runnable scenarios (phase0_noop, v60_pourover, compare_solvers, …)
├── docs/
│   ├── ARCHITECTURE.md             # ← this document
│   └── plans/                      # ← sub-module plans (see §7)
└── src/
    ├── lib.rs
    ├── engine/                     # orchestration glue (wires the blocks; owns no physics)
    │   ├── scene.rs                #   build a brew (dripper + dose + water + pour schedule)
    │   ├── simulator.rs            #   drive the ONE active solver; frame loop; emit state/metrics/profile
    │   ├── state.rs                #   canonical State snapshot (+ Metrics) for ui + profiling
    │   ├── registry.rs             #   SolverId + build_solver + the solver catalog loader
    │   └── solvers.json            #   ★ solver-description catalog (data, NOT trait code)
    ├── solvers/                    # ★ modular library — each a COMPLETE coffee+water method
    │   ├── base.rs                 #   Solver trait (the seam) + SolverInfo/Paradigm/Stability
    │   ├── noop.rs                 #   harness validators (noop_a / noop_b)
    │   ├── xpbd/                   #   ← PRIMARY (shipped)
    │   ├── mpm/                    #   ← Tampubolon mixture (fidelity reference)
    │   └── sph_mpm/                #   ← DFSPH water + MPM bed
    ├── models/                     # ★ SHARED physics + materials (imported by every solver; NOT a coupler)
    │   └── permeability · extraction · cohesion · wetting · thermal · Materials
    ├── emission/                   # ★ particle emission (coffee + water), independently testable
    ├── ui/                         # rendering + UI + debug overlays (consumes canonical state only)
    ├── profiling/                  # timestamp wrapper + dispatches-per-frame counter (local; simple)
    └── utils/                      # GpuContext, shared spatial hash, SDF, kernels, RNG, geometry builders, config, ParticleBuffers
```

The five modular work-blocks are `solvers/`, `emission/`, `ui/`, `profiling/`, and
`utils/`(+config); `engine/` is thin orchestration glue. `materials`/`geometry`/`options`
are folded into `models`/`utils` rather than standalone modules.

---

## 5. Core abstractions (top-level only; implementations live in sub-module plans)

### 5.1 The seam — `Solver` (solvers/base.rs)
Every complete method implements one interface, so `engine`, `ui`, and `profiling` drive any solver uniformly. `step()` advances a **whole frame** — water, bed, internal coupling, extraction all happen inside, however that method prefers.

Solver **descriptions are data, not trait code**: each solver's `SolverInfo` lives in the `engine/solvers.json` catalog (deserialized with serde) and is looked up by `SolverId`. Adding or amending a solver is a one-row data edit, so the trait carries no `descriptor()`.

```rust
// Deserialized from engine/solvers.json, keyed by SolverId.
pub struct SolverInfo {
    pub name: String,
    pub paradigm: Paradigm,     // PositionBased | ForceBased | Hybrid
    pub owns_grid: bool,
    pub stability: Stability,   // Unconditional | CflLimited { c }
}

pub trait Solver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self
        where Self: Sized;
    fn reset(&mut self, scene: &Scene);
    fn step(&mut self, dt: f32, input: &EmissionInput); // one frame; internal substepping/coupling/extraction
    fn particles(&self) -> ParticleBuffers;             // canonical particle buffers for rendering
    fn metrics(&self) -> Metrics;                       // yield, TDS, drawdown, evenness
    fn profile(&self) -> Profile;                       // per-pass GPU timestamps + dispatches-per-frame
}
```

### 5.2 Orchestration — `engine/`
- **Scene** — user-facing brew builder.
- **Simulator** — holds **one** `Box<dyn Solver>`; runs the frame loop; feeds `EmissionInput` from `ui`/`emission`; pumps `particles()`/`metrics()`/`profile()` out to `ui` and `profiling`. Re-`build`s a different method (by `SolverId`) on the same `Scene` for the runtime solver-switch.
- **State** — canonical snapshot decoupling solvers from rendering.

### 5.3 Data flow
`options` → `engine::Scene` → `Simulator` builds the active solver (from a registry keyed by config) → per frame: `vis` input → `solver.step` → `state`/`metrics`/`profile` → `vis` render + `profiler`. The shared spatial hash and `GpuContext` (`utils/`) are infrastructure used by solvers and the engine, owned by neither.

---

## 6. Build order (repo level; details in sub-module plans)

- **Phase 0 — scaffold.** Cargo workspace, `wgpu` GpuContext, the `CoffeeSolver` trait, a no-op solver, the timestamp+dispatch profiler harness. *Gate:* no-op solver steps and profiles at 60 fps locally; runtime-switch swaps it.
- **Phase 1+** — XPBD water core → bed → internal coupling + extraction → vis/scorecard → alternate solvers for comparison. Each phase's gate and internals live in the relevant sub-module plan.

The primary (XPBD) ships after the core loop validates; alternate solvers are added because the seam makes them cheap, and kept only if the profiler/validation says they earn it.

---

## 7. Sub-module plans (to follow)

Each becomes its own doc under `docs/plans/`. Proposed order reflects critical path (the XPBD solver carries the real design risk and the physics we've already worked out).

| Plan | Scope | Carries (from design history) |
|---|---|---|
| `solvers.md` | The seam in depth: `CoffeeSolver` contract, `SolverDescriptor`, the registry, the substep/scheduler conventions (multi-rate, frozen-bed fast path, KE watchdog), shared-buffer layout rules. | build-time-modularity principle; substep scheduling |
| `solver_xpbd.md` | **Primary.** Internal structure: water incompressibility (**density constraint + `s_corr` anti-clustering** — Patch A.1), bed (rest-shape + friction + cohesion, freeze-when-static), **internal coupling** (unilateral grain exclusion A.2 + the four interphase forces), wetting→extraction pipeline. | v6 §2–4, v7 Patch A, GIC-derived volume-fraction + interphase forces |
| `solver_mpm.md` | Tampubolon two-grid mixture as the **fidelity reference**: APIC transfer, Drucker-Prager sand, weakly-compressible/projected water, implicit two-grid drag. | the MPM-mixture analysis |
| `2026-06-09-001-feat-unified-twofield-solver-plan.md` | **Two-field multiphase** (`solvers/twofield/`): two velocity fields per grid node (water + solid), APIC transfers, Poisson-free incompressibility (coarse-grid pressure seed + fine local Jacobi on one MAC-consistent operator family), Klar Drucker-Prager elastoplastic bed with compaction-cap tamping memory, semi-implicit Laibe–Price exponential Darcy drag, mixture projection, GIC-style conserved wetting/swelling, and constraint-bubble pour cavities. Built as a validation ladder L0→L3 (water → bed → percolation → full coupling); all four physics rungs pass (8 green test suites). **The R9 real-time gate HALTS:** the 200k V60 saturated-pour scene runs ~68–180 ms/frame vs the 33 ms target — it scales ~linearly (1.05× µs/Kpart 40k→200k) but the constant is ~2–5× over budget, dominated (~80%) by the single-workgroup constraint-bubble row solve (see `PERF_NOTES.md`). NO-FALLBACK: a failing-but-honest gate, not a tuned pass. | the failed-attempts review + PB-MPM/Klar/Laibe–Price/GIC methods |
| `solver_sph_mpm.md` | DFSPH water + MPM bed; internal particle↔grid drag + buoyancy. | the SPH+MPM analysis |
| `models.md` | **Shared physics** imported by all solvers: Kozeny-Carman permeability, two-pool extraction kinetics (Arrhenius, flux-driven), saturation→cohesion curve, wetting/absorption, thermal exchange. Owns the calibration tables that make cross-solver comparison fair. | extraction kinetics, cohesion curve, permeability |
| `engine.md` | Scene/Simulator/State internals; registry; runtime solver-switch. | — |
| `vis.md` | Render (screen-space fluid, concentration→color, cross-section), brew scorecard, interaction controls, debug overlays. | — |
| `profiler.md` | Local profiler: `timestamp-query` wrapper, dispatches-per-frame counter, the `examples/compare_solvers` harness. (Simple now; browser profiling deferred.) | profiler-as-referee |
| `utils.md` | GpuContext, shared spatial hash, SDF helpers, kernels, seeded RNG. | — |

`materials/`, `geometry/`, and `options/` are light enough to fold into the plans that use them (or get short plans on demand).

---

## 8. Relationship to prior docs
This document supersedes the **architecture-level** content of v2–v8. The **physics-level** content is superseded by the sub-module plans in §7 (chiefly `solver_xpbd.md` and `models.md`). The v-numbered docs remain as design history and rationale.

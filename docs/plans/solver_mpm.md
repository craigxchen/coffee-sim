# coffeesim — `solver_mpm` Plan (fidelity reference)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/solvers/mpm/` · **Status:** plan (lower priority — built after the primary, for comparison)
**Depends on:** `models/`, `utils/` (GpuContext, kernels, SDF), `solvers/base.rs`

## Purpose
A continuum two-grid MPM **sand-water mixture** (Tampubolon et al. 2017), the rigorous-physics yardstick to A/B the shipped XPBD solver against via `examples/compare_solvers`. **Not a shipping candidate** — it is offline-leaning by design (its stiff momentum exchange forces an implicit/expensive solve). Its value is being the high-fidelity reference, run on the **same `models/`** so the comparison isolates the numerical method.

`SolverDescriptor`: `ForceBased`, `owns_grid: true`, `CflLimited`.

## Scope
**In:** APIC transfers, the two-velocity-field mixture, Drucker-Prager sand, the fluid grid, the internal two-grid drag coupling, extraction read-out. **Out:** the shared physics (`models.md`); the seam.

## Internal design
- **Background grid** (`utils` grid; not shared with any other solver's physics — internal). APIC P2G/G2P transfers.
- **Two velocity fields** per node — sand `v_s` and water `v_w` — so the **relative** velocity (Darcy flux, the extraction signal) is represented.
- **Sand**: Drucker-Prager elastoplastic return mapping (Hencky space); cohesion from `models/cohesion` (saturation-dependent).
- **Water**: weakly-compressible (or projected) fluid grid; incompressibility via the grid.
- **Internal coupling**: momentum exchange (drag) between the two grids, parameterized by `models/permeability`. **Stiff** for fine media → handled **implicitly** (the reason this method is expensive; explicit would force ~10⁻⁶ s steps). This implicit per-step solve is the offline cost; it gets worse for fine grinds (high drag → ill-conditioned).
- **Extraction**: per-particle two-pool pools (`models/extraction`), released into the water phase, advected on the grid/particles. Same kinetics as the primary.

## Stability / cost note
The volume-estimation inaccuracy of vanilla MPM forces either a small timestep (kills real-time) or a large one (occasional blow-up). This solver is **not** held to the real-time gate — it runs at whatever timestep keeps it stable, purely as the fidelity reference. That asymmetry is the whole reason it isn't the primary.

## Build phases & gates
1. APIC transfers + single-phase water grid. *Gate:* incompressible dam-break (offline ok).
2. Drucker-Prager sand, dry. *Gate:* angle-of-repose; static bed.
3. Two-grid drag coupling (implicit). *Gate:* drawdown vs grind matches the primary qualitatively.
4. Extraction read-out via `models`. *Gate:* yield/TDS in band; **comparable to XPBD on identical `models`** in `compare_solvers`.

## Open questions
- Preconditioner for the implicit drag solve (Tampubolon list it as future work; affects cost but not whether it's the reference).
- Whether to bother with capillary-action wetting here (the MPM mixture model omits it; `models/wetting` may not map cleanly) — acceptable for a reference.

## References
Tampubolon, Gast, Klár, Fu, Teran, Jiang, Museth, *Multi-species simulation of porous sand and water mixtures*, ACM TOG 2017; Klár et al. *Drucker-Prager Elastoplasticity for Sand Animation*, SIGGRAPH 2016; Hu et al. *MLS-MPM* 2018.

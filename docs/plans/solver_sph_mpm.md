# coffeesim — `solver_sph_mpm` Plan (alternate)

**Parent:** `docs/ARCHITECTURE.md` · **Module:** `src/solvers/sph_mpm/` · **Status:** plan (lower priority — alternate for comparison)
**Depends on:** `models/`, `utils/` (GpuContext, spatial hash, kernels, SDF), `solvers/base.rs`

## Purpose
A hybrid: **DFSPH water + MPM bed**, coupled internally at the particle/grid level. Sits on the middle of the fidelity/cost curve between the rigorous-but-offline MPM mixture and the real-time XPBD primary. Kept as an alternate to profile against the others on identical `models/`.

`SolverDescriptor`: `Hybrid`, `owns_grid: true` (the bed's MPM grid), `CflLimited`.

## Scope
**In:** DFSPH water, MPM Drucker-Prager bed, internal drag/buoyancy coupling, multi-rate integration, extraction. **Out:** shared physics (`models.md`); the seam.

## Internal design
- **Water — DFSPH**: constant-density + divergence-free solvers (per-particle stiffness factors), **warm-started**; **particle shifting** for long-horizon distribution health (prevents clumping over thousands of steps). Free surface natural. Incompressibility via the two local solvers — no global Poisson.
- **Bed — MPM**: Drucker-Prager elastoplastic on its own grid; rest-config + rate-gated plasticity for static hold; cohesion from `models/cohesion`.
- **Internal coupling** (the hybrid's hard part): one **shared spatial hash** (`utils/hash`) bins both species so water particles find bed neighbors. Per water particle: sample local `α_s` and `v_s` from bed neighbors → **implicit per-particle drag** (`models/permeability`; unconditionally stable for any grind) → scatter the momentum-conserving reaction onto bed grains (picked up by MPM P2G). Plus buoyancy from the SPH pressure field on grains. **Multi-rate**: subcycle the MPM bed against the DFSPH water step, holding the water field constant during sub-steps (CFD-DEM pattern; frozen-bed fast path when quiescent).
- **Extraction**: per-grain two-pool pools (`models/extraction`); solute released to water particles, advected (Lagrangian).

## Build phases & gates
1. DFSPH water (warm-start + particle shifting). *Gate:* incompressible; 3-min volume-stable; no clumping.
2. MPM bed, dry. *Gate:* angle-of-repose; static.
3. Shared-hash drag + buoyancy coupling + subcycling. *Gate:* drawdown vs grind; momentum conserved; comparable to XPBD on identical `models`.
4. Extraction via `models`. *Gate:* yield/TDS in band.

## Open questions
- Coupling-split convergence (DFSPH-implicit-drag ↔ MPM bed Gauss-Seidel split; 1–2 iterations).
- SPH boundary handling for the dripper SDF (boundary particles vs volume maps) — fiddlier than the primary's position-level SDF collision.
- Particle-shift ↔ scalar conservation: the shift must advect carried `c`/`T` or solute leaks over a long brew.

## References
Bender & Koschier, *Divergence-Free SPH* (2015/2017); Akinci et al. *Versatile Rigid-Fluid Coupling for Incompressible SPH* (2012); Klár et al. 2016; CFD-DEM subcycling practice.

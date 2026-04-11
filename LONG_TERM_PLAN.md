# Coffee Sim Plan

## Goal

Build a browser-native pour-over simulation where the key visible behaviors come from one coherent physical model:

- a coherent kettle jet
- a real perched water layer above the bed
- believable bed deformation and compaction
- filter-mediated drawdown and drip formation
- a cup that fills as a liquid body with real height
- extraction layered on top of the hydraulic model later

The central solver commitment is:

- `water should be treated as near-incompressible everywhere`

The bed and filter should affect water through coupling, resistance, storage, and boundary flux, not by making the water phase itself unrealistically soft.

## Diagnosis

The current code path is architecturally wrong for the long term:

- weakly-compressible water is doing work that should belong to pressure/incompressibility
- cup-specific damping and free-stream ballistic-preserve heuristics are compensating for missing global fluid behavior
- bed coupling removes momentum and mass through local shortcuts rather than through a clean two-phase interaction model

Those hacks are still acceptable as temporary demo scaffolding, but they are not the destination architecture.

## Governing Direction

### What We Are Learning From

This plan is guided by two ideas:

1. `SPlisHSPlasH-style free-surface behavior`
- coherent emitter/source treatment
- strong incompressibility
- robust boundaries
- pooled liquid emerges from the fluid solver rather than a "cup mode"

2. `Particle-laden-flow-style two-way coupling`
- water and particulate material remain separate phases
- the coffee bed is not "soft water"
- interaction happens through explicit momentum exchange, storage, permeability, and stress

We are not copying either system literally. We are adapting the useful parts to a browser-first WebGPU MPM pipeline.

## Solver Decision

Before Stage 1, we commit to one specific incompressibility path.

### Chosen Stage-1 Method

Use a `fixed-iteration Red-Black Gauss-Seidel pressure projection` on the MPM grid.

Reason:

- excellent fit for WebGPU storage-buffer passes
- predictable iteration budget
- materially better low-frequency convergence than plain Jacobi at the same iteration budget
- simpler than CG/MGPCG on the web
- much easier to integrate with the current grid-based MPM transfer than a hybrid PBF/DFSPH path

This means Stage 1 specifically consists of:

1. P2G to grid mass + provisional velocity
2. classify cells into air / surface / interior fluid / bed-coupled fluid
3. compute divergence on water-occupied cells
4. solve a pressure Poisson system with fixed Red-Black Gauss-Seidel iterations
5. project grid velocity
6. run G2P using projected velocities
7. recompute APIC affine state from the projected grid field

### Explicit Non-Choices

We are not doing, in Stage 1:

- "just raise bulk modulus"
- an unspecified "divergence-free correction pass"
- PBF/DFSPH bolted onto the current pipeline
- MGPCG / multigrid as the first browser implementation

Those may become later upgrades, but they are not the initial implementation target.

Fallback:

- plain Jacobi is acceptable only as a temporary fallback if RB-GS causes an unexpected implementation problem on the WebGPU side

## High-Level Architecture

### 1. Water Phase

Water remains explicit particles in the current WebGPU MPM pipeline, but the solver target changes:

- water should be globally near-incompressible
- free liquid should not collapse into one-particle-thick sheets
- free-fall streams should not numerically brake in midair
- pooled water should build hydrostatic height naturally

### 2. Coffee Bed Phase

The coffee bed remains explicit material particles:

- deformable
- porous
- compactable
- separate from the water phase

Its jobs:

- resist and redirect water motion
- store water in pore space
- exchange momentum with water
- compact under load
- change permeability as it compacts and wets

### 3. Paper Filter Phase

The paper filter becomes a real physical interface:

- supports the bed
- deforms under wet/dry load
- later meters the flux from slurry to free drip

This must be split into two implementation stages:

- `Stage 5a`: a sagging support surface / moving filter SDF proxy
- `Stage 5b`: a true deformable shell mesh only if 5a is not visually sufficient

### 4. Coupling

The important coupling should come from:

- pressure in the water phase
- two-way drag / momentum exchange
- bed storage (`s_h`, later `s_v`)
- compaction modifying porosity/permeability
- Darcy-style flux closures at the filter interface

The plan does **not** reject Darcy everywhere. It rejects a domain-wide Darcy solve as the primary engine. Darcy-style closures at local interfaces are appropriate and expected.

## Physical States

### Water Particle State

Water particles remain explicit and should support:

- position
- velocity
- mass
- deformation / APIC state
- optional transient classification:
  - `free_jet`
  - `surface_pool`
  - `bed_coupled`
  - `filter_flux`
  - `cup_pool`

### Coffee Bed Particle State

Each bed particle should eventually carry:

- `phi_h`
  - intergranular porosity
- `phi_v`
  - intragranular wettable capacity proxy
- `s_h`
  - intergranular saturation
- `s_v`
  - intragranular saturation
- `k_eff`
  - effective permeability
- `sigma_c`
  - compaction / effective stress proxy
- `psi_s`
  - fast-extractable fraction
- `psi_b`
  - slow-extractable fraction
- `c_h`
  - dissolved concentration in intergranular liquid
- `c_v`
  - dissolved concentration in intragranular liquid
- `temperature`
- existing deformation/rest state

### Filter State

The filter should eventually have:

- authored rest shape
- live support surface or shell state
- pinned rim
- local saturation
- local retained water
- local permeability / clogging state

### Free-Surface Cell Classification

This is mandatory for Stage 1.

Each grid cell must be classified each substep as one of:

- `air`
  - mass below occupancy threshold
  - pressure Dirichlet boundary `p = 0`
- `surface_fluid`
  - mass above occupancy threshold but adjacent to air
  - pressure Dirichlet boundary `p = 0`
- `interior_fluid`
  - water-occupied and not adjacent to air
  - pressure solve applies normally
- `bed_coupled_fluid`
  - water-occupied and in/near bed-support region
  - pressure solve applies with sink-augmented divergence

Stage-1 default detector:

- marker / occupancy threshold on grid mass
- not a level set

That means:

- `occupied = cell_mass > tau_occ`
- `surface = occupied && has_air_neighbor`
- `interior = occupied && !has_air_neighbor && !bed_coupled`

Default:

- `tau_occ = 0.25 * rho_ref * V_cell`

This is the cheapest browser-feasible choice and must be the default until measured artifacts justify a more expensive surface reconstruction.

## Core Solver Principles

### Water Should Stay Near-Incompressible

Water should be treated as near-incompressible:

- in the spout jet
- above the bed
- in the drip stream below the filter
- inside the cup

The bed and filter should alter water through coupling and resistance, not by weakening incompressibility.

### APIC / Projection Interaction

Pressure projection and APIC interact non-trivially.

We explicitly choose the simple defensible rule:

- `always recompute APIC affine C from projected grid velocities in G2P`

We accept the small angular-momentum cost instead of trying to preserve the pre-projection affine field.

### J Field Under Projection

With pressure projection driving the water stress, the per-particle `J` field no longer drives the water solver.

Milestone-1 rule:

- `J` is either removed from water-particle state entirely or retained only as a drift / consistency indicator
- `J` must not re-enter the water stress path once projection is the primary incompressibility mechanism

### Bed Cells Inside Projection

Water near and inside the bed cannot be treated the same as open free water.

Stage-1 engineering compromise:

- bed-coupled cells satisfy a `sink-augmented incompressibility` condition

Specifically:

- `div(u) = -m_abs / (rho * V * dt)`

where:

- `m_abs` is the water mass transferred into bed storage this step
- `rho * V` is the cell water reference mass

Clarifications:

- `rho * V` is the `reference` full-cell water mass, not the current cell mass, so the sink term remains well-defined in partially occupied cells
- for Milestone 1, `m_abs` is predicted from pre-projection cell velocities as a first-order approximation
- if residual mass drift exceeds budget, add a later second-stage re-evaluation using projected velocities

This lets water enter the bed at a controlled rate while keeping the projection linear and avoiding a hard incompressible piston at the bed interface.

### Momentum Conservation In Bed Coupling

Current bed coupling does not satisfy Newton's third law.

The long-term rule must be:

- any drag impulse taken from water is deposited into the bed phase as equal-and-opposite momentum, modulo external friction and boundary losses

This should happen through shared grid-side accumulators, not disconnected heuristics.

## CFL and Timestep Policy

This must be explicit.

### Current Risks

The current weakly-compressible water path already runs close to or beyond acoustic-CFL comfort for realistic increases in bulk modulus. Stage 1 should therefore not rely on raising `K` as the main incompressibility tool.

### Stage-1 Policy

Use adaptive substepping driven by:

- advective CFL
- projection iteration budget
- drag stiffness budget

Track live:

- `advective_cfl = max(|u|) * dt / dx`
- `acoustic_cfl = c_s * dt / dx` where applicable

Rules:

- maintain `advective_cfl <= 0.75`
- projection removes the need to rely on acoustic-CFL for free-water stability
- drag must be semi-implicit in Stage 2, not explicit

### Iteration / Budget Targets

Medium-quality target:

- projection: `24-32` RB-GS iterations per substep
- substeps: adaptive, starting from `5`
- total compute budget target: `<= 16 ms` on the reference machine

Hard limit:

- `max_substeps_per_frame = 10`

If advective CFL demands more:

- log a warning
- accept the violation temporarily
- do not let the frame silently spiral into unbounded substep counts

Fallback rule:

- if Stage 1 blows the frame budget, reduce projection iterations and accept a quantified residual
- do not revert to stronger cup-specific heuristics as the fallback

## Proposed Solver Upgrade Path

### Milestone 0: Instrumentation and Population Management

This is not a soft "cleanup" stage. It is a real milestone.

Work:

- add a GPU metrics buffer
- add staging-buffer readback on a throttled cadence
- add WASM exports and a JS HUD
- add a deterministic benchmark scene:
  - fixed seed
  - fixed dt
  - fixed spout motion
- add particle population management:
  - compact dead particles out of the active range
  - or maintain a freelist / active-count compaction pass
- define emission saturation behavior once `num_water` approaches `max_particles`
  - default: drop new emission with an explicit warning / metric rather than silently distorting flow rate
- isolate current cup and free-stream heuristics behind clearly removable code paths

Required metrics:

- `emitted_water_mass`
- `active_water_mass`
- `bed_held_water_mass`
- `cup_contained_water_mass`
- `filter_retained_water_mass`
- `outflow_mass_rate`
- `surface_pool_depth`
- `max_cell_div`
- `l2_cell_div`
- `projection_residual`
- `projection_iterations`
- `max_particle_speed`
- `vel_cap_clamp_count`
- `particle_count_by_phase`
- `deactivated_particles_per_step`
- `mass_balance_residual`
- `advective_cfl`
- `acoustic_cfl`

Exit criteria:

- metrics update in-browser
- readback does not break runtime
- two consecutive fixed-scene runs stay within `5%` on window-averaged or integrated main metrics

Parallelizable work:

- the Stage-1 design notes for free-surface classification, projection layout, APIC recompute, and bed-cell sink treatment can proceed in parallel with this milestone

### Milestone 1: Near-Incompressible Free Water

Upgrade the water solver itself.

Work:

- add free-surface classification
- compute divergence on occupied fluid cells
- run fixed-iteration RB-GS pressure projection
- project grid velocities
- recompute APIC affine state from projected velocities in G2P
- remove reliance on bulk-modulus-only stabilization as the primary liquid mechanism

Stage-1 success means:

- the free stream remains coherent without midair braking
- water in the cup builds height without cup-specific hacks being the main mechanism
- perched water above the bed begins to resemble real liquid

Required intermediate tests:

1. `Closed-box rest test`
- max `|div u| < 1e-3` in occupied fluid cells after 5 seconds

2. `Free stream into empty cup`
- stream accelerates until impact
- no visible pre-impact braking
- pooled water becomes thicker than a one-particle sheet

3. `Default V60 scene`
- run for `>= 30` simulated seconds with cup-specific pooling hack disabled or behind a feature flag
- pool height remains stable to within `5%` of expected trend

Failure modes to watch:

- ghost suction at free surface
- pressure pulling water into empty cells
- APIC ringing after projection
- frame budget collapse

### Milestone 2: Two-Way Water/Bed Momentum Exchange

Once water behaves more like water, improve bed coupling.

Work:

- move bed-water exchange to the grid/shared coupling path
- use `semi-implicit drag`, not explicit drag
- deposit equal-and-opposite drag impulse into the bed phase
- define drag from an explicit closure

Default closure:

- a Kozeny-Carman / Ergun-style permeability-derived drag law

Also deliver:

- bed lookup rebuild policy

Default:

- rebuild the bed lookup every substep using the existing control-volume logic

Deferred:

- dynamic hashing / more elaborate lookup acceleration only if profiling later proves the rebuild cost is too high

Exit criteria:

- bed deformation responds to actual momentum exchange
- post-bed flow depends on bed state, not a fixed drag constant
- momentum accounting is explainable and approximately conserved

Failure modes:

- explicit drag instability
- stale bed lookup causing wrong coupling
- water momentum loss without bed response

### Milestone 3: Surface Pooling Above the Bed

Reintroduce perched-water logic only after Milestone 1 is working.

Work:

- classify water relative to the top bed surface
- allow a real surface-pool regime above the bed
- compute `surface_pool_depth` from the actual free-surface detector, not an ad hoc height proxy
- let water remain free liquid above the bed when infiltration is limited

Dependency:

- if Milestone 1 does not give believable free liquid, Milestone 3 will not work

Exit criteria:

- moderate/high flow creates a clear perched layer
- lower flow creates less perched depth
- stopping the pour produces realistic drawdown

### Milestone 4: Intergranular Storage and Drainage

Add Moroney-style double-porosity structure in an MPM-compatible form.

Work:

- `s_h` becomes the main short-timescale hydraulic storage
- infiltration fills `s_h`
- drainage depends on:
  - local head
  - `k_eff`
  - compaction
  - filter flux availability
- add explicit `bed -> grid` exfiltration for oversaturated cells

Clarification:

- we do not use a domain-wide Darcy solve as the main engine
- we do use Darcy-style local flux closures where physically appropriate

Exit criteria:

- the bed no longer behaves like an instant sink
- retained water and drawdown look plausible
- water above the bed and water leaving the bed feel causally linked

### Milestone 5a: Sagging Filter Support Surface

Do the cheap version first.

Work:

- add a filter support surface or moving filter SDF whose shape depends on summed bed/water load
- bed rests on the filter support
- filter support sags under wet/dry load

This is the preferred first implementation because it is much cheaper than full cloth-shell coupling and should capture most of the visual benefit.

Exit criteria:

- bed no longer cuts through the paper
- filter support visibly sags under load
- bed support reads correctly in the scene

### Milestone 5b: True Deformable Filter Shell

Only do this if 5a fails the visual bar.

Work:

- pinned shell mesh
- structural / shear / bending constraints
- collision/support with the bed and later fluid loading

Exit criteria:

- deformed paper reads materially better than 5a
- shell remains stable and attached

### Milestone 6: Filter-Mediated Flux and Drip Formation

Make the filter control how water leaves the bed.

Work:

- compute local Darcy-style flux through the filter from:
  - head above the filter
  - local saturation
  - local filter permeability
- accumulate underside water until a drip or coherent outlet stream forms
- return the water to the free-liquid solver below the filter

Required interface from Milestone 5:

- `filter_sample(x)` or equivalent must expose:
  - local support position
  - local saturation
  - local permeability / resistance

Exit criteria:

- bed outflow is no longer just leftover particle momentum
- the drip stream emerges from filter-mediated flow
- clogging / overload can slow drawdown plausibly

### Milestone 7: Compaction, Effective Stress, Wet Memory, and Extraction

Finish the bed model after pressures are available.

Work:

- update `sigma_c` from `effective stress`, not load alone
- `sigma_c` modifies `phi_h` and `k_eff`
- add slower `s_h -> s_v` transfer
- add `psi_s`, `psi_b`, `c_h`, `c_v`, and temperature
- advect dissolved solute with the water phase rather than keeping it entirely on the bed

Explicit dependency:

- Milestone 7 depends on Milestone 1 producing usable pore pressures

Without pore pressure, compaction is just curve-fitting.

Exit criteria:

- repeated pours behave differently from fresh dry bed
- wet memory persists
- extraction gains a real fast/slow structure

## Implementation Notes

### What Not To Do

Do not keep investing in:

- cup-specific fake pooling as the long-term fix
- height-only triggers that make the stream brake before impact
- using water compressibility as the main porous-bed mechanism

Those may be acceptable as temporary demo patches, but not as the destination architecture.

### What To Borrow From SPlisHSPlasH

- coherent source treatment
- robust boundaries
- stronger incompressibility
- letting filling emerge from the liquid model

### What To Borrow From Particle-Laden Flow

- separate water and particulate phases
- two-way momentum exchange
- dense particulate material affecting flow through coupling, not through weakening the water phase

### Surface Tension

Surface tension is deferred through Milestone 7.

Implication:

- jet coherence in Milestones 1-3 must come from incompressibility and source support alone

Rule:

- if jet breakup, meniscus behavior, or capillary/wetting artifacts dominate the visual failures after Milestones 1-3, revisit surface tension as its own dedicated stage
- do not compensate for missing surface tension with more local heuristics

## Validation Plan

### Required Continuous Checks

Run after each meaningful solver change:

```bash
cargo test -p coffee-sim-wasm
cargo check -p coffee-sim-wasm --target wasm32-unknown-unknown
wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg
```

### Conservation Invariants

Must be conserved exactly or to tight tolerance:

- total water mass

Should be conserved approximately, modulo explicit drag, friction, and boundary loss:

- total momentum

May drift downward by design:

- total kinetic energy

Mass bookkeeping target:

- `< 0.1%` total mass error over a representative run

### Quantitative Metrics

Track:

- `emitted_water_mass`
- `active_free_water_mass`
- `bed_held_water_mass`
- `cup_contained_water_mass`
- `filter_retained_water_mass`
- `outflow_mass_rate`
- `surface_pool_depth`
- `avg_s_h`
- `avg_s_v`
- `avg_k_eff`
- `avg_sigma_c`
- `max_cell_div`
- `l2_cell_div`
- `projection_residual`
- `advective_cfl`
- `acoustic_cfl`

Success threshold for Stage 1:

- `rms(div u) / max(|u|, eps) < 1e-3` over water-occupied interior cells

### Key Visual Tests

#### Test A: Free Stream Into Empty Cup

Setup:

- disable bed or bypass the dripper
- emit a clean free stream into the cup

Expected:

- stream accelerates downward until impact
- no visible midair braking
- water builds a rising pooled body in the cup

Failure:

- stream slows before impact
- pool remains one-particle thick

#### Test B: Center Pour Onto Bed

Expected:

- visible perched water above the bed
- crater / bed deformation
- delayed but believable drawdown

#### Test C: Pour Stop / Drawdown

Expected:

- surface water drains gradually
- outflow decays gradually
- no instantaneous disappearance

#### Test D: Filter Support

Expected:

- paper visibly supports the bed
- paper sags under wet load
- bed does not clip through filter

#### Test E: Repeat Pour Memory

Expected:

- compacted / previously wetted regions behave differently on later pours

## Failure-Mode Appendix

### Milestone 0

Most likely failures:

- readback stalls the frame
- metrics are noisy across runs
- active/dead particle counts drift

Detection:

- HUD plots jitter wildly across repeated fixed scenarios
- particle count grows without corresponding active water mass

### Milestone 1

Most likely failures:

- ghost suction at the free surface
- pressure pulls water into empty cells
- APIC ringing after projection
- frame budget collapse

Detection:

- `max_cell_div` stays high
- projection residual stalls
- surface inflates into air
- FPS drops below target

### Milestone 2

Most likely failures:

- explicit drag instability
- stale bed lookup
- non-conservative momentum exchange

Detection:

- NaNs or velocity spikes in bed-coupled cells
- bed response lags or occurs in the wrong region
- momentum loss without corresponding bed motion

### Milestone 3

Most likely failures:

- surface pool never forms
- surface water hovers unnaturally
- pooled depth metric does not track visual pool

Detection:

- high-flow and low-flow pours produce the same pooled depth
- surface particles persist without draining

### Milestone 4

Most likely failures:

- bed becomes an instant sink again
- oversaturated regions have no backflow
- retained water and outflow decouple

Detection:

- `s_h` saturates without corresponding drainage
- outflow remains unrelated to pooled head

### Milestone 5

Most likely failures:

- filter support clips through bed
- sag is visually imperceptible
- shell/support becomes unstable

Detection:

- bed intersects filter geometry
- filter does not move under load
- deformation jitters or inverts

### Milestone 6

Most likely failures:

- filter flux behaves like a hidden teleport
- drip stream is not causally linked to filter load
- local overload does not change drawdown

Detection:

- outflow rate ignores filter saturation/permeability
- drip timing is identical across different load states

### Milestone 7

Most likely failures:

- compaction is only curve-fit and ignores pore pressure
- `s_v` mirrors `s_h` too closely
- extraction transport is uniform and non-advective

Detection:

- repeated pours show no wet-memory effect
- extraction remains effectively single-timescale

## Immediate Next Priorities

1. Build Milestone 0 properly: metrics, readback, benchmark scene, particle population management.
2. Write a short Stage-1 design note for:
   - free-surface classification
   - Jacobi projection
   - APIC recompute after projection
   - bed-cell sink treatment
3. Remove reliance on cup-specific pooling behavior as the long-term mechanism.
4. Continue paper-filter support work in parallel through Stage 5a.

## Success Criteria

This architectural direction is successful when:

- the free stream, perched water, and cup pooling all look like the same liquid governed by one coherent water model
- the coffee bed still meaningfully alters the flow through coupling, storage, compaction, and pressure
- the paper filter becomes the real transition from bed/slurry to drip
- cup filling and drawdown no longer depend on brittle location-specific heuristics

---

# Dual-Velocity-Field Bed Coupling

## Status: IMPLEMENTED

## Problem

The current bed-particle approach is a bandaid: bed particles deposit mass into the shared MPM grid for PIC-averaging repulsion but are excluded from the pressure solver, with a hardcoded 0.35 velocity damping factor. This doesn't model real soil-water physics — there's no constitutive stress model for the bed, no pore pressure, and the coupling is an arbitrary damping rather than physics-based drag.

## Reference: Anura3D Two-Phase MPM

Anura3D (an open-source geomechanics MPM code) uses:

1. **Two independent velocity fields on one grid** — solid skeleton and pore fluid each have their own mass, momentum, and velocity at every grid node.
2. **Darcy drag coupling** — `F_drag = (n^2 * mu / K) * (v_solid - v_water)` where K is intrinsic permeability, n is porosity, mu is fluid viscosity.
3. **Pore pressure** — updated from volumetric strain. Gradient drives both phases.
4. **Effective stress** — soil particles carry `sigma_effective = sigma_total - p_water * I`. A constitutive model operates on effective stress only.

## Design

### Grid layout change

Currently: one set of atomic accumulators (`grid[4*idx + 0..3]`) for mass + 3 momentum, decoded into `grid_vel[idx]` as `vec4<f32>(vx, vy, vz, mass)`.

New: **two sets** of accumulators — one for water, one for solid. During `grid_update`, decode both into two velocity fields.

**Add `grid_solid` atomic buffer and `grid_vel_solid` float buffer.** Water uses existing `grid` + `grid_vel`. Solid uses new buffers.

### Shader changes

#### clear_grid
Clear both water and solid buffers.

#### p2g — split by phase
- Water (`phase < 0.5`): deposit to `grid` — unchanged.
- Bed (`phase >= 0.5`): deposit to `grid_solid`. Same B-spline, same APIC. Also contribute elastic stress: `sigma = -K_bed * (1 - J)` via stress-augmented momentum.

#### grid_update — dual decode + Darcy drag
Decode water velocity from `grid` → `grid_vel`.
Decode solid velocity from `grid_solid` → `grid_vel_solid`.
On coupled nodes (both `mass_water > 0` and `mass_solid > 0`):
```
F_drag = (n^2 * mu / K) * (v_solid - v_water) * dt
v_water += F_drag / mass_water
v_solid -= F_drag / mass_solid
```

#### g2p — split by phase
- Water: gather from `grid_vel` — unchanged.
- Bed: gather from `grid_vel_solid`. Remove hardcoded 0.35 damping.

### Constitutive model — elastic bulk stress
```
stress = -K_bed * (1.0 - J)
```

## Implementation steps

1. Add `grid_solid` and `grid_vel_solid` buffers to state.rs
2. Update clear_grid to clear solid buffers
3. Split P2G: water → `grid`, bed → `grid_solid` with elastic stress
4. Dual grid_update: decode both fields, apply Darcy drag on coupled nodes
5. Split G2P: water from `grid_vel`, bed from `grid_vel_solid`, remove 0.35 damping
6. Clean up bed_coupling (remove redundant drag)
7. Test + tune K_bed and drag parameters

---

# Fix Bed Absorption, Pooling, and FPS

## Status: IMPLEMENTED

## Problems

1. **No absorption**: Water passes through the bed without being absorbed. `bed_coupling` runs AFTER G2P, so the velocity/pressure fields never see the absorption.
2. **No pooling**: `classify_cells` marks ALL cells with `solid_mass > 1e-6` as `CELL_BED_COUPLED`, skipping the pressure solver entirely.
3. **FPS regression**: Dual-velocity-field doubled grid buffers from ~24MB to ~47MB at 80x115x80.

## Changes

1. **Grid-level absorption in grid_update**: After Darcy drag, compute absorption sink on coupled cells. Reduce water mass before writing grid_vel. Credit bed via bed_delta. Store absorbed fraction in grid slot 2.
2. **Fix classify_cells**: Only BED_COUPLED if solid AND no water. Cells with both → fluid classification.
3. **Simplify bed_coupling**: Read absorbed fraction from grid, reduce particle mass.
4. **Reduce grid**: 80x115x80 → 64x92x64. Hot buffers drop from ~47MB to ~24MB.

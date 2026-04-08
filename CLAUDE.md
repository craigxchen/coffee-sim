# Browser-Native Coffee Physics Plan

## Purpose

This document describes the long-term simulation direction for `coffee-sim`.

The goal is not to build a generic physics engine. The goal is to build a browser-native, physically grounded simulation core specialized for pour-over coffee, with enough structure that we can increase sophistication over time without throwing the architecture away.

The key product constraint is:

- the final experience must remain interactive in a web app

The key simulation constraint is:

- realism is the highest priority, especially for water delivery, bed wetting, drawdown, and extraction

## Relationship to Existing Work

Older planning documents and prototypes are useful as background, but they are not binding on this plan.

This document should be driven by:

- physical realism
- browser-native execution constraints
- extensibility toward water, coffee-bed, and extraction coupling

That means:

- we should reuse ideas from earlier work only when they still serve the new architecture
- we should not preserve Darcy-first assumptions just because they existed before
- we should not treat any older module or document as the final authority on the new simulation design

The standard for this plan is whether it gives us the most physically credible and extensible browser-native simulator, not whether it matches legacy architecture.

## Design Principles

### 1. Build a domain-specific simulator, not a general-purpose engine

Generic engines solve the wrong problems for us. We care about:

- controlled water inflow from a kettle
- free-flight jet behavior
- impact on a filter bed
- infiltration through ground coffee
- pooling, drawdown, and channeling
- extraction and cup output

That is a specialized multiphysics problem. A coffee-specific simulator is justified.

### 2. Use one conserved physical state, but allow multiple local models

We should not think in terms of disconnected fake systems.

Instead:

- water mass should be conserved across the whole domain
- momentum should be accounted for across the whole domain
- the world should be represented in one shared coordinate system
- different regions can use different constitutive laws

This means:

- kettle outlet is an inflow boundary
- free water is a fluid
- coffee bed is a porous deformable medium
- extraction is a transported scalar field

These are different closures for the same physical world, not unrelated game systems.

### 3. Optimize for browser-native execution

The architecture must be shaped by WebGPU realities:

- passes should be regular and batchable
- memory layout must be compact and predictable
- neighbor-heavy methods should be used carefully
- full-frame CPU-GPU synchronization should be avoided in the hot path
- stability must come before maximum complexity

### 4. Separate simulation truth from rendering

Rendering should always be downstream of simulation state.

That means:

- no UI-only or render-only water-level hacks
- no visual fill cues that are not backed by simulation state
- surfaces, splats, foam, and bed visuals should all derive from solver outputs

### 5. Prefer a clean replacement over compatibility scaffolding

We should not carry dead simulation code, fallback engines, or long-lived transitional layers in the mainline.

That means:

- the new solver should replace the old one, not sit beside it indefinitely
- compatibility shims should be temporary and aggressively removed
- once a new path proves itself, the old simulation path should be deleted rather than maintained
- version history and branches are enough to preserve the old work; the active codebase should stay clean

## High-Level Architecture

The recommended long-term architecture has four major simulation layers:

1. `Inflow layer`
2. `Free-water layer`
3. `Coffee-bed layer`
4. `Extraction layer`

### Inflow Layer

The kettle is treated as a physically meaningful inflow boundary:

- controlled by kettle angle
- produces a flow rate and outlet speed
- injects mass and momentum into the simulation
- defines the source geometry for the stream

This replaces the fragile idea of "spawn particles near a nozzle and hope the solver behaves."

### Free-Water Layer

Free water outside the bed should be simulated with a browser-native particle-grid method:

- preferred foundation: `MLS-MPM` or `APIC/MLS-MPM`
- this captures free motion, impact, pooling, and coherent transport
- it avoids SPH neighborhood search as the dominant cost

This layer is responsible for:

- jet continuation after inflow
- splash and pooling in the dripper
- flow through the dripper throat
- water accumulation in the carafe

### Coffee-Bed Layer

The coffee bed should be represented by coffee-bed material particles on the same MPM grid, not by fully resolved grains.

Those bed particles should carry state such as:

- porosity
- permeability
- saturation
- compaction
- wetting state
- extraction state

This layer is responsible for:

- infiltration
- drainage
- flow resistance
- channel formation
- bloom-like behavior through local changes in saturation, permeability, and bed structure

This gives us a more unified particle-based model without paying the cost of simulating every ground individually.

### Extraction Layer

Extraction is a scalar transport problem on top of fluid/material motion.

This layer tracks:

- how much soluble material remains in a region of the bed
- how much dissolved material is in the water
- how extraction depends on wetness, contact time, and local flow

This layer is responsible for:

- TDS-style output metrics
- cup concentration estimates
- spatial extraction maps

## Why MPM Is the Right Foundation

The current SPH exploration has already shown a core browser problem:

- thin jet emission is hard to make robust with a lightweight SPH implementation
- neighborhood search is expensive
- emitter support and boundary treatment quickly become complex

WebGPU-Ocean is useful because it proves that `browser-native MLS-MPM` is practical:

- no neighbor search in the hot path
- regular `P2G -> grid update -> G2P` structure
- good particle counts on browser GPUs

However, WebGPU-Ocean is a starting point, not our final model. It is:

- demo-oriented
- box-bounded
- single-material
- tuned aggressively for speed

We should borrow the structure, not the product assumptions.

## Solver Recommendation

### Recommended Solver Stack

- `Water`: MLS-MPM / APIC-style particle-grid solver
- `Boundaries`: SDF-based collision and friction
- `Coffee bed`: MPM bed particles carrying porous-material state on the same grid
- `Extraction`: advected scalar field

### Not Recommended as the Main Long-Term Path

- pure SPH for water everywhere
- per-ground coffee particles from the start
- embedding a large native simulator directly into WASM

### Why Not Start with Per-Ground Coffee Particles

Fully resolving individual grounds would create a much harder problem:

- far more particles
- contact-heavy behavior
- hard-to-calibrate constitutive behavior
- poor browser scaling

The right compromise is not "field only" versus "every grain." It is a porous, particle-based bed represented by material points on the shared MPM grid.

## Proposed Simulation State

### World Space

We should define one stable world-space convention early:

- `Y+` is upward
- all geometry lives in one consistent metric-like scale
- kettle, dripper, filter, bed, and carafe all share the same coordinate system

We should avoid ad hoc scale changes between solver modules.

### Water Particle State

Initial water-particle state for MPM:

- position: `vec3`
- velocity: `vec3`
- affine velocity matrix `C`: `mat3`
- mass: `f32`
- phase/material id: `u32`
- optional per-particle temperature later if advection quality demands it
- optional dissolved-solids concentration later

Particles exist primarily to carry water mass and velocity information between grid transfers.

### Coffee-Bed Particle State

Initial coffee-bed particle state:

- position: `vec3`
- velocity: `vec3`
- affine matrix or deformation state as required by the selected MPM variant
- solid mass: `f32`
- pore-water content: `f32`
- porosity: `f32`
- permeability: `f32`
- compaction: `f32`
- temperature: `f32`
- extractable-solids state
- material id: `u32`

These are not grain-resolved particles. They are material points representing deformable porous coffee-bed state.

### Grid State

Base grid state:

- mass
- momentum
- velocity

Extended grid state over time:

- solid SDF sample or boundary mask
- temporary bed-coupling accumulators
- extraction scalar(s)
- temperature
- material flags

The water solver and coffee-bed particles should operate on the same grid, while longer-lived bed state stays primarily on the bed particles unless profiling proves a grid-cached copy is necessary.

### Geometry State

Static geometry:

- dripper SDF
- filter wall SDF
- carafe SDF
- cup/carafe drain region if needed

Dynamic geometry later:

- kettle spout transform
- bed surface deformation proxy

### UI-Controlled State

Interactive user state:

- kettle angle
- optional flow override
- camera state
- pause/reset
- debug view toggles

Later:

- grind size
- bed depth
- pour pattern
- agitation

## Pass Structure

The simulation should be organized into explicit passes that map well to WebGPU.

## Phase 1: Water-Only MPM

Minimal pass sequence:

1. clear grid
2. apply inflow boundary
3. P2G mass and momentum
4. P2G stress / affine transfer
5. grid update
6. SDF boundary projection / friction
7. G2P
8. render-copy or render-ready packing

This should already be enough to establish the new canonical water path.

## Phase 2: Coffee-Coupled Water

Expanded pass sequence:

1. clear grid
2. apply inflow boundary
3. update porous bed state
4. P2G for water particles
5. apply porous drag and infiltration on grid
6. apply pressure/stress and gravity
7. apply SDF boundary handling
8. G2P for water particles
9. advect extraction scalars
10. pack render outputs

## Inflow Model

The inflow model is critical and should be solver-native from the start.

### Responsibilities

- convert kettle angle into flow rate
- convert flow rate into mass flux
- convert outlet geometry into injected momentum
- supply a stable stream shape

### Implementation Direction

Preferred:

- inject mass and momentum directly onto grid nodes in an outlet patch
- seed particles only as needed to maintain particle representation

Fallback:

- inject pre-packed particle sheets, but only if tied tightly to the MPM grid

### Key Parameters

- outlet position
- outlet normal/direction
- outlet width and thickness
- flow rate
- exit speed
- outlet profile shape

### Coffee-Friendly Extension

Later we can allow kettle angle to influence:

- flow rate
- outlet velocity
- slight stream widening
- pour footprint on the bed

## Boundary Model

We should move to SDF-based boundaries immediately when building the MPM path.

### Reasons

- V60 geometry is not box-shaped
- the carafe is not box-shaped
- later bed surfaces and filters may vary

### Boundary Handling Strategy

At minimum:

- grid-level velocity correction against SDF normals
- configurable friction and restitution

Optional safety layer:

- particle-level post-G2P SDF correction

### Boundary Types We Need

- hard solids:
  - dripper wall
  - carafe wall
  - table or domain floor
- semi-physical interfaces later:
  - filter wall
  - coffee-bed top surface

## Coffee Bed Model

The coffee bed should begin as a porous material-point system, not as a pure field and not as a full granular simulation.

## Initial Bed State

Each bed particle should define:

- initial porosity
- initial permeability
- initial dry density
- initial extractable mass
- initial pore-water content
- initial compaction state
- placement inside the filter volume

## Evolving Bed State

Over time, bed particles should track:

- saturation
- permeability changes from wetting
- compaction
- extraction progress

### Phenomena We Want This To Produce

- bloom
- surface pooling
- slow penetration when bed is dry
- faster flow through wet channels
- nonuniform extraction due to uneven flow

### Possible Constitutive Direction

We do not need a perfect porous-media model on day one. A reasonable staged path is:

Stage A:

- bed particles with porous drag and pore-water storage
- scalar saturation and permeability carried on particles

Stage B:

- permeability and drag as a function of saturation and compaction
- bed-particle state feeding back into local resistance and channel preference

Stage C:

- plastic/deformable bed behavior
- local bed surface change
- stronger bed-structure response to wetting and agitation

### Stage Transition Criteria

We should not leave constitutive upgrades open-ended. Each stage should have explicit exit criteria.

Move from `Stage A -> Stage B` when one or more of the following are true:

- changing grind or permeability does not change drawdown time by at least `10-15%` in validation scenarios
- off-center pours do not produce persistent nonuniform saturation patterns
- dry-to-wet transition looks too binary because the bed has no history-dependent permeability change
- spiral versus center pours do not separate meaningfully in uniformity metrics

Move from `Stage B -> Stage C` when one or more of the following are true:

- compaction-sensitive behavior cannot be matched by permeability updates alone
- repeated pours do not leave persistent structural flow preference where experiments or reference runs indicate they should
- bloom or crater behavior requires actual bed-surface displacement rather than only scalar-field changes
- visual realism is blocked by the bed remaining geometrically static despite correct bulk drawdown

## Coupling Between Free Water and Coffee Bed

This is where realism will be won or lost.

### The Rule

Water should move between free-water particles and coffee-bed particles through explicit conserved fluxes on the shared grid.

Not:

- particle teleports
- render-only fill changes
- arbitrary state swaps with no mass accounting

### Interface Quantities

At the free-water / bed-particle interface, compute:

- local water head or pressure
- local bed permeability
- local saturation deficit
- infiltration flux
- momentum loss due to drag

### First Concrete Coupling Strategy

The first implementation should be particle-to-particle through the shared grid, not a vague handoff.

1. During `P2G`, water particles deposit mass and momentum to the shared grid as usual.
2. During their own transfer/update path, coffee-bed particles deposit the state needed to compute local porosity, permeability, compaction response, and pore-water demand on the shared grid.
3. During `grid update`, nodes occupied by both water and bed particles apply porous drag using bed-particle-derived permeability, porosity, saturation, and pore velocity.
4. In that same `grid update`, coupled nodes compute an absorption sink `m_abs` for the current substep from local saturation deficit, permeability, and hydraulic head.
5. Grid mass and momentum are reduced immediately by that sink before `G2P`, so the free-water solve sees the coupling in the same substep rather than one frame later.
6. The absorbed mass is credited to nearby bed particles as pore-water content and used to update their saturation and stored-fluid state for the same substep.
7. During `G2P`, water particles moving through bed-occupied nodes inherit the slowed grid velocity and a per-node sink fraction.
8. After `G2P`, each affected water particle reduces its own residual mass by the already-accounted sink fraction so particle state matches the grid-side transfer exactly.
9. If a water particle falls below a minimum residual mass threshold, it is deactivated and its remaining mass is credited to nearby bed particles exactly.
10. If local bed particles exceed their pore-water storage capacity or hydraulic storage limit, the excess mass is exfiltrated as a grid-native source on neighboring interface nodes in the next substep; particles are seeded from that source only for representation quality, not as the primary conservation mechanism.

This gives us a concrete first answer to the "what happens when water hits the bed?" question:

- it does not teleport
- it does not immediately disappear
- it first loses momentum through porous drag
- then it transfers mass into bed-particle pore storage at a saturation-dependent rate in the same substep
- and it can re-emerge through an explicit interface source if the bed locally overloads

### Exfiltration Rule

Exfiltration should be grid-native first, particle-native second.

For each oversaturated bed-particle neighborhood:

- compute excess stored mass above local capacity
- move that mass into one or more neighboring free-fluid interface cells along the interface normal and gravity-biased direction
- assign exfiltrated momentum from nearby bed-particle pore velocity plus gravity over the substep
- only seed new particles from this exfiltrated grid mass when needed to maintain particle sampling density or rendering continuity

This avoids making particle spawning the primary mechanism for bed outflow and keeps conservation centered on the grid state.

### Coupling Pass Order

For the first coupled browser implementation, the pass order should be:

1. clear grid
2. apply inflow boundary
3. P2G for free-water particles
4. P2G or grid-coupling transfer for coffee-bed particles
5. apply fluid stress, gravity, and grid normalization
6. apply SDF solid-boundary projection
7. apply porous drag and compute interface absorption sinks on coupled water/bed nodes
8. subtract absorbed mass and momentum from the grid for the current substep
9. G2P for free-water particles
10. update coffee-bed particles with absorbed pore water, saturation, and compaction response
11. reconcile per-particle residual mass with the already-applied sink fractions and deactivate fully absorbed water particles
12. exfiltrate oversaturated bed water back toward free-fluid interface cells
13. advect extraction scalars

### Conservation Expectations

- mass transfer must be exact
- momentum transfer should be explicit
- energy loss is allowed only through modeled dissipation

## Extraction Model

The extraction model should remain separate from mechanics.

### Minimum State

- extractable solids remaining in bed cells
- dissolved solids carried in water

### Minimum Drivers

- saturation
- contact time
- flow rate
- temperature

### Outputs We Ultimately Want

- cup volume
- estimated extraction yield
- estimated strength / TDS proxy
- spatial extraction map for debugging

### Temperature Ownership

Temperature should not be treated as an optional afterthought once extraction work begins.

The intended ownership is:

- M1-M3: no temperature solve required for water-only and geometry bring-up
- M4: add temperature field storage and transport hooks to the bed/extraction state, even if calibrated heating/cooling is still simple
- M5: extraction does not ship without temperature as an explicit driver

That means temperature may remain absent from the earliest fluid bring-up, but it is a required part of the first extraction-capable build.

## Surface Tension and Wetting

Surface tension is not optional for full realism in pour-over.

It matters for:

- drip and stream cohesion near the spout
- stream breakup and reconnection
- meniscus behavior at the bed surface and dripper walls
- capillary and wetting effects in the coffee bed

The plan should treat it as two related but distinct effects:

- `free-surface tension`: an explicit surface-tension term or curvature-driven force in the free-water solver
- `bed wetting/capillarity`: a capillary-pressure or wetting term in the coffee-bed particle model

Delivery plan:

- M1-M2 may use a no-surface-tension approximation for basic solver bring-up
- M3-M4 must evaluate whether the jet and bed-interface behavior are unrealistic without it
- if the answer is yes, add a targeted surface-tension model before declaring the inflow and bed coupling physically credible

## Rendering Plan

Rendering should be a client of simulation, not part of the solver design.

### Water Rendering

Near-term:

- particle splats or spheres for debugging

Mid-term:

- screen-space fluid rendering
- surface reconstruction for pooled water

### Coffee Bed Rendering

Near-term:

- static or gently deforming bed mesh
- color/height overlays for saturation and extraction debugging

Mid-term:

- visually wetting bed surface
- crater and bloom cues from actual simulation state

### Debug Views

We should add debug views early:

- particles only
- grid occupancy
- inflow patch
- SDF normals
- bed saturation
- bed permeability
- extraction field

## WebGPU and Performance Strategy

The engine must stay browser-native and interactive, which means we need discipline up front.

## Performance Principles

- keep passes regular
- avoid CPU readback in the hot path
- size the grid to the active domain
- expose quality tiers
- profile each compute pass

## Timestep Policy

Do not hardcode an overly aggressive timestep.

Use:

- adaptive or capped timestep
- substeps when flow or impact velocity gets large
- conservative defaults for laptop GPUs

## Grid Strategy

Do not lock the architecture to a demo-style fixed `64^3` box.

Possible strategies:

- dense fixed grid for first implementation
- region-sized grid around the active pour-over apparatus
- sparse tiles or bricks if dense-grid memory or bandwidth fails the browser budget

Dense grids are acceptable only as an early constrained choice, not as a hidden long-term commitment.

Design constraint:

- the data layout for M1 must keep sparse or tiled grids possible later without a full solver rewrite

That means:

- separate active simulation fields from rarely updated material fields where possible
- prefer structure-of-arrays layouts over monolithic per-cell structs
- treat the active-domain box as movable/resizable from the start
- make dense-grid viability a Milestone 1 and Milestone 2 checkpoint, not a Milestone 5 surprise
- avoid baking long-lived coffee-bed state into every hot grid buffer when that state can live on bed particles instead

Memory budget rule of thumb:

- M1 default mode should target well under `64 MB` of hot simulation buffers
- higher tiers should be allowed to grow, but not by baking rarely used fields into every hot grid pass

## Data Layout Priorities

- tightly packed structs
- avoid unnecessary matrix precision if reduced forms work
- keep render packing separate from sim storage
- keep hot grid buffers separate from cold material/debug buffers

## P2G Accumulation Strategy

WebGPU's lack of universal `atomicAdd` support for floats is a first-class design constraint, not an implementation detail.

Milestone 1 should assume:

- `P2G` mass and momentum accumulation use fixed-point integer atomics
- grid mass and each momentum component are accumulated in separate integer buffers
- fixed-point values are decoded back to float during grid normalization and update

Initial implementation direction:

- do not hard-code the fixed-point scale until we derive bounds from the chosen particle mass normalization, velocity caps, and per-cell occupancy
- accumulate `mass`, `momentum_x`, `momentum_y`, and `momentum_z` separately
- normalize particle mass units so per-cell accumulation stays in a numerically safe range
- keep stress transfer simple and conservative before introducing more aggressive packing tricks

Required pre-implementation analysis:

- choose an initial velocity cap for the free-water solver
- estimate worst-case particle contributions per cell under the selected spacing and interpolation stencil
- derive maximum possible per-cell mass and momentum before choosing the fixed-point scale
- add explicit overflow instrumentation in debug builds before trusting the path

The fixed-point scale should be a derived result of those bounds, not a guessed constant.

Optimization order:

1. derive safe fixed-point bounds and ship a correct atomic path
2. add particle binning or sorting early enough to achieve coherent `P2G` access
3. add workgroup-local accumulation to reduce global atomic pressure
4. restrict dispatch to the active domain before considering sparse-grid complexity

Particle binning should not be treated as a late luxury. If unsorted `P2G` misses the budget, cell or tile binning becomes part of Milestone 1 exit criteria.

## Quality Tiers

We should plan for tiers from the beginning:

- low: fewer particles, smaller grid, fewer substeps
- medium: default laptop mode
- high: stronger GPU mode

## Numerical Targets

We should make the first browser targets concrete enough to guide tradeoffs.

Initial development target:

- grid: `48 x 48 x 64`
- water particles: `20k-40k`
- substeps: `2`
- simulation budget: `<= 12 ms`
- total frame budget: `<= 16.6 ms` at `60 FPS`

Default interactive target:

- grid: `64 x 64 x 96`
- water particles: `40k-80k`
- substeps: `2-3`
- simulation budget: `<= 16 ms`
- total frame budget: `<= 25 ms`

High-quality target:

- grid: `96 x 96 x 128`
- water particles: `80k-150k`
- substeps: `3-4`
- target framerate: `30 FPS` on stronger GPUs

These targets are intentionally sized for a browser-native free-surface solver, not inherited from an older grid-first porous-flow design.

## Validation and Success Criteria

We need a way to tell whether the new engine is better, not just more complex.

## Test Infrastructure

The validation numbers are only meaningful if we build the harness to measure them.

Required infrastructure:

- deterministic benchmark scenes for water-only, geometry, bed infiltration, and extraction
- automated regression tests for conservation and solver stability
- performance benchmarks that record per-pass timings, particle counts, and grid dimensions
- debug counters for overflow, NaNs, clamped velocities, and deleted/seeded particles
- CI coverage for at least CPU-side reference checks and non-browser smoke tests

Core benchmark scenarios:

- box rest test
- no-bed inflow ballistic landing test
- V60/carafe impact and collection test
- 1D or column infiltration test
- oversaturation and exfiltration interface test
- extraction sanity scenario with known relative outcomes across pour patterns

Each milestone should add or update benchmarks, not rely on manual visual inspection alone.

## Water-Only Validation

- total mass error remains below `0.1%`
- with no bed present, inflow landing point stays within `1` grid cell of the expected ballistic impact region
- closed-container rest tests show no unbounded energy growth over `10 s` of simulated time
- water collides with V60 and carafe geometry without persistent penetration through SDF boundaries
- pooled water accumulates without solver freeze or runaway velocity spikes

## Coffee Validation

- 1D or column-style infiltration tests match the selected bed model's reference solution within `5%`
- total free-water mass absorbed by the bed matches bed water-content increase within `0.1%`
- dry bed resists infiltration more strongly than pre-wet bed in the same geometry
- changing grind or permeability produces measurable drawdown differences of at least `10%` in calibration scenarios
- uneven pours produce persistent saturation nonuniformity rather than instantly homogenizing

## Product Validation

- default quality tier remains interactive within the frame budgets defined above
- no visual-only hacks are required for cup fill, bed wetting, or stream shape
- kettle-angle changes produce believable changes in flow rate, impact footprint, and drawdown response
- once extraction is enabled, cumulative cup metrics trend toward physically plausible brew targets:
  - cumulative TDS: `1.15-1.45%`
  - extraction yield: `18-22%`
  - spiral pours should outperform center-only or edge-only pours in uniformity

## Proposed Repo Migration

This should be a clean rebuild with controlled cutovers, not a permanent dual-engine migration.

## Cleanup Roadmap

The clean-build policy should translate into an explicit deletion sequence.

### Preserve

Keep only the parts of the current browser app that are not solver-specific:

- `crates/sim-wasm/src/lib.rs` as the WASM binding surface, but rewrite its simulation-facing internals
- `crates/sim-wasm/src/renderer.rs` only to the extent that it remains a generic particle/grid renderer
- `crates/sim-wasm/src/sdf.rs` if its geometry sampling utilities remain useful for MPM boundaries
- the `www-3d` app shell, controls, and page structure

### Replace Early

These should be treated as replacement targets immediately:

- `crates/sim-wasm/src/gpu_sim_3d.rs`
- all SPH-specific bindings and settings in `crates/sim-wasm/src/lib.rs`
- all renderer assumptions that depend on SPH-only particle semantics

### Delete Once The New Path Is Viable

Do not delete the legacy solver as soon as the new path merely renders.

Delete it only after the new path can support the water-only development loop credibly:

- delete `crates/sim-wasm/src/gpu_sim_3d.rs`
- remove `mod gpu_sim_3d;` and its imports from `crates/sim-wasm/src/lib.rs`
- remove `renderer.rs` dependencies on `CoffeeGpuSim3D`, `Obstacle`, and `SimSettings3D`
- remove any remaining SPH-only UI state, metrics, and emitter code paths from the WASM boundary

Deletion gate:

- pause/reset works
- interactive stepping works
- debug particle rendering works
- geometry collision works
- performance instrumentation exists
- the new path is usable for daily development feedback, not just tech-demo screenshots

As soon as kettle inflow and geometry are working in the new solver:

- delete any remaining legacy emitter constants, recipe plumbing, and stream helpers that existed only to support the old SPH path
- remove dead code in the web UI that was only there to paper over SPH limitations

### Do Not Carry Forward

The following should not survive the rebuild unless they are rewritten for the new engine:

- fallback solver toggles
- compatibility shims between SPH and MPM state
- duplicate render paths that exist only to keep the old solver alive
- proxy visualizations that are not backed by the new simulation state

## Initial `mpm_3d` Module Layout

The first clean layout should separate simulation passes from bindings and rendering.

Suggested initial structure:

- `crates/sim-wasm/src/mpm_3d/mod.rs`
  - public simulation entry point and high-level orchestration
- `crates/sim-wasm/src/mpm_3d/state.rs`
  - particle buffers, grid buffers, uniforms, and quality-tier configuration
- `crates/sim-wasm/src/mpm_3d/pipelines.rs`
  - WebGPU pipeline creation, bind groups, and pass dispatch helpers
- `crates/sim-wasm/src/mpm_3d/inflow.rs`
  - kettle-angle to inflow mapping and outlet patch setup
- `crates/sim-wasm/src/mpm_3d/boundary.rs`
  - SDF boundary sampling, friction, and solid projection logic
- `crates/sim-wasm/src/mpm_3d/bed.rs`
  - coffee-bed particle state, pore-water storage, drag, compaction, and absorption/exfiltration logic
- `crates/sim-wasm/src/mpm_3d/debug.rs`
  - debug counters, optional probes, and visualization-oriented extracts
- `crates/sim-wasm/src/mpm_3d/shaders/*.wgsl`
  - `clear_grid.wgsl`
  - `inflow.wgsl`
  - `p2g_mass_momentum.wgsl`
  - `p2g_stress.wgsl`
  - `grid_update.wgsl`
  - `boundary_project.wgsl`
  - `g2p.wgsl`
  - later `bed_coupling.wgsl`, `extraction_advect.wgsl`

### WASM Binding Cutover

`crates/sim-wasm/src/lib.rs` should move toward a thin role:

- own the wasm-bindgen surface
- instantiate `mpm_3d`
- expose settings, stepping, metrics, and debug views
- avoid embedding solver logic directly in the binding file

### Renderer Cutover

`crates/sim-wasm/src/renderer.rs` should become solver-agnostic:

- accept generic particle buffers and optional grid/bed debug overlays
- stop importing SPH-specific simulation types
- treat the simulation as a data provider, not as a type-level dependency

## Step 1: Replace the Simulation Backend With a New MPM Path

Introduce the new browser-native solver as the canonical simulation backend:

- `crates/sim-wasm/src/mpm_3d/`
- supporting WGSL modules under `crates/sim-wasm/src/mpm_3d/shaders/`

Do not keep the old SPH path around past the deletion gate described above.

## Step 2: Keep the App Shell, Replace the Solver Internals

Preserve only the parts that are still clearly useful:

- UI controls
- camera/orbit controls
- render loop shell
- high-level WASM bindings

Do not preserve simulation compatibility layers beyond what is required to land the replacement.

## Step 3: Implement Water-Only Box MPM

Before adding V60 geometry:

- get stable MPM particles
- validate P2G/G2P loop
- validate gravity and box bounds

## Step 4: Add SDF Boundaries

Port the geometry infrastructure over:

- dripper
- filter
- carafe

Validate boundary behavior before adding coffee-bed logic.

## Step 5: Add Kettle Inflow Boundary

Implement a proper grid-native inflow driven by kettle angle and flow rate.

At this point, all remaining legacy emitter logic should be removed.

## Step 6: Add Porous Bed State

Introduce coffee-bed particles and infiltration logic without yet solving extraction.

## Step 7: Add Extraction Scalars

Once wetting and drawdown are convincing, layer in brew-relevant scalar transport.

## Milestones

### Milestone 1 Design Decisions

These choices should not remain open past Milestone 1:

- use `APIC/MLS-MPM` rather than plain PIC-style transfers
- use one dense shared grid for water and porous-bed coupling in the first implementation
- use fixed-point integer atomics for `P2G`
- use SDF boundaries from the start rather than box-only temporary boundaries
- keep conservation, coupling, and browser viability as the Milestone 1 decision anchors
- determine whether particle binning is required to hit the baseline frame budget
- determine whether the dense-grid memory budget is viable for the default tier
- keep long-lived coffee-bed state on bed particles unless profiling proves a grid copy is necessary

### Milestone 1: Water-Only MPM in a Box

Priority: `MVP-blocking`

Deliverables:

- stable water solver
- browser-native compute path
- debug particle rendering
- validated fixed-point `P2G` accumulation path
- measured frame-time breakdown against the initial development target
- measured memory footprint against the default-tier budget
- baseline benchmark and regression harness

### Milestone 2: Water + V60/Carafe Geometry

Priority: `MVP-blocking`

Deliverables:

- SDF geometry
- stable stream impact
- correct collection in the carafe
- verified SDF collision behavior under fast inflow

### Milestone 3: Kettle-Angle Inflow

Priority: `MVP-blocking`

Deliverables:

- angle-controlled flow rate
- angle-controlled outlet momentum
- stable inflow boundary
- resolved choice between pure grid inflow and grid inflow plus particle seeding

### Milestone 4: Porous Coffee Bed

Priority: `Core realism`

Deliverables:

- coffee-bed particles
- infiltration
- drawdown behavior
- validated coupling against dedicated bed-flow and extraction test scenarios
- explicit decision on whether the shared grid remains sufficient for bed coupling

### Milestone 5: Extraction

Priority: `Core realism`

Deliverables:

- dissolved-solids transport
- output metrics
- physically grounded cup statistics
- quantitative comparison to chosen brew-quality targets and bench scenarios

### Milestone 6: Higher-Fidelity Bed and Rendering

Priority: `Polish / realism extension`

Deliverables:

- bed compaction/deformation
- richer surface rendering
- stronger brew realism

## Decisions To Resolve By Milestone

These are still open, but they are scheduled decisions rather than indefinite questions.

Resolve by the end of `Milestone 1`:

- confirm that `APIC/MLS-MPM` is stable enough for the water-only path, or escalate to a stronger variant before geometry work continues
- confirm the fixed-point encoding scale and overflow margins for `P2G`
- confirm that one dense shared grid is sufficient for the initial apparatus size
- confirm whether particle binning is required for baseline performance
- confirm the hot-buffer memory footprint against the browser budget

Resolve by the end of `Milestone 3`:

- decide whether inflow remains pure grid injection or requires persistent particle seeding for advection and rendering quality

Resolve by the end of `Milestone 4`:

- decide whether the coffee-bed particles can continue sharing the same grid as free water or need an auxiliary structure
- decide whether Stage A or Stage B bed behavior is sufficient for realistic drawdown and channeling

Resolve by the end of `Milestone 5`:

- decide whether dense grids remain viable or whether sparse tiling is now justified by the measured active domain

## Immediate Next Step

The next engineering step should be:

- create `mpm_3d` as the new canonical simulation backend
- implement a minimal water-only `clear -> P2G -> grid update -> G2P -> render copy` loop
- keep geometry simple at first
- prove browser stability before layering in coffee physics

That is the best way to move fast without giving up the long-term realism target.

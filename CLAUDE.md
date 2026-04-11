# CLAUDE.md

This project is a browser-native physics simulation of pour-over coffee brewing.

## Coding Best Practices

- ALWAYS keep @CHANGELOG.md up to date with recent changes, making note of implementation mistakes so that they are not repeated.
- Don't tweak parameters or add heuristics to resolve fundamental issues with the physics engine.

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

## Simulation State

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


# Long-Term Plan

## Goal

Build a browser-native pour-over simulator where the visible behavior comes from
one coherent physical model:
- coherent kettle jet
- near-incompressible free water
- deformable porous coffee bed
- filter-mediated drawdown
- extraction layered on top later

## Solver Direction

The current architectural commitment is:
- water remains explicit particles on a shared MPM grid
- pressure projection enforces near-incompressibility
- the bed is a separate material layer on the same world state
- rendering remains downstream of solver state

The intended milestones are:
1. strong water incompressibility and stable free-surface behavior
2. physically cleaner water/bed coupling
3. believable pooled water above the bed
4. bed storage, drainage, and drawdown
5. filter support / filter-mediated outflow
6. extraction on top of the hydraulic state

## Constraints

- stay browser-native
- keep passes regular and GPU-friendly
- keep simulation truth separate from rendering
- prefer explicit conservation over heuristic visual hacks

## Current Open Direction

The current branch work is still pushing toward:
- better free-flight jet cohesion
- more physical bed compaction and shear behavior
- better filter contact and drawdown
- stronger validation and regression coverage

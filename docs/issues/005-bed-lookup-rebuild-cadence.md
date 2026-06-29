# Avoid rebuilding the full bed lookup every substep

## Tags

- `coffee:bed-coupling`
- `physics:solver-performance`

## Severity

Medium-High

## Problem

The bed lookup is cleared across the full grid and rebuilt every substep even
though the coffee bed usually changes more slowly than the water. This creates
extra grid-wide work and couples bed lookup cost to the simulation substep rate.

## Proposed fix

Investigate a bed lookup cache with explicit invalidation:

1. Rebuild only when bed particles move far enough to change cell occupancy.
2. Consider a lower-frequency rebuild cadence for stable bed scenes.
3. Separate static porous-bed occupancy from dynamic bed particle deformation.
4. Keep bed-water coupling deterministic enough for existing physics tests.

## Acceptance criteria

- Bed lookup rebuilds less often in stable scenes.
- Coffee-bed coupling and extraction behavior remain visually and numerically
  consistent.
- Debug scenes that stress bed motion still update lookup correctly.

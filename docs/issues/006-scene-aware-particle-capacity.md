# Use scene-aware or growable particle capacities

## Tags

- `physics:memory-layout`
- `ui:device-compatibility`

## Severity

Medium

## Problem

Particle-related GPU buffers are allocated for the scene's maximum particle
capacity up front. The default capacity is large enough for the full V60 scene,
but smaller debug scenes and water-only scenarios do not need the same memory
budget.

## Proposed fix

1. Define per-scene capacity budgets for water and bed particles.
2. Consider separate water and bed capacities instead of one shared maximum.
3. Optionally support grow-on-demand reallocation for long-running scenes.
4. Surface capacity/dropped-particle diagnostics clearly in the UI.

## Acceptance criteria

- Smaller scenes allocate smaller particle buffers.
- Default V60 behavior and capacity diagnostics remain unchanged or improve.
- Reallocation, if implemented, does not occur silently during hot frames without
  diagnostics.

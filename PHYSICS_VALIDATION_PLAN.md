# Physics Validation Plan

## Goal

Turn the browser app and native GPU tests into a more systematic validation
surface for the MPM solver.

## Near-Term Validation Targets

- water mass conservation
- hydrostatic rest stability
- free-stream continuity
- center-pour pool stability
- dry-bed settle stability
- bounded long-run dry-bed creep

## Tooling Direction

Short term:
- keep using `physics_tests.rs` as the authoritative regression surface
- expand scene-specific readback helpers only when they lock meaningful behavior

Later:
- add a browser-visible validation grid or dedicated validation mode
- expose clearer solver metrics and residuals
- compare scene outcomes against explicit acceptance thresholds

## Acceptance Principle

Validation should lock:
- conservation
- stability
- scene-level plausibility

Validation should not lock:
- temporary heuristics that are expected to be replaced
- branch-specific tuning that does not represent the intended architecture

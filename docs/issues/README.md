# Proposed Issues

These files are the source bodies for GitHub issues created with
`scripts/create-github-issues.sh`. The sparse-grid / active-cell redesign is
intentionally omitted here because it is already being worked on.

| Issue | Severity | Tags |
| --- | --- | --- |
| [Run `prepare_render` once per frame](001-prepare-render-once-per-frame.md) | High | `physics:solver-performance`, `ui:render-performance`, `ui:diagnostics` |
| [Reduce pressure solve cost without depending on sparse-grid work](002-pressure-solve-cost.md) | High | `physics:solver-performance`, `physics:numerics` |
| [Split the monolithic MPM bind group by pass family](003-split-mpm-bind-groups.md) | High | `physics:memory-layout`, `ui:device-compatibility` |
| [Reduce fixed-point atomic P2G contention](004-reduce-p2g-atomic-contention.md) | High | `physics:solver-performance`, `physics:numerics`, `physics:memory-layout` |
| [Avoid rebuilding full bed lookup every substep](005-bed-lookup-rebuild-cadence.md) | Medium-High | `coffee:bed-coupling`, `physics:solver-performance` |
| [Use scene-aware or growable particle capacities](006-scene-aware-particle-capacity.md) | Medium | `physics:memory-layout`, `ui:device-compatibility` |
| [Compact visible render instances](007-compact-visible-render-instances.md) | Medium | `ui:render-performance`, `ui:device-compatibility` |
| [Move inflow emission onto the GPU](008-gpu-side-inflow-emission.md) | Medium | `physics:inflow`, `physics:solver-performance` |
| [Decouple diagnostics from hot simulation/render passes](009-decouple-diagnostics-from-hot-passes.md) | Medium | `ui:diagnostics`, `physics:solver-performance` |
| [Separate water and bed data paths](010-separate-water-bed-data-paths.md) | Medium-High | `physics:memory-layout`, `coffee:bed-coupling`, `coffee:extraction-state` |

## Deferred / intentionally not filed

- Sparse active-cell execution, sparse clearing, and active-cell classification
  are excluded because that work is already in progress.
- Dirty rendering while paused is below the medium-severity cutoff for this
  issue pass.

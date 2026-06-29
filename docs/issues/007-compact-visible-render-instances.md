# Compact visible render instances

## Tags

- `ui:render-performance`
- `ui:device-compatibility`

## Severity

Medium

## Problem

Rendering uses `render_data` as an instance buffer and draws one quad per active
simulation particle count. Inactive or visually hidden particles are represented
with sentinel render data, but the renderer can still pay vertex/instance cost
for entries that do not contribute to the image.

## Proposed fix

1. Add a GPU compaction step that writes only visible particles into a compact
   render-instance buffer.
2. Use an indirect draw count if supported, or expose a copied visible count to
   the CPU at a low cadence if indirect draw is not viable.
3. Consider separate render paths for bed and water so dense bed particles can
   use a cheaper representation.

## Acceptance criteria

- Visual output matches the current particle rendering for visible particles.
- Scenes with many inactive or hidden particles submit fewer render instances.
- The fallback path remains compatible with WebGPU adapters that lack optional
  features.

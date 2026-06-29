# Separate water and bed data paths

## Tags

- `physics:memory-layout`
- `coffee:bed-coupling`
- `coffee:extraction-state`

## Severity

Medium-High

## Problem

Water and bed particles share several buffer layouts and dispatch paths. Some
water-specific kernels launch across water plus bed particles and then branch out
for non-water phases. This keeps indexing simple, but it mixes two materials
with different update cadence, render needs, and state ownership.

## Proposed fix

Evaluate splitting water and bed data more explicitly:

1. Separate water and bed particle ranges, buffers, or views.
2. Dispatch water-only kernels over water particles only.
3. Keep bed/extraction state in bed-specific buffers with clear ownership.
4. Preserve coupling paths through lookup buffers or explicit overlap lists.

## Acceptance criteria

- Water-only passes no longer pay per-bed-particle branch overhead.
- Bed/extraction state ownership becomes clearer.
- Existing coffee-bed and water-only debug scenes remain valid.

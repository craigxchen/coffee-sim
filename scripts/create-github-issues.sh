#!/usr/bin/env bash
set -euo pipefail

# Create the MPM performance labels and issues tracked in docs/issues.
#
# Usage:
#   scripts/create-github-issues.sh OWNER/REPO
#
# Requirements:
#   - GitHub CLI (`gh`) installed
#   - `gh auth login` completed with issue write access
#
# The sparse active-cell/grid work is intentionally omitted because it is
# already being worked on.

repo="${1:-${GH_REPO:-}}"
if [[ -z "$repo" ]]; then
  echo "usage: $0 OWNER/REPO" >&2
  echo "or set GH_REPO=OWNER/REPO" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI (gh) is required" >&2
  exit 127
fi

gh auth status >/dev/null

ensure_label() {
  local name="$1"
  local color="$2"
  local description="$3"
  if gh label list --repo "$repo" --search "$name" --json name --jq '.[].name' | grep -Fxq "$name"; then
    gh label edit "$name" --repo "$repo" --color "$color" --description "$description"
  else
    gh label create "$name" --repo "$repo" --color "$color" --description "$description"
  fi
}

create_issue() {
  local title="$1"
  local body_file="$2"
  shift 2
  local labels=()
  for label in "$@"; do
    labels+=(--label "$label")
  done

  if gh issue list --repo "$repo" --state all --search "${title} in:title" --json title --jq '.[].title' | grep -Fxq "$title"; then
    echo "skip existing issue: $title"
  else
    gh issue create --repo "$repo" --title "$title" --body-file "$body_file" "${labels[@]}"
  fi
}

ensure_label "physics:solver-performance" "1D76DB" "Core MPM transfer, grid, pressure, or scheduling performance"
ensure_label "physics:numerics" "0052CC" "Pressure, viscosity, stability, fixed-point ranges, or convergence"
ensure_label "physics:memory-layout" "5319E7" "GPU buffer layout, bind groups, particle layout, or capacity management"
ensure_label "physics:inflow" "0E8A16" "Spout emission, particle allocation, or inlet scheduling"
ensure_label "coffee:bed-coupling" "A0522D" "Porous-bed lookup, bed-water exchange, bed impulse transfer, or bed motion"
ensure_label "coffee:extraction-state" "D4A72C" "Solute, saturation, retained water, or extraction metric/state ownership"
ensure_label "ui:render-performance" "FBCA04" "Visual rendering performance without changing simulation truth"
ensure_label "ui:diagnostics" "BFDADC" "Debug metrics, readback cadence, profiling, HUD, or timeseries overhead"
ensure_label "ui:device-compatibility" "C5DEF5" "Constrained WebGPU adapters, integrated GPUs, or browser memory budgets"

create_issue "Run prepare_render once per frame instead of once per substep" \
  docs/issues/001-prepare-render-once-per-frame.md \
  "physics:solver-performance" "ui:render-performance" "ui:diagnostics"
create_issue "Reduce pressure solve cost without depending on sparse-grid work" \
  docs/issues/002-pressure-solve-cost.md \
  "physics:solver-performance" "physics:numerics"
create_issue "Split the monolithic MPM bind group by pass family" \
  docs/issues/003-split-mpm-bind-groups.md \
  "physics:memory-layout" "ui:device-compatibility"
create_issue "Reduce fixed-point atomic P2G contention" \
  docs/issues/004-reduce-p2g-atomic-contention.md \
  "physics:solver-performance" "physics:numerics" "physics:memory-layout"
create_issue "Avoid rebuilding the full bed lookup every substep" \
  docs/issues/005-bed-lookup-rebuild-cadence.md \
  "coffee:bed-coupling" "physics:solver-performance"
create_issue "Use scene-aware or growable particle capacities" \
  docs/issues/006-scene-aware-particle-capacity.md \
  "physics:memory-layout" "ui:device-compatibility"
create_issue "Compact visible render instances" \
  docs/issues/007-compact-visible-render-instances.md \
  "ui:render-performance" "ui:device-compatibility"
create_issue "Move inflow emission onto the GPU" \
  docs/issues/008-gpu-side-inflow-emission.md \
  "physics:inflow" "physics:solver-performance"
create_issue "Decouple diagnostics from hot simulation/render passes" \
  docs/issues/009-decouple-diagnostics-from-hot-passes.md \
  "ui:diagnostics" "physics:solver-performance"
create_issue "Separate water and bed data paths" \
  docs/issues/010-separate-water-bed-data-paths.md \
  "physics:memory-layout" "coffee:bed-coupling" "coffee:extraction-state"

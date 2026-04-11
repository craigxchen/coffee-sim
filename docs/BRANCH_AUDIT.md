# Branch Audit

Baseline for all decisions: `origin/main` at `0f3d345`.

## Branch Roles

| Branch | Role | Status |
| --- | --- | --- |
| `origin/main` | authoritative current mainline | keep |
| `codex/incompressible-rewrite` | historical source branch for the MPM rewrite | inspect-only |
| `dual-velocity-field` | mixed branch: one perf commit, otherwise experimental coupling work | selective |
| `dual-velocity-field-codex` | comparison branch superseded by later work | superseded |
| `grind-size-bed-drop` | experimental bed hydraulics/mechanics/UI branch | experimental |

## Commit Decisions

### `codex/incompressible-rewrite`

| Commit | Decision | Reason |
| --- | --- | --- |
| `38ea5ca` | defer | not in `origin/main`; introduces grid-coupled bed particles but is explicitly described as a bandaid |
| older commits already merged via `origin/main` | keep in baseline | already represented in current mainline |

Notes:
- do not branch from `codex/incompressible-rewrite`
- use it only to identify local deltas not already merged into `origin/main`

### `dual-velocity-field`

| Commit | Decision | Reason |
| --- | --- | --- |
| `3f1c873` | keep for mainline integration | isolated performance/runtime improvement set |
| `4ac6275` | defer | experimental bed pore-water redistribution |
| `d96fde3` | defer | solver/behavior change, not just cleanup |
| `c07433e` | defer | dual-field bed-coupling experiment |
| `5648e5d` | superseded | older variant of the same dual-field work |
| `2f6b322` | drop | merge artifact, not an integration target |
| `38ea5ca` | defer | inherited experimental bed rewrite commit |

### `dual-velocity-field-codex`

| Commit | Decision | Reason |
| --- | --- | --- |
| branch-only content | superseded | no unique mainline target until a concrete missing fix is identified |

### `grind-size-bed-drop`

| Commit | Decision | Reason |
| --- | --- | --- |
| `2477f42` | keep on branch only | snapshot commit preserving experimental branch state |
| `ddc0877` | defer | dry bed plastic yield projection is experimental |
| `60912e7` | defer | dry bed deformation-state work is experimental |
| `4ac6275` | defer | inherited redistribution work remains experimental |
| `3f1c873` | keep for mainline integration | inherited stable perf commit |
| remaining inherited dual-field commits | defer | solver realism work still under evaluation |

## Subsystem Decisions

| Subsystem | Source | Decision |
| --- | --- | --- |
| Core MPM browser stack | `origin/main` | keep as baseline |
| Legacy SPH-era runtime/demo paths | pre-MPM history | remove from active mainline |
| Shared `sim-core::sph` math/types | `origin/main` | keep for now; rename/extract later if desired |
| Perf hot-path wins | `3f1c873` | integrate into mainline |
| Dual-field water/bed coupling | `dual-velocity-field` | experimental |
| Bed redistribution | `4ac6275` | experimental |
| Grind-size-dependent bed hydraulics | `grind-size-bed-drop` | experimental |
| Bed deformation + granular plasticity | `grind-size-bed-drop` | experimental |
| Rigid-support scene / extra UI | `grind-size-bed-drop` | experimental until split from solver work |

## Mainline Integration Checklist

Mainline-ready now:
- docs refresh in this branch
- architecture note
- branch audit ledger
- changelog cleanup
- isolated perf commit `3f1c873`

Not mainline-ready yet:
- dual velocity fields
- redistribution and grind-size behavior
- bed deformation/plasticity work
- mixed experimental UI from `grind-size-bed-drop`

## Merge Mechanism

Use curated cherry-picks onto `origin/main`, not branch merges or rebases of the whole experimental branches.

Reason:
- `dual-velocity-field` and `grind-size-bed-drop` mix stable and experimental work
- both branches include solver changes that should not ride into `main` implicitly
- `origin/main` already contains the real MPM baseline

## Release Gates

Before merging audited mainline changes:
- `cargo test -p coffee-sim-wasm --lib`
- `cargo check -p coffee-sim-wasm --target wasm32-unknown-unknown`
- `wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg`
- `cargo fmt --check`
- `cargo clippy -p coffee-sim-wasm -- -D warnings`
- browser smoke pass for default, free-stream, and center-pour scenes

Before promoting experimental bed-physics work:
- full native test suite
- same wasm build
- browser checks for bed settle, rigid support, drawdown, and pour interaction
- explicit changelog entry for unresolved realism gaps

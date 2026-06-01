# AGENTS.md

Repository-wide instructions for coding agents.

## Repository Map

`coffee-sim` is a single Rust + `wgpu` + WGSL crate; the repo root is both the workspace
and the package. Modular blocks under `src/`:

- `engine/` — orchestration glue: `Scene`, `Simulator`, `State`/`Metrics`, the solver
  `registry`, and the solver-description catalog `engine/solvers.json`.
- `solvers/` — each complete fluid-sim algorithm is a `Solver` (the seam is `base.rs`);
  `noop.rs` are the harness validators. XPBD (primary), MPM, SPH+MPM land here.
- `models/` — shared physics (permeability, extraction, cohesion, wetting, thermal) +
  `Materials`, imported by every solver so cross-solver comparison is fair.
- `emission/` — particle emission (coffee + water), independently testable.
- `profiling/` — local `timestamp-query` + dispatches-per-frame profiler.
- `ui/` — rendering + controls + overlays; consumes only canonical state (one-way data flow).
- `utils/` — shared infra + config: `GpuContext`, spatial hash, SDF, kernels, seeded RNG,
  geometry builders, `Config`, `ParticleBuffers`.

`docs/ARCHITECTURE.md` is authoritative; sub-module plans live in `docs/plans/`. Salvaged
v1 reference values (calibration, units, kernel/SDF math, geometry, target bands) are in
`KEEP.md`. The superseded v1 MPM implementation remains on `origin/*` branches.

## Working Rules

- Prefer the smallest correct change over broad refactors.
- Keep physics fixes structural; avoid tuning-only patches for fundamental issues.
- **Never delete or `#[ignore]` tests to go green.** Tests lock conservation/stability/
  plausibility, not temporary heuristics.
- Keep simulation truth out of `ui` — solvers own physical state.
- Stay WASM/WebGPU-portable: target `wgpu::Limits::default()` (WebGPU baseline) and keep
  bind groups within 8 storage buffers per stage. Flag any native-only feature.
- `todo!()` stubs must compile and be inert — no half-wired pipelines.

## Verification

Start with the narrowest relevant check, then before landing a branch:

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo run --example phase0_noop`  (headless; skips when no GPU adapter is present)

## References

- Start with `README.md`
- Then use `docs/ARCHITECTURE.md`

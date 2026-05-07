# AGENTS.md

Repository-wide instructions for coding agents.

## Project Commands

- Never run `cargo insta accept` without explicit user approval.

## Repository Map

- `crates/sim-core` is shared math/types for the browser solver.
- `crates/sim-wasm` owns the WebGPU renderer and the MPM simulation under `mpm_3d/`.

## Working Rules

- Prefer the smallest correct change over broad refactors.
- Keep physics fixes structural; avoid tuning-only patches for fundamental issues.
- Do not edit generated artifacts unless the task specifically requires it.
- Keep documentation changes concrete and tied to real behavior.

## Verification

- Start with the narrowest relevant check.
- Before landing a meaningful branch, run:
  - `cargo fmt --check`
  - `cargo clippy -p coffee-sim-wasm -- -D warnings`
  - `cargo test -p coffee-sim-wasm --lib`
  - `wasm-pack build crates/sim-wasm --target web --release --out-dir www-3d/pkg`

## References

- Start with `README.md`
- Then use `docs/ARCHITECTURE.md` and `docs/ROADMAP.md`

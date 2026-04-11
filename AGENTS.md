# AGENTS.md

This file gives repository-wide instructions to coding agents. Keep it focused on stable, high-value guidance; add a nested `AGENTS.md` only when a subdirectory truly needs different rules.

## Project Commands

Never run `cargo insta accept` without explicit user approval.

## Repository Map

- `crates/sim-core` is a shared Rust math/settings/pour-scripting library
- `crates/sim-wasm` is the browser WASM binding that owns the WebGPU renderer and the MLS-MPM solver under mpm_3d/, implementing the water + bed coupling described in CLAUDE.md.

## Working Rules

- Prefer the smallest correct change over broad refactors.
- Match the existing crate and module boundaries unless a structural change is clearly necessary.
- Avoid editing generated artifacts, vendored code, or snapshot outputs unless the task specifically requires it.
- Avoid heuristic changes or parameter fine-tuning.

## Documentation Rules

- Keep documentation updates concrete and example-driven; do not leave behavior changes documented only in code.

## Verification

- Run the narrowest relevant check first, usually `cargo test -p <crate>` or a focused `cargo run -p ...` command.
- Do not run the full workspace verification suite after every small edit.
- Before committing or pushing a meaningful batch of changes, run `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo nextest run --no-fail-fast`.
- If snapshot tests change, call that out clearly in your summary and leave acceptance to the user.

## References

- Start with `README.md` for the product-level overview.


You are rigorously reviewing an implementation plan for a Rust + wgpu 29 + WGSL real-time pour-over coffee simulator (the `coffee-sim` rewrite). The plan adds general static solid-object boundaries defined via analytic SDFs (a V60 dripper cone + grains-only filter cone + catch cup as the first instances) to an existing two-species XPBD GPU solver.

Respond with a verdict on the FIRST LINE: **APPROVE**, **REVISE**, or **REJECT**. Then explain, rigorously and specifically, citing the plan's section/unit IDs.

The reviewer should weigh these project-specific facts (all verified against the codebase):
- Two-species XPBD GPU solver: phase 0=water, 1=grain. WGSL split common/water/bed/coupling/wetting.wgsl, concatenated via include_str!. Counting-sort neighbor grid.
- HARD constraints: max 8 storage buffers PER SHADER STAGE (device limit pinned to wgpu defaults; raising it is forbidden — it was v1's fatal mistake); NO portable WGSL float atomics; `Params` uniform must stay EXACTLY 256 bytes, byte-identical Rust(`#[repr(C)]`)↔WGSL, enforced by `const _: () = assert!(size_of::<Params>()==256)`.
- The ONLY boundary handler today is an axis-aligned box clamp inside `apply_dp` (common.wgsl:300-336) and re-applied in `apply_drag_pred` (coupling.wgsl:372-379). The clamp delta gives the boundary normal; grains get Coulomb friction (`floor_mu`); velocity is derived from the position delta in `finalize` (never explicitly reflected). Every writer of pred/pos MUST preserve the `.w` lane (moisture).
- `apply_dp` currently uses 5/8 storage buffers; `apply_drag_pred` 3/8. Six kernels (bed_project, compute_lambda, compute_dp, drag_water, drag_grain, buoyancy_grain) are AT 8/8 and cannot take a new binding.
- Existing scenes (dam/pour/bed/water) and four GPU test suites (water/bed/coupling/wetting) must stay green; perf optimization is explicitly DEFERRED (do not optimize for speed).
- The user is a physics reviewer who rejects hand-waving and probes failure modes (clumping, artificial-pressure artifacts, conservation drift). Volume + momentum conservation are hard constraints.

Be especially critical of: (1) whether folding the SDF push-out into `apply_dp`/`apply_drag_pred` (vs a separate pass) is correct and conservation-clean, and any ordering hazard between SDF push-out and the box clamp; (2) the cavity-SDF sign convention and the true-Euclidean cone distance (vs v1's radial approximation) — correctness at the apex, open top, and apex hole; (3) the storage-buffer budget after adding binding 19 to those two passes; (4) the Params byte-match after renaming `_pad_bucket`→`num_solids` and the 64-byte Primitive struct; (5) selectivity correctness (filter blocks grains, passes water) and whether grains seeded inside the cone can escape; (6) momentum/volume conservation as water drains through the filter into the cup; (7) any missing failure mode, test gap, or unstated assumption.

## Plan to Review


APPROVE

The revised doc is ready to hand to `/ce-plan`. It now states the PB-MPM bet honestly: stable local stiffness and visible push-back at few iterations, not free globally-stiff incompressibility; KD2 correctly calls the coarse pre-pass approximate global low-frequency pressure work.

Prior REVISE items were addressed materially:

- KD5 is now a hypothesis with sign/unit/smoothness/iteration/stress-partition audits.
- The XPBD claim is corrected: `src/solvers/xpbd/coupling.wgsl:228` uses `max(-lambda[i], 0.0)`, and `:257` / `:303` apply pairwise `spiky_grad` impulses, not a grid `∇λ`.
- Candidate D is added and treated as the closest rival, with a coherent fallback if PB-MPM does not out-bounce it.
- KD8 now gates the assembled two-velocity cost, not just single-phase water.
- KD10 correctly records current `GpuContext` grant `9` at `src/utils/gpu.rs:114`, current two-field widest pass `7` at `src/solvers/twofield/common.wgsl:9`, and the sanctioned `16` in `AGENTS.md:34`.
- Fixed-point headroom for PB-MPM displacement/correction state is explicitly called out instead of hidden behind existing mass/momentum FP math at `src/solvers/twofield/common.wgsl:124`.
- R1/R3/R4 now have real metric surfaces and anti-gaming guards.
- The old “~70%” pressure claim is gone; “pressure-pair loop plausibly dominates” is appropriately softened. The old `main` branch does show 40–100 RBGS pressure pairs and red/black dispatches in `crates/sim-wasm/src/mpm_3d/mod.rs:332`, `:361`, `:428`, and `:1663`.
- The “publishable core” rhetoric is gone.

One note for `/ce-plan`, not a blocker: the buffer/pass ledger and exact numeric thresholds for “materially exceeds,” “plausible band,” and “bounded” still need to be made concrete there. As a requirements/brainstorm artifact, the architecture decision is now honest, falsifiable enough to plan against, and has coherent D/B fallbacks if KD8 or the PB-MPM-vs-D bounce comparison fails.

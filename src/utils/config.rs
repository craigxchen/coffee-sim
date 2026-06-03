//! Solver/simulation configuration: the deterministic seed + solver numerics.

/// Declarative configuration handed to a solver at build time.
///
/// Carries the deterministic `seed` (determinism is load-bearing for reproducible runs and
/// fair cross-solver comparison) and the PBF water-core numerics. Physical material params
/// live in [`crate::models::Materials`]; scene geometry/gravity live in
/// [`crate::engine::Scene`].
#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,

    /// Sub-steps per frame (smaller sub-`dt`); raise only if the CFL/displacement gate trips.
    pub substeps: u32,

    // --- adaptive constraint-projection iterations (GPU-side early-exit + hard cap) ---
    /// Minimum iterations before early-exit may trigger.
    pub min_iters: u32,
    /// Hard cap on iterations per (sub)step.
    pub max_iters: u32,
    /// Early-exit threshold on the max density residual `‖C‖∞` (e.g. 0.01 = 1% density error).
    /// Set to 0 to disable early-exit (always runs `max_iters`).
    pub residual_tolerance: f32,

    // --- PBF density solve ---
    /// Constraint-force-mixing relaxation `ε` added to the gradient-sum denominator.
    pub relaxation_eps: f32,
    /// Under-relaxation `ω` on the position correction `Δp` (damps overshoot/oscillation).
    pub position_relaxation: f32,

    // --- s_corr artificial pressure (anti-clumping) ---
    pub s_corr_k: f32,
    pub s_corr_n: f32,
    /// `Δq = s_corr_dq_ratio · h` (the reference distance in the s_corr ratio).
    pub s_corr_dq_ratio: f32,
    /// Spiky-gradient `r→0` clamp: `r_eff = max(r, spiky_r_min_ratio · h)`.
    pub spiky_r_min_ratio: f32,
    /// Clamp the density correction to compression-only (`λ ≤ 0`) — anti-clump fallback.
    pub lambda_clamp_noncohesive: bool,

    // --- artifact damping / stability ---
    /// XSPH velocity-smoothing coefficient (damps s_corr surface jitter).
    pub xsph_viscosity_c: f32,
    /// Per-iteration position-correction cap as a fraction of `h` — no single overcorrection
    /// can launch a particle (anti-eruption, the main blow-up guard).
    pub max_correction_ratio: f32,
    /// Global per-step velocity damping (energy bleed so the pool settles; `1.0` = none).
    pub velocity_damping: f32,
    /// Velocity clamp `‖v‖ ≤ max_speed` (CFL + anti-blow-up backstop).
    pub max_speed: f32,

    // --- neighbor grid ---
    /// Fixed per-cell bucket capacity `K` (overflow is a hard correctness failure).
    pub bucket_capacity: u32,

    // --- initial seeding ---
    /// Initial position jitter as a fraction of particle spacing (breaks lattice symmetry).
    pub seed_jitter: f32,

    // --- granular bed (grain phase) ---
    /// Iteration cap for the contact solve (contacts converge slower than the density solve).
    pub bed_max_iters: u32,
    /// Early-exit threshold on the max normalized penetration `overlap/d` (e.g. 0.02 = 2%).
    pub bed_residual_tolerance: f32,
    /// Rebuild the neighbor grid every this-many bed iterations (more frequent than water).
    pub bed_regrid_interval: u32,
    /// Grain static-yield dead-band (scene units/s): below this a grain is snapped to rest.
    /// Honest quasi-static regularization (gravity + contacts still evaluated) — removes the
    /// sub-threshold jitter a Jacobi contact pile never fully settles, without freezing.
    pub grain_sleep_speed: f32,

    // --- water/bed coupling (mixed scenes) ---
    /// Solid-fraction packing clamp on `α_s` (~0.64 random close packing).
    pub packing_limit: f32,
    /// Under-relaxation on the grain-exclusion position correction (A.2).
    pub exclusion_relax: f32,
    /// Drag rate scale: `γ = drag_gamma / k` (Kozeny–Carman k); higher = stiffer drag. (step 2)
    pub drag_gamma: f32,
    /// Per-particle accumulated-drag-blend cap `β_max < 1` (anti-overshoot). (step 2)
    pub drag_beta_max: f32,
    /// Drag Jacobi sub-iterations per frame. 0 disables drag. (step 2)
    pub drag_subiters: u32,
    /// Buoyancy impulse scale on the PBF `λ` pressure proxy. 0 disables buoyancy. (step 3)
    pub buoyancy_scale: f32,
    /// A grain skips its static dead-band when its frame fluid-impulse exceeds this (wake). (step 2)
    pub wake_threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0x00C0_FFEE_5EED,
            substeps: 1,
            min_iters: 1,
            // Tuned against the dam-break gate: a tall column needs enough iterations to
            // propagate pressure up the pool (early-exit keeps calm frames cheap).
            max_iters: 20,
            residual_tolerance: 0.01,
            relaxation_eps: 1.0e-3,
            // Under-relax the Jacobi position solve (ω<1) — stabilizes it and keeps
            // per-iteration moves within a grid cell (the grid is rebuilt once/frame).
            position_relaxation: 0.5,
            // Artificial pressure (Monaghan anti-clustering): standard strength. With
            // compression-only λ it's the only short-range repulsion keeping spacing.
            s_corr_k: 0.1,
            s_corr_n: 4.0,
            s_corr_dq_ratio: 0.2,
            spiky_r_min_ratio: 0.01,
            // Compression-only correction (λ ≤ 0): resist over-density but apply NO cohesive
            // pull on under-dense surface particles — that pull is PBF's implicit surface
            // tension and makes the water bead into droplets. Safe now that the convergence
            // order is fixed (the earlier collapse was that accumulation bug, not this).
            lambda_clamp_noncohesive: true,
            // Light viscosity — just enough to damp jitter; not the over-damped 0.2.
            xsph_viscosity_c: 0.05,
            // THE stability lever: cap any single position correction to 0.12·h. This bounds
            // the velocity that corrections inject, which kills the deficient-neighborhood /
            // squeeze-out eruptions ("the fluid jumped"). Tighter is calmer (0.25 still left
            // occasional global jumps); too tight under-resolves the deep pool.
            max_correction_ratio: 0.15,
            // No global damping — the convergence fix removed the energy accumulation, so the
            // water can stay lively instead of looking syrupy. (1.0 = off; available as a knob.)
            velocity_damping: 1.0,
            // Velocity cap as a pure safety backstop, above believable water speeds.
            max_speed: 50.0,
            bucket_capacity: 64,
            seed_jitter: 0.1,
            // Granular bed: contacts need more iterations + a tighter penetration tolerance than
            // the density solve, and a more frequent grid rebuild (contact-heavy).
            bed_max_iters: 24,
            bed_residual_tolerance: 0.02,
            bed_regrid_interval: 2,
            grain_sleep_speed: 0.2,
            // Coupling: drag_gamma is a scale resolved through Kozeny-Carman at solver build.
            packing_limit: 0.64,
            exclusion_relax: 0.5,
            drag_gamma: 0.02,
            drag_beta_max: 0.8,
            drag_subiters: 4,
            buoyancy_scale: 0.0,
            wake_threshold: 0.05,
        }
    }
}

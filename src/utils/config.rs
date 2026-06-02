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
    /// Velocity clamp `‖v‖ ≤ max_speed` (CFL + anti-blow-up).
    pub max_speed: f32,

    // --- neighbor grid ---
    /// Fixed per-cell bucket capacity `K` (overflow is a hard correctness failure).
    pub bucket_capacity: u32,

    // --- initial seeding ---
    /// Initial position jitter as a fraction of particle spacing (breaks lattice symmetry).
    pub seed_jitter: f32,
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
            relaxation_eps: 1.0e-4,
            // Under-relax the Jacobi position solve (ω<1) — stabilizes it and keeps
            // per-iteration moves within a grid cell (the grid is rebuilt once/frame).
            position_relaxation: 0.5,
            s_corr_k: 0.1,
            s_corr_n: 4.0,
            s_corr_dq_ratio: 0.2,
            spiky_r_min_ratio: 0.01,
            lambda_clamp_noncohesive: false,
            // XSPH must be strong enough to dissipate the energy s_corr injects (else the
            // surface jitters / never settles).
            xsph_viscosity_c: 0.1,
            max_speed: 50.0,
            bucket_capacity: 64,
            seed_jitter: 0.1,
        }
    }
}

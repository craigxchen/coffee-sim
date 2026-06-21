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
    /// Reduced-unit Darcy drag scale multiplying `150·(1−ε)²/(ε³·d²)`.
    pub drag_scale: f32,
    /// Per-subiter aggregate drag cap `β_max < 1` (anti-overshoot). (step 2)
    pub drag_beta_max: f32,
    /// Drag Jacobi sub-iterations per frame. 0 disables drag. (step 2)
    pub drag_subiters: u32,
    /// Buoyancy impulse scale on the PBF `λ` pressure proxy. 0 disables buoyancy. (step 3)
    pub buoyancy_scale: f32,
    /// A grain skips its static dead-band while the local water flow speed `|mean(v_water)−v_grain|`
    /// (scene units/s) exceeds this — a drag-only, dt/subiter-invariant wake gate (KTD-9). A still or
    /// hydrostatic saturated bed reads ≈0 and sleeps; the pour mobilizes the surface. (step 2)
    pub wake_threshold: f32,
    /// Dynamic-pressure (momentum-flux) impact scale: a center pour's downward water momentum pushes
    /// grains down/out (crater) via a force ∝ (normal approach velocity)². Momentum-conserving,
    /// approach-gated. `0` disables the pass (existing scenes byte-unchanged); a brew scene opts in.
    pub impact_scale: f32,
    /// Twofield U5 gate: the solid phase is a DYNAMIC elastoplastic granular material (Klar
    /// Drucker-Prager + compaction cap; `solvers/twofield/plasticity.wgsl`). `false` (default)
    /// keeps the U6 kinematically frozen skeleton — existing scenes byte-unchanged; the dry-bed
    /// L1 scenes opt in. (U7 releases the saturated bed and this becomes the production path.)
    pub solid_dynamics: bool,

    // --- wetting / cohesion (Phase 1.4) ---
    /// Absorption rate constant `k_abs` (1/s) in the bounded uptake `(1−e^{−k·dt})`. Higher = a
    /// grain saturates faster once wet (~0.5 ⇒ ~2 s time constant). (step 4)
    pub absorb_rate: f32,
    /// Remaining-volume floor: a water particle is deactivated only when `f_w ≤ this` (drives
    /// exact volume conservation — must be ≪ `pbf_eps`). (step 4)
    pub absorb_roundoff: f32,
    /// PBF skips a water particle's density contribution when `f_w ≤ this` (stability near the wet
    /// front). Larger than `absorb_roundoff`, so skipped water is still absorption-eligible. (step 4)
    pub pbf_eps: f32,
    /// Extraction/thermal opt-in gate (step 5). 0 = the extraction + thermal passes don't run, so
    /// existing scenes are byte-unchanged; a brew scene sets it > 0 to enable dissolution kinetics.
    pub extract_rate: f32,

    // --- fines migration (Phase 6) ---
    /// Fines erosion/deposition rate scale + opt-in gate (1/s). 0 = the fines transfer pass and the
    /// fines→permeability deviation don't run, so existing scenes are byte-unchanged; a brew scene
    /// sets it > 0 (alongside `Materials::fines_fraction > 0`) to enable migration. (step 6)
    pub fines_rate: f32,

    // --- pour emission (spout settings) ---
    /// Pour spout nozzle radius (scene units). With the discharge coefficient it sets the effective
    /// flux area `A_eff = π·r²·discharge_coeff`, which fixes the stream's exit speed from the flow
    /// rate and the emitted-layer particle count (so the inlet packs to the fluid's rest density).
    pub nozzle_radius: f32,
    /// Orifice discharge coefficient `∈ (0,1]` (vena-contracta loss). Folded into `A_eff`.
    pub discharge_coeff: f32,

    // --- twofield U9 infiltration interface (opt-in; all default 0 / OFF so existing scenes
    // and the U2–U6 twofield suites are byte-unchanged) ---
    /// Twofield moisture phase-change absorption gate + rate `k_abs` (1/s, reduced sim units).
    /// Zero skips the grid-projected absorption passes entirely (no φ_s/swelling change); a
    /// positive value enables GIC-style absorption into grain moisture via `models::wetting`.
    /// Calibrated: 60–80% swelling complete ≤ 30 s physical (≈ 8.14 s sim) gives k_abs ≈
    /// 0.11–0.20 (see `tests/twofield_infiltration.rs` UNIT MAPPING).
    pub tf_absorb_rate: f32,
    /// Twofield wetting-front capillary suction body force (Green–Ampt magnitude, reduced
    /// units): the extra downward acceleration on water at the unsaturated front, derived from
    /// the suction head ψ_f ≈ 50–110 mm (≈ 1.39–3.05 su) — see the test's UNIT MAPPING. 0 =
    /// no suction (the bare-Darcy arm).
    pub tf_suction_accel: f32,
    /// Twofield bloom delay (reduced sim seconds): a dry hydrophobic grain ramps from no
    /// absorption/suction to full over this contact time. 0 = no bloom gate (grains wet on
    /// contact). Calibrated seconds-scale: ≈ 2–6 s physical ⇒ ≈ 0.54–1.63 s sim.
    pub tf_bloom_delay: f32,
    /// Twofield phase-selective filter floor: water drains through the y-min face (porous-jump
    /// outflow), the frozen solid skeleton is retained. 0 = sealed floor (the U6 default).
    pub tf_filter_floor: bool,

    // --- twofield U7 effective-stress coupling (opt-in; default 0 / OFF so the U2–U6 suites and
    // the dry U5 bed are byte-unchanged: c_max = 0 ⇒ `cohesion::for_saturation` ≡ 0 ⇒ the DP
    // yield uses `cohesion::dry()` exactly) ---
    /// Twofield wet-cohesion peak `c_max` (stress units): the saturation-dependent capillary
    /// cohesion bump (`models::cohesion::for_saturation`) fed into the U5 Drucker-Prager yield
    /// apex as a function of local grain saturation s = V_abs/V_cap. 0 = no wet cohesion (dry
    /// `cohesion::dry()`), so the dry bed and the U5 gates are bitwise unchanged.
    pub tf_wet_cohesion: f32,
    /// Saturation at which the wet-cohesion bump peaks (`models::cohesion::for_saturation`
    /// `s_peak`): capillary bridges strengthen to `tf_wet_cohesion` here, then collapse toward
    /// full saturation. Coffee-plausible ≈ 0.4. Unused when `tf_wet_cohesion = 0`.
    pub tf_cohesion_speak: f32,

    // --- twofield U1/U2 surface-weighted velocity-averaging dissipation (g2p_water; default
    // DISABLED so the off-path is byte-identical to the pure-PIC baseline) ---
    /// Surface (near-air) smoothing strength `c` for the velocity-averaging dissipation:
    /// `v = mix(v_own, v_grid, c)`. The bulk uses `c → 1` (full average = calm pool); the free
    /// surface uses `c → tf_flip_c_surface` (small = momentum-preserving = splash). The sentinel
    /// `1.0` ⇒ `v = v_grid` everywhere = the pure-PIC baseline (DISABLED).
    pub tf_flip_c_surface: f32,
    /// Local-density gate (in units of a rest-packed cell ≈ `8·particle_mass`) above which the
    /// bulk dissipation term ramps in. Unused while the knob is disabled.
    pub tf_flip_density_gate: f32,
    /// Merge-discriminator scale (the compressive-`div` / relative-approach term). `≤ 0` ⇒ the
    /// entire merge discriminator is OFF (the density-only negative control); the sentinel `0.0`
    /// is the disabled default.
    pub tf_flip_div_scale: f32,
    /// Separate water/splash velocity cap (the U3 splash-path clamp). `≤ 0` ⇒ fall back to the
    /// global `max_speed` (byte-identical); the sentinel `0.0` is the disabled default.
    pub tf_flip_water_splash_cap: f32,

    // --- PB-MPM liquid density constraint (U4; only read by `SolverId::Pbmpm`) ---
    /// PB-MPM iteration count: the `particle_update → grid_zero → p2g → grid_update → g2p` bundle
    /// repeats this many times per substep so the compliant density correction propagates
    /// spatially through the rebuilt grid. ~2–4 is the stiff-but-cheap range; 1 disables the
    /// iterative tightening (one transfer cycle).
    pub pbmpm_iteration_count: u32,
    /// PB-MPM rest liquid density target (`1/liquid_density` in the `alpha` term). For D as the
    /// velocity-gradient state the rest target is `tr(D) = 1/density − 1`, so `1.0` rests at zero
    /// divergence (the natural single-phase setpoint).
    pub pbmpm_liquid_density: f32,
    /// PB-MPM compliant volume-correction relaxation `∈ (0,1]`: 1 is the stiffest single-iteration
    /// push toward rest, smaller is softer/more damped. The stiffness ⇒ bounce lever.
    pub pbmpm_liquid_relaxation: f32,
    /// PB-MPM viscous (deviatoric/shear) correction weight. Small (~0.01) damps shear without
    /// over-thickening the pool; `0` disables the shear term.
    pub pbmpm_liquid_viscosity: f32,
    /// PB-MPM collider normal-velocity restitution (U5; net-new knob twofield never had). On a
    /// particle penetrating a solid (cup floor/wall) the into-solid normal velocity reflects as
    /// `v_n_out = −restitution·v_n_in`. `0.0` = free-slip stop (the constraint-only / no-rebound
    /// arm, R8); larger reflects more (a bouncier floor). Water is barely elastic, so the default
    /// is LOW (~0.1) — not bouncy-rubber. Clamped to `[0, 1]`.
    pub pbmpm_restitution: f32,
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
            // Artificial pressure (Monaghan anti-clustering): the only short-range repulsion keeping
            // spacing under compression-only λ. The SHARP exponent (n=16, vs the textbook ~4) confines
            // it to genuine pairing range (r → 0): it's ~0 at rest spacing yet stronger than n=4 at
            // r→0. A softer exponent has a long tail that, on a thin airborne stream (free fall, where
            // s_corr is one-sided with nothing to cancel it), accumulates into lateral spread and
            // disperses the column — water in flight should stay ballistic/coherent. Sharpening keeps
            // dense fluid pair-stable (dam-break: no clumping) while leaving the free stream alone.
            s_corr_k: 0.1,
            s_corr_n: 16.0,
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
            seed_jitter: 0.1,
            // Granular bed: contacts need more iterations + a tighter penetration tolerance than
            // the density solve, and a more frequent grid rebuild (contact-heavy).
            bed_max_iters: 24,
            bed_residual_tolerance: 0.02,
            bed_regrid_interval: 2,
            grain_sleep_speed: 0.2,
            // Coupling: Darcy drag is computed per-particle from local porosity in WGSL.
            packing_limit: 0.64,
            drag_scale: 0.30,
            drag_beta_max: 0.8,
            drag_subiters: 4,
            buoyancy_scale: 1.0,
            wake_threshold: 0.3, // local water flow speed (units/s) above grain_sleep_speed (KTD-9)
            impact_scale: 0.0, // dynamic-pressure crater coupling OFF by default (opt-in per scene)
            solid_dynamics: false, // U6 frozen skeleton by default; the U5 bed scenes opt in
            // Wetting OFF by default (like drag_subiters=0): scenes opt in with absorb_rate>0 until
            // the feature is calibrated. ~2 s saturation time constant when enabled; deactivate
            // water only at a tiny remaining fraction (exact conservation), skip from PBF above that.
            absorb_rate: 0.0,
            absorb_roundoff: 1.0e-3,
            pbf_eps: 0.05,
            // Extraction/thermal OFF by default (step 5; like absorb_rate): a brew scene opts in.
            extract_rate: 0.0,
            // Fines OFF by default (step 6; like extract_rate): a brew scene opts in.
            fines_rate: 0.0,
            // Pour spout: a thin stream by default; a pour scene tunes these to its grind/flow.
            nozzle_radius: 0.5,
            discharge_coeff: 1.0,
            // Twofield U9 infiltration interface: OFF by default — the absorption/suction/bloom
            // passes are CPU-elided (absorb_rate gate) and the filter floor is sealed, so the
            // U2–U6 twofield suites and every existing scene are byte-unchanged.
            tf_absorb_rate: 0.0,
            tf_suction_accel: 0.0,
            tf_bloom_delay: 0.0,
            tf_filter_floor: false,
            // U7 effective-stress coupling: OFF by default — c_max = 0 makes the wet-cohesion
            // curve identically zero, so the DP yield falls back to cohesion::dry() and the dry
            // U5 bed + the U2–U6 suites are byte-unchanged. A saturated brew bed opts in.
            tf_wet_cohesion: 0.0,
            tf_cohesion_speak: 0.4,
            // U1/U2 surface-weighted dissipation: DISABLED by default so g2p_water stays pure-PIC
            // and every existing twofield gate is byte-unchanged. c_surface = 1.0 ⇒ v = v_grid
            // (pure-PIC); div_scale = 0 ⇒ merge discriminator off; water_splash_cap = 0 ⇒ global
            // max_speed. density_gate is unused while disabled.
            tf_flip_c_surface: 1.0,
            tf_flip_density_gate: 0.5,
            tf_flip_div_scale: 0.0,
            tf_flip_water_splash_cap: 0.0,
            // PB-MPM liquid density constraint (U4). Conservative stiff-but-stable starting point:
            // 2 iterations (cheap, enough to bounce), relaxation 0.5 (a half-step compliant push —
            // does not detonate at the default cap on a fast pour), density 1.0 (rest at zero
            // divergence), and a small 0.01 viscous shear damp. Tuned live in the webapp (U4/U6).
            pbmpm_iteration_count: 2,
            pbmpm_liquid_density: 1.0,
            pbmpm_liquid_relaxation: 0.5,
            pbmpm_liquid_viscosity: 0.01,
            // Water is barely elastic: a LOW restitution so the floor bounce is a thin rebound, not
            // bouncy-rubber. 0 = free-slip stop (the constraint-only arm). Tuned live in U5/U6.
            pbmpm_restitution: 0.1,
        }
    }
}

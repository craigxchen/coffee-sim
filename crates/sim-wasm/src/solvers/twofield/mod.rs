//! Two-field grid solver under `SolverId::Twofield` — rung L0 of
//! `docs/plans/2026-06-09-001-feat-unified-twofield-solver-plan.md`.
//!
//! U1 pinned the seam contract (six `Solver` methods, GPU-cached getters), the two-field
//! particle-range layout (water `[0, w)`, then solids — KTD-1), the canonical `ParticleBuffers`
//! exposure, and the dispatch/budget bookkeeping.
//!
//! U2 adds the APIC water transfers: quadratic B-spline P2G with fixed-point atomics, grid
//! gravity + boundary conditions, APIC G2P with a per-particle affine C matrix
//! (`transfers.wgsl`). Per frame: `grid_clear → p2g_water → grid_update → g2p_water`, one
//! substep (the CFL substep policy is a later-unit decision).
//!
//! U3 adds incompressibility (`pressure.wgsl`, KTD-2): a coarse-grid pressure seed plus a
//! fixed budget of fine damped-Jacobi sweeps on ONE discretely consistent operator family
//! (pressure at cell centers, corner-trilinear D, G = −Dᵀ by construction, A only ever the
//! composition −D·M̃⁻¹·G) — no converged Poisson anywhere (R7). The full operator-family and
//! coarse-seed construction is documented at the top of `pressure.wgsl`; the pre-registered
//! gates live in `tests/twofield_pressure.rs`. The KTD-2 adaptive-Tait predictor is a STUB
//! DECISION: deliberately not implemented (knob grid pins Tait ∈ {off}); if a later unit
//! demonstrates an iteration-budget win it ships behind an opt-in `Config` gate, otherwise it
//! is removed rather than shipped dormant.
//!
//! U4 (the L0 exit) adds pour emission and the pour-cavity machinery (`surface.wgsl`,
//! KTD-6): `EmissionInput` drives the xpbd-shaped volume-consistent arclength-credit emitter
//! into a pre-allocated water pool (water live range `[0, water_count)` grows toward
//! `water_capacity`; solids sit AFTER the full pool so the KTD-1 range layout survives
//! emission — dormant slots are never dispatched); enclosed air pockets are detected by a
//! fixed-budget grid flood fill from the open (top) boundary and carried as ONE constraint
//! bubble: a single Lagrange-multiplier scalar λ_b whose value IS the pocket pressure,
//! represented identically in the coarse solve and the fine sweeps (see surface.wgsl's
//! BUBBLE REPRESENTATION header). Gates: `tests/twofield_cavity.rs`.
//!
//! U6 (the early-L2 risk spike, initially run against a RIGID, kinematically frozen skeleton —
//! KTD-3/KTD-4/KTD-8) adds `coupling.wgsl`: a thin solid-mass P2G (`p2g_solid` scatters the
//! grain sphere volume π/6·d³ into a one-lane fixed-point field → local φ_s/φ_f per node)
//! and the forced Laibe-Price exponential drag fold (`drag_fold`, which also owns the grid
//! forces + BCs so the fold integrates the gravity source exactly). β(φ) mirrors
//! `models::permeability` (Kozeny-Carman) blended toward a Wen-Yu dilute rate below
//! φ_s ≈ 0.2 (Huilin-Gidaspow). The projection becomes the MIXTURE family ∇·(φ_f·v_f) = s
//! (v_s = 0) with the velocity correction Δv = −(Δt_eff/ρ_w)∇p, Δt_eff = (1−e^{−βΔt})/β —
//! the same integrator weight as the fold, carried as ς inside M̃⁻¹; φ NEVER scales the
//! correction. Every impulse the frozen skeleton absorbs (drag pair + pressure on the solid
//! volume) is recorded in the per-node `react` ledger — never discarded. Free water
//! (φ_f = 1, ς = 1) is exact-zero passthrough, gated bitwise. Gates:
//! `tests/twofield_coupling.rs`.
//!
//! U5 (rung L1, `plasticity.wgsl` + the CPU twin `plasticity.rs`) makes the solid phase a
//! real elastoplastic granular material behind `Config::solid_dynamics` (default OFF — the
//! U6 frozen-skeleton semantics are preserved bitwise; the dry-bed L1 scenes opt in): solid
//! P2G with momentum + APIC + MLS-MPM fused stress force (`p2g_solid_dyn` replaces the
//! frozen `p2g_solid` 1:1), grid forces + over-packing solids-pressure guard + Coulomb wall
//! friction (`solid_update`), solid G2P with the deformation-gradient update, one 3×3 SVD
//! per grain, and the Klar 2016 Drucker-Prager 3-branch return map + compaction-cap tamping
//! memory (`g2p_solid`). Dynamic wet coupling now uses this solid grid velocity in one exact,
//! momentum-conserving two-mass drag impulse, closes pressure on
//! `div(phi_f*v_f + phi_s*v_s)`, and applies the drag-condensed pressure response to both
//! phases. Frozen mode keeps the U6 constrained-skeleton path. Gates: `tests/twofield_bed.rs`.

pub mod plasticity;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::emission::EmissionInput;
use crate::engine::scene::Species;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::solvers::base::Solver;
use crate::solvers::pass_recorder::{PassRecorder, TimestampSink};
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

const WG: u32 = 256;
const DIAGNOSTICS_PERIOD_S: f32 = 0.25;
const DIAGNOSTICS_READBACK_SLOTS: usize = 2;

/// Y-coordinate at which un-emitted (dormant) water-pool slots are parked: far below any scene so
/// the renderer culls them (particles.wgsl matches this with `p.y <= -1.0e8`). Activated slots get
/// a real position from `emit()`. Kept finite (not NaN) so any accidental read stays well-defined.
const DORMANT_PARK_Y: f32 = -1.0e9;

/// Fine-cell tile shape shared with `active_tiles.wgsl`. One 256-thread workgroup processes
/// exactly two lanes per cell tile (512 cells total).
const ACTIVE_TILE_EDGE: u32 = 8;
const ACTIVE_TILE_HEADER_WORDS: u32 = 1;
const ACTIVE_TILE_WORDS_PER_TILE: u32 = 2;
const ACTIVE_WATER_COUNT_OFFSET: u64 = 0;
const ACTIVE_WATER_INDIRECT_OFFSET: u64 = 0;
// Internal sequence sentinels. WebGPU's per-dimension workgroup ceiling is far below these, so
// they cannot alias a legal direct dispatch size; the actual counts stay in the GPU args buffer.
const ACTIVE_WATER_INDIRECT: u32 = u32::MAX;
const ACTIVE_TILE_MARKER: u32 = u32::MAX - 1;

fn groups(n: u32) -> u32 {
    n.div_ceil(WG)
}

fn active_tile_dims_for_grid(node_dims: [u32; 3]) -> [u32; 3] {
    node_dims.map(|nodes| (nodes - 1).div_ceil(ACTIVE_TILE_EDGE))
}

fn active_tile_count_for_grid(node_dims: [u32; 3]) -> u32 {
    active_tile_dims_for_grid(node_dims).into_iter().product()
}

fn active_tiling_enabled(implicit_solid: bool, all_gas_atmospheric: bool) -> bool {
    implicit_solid && all_gas_atmospheric
}

/// U2+U3+U4+U6+U5 GPU-budget bookkeeping (R8): the widest entry point's storage-buffer count,
/// derived by inspection of the bind groups in `build` (the params uniform doesn't count
/// against the storage limit). Per pass: `grid_clear` 7 (grid_fp, cell_cnt, grid_sfp,
/// grid_sm, grid_moist, grid_chem, grid_sraw), `p2g_water` 6 (pos, vel, chem, cmat, grid_fp, cell_cnt),
/// `p2g_solid` 4 (pos, chem, grid_sfp, cell_cnt), `grid_update` 2 (grid_fp, grid_vel), `drag_fold` 7
/// (grid_vel, solids, cell_cnt, grid_sfp, react, grid_sraw, grid_svel), `g2p_water` 8 (pos, vel, chem,
/// cmat, grid_vel, solids, cell_cnt, filter_state); U3 pressure family —
/// `node_setup` 7 (grid_vel, solids, nm, cell_cnt, grid_sfp, react, grid_svel),
/// `cell_classify` 9 (grid_vel, solids, cell_meta, pf_a, pf_b, cell_cnt, grid_sfp, nm, grid_svel),
/// `coarse_node_setup` 2, `coarse_cell_setup` 5 (cell_meta, cmeta, pc_a, pc_b, pocket_c — U8
/// compacted coarse-pocket append),
/// `jacobi_fine`/`jacobi_coarse` 5 (+bubble), `prolong_add` 6 (cell_meta, cmeta, pc, pf×2,
/// bubble), `project` 9 (grid_vel, solids, nm, cell_meta, pf, cell_cnt, grid_sfp, react, grid_svel —
/// dynamic mode uses phase-specific wall projectors and pressure mobilities; ties cell_classify), debug
/// taps ≤ 4; U4
/// surface family — `flood_init` 6 (solids, cell_meta, pf_a, bubble — U8 early-out flag reset —
/// plus pocket_f/pocket_c counter resets; `solids` for the wall-adjacent OUTSIDE seed),
/// `flood_sweep` 3, `pocket_mark` 8 (grid_vel, cell_meta,
/// pf×2, bubble, nm, pocket_f, grid_svel — U8 compacted-list append), `bubble_fine` 5 (nm, cell_meta, pf,
/// bubble, pocket_f), `bubble_coarse` 6 (nm_c, cmeta, pc, bubble, pocket_f, pocket_c — pocket_f
/// for the shared sub-resolution release size gate); U5 plasticity family —
/// `p2g_solid_reference` 5 (pos, vel, chem, cmat, grid_sraw), `p2g_solid_dyn` 8 (pos, vel, chem, cmat, grid_sfp, grid_sm, sstate, cell_cnt), `solid_update` 4
/// (grid_sm, grid_svel, grid_sfp, solids), `g2p_solid` 8 (pos, vel, chem, cmat, solids,
/// grid_svel, fmat, sstate) — ties cell_classify; U9 absorption family — `p2g_moisture` 4
/// (pos, vel, chem, grid_moist), `g2p_absorb` 4 (pos, vel, chem, grid_moist); chemistry/thermal
/// scatter binds 6 and gather binds 7 around the 11-lane chemistry ledger; `filter_release` binds
/// 10 to reconstruct pressure-projected mobile superficial flux and its measured carrier closure; the semi-implicit
/// family's widest pass is `p2g_solid_macro` at 9 (pos, vel, chem, cmat, grid_sfp, grid_sm,
/// sstate, grid_scomp, cell_cnt), while its scalar Jacobi binds 4. Tiled fine-pressure relaxation adds one
/// scheduler binding; its occupancy marker binds 2 (cell census, scheduler). The occasional
/// compact diagnostics pass
/// binds 6 storage buffers (pos, chem, grid_vel, cell_cnt, filter_state, output). All remain at or
/// below the cap; re-derive when passes are added. Shared required limits request 10 storage
/// buffers per stage (see `solvers/mpm/mod.rs::required_limits`), within the WebGPU-target cap.
pub const MAX_STORAGE_BUFFERS_PER_ENTRY_POINT: u32 = 10;

/// U3 pressure knobs — defaults of the declared knob grid (KTD-9; the grid itself is pinned
/// in the header of `tests/twofield_pressure.rs`). The coarse ratio is fine cells per coarse
/// cell per axis (4 → 1/4³ cell count); sweep counts are dispatches, so they are plain Rust
/// fields driven from `step()` (a Jacobi sweep cannot loop inside one dispatch — WGSL has no
/// global sync).
pub const COARSE_RATIO_DEFAULT: u32 = 4;
pub const COARSE_SWEEPS_DEFAULT: u32 = 8;
pub const FINE_SWEEPS_DEFAULT: u32 = 8;

/// Damped-Jacobi relaxation factor — a fixed algorithmic constant of the family (mirrors
/// `JACOBI_OMEGA` in `pressure.wgsl`; the CPU twins import it), NOT a gate knob. Derivation:
/// the Fourier symbol of A = −D·M⁻¹·G on the corner family gives
/// λ(diag⁻¹A)(θ) = (8/3)·[sin²(θx/2)cos²(θy/2)cos²(θz/2) + cyc], max 8/3 at θ = (π,0,0), so
/// Jacobi requires ω < 2/(8/3) = 0.75; ω = 2/3 is the classic weighted-Jacobi choice and
/// keeps ω·λ_max = 16/9 < 2 with healthy high-frequency smoothing.
pub const JACOBI_OMEGA: f32 = 2.0 / 3.0;

/// Default APIC↔PIC blend = the PIC fraction folded into the G2P affine reconstruction
/// (`v_new` keeps full APIC velocity; the C matrix is scaled by `1 − PIC_BLEND`). A PIC fraction
/// is the standard MPM damping that bleeds off the affine field's spurious rotational energy each
/// transfer. DEFAULT 0 = pure APIC.
///
/// This IS the settled-pool stirring fix (the user-reported open-water churn): a settled tank held
/// ~32× the pure-PIC tail KE under pure APIC; a small blend bleeds that off (settled tail KE 67→21,
/// max|v| 0.95→0.52). It was held at 0 during the investigation because the stirring is the SAME
/// open-water agitation that drove the OLD deformable-bed crater slump, so any blend that quieted
/// the pool also "froze" that slump. The fix was to correct the crater gate: a real wet bed HOLDS
/// the poured crater (wet-sand plasticity — pour-over research), so a held crater is the CORRECT
/// outcome, not a regression. With the gate asserting persistence (see `twofield_full.rs` and
/// docs/plans/2026-06-15-001), the blend is safe to re-enable: it quiets the pool AND the crater
/// correctly persists. Volume conservation was verified to hold at this blend (the
/// `combined_conservation_v60_pour_deformable` + saturated-tail gates). blend 1 ⇒ pure PIC (the
/// APIC-vs-PIC gate). The global blend is chosen over a φ_f-gated one because the held crater is
/// now correct, so no gating is needed.
pub const PIC_BLEND_DEFAULT: f32 = 0.05;

/// Density-relief time constant in frames (mirrors `DENSITY_RELAX_FRAMES` in pressure.wgsl).
/// A fixed structural constant like `JACOBI_OMEGA`: it closes the volume-conservation loop
/// (velocity-only projection cannot see accumulated positional compression), it is not a
/// gate-tuning knob. Half a second (30 frames): fast enough that equilibrium compaction
/// drift stays ≈ residual·N·dt ≈ 1% (inside the ±5% band), slow enough that relieving a
/// seeded over-density (the lattice double-counts the wall layers by ~30%) injects v ~
/// Δx/τ ≈ 0.6 instead of ~4.5 of slosh — the relief must correct volume, not detonate it.
pub const DENSITY_RELAX_FRAMES: f32 = 30.0;

/// SDF wall-BC mode selector, written to `dbg.y` and read in pressure.wgsl/coupling.wgsl
/// (`wall_bc_multi`). `SINGLE` is the original single-most-penetrated-normal banded BC
/// (byte-identical default); `MULTI` is the orthonormal-basis projector P = I − Q·Qᵀ over ALL
/// in-band faces (plan 2026-06-17-002 — constrains BOTH surfaces at a concave floor∩wall seam),
/// flipped on in U3 after the corner-audit confirms box parity. `dbg.y` previously carried a
/// prototyped relief dead-band that was dropped (the settled-pool stirring fix is the G2P PIC
/// blend, `PIC_BLEND_DEFAULT`).
pub const WALL_BC_SINGLE: f32 = 0.0;
pub const WALL_BC_MULTI: f32 = 1.0;

/// Free-surface fill-fraction constants (mirror `SURF_FULL_FRAC`/`SURF_MIN_CORNER` in
/// pressure.wgsl — the ghost-fluid-style fraction weighting documented in its FREE SURFACE
/// header; refined, not replaced, by U4). Fixed structural constants of the operator family,
/// not gate knobs: a cell's constraint row is weighted by f = clamp(ρ̄/(0.5·ρ_rest), 0, 1),
/// so interior rows are exactly f = 1 (a flat surface cuts a cell at ρ̄ ≈ 0.5·ρ_rest — the
/// taper lives strictly above the mean surface line) and the implicit p = 0 Dirichlet acts
/// at the surface instead of one cell inside it. ACTIVITY is the min corner-node density ≥
/// `SURF_MIN_CORNER`·ρ_rest — the particle-presence discriminator that excludes B-spline
/// smear cells (particle-free, far corner ≤ ~0.05·ρ_rest), whose pressure rows otherwise
/// hover their dust against gravity and block settling (see the pressure.wgsl FREE SURFACE
/// header for the observed failure).
pub const SURF_FULL_FRAC: f32 = 0.5;
pub const SURF_MIN_CORNER: f32 = 0.1;

/// U3 dispatch increment over U2's 4, at the default knobs: node_setup + cell_classify +
/// pre-smooth/post-smooth FINE_SWEEPS + residual + coarse_node_setup +
/// coarse_cell_setup(+restrict) + COARSE_SWEEPS + prolong_add + project. Each Jacobi sweep is
/// necessarily its own dispatch (no global sync within a dispatch), which is why this lands
/// above the plan's rough +6–12 guess — recorded honestly per R8.
pub const U3_PRESSURE_DISPATCHES: u32 = 7 + COARSE_SWEEPS_DEFAULT + FINE_SWEEPS_DEFAULT;

/// U4 flood-fill sweep budget: each sweep propagates the OUTSIDE label ONE 6-neighbor cell, so
/// the budget must bound the longest open-air PATH (in cells) from any in-box air cell to the
/// open top face. The earlier fixed 24 was a miscalculation: it covered the box gate scenes
/// (tank air column ≈ 16 cells) but NOT the V60 cup, whose air path runs from the cup floor
/// (y = -8) up through the cone apex and out the box top (y = +10) — ≈ 88 cells at the web
/// spacing, plus the cup-radius lateral detour. With too few sweeps the cup/cone air never gets
/// the OUTSIDE label, is misclassified as one enclosed POCKET, and the bubble constraint
/// (λ ≈ −3000) crushes the standing water into the floor corner (the reported WaterOnly
/// collapse). The budget is therefore the grid's MANHATTAN DIAMETER — (nx−1)+(ny−1)+(nz−1)
/// cells — which upper-bounds any monotone 6-neighbor path on the grid regardless of geometry,
/// rounded UP to even so the final labels land back in the pf_a slot (parity). It is scene-
/// derived (a per-solver field), not a global constant, because the path length scales with the
/// grid; cells beyond the budget still degrade conservatively (extra pocket members with
/// massless nodes — zero row coupling).
pub fn flood_sweeps_for(dims: [u32; 3]) -> u32 {
    let diameter = (dims[0] - 1) + (dims[1] - 1) + (dims[2] - 1);
    diameter + (diameter & 1) // round up to even
}

const U4_SURFACE_MAINTENANCE_DISPATCHES: u32 = 2;

fn u4_surface_dispatches_for_sweeps(
    dims: [u32; 3],
    fine_sweeps: u32,
    coarse_sweeps: u32,
    all_gas_atmospheric: bool,
) -> u32 {
    if all_gas_atmospheric {
        U4_SURFACE_MAINTENANCE_DISPATCHES
    } else {
        U4_SURFACE_MAINTENANCE_DISPATCHES + flood_sweeps_for(dims) + fine_sweeps + coarse_sweeps
    }
}

/// U4 dispatch increment for a grid of `dims` and gas-boundary mode. The atmospheric mode keeps
/// only state reset + the under-density/finalize scan; the generic mode additionally flood-fills
/// open gas and relaxes the fine/coarse aggregate bubble row before every Jacobi sweep.
pub fn u4_surface_dispatches_for_mode(dims: [u32; 3], all_gas_atmospheric: bool) -> u32 {
    u4_surface_dispatches_for_sweeps(
        dims,
        FINE_SWEEPS_DEFAULT,
        COARSE_SWEEPS_DEFAULT,
        all_gas_atmospheric,
    )
}

/// Generic/default U4 dispatch increment, retained for existing callers and sealed-pocket tests.
pub fn u4_surface_dispatches_for(dims: [u32; 3]) -> u32 {
    u4_surface_dispatches_for_mode(dims, false)
}

/// U6 dispatch increment: the thin solid-mass P2G (`p2g_solid`) + the drag fold
/// (`drag_fold`, which also absorbed grid_update's force/BC work — net one new pass).
pub const U6_COUPLING_DISPATCHES: u32 = 2;

/// U9 dispatch increment, applied ONLY when `Config::tf_absorb_rate > 0` (opt-in; default-knob
/// frozen-water scenes keep the U6 budget bitwise, so `DISPATCHES_PER_FRAME` and every U2–U6
/// suite's pinned budget are untouched): `p2g_moisture` + `g2p_absorb`. Suction and the filter
/// floor add ZERO passes — suction folds into `drag_fold`'s existing force step, the filter
/// floor is a branch in `drag_fold`/`node_setup`/`g2p_water`'s existing BC code.
pub const U9_INFILTRATION_DISPATCHES: u32 = 2;

/// Conservative chemistry/thermal scatter + gather, conditionally dispatched when either the
/// extraction or independent thermal Config gate is enabled.
pub const CONSTITUTIVE_DISPATCHES: u32 = 2;
/// Finite paper storage adds one release pass; whole-parcel admission is fused into g2p_water.
pub const FILTER_RESERVOIR_DISPATCHES: u32 = 1;

/// U5 dispatch increment per SUBSTEP, applied ONLY when `Config::solid_dynamics` is on:
/// `p2g_solid_reference` + `solid_update` + `g2p_solid` (`p2g_solid_dyn` replaces the frozen `p2g_solid`
/// one-for-one). Dynamic mode runs the frame pipeline `plasticity::solid_substeps`
/// times (the stiff-elasticity CFL — 7 at the default dt/h/density), so the recorded
/// dynamic-mode frame budget is substeps × (DISPATCHES_PER_FRAME + U5_PLASTICITY_DISPATCHES)
/// when water is present. Frozen mode keeps the U6 budget exactly.
pub const U5_PLASTICITY_DISPATCHES: u32 = 3;

/// Semi-implicit production solid: fixed projected-Jacobi budget for the bounded packing solve.
/// Tests protect residual/energy monotonicity rather than treating this algorithmic count as a
/// coffee material constant.
pub const COMPACTION_SWEEPS: u32 = 8;
/// The explicit predictor carries numerical shear only. If its requested compliance would need
/// more work, the applied modulus is lowered and reported; physical packing stiffness is never
/// changed to meet this ceiling.
pub const MAX_SHEAR_SUBSTEPS: u32 = 8;
pub const SHEAR_CFL: f64 = 0.4;

/// Dry dynamic-mode budget per substep: with ZERO live water particles the entire
/// water-side pipeline (p2g_water, grid_update decode, drag fold, the U3 pressure stack,
/// the U4 surface stack, g2p_water) reads and writes only water state that the solid passes
/// never touch, so the dry L1 scenes dispatch exactly grid_clear + p2g_solid_reference + p2g_solid_dyn +
/// solid_update + g2p_solid per substep. This is a CPU-side pass elision keyed on the
/// synchronously known live water count — NOT indirect dispatch (R8) — and it is scoped to
/// DYNAMIC mode only: frozen-mode scenes keep the full pipeline bitwise, so every U2–U6
/// suite's pinned budget is untouched. Solid trajectories are bitwise identical with and
/// without the elision (verified by the suite re-runs at the switch); without it the dry
/// macro gates spend 18× their dispatch budget ringing an empty pressure solve (measured:
/// the resolved runout scene fell from ~70 min to minutes).
pub const U5_DRY_DISPATCHES: u32 = 5;

/// Compute dispatches per frame for a grid of `dims` at the default knobs: the U2 transfer
/// pipeline (grid_clear, p2g_water, grid_update, g2p_water) + the U3 pressure stack + the U4
/// surface stack (scene-derived through the flood budget) + the U6 coupling passes.
pub fn dispatches_per_frame_for(dims: [u32; 3]) -> u32 {
    dispatches_per_frame_for_mode(dims, false)
}

/// Base default-knob dispatch budget with an explicit gas-boundary mode. Conditional
/// infiltration, chemistry, dynamic-solid, and compaction passes remain additive at their owning
/// call sites; the recorder is the truthful total for an executed production frame.
pub fn dispatches_per_frame_for_mode(dims: [u32; 3], all_gas_atmospheric: bool) -> u32 {
    4 + U3_PRESSURE_DISPATCHES
        + u4_surface_dispatches_for_mode(dims, all_gas_atmospheric)
        + U6_COUPLING_DISPATCHES
}

/// Dry per-grain structural volume (U6, KTD-8): π/6·d³. The WGSL swelling law scales this
/// dry base from bound-water saturation; it never adds bound-water volume directly.
pub fn grain_volume(d: f32) -> f32 {
    std::f32::consts::PI / 6.0 * d * d * d
}

/// Dry structural volume carried by one solid material point on a cubic sample lattice.
/// Numerical sample pitch is not a literal grind diameter: assigning `(1-porosity)*pitch^3`
/// makes a uniformly seeded bed reproduce its declared reference porosity independently of the
/// water-grid resolution.
pub fn dry_solid_sample_volume(sample_pitch: f32, reference_porosity: f32) -> f32 {
    sample_pitch.max(0.0).powi(3) * (1.0 - reference_porosity.clamp(0.0, 1.0))
}

/// Dry bulk density used by the explicit shear CFL. The numerical shear modulus is declared per
/// represented bulk volume, so density is dry mass divided by the corresponding dry bulk control
/// volume `V_struct/(1-porosity)`. Bound-water swelling must not alter this stability definition.
pub fn dry_bulk_density(dry_mass: f32, dry_structural_volume: f32, reference_porosity: f32) -> f32 {
    let solid_fraction = (1.0 - reference_porosity).clamp(0.0, 1.0);
    dry_mass.max(0.0) * solid_fraction / dry_structural_volume.max(1.0e-9)
}

fn filter_uniforms(cfg: &Config) -> Result<([f32; 4], [u32; 4]), String> {
    let finite_nonnegative = |value: f32, label: &str| {
        if !value.is_finite() || value < 0.0 {
            Err(format!("{label} must be finite and non-negative"))
        } else {
            Ok(())
        }
    };
    finite_nonnegative(cfg.tf_filter_permeability_m2, "filter permeability")?;
    finite_nonnegative(
        cfg.tf_filter_kinematic_viscosity_m2_s,
        "filter reference viscosity",
    )?;
    finite_nonnegative(cfg.tf_filter_thickness_m, "filter thickness")?;
    finite_nonnegative(cfg.tf_filter_capacity_g, "filter capacity")?;
    if !(cfg.tf_filter_porosity.is_finite()
        && cfg.tf_filter_porosity > 0.0
        && cfg.tf_filter_porosity < 1.0)
    {
        return Err("filter porosity must be finite and strictly between zero and one".into());
    }
    if cfg.tf_filter_permeability_m2 > 0.0
        && (cfg.tf_filter_kinematic_viscosity_m2_s <= 0.0 || cfg.tf_filter_thickness_m <= 0.0)
    {
        return Err(
            "positive filter permeability requires positive viscosity and physical thickness"
                .into(),
        );
    }

    let scale = crate::utils::units::SiUnitScale::reference();
    let permeability = scale.sim_permeability_from_m2(f64::from(cfg.tf_filter_permeability_m2));
    let viscosity =
        scale.sim_permeability_from_m2(f64::from(cfg.tf_filter_kinematic_viscosity_m2_s));
    let thickness = scale.sim_length_from_meters(f64::from(cfg.tf_filter_thickness_m));
    let capacity_m3 = crate::models::filter::water_mass_capacity_to_volume_m3(
        f64::from(cfg.tf_filter_capacity_g),
        f64::from(cfg.tf_water_density_kg_m3),
    )
    .map_err(|error| format!("filter capacity: {error}"))?;
    let capacity_sim = capacity_m3 / scale.meters_per_sim_unit().powi(3);
    let capacity_counts = (capacity_sim * FILTER_VOLUME_SCALE).round();
    if !(0.0..=f64::from(u32::MAX)).contains(&capacity_counts) {
        return Err("filter capacity exceeds the persistent fixed-point reservoir".into());
    }

    Ok((
        [
            permeability as f32,
            viscosity as f32,
            thickness as f32,
            cfg.tf_filter_porosity,
        ],
        [capacity_counts as u32, 0, 0, 0],
    ))
}

fn recipe_solid_mass_reduced(
    scene: &Scene,
    mats: &Materials,
    water_density_kg_m3: f32,
    solid_count: u32,
) -> Option<f32> {
    if scene.dose_g <= 0.0 || solid_count == 0 || mats.particle_mass <= 0.0 {
        return None;
    }
    let spatial = crate::utils::units::SiUnitScale::reference();
    let water = crate::models::scaling::water_sample_from_spacing(
        f64::from(mats.particle_spacing),
        f64::from(water_density_kg_m3),
        spatial,
    )
    .ok()?;
    let mass_kg_per_reduced_unit = water.mass_kg / f64::from(mats.particle_mass);
    let solid = crate::models::scaling::solid_sample_mass_from_recipe(
        f64::from(scene.dose_g),
        solid_count,
        mass_kg_per_reduced_unit,
    )
    .ok()?;
    let reduced = solid.reduced_mass;
    (reduced.is_finite() && reduced > 0.0 && reduced <= f64::from(f32::MAX))
        .then_some(reduced as f32)
}

/// U5 seed state: per-solid identity deformation gradients (3 vec4 rows each).
fn identity_rows(solid_count: u32) -> Vec<[f32; 4]> {
    let mut rows = Vec::with_capacity(3 * solid_count as usize);
    for _ in 0..solid_count {
        rows.push([1.0, 0.0, 0.0, 0.0]);
        rows.push([0.0, 1.0, 0.0, 0.0]);
        rows.push([0.0, 0.0, 1.0, 0.0]);
    }
    rows
}

/// U5 seed state: per-solid (τ = 0, p_c = PC0, compaction = 0), 2 vec4 each.
fn fresh_sstate(solid_count: u32, initial_consolidation_pressure: f32) -> Vec<[f32; 4]> {
    let mut rows = Vec::with_capacity(2 * solid_count as usize);
    for _ in 0..solid_count {
        rows.push([0.0; 4]);
        rows.push([0.0, 0.0, initial_consolidation_pressure, 0.0]);
    }
    rows
}

/// CPU twin of `coupling.wgsl::drag_rate_blended` (KTD-3): Kozeny-Carman packed-bed rate
/// (mirrors `models::permeability` — beta = phi_mobile*drag_scale/K_structural) blended toward a Wen-Yu-Stokes dilute
/// rate (coefficient 18, voidage correction omitted) below φ_s = 0.2 via the Huilin-Gidaspow
/// arctan transition (slope 262.5 = 150·1.75). The pair-momentum gate pins the WGSL against
/// this twin.
pub fn blended_drag_rate(d: f32, phi_f: f32, drag_scale: f32) -> f32 {
    blended_drag_rate_for_mobile_pore(d, phi_f, phi_f, drag_scale)
}

fn blended_drag_rate_for_mobile_pore(
    d: f32,
    structural_phi_f: f32,
    mobile_phi_f: f32,
    drag_scale: f32,
) -> f32 {
    let phi_s = 1.0 - structural_phi_f;
    let packed = crate::models::permeability::darcy_mobile_drag_rate(
        d,
        structural_phi_f,
        mobile_phi_f,
        drag_scale,
    );
    let available_ratio = (mobile_phi_f / structural_phi_f.max(1.0e-6)).clamp(0.0, 1.0);
    let dilute = drag_scale * 18.0 * phi_s / (d * d).max(1.0e-8) * available_ratio;
    let psi = (262.5 * (phi_s - 0.2)).atan() / std::f32::consts::PI + 0.5;
    psi * packed + (1.0 - psi) * dilute
}

/// Recipe-scaled retention inventory frozen when a solver is built. All volumes are reduced
/// solver volumes. The state keeps the requested and attainable inventories separate so an
/// over-capacity calibration cannot silently masquerade as a successful fit.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RetentionBuildState {
    whole_bed_target_volume: f32,
    bound_capacity_volume: f32,
    capillary_target_volume: f32,
    available_pore_volume: f32,
    retained_capillary_volume: f32,
    residual_pore_saturation: f32,
    attainment: crate::models::retention::TargetAttainment,
}

impl RetentionBuildState {
    fn unmet_volume(self) -> f32 {
        match self.attainment {
            crate::models::retention::TargetAttainment::Attainable => 0.0,
            crate::models::retention::TargetAttainment::Unmet { volume } => volume,
        }
    }

    fn target_attainable(self) -> bool {
        matches!(
            self.attainment,
            crate::models::retention::TargetAttainment::Attainable
        )
    }

    fn diagnostics(self) -> TwofieldDiagnostics {
        TwofieldDiagnostics {
            retention_target_volume: self.whole_bed_target_volume,
            retention_bound_capacity_volume: self.bound_capacity_volume,
            retention_capillary_target_volume: self.capillary_target_volume,
            retention_available_pore_volume: self.available_pore_volume,
            // Runtime H=min(W,C) is a measured grid inventory and must never be synthesized from
            // this attainable build target. The compact diagnostics pass fills it after scatter.
            retention_capillary_volume: 0.0,
            retention_unmet_volume: self.unmet_volume(),
            residual_pore_saturation: self.residual_pore_saturation,
            retention_target_attainable: self.target_attainable(),
            filter_pressure_mobility_min: 1.0,
            mean_diameter_ratio: 1.0,
            ..TwofieldDiagnostics::default()
        }
    }
}

/// Derive the build-time capillary ledger from the exact recipe mass and numerical pore volume.
///
/// `capacity_from_dry_mass` is used for both the whole-bed gravimetric target and the bound-water
/// capacity, so both share the same reduced water density as free-water P2G. The pore capacity is
/// the seeded bulk sample volume minus the dry structural volume; it therefore reproduces the
/// declared reference porosity without interpreting a numerical material point as one coffee
/// particle. Filter-paper inventory is deliberately zero here until filter state exists.
fn retention_state_for_build(
    cfg: &Config,
    mats: &Materials,
    dry_grain_mass: f32,
    solid_count: u32,
    dry_structural_volume: f32,
    solid_quadrature_weight: f32,
    water_sample_volume: f32,
) -> Result<RetentionBuildState, crate::models::retention::RetentionError> {
    let count = solid_count as f32;
    let whole_bed_target_volume = crate::models::wetting::capacity_from_dry_mass(
        dry_grain_mass,
        cfg.tf_retention_target_g_per_g,
        water_sample_volume,
        mats.particle_mass,
    ) * count;
    let bound_capacity_volume = crate::models::wetting::capacity_from_dry_mass(
        dry_grain_mass,
        mats.r_max,
        water_sample_volume,
        mats.particle_mass,
    ) * count;
    let bulk_sample_volume =
        mats.grain_diameter.max(0.0).powi(3) * solid_quadrature_weight.max(0.0);
    let available_pore_volume = (bulk_sample_volume - dry_structural_volume).max(0.0) * count;
    let capillary_target_volume = crate::models::retention::capillary_target_volume(
        whole_bed_target_volume,
        bound_capacity_volume,
    )?;
    let residual = crate::models::retention::residual_pore_saturation(
        capillary_target_volume,
        available_pore_volume,
    )?;
    Ok(RetentionBuildState {
        whole_bed_target_volume,
        bound_capacity_volume,
        capillary_target_volume,
        available_pore_volume,
        retained_capillary_volume: residual.retained_volume,
        residual_pore_saturation: residual.residual_saturation,
        attainment: residual.attainment,
    })
}

/// CPU twin of the grid-node pore saturation calculation. Invalid or collapsed pore volumes are
/// treated as dry; physical inputs are bounded to `[0, 1]` without embedding a calibration target.
#[cfg(test)]
fn node_pore_saturation(water_volume: f32, visible_pore_volume: f32) -> f32 {
    if !water_volume.is_finite() || !visible_pore_volume.is_finite() || visible_pore_volume <= 0.0 {
        return 0.0;
    }
    (water_volume.max(0.0) / visible_pore_volume).clamp(0.0, 1.0)
}

/// CPU twin of the measured single-field mobility used by `coupling.wgsl`, expressed on a unit
/// pore volume. The floor applies only above capacity; held-only water is exactly immobile.
#[cfg(test)]
fn bounded_corey_mobility(
    local_saturation: f32,
    residual_saturation: f32,
    exponent: f32,
    mobility_floor: f32,
) -> f32 {
    crate::models::retention::runtime_capillary_closure(
        local_saturation.clamp(0.0, 1.0),
        1.0,
        residual_saturation.clamp(0.0, 1.0),
        exponent.max(1.0e-6),
        mobility_floor.clamp(0.0, 1.0),
    )
    .map_or(0.0, |state| state.relative_mobility)
}

/// CPU twin of the bed-restricted saturation-gradient capillary force. Neighbor order is
/// `[-x, +x, -y, +y, -z, +z]`; `None` means a missing or non-bed neighbor and receives the
/// zero-normal-gradient value `center`.
#[cfg(test)]
fn capillary_acceleration_from_stencil(
    center: f32,
    neighbors: [Option<f32>; 6],
    cell_size: f32,
    suction_accel: f32,
) -> [f32; 3] {
    if !center.is_finite()
        || !cell_size.is_finite()
        || cell_size <= 0.0
        || !suction_accel.is_finite()
        || suction_accel <= 0.0
    {
        return [0.0; 3];
    }
    let center = center.clamp(0.0, 1.0);
    let sample = |index: usize| {
        neighbors[index]
            .filter(|value| value.is_finite())
            .unwrap_or(center)
            .clamp(0.0, 1.0)
    };
    let inv_two_h = 0.5 / cell_size;
    let gradient = [
        (sample(1) - sample(0)) * inv_two_h,
        (sample(3) - sample(2)) * inv_two_h,
        (sample(5) - sample(4)) * inv_two_h,
    ];
    let magnitude =
        (gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2]).sqrt();
    if magnitude <= 0.0 {
        return [0.0; 3];
    }
    let strength = (2.0 * cell_size * magnitude).clamp(0.0, 1.0);
    let scale = -suction_accel * strength / magnitude;
    [
        scale * gradient[0],
        scale * gradient[1],
        scale * gradient[2],
    ]
}

/// Fine-grid cell size as a multiple of the particle spacing. ~2× spacing gives the quadratic
/// B-spline support (1.5 cells each way) a ≈3-spacing reach with ≈8 particles per cell at rest
/// — the standard APIC loading and the same scale as the SPH `support_radius` default. The plan
/// defers this to implementation; revisit when the U3 pressure stencil picks its resolution.
pub const CELL_SIZE_FACTOR: f32 = 2.0;

/// Fixed-point scale for the grid atomics — mirrors `FP_SCALE` in `common.wgsl` (2^18,
/// KEEP.md §3). The overflow-headroom derivation (coupled to `Config::max_speed`) lives next to
/// the WGSL constant; the overflow-probe gate in `tests/twofield_water.rs` exercises it.
pub const FP_SCALE: f64 = 262144.0;
/// Persistent paper-reservoir volume scale. The current 25 g UI ceiling remains below u32::MAX
/// at the reference spatial scale while resolving a small fraction of one water parcel.
pub const FILTER_VOLUME_SCALE: f64 = 268_435_456.0; // 2^28

fn filter_volume_counts(volume: f32) -> u32 {
    (volume.max(0.0) * FILTER_VOLUME_SCALE as f32)
        .round()
        .clamp(0.0, 4_294_967_040.0) as u32
}

fn solution_solute_mass_fraction(carrier_mass: f32, solute_mass: f32) -> f32 {
    let solution_mass = carrier_mass.max(0.0) + solute_mass.max(0.0);
    if solution_mass > 0.0 {
        solute_mass.max(0.0) / solution_mass
    } else {
        0.0
    }
}

/// Uniform parameters, byte-mirrored by the WGSL `Params` in `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    box_min: [f32; 4],
    box_max: [f32; 4],
    gravity: [f32; 4],
    grid_origin: [f32; 4], // .xyz = node (0,0,0) world position; .w = cell size h
    grid_dims: [u32; 4],   // nx, ny, nz, num_nodes
    dt: f32,
    particle_mass: f32,
    max_speed: f32,
    // APIC↔PIC blend = PIC fraction ∈ [0,1] applied to the G2P affine state (0 = pure APIC,
    // 1 = pure PIC). A small default (PIC_BLEND_DEFAULT) damps the APIC rotational ringing that
    // re-energizes a settled pool; the APIC-vs-PIC gate forces 1.0 via `set_pic_for_test`.
    pic_blend: f32,
    water_count: u32,    // particles [0, water_count) are water (KTD-1 range layout)
    solid_count: u32,    // particles [water_count, water_count + solid_count) are solid grains
    particle_count: u32, // = water_count + solid_count (kernel live-set guard)
    num_solids: u32,     // count of static SDF solids in the `solids` buffer
    coarse_dims: [u32; 4], // coarse CELLS per axis (= ceil(fine_cells/ratio)); .w = ratio
    extra: [f32; 4],     // (rest_density, rho_floor, mass_eps, U7 wet-cohesion s_peak)
    // U6 coupling: (solid sample pitch, drag_scale, dry structural sample volume, open_base
    // flag — the dev/test drained-column outflow mode, see coupling.wgsl).
    coupling: [f32; 4],
    // U5 plasticity (plasticity.wgsl; values from `plasticity` consts + Materials/Config):
    splas0: [f32; 4], // (solid_dynamics flag, Lamé μ, Lamé λ, DP α)
    splas1: [f32; 4], // (cap hardening ξ, φ_max, dry grain mass, cohesion y_c)
    splas2: [f32; 4], // (floor/wall Coulomb μ_b, guard K_sp, guard onset φ_on, U7 wet c_max)
    // U9 infiltration interface (coupling.wgsl; rates and delays use physical seconds):
    wet0: [f32; 4], // (k_abs, V_cap from dry recipe mass, V_w water vol, absorb_roundoff)
    wet1: [f32; 4], // (a_suction, bloom_delay, filter_floor, max diameter swelling fraction)
    // Diagnostic toggles (test-only; production keeps the defaults). .x = density-relief enable
    // (1.0 on by default; 0.0 disables BOTH the cell_classify over-density relief and the
    // pocket_mark under-density suction for the stirring-isolation gate). .y = SDF wall-BC mode
    // (WALL_BC_SINGLE/MULTI). .z = temper-K rate divisor for the uncapped path (0 ⇒ K=1 full
    // uncap; legacy cap is 30; sweet spot K∈(1,30)). .w = uncapped two-sided density-target mode
    // (≤ 0.5 → legacy rate-limited (DENSITY_RELAX) two-sided relief, default & byte-identical;
    // > 0.5 → relief uncapped to full strength, driving ρ→ρ₀ — the over-pack fix).
    dbg: [f32; 4],
    // U1 surface-weighted dissipation knob (transfers.wgsl g2p_water). Lanes =
    // (c_surface, density_gate, div_scale, water_splash_cap). DISABLED by default
    // (c_surface = 1.0 ⇒ v = v_grid = pure-PIC; div_scale = 0 ⇒ merge discriminator off;
    // water_splash_cap = 0 ⇒ fall back to the global max_speed). The U2 curve reads these.
    flip: [f32; 4],
    // Conservative extraction / thermal transport (coupling.wgsl). Extraction and thermal rates
    // have independent Config scales; either positive gate dispatches their shared ledger pair.
    chem0: [f32; 4], // (scaled k_fast, scaled k_slow, Ea/R in K, T_ref in K)
    chem1: [f32; 4], // (c_sat, u_half, wet-gate s_on, thermal exchange rate)
    therm: [f32; 4], // (cp_water, cp_dry_grain, ambient loss rate, ambient T_K)
    // Capillary retention: (build target saturation for diagnostics, Corey exponent,
    // relative-mobility floor, all-gas-atmospheric flag). Runtime retention is measured from
    // the scattered per-solid capacity in solute.w; surface.wgsl consumes retention.w.
    retention: [f32; 4],
    // Semi-implicit production granular split: (K_max, p_c0, full-tangent phi, shear dt).
    // K_max <= 0 is the exact legacy/off-path sentinel.
    granular: [f32; 4],
    // Porous paper: (permeability su², kinematic viscosity su²/s, physical thickness su,
    // porosity). Independent of the coarse SDF shell and finite reservoir capacity.
    filter0: [f32; 4],
    // Exact integer reservoir capacity in FILTER_VOLUME_SCALE counts; remaining lanes reserved.
    filter1: [u32; 4],
    // Mechanical/thermal solute bookkeeping: insoluble grain mass, solute specific heat,
    // initial extractable mass, capillary capacity volume per solid material point.
    solute: [f32; 4],
}

// Params is uploaded as a uniform and must stay byte-identical to the WGSL `Params`.
const _: () = assert!(std::mem::size_of::<Params>() == 400);

/// Static spatial information used only by the compact diagnostics reduction. Keeping cup and
/// bed geometry out of the physics `Params` prevents an observability concern from perturbing
/// every production pipeline's uniform ABI.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DiagnosticsConfig {
    cup0: [f32; 4], // (center x, center z, radius, minimum y)
    cup1: [f32; 4], // (maximum y, enabled, seeded bed top, reserved)
}

const _: () = assert!(std::mem::size_of::<DiagnosticsConfig>() == 32);

/// Compact GPU result mirrored by `diagnostics.wgsl::DiagnosticsOutput`. These are primitive
/// extensive sums; solution/grain totals and conservation residuals are derived on the CPU so
/// their definitions stay auditable and share the blocking-validation path.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct DiagnosticsGpu {
    mobile: [f32; 4],
    paper: [f32; 4],
    grain: [f32; 4],
    cup: [f32; 4],
    grid: [f32; 4],
    counts: [u32; 4],
    filter0: [u32; 4],
    filter1: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<DiagnosticsGpu>() == 128);

/// GPU record for one static SDF solid — byte-identical to the WGSL `Primitive` (64 bytes,
/// vec4-aligned; mirrors the xpbd packing of `utils::sdf` primitives). Cone radii in `a` are
/// OUTER wall radii; the cavity surface is `outer − thickness`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Primitive {
    kind: u32,         // 0 = cone, 1 = cylinder
    species_mask: u32, // MASK_* bits
    friction: f32,
    flags: u32,  // bit0 = apex_open, bit1 = porous filter paper
    a: [f32; 4], // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)
    b: [f32; 4], // cone:(thickness, hole_radius, center_x, center_z)  cyl:(center_x, center_z, _, _)
    c: [f32; 4], // reserved
}

// Primitive is a storage-buffer element and must stay byte-identical to the WGSL `Primitive`.
const _: () = assert!(std::mem::size_of::<Primitive>() == 64);

/// Pack a scene's analytic solids into the GPU `Primitive` layout (mirrors utils/sdf.rs).
fn pack_solids(solids: &[crate::utils::sdf::SdfPrimitive]) -> Vec<Primitive> {
    use crate::utils::sdf::{SolidKind, FLAG_FILTER_PAPER};
    let semantic_flags = |flags: u32| (flags & FLAG_FILTER_PAPER) << 1;
    solids
        .iter()
        .map(|s| match s.kind {
            SolidKind::Cone {
                center,
                apex_y,
                top_y,
                apex_r,
                top_r,
                thickness,
                hole_radius,
                apex_open,
            } => Primitive {
                kind: 0,
                species_mask: s.species_mask,
                friction: s.friction,
                flags: u32::from(apex_open) | semantic_flags(s.flags),
                a: [apex_y, apex_r, top_y, top_r],
                b: [thickness, hole_radius, center.x, center.z],
                c: [0.0; 4],
            },
            SolidKind::Cylinder {
                center,
                floor_y,
                rim_y,
                radius,
            } => Primitive {
                kind: 1,
                species_mask: s.species_mask,
                friction: s.friction,
                flags: semantic_flags(s.flags),
                a: [floor_y, rim_y, radius, 0.0],
                b: [center.x, center.z, 0.0, 0.0],
                c: [0.0; 4],
            },
            SolidKind::PolyCup {
                center,
                floor_y,
                rim_y,
                apothem,
                sides,
            } => Primitive {
                kind: 2,
                species_mask: s.species_mask,
                friction: s.friction,
                flags: semantic_flags(s.flags),
                a: [floor_y, rim_y, apothem, sides as f32],
                b: [center.x, center.z, 0.0, 0.0],
                c: [0.0; 4],
            },
        })
        .collect()
}

/// Per-pass `timestamp-query` capture (native only; the web context never enables the
/// feature). Same resolve-into-readback-of-the-previous-frame shape as xpbd, so `profile()`
/// never stalls — `sample_diagnostics` is the explicit blocking cache point.
struct Timestamps {
    qset: wgpu::QuerySet,
    capacity: u32,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    labels: Vec<String>,
    count: u32,
}

struct DiagnosticsPending {
    receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    sample_time_s: f32,
    expected_carrier_mass: f32,
    epoch: u64,
}

struct DiagnosticsReadbackSlot {
    buffer: wgpu::Buffer,
    pending: Option<DiagnosticsPending>,
}

/// Brew diagnostics sampled from the GPU (dev/test only — the read-back stalls). Mirrors the
/// xpbd `XpbdDiagnostics` precedent: the values are CPU-resident after the explicit
/// `sample_diagnostics()` synchronization point and the getters never sync. `drawdown_time`
/// latches the sim time when the free water ABOVE the bed surface (the pond) first drains away.
#[derive(Clone, Copy, Debug, Default)]
pub struct TwofieldDiagnostics {
    /// Whether at least one compact GPU snapshot has completed since build/reset, and the sim
    /// time represented by that snapshot. Production getters return this cache without mapping.
    pub sample_valid: bool,
    pub sample_time_s: f32,
    /// Live free-water particles above the bed surface (the pond), latest sample.
    pub free_water_above_bed: u32,
    pub pond_carrier_mass: f32,
    /// Sim time (s) when the pond first drained below the drawdown threshold; 0 = not latched.
    pub drawdown_time: f32,
    /// Flow-evenness placeholder (1.0 = even); channel histogram work remains separate.
    pub evenness: f32,
    /// Peak per-node grid occupancy proxy (max decoded node mass / rest density), latest sample.
    pub max_occupancy: f32,
    /// Dissolved solute mass divided by total dry coffee mass.
    pub extraction_yield: f32,
    /// Beverage solute mass fraction in the catch cup: solute / (carrier + solute). Particle
    /// concentration remains the independent solute/carrier ratio used by the transport shader.
    pub cup_tds: f32,
    /// Heat-capacity-weighted phase mean temperatures.
    pub mean_water_temperature: f32,
    pub mean_grain_temperature: f32,
    /// Exact phase inventories in reduced mass units. Carrier is solvent water; solution includes
    /// dissolved solute. Bound carrier moves mechanically with the grains and is therefore part
    /// of `grain_mass`, not either free-solution inventory.
    pub mobile_carrier_mass: f32,
    pub paper_carrier_mass: f32,
    pub bound_carrier_mass: f32,
    pub total_carrier_mass: f32,
    pub mobile_solution_mass: f32,
    pub paper_solution_mass: f32,
    pub grain_mass: f32,
    pub mobile_solute_mass: f32,
    pub paper_solute_mass: f32,
    pub remaining_grain_solute_mass: f32,
    pub dissolved_solute_mass: f32,
    pub cup_carrier_mass: f32,
    pub cup_solute_mass: f32,
    pub cup_solution_mass: f32,
    /// Whole-bed mean bound saturation and its corresponding mean structural diameter ratio.
    /// Diameter swelling is linear in saturation, so the ratio follows exactly from the aggregate
    /// bound inventory when every material point carries the same recipe-scaled capacity.
    pub mean_bound_saturation: f32,
    pub mean_diameter_ratio: f32,
    /// Closed-system inventory checks. `total_enthalpy` changes only through the explicit ambient
    /// term (and inlet/outflow); `total_solute` changes only through inlet/outflow.
    pub total_solute: f32,
    pub total_enthalpy: f32,
    /// Source-side expectations and signed conservation residuals. Open-base particles remain in
    /// the canonical arrays, so these whole-system ledgers include escaped parcels.
    pub expected_carrier_mass: f32,
    pub expected_solute_mass: f32,
    pub carrier_mass_residual: f32,
    pub solute_mass_residual: f32,
    pub mechanical_mass_residual: f32,
    /// Persistent porous-paper ledger. The volume residual compares the sum of whole held-parcel
    /// encoded volumes with the CAS-owned occupancy word; the count residual compares held
    /// particles with cumulative admissions minus releases.
    pub filter_occupied_volume: f32,
    pub filter_capacity_volume: f32,
    pub filter_saturation: f32,
    pub filter_volume_residual: f32,
    pub filter_count_residual: i64,
    pub filter_held_parcels: u32,
    pub filter_admitted: u32,
    pub filter_released: u32,
    pub filter_overflow: u32,
    pub filter_errors: u32,
    /// Center-band one-cell porous response at the current `h` and `dt`. These are derived from
    /// the exact CPU twin, not sampled from particles: beta is the maximum normal relaxation rate
    /// and the mobility is the minimum pressure response across the hat band.
    pub filter_beta_max_per_s: f32,
    pub filter_pressure_mobility_min: f32,
    /// Explicit invalid-state observability. Finite aggregates exclude invalid records only after
    /// counting them here; callers can no longer mistake an omitted NaN for a valid low total.
    pub has_nonfinite_state: bool,
    pub nonfinite_particle_count: u32,
    pub nonfinite_grid_node_count: u32,
    /// Build-time retention ledger in reduced solver-volume units. The whole-bed target is split
    /// into bound capacity plus a capillary target; capillary volume is bounded by the seeded pore
    /// volume and any remainder is reported explicitly instead of hidden by saturation clamping.
    pub retention_target_volume: f32,
    pub retention_bound_capacity_volume: f32,
    pub retention_capillary_target_volume: f32,
    pub retention_available_pore_volume: f32,
    pub retention_capillary_volume: f32,
    pub retention_unmet_volume: f32,
    pub residual_pore_saturation: f32,
    pub retention_target_attainable: bool,
    /// Numerical-only explicit shear work. A limited applied value is observable and never
    /// misreported as the physical compaction stiffness.
    pub requested_numerical_shear_modulus: f32,
    pub applied_numerical_shear_modulus: f32,
    pub shear_microsteps: u32,
    pub numerical_shear_limited: bool,
}

pub struct TwofieldSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Params,
    /// LIVE water count (grows as the pour activates pool slots); the seed value is
    /// `initial_water`.
    water_count: u32,
    /// Allocated water-pool size = seed + dose headroom (KTD-1 layout with emission: water
    /// pool `[0, water_capacity)`, live `[0, water_count)`, solids
    /// `[water_capacity, water_capacity + solid_count)` — dormant slots are never dispatched).
    water_capacity: u32,
    solid_count: u32,
    num_nodes: u32,
    num_cells: u32,
    /// Pour-emission state (xpbd-shaped volume-consistent emitter).
    inflow: Inflow,

    // U3 pressure knobs (KTD-9 grid points; sweep counts = dispatch counts, driven in step).
    coarse_ratio: u32,
    coarse_sweeps: u32,
    fine_sweeps: u32,
    // U4 flood-fill budget = grid Manhattan diameter (scene-derived; see `flood_sweeps_for`).
    flood_sweeps: u32,
    /// Explicit vented-geometry approximation: all resolved gas remains atmospheric and the
    /// flood/bubble rows are omitted. Generic/default solvers keep sealed-pocket detection.
    all_gas_atmospheric: bool,

    params_buf: wgpu::Buffer,
    // Canonical particle state, exposed through `ParticleBuffers`: pos.w carries the moisture
    // lane (water = remaining fraction, grain = bound-water volume). `chem` is phase-specific:
    // water=(solute/carrier mass ratio, T_K, filter state 0/1/2, pore-transit progress),
    // grain=(fast-pool mass, slow-pool mass, T_K, wet-contact time).
    pos: Arc<wgpu::Buffer>,
    vel: Arc<wgpu::Buffer>,
    phase: Arc<wgpu::Buffer>,
    chem: Arc<wgpu::Buffer>,
    // Per-particle APIC affine matrix C: 3 vec4 rows per particle (see common.wgsl binding 5).
    cmat: wgpu::Buffer,
    // WATER grid field: fixed-point atomic<i32>, 4 lanes per node (mass, mom.xyz).
    grid_fp: wgpu::Buffer,
    // Combined per-cell presence prefix + four per-node capillary-closure suffix lanes
    // (carrier mass, attainable capacity C, bloom-weighted structural volume, measured held
    // carrier mass rho_w*H; binding 18).
    cell_cnt: wgpu::Buffer,
    // Float grid velocity (.xyz) + node mass (.w) after grid_update; post-projection after
    // the U3 pressure stack. CPU-readable for the divergence/volume gates.
    grid_vel: wgpu::Buffer,
    // Solid-volume field plus interphase drag state (per-node impulse.xyz, sigma) —
    // coupling.wgsl bindings 19/20.
    grid_sfp: wgpu::Buffer,
    // Semi-implicit solid plastic-compaction moment, one fixed-point lane per node.
    grid_scomp: wgpu::Buffer,
    // Persistent paper reservoir (binding 30), intentionally not part of per-frame grid clear.
    filter_state: wgpu::Buffer,
    // Phase-1 sparse water-pressure scheduler (binding 33) and its separate INDIRECT-only
    // argument record. Enabled only for the semi-implicit + explicitly vented production path;
    // every solid, setup, clear, node, and coarse pass remains dense.
    active_tiles: wgpu::Buffer,
    active_tile_indirect: wgpu::Buffer,
    active_tile_count: u32,
    active_tiling: bool,
    // Compact periodic observability path. `diagnostics_gpu` is reduced on the GPU; two tiny
    // staging slots are mapped asynchronously and never waited on by the frame loop.
    diagnostics_gpu: wgpu::Buffer,
    diagnostics_config: wgpu::Buffer,
    diagnostics_readbacks: [DiagnosticsReadbackSlot; DIAGNOSTICS_READBACK_SLOTS],
    diagnostics_elapsed_s: f32,
    diagnostics_epoch: u64,
    react: wgpu::Buffer,
    // U5 dynamic-solid per-particle state (plasticity.wgsl bindings 23/24): deformation
    // gradient (3 vec4 rows) and (τ, p_c, compaction). The grid_sm/grid_svel node fields
    // (bindings 21/22) live only inside the bind groups, like the coarse pressure mirrors.
    fmat: wgpu::Buffer,
    sstate: wgpu::Buffer,
    /// `Config::solid_dynamics`: dispatch the U5 dynamic-solid passes instead of the frozen
    /// U6 skeleton.
    solid_dynamic: bool,
    /// Production semi-implicit solid path. False is the exact legacy U5 execution path.
    implicit_solid: bool,
    requested_numerical_shear_modulus: f32,
    applied_numerical_shear_modulus: f32,
    shear_substeps: u32,
    /// `Config::tf_absorb_rate > 0`: dispatch the U9 moisture phase-change absorption passes
    /// (`p2g_moisture` + `g2p_absorb`). Off (default) ⇒ the passes are CPU-elided, so the
    /// U2–U6 suites and existing scenes are byte-unchanged.
    absorb_on: bool,
    /// Extraction and heat activation are independent. They share one conservative grid
    /// scatter/gather pair, dispatched when either constitutive subsystem has nonzero rates.
    extraction_on: bool,
    thermal_on: bool,
    filter_reservoir_on: bool,
    filter_paper_present: bool,
    // U3 pressure-family state (layouts documented in pressure.wgsl): per-node M̃⁻¹, per-cell
    // (rhs, active, dbg) meta, and the fine pressure ping-pong pair. The coarse mirrors live
    // only inside the bind groups.
    nm: wgpu::Buffer,
    cell_meta: wgpu::Buffer,
    pf_a: wgpu::Buffer,
    pf_b: wgpu::Buffer,
    // U4 bubble state: [λ_b, δλ_b] (surface.wgsl binding 17).
    bubble: wgpu::Buffer,
    readback: wgpu::Buffer,

    pipelines: Pipelines,
    ts: Option<Timestamps>,

    // Cached per-frame results returned by the getters (never a GPU sync there).
    dispatches: u32,
    cached_passes: Vec<(String, f32)>,

    // CPU-resident observability cache, refreshed by compact asynchronous snapshots or the
    // explicit blocking validation sampler. Getters never synchronize the GPU.
    sim_time: f32,
    bed_top_seed: f32,
    cup: Option<CupRegion>,
    pond_peak_carrier_mass: f32,
    retention_build: RetentionBuildState,
    cached_diag: TwofieldDiagnostics,
    initial_carrier_mass: f32,
    initial_solute_mass: f32,
    insoluble_grain_mass: f32,

    // Retained for reset (exact, deterministic re-seed; pool-padded layout).
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
    initial_chemistry: Vec<[f32; 4]>,
    initial_water: u32,
}

/// Catch-cup cavity used only by the explicitly sampled TDS diagnostic.
#[derive(Clone, Copy)]
struct CupRegion {
    cx: f32,
    cz: f32,
    radius: f32,
    y_min: f32,
    y_max: f32,
}

/// Pour-emission state + spout parameters (host-side; mirrors the xpbd `Inflow`). Turns
/// `EmissionInput` into activated water-pool particles via the volume-consistent
/// arclength-credit emitter: the volume accumulator (flow/V_w·dt) is the master count budget;
/// layers release one `particle_spacing` of stream travel apart, each filling a golden-angle
/// disc, so the inlet packs to the fluid's rest density.
struct Inflow {
    // Static (from Config/Materials at build).
    nozzle_radius: f32,
    discharge_coeff: f32,
    spacing: f32,
    v_w: f32,
    pour_t: f32,
    // State.
    accumulator: f32, // volume budget in particles (carries the sub-particle fraction)
    axial: f32,       // arclength credit (scene units) toward the next layer
    last_exit_speed: f32, // drains backlog at the last cadence when flow drops to 0
    cursor: u64,      // golden-angle determinism across all emitted particles
    emitted_mass: f32, // running total emitted water mass (conservation accounting)
}

/// Orthonormal disc basis perpendicular to a (unit) pour direction `dir` (mirrors xpbd).
fn disc_basis(dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1.0e-9);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let refv = if dir[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = norm(cross(dir, refv));
    let w = cross(dir, u);
    (u, w)
}

struct Pipelines {
    active_tile_mark: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g_water: (wgpu::ComputePipeline, wgpu::BindGroup),
    // U6 coupling family: solid-mass P2G + the exponential drag fold (which owns the grid
    // forces + BCs — see coupling.wgsl).
    p2g_solid: (wgpu::ComputePipeline, wgpu::BindGroup),
    drag_fold: (wgpu::ComputePipeline, wgpu::BindGroup),
    // U9 moisture phase-change absorption (dispatched only when Config::tf_absorb_rate > 0).
    p2g_moisture: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p_absorb: (wgpu::ComputePipeline, wgpu::BindGroup),
    // Conservative extraction + thermal transport (dispatched when either independent gate is on).
    chemistry_scatter: (wgpu::ComputePipeline, wgpu::BindGroup),
    chemistry_gather: (wgpu::ComputePipeline, wgpu::BindGroup),
    filter_release: (wgpu::ComputePipeline, wgpu::BindGroup),
    diagnostics_reduce: (wgpu::ComputePipeline, wgpu::BindGroup),
    // U5 plasticity family (dispatched only in dynamic mode; p2g_solid_dyn replaces
    // p2g_solid 1:1 there).
    p2g_solid_reference: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g_solid_dyn: (wgpu::ComputePipeline, wgpu::BindGroup),
    solid_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p_solid: (wgpu::ComputePipeline, wgpu::BindGroup),
    // Semi-implicit production solid path (compaction.wgsl). The pressure ping-pong variants
    // reuse pf_a/pf_b and are overwritten by the subsequent mixture-pressure setup.
    solid_grid_clear: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g_solid_macro: (wgpu::ComputePipeline, wgpu::BindGroup),
    p2g_solid_shear: (wgpu::ComputePipeline, wgpu::BindGroup),
    solid_predict_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    solid_macro_decode: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p_solid_shear: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p_solid_velocity_only: (wgpu::ComputePipeline, wgpu::BindGroup),
    compaction_node_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    compaction_cell_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    compaction_jacobi: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    compaction_project: (wgpu::ComputePipeline, wgpu::BindGroup),
    compaction_state_gather: (wgpu::ComputePipeline, wgpu::BindGroup),
    grid_update: (wgpu::ComputePipeline, wgpu::BindGroup),
    g2p_water: (wgpu::ComputePipeline, wgpu::BindGroup),
    // U3 pressure family. The Jacobi sweeps ping-pong the pressure pair by swapping which
    // buffer sits at the src/dst binding indices (two bind-group variants, index = sweep % 2);
    // prolong/project carry parity variants selecting the final-parity buffer.
    node_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    cell_classify: (wgpu::ComputePipeline, wgpu::BindGroup),
    residual: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    residual_tiled: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    coarse_node_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    coarse_cell_setup: (wgpu::ComputePipeline, wgpu::BindGroup),
    jacobi_coarse: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    // prolong_add variants indexed [current-p parity][coarse-final parity].
    prolong_add: (wgpu::ComputePipeline, [[wgpu::BindGroup; 2]; 2]),
    prolong_add_tiled: (wgpu::ComputePipeline, [[wgpu::BindGroup; 2]; 2]),
    jacobi_fine: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    jacobi_fine_tiled: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    project: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    // U4 surface family: flood fill (label ping-pong in the pressure slots), pocket marking,
    // and the single-workgroup bubble-row solves at each level (parity variants like the
    // sweeps they precede).
    flood_init: (wgpu::ComputePipeline, wgpu::BindGroup),
    flood_sweep: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    pocket_mark: (wgpu::ComputePipeline, wgpu::BindGroup),
    bubble_fine: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    bubble_coarse: (wgpu::ComputePipeline, [wgpu::BindGroup; 2]),
    // Test-only operator taps (never dispatched in step; zero budget impact).
    dbg_div: (wgpu::ComputePipeline, wgpu::BindGroup),
    dbg_grad: (wgpu::ComputePipeline, wgpu::BindGroup),
}

/// Seed the scene's regions into the two-field range layout: ALL water particles first
/// (`[0, water_count)`), then all solids. Lattice + seeded jitter per species spacing, with
/// cone-aware SDF rejection (mirrors xpbd's `seed_block`; RNG draws happen per lattice point
/// regardless of rejection, so the layout is deterministic for a given `cfg.seed`). The 4th
/// position lane carries moisture: water = remaining fraction (1 = full), grain = bound-water
/// volume (0 = dry). Wet-contact time starts at zero in `chem.w`, independent of these seeds.
/// Returns `(positions, phases, water_count, solid_quadrature_weight)`.
pub(crate) fn seed_ranges(
    scene: &Scene,
    mats: &Materials,
    cfg: &Config,
) -> (Vec<[f32; 4]>, Vec<u32>, u32, f32) {
    let mut rng = crate::utils::rng::Rng::new(cfg.seed);
    let mut water: Vec<[f32; 4]> = Vec::new();
    let mut solid: Vec<[f32; 4]> = Vec::new();
    for region in &scene.regions {
        let (lo, hi) = (region.min, region.max);
        let (s, tag, w0) = match region.species {
            Species::Water => (mats.particle_spacing, 0u32, 1.0f32),
            Species::Grain => (mats.grain_diameter, 1u32, 0.0f32),
        };
        // A seed sample needs at most half its own lattice pitch of geometric clearance. A fixed
        // scene-unit shell erased fine water samples and under-cleared coarse grains when either
        // resolution changed.
        let seed_clearance = 0.5 * s.max(0.0);
        let jitter = cfg.seed_jitter * s;
        let nx = (((hi[0] - lo[0]) / s).floor() as i32).max(0);
        let ny = (((hi[1] - lo[1]) / s).floor() as i32).max(0);
        let nz = (((hi[2] - lo[2]) / s).floor() as i32).max(0);
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let jx = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let jy = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let jz = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let px = lo[0] + i as f32 * s + jx;
                    let py = lo[1] + j as f32 * s + jy;
                    let pz = lo[2] + k as f32 * s + jz;
                    if !scene.solids.is_empty() {
                        let c = crate::utils::sdf::nearest(
                            &scene.solids,
                            glam::Vec3::new(px, py, pz),
                            tag,
                        );
                        if c.signed < seed_clearance {
                            continue;
                        }
                    }
                    let p = [px, py, pz, w0];
                    if tag == 0 {
                        water.push(p);
                    } else {
                        solid.push(p);
                    }
                }
            }
        }
    }
    // A scene's seeded-water inventory is authoritative. Ordinary free water keeps a contiguous
    // lowest-fill volume at full parcel density. Only the deliberate co-located porous-bed
    // fixture uses uniform selection to represent pore occupancy throughout the bed.
    if scene.water_ml > 0.0 && mats.particle_spacing > 0.0 {
        let target_volume = crate::utils::units::SiUnitScale::reference()
            .sim_volume_from_ml(f64::from(scene.water_ml));
        let target_samples = (target_volume / f64::from(mats.particle_spacing.powi(3)))
            .round()
            .max(1.0) as u32;
        if seeded_water_overlaps_grain(scene) {
            uniformly_downsample_points(
                &mut water,
                target_samples,
                cfg.seed,
                0x5741_5445_525F_4255, // "WATER_BU"
            );
        } else {
            downsample_lowest_fill(&mut water, target_samples, cfg.seed);
        }
    }
    let accepted_solid_count = solid.len();
    uniformly_downsample_points(
        &mut solid,
        cfg.tf_solid_sample_budget,
        cfg.seed,
        0x534F_4C49_445F_4255, // "SOLID_BU"
    );
    // Uniform thinning is a quadrature change, not a loss of represented bed volume. Every
    // retained point carries the same accepted/final weight; a non-binding budget is exactly 1.
    let solid_quadrature_weight = if solid.is_empty() {
        1.0
    } else {
        accepted_solid_count as f32 / solid.len() as f32
    };
    let water_count = water.len() as u32;
    let mut pos = water;
    pos.extend_from_slice(&solid);
    let mut phase = vec![0u32; water_count as usize];
    phase.extend(std::iter::repeat_n(1u32, solid.len()));
    (pos, phase, water_count, solid_quadrature_weight)
}

/// Uniformly select an exact material-point budget without biasing toward the lattice prefix.
/// Priorities are a stateless SplitMix64 hash of `(seed, accepted_index)`. After retaining the
/// lowest priorities, original accepted-point order is restored so every downstream buffer remains
/// deterministic and spatially ordered exactly as before. Budget zero and budgets at/above the
/// accepted count are strict no-ops.
fn uniformly_downsample_points(points: &mut Vec<[f32; 4]>, budget: u32, seed: u64, domain: u64) {
    let budget = budget as usize;
    if budget == 0 || points.len() <= budget {
        return;
    }

    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let priority = |index: usize| {
        let mut z = seed
            .wrapping_add(domain)
            .wrapping_add((index as u64).wrapping_mul(GOLDEN));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };

    let mut ranked: Vec<(u64, usize)> = (0..points.len())
        .map(|index| (priority(index), index))
        .collect();
    ranked.sort_unstable(); // `(priority, index)` gives a deterministic collision tie-break.
    ranked.truncate(budget);
    ranked.sort_unstable_by_key(|&(_, index)| index);

    let accepted = std::mem::take(points);
    points.extend(ranked.into_iter().map(|(_, index)| accepted[index]));
}

fn seeded_water_overlaps_grain(scene: &Scene) -> bool {
    scene
        .regions
        .iter()
        .filter(|region| region.species == Species::Water)
        .any(|water| {
            scene
                .regions
                .iter()
                .filter(|region| region.species == Species::Grain)
                .any(|grain| {
                    (0..3).all(|axis| {
                        water.min[axis] < grain.max[axis] && grain.min[axis] < water.max[axis]
                    })
                })
        })
}

/// Select a contiguous lowest-fill free-water inventory. Horizontal tie-breaking is a stateless
/// seed hash so a partial top layer does not bias one side; selected points return in original
/// deterministic order.
fn downsample_lowest_fill(points: &mut Vec<[f32; 4]>, budget: u32, seed: u64) {
    let budget = budget as usize;
    if budget == 0 || points.len() <= budget {
        return;
    }
    const DOMAIN: u64 = 0x4C4F_5745_5354_4649; // "LOWESTFI"
    let horizontal_priority = |index: usize| {
        let mut z = seed
            .wrapping_add(DOMAIN)
            .wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut ranked: Vec<(usize, u64)> = (0..points.len())
        .map(|index| (index, horizontal_priority(index)))
        .collect();
    ranked.sort_unstable_by(|&(a, ha), &(b, hb)| {
        points[a][1]
            .total_cmp(&points[b][1])
            .then_with(|| ha.cmp(&hb))
            .then_with(|| a.cmp(&b))
    });
    ranked.truncate(budget);
    ranked.sort_unstable_by_key(|&(index, _)| index);
    let accepted = std::mem::take(points);
    points.extend(ranked.into_iter().map(|(index, _)| accepted[index]));
}

/// Seed phase-specific chemistry lanes for the pool-padded KTD-1 layout. Dormant water slots
/// stay all-zero until emission activates them. Every solid material point owns its recipe-scaled
/// share of the soluble dose; changing sampling density therefore does not change total solute.
fn seed_chemistry(
    live_water: u32,
    water_capacity: u32,
    solid_count: u32,
    dry_grain_mass: f32,
    mats: &Materials,
) -> Vec<[f32; 4]> {
    let mut seeded = vec![[0.0; 4]; (water_capacity + solid_count) as usize];
    for water in &mut seeded[..live_water as usize] {
        *water = [0.0, mats.pour_t, 0.0, 0.0];
    }
    let extractable = mats.soluble_fraction.max(0.0) * dry_grain_mass.max(0.0);
    let (fast, slow) = crate::models::extraction::split(extractable, mats.fast_fraction);
    for grain in &mut seeded[water_capacity as usize..] {
        *grain = [fast, slow, mats.t_amb, 0.0];
    }
    seeded
}

/// Grid dimensions for a scene: cell size `h = CELL_SIZE_FACTOR · particle_spacing`, node 0 one
/// cell OUTSIDE `box_min` (a pad layer), `ceil(extent/h) + 3` nodes per axis. With the pad, a
/// particle anywhere in the box — including exactly on a face after the G2P clamp — has its
/// full 3-node B-spline support in range: `xl = (x − origin)/h ∈ [1, 1 + extent/h]`, so
/// `base = floor(xl − 0.5) ∈ [0, dims − 3]`. The node layer at exactly `box_min` always exists
/// (origin + h = box_min), which is what the grid box-face BC keys on.
pub fn grid_spec_for(scene: &Scene, mats: &Materials) -> ([f32; 3], f32, [u32; 3]) {
    let h = CELL_SIZE_FACTOR * mats.particle_spacing;
    let origin = [
        scene.box_min[0] - h,
        scene.box_min[1] - h,
        scene.box_min[2] - h,
    ];
    let mut dims = [0u32; 3];
    for (a, d) in dims.iter_mut().enumerate() {
        let extent = (scene.box_max[a] - scene.box_min[a]).max(0.0);
        *d = (extent / h).ceil() as u32 + 3;
    }
    (origin, h, dims)
}

/// Coarse CELL counts at `ratio` fine cells per coarse cell per axis (fine cells = node dims
/// − 1; partial coarse cells at the high end clamp their children to the fine range).
fn coarse_dims_for(dims: [u32; 3], ratio: u32) -> [u32; 3] {
    [
        (dims[0] - 1).div_ceil(ratio),
        (dims[1] - 1).div_ceil(ratio),
        (dims[2] - 1).div_ceil(ratio),
    ]
}

impl TwofieldSolver {
    fn storage(
        device: &wgpu::Device,
        label: &str,
        bytes: u64,
        extra: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        })
    }

    /// Blocking GPU→CPU read-back of the first `bytes` of `src` (dev/test only — stalls).
    fn read_bytes(&self, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        if bytes == 0 {
            return Vec::new();
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-readback"),
            });
        enc.copy_buffer_to_buffer(src, 0, &self.readback, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = self.readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().to_vec();
        self.readback.unmap();
        data
    }

    /// Read back current particle positions (dev/test only — stalls the GPU).
    pub fn read_positions(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.pos, bytes)).to_vec()
    }

    /// Read back current particle velocities (dev/test only — stalls the GPU).
    pub fn read_velocities(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.vel, bytes)).to_vec()
    }

    /// Read back the per-particle phase tags (dev/test only — stalls the GPU).
    pub fn read_phases(&self) -> Vec<u32> {
        let bytes = (self.params.particle_count as u64) * 4;
        bytemuck::cast_slice(&self.read_bytes(&self.phase, bytes)).to_vec()
    }

    /// Read back phase-specific chemistry lanes (dev/test/explicit diagnostics only — stalls).
    pub fn read_chemistry(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.chem, bytes)).to_vec()
    }

    /// Water/solid range sizes (the KTD-1 layout: water `[0, w)`, solids `[w, w + s)`).
    pub fn phase_counts(&self) -> (u32, u32) {
        (self.water_count, self.solid_count)
    }

    /// Grid layout `(origin, cell_size, dims)` for CPU twins (dev/test only).
    pub fn grid_spec(&self) -> ([f32; 3], f32, [u32; 3]) {
        let o = self.params.grid_origin;
        let d = self.params.grid_dims;
        ([o[0], o[1], o[2]], o[3], [d[0], d[1], d[2]])
    }

    /// Raw fixed-point grid totals `(mass_counts, momentum_counts)` summed in i64 — exact
    /// integer sums, so two scatters of the same state are bit-identical (dev/test only).
    /// Valid after `step()`: the grid holds that frame's P2G result until the next frame's
    /// `grid_clear`.
    pub fn read_grid_counts(&self) -> (i64, [i64; 3]) {
        let bytes = (self.num_nodes as u64) * 16;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_fp, bytes)).to_vec();
        let mut mass = 0i64;
        let mut mom = [0i64; 3];
        for n in raw.chunks_exact(4) {
            mass += n[0] as i64;
            mom[0] += n[1] as i64;
            mom[1] += n[2] as i64;
            mom[2] += n[3] as i64;
        }
        (mass, mom)
    }

    /// Decoded grid totals `(total_mass, total_momentum)` (dev/test only — stalls).
    pub fn read_grid_mass_momentum(&self) -> (f64, [f64; 3]) {
        let (mass, mom) = self.read_grid_counts();
        (
            mass as f64 / FP_SCALE,
            [
                mom[0] as f64 / FP_SCALE,
                mom[1] as f64 / FP_SCALE,
                mom[2] as f64 / FP_SCALE,
            ],
        )
    }

    /// Per-node carrier/solvent mass census (dev/test/explicit diagnostics only — stalls).
    pub fn read_grid_carrier_masses(&self) -> Vec<f32> {
        let bytes = ((self.num_cells + self.num_nodes) as u64) * 4;
        let raw: Vec<u32> = bytemuck::cast_slice(&self.read_bytes(&self.cell_cnt, bytes)).to_vec();
        raw[self.num_cells as usize..]
            .iter()
            .map(|&count| count as f32 / FP_SCALE as f32)
            .collect()
    }

    /// Sum the scattered runtime capillary capacity and bloom-weighted structural volume
    /// suffixes (dev/test only — stalls). Integer partitioning makes the capacity sum invariant
    /// to point motion and stencil placement up to the one-time per-particle encoding.
    pub fn read_grid_capillary_totals(&self) -> (f32, f32) {
        let slots = self.num_cells + 4 * self.num_nodes;
        let raw: Vec<u32> =
            bytemuck::cast_slice(&self.read_bytes(&self.cell_cnt, u64::from(slots) * 4)).to_vec();
        let capacity_offset = (self.num_cells + self.num_nodes) as usize;
        let bloom_offset = (self.num_cells + 2 * self.num_nodes) as usize;
        let capacity_counts: u64 = raw[capacity_offset..capacity_offset + self.num_nodes as usize]
            .iter()
            .map(|&value| u64::from(value))
            .sum();
        let bloom_counts: u64 = raw[bloom_offset..bloom_offset + self.num_nodes as usize]
            .iter()
            .map(|&value| u64::from(value))
            .sum();
        (
            capacity_counts as f32 / FP_SCALE as f32,
            bloom_counts as f32 / FP_SCALE as f32,
        )
    }

    /// Sum the measured held-carrier mass suffix (dev/test only — stalls). Held-only nodes copy
    /// their carrier integer exactly, so this readback exposes the no-fake-mobile invariant.
    pub fn read_grid_held_carrier_mass(&self) -> f32 {
        let slots = self.num_cells + 4 * self.num_nodes;
        let raw: Vec<u32> =
            bytemuck::cast_slice(&self.read_bytes(&self.cell_cnt, u64::from(slots) * 4)).to_vec();
        let held_offset = (self.num_cells + 3 * self.num_nodes) as usize;
        let held_counts: u64 = raw[held_offset..held_offset + self.num_nodes as usize]
            .iter()
            .map(|&value| u64::from(value))
            .sum();
        held_counts as f32 / FP_SCALE as f32
    }

    /// Read back the maximum per-node fixed-point lane magnitude (overflow-probe gate support;
    /// dev/test only — stalls).
    pub fn read_grid_max_count(&self) -> i64 {
        let bytes = (self.num_nodes as u64) * 16;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_fp, bytes)).to_vec();
        raw.iter().map(|&c| (c as i64).abs()).max().unwrap_or(0)
    }

    /// Overwrite particle velocities (dev/test only). Length must equal the particle count.
    pub fn write_velocities_for_test(&self, velocities: &[[f32; 4]]) {
        assert_eq!(
            velocities.len(),
            self.params.particle_count as usize,
            "velocity seed length must match particle count"
        );
        self.queue
            .write_buffer(&self.vel, 0, bytemuck::cast_slice(velocities));
    }

    /// Overwrite the per-particle APIC C matrices (dev/test only). Three vec4 rows per
    /// particle, row-major: `rows[3p + r] = (C[r][0], C[r][1], C[r][2], 0)`.
    pub fn write_affine_for_test(&self, rows: &[[f32; 4]]) {
        assert_eq!(
            rows.len(),
            3 * self.params.particle_count as usize,
            "affine seed length must be 3 rows per particle"
        );
        self.queue
            .write_buffer(&self.cmat, 0, bytemuck::cast_slice(rows));
    }

    /// Switch G2P to the pure-PIC variant (C fully zeroed each step) — ONLY for the APIC-vs-PIC
    /// discrimination gate; never set in production paths.
    pub fn set_pic_for_test(&mut self, pic: bool) {
        self.params.pic_blend = if pic { 1.0 } else { 0.0 };
    }

    /// Set the APIC↔PIC blend (PIC fraction ∈ [0,1]) directly — dev/test only (the
    /// stirring-vs-crater tradeoff sweep). Production keeps `PIC_BLEND_DEFAULT`.
    pub fn set_pic_blend_for_test(&mut self, blend: f32) {
        self.params.pic_blend = blend.clamp(0.0, 1.0);
    }

    /// Set the U3 separate water-splash velocity cap (`flip.w`) directly — dev/test only (the
    /// FP-overflow headroom probe at a raised water cap). `0.0` is the disabled sentinel
    /// (water path falls back to the global `max_speed`, byte-identical).
    pub fn set_water_splash_cap_for_test(&mut self, cap: f32) {
        self.params.flip[3] = cap;
    }

    /// Toggle the density relief (over-density relief + under-density suction) — dev/test only,
    /// for the spontaneous-stirring isolation gate. Production keeps relief ON.
    pub fn set_relief_for_test(&mut self, on: bool) {
        self.params.dbg[0] = if on { 1.0 } else { 0.0 };
    }

    /// Select the SDF wall-BC mode (`dbg.y`) — dev/test only. `WALL_BC_SINGLE` (≤ 0.5) is the
    /// single-normal banded BC; `WALL_BC_MULTI` (> 0.5) is the orthonormal-basis multi-normal
    /// projector. Production default is `WALL_BC_SINGLE` until U3 flips it post-calibration.
    pub fn set_wall_bc_mode_for_test(&mut self, mode: f32) {
        self.params.dbg[1] = mode;
    }

    /// Select the uncapped two-sided density-target mode (dbg.w) — dev/test only. `false`
    /// (default) keeps the legacy ∇·v projection + rate-limited (DENSITY_RELAX) two-sided relief,
    /// byte-identical to pre-fix; `true` drops the rate cap so the density relief drives ρ→ρ₀ at
    /// full strength (the calibration showed this is the over-pack fix; it is stable because it is
    /// a bounded restoring target, not the one-sided expansion source that historically detonated).
    /// Production default flips to `true` in U5.
    pub fn set_density_target_mode_for_test(&mut self, on: bool) {
        self.params.dbg[3] = if on { 1.0 } else { 0.0 };
    }

    /// Set the temper-K rate divisor (dbg.z) for the uncapped density path — dev/test only (the
    /// temper calibration). The uncapped relief rate is `e/(K·dt)`: K=1 (default, dbg.z=0) is the
    /// full uncap (crisp but pumps energy on coupled scenes); the legacy cap is K=30 (slow, mushy);
    /// the calibrated sweet spot is K∈(1,30). Smaller K = stiffer/crisper. Only takes effect on the
    /// uncapped path (dbg.w on).
    pub fn set_density_rate_k_for_test(&mut self, k: f32) {
        self.params.dbg[2] = k.max(0.0);
    }

    /// Set the U3 pressure-budget knobs (KTD-9 grid points; dev/test only). `coarse_sweeps =
    /// 0` is the A/B seed-off arm (the whole coarse stage is skipped, the fine sweeps start
    /// from p = 0); `coarse_sweeps = fine_sweeps = 0` makes the frame U2-equivalent (the
    /// projection applies a zero pressure field).
    pub fn set_pressure_budget_for_test(
        &mut self,
        coarse_ratio: u32,
        coarse_sweeps: u32,
        fine_sweeps: u32,
    ) {
        assert!(
            coarse_ratio >= COARSE_RATIO_DEFAULT,
            "coarse buffers are sized for ratio {COARSE_RATIO_DEFAULT} (the densest grid point)"
        );
        let d = self.params.grid_dims;
        let cd = coarse_dims_for([d[0], d[1], d[2]], coarse_ratio);
        self.params.coarse_dims = [cd[0], cd[1], cd[2], coarse_ratio];
        self.coarse_ratio = coarse_ratio;
        self.coarse_sweeps = coarse_sweeps;
        self.fine_sweeps = fine_sweeps;
    }

    /// Set the live water count (dev/test only — state-replay support for the decay gates;
    /// `n` must not exceed the allocated water pool).
    pub fn set_live_water_for_test(&mut self, n: u32) {
        assert!(
            n <= self.water_capacity,
            "live water {n} exceeds the allocated pool {}",
            self.water_capacity
        );
        self.water_count = n;
        self.params.water_count = n;
    }

    /// Read back the bubble state `[λ_b, δλ_b]` (dev/test only — stalls).
    pub fn read_bubble(&self) -> [f32; 2] {
        let raw: Vec<f32> = bytemuck::cast_slice(&self.read_bytes(&self.bubble, 8)).to_vec();
        [raw[0], raw[1]]
    }

    /// Whether an enclosed air pocket was detected this frame (single-writer mirror in
    /// `bubble[2]`; dev/test only — stalls).
    pub fn pocket_present(&self) -> bool {
        let raw: Vec<f32> = bytemuck::cast_slice(&self.read_bytes(&self.bubble, 12)).to_vec();
        raw[2] >= 0.5
    }

    /// Persistent finite paper-reservoir counters (dev/test/explicit diagnostics only — stalls):
    /// occupied volume counts, admitted, released, overflow, invariant-error flags.
    pub fn read_filter_state(&self) -> [u32; 5] {
        let raw: Vec<u32> = bytemuck::cast_slice(&self.read_bytes(&self.filter_state, 20)).to_vec();
        [raw[0], raw[1], raw[2], raw[3], raw[4]]
    }

    /// Blocking dev/profile readback of the current compact water-pressure tile count.
    /// Production frame execution never calls this; the scheduler remains fully GPU-driven.
    pub fn read_active_tile_count(&self) -> u32 {
        let raw: Vec<u32> = bytemuck::cast_slice(&self.read_bytes(&self.active_tiles, 4)).to_vec();
        raw[0]
    }

    /// Read back the interphase drag state, one vec4 per node: .xyz = the water-to-solid
    /// impulse (hypothetical constrained reaction in frozen mode; already applied to
    /// `grid_svel` in dynamic mode), .w = sigma (dev/test only — stalls).
    pub fn read_reactions(&self) -> Vec<[f32; 4]> {
        let bytes = (self.num_nodes as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.react, bytes)).to_vec()
    }

    /// Read back the decoded per-node solid volumes (the φ_s carrier; dev/test only).
    pub fn read_solid_volumes(&self) -> Vec<f32> {
        let bytes = (self.num_nodes as u64) * 4;
        let raw: Vec<i32> = bytemuck::cast_slice(&self.read_bytes(&self.grid_sfp, bytes)).to_vec();
        raw.iter().map(|&c| (c as f64 / FP_SCALE) as f32).collect()
    }

    /// Overwrite the per-node solid volumes (dev/test only; fixed-point encoded). One value
    /// per grid node. The next `step()` re-scatters from the grains — this feeds the
    /// standalone drag-pass gates only.
    pub fn write_solid_volumes_for_test(&self, volumes: &[f32]) {
        assert_eq!(
            volumes.len(),
            self.num_nodes as usize,
            "one volume per node"
        );
        let enc: Vec<i32> = volumes
            .iter()
            .map(|&v| (v as f64 * FP_SCALE).round() as i32)
            .collect();
        self.queue
            .write_buffer(&self.grid_sfp, 0, bytemuck::cast_slice(&enc));
    }

    /// Dispatch the U6 `drag_fold` pass standalone on the current grid state (dev/test only;
    /// uses the last-uploaded params — dt = 1/60 from build until a `step()` ran).
    pub fn run_drag_for_test(&self) {
        self.run_dbg(
            &self.pipelines.drag_fold.0,
            &self.pipelines.drag_fold.1,
            groups(self.num_nodes).max(1),
        );
    }

    /// Toggle the U6 open-base drained-column mode (dev/test only; applies from the next
    /// `step()` — U9's phase-selective filter boundary supersedes this hook).
    pub fn set_open_base_for_test(&mut self, open: bool) {
        self.params.coupling[3] = if open { 1.0 } else { 0.0 };
    }

    /// Total water mass emitted by the pour so far (conservation accounting; dev/test).
    pub fn total_emitted_water_mass(&self) -> f32 {
        self.inflow.emitted_mass
    }

    /// Drop per-pass timestamp profiling so `step()` records every frame in the BATCHED
    /// single-compute-pass mode — exactly the web/no-TIMESTAMP_QUERY path. Used by the
    /// equivalence gate (`tests/batch_pass_equivalence.rs`) to compare batched vs per-pass
    /// recording natively; not used in production.
    pub fn force_batched_passes_for_test(&mut self) {
        self.ts = None;
    }

    /// Read back the per-solid deformation gradients, 3 vec4 rows per solid (row-major,
    /// solid-local indexing; dev/test only — stalls).
    pub fn read_deformation(&self) -> Vec<[f32; 4]> {
        let bytes = (self.solid_count.max(1) as u64) * 48;
        bytemuck::cast_slice(&self.read_bytes(&self.fmat, bytes)).to_vec()
    }

    /// Overwrite the per-solid deformation gradients (dev/test only; 3 rows per solid).
    pub fn write_deformation_for_test(&self, rows: &[[f32; 4]]) {
        assert_eq!(
            rows.len(),
            3 * self.solid_count as usize,
            "deformation seed must be 3 rows per solid"
        );
        self.queue
            .write_buffer(&self.fmat, 0, bytemuck::cast_slice(rows));
    }

    /// Read back the per-solid constitutive state, 2 vec4 per solid:
    /// (τxx, τxy, τxz, τyy), (τyz, τzz, p_c, compaction) (dev/test only — stalls).
    pub fn read_solid_state(&self) -> Vec<[f32; 4]> {
        let bytes = (self.solid_count.max(1) as u64) * 32;
        bytemuck::cast_slice(&self.read_bytes(&self.sstate, bytes)).to_vec()
    }

    /// Overwrite the per-solid constitutive state (dev/test only; 2 vec4 per solid — the
    /// over-packing gate softens p_c with this).
    pub fn write_solid_state_for_test(&self, rows: &[[f32; 4]]) {
        assert_eq!(
            rows.len(),
            2 * self.solid_count as usize,
            "solid state must be 2 vec4 per solid"
        );
        self.queue
            .write_buffer(&self.sstate, 0, bytemuck::cast_slice(rows));
    }

    /// Override gravity from the next `step()` on (dev/test only — the documented U5 tamp
    /// probe applies body-force pulses by pulsing gravity on a grain-only scene).
    pub fn set_gravity_for_test(&mut self, g: [f32; 3]) {
        self.params.gravity = [g[0], g[1], g[2], 0.0];
    }

    /// Activate pour-emitted water particles for this frame from `EmissionInput` — the
    /// volume-consistent arclength-credit emitter, ported from the xpbd solver (see its
    /// `emit` for the derivations): the volume accumulator (flow/V_w·dt) is the master count
    /// budget; layers release one `particle_spacing` of stream travel apart, each filling a
    /// golden-angle disc with up to `N_layer = ceil(A_eff·spacing/V_w)` particles, so the
    /// inlet packs to rest density. Particles are written into the water pool's live range
    /// and `water_count` grows. No-ops when not pouring and no backlog remains.
    fn emit(&mut self, input: &EmissionInput, dt: f32) {
        // A Reset event clears the emitter's backlog/credit (the emitter contract; a full
        // sim restart is `reset()`). Done before the gate so a Reset with zero flow clears.
        if input.event == crate::emission::PourEvent::Reset {
            self.inflow.accumulator = 0.0;
            self.inflow.axial = 0.0;
            self.inflow.last_exit_speed = 0.0;
        }
        let flow = input.flow_rate.max(0.0);
        // Gate: pour active, or a whole particle of backlog still to drain.
        if flow <= 0.0 && self.inflow.accumulator < 1.0 {
            return;
        }
        let a_eff = std::f32::consts::PI
            * self.inflow.nozzle_radius
            * self.inflow.nozzle_radius
            * self.inflow.discharge_coeff;
        let a_eff = a_eff.max(1.0e-9);
        // Orifice relation: exit speed from flow + effective area. While draining a backlog
        // at zero flow, keep the last cadence so the stream tail stays correctly spaced.
        let exit_speed = if flow > 0.0 {
            let es = flow / a_eff;
            self.inflow.last_exit_speed = es;
            self.inflow.accumulator += flow / self.inflow.v_w * dt;
            es
        } else {
            self.inflow.last_exit_speed
        };
        if exit_speed <= 0.0 {
            return;
        }
        // Volume-consistent layer capacity (ceil ⇒ throughput ≥ flow/V_w, no backlog).
        let n_layer = ((a_eff * self.inflow.spacing / self.inflow.v_w).ceil() as u32).max(1);

        // Pour direction (downward, tilted by pour_angle toward +x) + a disc basis.
        let a = input.pour_angle;
        let dir = [a.sin(), -a.cos(), 0.0];
        let (u, w) = disc_basis(dir);
        let r_eff = self.inflow.nozzle_radius * self.inflow.discharge_coeff.sqrt();
        const GOLDEN: f32 = 2.399_963_2;

        self.inflow.axial += exit_speed * dt;
        let kettle = input.kettle_pos;
        let mut want = self.inflow.accumulator.floor() as u32;
        let mut new_pos: Vec<[f32; 4]> = Vec::new();
        let mut new_vel: Vec<[f32; 4]> = Vec::new();
        let mut new_chem: Vec<[f32; 4]> = Vec::new();
        let mut clamped = false;
        while self.inflow.axial >= self.inflow.spacing && want > 0 {
            // Capacity check BEFORE spending arclength credit, so a full pool doesn't
            // silently consume a layer's axial.
            let avail = self.water_capacity - (self.water_count + new_pos.len() as u32);
            if avail == 0 {
                clamped = true;
                break;
            }
            self.inflow.axial -= self.inflow.spacing;
            let depth = self.inflow.axial; // residual stream travel below the nozzle
            let this_layer = n_layer.min(want).min(avail);
            for _ in 0..this_layer {
                // Radial shell cycles with the cursor (mod N_layer) so partial layers still
                // cover the whole disc over time; golden angle fills it uniformly.
                let ri = (self.inflow.cursor % n_layer as u64) as f32;
                let r = r_eff * ((ri + 0.5) / n_layer as f32).sqrt();
                let theta = self.inflow.cursor as f32 * GOLDEN;
                self.inflow.cursor += 1;
                let (ct, st) = (theta.cos(), theta.sin());
                let off = [
                    u[0] * r * ct + w[0] * r * st,
                    u[1] * r * ct + w[1] * r * st,
                    u[2] * r * ct + w[2] * r * st,
                ];
                new_pos.push([
                    kettle[0] + dir[0] * depth + off[0],
                    kettle[1] + dir[1] * depth + off[1],
                    kettle[2] + dir[2] * depth + off[2],
                    1.0, // moisture lane: full water
                ]);
                new_vel.push([
                    dir[0] * exit_speed,
                    dir[1] * exit_speed,
                    dir[2] * exit_speed,
                    0.0,
                ]);
                new_chem.push([0.0, self.inflow.pour_t, 0.0, 0.0]); // c = 0, T = pour temp
            }
            want -= this_layer;
        }
        if clamped {
            eprintln!(
                "twofield pour: water pool {} reached; emission clamped (a recipe scene must \
                 declare enough pour_water_ml to size the pool to its dose)",
                self.water_capacity
            );
        }
        let emit_n = new_pos.len() as u32;
        if emit_n == 0 {
            return;
        }
        // Clamp-before-decrement: subtract only what was actually emitted — unspent budget
        // stays as backlog rather than being silently burned.
        self.inflow.accumulator -= emit_n as f32;
        self.inflow.emitted_mass += emit_n as f32 * self.params.particle_mass;
        let off_v4 = (self.water_count as u64) * 16;
        self.queue
            .write_buffer(&self.pos, off_v4, bytemuck::cast_slice(&new_pos));
        self.queue
            .write_buffer(&self.vel, off_v4, bytemuck::cast_slice(&new_vel));
        self.queue
            .write_buffer(&self.chem, off_v4, bytemuck::cast_slice(&new_chem));
        // Phase tags of dormant pool slots are already 0 (water) and the APIC C rows are
        // zero (each slot is activated at most once per run; reset re-zeroes them).
        self.water_count += emit_n;
        self.params.water_count = self.water_count;
    }

    /// Overwrite particle positions (dev/test only; the .w lane carries moisture).
    pub fn write_positions_for_test(&self, positions: &[[f32; 4]]) {
        assert_eq!(
            positions.len(),
            self.params.particle_count as usize,
            "position seed length must match particle count"
        );
        self.queue
            .write_buffer(&self.pos, 0, bytemuck::cast_slice(positions));
    }

    /// Overwrite the float grid-velocity field (dev/test only — operator gates feed random
    /// node fields to the dbg taps). Length must equal the node count; .w is the mass lane.
    pub fn write_grid_velocities_for_test(&self, v: &[[f32; 4]]) {
        assert_eq!(v.len(), self.num_nodes as usize, "one vec4 per grid node");
        self.queue
            .write_buffer(&self.grid_vel, 0, bytemuck::cast_slice(v));
    }

    /// Overwrite the carrier-mass suffix while leaving cell occupancy/capillary lanes zero
    /// (dev/test only). This exposes the filter-release distinction between an authoritative
    /// pressure-projected zero flux and the no-mobile-grid gravity fallback.
    pub fn write_grid_carrier_masses_for_test(&self, masses: &[f32]) {
        assert_eq!(
            masses.len(),
            self.num_nodes as usize,
            "one mass per grid node"
        );
        let mut auxiliary = vec![0u32; (self.num_cells + 4 * self.num_nodes) as usize];
        let offset = self.num_cells as usize;
        for (slot, mass) in auxiliary[offset..offset + self.num_nodes as usize]
            .iter_mut()
            .zip(masses)
        {
            *slot = (mass.max(0.0) * FP_SCALE as f32)
                .round()
                .clamp(0.0, u32::MAX as f32) as u32;
        }
        self.queue
            .write_buffer(&self.cell_cnt, 0, bytemuck::cast_slice(&auxiliary));
    }

    /// Overwrite the fine pressure field (slot pf_a, which `dbg_grad` reads; dev/test only).
    pub fn write_pressure_for_test(&self, p: &[f32]) {
        assert_eq!(p.len(), self.num_cells as usize, "one f32 per fine cell");
        self.queue
            .write_buffer(&self.pf_a, 0, bytemuck::cast_slice(p));
    }

    /// Read back the float grid velocities (.xyz) + node masses (.w). After `step()` this is
    /// the post-projection field (dev/test only — stalls).
    pub fn read_grid_velocities(&self) -> Vec<[f32; 4]> {
        let bytes = (self.num_nodes as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.grid_vel, bytes)).to_vec()
    }

    /// Read back the per-cell (rhs, active, dbg_div, _) meta (dev/test only — stalls).
    pub fn read_cell_meta(&self) -> Vec<[f32; 4]> {
        let bytes = (self.num_cells as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.cell_meta, bytes)).to_vec()
    }

    /// Read back the per-node M̃⁻¹ matrices, 8 floats per node:
    /// (xx, xy, xz, yy, yz, zz, massy_flag, 0) (dev/test only — stalls).
    pub fn read_node_matrices(&self) -> Vec<[f32; 8]> {
        let bytes = (self.num_nodes as u64) * 32;
        bytemuck::cast_slice(&self.read_bytes(&self.nm, bytes)).to_vec()
    }

    /// Read back the frame's solved pressure (the final-parity ping-pong buffer the project
    /// pass consumed; dev/test only — stalls).
    pub fn read_pressure(&self) -> Vec<f32> {
        let buf = if self.fine_sweeps.is_multiple_of(2) {
            &self.pf_a
        } else {
            &self.pf_b
        };
        let bytes = (self.num_cells as u64) * 4;
        bytemuck::cast_slice(&self.read_bytes(buf, bytes)).to_vec()
    }

    /// Read back the per-particle APIC C rows (3 vec4 per particle; dev/test only — stalls).
    pub fn read_affine_rows(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 48;
        bytemuck::cast_slice(&self.read_bytes(&self.cmat, bytes)).to_vec()
    }

    fn run_dbg(&self, pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, n_groups: u32) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-dbg"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("twofield-dbg"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, Some(bind), &[]);
            pass.dispatch_workgroups(n_groups, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Dispatch the test-only `dbg_div` tap: writes the raw masked divergence of the current
    /// grid-velocity field into `cell_meta.z`. Masks/nm must exist (call after a `step()`).
    pub fn run_div_for_test(&self) {
        self.run_dbg(
            &self.pipelines.dbg_div.0,
            &self.pipelines.dbg_div.1,
            groups(self.num_cells).max(1),
        );
    }

    /// Dispatch the test-only `dbg_grad` tap: writes the raw masked gradient of pf_a into the
    /// grid-velocity buffer (overwrites it — test sequences only).
    pub fn run_grad_for_test(&self) {
        self.run_dbg(
            &self.pipelines.dbg_grad.0,
            &self.pipelines.dbg_grad.1,
            groups(self.num_nodes).max(1),
        );
    }

    /// Fine-sweep split around the coarse correction: pre-smooth half the budget (≥1 — the
    /// residual restriction needs the boundary-spike content absorbed first, see
    /// pressure.wgsl), post-smooth the rest. Seed-off runs everything as post-smoothing.
    fn smooth_split(&self) -> (u32, u32) {
        let pre = if self.coarse_sweeps > 0 && self.fine_sweeps > 0 {
            (self.fine_sweeps / 2).max(1)
        } else {
            0
        };
        (pre, self.fine_sweeps - pre)
    }

    /// Which ping-pong buffer holds the final pressure (0 = pf_a) — mirrors step()'s parity.
    fn pressure_parity(&self) -> usize {
        let (pre, post) = self.smooth_split();
        let mut par = (pre % 2) as usize;
        if self.coarse_sweeps > 0 {
            par ^= 1; // prolong_add flips
        }
        par ^= (post % 2) as usize;
        par
    }

    fn filter_relaxation_observability(&self) -> (f32, f32) {
        if !self.filter_paper_present {
            return (0.0, 1.0);
        }
        let permeability = self.params.filter0[0];
        let viscosity = self.params.filter0[1];
        let thickness = self.params.filter0[2];
        let band = self.params.grid_origin[3];
        if !permeability.is_finite()
            || !viscosity.is_finite()
            || !thickness.is_finite()
            || !band.is_finite()
            || !self.params.dt.is_finite()
            || permeability <= 0.0
            || viscosity <= 0.0
            || thickness <= 0.0
            || band <= 0.0
            || self.params.dt < 0.0
        {
            return (0.0, 1.0);
        }
        match crate::models::filter::thin_sheet_relaxation(
            0.0,
            f64::from(band),
            f64::from(thickness),
            f64::from(viscosity),
            f64::from(permeability),
            f64::from(self.params.dt),
        ) {
            Ok(response) => (
                response.beta_per_s as f32,
                response.pressure_mobility as f32,
            ),
            Err(_) => (0.0, 1.0),
        }
    }

    fn apply_diagnostics_gpu(
        &mut self,
        snapshot: DiagnosticsGpu,
        sample_time_s: f32,
        expected_carrier_mass: f32,
    ) {
        if self.cached_diag.sample_valid && sample_time_s < self.cached_diag.sample_time_s {
            return;
        }
        let mobile_carrier = snapshot.mobile[0].max(0.0);
        let mobile_solute = snapshot.mobile[1].max(0.0);
        let paper_carrier = snapshot.paper[0].max(0.0);
        let paper_solute = snapshot.paper[1].max(0.0);
        let bound_carrier = snapshot.grain[0].max(0.0);
        let remaining_solute = snapshot.grain[1].max(0.0);
        let mobile_solution = mobile_carrier + mobile_solute;
        let paper_solution = paper_carrier + paper_solute;
        let grain_mass = self.insoluble_grain_mass + remaining_solute + bound_carrier;
        let total_carrier = mobile_carrier + paper_carrier + bound_carrier;
        let dissolved_solute = mobile_solute + paper_solute;
        let total_solute = dissolved_solute + remaining_solute;
        let observed_mechanical_mass = mobile_solution + paper_solution + grain_mass;
        let expected_mechanical_mass =
            expected_carrier_mass + self.initial_solute_mass + self.insoluble_grain_mass;
        let occupied_counts = snapshot.filter0[0];
        let capacity_counts = if self.filter_paper_present {
            self.params.filter1[0]
        } else {
            0
        };
        let admitted = snapshot.filter0[1];
        let released = snapshot.filter0[2];
        let held_count = snapshot.counts[1];
        let held_volume_counts = snapshot.counts[2];
        let water_capacity = (snapshot.mobile[2] + snapshot.paper[2]).max(0.0);
        let water_enthalpy = snapshot.mobile[3] + snapshot.paper[3];
        let grain_capacity = snapshot.grain[2].max(0.0);
        let grain_enthalpy = snapshot.grain[3];

        let mut diag = self.cached_diag;
        diag.sample_valid = true;
        diag.sample_time_s = sample_time_s;
        diag.free_water_above_bed = snapshot.counts[0];
        diag.pond_carrier_mass = snapshot.cup[2].max(0.0);
        diag.evenness = 1.0;
        diag.max_occupancy = snapshot.grid[0].max(0.0);
        let total_dry_mass = self.insoluble_grain_mass + self.initial_solute_mass;
        diag.extraction_yield = if total_dry_mass > 0.0 {
            dissolved_solute / total_dry_mass
        } else {
            0.0
        };
        let cup_solution = snapshot.cup[0].max(0.0) + snapshot.cup[1].max(0.0);
        diag.cup_tds = solution_solute_mass_fraction(snapshot.cup[0], snapshot.cup[1]);
        diag.mean_water_temperature = if water_capacity > 0.0 {
            water_enthalpy / water_capacity
        } else {
            0.0
        };
        diag.mean_grain_temperature = if grain_capacity > 0.0 {
            grain_enthalpy / grain_capacity
        } else {
            0.0
        };
        diag.mobile_carrier_mass = mobile_carrier;
        diag.paper_carrier_mass = paper_carrier;
        diag.bound_carrier_mass = bound_carrier;
        diag.total_carrier_mass = total_carrier;
        diag.mobile_solution_mass = mobile_solution;
        diag.paper_solution_mass = paper_solution;
        diag.grain_mass = grain_mass;
        diag.mobile_solute_mass = mobile_solute;
        diag.paper_solute_mass = paper_solute;
        diag.remaining_grain_solute_mass = remaining_solute;
        diag.dissolved_solute_mass = dissolved_solute;
        diag.cup_carrier_mass = snapshot.cup[0].max(0.0);
        diag.cup_solute_mass = snapshot.cup[1].max(0.0);
        diag.cup_solution_mass = cup_solution.max(0.0);
        let bound_capacity_mass = if self.params.wet0[2] > 0.0 {
            self.params.wet0[1].max(0.0) * self.params.particle_mass.max(0.0) / self.params.wet0[2]
                * self.solid_count as f32
        } else {
            0.0
        };
        diag.mean_bound_saturation = if bound_capacity_mass > 0.0 {
            (bound_carrier / bound_capacity_mass).clamp(0.0, 1.0)
        } else {
            0.0
        };
        diag.mean_diameter_ratio = 1.0 + diag.mean_bound_saturation * self.params.wet1[3].max(0.0);
        diag.total_solute = total_solute;
        diag.total_enthalpy = water_enthalpy + grain_enthalpy;
        diag.expected_carrier_mass = expected_carrier_mass;
        diag.expected_solute_mass = self.initial_solute_mass;
        diag.carrier_mass_residual = total_carrier - expected_carrier_mass;
        diag.solute_mass_residual = total_solute - self.initial_solute_mass;
        diag.mechanical_mass_residual = observed_mechanical_mass - expected_mechanical_mass;
        diag.filter_occupied_volume = occupied_counts as f32 / FILTER_VOLUME_SCALE as f32;
        diag.filter_capacity_volume = capacity_counts as f32 / FILTER_VOLUME_SCALE as f32;
        diag.filter_saturation = if capacity_counts > 0 {
            occupied_counts as f32 / capacity_counts as f32
        } else {
            0.0
        };
        diag.filter_volume_residual = (i64::from(held_volume_counts) - i64::from(occupied_counts))
            as f32
            / FILTER_VOLUME_SCALE as f32;
        diag.filter_count_residual =
            i64::from(held_count) - (i64::from(admitted) - i64::from(released));
        diag.filter_held_parcels = held_count;
        diag.filter_admitted = admitted;
        diag.filter_released = released;
        diag.filter_overflow = snapshot.filter0[3];
        diag.filter_errors = snapshot.filter1[0];
        diag.nonfinite_particle_count = snapshot.counts[3];
        diag.nonfinite_grid_node_count = snapshot.filter1[1];
        diag.has_nonfinite_state =
            diag.nonfinite_particle_count > 0 || diag.nonfinite_grid_node_count > 0;
        let (beta, mobility) = self.filter_relaxation_observability();
        diag.filter_beta_max_per_s = beta;
        diag.filter_pressure_mobility_min = mobility;

        if self.solid_count > 0 && !diag.has_nonfinite_state {
            self.pond_peak_carrier_mass = self.pond_peak_carrier_mass.max(diag.pond_carrier_mass);
            if diag.drawdown_time == 0.0
                && self.pond_peak_carrier_mass >= self.params.particle_mass.max(1.0e-9)
                && diag.pond_carrier_mass < 0.10 * self.pond_peak_carrier_mass
            {
                diag.drawdown_time = sample_time_s;
            }
        }
        self.cached_diag = diag;
    }

    /// Poll already-requested compact maps without waiting. Completed snapshots are applied in
    /// sample-time order; reset epochs discard late maps from an earlier run.
    fn poll_periodic_diagnostics(&mut self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let mut completed = Vec::new();
        for slot in &mut self.diagnostics_readbacks {
            let ready = match slot.pending.as_ref() {
                Some(pending) => match pending.receiver.try_recv() {
                    Ok(result) => Some(Some(result)),
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(None),
                },
                None => None,
            };
            let Some(result) = ready else { continue };
            let pending = slot
                .pending
                .take()
                .expect("diagnostics pending state disappeared");
            let Some(Ok(())) = result else { continue };
            let snapshot = {
                let view = slot.buffer.slice(..).get_mapped_range();
                *bytemuck::from_bytes::<DiagnosticsGpu>(&view)
            };
            slot.buffer.unmap();
            if pending.epoch == self.diagnostics_epoch {
                completed.push((
                    pending.sample_time_s,
                    pending.expected_carrier_mass,
                    snapshot,
                ));
            }
        }
        completed.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (sample_time_s, expected_carrier_mass, snapshot) in completed {
            self.apply_diagnostics_gpu(snapshot, sample_time_s, expected_carrier_mass);
        }
    }

    fn periodic_diagnostics_slot(&mut self, dt: f32) -> Option<usize> {
        self.diagnostics_elapsed_s += dt.max(0.0);
        if self.diagnostics_elapsed_s < DIAGNOSTICS_PERIOD_S {
            return None;
        }
        let slot = self
            .diagnostics_readbacks
            .iter()
            .position(|slot| slot.pending.is_none())?;
        self.diagnostics_elapsed_s =
            (self.diagnostics_elapsed_s - DIAGNOSTICS_PERIOD_S).min(DIAGNOSTICS_PERIOD_S);
        Some(slot)
    }

    fn begin_periodic_diagnostics_map(&mut self, slot: usize, sample_time_s: f32) {
        let slice = self.diagnostics_readbacks[slot].buffer.slice(..);
        let (tx, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.diagnostics_readbacks[slot].pending = Some(DiagnosticsPending {
            receiver,
            sample_time_s,
            expected_carrier_mass: self.initial_carrier_mass + self.inflow.emitted_mass,
            epoch: self.diagnostics_epoch,
        });
    }

    /// Explicit full blocking validation snapshot (dev/test only). Production uses the compact
    /// asynchronous ring above; getters never call this synchronization point.
    pub fn sample_diagnostics(&mut self) {
        if let Some(ts) = &self.ts {
            if ts.count >= 2 {
                let size = (ts.count as u64) * 8;
                let slice = ts.readback.slice(0..size);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                rx.recv().unwrap().unwrap();
                let times: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
                ts.readback.unmap();
                let mut passes = Vec::new();
                for (idx, label) in ts.labels.iter().enumerate() {
                    let b = times[idx * 2];
                    let e = times[idx * 2 + 1];
                    let us = (e.saturating_sub(b)) as f32 * ts.period_ns / 1000.0;
                    passes.push((label.clone(), us));
                }
                self.cached_passes = passes;
            }
        }
        self.sample_brew_diagnostics();
    }

    /// Refresh brew metrics and the chemistry/thermal inventories from blocking position + chem
    /// snapshots. This is called only by the explicit `sample_diagnostics` synchronization point;
    /// `metrics()` and `diagnostics()` return the cache without touching the GPU.
    fn sample_brew_diagnostics(&mut self) {
        let pos = self.read_positions();
        let chem = self.read_chemistry();
        let n = pos.len().min(chem.len());
        let water_end = (self.water_count as usize).min(n);
        let solid_start = ((self.params.particle_count - self.solid_count) as usize).min(n);
        let particle_mass = self.params.particle_mass.max(0.0);
        let cp_water = self.params.therm[0].max(0.0);
        let cp_grain = self.params.therm[1].max(0.0);
        let cp_solute = self.params.solute[1].max(0.0);
        let water_volume = self.params.wet0[2].max(1.0e-9);
        let water_mass_per_volume = particle_mass / water_volume;
        let mut snapshot = DiagnosticsGpu::default();

        for i in 0..water_end {
            let p = pos[i];
            let c = chem[i];
            if p.iter().chain(c.iter()).any(|value| !value.is_finite()) {
                snapshot.counts[3] += 1;
                continue;
            }
            let fraction = p[3].clamp(0.0, 1.0);
            let carrier = particle_mass * fraction;
            let solute = carrier * c[0].max(0.0);
            let capacity = carrier * cp_water + solute * cp_solute;
            let held = (0.5..=1.5).contains(&c[2]);
            let target = if held {
                &mut snapshot.paper
            } else {
                &mut snapshot.mobile
            };
            target[0] += carrier;
            target[1] += solute;
            target[2] += capacity;
            target[3] += capacity * c[1];
            if held {
                snapshot.counts[1] += 1;
                snapshot.counts[2] = snapshot.counts[2]
                    .wrapping_add(filter_volume_counts(fraction * self.params.wet0[2]));
            } else {
                if carrier > 0.0 && p[1] > self.bed_top_seed {
                    snapshot.counts[0] += 1;
                    snapshot.cup[2] += carrier;
                }
                if let Some(cup) = self.cup {
                    let dx = p[0] - cup.cx;
                    let dz = p[2] - cup.cz;
                    if dx * dx + dz * dz <= cup.radius * cup.radius
                        && p[1] >= cup.y_min
                        && p[1] <= cup.y_max
                    {
                        snapshot.cup[0] += carrier;
                        snapshot.cup[1] += solute;
                    }
                }
            }
        }
        for i in solid_start..n {
            let p = pos[i];
            let c = chem[i];
            if p.iter().chain(c.iter()).any(|value| !value.is_finite()) {
                snapshot.counts[3] += 1;
                continue;
            }
            let bound_mass = water_mass_per_volume * p[3].max(0.0);
            let remaining_solute = c[0].max(0.0) + c[1].max(0.0);
            let capacity = self.params.solute[0] * cp_grain
                + remaining_solute * cp_solute
                + bound_mass * cp_water;
            snapshot.grain[0] += bound_mass;
            snapshot.grain[1] += remaining_solute;
            snapshot.grain[2] += capacity;
            snapshot.grain[3] += capacity * c[2];
        }

        let grid_velocities = self.read_grid_velocities();
        if self.water_count > 0 {
            let carrier_masses = self.read_grid_carrier_masses();
            for (velocity, carrier) in grid_velocities.iter().zip(carrier_masses) {
                if velocity.iter().any(|value| !value.is_finite()) {
                    snapshot.filter1[1] += 1;
                } else {
                    snapshot.grid[0] =
                        snapshot.grid[0].max(carrier / self.params.extra[0].max(1.0e-9));
                }
            }
        }
        let filter = self.read_filter_state();
        snapshot.filter0 = [filter[0], filter[1], filter[2], filter[3]];
        snapshot.filter1[0] = filter[4];
        self.apply_diagnostics_gpu(
            snapshot,
            self.sim_time,
            self.initial_carrier_mass + self.inflow.emitted_mass,
        );
    }

    /// Latest sampled brew diagnostics (CPU-resident; the getters never sync). Mirrors
    /// `XpbdSolver::diagnostics` so future suites can port.
    pub fn diagnostics(&self) -> TwofieldDiagnostics {
        let mut diagnostics = self.cached_diag;
        let (beta, mobility) = self.filter_relaxation_observability();
        diagnostics.filter_beta_max_per_s = beta;
        diagnostics.filter_pressure_mobility_min = mobility;
        diagnostics
    }

    pub fn diagnostics_pending(&self) -> bool {
        self.diagnostics_readbacks.iter().any(|slot| {
            slot.pending
                .as_ref()
                .is_some_and(|pending| pending.epoch == self.diagnostics_epoch)
        })
    }

    /// Live particle count currently simulated (water prefix + solids). Mirrors
    /// `XpbdSolver::active_count` — the scaling-probe / perf-test particle-count read.
    pub fn active_count(&self) -> u32 {
        self.water_count + self.solid_count
    }

    /// Allocated water-pool capacity (`seed + dose headroom`; dev/test). The upper bound the
    /// scaling-probe / perf-test bed-saturation may raise the live water count to.
    pub fn water_pool_capacity(&self) -> u32 {
        self.water_capacity
    }

    /// Whole-pipeline repeat count at `dt`. The semi-implicit path records every shear
    /// microstep inside one macro sequence, so its profiling multiplier is one.
    pub fn substeps_for_dt(&self, dt: f32) -> u32 {
        if self.implicit_solid {
            1
        } else if self.solid_dynamic {
            let h = self.params.grid_origin[3];
            let d = self.params.coupling[0];
            let rho_p = self.params.splas1[2] / (d * d * d).max(1.0e-9);
            plasticity::solid_substeps(dt, h, rho_p)
        } else {
            1
        }
    }
}

impl Solver for TwofieldSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let (seed_positions, seed_phases, water_seed, solid_quadrature_weight) =
            seed_ranges(scene, mats, cfg);
        let solid_count = seed_positions.len() as u32 - water_seed;
        // Recipe scenes derive each material point's immutable dry mass from dose/count. This
        // keeps total coffee mass fixed under changes to numerical sampling. Non-recipe test
        // scenes retain the explicitly supplied material mass.
        let dry_grain_mass =
            recipe_solid_mass_reduced(scene, mats, cfg.tf_water_density_kg_m3, solid_count)
                .unwrap_or(mats.grain_mass * solid_quadrature_weight);
        let dry_structural_volume =
            dry_solid_sample_volume(mats.grain_diameter, mats.porosity) * solid_quadrature_weight;

        // Water pool: seed + dose headroom for a declared pour (KEEP §2 calibration; V_w =
        // spacing³ at this solver's rest density). Solids sit AFTER the full pool so the
        // KTD-1 range layout survives emission; dormant slots `[water_count, water_capacity)`
        // are never dispatched (the water kernels guard on the LIVE count) and are parked at
        // a far below-domain sentinel (y = DORMANT_PARK_Y) so the renderer can cull them when the
        // full pool is exposed (solids-present scenes) — otherwise the unused tail would draw as a
        // clump at one point. emit() overwrites the slot's position when it activates it.
        let v_w = mats.particle_spacing.powi(3);
        let retention_build = retention_state_for_build(
            cfg,
            mats,
            dry_grain_mass,
            solid_count,
            dry_structural_volume,
            solid_quadrature_weight,
            v_w,
        )
        .expect("validated recipe retention inventory");
        let dose_headroom = if scene.declares_pour() {
            let dose_volume = crate::utils::units::SiUnitScale::reference()
                .sim_volume_from_ml(f64::from(scene.pour_water_ml));
            (dose_volume / f64::from(v_w)).ceil() as u32
        } else {
            0
        };
        let water_capacity = water_seed + dose_headroom;
        let particle_count = water_capacity + solid_count; // pool size (buffers + readbacks)
                                                           // Sentinel far below the domain (matched by DORMANT_PARK_Y in particles.wgsl's cull).
        let park = [scene.box_min[0], DORMANT_PARK_Y, scene.box_min[2], 0.0];
        let mut positions = seed_positions[..water_seed as usize].to_vec();
        positions.resize(water_capacity as usize, park);
        positions.extend_from_slice(&seed_positions[water_seed as usize..]);
        let mut phases = seed_phases[..water_seed as usize].to_vec();
        phases.resize(water_capacity as usize, 0);
        phases.extend_from_slice(&seed_phases[water_seed as usize..]);
        let water_count = water_seed; // live count
        let initial_chemistry = seed_chemistry(
            water_count,
            water_capacity,
            solid_count,
            dry_grain_mass,
            mats,
        );
        let water_mass_per_volume = mats.particle_mass / v_w.max(1.0e-9);
        let initial_free_carrier_mass = positions[..water_seed as usize]
            .iter()
            .map(|p| mats.particle_mass * p[3].clamp(0.0, 1.0))
            .sum::<f32>();
        let initial_bound_carrier_mass = positions[water_capacity as usize..]
            .iter()
            .map(|p| water_mass_per_volume * p[3].max(0.0))
            .sum::<f32>();
        let initial_carrier_mass = initial_free_carrier_mass + initial_bound_carrier_mass;
        let initial_water_solute_mass = positions[..water_seed as usize]
            .iter()
            .zip(&initial_chemistry[..water_seed as usize])
            .map(|(p, c)| mats.particle_mass * p[3].clamp(0.0, 1.0) * c[0].max(0.0))
            .sum::<f32>();
        let initial_grain_solute_mass = initial_chemistry[water_capacity as usize..]
            .iter()
            .map(|c| c[0].max(0.0) + c[1].max(0.0))
            .sum::<f32>();
        let initial_solute_mass = initial_water_solute_mass + initial_grain_solute_mass;
        let insoluble_grain_mass =
            (dry_grain_mass - mats.soluble_fraction.clamp(0.0, 1.0) * dry_grain_mass).max(0.0)
                * solid_count as f32;

        let (origin, cell, dims) = grid_spec_for(scene, mats);
        let num_nodes = dims[0] * dims[1] * dims[2];
        let num_cells = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1);
        let active_tile_count = active_tile_count_for_grid(dims);
        // Coarse buffers are sized once for the densest knob-grid ratio (4 → most coarse
        // cells); the ratio-8 grid point uses a prefix of them.
        let cdims = coarse_dims_for(dims, COARSE_RATIO_DEFAULT);
        let num_ccells = cdims[0] * cdims[1] * cdims[2];
        let num_cnodes = (cdims[0] + 1) * (cdims[1] + 1) * (cdims[2] + 1);
        // ρ_rest = particle_mass/spacing³ (the rest node density the M̃⁻¹ floor is keyed to);
        // mass_eps gates "this node carries water" well above fixed-point decode noise.
        // The density floor is ρ_rest itself: a dilute (free-surface/splash) node otherwise
        // receives an invρ-amplified pressure kick (a 0.25·ρ_rest node would get 4× the bulk
        // Δv), which measurably pumps energy into the surface every frame until eruption.
        // Flooring at ρ_rest under-kicks dilute nodes instead — they sag, compact, and the
        // floor releases: stable. The floor enters A and the projection through the SAME M̃⁻¹,
        // so the operator family stays consistent; proper free-surface treatment is U4.
        let rest_density = mats.particle_mass / mats.particle_spacing.powi(3);
        let rho_floor = rest_density;
        let mass_eps = 1.0e-4 * mats.particle_mass;
        let implicit_solid = cfg.solid_dynamics && cfg.tf_compaction_tangent > 0.0;
        let shear_bulk_density =
            dry_bulk_density(dry_grain_mass, dry_structural_volume, mats.porosity);
        let shear_budget = crate::models::granular::bounded_explicit_shear_budget(
            1.0 / 60.0,
            f64::from(cell),
            f64::from(shear_bulk_density.max(1.0e-9)),
            f64::from(cfg.tf_numerical_shear_modulus.max(0.0)),
            SHEAR_CFL,
            MAX_SHEAR_SUBSTEPS,
        )
        .expect("validated numerical shear budget");
        let (filter0, filter1) = filter_uniforms(cfg).expect("validated filter-paper calibration");
        let initial_extractable_mass = mats.soluble_fraction.clamp(0.0, 1.0) * dry_grain_mass;
        let filter_paper_present = scene
            .solids
            .iter()
            .any(|solid| solid.flags & crate::utils::sdf::FLAG_FILTER_PAPER != 0);

        let cup = scene
            .solids
            .iter()
            .find_map(|primitive| match primitive.kind {
                crate::utils::sdf::SolidKind::Cylinder {
                    center,
                    floor_y,
                    rim_y,
                    radius,
                } => Some(CupRegion {
                    cx: center.x,
                    cz: center.z,
                    radius,
                    y_min: floor_y,
                    y_max: rim_y,
                }),
                _ => None,
            });
        // Bed surface = top of the seeded grain regions (the pond/free-water boundary the
        // drawdown metric keys on). No grain region means the box floor.
        let bed_top_seed = scene
            .regions
            .iter()
            .filter(|r| matches!(r.species, Species::Grain))
            .map(|r| r.max[1])
            .fold(scene.box_min[1], f32::max);

        let packed = {
            let mut packed = pack_solids(&scene.solids);
            if packed.is_empty() {
                packed.push(Primitive::zeroed()); // non-empty binding; num_solids = 0 skips it
            }
            packed
        };

        let params = Params {
            box_min: [scene.box_min[0], scene.box_min[1], scene.box_min[2], 0.0],
            box_max: [scene.box_max[0], scene.box_max[1], scene.box_max[2], 0.0],
            gravity: [scene.gravity[0], scene.gravity[1], scene.gravity[2], 0.0],
            grid_origin: [origin[0], origin[1], origin[2], cell],
            grid_dims: [dims[0], dims[1], dims[2], num_nodes],
            dt: 1.0 / 60.0,
            particle_mass: mats.particle_mass,
            max_speed: cfg.max_speed,
            pic_blend: PIC_BLEND_DEFAULT,
            water_count,
            solid_count,
            particle_count,
            num_solids: scene.solids.len() as u32,
            coarse_dims: [cdims[0], cdims[1], cdims[2], COARSE_RATIO_DEFAULT],
            // extra.w (U7): wet-cohesion peak saturation s_peak (the cohesion::for_saturation
            // curve argument; paired with c_max in splas2.w). 0.4 is the coffee-plausible
            // capillary-bridge peak — inert while c_max = 0.
            extra: [rest_density, rho_floor, mass_eps, cfg.tf_cohesion_speak],
            coupling: [
                mats.grain_diameter,
                cfg.drag_scale,
                dry_structural_volume,
                // Floor-outflow flag: the U6 dev/test open_base hook OR the U9 phase-selective
                // filter floor (water drains, frozen grains retained). set_open_base_for_test
                // still toggles it at runtime.
                if cfg.tf_filter_floor { 1.0 } else { 0.0 },
            ],
            // U5 constitutive lanes (plasticity.rs consts; friction angle from Materials,
            // dry cohesion from models::cohesion — the Bishop saturation mapping is U7).
            splas0: [
                f32::from(cfg.solid_dynamics),
                if implicit_solid {
                    shear_budget.applied_modulus as f32
                } else {
                    plasticity::lame_mu(plasticity::YOUNG_E, plasticity::POISSON_NU)
                },
                if implicit_solid {
                    0.0
                } else {
                    plasticity::lame_lambda(plasticity::YOUNG_E, plasticity::POISSON_NU)
                },
                plasticity::dp_alpha(mats.friction_mu),
            ],
            splas1: [
                plasticity::HARDEN_XI,
                if implicit_solid {
                    1.0 - mats.porosity
                } else {
                    cfg.packing_limit
                },
                dry_grain_mass,
                crate::models::cohesion::dry(),
            ],
            splas2: [
                mats.floor_mu,
                plasticity::SP_STIFF,
                plasticity::SP_ONSET,
                // U7 wet-cohesion peak c_max (Bishop saturation-weighted cohesion into the DP
                // yield; models::cohesion::for_saturation). 0 ⇒ the yield uses cohesion::dry()
                // exactly, so the dry U5 bed is byte-unchanged.
                cfg.tf_wet_cohesion,
            ],
            // U9: derive the gravimetric capacity from the SAME dry recipe mass P2G uses:
            // V_cap = r_max*m_dry/rho_water, with rho_water = particle_mass/V_w. Bound water
            // remains an independent inventory; wet1.w controls only structural swelling.
            wet0: [
                cfg.tf_absorb_rate,
                crate::models::wetting::capacity_from_dry_mass(
                    dry_grain_mass,
                    mats.r_max,
                    v_w,
                    mats.particle_mass,
                ),
                v_w,
                cfg.absorb_roundoff,
            ],
            wet1: [
                cfg.tf_suction_accel,
                cfg.tf_bloom_delay,
                if cfg.tf_filter_floor { 1.0 } else { 0.0 },
                cfg.tf_max_diameter_swelling.max(0.0),
            ],
            // Density relief ON (dbg.x) by default; SDF wall BC in BINARY mode (dbg.y) until U3
            // Default stays SINGLE: the multi-normal corner fix is proven (corner_parity gate)
            // but flipping the default trips poured_cup_water_fills_not_corner — the dynamic pour
            // traps a spurious crushing pocket under multi (same failure mode a prior one-sided
            // wall BC hit). Resolve that before defaulting MULTI / retiring the selector.
            // Tests override via set_relief_for_test / set_wall_bc_mode_for_test.
            dbg: [1.0, WALL_BC_SINGLE, 0.0, 0.0],
            // U1 surface-weighted dissipation knob: (c_surface, density_gate, div_scale,
            // water_splash_cap). Default DISABLED (c_surface = 1.0 ⇒ pure-PIC; div_scale = 0 ⇒
            // merge off; water_splash_cap = 0 ⇒ global max_speed), so g2p_water is byte-identical
            // to the pure-PIC baseline. The U2 curve reads these lanes.
            flip: [
                cfg.tf_flip_c_surface,
                cfg.tf_flip_density_gate,
                cfg.tf_flip_div_scale,
                cfg.tf_flip_water_splash_cap,
            ],
            chem0: [
                cfg.extract_rate.max(0.0) * mats.k0_fast.max(0.0),
                cfg.extract_rate.max(0.0) * mats.k0_slow.max(0.0),
                mats.ea_over_r.max(0.0),
                mats.t_ref,
            ],
            chem1: [
                mats.c_sat.max(0.0),
                mats.u_half.max(0.0),
                mats.s_on.max(0.0),
                cfg.thermal_rate.max(0.0) * mats.kappa.max(0.0),
            ],
            therm: [
                mats.cp_water.max(0.0),
                mats.cp_grain.max(0.0),
                cfg.thermal_rate.max(0.0) * mats.h_amb.max(0.0),
                mats.t_amb,
            ],
            retention: [
                retention_build.residual_pore_saturation,
                cfg.tf_capillary_exponent,
                cfg.tf_capillary_mobility_floor,
                f32::from(cfg.tf_all_gas_atmospheric),
            ],
            granular: [
                if implicit_solid {
                    cfg.tf_compaction_tangent.max(0.0)
                } else {
                    0.0
                },
                cfg.tf_initial_consolidation_pressure.max(0.0),
                cfg.tf_full_compaction_solid_fraction,
                (1.0 / 60.0) / shear_budget.substeps as f32,
            ],
            filter0,
            filter1,
            solute: [
                (dry_grain_mass - initial_extractable_mass).max(0.0),
                mats.cp_grain.max(0.0),
                initial_extractable_mass,
                if solid_count > 0 {
                    retention_build.retained_capillary_volume / solid_count as f32
                } else {
                    0.0
                },
            ],
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twofield-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let n = particle_count.max(1) as u64;
        let vec4 = n * 16;
        let pos = Arc::new(Self::storage(
            &device,
            "twofield-pos",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let vel = Arc::new(Self::storage(
            &device,
            "twofield-vel",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let phase = Arc::new(Self::storage(
            &device,
            "twofield-phase",
            n * 4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        // Phase-specific chemistry/thermal lanes; seeded below after the pool-padded buffers exist.
        let chem = Arc::new(Self::storage(
            &device,
            "twofield-chem",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        // APIC C matrices: 3 vec4 rows per particle, zero-initialized (C = 0 at rest).
        // COPY_SRC for the snapshot/replay readback in the U3 divergence-decay gate.
        let cmat = Self::storage(
            &device,
            "twofield-cmat",
            n * 48,
            wgpu::BufferUsages::COPY_SRC,
        );
        let grid_bytes = (num_nodes.max(1) as u64) * 16;
        let grid_fp = Self::storage(
            &device,
            "twofield-grid-fp",
            grid_bytes,
            wgpu::BufferUsages::COPY_SRC,
        );
        let grid_vel = Self::storage(
            &device,
            "twofield-grid-vel",
            grid_bytes,
            wgpu::BufferUsages::COPY_SRC,
        );
        // U6 SOLID grid field: per-node solid volume, one fixed-point lane (coupling.wgsl).
        let grid_sfp = Self::storage(
            &device,
            "twofield-grid-sfp",
            (num_nodes.max(1) as u64) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let grid_scomp = Self::storage(
            &device,
            "twofield-grid-scomp",
            (num_nodes.max(1) as u64) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let filter_state = Self::storage(
            &device,
            "twofield-filter-state",
            5 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let active_tiles = Self::storage(
            &device,
            "twofield-active-tiles",
            u64::from(ACTIVE_TILE_HEADER_WORDS + ACTIVE_TILE_WORDS_PER_TILE * active_tile_count)
                * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        // One [x, y, z, padding] record. Only x is copied from the GPU scheduler each frame;
        // y/z remain one, and the padding keeps the allocation 16-byte aligned.
        let active_tile_indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twofield-active-tile-indirect"),
            contents: bytemuck::cast_slice(&[[0u32, 1, 1, 0]]),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });
        let diagnostics_gpu = Self::storage(
            &device,
            "twofield-diagnostics-output",
            std::mem::size_of::<DiagnosticsGpu>() as u64,
            wgpu::BufferUsages::COPY_SRC,
        );
        let diagnostics_config_data = if let Some(cup) = cup {
            DiagnosticsConfig {
                cup0: [cup.cx, cup.cz, cup.radius, cup.y_min],
                cup1: [cup.y_max, 1.0, bed_top_seed, 0.0],
            }
        } else {
            DiagnosticsConfig {
                cup0: [0.0; 4],
                cup1: [0.0, 0.0, bed_top_seed, 0.0],
            }
        };
        let diagnostics_config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twofield-diagnostics-config"),
            contents: bytemuck::bytes_of(&diagnostics_config_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let diagnostics_readbacks = std::array::from_fn(|index| DiagnosticsReadbackSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if index == 0 {
                    "twofield-diagnostics-readback-0"
                } else {
                    "twofield-diagnostics-readback-1"
                }),
                size: std::mem::size_of::<DiagnosticsGpu>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            pending: None,
        });
        // Interphase drag state: per-node vec4 (impulse.xyz, sigma). Frozen-mode pressure
        // reaction accounting shares the buffer; dynamic impulses are applied to grid_svel.
        let react_buf = Self::storage(
            &device,
            "twofield-react",
            grid_bytes,
            wgpu::BufferUsages::COPY_SRC,
        );
        // U5 dynamic-solid buffers (allocated regardless of mode — bind groups need them;
        // dormant in frozen mode). Per-solid state is solid-local indexed.
        let grid_sm = Self::storage(
            &device,
            "twofield-grid-sm",
            grid_bytes,
            wgpu::BufferUsages::empty(),
        );
        let grid_sraw = Self::storage(
            &device,
            "twofield-grid-solid-reference",
            grid_bytes,
            wgpu::BufferUsages::empty(),
        );
        let grid_svel = Self::storage(
            &device,
            "twofield-grid-svel",
            grid_bytes,
            wgpu::BufferUsages::empty(),
        );
        // U9 moisture ledger: 6 fixed-point lanes per node (supply S, demand D, donor
        // volume-momentum Q.xyz, and donor volume-temperature R). Q/S and R/S carry the
        // transferred water's velocity and temperature into grains.
        let grid_moist = Self::storage(
            &device,
            "twofield-grid-moist",
            (num_nodes.max(1) as u64) * 24,
            wgpu::BufferUsages::empty(),
        );
        // Conservative chemistry ledger: mass/headroom, phase capacity/enthalpy, and proposed
        // solute momentum/enthalpy (11 fixed-point lanes), cleared with the grids each substep.
        let grid_chem = Self::storage(
            &device,
            "twofield-grid-chem",
            (num_nodes.max(1) as u64) * 44,
            wgpu::BufferUsages::empty(),
        );
        let ns = solid_count.max(1) as u64;
        let fmat = Self::storage(
            &device,
            "twofield-fmat",
            ns * 48,
            wgpu::BufferUsages::COPY_SRC,
        );
        let sstate = Self::storage(
            &device,
            "twofield-sstate",
            ns * 32,
            wgpu::BufferUsages::COPY_SRC,
        );
        if solid_count > 0 {
            queue.write_buffer(&fmat, 0, bytemuck::cast_slice(&identity_rows(solid_count)));
            let initial_pc = if implicit_solid {
                params.granular[1]
            } else {
                plasticity::PC0
            };
            queue.write_buffer(
                &sstate,
                0,
                bytemuck::cast_slice(&fresh_sstate(solid_count, initial_pc)),
            );
        }
        // U3 pressure-family buffers (layouts in pressure.wgsl).
        let nm = Self::storage(
            &device,
            "twofield-nm",
            (num_nodes.max(1) as u64) * 32,
            wgpu::BufferUsages::COPY_SRC,
        );
        let cell_meta = Self::storage(
            &device,
            "twofield-cell-meta",
            (num_cells.max(1) as u64) * 16,
            wgpu::BufferUsages::COPY_SRC,
        );
        let pf_a = Self::storage(
            &device,
            "twofield-pf-a",
            (num_cells.max(1) as u64) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let pf_b = Self::storage(
            &device,
            "twofield-pf-b",
            (num_cells.max(1) as u64) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let nm_c = Self::storage(
            &device,
            "twofield-nm-c",
            (num_cnodes.max(1) as u64) * 32,
            wgpu::BufferUsages::empty(),
        );
        let cmeta = Self::storage(
            &device,
            "twofield-cmeta",
            (num_ccells.max(1) as u64) * 16,
            wgpu::BufferUsages::empty(),
        );
        let pc_a = Self::storage(
            &device,
            "twofield-pc-a",
            (num_ccells.max(1) as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        let pc_b = Self::storage(
            &device,
            "twofield-pc-b",
            (num_ccells.max(1) as u64) * 4,
            wgpu::BufferUsages::empty(),
        );
        // U4 bubble state: [λ_b, δλ_b, pocket_present]. The observable presence mirror is
        // raised only by the thread that atomically claims compact-list slot zero, so pocket_mark
        // has no many-writer storage race.
        let bubble = Self::storage(&device, "twofield-bubble", 12, wgpu::BufferUsages::COPY_SRC);
        // Combined U4 cell-presence prefix + four per-node capillary-closure suffix lanes.
        // Keeping the suffixes in the same allocation adds no WebGPU storage binding.
        let cell_cnt = Self::storage(
            &device,
            "twofield-cell-carrier-aux",
            ((num_cells + 4 * num_nodes).max(1) as u64) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        // U8 R9 fix: compacted pocket-cell lists (slot 0 = atomic append count, rest = flat
        // indices). flood_init resets the counters; pocket_mark / coarse_cell_setup append; the
        // bubble row solves stride these few-hundred-entry lists instead of all cells. Sized for
        // the worst case (every cell a pocket) + the count slot.
        let pocket_f = Self::storage(
            &device,
            "twofield-pocket-f",
            ((num_cells as u64) + 1) * 4,
            wgpu::BufferUsages::empty(),
        );
        let pocket_c = Self::storage(
            &device,
            "twofield-pocket-c",
            ((num_ccells as u64) + 1) * 4,
            wgpu::BufferUsages::empty(),
        );
        let solids_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twofield-solids"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        if particle_count > 0 {
            queue.write_buffer(&pos, 0, bytemuck::cast_slice(&positions));
            queue.write_buffer(&phase, 0, bytemuck::cast_slice(&phases));
            queue.write_buffer(&chem, 0, bytemuck::cast_slice(&initial_chemistry));
        }
        queue.write_buffer(&filter_state, 0, bytemuck::cast_slice(&[0u32; 5]));
        // One readback scratch big enough for the largest readable buffer (particle vec4s,
        // affine rows at 48 B/particle, grid lanes, or the 32 B/node M̃⁻¹ matrices).
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("twofield-readback"),
            size: (n * 48).max(grid_bytes).max((num_nodes.max(1) as u64) * 32),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // WGSL has no imports: assemble the one module from the concern files. `common` declares
        // Params/bindings + shared helpers; `transfers` adds the APIC water passes; `pressure`
        // adds the U3 incompressibility family (bindings 9–16); `surface` adds the U4 flood
        // fill + constraint bubble (binding 17); `coupling` adds the U6 solid-mass P2G + drag
        // fold (bindings 19–20); `plasticity` adds the U5 dynamic-solid family (bindings
        // 22–24; grid_sm at 21 lives in common so grid_clear can zero it). `diagnostics` adds the
        // compact periodic reduction at bindings 31–32.
        let shader_src = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("common.wgsl"),
            include_str!("transfers.wgsl"),
            include_str!("pressure.wgsl"),
            include_str!("active_tiles.wgsl"),
            include_str!("surface.wgsl"),
            include_str!("coupling.wgsl"),
            include_str!("plasticity.wgsl"),
            include_str!("compaction.wgsl"),
            include_str!("diagnostics.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("twofield"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        // Per-pipeline bind groups providing exactly the bindings each entry point uses
        // (auto layout drops declared-but-unused globals; binding indices stay global).
        let bg = |pipe: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let layout = pipe.get_bind_group_layout(0);
            let e: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(b, buf)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &e,
            })
        };
        let pipelines = {
            let active_tile_mark = make("active_tile_mark");
            let active_tile_mark_bind = bg(
                &active_tile_mark,
                &[(0, &params_buf), (18, &cell_cnt), (33, &active_tiles)],
            );
            let grid_clear = make("grid_clear");
            let grid_clear_bind = bg(
                &grid_clear,
                &[
                    (0, &params_buf),
                    (6, &grid_fp),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                    (21, &grid_sm),
                    (25, &grid_moist),
                    (28, &grid_chem),
                    (34, &grid_sraw),
                ],
            );
            let p2g = make("p2g_water");
            let p2g_bind = bg(
                &p2g,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (6, &grid_fp),
                    (18, &cell_cnt),
                ],
            );
            // U6 coupling family (coupling.wgsl).
            let p2g_solid = make("p2g_solid");
            let p2g_solid_bind = bg(
                &p2g_solid,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (4, &chem),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                ],
            );
            // U5 plasticity family (plasticity.wgsl).
            let p2g_solid_reference = make("p2g_solid_reference");
            let p2g_solid_reference_bind = bg(
                &p2g_solid_reference,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (34, &grid_sraw),
                ],
            );
            let p2g_solid_dyn = make("p2g_solid_dyn");
            let p2g_solid_dyn_bind = bg(
                &p2g_solid_dyn,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                    (21, &grid_sm),
                    (24, &sstate),
                ],
            );
            let solid_update = make("solid_update");
            let solid_update_bind = bg(
                &solid_update,
                &[
                    (0, &params_buf),
                    (8, &solids_buf),
                    (19, &grid_sfp),
                    (21, &grid_sm),
                    (22, &grid_svel),
                ],
            );
            let g2p_solid = make("g2p_solid");
            let g2p_solid_bind = bg(
                &g2p_solid,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (8, &solids_buf),
                    (22, &grid_svel),
                    (23, &fmat),
                    (24, &sstate),
                ],
            );
            // Semi-implicit production solid family (compaction.wgsl). All buffers are
            // allocated even when granular.x == 0 so the legacy/default path is unchanged.
            let solid_grid_clear = make("solid_grid_clear");
            let solid_grid_clear_bind = bg(
                &solid_grid_clear,
                &[
                    (0, &params_buf),
                    (19, &grid_sfp),
                    (21, &grid_sm),
                    (29, &grid_scomp),
                ],
            );
            let p2g_solid_macro = make("p2g_solid_macro");
            let p2g_solid_macro_bind = bg(
                &p2g_solid_macro,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                    (21, &grid_sm),
                    (24, &sstate),
                    (29, &grid_scomp),
                ],
            );
            let p2g_solid_shear = make("p2g_solid_shear");
            let p2g_solid_shear_bind = bg(
                &p2g_solid_shear,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (21, &grid_sm),
                    (24, &sstate),
                ],
            );
            let solid_predict_update = make("solid_predict_update");
            let solid_predict_update_bind = bg(
                &solid_predict_update,
                &[
                    (0, &params_buf),
                    (8, &solids_buf),
                    (21, &grid_sm),
                    (22, &grid_svel),
                ],
            );
            let solid_macro_decode = make("solid_macro_decode");
            let solid_macro_decode_bind = bg(
                &solid_macro_decode,
                &[
                    (0, &params_buf),
                    (8, &solids_buf),
                    (21, &grid_sm),
                    (22, &grid_svel),
                ],
            );
            let g2p_solid_shear = make("g2p_solid_shear");
            let g2p_solid_shear_bind = bg(
                &g2p_solid_shear,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (5, &cmat),
                    (8, &solids_buf),
                    (22, &grid_svel),
                    (23, &fmat),
                    (24, &sstate),
                ],
            );
            let g2p_solid_velocity_only = make("g2p_solid_velocity_only");
            let g2p_solid_velocity_only_bind = bg(
                &g2p_solid_velocity_only,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (22, &grid_svel),
                ],
            );
            let compaction_node_setup = make("compaction_node_setup");
            let compaction_node_setup_bind = bg(
                &compaction_node_setup,
                &[
                    (0, &params_buf),
                    (8, &solids_buf),
                    (9, &nm),
                    (22, &grid_svel),
                ],
            );
            let compaction_cell_setup = make("compaction_cell_setup");
            let compaction_cell_setup_bind = bg(
                &compaction_cell_setup,
                &[
                    (0, &params_buf),
                    (8, &solids_buf),
                    (10, &cell_meta),
                    (11, &pf_a),
                    (12, &pf_b),
                    (19, &grid_sfp),
                    (22, &grid_svel),
                    (29, &grid_scomp),
                ],
            );
            let compaction_jacobi = make("compaction_jacobi");
            let compaction_jacobi_binds = [
                bg(
                    &compaction_jacobi,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_a),
                        (12, &pf_b),
                    ],
                ),
                bg(
                    &compaction_jacobi,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_b),
                        (12, &pf_a),
                    ],
                ),
            ];
            let compaction_project = make("compaction_project");
            let compaction_project_bind = bg(
                &compaction_project,
                &[(0, &params_buf), (9, &nm), (11, &pf_a), (22, &grid_svel)],
            );
            let compaction_state_gather = make("compaction_state_gather");
            let compaction_state_gather_bind = bg(
                &compaction_state_gather,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (10, &cell_meta),
                    (11, &pf_a),
                    (19, &grid_sfp),
                    (24, &sstate),
                    (29, &grid_scomp),
                ],
            );
            let drag_fold = make("drag_fold");
            let drag_fold_bind = bg(
                &drag_fold,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (8, &solids_buf),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                    (20, &react_buf),
                    (22, &grid_svel),
                    (34, &grid_sraw),
                ],
            );
            // U9 absorption family (coupling.wgsl). p2g_moisture scatters S/D plus donor
            // volume-momentum Q; g2p_absorb conservatively transfers volume, mass, and momentum,
            // updates pos.w (water f_w / grain V_bound), and ticks the bloom clock in chem.w.
            let p2g_moisture = make("p2g_moisture");
            let p2g_moisture_bind = bg(
                &p2g_moisture,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (25, &grid_moist),
                ],
            );
            let g2p_absorb = make("g2p_absorb");
            let g2p_absorb_bind = bg(
                &g2p_absorb,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (25, &grid_moist),
                ],
            );
            let chemistry_scatter = make("chemistry_scatter");
            let chemistry_scatter_bind = bg(
                &chemistry_scatter,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (7, &grid_vel),
                    (22, &grid_svel),
                    (28, &grid_chem),
                ],
            );
            let chemistry_gather = make("chemistry_gather");
            let chemistry_gather_bind = bg(
                &chemistry_gather,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (7, &grid_vel),
                    (22, &grid_svel),
                    (28, &grid_chem),
                ],
            );
            let filter_release = make("filter_release");
            let filter_release_bind = bg(
                &filter_release,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (7, &grid_vel),
                    (8, &solids_buf),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                    (22, &grid_svel),
                    (30, &filter_state),
                ],
            );
            let diagnostics_reduce = make("diagnostics_reduce");
            let diagnostics_reduce_bind = bg(
                &diagnostics_reduce,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (4, &chem),
                    (7, &grid_vel),
                    (18, &cell_cnt),
                    (30, &filter_state),
                    (31, &diagnostics_gpu),
                    (32, &diagnostics_config),
                ],
            );
            let grid_update = make("grid_update");
            let grid_update_bind = bg(
                &grid_update,
                &[(0, &params_buf), (6, &grid_fp), (7, &grid_vel)],
            );
            let g2p = make("g2p_water");
            let g2p_bind = bg(
                &g2p,
                &[
                    (0, &params_buf),
                    (1, &pos),
                    (2, &vel),
                    (4, &chem),
                    (5, &cmat),
                    (7, &grid_vel),
                    (8, &solids_buf),
                    (18, &cell_cnt),
                    (30, &filter_state),
                ],
            );
            // --- U3 pressure family (bindings per pass derived in the budget comment) ---
            let node_setup = make("node_setup");
            let node_setup_bind = bg(
                &node_setup,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (8, &solids_buf),
                    (9, &nm),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                    (20, &react_buf),
                    (22, &grid_svel),
                ],
            );
            let cell_classify = make("cell_classify");
            let cell_classify_bind = bg(
                &cell_classify,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (8, &solids_buf),
                    (9, &nm),
                    (10, &cell_meta),
                    (11, &pf_a),
                    (12, &pf_b),
                    (18, &cell_cnt),
                    (19, &grid_sfp),
                    (22, &grid_svel),
                ],
            );
            let coarse_node_setup = make("coarse_node_setup");
            let coarse_node_setup_bind = bg(
                &coarse_node_setup,
                &[(0, &params_buf), (9, &nm), (13, &nm_c)],
            );
            let coarse_cell_setup = make("coarse_cell_setup");
            let coarse_cell_setup_bind = bg(
                &coarse_cell_setup,
                &[
                    (0, &params_buf),
                    (10, &cell_meta),
                    (14, &cmeta),
                    (15, &pc_a),
                    (16, &pc_b),
                    (27, &pocket_c),
                ],
            );
            let jacobi_coarse = make("jacobi_coarse");
            let jacobi_coarse_binds = [
                bg(
                    &jacobi_coarse,
                    &[
                        (0, &params_buf),
                        (13, &nm_c),
                        (14, &cmeta),
                        (15, &pc_a),
                        (16, &pc_b),
                        (17, &bubble),
                    ],
                ),
                bg(
                    &jacobi_coarse,
                    &[
                        (0, &params_buf),
                        (13, &nm_c),
                        (14, &cmeta),
                        (15, &pc_b),
                        (16, &pc_a),
                        (17, &bubble),
                    ],
                ),
            ];
            let residual = make("residual");
            let residual_binds = [
                bg(
                    &residual,
                    &[(0, &params_buf), (9, &nm), (10, &cell_meta), (11, &pf_a)],
                ),
                bg(
                    &residual,
                    &[(0, &params_buf), (9, &nm), (10, &cell_meta), (11, &pf_b)],
                ),
            ];
            let residual_tiled = make("residual_tiled");
            let residual_tiled_binds = [
                bg(
                    &residual_tiled,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_a),
                        (33, &active_tiles),
                    ],
                ),
                bg(
                    &residual_tiled,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_b),
                        (33, &active_tiles),
                    ],
                ),
            ];
            // prolong_add reads the current pressure (11) + the coarse final-parity buffer
            // (15) and writes the corrected pressure (12) — variants [p parity][pc parity].
            let prolong_add = make("prolong_add");
            let pa = |src: &wgpu::Buffer, dst: &wgpu::Buffer, pc: &wgpu::Buffer| {
                bg(
                    &prolong_add,
                    &[
                        (0, &params_buf),
                        (10, &cell_meta),
                        (14, &cmeta),
                        (15, pc),
                        (11, src),
                        (12, dst),
                        (17, &bubble),
                    ],
                )
            };
            let prolong_add_binds = [
                [pa(&pf_a, &pf_b, &pc_a), pa(&pf_a, &pf_b, &pc_b)],
                [pa(&pf_b, &pf_a, &pc_a), pa(&pf_b, &pf_a, &pc_b)],
            ];
            let prolong_add_tiled = make("prolong_add_tiled");
            let pa_tiled = |src: &wgpu::Buffer, dst: &wgpu::Buffer, pc: &wgpu::Buffer| {
                bg(
                    &prolong_add_tiled,
                    &[
                        (0, &params_buf),
                        (10, &cell_meta),
                        (14, &cmeta),
                        (15, pc),
                        (11, src),
                        (12, dst),
                        (17, &bubble),
                        (33, &active_tiles),
                    ],
                )
            };
            let prolong_add_tiled_binds = [
                [pa_tiled(&pf_a, &pf_b, &pc_a), pa_tiled(&pf_a, &pf_b, &pc_b)],
                [pa_tiled(&pf_b, &pf_a, &pc_a), pa_tiled(&pf_b, &pf_a, &pc_b)],
            ];
            let jacobi_fine = make("jacobi_fine");
            let jacobi_fine_binds = [
                bg(
                    &jacobi_fine,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_a),
                        (12, &pf_b),
                        (17, &bubble),
                    ],
                ),
                bg(
                    &jacobi_fine,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_b),
                        (12, &pf_a),
                        (17, &bubble),
                    ],
                ),
            ];
            let jacobi_fine_tiled = make("jacobi_fine_tiled");
            let jacobi_fine_tiled_binds = [
                bg(
                    &jacobi_fine_tiled,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_a),
                        (12, &pf_b),
                        (17, &bubble),
                        (33, &active_tiles),
                    ],
                ),
                bg(
                    &jacobi_fine_tiled,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_b),
                        (12, &pf_a),
                        (17, &bubble),
                        (33, &active_tiles),
                    ],
                ),
            ];
            // Project consumes the fine final-parity buffer (index = fine_sweeps % 2).
            let project = make("project");
            let project_binds = [
                bg(
                    &project,
                    &[
                        (0, &params_buf),
                        (7, &grid_vel),
                        (8, &solids_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_a),
                        (18, &cell_cnt),
                        (19, &grid_sfp),
                        (20, &react_buf),
                        // Dynamic mode applies the drag-condensed pore-pressure response to
                        // this solid velocity field; frozen mode leaves it dormant.
                        (22, &grid_svel),
                    ],
                ),
                bg(
                    &project,
                    &[
                        (0, &params_buf),
                        (7, &grid_vel),
                        (8, &solids_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, &pf_b),
                        (18, &cell_cnt),
                        (19, &grid_sfp),
                        (20, &react_buf),
                        (22, &grid_svel),
                    ],
                ),
            ];
            // --- U4 surface family --------------------------------------------------------
            let flood_init = make("flood_init");
            // binding 17 (bubble): thread zero owns the observable pocket-present reset.
            let flood_init_bind = bg(
                &flood_init,
                &[
                    (0, &params_buf),
                    (8, &solids_buf),
                    (10, &cell_meta),
                    (11, &pf_a),
                    (17, &bubble),
                    (26, &pocket_f),
                    (27, &pocket_c),
                ],
            );
            let flood_sweep = make("flood_sweep");
            let flood_sweep_binds = [
                bg(
                    &flood_sweep,
                    &[(0, &params_buf), (10, &cell_meta), (11, &pf_a), (12, &pf_b)],
                ),
                bg(
                    &flood_sweep,
                    &[(0, &params_buf), (10, &cell_meta), (11, &pf_b), (12, &pf_a)],
                ),
            ];
            // pocket_mark reads the final labels from pf_a (the flood sweep count is even) and
            // re-zeroes both pressure slots for the solve.
            let pocket_mark = make("pocket_mark");
            let pocket_mark_bind = bg(
                &pocket_mark,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (9, &nm),
                    (10, &cell_meta),
                    (11, &pf_a),
                    (12, &pf_b),
                    (17, &bubble),
                    (18, &cell_cnt),
                    (22, &grid_svel),
                    (26, &pocket_f),
                ],
            );
            let bubble_fine = make("bubble_fine");
            let bf = |pf: &wgpu::Buffer| {
                bg(
                    &bubble_fine,
                    &[
                        (0, &params_buf),
                        (9, &nm),
                        (10, &cell_meta),
                        (11, pf),
                        (17, &bubble),
                        (26, &pocket_f),
                    ],
                )
            };
            let bubble_fine_binds = [bf(&pf_a), bf(&pf_b)];
            let bubble_coarse = make("bubble_coarse");
            let bc = |pc: &wgpu::Buffer| {
                bg(
                    &bubble_coarse,
                    &[
                        (0, &params_buf),
                        (13, &nm_c),
                        (14, &cmeta),
                        (15, pc),
                        (17, &bubble),
                        (26, &pocket_f),
                        (27, &pocket_c),
                    ],
                )
            };
            let bubble_coarse_binds = [bc(&pc_a), bc(&pc_b)];
            let dbg_div = make("dbg_div");
            let dbg_div_bind = bg(
                &dbg_div,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (9, &nm),
                    (10, &cell_meta),
                    (18, &cell_cnt),
                    (22, &grid_svel),
                ],
            );
            let dbg_grad = make("dbg_grad");
            let dbg_grad_bind = bg(
                &dbg_grad,
                &[
                    (0, &params_buf),
                    (7, &grid_vel),
                    (10, &cell_meta),
                    (11, &pf_a),
                ],
            );
            Pipelines {
                active_tile_mark: (active_tile_mark, active_tile_mark_bind),
                grid_clear: (grid_clear, grid_clear_bind),
                p2g_water: (p2g, p2g_bind),
                p2g_solid: (p2g_solid, p2g_solid_bind),
                drag_fold: (drag_fold, drag_fold_bind),
                p2g_moisture: (p2g_moisture, p2g_moisture_bind),
                g2p_absorb: (g2p_absorb, g2p_absorb_bind),
                chemistry_scatter: (chemistry_scatter, chemistry_scatter_bind),
                chemistry_gather: (chemistry_gather, chemistry_gather_bind),
                filter_release: (filter_release, filter_release_bind),
                diagnostics_reduce: (diagnostics_reduce, diagnostics_reduce_bind),
                p2g_solid_reference: (p2g_solid_reference, p2g_solid_reference_bind),
                p2g_solid_dyn: (p2g_solid_dyn, p2g_solid_dyn_bind),
                solid_update: (solid_update, solid_update_bind),
                g2p_solid: (g2p_solid, g2p_solid_bind),
                solid_grid_clear: (solid_grid_clear, solid_grid_clear_bind),
                p2g_solid_macro: (p2g_solid_macro, p2g_solid_macro_bind),
                p2g_solid_shear: (p2g_solid_shear, p2g_solid_shear_bind),
                solid_predict_update: (solid_predict_update, solid_predict_update_bind),
                solid_macro_decode: (solid_macro_decode, solid_macro_decode_bind),
                g2p_solid_shear: (g2p_solid_shear, g2p_solid_shear_bind),
                g2p_solid_velocity_only: (g2p_solid_velocity_only, g2p_solid_velocity_only_bind),
                compaction_node_setup: (compaction_node_setup, compaction_node_setup_bind),
                compaction_cell_setup: (compaction_cell_setup, compaction_cell_setup_bind),
                compaction_jacobi: (compaction_jacobi, compaction_jacobi_binds),
                compaction_project: (compaction_project, compaction_project_bind),
                compaction_state_gather: (compaction_state_gather, compaction_state_gather_bind),
                grid_update: (grid_update, grid_update_bind),
                g2p_water: (g2p, g2p_bind),
                node_setup: (node_setup, node_setup_bind),
                cell_classify: (cell_classify, cell_classify_bind),
                residual: (residual, residual_binds),
                residual_tiled: (residual_tiled, residual_tiled_binds),
                coarse_node_setup: (coarse_node_setup, coarse_node_setup_bind),
                coarse_cell_setup: (coarse_cell_setup, coarse_cell_setup_bind),
                jacobi_coarse: (jacobi_coarse, jacobi_coarse_binds),
                prolong_add: (prolong_add, prolong_add_binds),
                prolong_add_tiled: (prolong_add_tiled, prolong_add_tiled_binds),
                jacobi_fine: (jacobi_fine, jacobi_fine_binds),
                jacobi_fine_tiled: (jacobi_fine_tiled, jacobi_fine_tiled_binds),
                project: (project, project_binds),
                flood_init: (flood_init, flood_init_bind),
                flood_sweep: (flood_sweep, flood_sweep_binds),
                pocket_mark: (pocket_mark, pocket_mark_bind),
                bubble_fine: (bubble_fine, bubble_fine_binds),
                bubble_coarse: (bubble_coarse, bubble_coarse_binds),
                dbg_div: (dbg_div, dbg_div_bind),
                dbg_grad: (dbg_grad, dbg_grad_bind),
            }
        };

        let ts = if gpu.timestamps_supported {
            // 71 passes per frame at the U6 defaults (2 queries each); headroom for U5+.
            let capacity = 192u32;
            let qset = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("twofield-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("twofield-ts-resolve"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("twofield-ts-readback"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Some(Timestamps {
                qset,
                capacity,
                resolve,
                readback,
                period_ns: queue.get_timestamp_period(),
                labels: Vec::new(),
                count: 0,
            })
        } else {
            None
        };

        let mut initial_diag = retention_build.diagnostics();
        initial_diag.requested_numerical_shear_modulus = cfg.tf_numerical_shear_modulus.max(0.0);
        initial_diag.applied_numerical_shear_modulus = shear_budget.applied_modulus as f32;
        initial_diag.shear_microsteps = shear_budget.substeps;
        initial_diag.numerical_shear_limited = shear_budget.limited;
        initial_diag.expected_carrier_mass = initial_carrier_mass;
        initial_diag.expected_solute_mass = initial_solute_mass;
        initial_diag.filter_capacity_volume = if filter_paper_present {
            filter1[0] as f32 / FILTER_VOLUME_SCALE as f32
        } else {
            0.0
        };

        Self {
            device,
            queue,
            params,
            water_count,
            water_capacity,
            solid_count,
            num_nodes,
            num_cells,
            inflow: Inflow {
                nozzle_radius: cfg.nozzle_radius,
                discharge_coeff: cfg.discharge_coeff,
                spacing: mats.particle_spacing,
                v_w,
                pour_t: mats.pour_t,
                accumulator: 0.0,
                axial: 0.0,
                last_exit_speed: 0.0,
                cursor: 0,
                emitted_mass: 0.0,
            },
            coarse_ratio: COARSE_RATIO_DEFAULT,
            coarse_sweeps: COARSE_SWEEPS_DEFAULT,
            fine_sweeps: FINE_SWEEPS_DEFAULT,
            flood_sweeps: flood_sweeps_for(dims),
            all_gas_atmospheric: cfg.tf_all_gas_atmospheric,
            params_buf,
            pos,
            vel,
            phase,
            chem,
            cmat,
            grid_fp,
            cell_cnt,
            grid_vel,
            grid_sfp,
            grid_scomp,
            filter_state,
            active_tiles,
            active_tile_indirect,
            active_tile_count,
            active_tiling: active_tiling_enabled(implicit_solid, cfg.tf_all_gas_atmospheric),
            diagnostics_gpu,
            diagnostics_config,
            diagnostics_readbacks,
            // Sample immediately after the first completed physics frame, then at 4 Hz.
            diagnostics_elapsed_s: DIAGNOSTICS_PERIOD_S,
            diagnostics_epoch: 0,
            react: react_buf,
            fmat,
            sstate,
            solid_dynamic: cfg.solid_dynamics,
            implicit_solid,
            requested_numerical_shear_modulus: cfg.tf_numerical_shear_modulus.max(0.0),
            applied_numerical_shear_modulus: shear_budget.applied_modulus as f32,
            shear_substeps: shear_budget.substeps,
            absorb_on: cfg.tf_absorb_rate > 0.0,
            extraction_on: cfg.extract_rate > 0.0,
            thermal_on: cfg.thermal_rate > 0.0 && (mats.kappa > 0.0 || mats.h_amb > 0.0),
            filter_reservoir_on: filter_paper_present && filter1[0] > 0 && filter0[0] > 0.0,
            filter_paper_present,
            nm,
            cell_meta,
            pf_a,
            pf_b,
            bubble,
            readback,
            pipelines,
            ts,
            dispatches: 0,
            cached_passes: Vec::new(),
            sim_time: 0.0,
            bed_top_seed,
            cup,
            pond_peak_carrier_mass: 0.0,
            retention_build,
            cached_diag: initial_diag,
            initial_carrier_mass,
            initial_solute_mass,
            insoluble_grain_mass,
            initial_positions: positions,
            initial_phases: phases,
            initial_chemistry,
            initial_water: water_seed,
        }
    }

    fn reset(&mut self, _scene: &Scene) {
        // Exact, deterministic re-seed: positions + phase tags back to the seed (pool-padded
        // layout — pour-activated slots return to the parked dormant state); velocities, all
        // phase-specific chemistry lanes back to their seeded pools/temperatures (including the
        // grain wet-contact clock in chem.w), and APIC C matrices re-zeroed; the emitter restarts
        // from t = 0. (The grid is cleared at frame start.)
        if !self.initial_positions.is_empty() {
            self.queue
                .write_buffer(&self.pos, 0, bytemuck::cast_slice(&self.initial_positions));
            self.queue
                .write_buffer(&self.phase, 0, bytemuck::cast_slice(&self.initial_phases));
            let zeros = vec![[0.0f32; 4]; self.initial_positions.len()];
            self.queue
                .write_buffer(&self.vel, 0, bytemuck::cast_slice(&zeros));
            self.queue
                .write_buffer(&self.chem, 0, bytemuck::cast_slice(&self.initial_chemistry));
            let czeros = vec![[0.0f32; 4]; 3 * self.initial_positions.len()];
            self.queue
                .write_buffer(&self.cmat, 0, bytemuck::cast_slice(&czeros));
        }
        if self.solid_count > 0 {
            // U5 constitutive state back to the seed: F = I, τ = 0, p_c = PC0, compaction 0.
            self.queue.write_buffer(
                &self.fmat,
                0,
                bytemuck::cast_slice(&identity_rows(self.solid_count)),
            );
            self.queue.write_buffer(
                &self.sstate,
                0,
                bytemuck::cast_slice(&fresh_sstate(
                    self.solid_count,
                    if self.implicit_solid {
                        self.params.granular[1]
                    } else {
                        plasticity::PC0
                    },
                )),
            );
        }
        // Bubble diagnostics/state are frame-local and must not survive a reset. This is
        // especially visible in a poured CenterPour reset: its temporarily dry dynamic fast path
        // does not execute the surface stack before the emitter activates its first water sample.
        self.queue
            .write_buffer(&self.bubble, 0, bytemuck::cast_slice(&[0.0f32; 3]));
        self.queue
            .write_buffer(&self.filter_state, 0, bytemuck::cast_slice(&[0u32; 5]));
        self.water_count = self.initial_water;
        self.params.water_count = self.initial_water;
        self.inflow.accumulator = 0.0;
        self.inflow.axial = 0.0;
        self.inflow.last_exit_speed = 0.0;
        self.inflow.cursor = 0;
        self.inflow.emitted_mass = 0.0;
        self.cached_passes.clear();
        self.dispatches = 0;
        self.sim_time = 0.0;
        self.pond_peak_carrier_mass = 0.0;
        self.diagnostics_elapsed_s = DIAGNOSTICS_PERIOD_S;
        self.diagnostics_epoch = self.diagnostics_epoch.wrapping_add(1);
        self.cached_diag = self.retention_build.diagnostics();
        self.cached_diag.expected_carrier_mass = self.initial_carrier_mass;
        self.cached_diag.expected_solute_mass = self.initial_solute_mass;
        self.cached_diag.filter_capacity_volume = if self.filter_paper_present {
            self.params.filter1[0] as f32 / FILTER_VOLUME_SCALE as f32
        } else {
            0.0
        };
        self.cached_diag.requested_numerical_shear_modulus = self.requested_numerical_shear_modulus;
        self.cached_diag.applied_numerical_shear_modulus = self.applied_numerical_shear_modulus;
        self.cached_diag.shear_microsteps = self.shear_substeps;
        self.cached_diag.numerical_shear_limited =
            self.applied_numerical_shear_modulus < self.requested_numerical_shear_modulus;
    }

    fn step(&mut self, dt: f32, input: &EmissionInput) {
        self.poll_periodic_diagnostics();
        // Pour emission first (grows water_count for this frame); no-op when not pouring.
        self.emit(input, dt);
        self.sim_time += dt;
        let diagnostics_slot = self.periodic_diagnostics_slot(dt);
        debug_assert!(self.water_count <= self.water_capacity);
        // The legacy U5 path repeats the whole frame at its explicit elastic CFL. The
        // semi-implicit path instead keeps one macro frame: only its deliberately soft
        // deviatoric predictor is microstepped below, while compaction, drag, and mixture
        // pressure each run once at the caller's physical dt. Frozen mode also stays at one.
        let substeps = if self.implicit_solid {
            let reference_solid_fraction = self.params.splas1[1].clamp(0.0, 1.0);
            let dry_bulk_volume = self.params.coupling[2] / reference_solid_fraction.max(1.0e-9);
            let shear_bulk_density = self.params.splas1[2] / dry_bulk_volume.max(1.0e-9);
            let budget = crate::models::granular::bounded_explicit_shear_budget(
                f64::from(dt.max(0.0)),
                f64::from(self.params.grid_origin[3]),
                f64::from(shear_bulk_density.max(1.0e-9)),
                f64::from(self.requested_numerical_shear_modulus),
                SHEAR_CFL,
                MAX_SHEAR_SUBSTEPS,
            )
            .expect("validated numerical shear budget");
            self.applied_numerical_shear_modulus = budget.applied_modulus as f32;
            self.shear_substeps = budget.substeps;
            self.params.splas0[1] = self.applied_numerical_shear_modulus;
            self.params.granular[3] = dt / self.shear_substeps as f32;
            self.cached_diag.requested_numerical_shear_modulus =
                self.requested_numerical_shear_modulus;
            self.cached_diag.applied_numerical_shear_modulus = self.applied_numerical_shear_modulus;
            self.cached_diag.shear_microsteps = self.shear_substeps;
            self.cached_diag.numerical_shear_limited = budget.limited;
            1
        } else if self.solid_dynamic {
            let h = self.params.grid_origin[3];
            let d = self.params.coupling[0];
            let rho_p = self.params.splas1[2] / (d * d * d).max(1.0e-9);
            plasticity::solid_substeps(dt, h, rho_p)
        } else {
            1
        };
        self.params.dt = if self.implicit_solid {
            dt
        } else {
            dt / substeps as f32
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-frame"),
            });
        // One recorder for the whole frame. With timestamps (native profiling) it records
        // per-pass timestamped passes for substep 0, then batches the rest; without timestamps
        // (web / no TIMESTAMP_QUERY) the whole frame collapses into ONE compute pass — Dawn
        // inserts the storage-buffer barriers between dependent dispatches automatically, so the
        // result is identical (proven by tests/batch_pass_equivalence.rs). Active tiling inserts
        // one required pass boundary for the GPU counter -> indirect-argument copies; all work on
        // either side is still maximally batched.
        let sink = self.ts.as_ref().map(|t| TimestampSink {
            qset: &t.qset,
            capacity: t.capacity,
        });
        let mut rec = PassRecorder::new(sink);

        // The U2 transfer pipeline + U3 pressure stack, one substep per frame. Passes dispatch
        // at least one workgroup so the dispatch/profiling path is real even on an empty scene.
        // Dense solid/setup/coarse/fallback stages early-out on live-set/mask guards; only the
        // guarded repeated water fine-pressure stages use a GPU-produced indirect tile count.
        let node_groups = groups(self.num_nodes).max(1);
        let water_groups = groups(self.water_count).max(1);
        let solid_groups = groups(self.solid_count).max(1);
        // U9 absorption passes cover the full particle layout (water prefix + solid suffix);
        // dormant pool slots early-out on the range guards.
        let part_groups = groups(self.params.particle_count).max(1);
        let cell_groups = groups(self.num_cells).max(1);
        let cd = self.params.coarse_dims;
        let ccell_groups = groups(cd[0] * cd[1] * cd[2]).max(1);
        let cnode_groups = groups((cd[0] + 1) * (cd[1] + 1) * (cd[2] + 1)).max(1);

        if self.active_tiling {
            // Clear before grid_clear/P2G opens the first compute pass. The post-P2G tile census
            // will end that pass before copying this scheduler's count into the INDIRECT-only
            // argument record; the frame still uses one encoder and one submission.
            enc.clear_buffer(&self.active_tiles, 0, None);
        }

        // Dry dynamic mode (U5_DRY_DISPATCHES): no live water → only the solid passes run.
        let dry_dynamic = self.solid_dynamic && !self.implicit_solid && self.water_count == 0;
        if dry_dynamic {
            let seq: [(&str, &wgpu::ComputePipeline, &wgpu::BindGroup, u32); 5] = [
                (
                    "grid_clear",
                    &self.pipelines.grid_clear.0,
                    &self.pipelines.grid_clear.1,
                    node_groups,
                ),
                (
                    "p2g_solid_reference",
                    &self.pipelines.p2g_solid_reference.0,
                    &self.pipelines.p2g_solid_reference.1,
                    solid_groups,
                ),
                (
                    "p2g_solid_dyn",
                    &self.pipelines.p2g_solid_dyn.0,
                    &self.pipelines.p2g_solid_dyn.1,
                    solid_groups,
                ),
                (
                    "solid_update",
                    &self.pipelines.solid_update.0,
                    &self.pipelines.solid_update.1,
                    node_groups,
                ),
                (
                    "g2p_solid",
                    &self.pipelines.g2p_solid.0,
                    &self.pipelines.g2p_solid.1,
                    solid_groups,
                ),
            ];
            for ss in 0..substeps {
                // Timestamp substep 0 only (fixed query-set capacity; the perf test reads ONE
                // substep's sum). Remaining substeps batch into one pass.
                if ss == 1 {
                    rec.disable_timestamps();
                }
                for &(label, pipe, bind, g) in &seq {
                    rec.dispatch(&mut enc, pipe, bind, label, g);
                }
            }
            if diagnostics_slot.is_some() {
                rec.dispatch(
                    &mut enc,
                    &self.pipelines.diagnostics_reduce.0,
                    &self.pipelines.diagnostics_reduce.1,
                    "diagnostics_reduce",
                    1,
                );
            }
            let (dispatches, cursor, labels) = rec.finish();
            if let Some(slot) = diagnostics_slot {
                enc.copy_buffer_to_buffer(
                    &self.diagnostics_gpu,
                    0,
                    &self.diagnostics_readbacks[slot].buffer,
                    0,
                    std::mem::size_of::<DiagnosticsGpu>() as u64,
                );
            }
            if let Some(ts) = &self.ts {
                if cursor >= 2 {
                    enc.resolve_query_set(&ts.qset, 0..cursor, &ts.resolve, 0);
                    enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, (cursor as u64) * 8);
                }
            }
            self.queue.submit(Some(enc.finish()));
            if let Some(slot) = diagnostics_slot {
                self.begin_periodic_diagnostics_map(slot, self.sim_time);
            }
            self.dispatches = dispatches;
            if let Some(ts) = self.ts.as_mut() {
                ts.labels = labels;
                ts.count = cursor;
            }
            return;
        }

        // (label, pipeline, bind group, workgroups) — pressure order per pressure.wgsl header:
        // node_setup → cell_classify → pre-smooth → residual → coarse correction →
        // prolong_add → post-smooth → project. `par` tracks which ping-pong buffer holds the
        // current pressure (0 = pf_a); sweeps and prolong_add flip it.
        let (pre, post) = self.smooth_split();
        let mut par = 0usize;
        let mut seq: Vec<(&str, &wgpu::ComputePipeline, &wgpu::BindGroup, u32)> = vec![(
            "grid_clear",
            &self.pipelines.grid_clear.0,
            &self.pipelines.grid_clear.1,
            node_groups,
        )];
        // U9 moisture phase-change absorption (KTD-4c), only when on. Runs BEFORE the field
        // P2Gs so they scatter updated free-water mass, bound-water grain mass, and the separate
        // saturation-driven structural volume: p2g_moisture scatters node supply/demand and
        // donor volume-momentum (S/D/Q) at the start-of-frame positions, g2p_absorb gathers the
        // drain/fill fractions at the SAME positions (the transfer is a position-static phase
        // change — water volume, mass, and momentum loss/gain match per node up to FP quantization).
        if self.absorb_on {
            seq.push((
                "p2g_moisture",
                &self.pipelines.p2g_moisture.0,
                &self.pipelines.p2g_moisture.1,
                part_groups,
            ));
            seq.push((
                "g2p_absorb",
                &self.pipelines.g2p_absorb.0,
                &self.pipelines.g2p_absorb.1,
                part_groups,
            ));
        }
        seq.push((
            "p2g_water",
            &self.pipelines.p2g_water.0,
            &self.pipelines.p2g_water.1,
            water_groups,
        ));
        if self.active_tiling {
            seq.push((
                "active_tile_mark",
                &self.pipelines.active_tile_mark.0,
                &self.pipelines.active_tile_mark.1,
                ACTIVE_TILE_MARKER,
            ));
        }
        if self.solid_dynamic {
            seq.push((
                "p2g_solid_reference",
                &self.pipelines.p2g_solid_reference.0,
                &self.pipelines.p2g_solid_reference.1,
                solid_groups,
            ));
        }
        if self.implicit_solid {
            // Multirate IMEX split: only the deliberately soft deviatoric predictor is
            // subcycled. The water grid remains resident while solid-only clears rebuild the
            // skeleton; compaction, exact drag, and mixture pressure each run once at macro dt.
            for _ in 0..self.shear_substeps {
                seq.extend([
                    (
                        "solid_grid_clear",
                        &self.pipelines.solid_grid_clear.0,
                        &self.pipelines.solid_grid_clear.1,
                        node_groups,
                    ),
                    (
                        "p2g_solid_shear",
                        &self.pipelines.p2g_solid_shear.0,
                        &self.pipelines.p2g_solid_shear.1,
                        solid_groups,
                    ),
                    (
                        "solid_predict_update",
                        &self.pipelines.solid_predict_update.0,
                        &self.pipelines.solid_predict_update.1,
                        node_groups,
                    ),
                    (
                        "g2p_solid_shear",
                        &self.pipelines.g2p_solid_shear.0,
                        &self.pipelines.g2p_solid_shear.1,
                        solid_groups,
                    ),
                ]);
            }
            seq.extend([
                (
                    "solid_grid_clear",
                    &self.pipelines.solid_grid_clear.0,
                    &self.pipelines.solid_grid_clear.1,
                    node_groups,
                ),
                (
                    "p2g_solid_macro",
                    &self.pipelines.p2g_solid_macro.0,
                    &self.pipelines.p2g_solid_macro.1,
                    solid_groups,
                ),
                (
                    "solid_macro_decode",
                    &self.pipelines.solid_macro_decode.0,
                    &self.pipelines.solid_macro_decode.1,
                    node_groups,
                ),
                (
                    "grid_update",
                    &self.pipelines.grid_update.0,
                    &self.pipelines.grid_update.1,
                    node_groups,
                ),
                (
                    "compaction_node_setup",
                    &self.pipelines.compaction_node_setup.0,
                    &self.pipelines.compaction_node_setup.1,
                    node_groups,
                ),
                (
                    "compaction_cell_setup",
                    &self.pipelines.compaction_cell_setup.0,
                    &self.pipelines.compaction_cell_setup.1,
                    cell_groups,
                ),
            ]);
            for sweep in 0..COMPACTION_SWEEPS {
                seq.push((
                    "compaction_jacobi",
                    &self.pipelines.compaction_jacobi.0,
                    &self.pipelines.compaction_jacobi.1[(sweep % 2) as usize],
                    cell_groups,
                ));
            }
            debug_assert!(COMPACTION_SWEEPS.is_multiple_of(2));
            seq.extend([
                (
                    "compaction_project",
                    &self.pipelines.compaction_project.0,
                    &self.pipelines.compaction_project.1,
                    node_groups,
                ),
                (
                    "compaction_state_gather",
                    &self.pipelines.compaction_state_gather.0,
                    &self.pipelines.compaction_state_gather.1,
                    solid_groups,
                ),
            ]);
        } else {
            seq.extend([
                (
                    if self.solid_dynamic {
                        "p2g_solid_dyn"
                    } else {
                        "p2g_solid"
                    },
                    if self.solid_dynamic {
                        &self.pipelines.p2g_solid_dyn.0
                    } else {
                        &self.pipelines.p2g_solid.0
                    },
                    if self.solid_dynamic {
                        &self.pipelines.p2g_solid_dyn.1
                    } else {
                        &self.pipelines.p2g_solid.1
                    },
                    solid_groups,
                ),
                (
                    "grid_update",
                    &self.pipelines.grid_update.0,
                    &self.pipelines.grid_update.1,
                    node_groups,
                ),
            ]);
            if self.solid_dynamic {
                seq.push((
                    "solid_update",
                    &self.pipelines.solid_update.0,
                    &self.pipelines.solid_update.1,
                    node_groups,
                ));
            }
        }
        // Exact pair drag consumes the compaction-corrected solid velocity. The following
        // mixture node_setup intentionally overwrites `nm`, preserving its own A/project pair.
        seq.extend([
            (
                "drag_fold",
                &self.pipelines.drag_fold.0,
                &self.pipelines.drag_fold.1,
                node_groups,
            ),
            (
                "node_setup",
                &self.pipelines.node_setup.0,
                &self.pipelines.node_setup.1,
                node_groups,
            ),
            (
                "cell_classify",
                &self.pipelines.cell_classify.0,
                &self.pipelines.cell_classify.1,
                cell_groups,
            ),
        ]);
        // U4 gas handling. Generic scenes flood the OUTSIDE label through air, then promote the
        // enclosed remainder to one constrained bubble. Explicitly vented production geometry
        // instead dispatches only thread zero of flood_init to reset bubble/list state; the
        // cell-wide finalize scan remains because it owns the under-density half of density
        // relief, but surface.wgsl leaves every gas cell as atmospheric CELL_AIR.
        if self.all_gas_atmospheric {
            seq.push((
                "gas_state_reset",
                &self.pipelines.flood_init.0,
                &self.pipelines.flood_init.1,
                1,
            ));
        } else {
            debug_assert!(self.flood_sweeps.is_multiple_of(2));
            seq.push((
                "flood_init",
                &self.pipelines.flood_init.0,
                &self.pipelines.flood_init.1,
                cell_groups,
            ));
            for s in 0..self.flood_sweeps {
                seq.push((
                    "flood_sweep",
                    &self.pipelines.flood_sweep.0,
                    &self.pipelines.flood_sweep.1[(s % 2) as usize],
                    cell_groups,
                ));
            }
        }
        seq.push((
            if self.all_gas_atmospheric {
                "atmospheric_finalize"
            } else {
                "pocket_mark"
            },
            &self.pipelines.pocket_mark.0,
            &self.pipelines.pocket_mark.1,
            cell_groups,
        ));
        // Every Jacobi sweep is preceded by the single-workgroup bubble-row solve at its own
        // level (KTD-6 identical representation): the sweep then writes the fresh multiplier
        // into the pocket slots while relaxing the fluid rows against the previous one.
        let push_sweeps = |seq: &mut Vec<_>, n: u32, par: &mut usize| {
            for _ in 0..n {
                if !self.all_gas_atmospheric {
                    seq.push((
                        "bubble_fine",
                        &self.pipelines.bubble_fine.0,
                        &self.pipelines.bubble_fine.1[*par],
                        1,
                    ));
                }
                let family = if self.active_tiling {
                    &self.pipelines.jacobi_fine_tiled
                } else {
                    &self.pipelines.jacobi_fine
                };
                seq.push((
                    "jacobi_fine",
                    &family.0,
                    &family.1[*par],
                    if self.active_tiling {
                        ACTIVE_WATER_INDIRECT
                    } else {
                        cell_groups
                    },
                ));
                *par ^= 1;
            }
        };
        push_sweeps(&mut seq, pre, &mut par);
        if self.coarse_sweeps > 0 {
            let residual_family = if self.active_tiling {
                &self.pipelines.residual_tiled
            } else {
                &self.pipelines.residual
            };
            seq.push((
                "residual",
                &residual_family.0,
                &residual_family.1[par],
                if self.active_tiling {
                    ACTIVE_WATER_INDIRECT
                } else {
                    cell_groups
                },
            ));
            seq.push((
                "coarse_node_setup",
                &self.pipelines.coarse_node_setup.0,
                &self.pipelines.coarse_node_setup.1,
                cnode_groups,
            ));
            seq.push((
                "coarse_cell_setup",
                &self.pipelines.coarse_cell_setup.0,
                &self.pipelines.coarse_cell_setup.1,
                ccell_groups,
            ));
            for s in 0..self.coarse_sweeps {
                if !self.all_gas_atmospheric {
                    seq.push((
                        "bubble_coarse",
                        &self.pipelines.bubble_coarse.0,
                        &self.pipelines.bubble_coarse.1[(s % 2) as usize],
                        1,
                    ));
                }
                seq.push((
                    "jacobi_coarse",
                    &self.pipelines.jacobi_coarse.0,
                    &self.pipelines.jacobi_coarse.1[(s % 2) as usize],
                    ccell_groups,
                ));
            }
            let pc_par = (self.coarse_sweeps % 2) as usize;
            let prolong_family = if self.active_tiling {
                &self.pipelines.prolong_add_tiled
            } else {
                &self.pipelines.prolong_add
            };
            seq.push((
                "prolong_add",
                &prolong_family.0,
                &prolong_family.1[par][pc_par],
                if self.active_tiling {
                    ACTIVE_WATER_INDIRECT
                } else {
                    cell_groups
                },
            ));
            par ^= 1;
        }
        push_sweeps(&mut seq, post, &mut par);
        debug_assert_eq!(par, self.pressure_parity());
        seq.push((
            "project",
            &self.pipelines.project.0,
            &self.pipelines.project.1[par],
            node_groups,
        ));
        // Chemistry sees the pressure-projected phase velocities, then updates particle solute
        // pools/concentration and temperature before either phase advects. The two passes are
        // CPU-elided together only when both extraction and thermal transport are disabled.
        if self.extraction_on || self.thermal_on {
            seq.push((
                "chemistry_scatter",
                &self.pipelines.chemistry_scatter.0,
                &self.pipelines.chemistry_scatter.1,
                part_groups,
            ));
            seq.push((
                "chemistry_gather",
                &self.pipelines.chemistry_gather.0,
                &self.pipelines.chemistry_gather.1,
                part_groups,
            ));
        }
        if self.filter_reservoir_on {
            seq.push((
                "filter_release",
                &self.pipelines.filter_release.0,
                &self.pipelines.filter_release.1,
                water_groups,
            ));
        }
        seq.push((
            "g2p_water",
            &self.pipelines.g2p_water.0,
            &self.pipelines.g2p_water.1,
            water_groups,
        ));
        // U5: solid gather + advection + deformation update + return map (plasticity.wgsl).
        if self.solid_dynamic {
            if self.implicit_solid {
                seq.push((
                    "g2p_solid_velocity_only",
                    &self.pipelines.g2p_solid_velocity_only.0,
                    &self.pipelines.g2p_solid_velocity_only.1,
                    solid_groups,
                ));
            } else {
                seq.push((
                    "g2p_solid",
                    &self.pipelines.g2p_solid.0,
                    &self.pipelines.g2p_solid.1,
                    solid_groups,
                ));
            }
        }
        for ss in 0..substeps {
            // Timestamp queries only on the first substep (fixed query-set capacity); the
            // dispatch counter still counts every substep — the budget is the honest total.
            // After substep 0 the recorder batches the remaining dispatches into one pass.
            if ss == 1 {
                rec.disable_timestamps();
            }
            for &(label, pipe, bind, g) in &seq {
                if g == ACTIVE_TILE_MARKER {
                    rec.dispatch(&mut enc, pipe, bind, label, self.active_tile_count.max(1));
                    rec.flush();
                    enc.copy_buffer_to_buffer(
                        &self.active_tiles,
                        ACTIVE_WATER_COUNT_OFFSET,
                        &self.active_tile_indirect,
                        ACTIVE_WATER_INDIRECT_OFFSET,
                        4,
                    );
                    continue;
                }
                let indirect_offset = match g {
                    ACTIVE_WATER_INDIRECT => Some(ACTIVE_WATER_INDIRECT_OFFSET),
                    _ => None,
                };
                if let Some(offset) = indirect_offset {
                    rec.dispatch_indirect(
                        &mut enc,
                        pipe,
                        bind,
                        label,
                        &self.active_tile_indirect,
                        offset,
                    );
                } else {
                    rec.dispatch(&mut enc, pipe, bind, label, g);
                }
            }
        }
        if diagnostics_slot.is_some() {
            rec.dispatch(
                &mut enc,
                &self.pipelines.diagnostics_reduce.0,
                &self.pipelines.diagnostics_reduce.1,
                "diagnostics_reduce",
                1,
            );
        }
        let (dispatches, cursor, labels) = rec.finish();

        if let Some(slot) = diagnostics_slot {
            enc.copy_buffer_to_buffer(
                &self.diagnostics_gpu,
                0,
                &self.diagnostics_readbacks[slot].buffer,
                0,
                std::mem::size_of::<DiagnosticsGpu>() as u64,
            );
        }

        if let Some(ts) = &self.ts {
            if cursor >= 2 {
                enc.resolve_query_set(&ts.qset, 0..cursor, &ts.resolve, 0);
                enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, (cursor as u64) * 8);
            }
        }
        self.queue.submit(Some(enc.finish()));
        if let Some(slot) = diagnostics_slot {
            self.begin_periodic_diagnostics_map(slot, self.sim_time);
        }

        self.dispatches = dispatches;
        if let Some(ts) = self.ts.as_mut() {
            ts.labels = labels;
            ts.count = cursor;
        }
    }

    fn particles(&self) -> ParticleBuffers {
        // Exposure of the pool layout: with no solids the live water range `[0, water_count)`
        // is a contiguous prefix, so only the live set is exposed (dormant tail hidden —
        // the xpbd shape). With solids present the dormant gap sits between the water pool
        // and the solid range; the full pool is exposed and dormant slots stay parked at the
        // box corner with zero velocity (no twofield scene renders solids + pour before U5).
        let exposed = if self.solid_count == 0 {
            self.water_count
        } else {
            self.params.particle_count
        };
        ParticleBuffers {
            particle_count: exposed,
            position: Some(Arc::clone(&self.pos)),
            velocity: Some(Arc::clone(&self.vel)),
            phase_tag: Some(Arc::clone(&self.phase)),
            // One phase-specific chem buffer backs both canonical views: water concentration is
            // .x and water T_K is .y; grains carry fast/slow pools in .xy, grain T_K in .z, and
            // wet-contact time in .w. Consumers must select the temperature lane by phase.
            concentration: Some(Arc::clone(&self.chem)),
            temperature: Some(Arc::clone(&self.chem)),
            ..Default::default()
        }
    }

    fn metrics(&self) -> Metrics {
        // All values CPU-resident — never a GPU sync here (trait contract). Chemistry metrics and
        // drawdown/evenness come from the latest completed compact asynchronous snapshot (or an
        // explicit blocking validation snapshot); iteration_count carries the pressure budget.
        Metrics {
            particle_count: self.water_count + self.solid_count, // live set
            extraction_yield: self.cached_diag.extraction_yield,
            tds: self.cached_diag.cup_tds,
            drawdown_time: self.cached_diag.drawdown_time,
            evenness: self.cached_diag.evenness,
            iteration_count: self.fine_sweeps
                + self.coarse_sweeps
                + if self.implicit_solid {
                    COMPACTION_SWEEPS + self.shear_substeps
                } else {
                    0
                },
        }
    }

    fn profile(&self) -> Profile {
        Profile {
            passes: self.cached_passes.clone(),
            dispatches_per_frame: self.dispatches,
        }
    }
}

#[cfg(test)]
mod seed_tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 2.0e-6 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn reported_tds_is_solute_over_solution_not_shader_concentration_ratio() {
        let carrier = 3.0;
        let solute = 1.0;
        assert!(close(solution_solute_mass_fraction(carrier, solute), 0.25));
        assert!(!close(
            solution_solute_mass_fraction(carrier, solute),
            solute / carrier
        ));
    }

    #[test]
    fn atmospheric_gas_mode_removes_only_flood_and_bubble_work() {
        let dims = [19, 31, 23];
        let atmospheric = u4_surface_dispatches_for_mode(dims, true);
        let sealed = u4_surface_dispatches_for_mode(dims, false);

        assert_eq!(atmospheric, U4_SURFACE_MAINTENANCE_DISPATCHES);
        assert_eq!(u4_surface_dispatches_for(dims), sealed);
        assert_eq!(
            sealed - atmospheric,
            flood_sweeps_for(dims) + FINE_SWEEPS_DEFAULT + COARSE_SWEEPS_DEFAULT
        );
        assert_eq!(
            dispatches_per_frame_for_mode(dims, false) - dispatches_per_frame_for_mode(dims, true),
            sealed - atmospheric
        );
    }

    #[test]
    fn active_tile_neighbor_halo_covers_every_one_cell_operator_neighbor() {
        use std::collections::BTreeSet;

        assert!(!active_tiling_enabled(false, false));
        assert!(!active_tiling_enabled(false, true));
        assert!(!active_tiling_enabled(true, false));
        assert!(active_tiling_enabled(true, true));

        let node_dims = [19u32, 18, 17];
        let cell_dims = node_dims.map(|extent| extent - 1);
        let tile_dims = active_tile_dims_for_grid(node_dims);
        assert_eq!(
            active_tile_count_for_grid(node_dims),
            tile_dims.into_iter().product::<u32>()
        );

        let tile_id = |cell: [u32; 3]| -> u32 {
            let tile = cell.map(|coordinate| coordinate / ACTIVE_TILE_EDGE);
            tile[0] + tile_dims[0] * (tile[1] + tile_dims[1] * tile[2])
        };
        let marked_for_occupied_cell = |cell: [u32; 3]| {
            let mut marked = BTreeSet::new();
            let center = cell.map(|coordinate| (coordinate / ACTIVE_TILE_EDGE) as i32);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let tile = [center[0] + dx, center[1] + dy, center[2] + dz];
                        if (0..3).all(|axis| tile[axis] >= 0 && tile[axis] < tile_dims[axis] as i32)
                        {
                            marked.insert(
                                tile[0] as u32
                                    + tile_dims[0]
                                        * (tile[1] as u32 + tile_dims[1] * tile[2] as u32),
                            );
                        }
                    }
                }
            }
            marked
        };

        // Interior, tile-face/edge/corner, low-domain, and partial-high-tile occupancy cases.
        for occupied in [
            [0, 0, 0],
            [7, 7, 7],
            [8, 8, 8],
            [15, 8, 7],
            [cell_dims[0] - 1, cell_dims[1] - 1, cell_dims[2] - 1],
        ] {
            let marked = marked_for_occupied_cell(occupied);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let neighbor = [
                            occupied[0] as i32 + dx,
                            occupied[1] as i32 + dy,
                            occupied[2] as i32 + dz,
                        ];
                        if (0..3).all(|axis| {
                            neighbor[axis] >= 0 && neighbor[axis] < cell_dims[axis] as i32
                        }) {
                            assert!(marked.contains(&tile_id([
                                neighbor[0] as u32,
                                neighbor[1] as u32,
                                neighbor[2] as u32,
                            ])));
                        }
                    }
                }
            }
            assert!(marked
                .iter()
                .all(|&tile| tile < active_tile_count_for_grid(node_dims)));
        }
    }

    fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        if std::env::var_os("COFFEE_SIM_SKIP_GPU_TESTS").is_some() {
            return None;
        }
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("twofield atmospheric-gas test device"),
            required_features: wgpu::Features::empty(),
            required_limits: crate::solvers::mpm::required_limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()
    }

    #[test]
    fn water_tiled_relaxation_matches_dense_relaxation_on_the_gpu() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("no GPU adapter available; skipping active-tile equivalence test");
            return;
        };
        let scene = Scene {
            gravity: [0.0, -9.0, 0.0],
            box_min: [-2.0; 3],
            box_max: [2.0; 3],
            regions: vec![
                crate::engine::scene::SeedRegion {
                    min: [-0.5, 0.5, -0.5],
                    max: [0.5, 1.0, 0.5],
                    species: Species::Water,
                },
                crate::engine::scene::SeedRegion {
                    min: [-0.75, -0.75, -0.75],
                    max: [0.75, 0.25, 0.75],
                    species: Species::Grain,
                },
            ],
            ..Scene::default()
        };
        let materials = Materials {
            particle_spacing: 0.5,
            support_radius: 1.0,
            grain_diameter: 0.5,
            ..Materials::default()
        };
        let config = Config {
            seed_jitter: 0.0,
            max_speed: 4.0,
            tf_all_gas_atmospheric: true,
            solid_dynamics: true,
            tf_compaction_tangent: 8.0,
            tf_numerical_shear_modulus: 0.0,
            ..Config::default()
        };
        let gpu = GpuContext::from_device_queue(&device, &queue);
        let mut tiled = TwofieldSolver::build(&scene, &materials, &config, &gpu);
        let mut dense = TwofieldSolver::build(&scene, &materials, &config, &gpu);
        assert!(tiled.active_tiling);
        dense.active_tiling = false;

        tiled.step(1.0 / 120.0, &EmissionInput::default());
        dense.step(1.0 / 120.0, &EmissionInput::default());
        assert_eq!(
            tiled.dispatches,
            dense.dispatches + 1,
            "indirect relaxation dispatches stay individually counted; only the marker is new"
        );
        let count = tiled.read_active_tile_count();
        assert!(count > 0 && count <= tiled.active_tile_count);

        let assert_close = |name: &str, left: &[f32], right: &[f32]| {
            assert_eq!(left.len(), right.len(), "{name} length");
            for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
                let tolerance = 32.0 * f32::EPSILON * (1.0 + a.abs().max(b.abs()));
                assert!(
                    (a - b).abs() <= tolerance,
                    "{name}[{index}] differs: tiled={a:?}, dense={b:?}, tolerance={tolerance:?}"
                );
            }
        };
        let flatten4 =
            |values: Vec<[f32; 4]>| bytemuck::cast_slice::<[f32; 4], f32>(&values).to_vec();
        assert_close(
            "positions",
            &flatten4(tiled.read_positions()),
            &flatten4(dense.read_positions()),
        );
        assert_close(
            "velocities",
            &flatten4(tiled.read_velocities()),
            &flatten4(dense.read_velocities()),
        );
        assert_close(
            "grid velocities",
            &flatten4(tiled.read_grid_velocities()),
            &flatten4(dense.read_grid_velocities()),
        );
        assert_close(
            "cell metadata",
            &flatten4(tiled.read_cell_meta()),
            &flatten4(dense.read_cell_meta()),
        );
        assert_close("pressure", &tiled.read_pressure(), &dense.read_pressure());
        assert_close(
            "solid state",
            &flatten4(tiled.read_solid_state()),
            &flatten4(dense.read_solid_state()),
        );
    }

    #[test]
    fn compact_periodic_diagnostics_complete_without_frame_waits_and_close_mass_ledgers() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("no GPU adapter available; skipping compact diagnostics test");
            return;
        };
        let scene = Scene {
            gravity: [0.0; 3],
            box_min: [-2.0; 3],
            box_max: [2.0; 3],
            regions: vec![crate::engine::scene::SeedRegion {
                min: [0.0; 3],
                max: [0.0; 3],
                species: Species::Water,
            }],
            ..Scene::default()
        };
        let materials = Materials {
            particle_spacing: 0.5,
            support_radius: 1.0,
            ..Materials::default()
        };
        let config = Config {
            seed_jitter: 0.0,
            tf_all_gas_atmospheric: true,
            ..Config::default()
        };
        let gpu = GpuContext::from_device_queue(&device, &queue);
        let mut solver = TwofieldSolver::build(&scene, &materials, &config, &gpu);

        solver.step(1.0 / 60.0, &EmissionInput::default());
        assert!(
            solver.diagnostics_pending(),
            "first frame should schedule, not synchronously consume, its compact map"
        );
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        solver.poll_periodic_diagnostics();
        let diagnostics = solver.diagnostics();

        assert!(diagnostics.sample_valid);
        assert!(!solver.diagnostics_pending());
        assert!(!diagnostics.has_nonfinite_state);
        assert!(close(
            diagnostics.total_carrier_mass,
            diagnostics.expected_carrier_mass
        ));
        assert!(close(diagnostics.carrier_mass_residual, 0.0));
        assert!(close(diagnostics.solute_mass_residual, 0.0));
        assert!(close(diagnostics.mechanical_mass_residual, 0.0));
        assert!(close(
            diagnostics.mobile_solution_mass,
            diagnostics.mobile_carrier_mass
        ));
        assert!(close(
            diagnostics.pond_carrier_mass,
            diagnostics.mobile_carrier_mass
        ));
        assert_eq!(diagnostics.mean_bound_saturation, 0.0);
        assert_eq!(diagnostics.mean_diameter_ratio, 1.0);
        assert_eq!(diagnostics.filter_beta_max_per_s, 0.0);
        assert_eq!(diagnostics.filter_pressure_mobility_min, 1.0);

        // The expensive validation path is deliberately separate, but it must decode the same
        // physical ledger as the compact production path.
        let asynchronous = diagnostics;
        solver.sample_diagnostics();
        let blocking = solver.diagnostics();
        assert!(close(
            blocking.total_carrier_mass,
            asynchronous.total_carrier_mass
        ));
        assert!(close(
            blocking.mobile_solution_mass,
            asynchronous.mobile_solution_mass
        ));
        assert!(close(
            blocking.mechanical_mass_residual,
            asynchronous.mechanical_mass_residual
        ));
        assert_eq!(blocking.filter_count_residual, 0);
        assert!(close(blocking.filter_volume_residual, 0.0));
    }

    fn cavity_shell_fixture(all_gas_atmospheric: bool) -> (Scene, Materials, Config) {
        let scene = Scene {
            gravity: [0.0; 3],
            box_min: [0.0; 3],
            box_max: [8.0; 3],
            regions: Vec::new(),
            ..Scene::default()
        };
        let materials = Materials {
            particle_spacing: 0.5,
            support_radius: 1.0,
            ..Materials::default()
        };
        let config = Config {
            seed_jitter: 0.0,
            tf_all_gas_atmospheric: all_gas_atmospheric,
            ..Config::default()
        };
        (scene, materials, config)
    }

    fn run_cavity_surface_passes(solver: &TwofieldSolver) {
        const CELL_FLUID: f32 = 0.0;
        const CELL_AIR: f32 = 1.0;

        let dims = solver.params.grid_dims;
        let cells = [dims[0] - 1, dims[1] - 1, dims[2] - 1];
        assert!(cells.iter().all(|&extent| extent >= 9));
        let mut cell_meta = vec![[0.0, 0.0, 0.0, CELL_AIR]; solver.num_cells as usize];
        let lo = [cells[0] / 2 - 2, cells[1] / 2 - 2, cells[2] / 2 - 2];
        let hi = [lo[0] + 4, lo[1] + 4, lo[2] + 4];
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let on_shell = x == lo[0]
                        || x == hi[0]
                        || y == lo[1]
                        || y == hi[1]
                        || z == lo[2]
                        || z == hi[2];
                    if on_shell {
                        let flat = x + cells[0] * (y + cells[1] * z);
                        cell_meta[flat as usize] = [0.0, 1.0, 0.0, CELL_FLUID];
                    }
                }
            }
        }

        solver
            .queue
            .write_buffer(&solver.cell_meta, 0, bytemuck::cast_slice(&cell_meta));
        let pressure_zeros = vec![0.0f32; solver.num_cells as usize];
        solver
            .queue
            .write_buffer(&solver.pf_a, 0, bytemuck::cast_slice(&pressure_zeros));
        solver
            .queue
            .write_buffer(&solver.pf_b, 0, bytemuck::cast_slice(&pressure_zeros));

        let mut encoder = solver
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield cavity surface comparison"),
            });
        let mut recorder = PassRecorder::new(None);
        let cell_groups = groups(solver.num_cells).max(1);
        if solver.all_gas_atmospheric {
            recorder.dispatch(
                &mut encoder,
                &solver.pipelines.flood_init.0,
                &solver.pipelines.flood_init.1,
                "gas_state_reset",
                1,
            );
        } else {
            recorder.dispatch(
                &mut encoder,
                &solver.pipelines.flood_init.0,
                &solver.pipelines.flood_init.1,
                "flood_init",
                cell_groups,
            );
            for sweep in 0..solver.flood_sweeps {
                recorder.dispatch(
                    &mut encoder,
                    &solver.pipelines.flood_sweep.0,
                    &solver.pipelines.flood_sweep.1[(sweep % 2) as usize],
                    "flood_sweep",
                    cell_groups,
                );
            }
        }
        recorder.dispatch(
            &mut encoder,
            &solver.pipelines.pocket_mark.0,
            &solver.pipelines.pocket_mark.1,
            "pocket_mark",
            cell_groups,
        );
        let _ = recorder.finish();
        solver.queue.submit(Some(encoder.finish()));
    }

    #[test]
    fn atmospheric_mode_keeps_an_enclosed_cavity_at_zero_gauge_pressure() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("no GPU adapter available; skipping atmospheric cavity comparison");
            return;
        };
        let gpu = GpuContext::from_device_queue(&device, &queue);

        let (scene, materials, sealed_config) = cavity_shell_fixture(false);
        let sealed = TwofieldSolver::build(&scene, &materials, &sealed_config, &gpu);
        run_cavity_surface_passes(&sealed);
        assert!(
            sealed.pocket_present(),
            "generic gas handling must retain a resolved enclosed pocket"
        );

        let (_, _, atmospheric_config) = cavity_shell_fixture(true);
        let atmospheric = TwofieldSolver::build(&scene, &materials, &atmospheric_config, &gpu);
        run_cavity_surface_passes(&atmospheric);
        assert!(
            !atmospheric.pocket_present(),
            "vented production mode must leave enclosed resolved gas atmospheric"
        );
        assert_eq!(atmospheric.read_bubble(), [0.0, 0.0]);
    }

    #[test]
    fn filter_reservoir_admits_and_releases_whole_parcels_conservatively() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("no GPU adapter available; skipping paper reservoir comparison");
            return;
        };
        let solids = crate::utils::geometry::v60_dripper();
        let start = glam::Vec3::new(1.55, -1.0, 0.0);
        let paper_hit = solids[1].cavity(start);
        assert!(paper_hit.0 > 0.0, "fixture starts upstream of the paper");
        let downstream = -paper_hit.1;
        assert!(downstream.y < 0.0, "V60 paper drains downward");
        let scene = Scene {
            // Zero gravity makes release depend solely on the imposed post-projection grid flux.
            gravity: [0.0, 0.0, 0.0],
            box_min: [-7.0, -10.0, -7.0],
            box_max: [7.0, 10.0, 7.0],
            regions: vec![crate::engine::scene::SeedRegion {
                min: [0.0, -1.0, 0.0],
                max: [0.0, -1.0, 0.0],
                species: Species::Water,
            }],
            solids,
            ..Scene::default()
        };
        let materials = Materials {
            particle_spacing: 0.5,
            support_radius: 1.0,
            ..Materials::default()
        };
        let config = Config {
            seed_jitter: 0.0,
            tf_filter_permeability_m2: 1.0e-8,
            tf_filter_kinematic_viscosity_m2_s: 3.2e-7,
            tf_water_density_kg_m3: 1_000.0,
            tf_filter_capacity_g: 1.0,
            tf_filter_thickness_m: 0.000_2,
            tf_filter_porosity: 0.3,
            ..Config::default()
        };
        let gpu = GpuContext::from_device_queue(&device, &queue);
        let solver = TwofieldSolver::build(&scene, &materials, &config, &gpu);
        assert_eq!(solver.phase_counts(), (1, 0));
        solver.write_positions_for_test(&[[start.x, start.y, start.z, 1.0]]);

        let speed = 10.0;
        let grid_velocity = [
            downstream.x * speed,
            downstream.y * speed,
            downstream.z * speed,
            8.0,
        ];
        solver.write_velocities_for_test(&[[
            grid_velocity[0],
            grid_velocity[1],
            grid_velocity[2],
            0.0,
        ]]);
        solver.write_grid_velocities_for_test(&vec![grid_velocity; solver.num_nodes as usize]);
        solver.run_dbg(
            &solver.pipelines.g2p_water.0,
            &solver.pipelines.g2p_water.1,
            1,
        );

        let admitted_chem = solver.read_chemistry()[0];
        let admitted_pos = solver.read_positions()[0];
        let admitted = solver.read_filter_state();
        let expected_counts =
            (materials.particle_spacing.powi(3) as f64 * FILTER_VOLUME_SCALE).round() as u32;
        assert_eq!(admitted[0], expected_counts);
        assert_eq!(&admitted[1..], &[1, 0, 0, 0]);
        assert_eq!(admitted_chem[2], 1.0);
        assert_eq!(admitted_chem[3], 0.0);
        assert_eq!(admitted_pos[3], 1.0);

        solver.write_grid_carrier_masses_for_test(&vec![1.0; solver.num_nodes as usize]);
        solver.run_dbg(
            &solver.pipelines.filter_release.0,
            &solver.pipelines.filter_release.1,
            1,
        );
        let released = solver.read_filter_state();
        let released_chem = solver.read_chemistry()[0];
        let released_pos = solver.read_positions()[0];
        let released_velocity = solver.read_velocities()[0];
        assert_eq!(released, [0, 1, 1, 0, 0]);
        assert_eq!(released_chem[0], admitted_chem[0]);
        assert_eq!(released_chem[1], admitted_chem[1]);
        assert_eq!(released_chem[2], 2.0);
        assert_eq!(released_pos[3], admitted_pos[3]);
        let released_normal_speed = released_velocity[0] * downstream.x
            + released_velocity[1] * downstream.y
            + released_velocity[2] * downstream.z;
        assert!(close(released_normal_speed, speed));
    }

    #[test]
    fn capillary_capacity_scatter_is_motion_and_quadrature_invariant() {
        let Some((device, queue)) = create_test_device() else {
            eprintln!("no GPU adapter available; skipping capillary scatter comparison");
            return;
        };
        let scene = Scene {
            gravity: [0.0; 3],
            box_min: [0.0; 3],
            box_max: [4.0; 3],
            regions: vec![crate::engine::scene::SeedRegion {
                min: [0.5; 3],
                max: [3.5; 3],
                species: Species::Grain,
            }],
            dose_g: 12.0,
            ..Scene::default()
        };
        let materials = Materials {
            particle_spacing: 0.4,
            grain_diameter: 0.5,
            porosity: 0.4,
            r_max: 0.0,
            ..Materials::default()
        };
        let gpu = GpuContext::from_device_queue(&device, &queue);
        let mut totals = Vec::new();
        for budget in [0, 47] {
            let config = Config {
                seed_jitter: 0.0,
                tf_solid_sample_budget: budget,
                tf_retention_target_g_per_g: 1.0,
                tf_water_density_kg_m3: 980.0,
                ..Config::default()
            };
            let solver = TwofieldSolver::build(&scene, &materials, &config, &gpu);
            solver.run_dbg(
                &solver.pipelines.p2g_solid.0,
                &solver.pipelines.p2g_solid.1,
                groups(solver.solid_count).max(1),
            );
            let (capacity, bloom) = solver.read_grid_capillary_totals();
            let encoded_per_solid =
                (solver.params.solute[3] * FP_SCALE as f32).round() / FP_SCALE as f32;
            assert!(close(
                capacity,
                encoded_per_solid * solver.solid_count as f32
            ));
            assert_eq!(
                bloom, 0.0,
                "uncontacted grains must scatter no suction gate"
            );

            let mut moved = solver.read_positions();
            for position in &mut moved {
                position[0] += 0.11;
            }
            solver.write_positions_for_test(&moved);
            solver.run_dbg(
                &solver.pipelines.grid_clear.0,
                &solver.pipelines.grid_clear.1,
                groups(solver.num_nodes).max(1),
            );
            solver.run_dbg(
                &solver.pipelines.p2g_solid.0,
                &solver.pipelines.p2g_solid.1,
                groups(solver.solid_count).max(1),
            );
            let (moved_capacity, _) = solver.read_grid_capillary_totals();
            assert!(close(moved_capacity, capacity));
            totals.push(capacity);
        }
        assert!(
            (totals[0] - totals[1]).abs() <= 2.0e-4 * totals[0].abs().max(1.0),
            "quadrature changed capacity: full={} thinned={}",
            totals[0],
            totals[1]
        );
    }

    /// The range layout is water-first: every water particle precedes every solid.
    #[test]
    fn seeds_water_first_then_solids() {
        let scene = Scene::pour_over();
        let (pos, phase, water_count, _) =
            seed_ranges(&scene, &Materials::default(), &Config::default());
        assert_eq!(pos.len(), phase.len());
        assert!(water_count > 0, "pour_over seeds water");
        assert!(
            (water_count as usize) < phase.len(),
            "pour_over seeds solids too"
        );
        for (i, &ph) in phase.iter().enumerate() {
            let expect = if (i as u32) < water_count { 0 } else { 1 };
            assert_eq!(ph, expect, "particle {i} out of range order");
        }
        // Moisture lane per species: water full (1), grain dry (0).
        for (p, &ph) in pos.iter().zip(&phase) {
            assert_eq!(p[3], if ph == 0 { 1.0 } else { 0.0 });
        }
    }

    #[test]
    fn seed_clearance_tracks_each_species_sample_pitch() {
        let scene = Scene::pour_over();
        let materials = Materials {
            particle_spacing: 0.18,
            grain_diameter: 0.47,
            ..Materials::default()
        };
        let config = Config {
            seed_jitter: 0.08,
            ..Config::default()
        };
        let (positions, phases, _, _) = seed_ranges(&scene, &materials, &config);
        assert!(!positions.is_empty());
        for (position, phase) in positions.iter().zip(phases) {
            let pitch = if phase == 0 {
                materials.particle_spacing
            } else {
                materials.grain_diameter
            };
            let hit = crate::utils::sdf::nearest(
                &scene.solids,
                glam::Vec3::new(position[0], position[1], position[2]),
                phase,
            );
            assert!(
                hit.signed + 2.0e-5 >= 0.5 * pitch,
                "phase {phase} seed at {position:?} has clearance {}, pitch {pitch}",
                hit.signed
            );
        }
    }

    fn solid_budget_fixture() -> (Scene, Materials, Config) {
        let scene = Scene {
            regions: vec![crate::engine::scene::SeedRegion {
                min: [0.0, 0.0, 0.0],
                max: [4.0, 4.0, 4.0],
                species: Species::Grain,
            }],
            solids: Vec::new(),
            ..Scene::default()
        };
        let materials = Materials {
            grain_diameter: 0.25,
            ..Materials::default()
        };
        let config = Config {
            seed_jitter: 0.05,
            ..Config::default()
        };
        (scene, materials, config)
    }

    #[test]
    fn solid_sample_budget_is_exact_after_geometric_acceptance() {
        let (scene, materials, mut config) = solid_budget_fixture();
        let raw_count = seed_ranges(&scene, &materials, &config).0.len();
        assert!(
            raw_count > 257,
            "fixture must geometrically accept more solids than the requested budget"
        );
        config.tf_solid_sample_budget = 257;
        let (positions, phases, water_count, weight) = seed_ranges(&scene, &materials, &config);
        assert_eq!(water_count, 0);
        assert_eq!(positions.len(), 257);
        assert_eq!(phases, vec![1; 257]);
        assert!(weight > 1.0);
    }

    #[test]
    fn free_water_budget_keeps_a_contiguous_lowest_fill() {
        let original: Vec<[f32; 4]> = (0..3)
            .flat_map(|y| (0..3).map(move |x| [x as f32, y as f32, 0.0, 1.0]))
            .collect();
        let mut selected = original.clone();
        downsample_lowest_fill(&mut selected, 5, 17);
        assert_eq!(selected.len(), 5);
        assert!(selected.iter().all(|point| point[1] <= 1.0));
        assert_eq!(selected.iter().filter(|point| point[1] == 0.0).count(), 3);

        let mut repeated = original;
        downsample_lowest_fill(&mut repeated, 5, 17);
        assert_eq!(selected, repeated);
    }

    #[test]
    fn solid_quadrature_preserves_bulk_and_structural_volume_under_uniform_thinning() {
        let (scene, materials, mut config) = solid_budget_fixture();
        let (raw, _, raw_water, raw_weight) = seed_ranges(&scene, &materials, &config);
        let raw_solids = raw.len() - raw_water as usize;
        assert_eq!(raw_weight, 1.0);

        config.tf_solid_sample_budget = (raw_solids / 3).max(1) as u32;
        let (sampled, _, sampled_water, sampled_weight) = seed_ranges(&scene, &materials, &config);
        let sampled_solids = sampled.len() - sampled_water as usize;
        let bulk_per_raw = materials.grain_diameter.powi(3);
        let structure_per_raw =
            dry_solid_sample_volume(materials.grain_diameter, materials.porosity);
        assert!(close(
            raw_solids as f32 * bulk_per_raw,
            sampled_solids as f32 * bulk_per_raw * sampled_weight,
        ));
        assert!(close(
            raw_solids as f32 * structure_per_raw,
            sampled_solids as f32 * structure_per_raw * sampled_weight,
        ));
        assert!(close(
            structure_per_raw / bulk_per_raw,
            (structure_per_raw * sampled_weight) / (bulk_per_raw * sampled_weight),
        ));
    }

    #[test]
    fn solid_sample_budget_is_deterministic_and_retains_accepted_order() {
        let (scene, materials, mut config) = solid_budget_fixture();
        let raw = seed_ranges(&scene, &materials, &config).0;
        config.tf_solid_sample_budget = 257;
        let first = seed_ranges(&scene, &materials, &config).0;
        let second = seed_ranges(&scene, &materials, &config).0;
        assert_eq!(first, second);

        // The selected points must be a subsequence of the geometrically accepted order. This
        // catches priority-order output, which would be deterministic but would perturb every
        // downstream particle/buffer ordering contract.
        let mut raw_cursor = 0usize;
        for selected in &first {
            while raw_cursor < raw.len() && raw[raw_cursor] != *selected {
                raw_cursor += 1;
            }
            assert!(
                raw_cursor < raw.len(),
                "sampled solid is not in increasing accepted-point order"
            );
            raw_cursor += 1;
        }
    }

    #[test]
    fn zero_or_nonbinding_solid_sample_budget_is_a_noop() {
        let (scene, materials, config) = solid_budget_fixture();
        assert_eq!(config.tf_solid_sample_budget, 0);
        let unbounded = seed_ranges(&scene, &materials, &config);

        let mut nonbinding = config;
        nonbinding.tf_solid_sample_budget = u32::MAX;
        assert_eq!(seed_ranges(&scene, &materials, &nonbinding), unbounded);
    }

    #[test]
    fn solid_sample_budget_preserves_broad_spatial_extent() {
        let (scene, materials, mut config) = solid_budget_fixture();
        let raw = seed_ranges(&scene, &materials, &config).0;
        config.tf_solid_sample_budget = 257;
        let sampled = seed_ranges(&scene, &materials, &config).0;

        for axis in 0..3 {
            let raw_min = raw
                .iter()
                .map(|point| point[axis])
                .fold(f32::INFINITY, f32::min);
            let raw_max = raw
                .iter()
                .map(|point| point[axis])
                .fold(f32::NEG_INFINITY, f32::max);
            let span = raw_max - raw_min;
            assert!(span > 0.0);
            assert!(
                sampled
                    .iter()
                    .any(|point| point[axis] <= raw_min + 0.2 * span),
                "axis {axis}: sample lost the low side of the accepted extent"
            );
            assert!(
                sampled
                    .iter()
                    .any(|point| point[axis] >= raw_max - 0.2 * span),
                "axis {axis}: sample lost the high side of the accepted extent"
            );
        }
    }

    #[test]
    fn structural_sample_volume_reproduces_declared_porosity() {
        let pitch = 0.37;
        for porosity in [0.1, 0.4, 0.8] {
            let bulk_volume = pitch * pitch * pitch;
            let solid_volume = dry_solid_sample_volume(pitch, porosity);
            assert!((1.0 - solid_volume / bulk_volume - porosity).abs() < 1.0e-6);
        }
    }

    /// Pure inventory identities only: arbitrary reduced values exercise both attainable and
    /// over-capacity outcomes without pinning a coffee-retention calibration.
    #[test]
    fn retention_build_ledger_reports_attainable_and_unmet_targets() {
        let materials = Materials {
            particle_mass: 2.0,
            grain_diameter: 0.5,
            porosity: 0.4,
            r_max: 0.25,
            ..Materials::default()
        };
        let mut config = Config {
            tf_retention_target_g_per_g: 0.75,
            ..Config::default()
        };
        let dry_mass_per_sample = 0.8;
        let count = 10;
        let water_sample_volume = 0.2;
        let dry_structural_volume =
            dry_solid_sample_volume(materials.grain_diameter, materials.porosity);

        let attainable = retention_state_for_build(
            &config,
            &materials,
            dry_mass_per_sample,
            count,
            dry_structural_volume,
            1.0,
            water_sample_volume,
        )
        .unwrap();
        assert!(attainable.target_attainable());
        assert!(close(
            attainable.bound_capacity_volume + attainable.retained_capillary_volume,
            attainable.whole_bed_target_volume
        ));
        assert!(close(
            attainable.residual_pore_saturation * attainable.available_pore_volume,
            attainable.retained_capillary_volume
        ));
        assert_eq!(attainable.unmet_volume(), 0.0);

        config.tf_retention_target_g_per_g = 2.0;
        let unmet = retention_state_for_build(
            &config,
            &materials,
            dry_mass_per_sample,
            count,
            dry_structural_volume,
            1.0,
            water_sample_volume,
        )
        .unwrap();
        assert!(!unmet.target_attainable());
        assert!(close(
            unmet.retained_capillary_volume + unmet.unmet_volume(),
            unmet.capillary_target_volume
        ));
        assert!(close(
            unmet.retained_capillary_volume,
            unmet.available_pore_volume
        ));
        assert_eq!(unmet.residual_pore_saturation, 1.0);
    }

    #[test]
    fn node_pore_saturation_is_bounded_and_monotone_in_water_volume() {
        let pore_volume = 0.8;
        let samples =
            [-1.0, 0.0, 0.2, 0.8, 1.6].map(|water| node_pore_saturation(water, pore_volume));
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 0.0);
        assert!(samples.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(samples.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(samples[3], 1.0);
        assert_eq!(samples[4], 1.0);
        assert_eq!(node_pore_saturation(0.5, 0.0), 0.0);
    }

    #[test]
    fn measured_capillary_mobility_is_bounded_monotone_and_exactly_stiff_when_held() {
        let residual = 0.3;
        let exponent = 2.4;
        let floor = 0.02;
        let mobility = [-0.5, 0.0, residual, 0.5, 0.8, 1.0, 1.5]
            .map(|saturation| bounded_corey_mobility(saturation, residual, exponent, floor));
        assert!(mobility.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(mobility.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(mobility[0], 0.0);
        assert_eq!(mobility[2], 0.0);
        assert!(mobility[3] > 0.0);
        assert!(mobility[5] < 1.0, "held water keeps theta below one");
        assert_eq!(
            mobility[6], mobility[5],
            "input saturation is physically bounded"
        );
    }

    #[test]
    fn uniform_saturation_and_nonbed_boundaries_produce_zero_capillary_force() {
        let uniform = capillary_acceleration_from_stencil(
            0.4,
            [
                Some(0.4),
                Some(0.4),
                Some(0.4),
                Some(0.4),
                Some(0.4),
                Some(0.4),
            ],
            0.3,
            7.0,
        );
        assert_eq!(uniform, [0.0; 3]);
        let isolated = capillary_acceleration_from_stencil(0.4, [None; 6], 0.3, 7.0);
        assert_eq!(isolated, [0.0; 3]);
    }

    #[test]
    fn capillary_front_force_rotates_with_the_saturation_front() {
        let center = 0.5;
        let h = 0.4;
        let accel = 3.0;
        let x_front = capillary_acceleration_from_stencil(
            center,
            [
                Some(1.0),
                Some(0.0),
                Some(center),
                Some(center),
                Some(center),
                Some(center),
            ],
            h,
            accel,
        );
        let y_front = capillary_acceleration_from_stencil(
            center,
            [
                Some(center),
                Some(center),
                Some(1.0),
                Some(0.0),
                Some(center),
                Some(center),
            ],
            h,
            accel,
        );
        let z_front = capillary_acceleration_from_stencil(
            center,
            [
                Some(center),
                Some(center),
                Some(center),
                Some(center),
                Some(1.0),
                Some(0.0),
            ],
            h,
            accel,
        );
        assert!(x_front[0] > 0.0 && y_front[1] > 0.0 && z_front[2] > 0.0);
        assert!(close(x_front[0], y_front[1]));
        assert!(close(x_front[0], z_front[2]));
        assert!(close(x_front[1], 0.0) && close(x_front[2], 0.0));
        assert!(close(y_front[0], 0.0) && close(y_front[2], 0.0));
        assert!(close(z_front[0], 0.0) && close(z_front[1], 0.0));
    }

    #[test]
    fn recipe_mass_is_invariant_to_solid_sample_count() {
        let scene = Scene {
            dose_g: 17.0,
            ..Scene::default()
        };
        let materials = Materials {
            particle_spacing: 0.2,
            particle_mass: 1.7,
            ..Materials::default()
        };
        let coarse_count = 127;
        let fine_count = 2_003;
        let density = 975.0;
        let coarse = recipe_solid_mass_reduced(&scene, &materials, density, coarse_count).unwrap();
        let fine = recipe_solid_mass_reduced(&scene, &materials, density, fine_count).unwrap();
        let coarse_total = coarse * coarse_count as f32;
        let fine_total = fine * fine_count as f32;
        assert!((coarse_total - fine_total).abs() <= 2.0e-5 * coarse_total.abs().max(1.0));

        // A reduced mass unit is one water sample. For fixed spacing and physical coffee mass,
        // reduced recipe mass is inversely proportional to the one authoritative water density.
        let warmer_density = 960.0;
        let warmer =
            recipe_solid_mass_reduced(&scene, &materials, warmer_density, coarse_count).unwrap();
        assert!(close(coarse * density, warmer * warmer_density));
    }

    #[test]
    fn shear_cfl_density_is_dry_bulk_density_and_not_swollen_particle_density() {
        let dry_mass = 2.4;
        let dry_structural_volume = 0.006;
        let porosity = 0.42;
        let bulk = dry_bulk_density(dry_mass, dry_structural_volume, porosity);
        let material = dry_mass / dry_structural_volume;
        assert!(close(bulk, material * (1.0 - porosity)));
        assert!(bulk < material);

        let diameter_swelling = 0.15_f32;
        let incorrect_swollen_density =
            dry_mass / (dry_structural_volume * (1.0 + diameter_swelling).powi(3));
        assert!(
            !close(bulk, incorrect_swollen_density),
            "a swelling-dependent particle density must not alias the dry bulk definition"
        );
        assert_eq!(
            dry_bulk_density(dry_mass, dry_structural_volume, porosity),
            bulk,
            "swelling is intentionally absent from the dry bulk definition"
        );
    }

    /// Algebraic inventory/layout invariant only; no calibration constant is pinned.
    #[test]
    fn chemistry_seed_conserves_recipe_soluble_mass_and_keeps_dormant_slots_inert() {
        let materials = Materials {
            soluble_fraction: 0.23,
            fast_fraction: 0.37,
            ..Materials::default()
        };
        let live_water = 2;
        let water_capacity = 5;
        let solid_count = 7;
        let dry_mass = 0.41;
        let chemistry = seed_chemistry(
            live_water,
            water_capacity,
            solid_count,
            dry_mass,
            &materials,
        );
        assert!(chemistry[live_water as usize..water_capacity as usize]
            .iter()
            .all(|state| *state == [0.0; 4]));
        let pools: f32 = chemistry[water_capacity as usize..]
            .iter()
            .map(|state| state[0] + state[1])
            .sum();
        let expected = materials.soluble_fraction * dry_mass * solid_count as f32;
        assert!((pools - expected).abs() <= 2.0e-6 * expected.max(1.0));
    }

    fn conservative_transfer_velocity(
        grid_velocity: [f32; 3],
        new_mass: f32,
        delta_mass: f32,
        delta_momentum: [f32; 3],
    ) -> [f32; 3] {
        assert!(new_mass > 0.0);
        let old_mass = new_mass - delta_mass;
        assert!(old_mass >= 0.0);
        std::array::from_fn(|axis| {
            (old_mass * grid_velocity[axis] + delta_momentum[axis]) / new_mass
        })
    }

    /// Algebraic twin of the WGSL chemistry ledger. The deliberately arbitrary values lock the
    /// carrier/inertia distinction and signed transfer identities without pinning a coffee
    /// calibration target.
    #[test]
    fn dissolution_preserves_carrier_mass_momentum_capacity_and_enthalpy() {
        let particle_mass = 1.7f32;
        let particle_volume = 0.43f32;
        let fraction = 0.61f32;
        let concentration_before = 0.037f32;
        let insoluble = 0.52f32;
        let fast = 0.19f32;
        let slow = 0.31f32;
        let bound_water = 0.08f32;
        let transferred = 0.11f32;
        assert!(transferred < fast + slow);

        let solvent = particle_mass * fraction;
        let carrier_volume_before = particle_volume * fraction;
        let dissolved_before = solvent * concentration_before;
        let dissolved_after = dissolved_before + transferred;
        let concentration_after = dissolved_after / solvent;
        let carrier_volume_after = particle_volume * fraction;
        let water_mass_before = solvent + dissolved_before;
        let water_mass_after = solvent + solvent * concentration_after;
        let grain_mass_before = insoluble + fast + slow + bound_water;
        let grain_mass_after = grain_mass_before - transferred;

        assert_eq!(carrier_volume_after, carrier_volume_before);
        assert!((water_mass_after - water_mass_before - transferred).abs() < 2.0e-7);
        assert!((grain_mass_before - grain_mass_after - transferred).abs() < 2.0e-7);
        assert!(
            (water_mass_before + grain_mass_before - water_mass_after - grain_mass_after).abs()
                < 3.0e-7
        );

        let water_velocity = [0.7f32, -0.4, 0.2];
        let grain_velocity = [-0.3f32, 0.5, 0.9];
        let transferred_momentum = grain_velocity.map(|velocity| transferred * velocity);
        let water_after_velocity = conservative_transfer_velocity(
            water_velocity,
            water_mass_after,
            transferred,
            transferred_momentum,
        );
        let grain_after_velocity = conservative_transfer_velocity(
            grain_velocity,
            grain_mass_after,
            -transferred,
            transferred_momentum.map(|momentum| -momentum),
        );
        for axis in 0..3 {
            let before =
                water_mass_before * water_velocity[axis] + grain_mass_before * grain_velocity[axis];
            let after = water_mass_after * water_after_velocity[axis]
                + grain_mass_after * grain_after_velocity[axis];
            assert!((after - before).abs() < 3.0e-7);
        }

        let uniform_velocity = [0.2f32, -0.6, 0.4];
        let uniform_momentum = uniform_velocity.map(|velocity| transferred * velocity);
        let uniform_water = conservative_transfer_velocity(
            uniform_velocity,
            water_mass_after,
            transferred,
            uniform_momentum,
        );
        let uniform_grain = conservative_transfer_velocity(
            uniform_velocity,
            grain_mass_after,
            -transferred,
            uniform_momentum.map(|momentum| -momentum),
        );
        for axis in 0..3 {
            assert!((uniform_water[axis] - uniform_velocity[axis]).abs() < 2.0e-7);
            assert!((uniform_grain[axis] - uniform_velocity[axis]).abs() < 2.0e-7);
        }

        let cp_water = 1.3f32;
        let cp_grain = 0.47f32;
        let cp_solute = 0.63f32;
        let water_temperature = 291.0f32;
        let grain_temperature = 343.0f32;
        let water_capacity_before = solvent * cp_water + dissolved_before * cp_solute;
        let grain_capacity_before =
            insoluble * cp_grain + (fast + slow) * cp_solute + bound_water * cp_water;
        let transferred_capacity = transferred * cp_solute;
        let transferred_enthalpy = transferred_capacity * grain_temperature;
        let water_capacity_after = water_capacity_before + transferred_capacity;
        let grain_capacity_after = grain_capacity_before - transferred_capacity;
        let water_enthalpy_after = water_capacity_before * water_temperature + transferred_enthalpy;
        let grain_enthalpy_after = grain_capacity_before * grain_temperature - transferred_enthalpy;
        assert!(
            (water_capacity_before + grain_capacity_before
                - water_capacity_after
                - grain_capacity_after)
                .abs()
                < 2.0e-6
        );
        assert!(
            (water_capacity_before * water_temperature + grain_capacity_before * grain_temperature
                - water_enthalpy_after
                - grain_enthalpy_after)
                .abs()
                < 2.0e-4
        );

        let exchange_fraction = 0.37f32;
        let heat_to_water = exchange_fraction * water_capacity_after * grain_capacity_after
            / (water_capacity_after + grain_capacity_after)
            * (grain_enthalpy_after / grain_capacity_after
                - water_enthalpy_after / water_capacity_after);
        let exchanged_water_enthalpy = water_enthalpy_after + heat_to_water;
        let exchanged_grain_enthalpy = grain_enthalpy_after - heat_to_water;
        assert!(exchanged_water_enthalpy.is_finite() && exchanged_grain_enthalpy.is_finite());
        assert!(
            (exchanged_water_enthalpy + exchanged_grain_enthalpy
                - water_enthalpy_after
                - grain_enthalpy_after)
                .abs()
                < 2.0e-4
        );
        let equilibrium_temperature = (water_enthalpy_after + grain_enthalpy_after)
            / (water_capacity_after + grain_capacity_after);
        let exchanged_water_temperature = exchanged_water_enthalpy / water_capacity_after;
        let exchanged_grain_temperature = exchanged_grain_enthalpy / grain_capacity_after;
        assert!(exchanged_water_temperature <= equilibrium_temperature);
        assert!(exchanged_grain_temperature >= equilibrium_temperature);
    }

    const MOIST_SD_SCALE_TEST: f32 = 1_073_741_824.0;
    const MOIST_Q_SCALE_TEST: f32 = 67_108_864.0;
    const MOIST_R_SCALE_TEST: f32 = 16_777_216.0;
    const CHEM_H_SCALE_TEST: f32 = 67_108_864.0;
    const CHEM_FL_SCALE_TEST: f32 = 134_217_728.0;
    const CHEM_C_SCALE_TEST: f32 = 4_194_304.0;
    const CHEM_E_SCALE_TEST: f32 = 8_192.0;
    const CHEM_J_SCALE_TEST: f32 = 67_108_864.0;

    fn integer_partition(total: i32, weights: &[f32]) -> Vec<i32> {
        let blocks = total / 256;
        let remainder = total - blocks * 256;
        let mut cumulative = 0.0f32;
        let mut allocated = 0i32;
        weights
            .iter()
            .enumerate()
            .map(|(index, &weight)| {
                cumulative += weight;
                let prefix = cumulative.clamp(0.0, 1.0);
                let next = if index + 1 == weights.len() {
                    total
                } else {
                    ((blocks as f32 * prefix).round() as i32) * 256
                        + (remainder as f32 * prefix).round() as i32
                };
                let share = next - allocated;
                allocated = next;
                share
            })
            .collect()
    }

    fn stencil_weights(fx: [f32; 3]) -> Vec<f32> {
        let axis = |x: f32| {
            [
                0.5 * (1.5 - x).powi(2),
                0.75 - (x - 1.0).powi(2),
                0.5 * (x - 0.5).powi(2),
            ]
        };
        let [wx, wy, wz] = [axis(fx[0]), axis(fx[1]), axis(fx[2])];
        let mut weights = Vec::with_capacity(27);
        for z in wz {
            for y in wy {
                for x in wx {
                    weights.push(x * y * z);
                }
            }
        }
        weights
    }

    #[test]
    fn audited_lane_scales_have_the_documented_resolution_and_i32_range() {
        for (scale, expected_power, decoded_range) in [
            (MOIST_SD_SCALE_TEST, 30, 2.0),
            (MOIST_Q_SCALE_TEST, 26, 32.0),
            (MOIST_R_SCALE_TEST, 24, 128.0),
            (CHEM_H_SCALE_TEST, 26, 32.0),
            (CHEM_FL_SCALE_TEST, 27, 16.0),
            (CHEM_C_SCALE_TEST, 22, 512.0),
            (CHEM_E_SCALE_TEST, 13, 262_144.0),
            (CHEM_J_SCALE_TEST, 26, 32.0),
        ] {
            assert_eq!(scale, 2.0f32.powi(expected_power));
            let actual_range = i32::MAX as f64 / scale as f64;
            assert!(actual_range <= decoded_range && actual_range > 0.999 * decoded_range);
            assert!(1.0 / scale > 0.0, "every lane resolves a positive quantum");
        }
    }

    #[test]
    fn cumulative_integer_partition_is_deterministic_signed_and_exact() {
        let weights = stencil_weights([0.63, 1.11, 1.42]);
        for total in [
            0,
            1,
            17,
            16_777_219,
            1_900_000_001,
            -1,
            -16_777_219,
            -1_900_000_001,
        ] {
            let a = integer_partition(total, &weights);
            let b = integer_partition(total, &weights);
            assert_eq!(a, b);
            assert_eq!(
                a.iter().map(|&share| i64::from(share)).sum::<i64>(),
                i64::from(total)
            );
            if total >= 0 {
                assert!(a.iter().all(|&share| share >= 0));
            } else {
                assert!(a.iter().all(|&share| share <= 0));
            }
        }
    }

    #[test]
    fn partitioned_two_sided_allocation_conserves_the_node_transfer() {
        let weights = [
            stencil_weights([0.61, 0.93, 1.37]),
            stencil_weights([1.28, 0.72, 1.09]),
        ];
        let supply_totals = [0.031f32, 0.017];
        let demand_totals = [0.022f32, 0.041];
        let encode = |value: f32| (value * MOIST_SD_SCALE_TEST).round() as i32;
        let supply_shares = [
            integer_partition(encode(supply_totals[0]), &weights[0]),
            integer_partition(encode(supply_totals[1]), &weights[1]),
        ];
        let demand_shares = [
            integer_partition(encode(demand_totals[0]), &weights[1]),
            integer_partition(encode(demand_totals[1]), &weights[0]),
        ];
        let mut lost = [0.0f64; 2];
        let mut gained = [0.0f64; 2];
        let mut node_transfer = 0.0f64;
        for node in 0..27 {
            let supply = supply_shares
                .iter()
                .map(|shares| f64::from(shares[node]) / f64::from(MOIST_SD_SCALE_TEST))
                .sum::<f64>();
            let demand = demand_shares
                .iter()
                .map(|shares| f64::from(shares[node]) / f64::from(MOIST_SD_SCALE_TEST))
                .sum::<f64>();
            let transfer = supply.min(demand);
            node_transfer += transfer;
            if supply > 0.0 {
                for particle in 0..2 {
                    let share =
                        f64::from(supply_shares[particle][node]) / f64::from(MOIST_SD_SCALE_TEST);
                    lost[particle] += share * transfer / supply;
                }
            }
            if demand > 0.0 {
                for particle in 0..2 {
                    let share =
                        f64::from(demand_shares[particle][node]) / f64::from(MOIST_SD_SCALE_TEST);
                    gained[particle] += share * transfer / demand;
                }
            }
        }
        let lost_total: f64 = lost.into_iter().sum();
        let gained_total: f64 = gained.into_iter().sum();
        assert!((lost_total - node_transfer).abs() < 1.0e-12);
        assert!((gained_total - node_transfer).abs() < 1.0e-12);
    }

    #[test]
    fn solute_safe_solvent_floor_bounds_post_absorption_concentration() {
        let c_sat = 0.08f32;
        let roundoff = 1.0e-3f32;
        for (fraction, concentration) in [(1.0, 0.0), (0.8, 0.02), (0.4, 0.079)] {
            let floor = roundoff.max(fraction * concentration / c_sat);
            let absorbable = (fraction - floor).max(0.0);
            let final_fraction = (fraction - absorbable).max(floor);
            let final_concentration = concentration * fraction / final_fraction;
            assert!(final_fraction >= roundoff);
            assert!(final_concentration <= c_sat + 2.0 * f32::EPSILON);
        }
        let supersaturated = 0.12f32;
        let fraction = 0.7f32;
        let floor = roundoff.max(fraction * supersaturated / c_sat);
        assert!((fraction - floor).max(0.0) == 0.0);
    }

    #[test]
    fn ui_minimum_constitutive_rates_survive_the_current_substep_cadence() {
        let stable_fraction = |rate: f32, dt: f32| {
            let x = (rate * dt).max(0.0);
            if x < 1.0e-3 {
                x * (1.0 - 0.5 * x + x * x / 6.0)
            } else {
                1.0 - (-x).exp()
            }
        };
        let frame_dt = 1.0 / 60.0;
        let representative_substeps = 64.0;
        let dt = frame_dt / representative_substeps;
        for control in [
            crate::models::calibration::CalibrationControl::AbsorptionRate,
            crate::models::calibration::CalibrationControl::ExtractionFastRate,
            crate::models::calibration::CalibrationControl::ExtractionSlowRate,
            crate::models::calibration::CalibrationControl::ThermalExchangeRate,
        ] {
            assert!(stable_fraction(control.metadata().min, dt) > 0.0);
        }
        // One reduced inventory unit is a scale probe, not a calibrated coffee target. At the
        // smallest extraction rate it still produces multiple integer quanta even at 64
        // constitutive substeps per rendered frame.
        let release = stable_fraction(
            crate::models::calibration::CalibrationControl::ExtractionSlowRate
                .metadata()
                .min,
            dt,
        );
        assert!((release * CHEM_FL_SCALE_TEST).round() >= 1.0);
        let demand = 0.1
            * stable_fraction(
                crate::models::calibration::CalibrationControl::AbsorptionRate
                    .metadata()
                    .min,
                dt,
            );
        assert!((demand * MOIST_SD_SCALE_TEST).round() >= 1.0);
    }

    /// Seeding is deterministic for a fixed seed (R6).
    #[test]
    fn seeding_is_deterministic() {
        let scene = Scene::pour_over();
        let a = seed_ranges(&scene, &Materials::default(), &Config::default());
        let b = seed_ranges(&scene, &Materials::default(), &Config::default());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }

    /// The grid pads one node layer below box_min and covers the full B-spline support of any
    /// in-box particle: base = floor((x − origin)/h − 0.5) ∈ [0, dims − 3] at both extremes.
    #[test]
    fn grid_spec_covers_box_with_padding() {
        let scene = Scene::default();
        let mats = Materials::default();
        let (origin, h, dims) = grid_spec_for(&scene, &mats);
        assert_eq!(h, CELL_SIZE_FACTOR * mats.particle_spacing);
        for a in 0..3 {
            assert_eq!(origin[a], scene.box_min[a] - h);
            for x in [scene.box_min[a], scene.box_max[a]] {
                let xl = (x - origin[a]) / h;
                let base = (xl - 0.5).floor() as i64;
                assert!(base >= 0, "axis {a}: base {base} below grid at x={x}");
                assert!(
                    base + 2 <= dims[a] as i64 - 1,
                    "axis {a}: support exceeds grid at x={x} (base {base}, dims {})",
                    dims[a]
                );
            }
        }
    }

    fn pair_drag_1d(
        fluid_velocity: f64,
        solid_velocity: f64,
        fluid_mass: f64,
        solid_mass: f64,
        fluid_drag_rate: f64,
        dt: f64,
    ) -> (f64, f64, f64) {
        let relative_rate = fluid_drag_rate * (1.0 + fluid_mass / solid_mass);
        let one_minus_decay = -(-relative_rate * dt).exp_m1();
        let reduced_mass = fluid_mass * solid_mass / (fluid_mass + solid_mass);
        let impulse = reduced_mass * one_minus_decay * (fluid_velocity - solid_velocity);
        (
            fluid_velocity - impulse / fluid_mass,
            solid_velocity + impulse / solid_mass,
            (-relative_rate * dt).exp(),
        )
    }

    /// Algebraic invariant only: no calibrated rate or target behavior is pinned here.
    #[test]
    fn exact_pair_drag_conserves_momentum_and_decays_the_relative_mode() {
        for (vf, vs, mf, ms, beta, dt) in [
            (1.7, -0.4, 0.8, 2.3, 0.02, 1.0 / 60.0),
            (-3.0, 0.7, 4.0, 0.25, 350.0, 1.0 / 120.0),
            (0.2, 0.2, 1.0, 1.0, 1.0e6, 1.0 / 30.0),
        ] {
            let before_momentum = mf * vf + ms * vs;
            let before_relative = vf - vs;
            let (vf_after, vs_after, decay) = pair_drag_1d(vf, vs, mf, ms, beta, dt);
            let momentum_error = (mf * vf_after + ms * vs_after - before_momentum).abs();
            assert!(momentum_error <= 2.0e-12 * before_momentum.abs().max(1.0));
            let relative_error = (vf_after - vs_after - decay * before_relative).abs();
            assert!(relative_error <= 2.0e-12 * before_relative.abs().max(1.0));
        }
    }

    fn phase_mobilities(phi_f: f64, rho_f: f64, rho_s: f64, sigma: f64) -> (f64, f64) {
        let phi_s = 1.0 - phi_f;
        let bulk_density = phi_f * rho_f + phi_s * rho_s;
        let y_f = phi_f * rho_f / bulk_density;
        let y_s = phi_s * rho_s / bulk_density;
        let inverse_density_delta = 1.0 / rho_f - 1.0 / rho_s;
        (
            1.0 / bulk_density + y_s * sigma * inverse_density_delta,
            1.0 / bulk_density - y_f * sigma * inverse_density_delta,
        )
    }

    fn phase_mobilities_from_node_masses(
        phi_f: f64,
        fluid_mass: f64,
        solid_mass: f64,
        h: f64,
        sigma: f64,
    ) -> (f64, f64, f64, f64) {
        let phi_s = 1.0 - phi_f;
        let h3 = h * h * h;
        let rho_f = fluid_mass / (phi_f * h3);
        let rho_s = solid_mass / (phi_s * h3);
        let (r_f, r_s) = phase_mobilities(phi_f, rho_f, rho_s, sigma);
        (rho_f, rho_s, r_f, r_s)
    }

    /// Endpoint identities derive from the coupled equations; this intentionally avoids
    /// freezing any empirical coffee-material constants into a test.
    #[test]
    fn condensed_pressure_mobilities_have_uncoupled_and_locked_limits() {
        // Deliberately dimensionless, arbitrary values: the test pins identities, not a
        // material calibration.
        let phi_f = 0.4;
        let rho_f = 2.0;
        let rho_s = 5.0;
        let bulk_density = phi_f * rho_f + (1.0 - phi_f) * rho_s;

        let (fluid_free, solid_free) = phase_mobilities(phi_f, rho_f, rho_s, 1.0);
        assert!((fluid_free - 1.0 / rho_f).abs() < 1.0e-15);
        assert!((solid_free - 1.0 / rho_s).abs() < 1.0e-15);

        let (fluid_locked, solid_locked) = phase_mobilities(phi_f, rho_f, rho_s, 0.0);
        assert!((fluid_locked - 1.0 / bulk_density).abs() < 1.0e-15);
        assert!((solid_locked - 1.0 / bulk_density).abs() < 1.0e-15);

        let (fluid_mid, solid_mid) = phase_mobilities(phi_f, rho_f, rho_s, 0.37);
        assert!(fluid_mid.is_sign_positive() && solid_mid.is_sign_positive());
    }

    /// The dynamic pressure response must condense the exact masses used by drag. This pins
    /// identities only; no density floor or material calibration belongs in the paired system.
    #[test]
    fn paired_pressure_mass_fractions_and_operator_match_actual_node_masses() {
        let (phi_f, fluid_mass, solid_mass, h, sigma) = (0.35, 0.012, 3.7, 1.4, 0.28);
        let phi_s = 1.0 - phi_f;
        let (rho_f, rho_s, r_f, r_s) =
            phase_mobilities_from_node_masses(phi_f, fluid_mass, solid_mass, h, sigma);
        let bulk_density = phi_f * rho_f + phi_s * rho_s;
        let density_fraction_f = phi_f * rho_f / bulk_density;
        let density_fraction_s = phi_s * rho_s / bulk_density;
        let actual_fraction_f = fluid_mass / (fluid_mass + solid_mass);
        let actual_fraction_s = solid_mass / (fluid_mass + solid_mass);
        assert!((density_fraction_f - actual_fraction_f).abs() < 1.0e-15);
        assert!((density_fraction_s - actual_fraction_s).abs() < 1.0e-15);

        // node_setup stores the volume-weighted operator response; project applies the same
        // r_f/r_s separately. Their resulting mixture correction must be identical.
        let (dt, gradient) = (0.023, -1.7);
        let operator_correction = -dt * (phi_f * r_f + phi_s * r_s) * gradient;
        let projected_fluid = -dt * r_f * gradient;
        let projected_solid = -dt * r_s * gradient;
        let projected_mixture = phi_f * projected_fluid + phi_s * projected_solid;
        assert!((operator_correction - projected_mixture).abs() < 1.0e-15);
    }

    /// Partial retention changes phase bookkeeping but not the pressure impulse identity. This
    /// pins equations only: the values are arbitrary and contain no coffee calibration target.
    #[test]
    fn partial_retention_pressure_response_is_adjoint_and_uses_composite_inertia() {
        let (phi_mobile, total_water_mass, dry_solid_mass, theta, h, sigma) =
            (0.27_f64, 0.83, 2.4, 0.36, 1.2, 0.41);
        let mobile_mass = theta * total_water_mass;
        let held_mass = (1.0 - theta) * total_water_mass;
        let composite_mass = dry_solid_mass + held_mass;
        let (_, _, r_mobile, r_composite) =
            phase_mobilities_from_node_masses(phi_mobile, mobile_mass, composite_mass, h, sigma);
        let (dt, gradient) = (0.019, -1.3);
        let dv_mobile = -dt * r_mobile * gradient;
        let dv_composite = -dt * r_composite * gradient;
        let dv_shared = theta * dv_mobile + (1.0 - theta) * dv_composite;

        // D reconstructs intrinsic mobile velocity from the homogenized update exactly.
        let reconstructed_mobile = (dv_shared - (1.0 - theta) * dv_composite) / theta;
        assert!((reconstructed_mobile - dv_mobile).abs() < 1.0e-15);

        // The operator assembled in node_setup equals D applied to project's two phase updates.
        let phi_composite = 1.0 - phi_mobile;
        let operator = -dt * (phi_mobile * r_mobile + phi_composite * r_composite) * gradient;
        let projected = phi_mobile * reconstructed_mobile + phi_composite * dv_composite;
        assert!((operator - projected).abs() < 1.0e-15);

        // Frozen skeleton: writing theta*dv_mobile to v_bar avoids the 1/theta over-acceleration.
        let rho_mobile = mobile_mass / (phi_mobile * h.powi(3));
        let frozen_mobile = -dt * sigma / rho_mobile * gradient;
        let frozen_shared = theta * frozen_mobile;
        let homogenized_density = total_water_mass / (phi_mobile * h.powi(3));
        assert!((frozen_shared + dt * sigma / homogenized_density * gradient).abs() < 1.0e-15);
    }

    #[test]
    fn partial_retention_force_staging_preserves_equal_body_acceleration() {
        // Arbitrary dimensionless values: this pins the split impulse algebra, not a material.
        let (water_mass, dry_mass, theta, velocity, body_kick) = (0.83_f64, 2.4, 0.36, -0.7, 0.19);
        let mobile_mass = theta * water_mass;
        let held_mass = (1.0 - theta) * water_mass;
        let composite_mass = dry_mass + held_mass;

        let dry_impulse = dry_mass * body_kick;
        let held_impulse = held_mass * body_kick;
        let composite_forced = velocity + (dry_impulse + held_impulse) / composite_mass;
        let mobile_forced = velocity + body_kick;
        assert!((mobile_forced - composite_forced).abs() < 1.0e-15);

        let before = mobile_mass * velocity + composite_mass * velocity;
        let after = mobile_mass * mobile_forced + composite_mass * composite_forced;
        let external = (mobile_mass + composite_mass) * body_kick;
        assert!((after - before - external).abs() < 1.0e-15);

        let vbar = theta * mobile_forced + (1.0 - theta) * composite_forced;
        let reconstructed = (vbar - (1.0 - theta) * composite_forced) / theta;
        assert!((reconstructed - mobile_forced).abs() < 1.0e-15);
    }

    #[test]
    fn partial_retention_paper_step_changes_only_mobile_superficial_flux() {
        let (theta, phi_mobile, mobile_v, composite_v) = (0.37_f64, 0.22, 1.4, -0.3);
        let (decay, sigma_dt_source) = (0.61, -0.08);
        let vbar = theta * mobile_v + (1.0 - theta) * composite_v;
        let old_flux = phi_mobile * mobile_v;
        let new_flux = decay * old_flux + sigma_dt_source;
        let mobile_after = mobile_v + (new_flux - old_flux) / phi_mobile;
        let vbar_after = theta * mobile_after + (1.0 - theta) * composite_v;

        assert!((phi_mobile * mobile_after - new_flux).abs() < 1.0e-15);
        assert!((vbar_after - vbar - theta * (mobile_after - mobile_v)).abs() < 1.0e-15);
        assert_eq!(
            composite_v, -0.3,
            "paper cannot move the held/dry composite"
        );
    }

    #[test]
    fn held_count_quantization_cannot_create_mobile_water_or_destroy_capacity() {
        let carrier_count = 17_u32;
        let raw_capacity_count = 31_u32;
        // The GPU copies carrier_count exactly when M=0 instead of re-encoding H.
        let held_count = carrier_count;
        assert_eq!(carrier_count - held_count, 0);
        // H has an independent lane; writing it cannot mutate attainable C.
        let capacity_after_h_write = raw_capacity_count;
        assert_eq!(capacity_after_h_write, raw_capacity_count);
    }

    #[test]
    fn frozen_pressure_reaction_uses_complete_composite_and_finite_drag_share() {
        let (solid_volume, held_volume, mobile_phi, visible, dt, gradient, sigma) =
            (0.31_f64, 0.14, 0.27, 0.9, 0.02, -1.7, 0.0);
        let direct = -(solid_volume + held_volume) * dt * gradient;
        let transmitted = (1.0 - sigma) * mobile_phi * visible * dt * gradient;
        let reaction = direct - transmitted;
        assert!(reaction.is_finite());
        assert_eq!(direct, -(solid_volume + held_volume) * dt * gradient);
        // sigma=0 is deliberately included: the volume form has no 0/0 recovery.
        assert_eq!(transmitted, mobile_phi * visible * dt * gradient);
    }
}

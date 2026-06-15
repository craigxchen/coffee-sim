//! Twofield U3 gates: incompressibility = coarse-grid pressure seed + fine local Jacobi
//! cleanup on ONE discretely consistent operator family (no converged Poisson) — plan
//! 2026-06-09-001 U3 / KTD-2.
//!
//! Operator family under test (construction documented in `pressure.wgsl`): pressure lives at
//! CELL CENTERS, velocity at NODES (the U2 collocated grid). D = corner-trilinear (FV)
//! divergence of a cell's 8 corner-node velocities, row-weighted by the cell fill fraction
//! f ∈ [0, 1] from grid mass (D_f = F·D — the ghost-fluid-style free-surface taper; interior
//! rows have f = 1 exactly); G = −D_fᵀ BY CONSTRUCTION; empty cells (f = 0) and wall-Neumann
//! conditions enter through the masks and the per-node constrained inverse-mass matrix M̃⁻¹;
//! the Laplacian exists ONLY as the composition A = −D_f·M̃⁻¹·(−D_fᵀ) — never a hand-written
//! stencil. That hand-written-stencil shortcut is exactly how main got A ≠ D·G and an
//! unbeatable divergence floor; these gates exist to make that unshippable.
//! The SURF_FULL_FRAC/SURF_MIN_CORNER fill-fraction constants are fixed structural constants
//! of the family (like JACOBI_OMEGA, documented in pressure.wgsl/mod.rs), not knob-grid
//! entries.
//!
//! CPU twins of D, G, M̃⁻¹, A, R, P and the damped-Jacobi sweep live at the bottom of this
//! file (f64). The adjointness/MMS/SPD gates run on the twins AND on GPU readbacks (via the
//! test-only `dbg_div`/`dbg_grad` entry points + the nm/cell_meta readbacks).
//!
//! KNOB GRID (KTD-9) — every tunable these gates depend on; exhausting it without a pass IS
//! the halt condition, and reaching for any knob outside it is a redesign decision:
//!   coarse ratio          {4, 8}        (fine cells per coarse cell per axis; default 4)
//!   coarse Jacobi sweeps  {4, 8, 16}    (default 8; 0 exists only as the A/B seed-off arm)
//!   fine Jacobi sweeps    {4, 8, 16}    (default 8)
//!   adaptive-Tait         {off}         (stub decision — documented in twofield/mod.rs)
//!   divergence tolerance  DIV_TOL       (pre-registered below; never tuned to results)

// Axis loops (`for a in 0..3`) index parallel arrays; the iterator rewrite obscures that.
#![allow(clippy::needless_range_loop)]

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{
    dispatches_per_frame_for, u4_surface_dispatches_for, TwofieldSolver, COARSE_RATIO_DEFAULT,
    COARSE_SWEEPS_DEFAULT, FINE_SWEEPS_DEFAULT, JACOBI_OMEGA, MAX_STORAGE_BUFFERS_PER_ENTRY_POINT,
    U3_PRESSURE_DISPATCHES, U6_COUPLING_DISPATCHES,
};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::rng::Rng;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;
const SPACING: f32 = 1.0; // Materials::default().particle_spacing

// --- knob grid (KTD-9), mirrored from the header --------------------------------------------
const KNOB_COARSE_RATIOS: [u32; 2] = [4, 8];
const KNOB_COARSE_SWEEPS: [u32; 3] = [4, 8, 16];
const KNOB_FINE_SWEEPS: [u32; 3] = [4, 8, 16];
const KNOB_TAIT_PREDICTOR: [bool; 1] = [false]; // stubbed off (documented in twofield/mod.rs)

/// Pre-registered post-projection RMS-divergence tolerance (interior cells, units 1/s).
/// Derivation, fixed BEFORE any results were looked at: the visible-error budget is the xpbd
/// settled-tank density band of ±5%. Mass conservation ties density drift to divergence,
/// ρ̇/ρ = −∇·v, so a sustained interior divergence d compounds density by d·dt per frame.
/// Requiring each frame's increment ≤ 0.5% — 10× inside the band, so ~10 frames of coherent
/// one-signed residual still stay inside the envelope — gives d ≤ 0.005/dt = 0.005·60 = 0.3.
const DIV_TOL: f64 = 0.3;

/// Divergence-decay probe budgets. 32 exceeds the knob grid and is printed for curve context
/// only — the tolerance must be reached at a fine-sweep budget ≤ 16 (the grid's edge).
const DECAY_BUDGETS: [u32; 4] = [4, 8, 16, 32];
/// Required error-reduction factor per doubling of the fine-sweep budget (plan: ≥2×) until
/// the curve falls below DIV_TOL. A plateau above tolerance FAILS (main's A≠D·G signature).
const DECAY_FACTOR_MIN: f64 = 2.0;

/// Interior = active cells whose whole Chebyshev-2 cell neighbourhood is active (≥2 cells
/// from any surface/wall mask boundary — surface cells are gated in U4, not here).
const INTERIOR_MARGIN: i64 = 2;

// --- independent volume gates (pre-registered bands) -----------------------------------------
/// Settled-tank water-height drift over 600 steps, in particle spacings.
const HEIGHT_DRIFT_TOL: f64 = 0.5;
/// Deep-interior mean node-density drift over 600 steps (the xpbd visible band).
const DENSITY_DRIFT_TOL: f64 = 0.05;
/// |net volumetric flux| through the closed box walls (units³/s). The settled tank holds
/// ≈3.5k units³, so 2.0 units³/s ≈ 0.06%/s — far below anything visible, far above noise.
const NET_WALL_FLUX_TOL: f64 = 2.0;

// --- hydrostatic profile gate (pre-registered) ------------------------------------------------
const HYDRO_R2_MIN: f64 = 0.90;
/// Fitted |dp/dy| must sit within this band around ρ_rest·|g|.
const HYDRO_SLOPE_BAND: (f64, f64) = (0.6, 1.4);

// --- long-run gate (pre-registered) -----------------------------------------------------------
const LONGRUN_STEPS: u32 = 2000;
const LONGRUN_MAX_SPEED: f32 = 5.0;
const LONGRUN_POP_RISE: f32 = 1.0; // max particle-y rise in spacings (popcorn guard)

fn water_scene(
    box_min: [f32; 3],
    box_max: [f32; 3],
    gravity: [f32; 3],
    regions: &[([f32; 3], [f32; 3])],
) -> Scene {
    Scene {
        gravity,
        box_min,
        box_max,
        regions: regions
            .iter()
            .map(|&(min, max)| SeedRegion {
                min,
                max,
                species: Species::Water,
            })
            .collect(),
        solids: Vec::new(),
        ..Scene::default()
    }
}

/// Settled-tank scene: full-width water so the only free surface is the top.
/// Box 16×32×16, water depth `depth` (units). ρ_rest = 1, |g| = 20.
///
/// Seeded AT REST DENSITY: `seed_ranges` places a lattice with INCLUSIVE bounds (n+1 layers
/// across n spacings), so a region flush with the walls seeds ≈9% over-dense with doubled
/// wall layers — the relief then (correctly) expands the column toward its true rest volume
/// for hundreds of frames, which a "settled tank stays settled" gate would misread as drift.
/// The half-spacing inset yields exactly one particle per spacing³ with the standard
/// half-spacing wall gap, surface at y = `depth`.
fn tank_scene(depth: f32) -> Scene {
    water_scene(
        [0.0; 3],
        [16.0, 32.0, 16.0],
        [0.0, -20.0, 0.0],
        &[([0.5, 0.5, 0.5], [15.6, depth - 0.4, 15.6])],
    )
}

/// Dam-break scene: a fat column against one wall (rest-density inset seeding, see
/// `tank_scene`). The decay gate measures at frame 45 — the column is collapsing fast
/// (divergence-rich, broadband) but still thick enough to hold ≥50 interior (margin-2)
/// cells; by ~frame 60 the splash sheet is thinner than the margin. The box is 24 deep so
/// the margin-2 interior keeps a z-extent of several cells mid-splash.
fn dam_scene() -> Scene {
    water_scene(
        [0.0; 3],
        [32.0, 24.0, 24.0],
        [0.0, -20.0, 0.0],
        &[([0.5, 0.5, 0.5], [13.6, 15.6, 23.6])],
    )
}

fn all_finite(rows: &[[f32; 4]]) -> bool {
    rows.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

fn runif(rng: &mut Rng) -> f64 {
    rng.next_f32() as f64 * 2.0 - 1.0
}

// ==============================================================================================
// CPU-only operator gates (no GPU needed; f64 twins, machine precision)
// ==============================================================================================

/// The chosen defaults must be points of the declared knob grid (KTD-9 sanity).
#[test]
fn knob_grid_contains_the_chosen_defaults() {
    assert!(KNOB_COARSE_RATIOS.contains(&COARSE_RATIO_DEFAULT));
    assert!(KNOB_COARSE_SWEEPS.contains(&COARSE_SWEEPS_DEFAULT));
    assert!(KNOB_FINE_SWEEPS.contains(&FINE_SWEEPS_DEFAULT));
    assert!(KNOB_TAIT_PREDICTOR.contains(&false), "Tait ships OFF");
    const { assert!(DIV_TOL > 0.0) };
}

/// Discrete adjointness u·(Gp) = −(Du)·p to machine precision for random fields on three
/// masked domains (closed tank, sloped wall, hole-masked interior), plus the same identity
/// for the R/P pair (⟨Ru, v⟩_c = ⟨u, Pv⟩_f with volume weighting).
#[test]
fn cpu_adjointness_d_g_and_r_p_on_masked_domains() {
    let mut rng = Rng::new(0x00AD_01A7);
    for (name, tw) in cpu_domains() {
        let u: Vec<[f64; 3]> = (0..tw.num_nodes())
            .map(|_| [runif(&mut rng), runif(&mut rng), runif(&mut rng)])
            .collect();
        let p: Vec<f64> = (0..tw.num_cells()).map(|_| runif(&mut rng)).collect();
        let du = tw.div(&u);
        let gp = tw.grad(&p);
        let lhs: f64 = u
            .iter()
            .zip(&gp)
            .map(|(a, b)| a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            .sum();
        let rhs: f64 = -du.iter().zip(&p).map(|(a, b)| a * b).sum::<f64>();
        let scale = du.iter().map(|d| d.abs()).sum::<f64>().max(1.0);
        assert!(
            (lhs - rhs).abs() <= 1e-12 * scale,
            "{name}: adjointness broke: u·Gp {lhs} vs −Du·p {rhs}"
        );

        // R/P transpose pair at both knob-grid ratios.
        for &r in &KNOB_COARSE_RATIOS {
            let ct = tw.coarsen(r as usize);
            let f: Vec<f64> = (0..tw.num_cells()).map(|_| runif(&mut rng)).collect();
            let vc: Vec<f64> = (0..ct.num_cells()).map(|_| runif(&mut rng)).collect();
            let rf = tw.restrict(&ct, r as usize, &f);
            let pv = tw.prolong(&ct, r as usize, &vc);
            // ⟨Rf, v⟩_c·Vc = ⟨f, Pv⟩_f·Vf with Vc = r³·Vf → r³·Σ(Rf)v = Σ f·(Pv).
            let lhs = (r as f64).powi(3) * rf.iter().zip(&vc).map(|(a, b)| a * b).sum::<f64>();
            let rhs: f64 = f.iter().zip(&pv).map(|(a, b)| a * b).sum();
            let scale = rhs.abs().max(1.0);
            assert!(
                (lhs - rhs).abs() <= 1e-12 * scale,
                "{name} ratio {r}: R/P adjointness broke: {lhs} vs {rhs}"
            );
        }
    }
}

/// MMS order-of-accuracy on D and G: smooth manufactured fields, observed log-log error slope
/// matches the formal second order of the corner-trilinear pair.
#[test]
fn cpu_mms_order_of_accuracy_d_and_g() {
    let l = 16.0f64;
    let mut errs_d = Vec::new();
    let mut errs_g = Vec::new();
    let mut hs = Vec::new();
    for n in [8usize, 16, 32] {
        let h = l / n as f64;
        let tw = Twin::full_box([n, n, n], h);
        // Node samples of the manufactured velocity (uploaded everywhere, so D sees no
        // stencil truncation); analytic divergence compared at active cell centers.
        let u: Vec<[f64; 3]> = (0..tw.num_nodes()).map(|i| mms_u(tw.node_pos(i))).collect();
        let du = tw.div(&u);
        let mut se = 0.0;
        let mut cnt = 0usize;
        for c in 0..tw.num_cells() {
            if tw.active[c] {
                let e = du[c] - mms_div(tw.cell_center(c));
                se += e * e;
                cnt += 1;
            }
        }
        errs_d.push((se / cnt as f64).sqrt());

        let p: Vec<f64> = (0..tw.num_cells())
            .map(|i| mms_p(tw.cell_center(i)))
            .collect();
        let gp = tw.grad(&p);
        let mut se = 0.0;
        let mut cnt = 0usize;
        for nidx in 0..tw.num_nodes() {
            // Only nodes with a complete (8-cell) active neighbourhood — G truncates at masks.
            if tw.node_fully_surrounded(nidx) {
                let g = mms_gradp(tw.node_pos(nidx));
                for a in 0..3 {
                    let e = gp[nidx][a] - g[a];
                    se += e * e;
                }
                cnt += 3;
            }
        }
        errs_g.push((se / cnt as f64).sqrt());
        hs.push(h);
    }
    for (label, errs) in [("D", &errs_d), ("G", &errs_g)] {
        for i in 1..errs.len() {
            let slope = (errs[i - 1] / errs[i]).log2() / (hs[i - 1] / hs[i]).log2();
            println!(
                "twofield U3 CPU MMS {label}: h {} -> {}: err {:.3e} -> {:.3e}, slope {slope:.2}",
                hs[i - 1],
                hs[i],
                errs[i - 1],
                errs[i]
            );
            assert!(
                (1.7..=2.35).contains(&slope),
                "{label}: observed order {slope:.2} not ~2"
            );
        }
    }
}

/// The ASSEMBLED A = −D·M̃⁻¹·G: symmetric, positive on the Dirichlet-bounded masked domains,
/// rows summing consistently with the masks (zero deep in the interior, positive next to the
/// implicit-Dirichlet surface).
#[test]
fn cpu_assembled_a_spd_and_row_sums() {
    let mut rng = Rng::new(0x5D_D0);
    for (name, tw) in cpu_domains() {
        let x: Vec<f64> = (0..tw.num_cells())
            .map(|c| if tw.active[c] { runif(&mut rng) } else { 0.0 })
            .collect();
        let y: Vec<f64> = (0..tw.num_cells())
            .map(|c| if tw.active[c] { runif(&mut rng) } else { 0.0 })
            .collect();
        let ax = tw.apply_a(&x);
        let ay = tw.apply_a(&y);
        let xay: f64 = x.iter().zip(&ay).map(|(a, b)| a * b).sum();
        let yax: f64 = y.iter().zip(&ax).map(|(a, b)| a * b).sum();
        let scale = xay.abs().max(yax.abs()).max(1.0);
        assert!(
            (xay - yax).abs() <= 1e-12 * scale,
            "{name}: A not symmetric: x·Ay {xay} vs y·Ax {yax}"
        );
        let xax: f64 = x.iter().zip(&ax).map(|(a, b)| a * b).sum();
        assert!(xax > 0.0, "{name}: A not positive: x·Ax = {xax}");

        // Row sums = A·1 on the active set: "consistent with masks" means EXACTLY zero deep
        // in the interior (a locally constant field is in the null space there — constants
        // produce no gradient at fully-surrounded free nodes), and nonzero Dirichlet leakage
        // near the mask boundary. (A is PSD but NOT an M-matrix: oblique wall projectors give
        // small signed row sums near sloped walls — that is legitimate, so no sign claim off
        // the boundary; SPD-ness is what the symmetry/positivity checks above pin.)
        let ones: Vec<f64> = tw
            .active
            .iter()
            .map(|&a| if a { 1.0 } else { 0.0 })
            .collect();
        let a1 = tw.apply_a(&ones);
        let row_scale = tw.diag_a().iter().cloned().fold(0.0f64, f64::max);
        let mut deep_max = 0.0f64;
        let mut boundary_pos = false;
        for c in 0..tw.num_cells() {
            if !tw.active[c] {
                continue;
            }
            assert!(
                a1[c].abs() <= 8.0 * row_scale,
                "{name}: wild row sum {} at {c}",
                a1[c]
            );
            if tw.cell_is_interior(c, INTERIOR_MARGIN) && tw.cell_nodes_free(c, INTERIOR_MARGIN) {
                deep_max = deep_max.max(a1[c].abs());
            } else if a1[c] > 1e-9 {
                boundary_pos = true;
            }
        }
        assert!(
            deep_max <= 1e-12,
            "{name}: interior row sums must vanish (constant in local null space): {deep_max}"
        );
        assert!(
            boundary_pos,
            "{name}: no Dirichlet leakage found — masks not exercised"
        );
    }
}

/// Two-grid convergence factor on masked domains: the (coarse seed + fine sweeps) cycle beats
/// fine-only at equal cost on the low-frequency mode (the unreachable-by-local-iteration mode
/// that motivated the coarse seed). High-frequency factor printed for the record.
#[test]
fn cpu_two_grid_beats_fine_only_on_the_low_mode() {
    for (name, tw) in cpu_domains() {
        let lo = tw.mode_low();
        let hi = tw.mode_high();
        for (mode_name, mode) in [("low", &lo), ("high", &hi)] {
            let rhs = tw.apply_a(mode);
            let norm0 = rms_masked(mode, &tw.active);
            // Seeded cycle at the defaults.
            let p_seeded = two_grid_cycle(
                &tw,
                COARSE_RATIO_DEFAULT as usize,
                COARSE_SWEEPS_DEFAULT,
                FINE_SWEEPS_DEFAULT,
                &rhs,
            );
            // Fine-only at equal cost: coarse work is ≤ Nc/ratio³ fine-sweep equivalents plus
            // the two transfers — one extra fine sweep over-pays for it.
            let mut p_fine = vec![0.0; tw.num_cells()];
            for _ in 0..(FINE_SWEEPS_DEFAULT + 1) {
                p_fine = tw.jacobi(&p_fine, &rhs);
            }
            let e_seeded = rms_err(&p_seeded, mode, &tw.active) / norm0;
            let e_fine = rms_err(&p_fine, mode, &tw.active) / norm0;
            println!(
                "twofield U3 two-grid [{name}/{mode_name}]: seeded {e_seeded:.4} vs fine-only {e_fine:.4}"
            );
            if mode_name == "low" {
                assert!(
                    e_seeded < e_fine,
                    "{name}: coarse seed must beat fine-only on the low mode \
                     (seeded {e_seeded:.4} vs fine-only {e_fine:.4})"
                );
            }
        }
    }
}

// ==============================================================================================
// GPU operator gates (twin pins + adjointness on readbacks)
// ==============================================================================================

/// GPU D/G match the f64 twins on masked scenes (closed tank, V60 cone slope, partially
/// masked interior), and the GPU outputs satisfy the adjointness identity directly.
#[test]
fn gpu_operators_match_twins_and_are_adjoint() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let mut rng = Rng::new(0x6B_0B5);
    for (name, scene) in gpu_mask_scenes() {
        let mut solver =
            TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
        solver.step(DT, &EmissionInput::default()); // populates masks + nm
        let tw = twin_from_gpu(&solver);
        let n_active = tw.active.iter().filter(|&&a| a).count();
        assert!(
            n_active > 20,
            "{name}: scene produced only {n_active} active cells"
        );

        // Random node field u: GPU dbg_div vs twin D, then the dot-product identity.
        let u: Vec<[f32; 4]> = (0..tw.num_nodes())
            .map(|_| {
                [
                    runif(&mut rng) as f32,
                    runif(&mut rng) as f32,
                    runif(&mut rng) as f32,
                    0.0,
                ]
            })
            .collect();
        solver.write_grid_velocities_for_test(&u);
        solver.run_div_for_test();
        let du_gpu: Vec<f64> = solver
            .read_cell_meta()
            .iter()
            .map(|m| m[2] as f64)
            .collect();
        let u64v: Vec<[f64; 3]> = u
            .iter()
            .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
            .collect();
        let du_twin = tw.div(&u64v);
        for c in 0..tw.num_cells() {
            assert!(
                (du_gpu[c] - du_twin[c]).abs() <= 2e-4 + 1e-4 * du_twin[c].abs(),
                "{name}: GPU D vs twin at cell {c}: {} vs {}",
                du_gpu[c],
                du_twin[c]
            );
        }

        // Random cell field p: GPU dbg_grad vs twin G.
        let p: Vec<f32> = (0..tw.num_cells())
            .map(|_| runif(&mut rng) as f32)
            .collect();
        solver.write_pressure_for_test(&p);
        solver.run_grad_for_test();
        let gp_gpu = solver.read_grid_velocities();
        let p64: Vec<f64> = p.iter().map(|&x| x as f64).collect();
        let gp_twin = tw.grad(&p64);
        for n in 0..tw.num_nodes() {
            for a in 0..3 {
                assert!(
                    (gp_gpu[n][a] as f64 - gp_twin[n][a]).abs()
                        <= 2e-4 + 1e-4 * gp_twin[n][a].abs(),
                    "{name}: GPU G vs twin at node {n} axis {a}: {} vs {}",
                    gp_gpu[n][a],
                    gp_twin[n][a]
                );
            }
        }

        // Adjointness on the GPU readbacks themselves: u·(Gp) = −(Du)·p (f32 budget).
        let lhs: f64 = u64v
            .iter()
            .zip(&gp_gpu)
            .map(|(a, b)| a[0] * b[0] as f64 + a[1] * b[1] as f64 + a[2] * b[2] as f64)
            .sum();
        let rhs: f64 = -du_gpu.iter().zip(&p64).map(|(a, b)| a * b).sum::<f64>();
        let scale = du_gpu.iter().map(|d| d.abs()).sum::<f64>().max(1.0);
        assert!(
            (lhs - rhs).abs() <= 1e-3 * scale,
            "{name}: GPU adjointness broke: {lhs} vs {rhs} (scale {scale})"
        );
        println!(
            "twofield U3 GPU operators [{name}]: {n_active} active cells, twin-pinned + adjoint"
        );
    }
}

/// The assembled GPU A = −D·M̃⁻¹·G (D and G straight from the GPU kernels, M̃⁻¹ from the
/// GPU-read nm buffer) is symmetric and positive on a real masked scene.
#[test]
fn gpu_assembled_a_symmetric_and_positive() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let scene = tank_scene(12.0);
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    solver.step(DT, &EmissionInput::default());
    let tw = twin_from_gpu(&solver);
    let mut rng = Rng::new(0xA5_A5);
    let mk = |rng: &mut Rng| -> Vec<f32> {
        (0..tw.num_cells())
            .map(|c| if tw.active[c] { runif(rng) as f32 } else { 0.0 })
            .collect()
    };
    let x = mk(&mut rng);
    let y = mk(&mut rng);
    // A z via the pipeline: dbg_grad (GPU G) → CPU M̃⁻¹ (from the GPU nm readback) → upload →
    // dbg_div (GPU D) → negate. The Laplacian is the composition, exactly as in the solve.
    let apply_a_gpu = |z: &[f32]| -> Vec<f64> {
        solver.write_pressure_for_test(z);
        solver.run_grad_for_test();
        let g = solver.read_grid_velocities();
        let mg: Vec<[f32; 4]> = (0..tw.num_nodes())
            .map(|n| {
                let v = tw.minv(n, [g[n][0] as f64, g[n][1] as f64, g[n][2] as f64]);
                [v[0] as f32, v[1] as f32, v[2] as f32, 0.0]
            })
            .collect();
        solver.write_grid_velocities_for_test(&mg);
        solver.run_div_for_test();
        solver
            .read_cell_meta()
            .iter()
            .map(|m| -(m[2] as f64))
            .collect()
    };
    let ax = apply_a_gpu(&x);
    let ay = apply_a_gpu(&y);
    let dot = |a: &[f32], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| *x as f64 * y).sum() };
    let xay = dot(&x, &ay);
    let yax = dot(&y, &ax);
    let scale = xay.abs().max(yax.abs()).max(1.0);
    assert!(
        (xay - yax).abs() <= 2e-3 * scale,
        "GPU assembled A not symmetric: x·Ay {xay} vs y·Ax {yax}"
    );
    let xax = dot(&x, &ax);
    assert!(xax > 0.0, "GPU assembled A not positive: x·Ax {xax}");
    println!("twofield U3 GPU assembled A: x·Ay {xay:.4} = y·Ax {yax:.4}, x·Ax {xax:.4} > 0");
}

/// MMS order-of-accuracy on the GPU D and G kernels across three grid resolutions
/// (particle spacing 2.0 / 1.0 / 0.5 → h = 4 / 2 / 1 on a fixed 24³ box).
#[test]
fn gpu_mms_order_d_and_g() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let mut errs_d = Vec::new();
    let mut errs_g = Vec::new();
    let mut hs = Vec::new();
    for spacing in [2.0f32, 1.0, 0.5] {
        let scene = water_scene(
            [0.0; 3],
            [24.0; 3],
            [0.0; 3],
            &[([0.0; 3], [24.0; 3])], // fill the whole box so the masks are full
        );
        let mats = Materials {
            particle_spacing: spacing,
            ..Materials::default()
        };
        let mut solver = TwofieldSolver::build(&scene, &mats, &Config::default(), &gpu);
        solver.step(DT, &EmissionInput::default());
        let tw = twin_from_gpu(&solver);
        let u: Vec<[f32; 4]> = (0..tw.num_nodes())
            .map(|n| {
                let m = mms_u(tw.node_pos(n));
                [m[0] as f32, m[1] as f32, m[2] as f32, 0.0]
            })
            .collect();
        solver.write_grid_velocities_for_test(&u);
        solver.run_div_for_test();
        let meta = solver.read_cell_meta();
        let mut se = 0.0;
        let mut cnt = 0usize;
        for c in 0..tw.num_cells() {
            // Full rows only (f = 1): the fraction-tapered surface rows are first-order by
            // design (gated in U4); the order claim is for the interior family.
            if tw.frac[c] >= 0.999 {
                let e = meta[c][2] as f64 - mms_div(tw.cell_center(c));
                se += e * e;
                cnt += 1;
            }
        }
        assert!(cnt > 20, "spacing {spacing}: too few full cells ({cnt})");
        errs_d.push((se / cnt as f64).sqrt());

        let p: Vec<f32> = (0..tw.num_cells())
            .map(|c| mms_p(tw.cell_center(c)) as f32)
            .collect();
        solver.write_pressure_for_test(&p);
        solver.run_grad_for_test();
        let gp = solver.read_grid_velocities();
        let mut se = 0.0;
        let mut cnt = 0usize;
        for n in 0..tw.num_nodes() {
            if tw.node_fully_surrounded(n) {
                let g = mms_gradp(tw.node_pos(n));
                for a in 0..3 {
                    let e = gp[n][a] as f64 - g[a];
                    se += e * e;
                }
                cnt += 3;
            }
        }
        errs_g.push((se / cnt as f64).sqrt());
        hs.push(tw.h);
    }
    for (label, errs) in [("D", &errs_d), ("G", &errs_g)] {
        for i in 1..errs.len() {
            let slope = (errs[i - 1] / errs[i]).log2() / (hs[i - 1] / hs[i]).log2();
            println!(
                "twofield U3 GPU MMS {label}: h {} -> {}: err {:.3e} -> {:.3e}, slope {slope:.2}",
                hs[i - 1],
                hs[i],
                errs[i - 1],
                errs[i]
            );
            assert!(
                (1.7..=2.35).contains(&slope),
                "GPU {label}: observed order {slope:.2} not ~2"
            );
        }
    }
}

// ==============================================================================================
// Divergence decay (the rung's reason to exist) + A/B + hydrostatic
// ==============================================================================================

struct Snapshot {
    pos: Vec<[f32; 4]>,
    vel: Vec<[f32; 4]>,
    cmat: Vec<[f32; 4]>,
}

fn snapshot(solver: &TwofieldSolver) -> Snapshot {
    Snapshot {
        pos: solver.read_positions(),
        vel: solver.read_velocities(),
        cmat: solver.read_affine_rows(),
    }
}

fn restore(solver: &TwofieldSolver, s: &Snapshot) {
    solver.write_positions_for_test(&s.pos);
    solver.write_velocities_for_test(&s.vel);
    solver.write_affine_for_test(&s.cmat);
}

/// Post-projection RMS divergence over interior cells, via the projection's own D (the
/// twin is pinned to the GPU D by `gpu_operators_match_twins_and_are_adjoint`). U4 pocket
/// cells (cell_meta.w = 2: enclosed air under the AGGREGATE bubble constraint) are excluded
/// from the census, exactly as air cells always were — only fluid rows promise per-cell
/// divergence-freedom; this preserves, not loosens, the U3 census semantics.
fn rms_interior_div(solver: &TwofieldSolver) -> (f64, usize) {
    let tw = twin_from_gpu(solver);
    let pocket: Vec<bool> = solver.read_cell_meta().iter().map(|m| m[3] > 1.5).collect();
    let gv = solver.read_grid_velocities();
    let u: Vec<[f64; 3]> = gv
        .iter()
        .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .collect();
    let div = tw.div(&u);
    let mut se = 0.0;
    let mut cnt = 0usize;
    for c in 0..tw.num_cells() {
        if tw.active[c] && !pocket[c] && tw.cell_is_interior(c, INTERIOR_MARGIN) {
            se += div[c] * div[c];
            cnt += 1;
        }
    }
    ((se / cnt.max(1) as f64).sqrt(), cnt)
}

/// Run the decay probe: advance a donor with the defaults, snapshot the particle state, then
/// for each fine-sweep budget rebuild, restore, take ONE step (identical pre-projection state
/// by fixed-point determinism), and measure post-projection interior RMS divergence.
fn decay_curve(gpu: &GpuContext, scene: &Scene, warm_frames: u32) -> Vec<(u32, f64, usize)> {
    let mats = Materials::default();
    let cfg = Config::default();
    let mut donor = TwofieldSolver::build(scene, &mats, &cfg, gpu);
    let input = EmissionInput::default();
    for _ in 0..warm_frames {
        donor.step(DT, &input);
    }
    let snap = snapshot(&donor);
    DECAY_BUDGETS
        .iter()
        .map(|&nf| {
            let mut s = TwofieldSolver::build(scene, &mats, &cfg, gpu);
            restore(&s, &snap);
            s.set_pressure_budget_for_test(COARSE_RATIO_DEFAULT, COARSE_SWEEPS_DEFAULT, nf);
            s.step(DT, &input);
            let (rms, cells) = rms_interior_div(&s);
            (nf, rms, cells)
        })
        .collect()
}

/// Assert the pre-registered decay gate on a measured curve: ≥2× per doubling until below
/// DIV_TOL; tolerance must be reached within the knob grid (fine sweeps ≤ 16); a plateau
/// above tolerance fails.
fn assert_decay_gate(label: &str, curve: &[(u32, f64, usize)]) {
    for &(nf, rms, cells) in curve {
        println!("twofield U3 divergence decay [{label}]: fine sweeps {nf:2} -> RMS {rms:.4e} ({cells} interior cells)");
        assert!(
            cells >= 50,
            "{label}: only {cells} interior cells — gate would be vacuous"
        );
        assert!(rms.is_finite(), "{label}: non-finite divergence");
    }
    let mut reached: Option<u32> = None;
    for i in 0..curve.len() {
        let (nf, rms, _) = curve[i];
        if rms < DIV_TOL {
            reached = Some(nf);
            break;
        }
        // Still above tolerance: the next doubling must cut it by ≥ DECAY_FACTOR_MIN.
        assert!(
            i + 1 < curve.len(),
            "{label}: divergence never fell below the pre-registered DIV_TOL {DIV_TOL} \
             (last RMS {rms:.4e} at {nf} sweeps) — plateau above tolerance FAILS"
        );
        let (nf2, rms2, _) = curve[i + 1];
        let factor = rms / rms2;
        assert!(
            factor >= DECAY_FACTOR_MIN || rms2 < DIV_TOL,
            "{label}: decay factor {factor:.2} from {nf} -> {nf2} sweeps below the required \
             {DECAY_FACTOR_MIN} while still above tolerance — the A≠D·G plateau signature"
        );
    }
    let reached = reached.expect("checked above");
    assert!(
        reached <= 16,
        "{label}: tolerance reached only at {reached} sweeps — outside the declared knob grid"
    );
}

/// Divergence decay on a settled static tank. Depth 20 (not 12): the interior-cell census
/// requires margin-2 FULL (f = 1) cells, and the corrected surface classification no longer
/// counts smear/surface cells as full, so the shallower tank holds < 50 interior cells and
/// would trip the vacuousness guard.
#[test]
fn divergence_decay_static_tank() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let curve = decay_curve(&gpu, &tank_scene(20.0), 240);
    assert_decay_gate("static tank", &curve);
}

/// Divergence decay mid-splash in a dam break (the broadband hard case).
#[test]
fn divergence_decay_dam_break_midsplash() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let curve = decay_curve(&gpu, &dam_scene(), 45);
    assert_decay_gate("dam break", &curve);
}

/// A/B coarse seed on vs off at one-step-equal state: seed-on reaches a lower divergence at
/// equal fine sweeps AND matches/beats seed-off at double the fine sweeps (materially fewer
/// sweeps for the same tolerance); plus the settled-tank pressure profile is hydrostatic
/// (linear in depth, slope ≈ ρ_rest·|g|).
#[test]
fn coarse_seed_ab_and_hydrostatic_profile() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let scene = tank_scene(20.0); // deep: the hydrostatic mode is the low-frequency content
    let mats = Materials::default();
    let cfg = Config::default();
    let input = EmissionInput::default();
    let mut donor = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..240 {
        donor.step(DT, &input);
    }
    let snap = snapshot(&donor);

    let measure = |coarse_sweeps: u32, nf: u32| -> f64 {
        let mut s = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
        restore(&s, &snap);
        s.set_pressure_budget_for_test(COARSE_RATIO_DEFAULT, coarse_sweeps, nf);
        s.step(DT, &input);
        rms_interior_div(&s).0
    };
    let on_8 = measure(COARSE_SWEEPS_DEFAULT, 8);
    let off_8 = measure(0, 8);
    let off_16 = measure(0, 16);
    println!(
        "twofield U3 A/B coarse seed: on@8 {on_8:.4e} | off@8 {off_8:.4e} | off@16 {off_16:.4e}"
    );
    assert!(
        on_8 < off_8,
        "coarse seed must reduce divergence at equal fine sweeps (on {on_8:.4e} vs off {off_8:.4e})"
    );
    assert!(
        on_8 <= off_16,
        "coarse seed must be worth at least a doubling of fine sweeps \
         (on@8 {on_8:.4e} vs off@16 {off_16:.4e})"
    );

    // Hydrostatic profile from the settled donor (defaults): layer-mean pressure linear in
    // depth. ρ_rest = particle_mass/spacing³ = 1, |g| = 20 → expected slope 20/unit depth.
    donor.step(DT, &input);
    let tw = twin_from_gpu(&donor);
    let p = donor.read_pressure();
    let ncy = tw.nc[1];
    let mut layers: Vec<(f64, f64)> = Vec::new(); // (y_center, mean p)
    for j in 0..ncy {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for k in 0..tw.nc[2] {
            for i in 0..tw.nc[0] {
                let c = tw.cidx(i, j, k);
                if tw.active[c] && tw.cell_is_interior(c, INTERIOR_MARGIN) {
                    sum += p[c] as f64;
                    cnt += 1;
                }
            }
        }
        if cnt > 0 {
            layers.push((tw.cell_center(tw.cidx(0, j, 0))[1], sum / cnt as f64));
        }
    }
    assert!(
        layers.len() >= 3,
        "need ≥3 interior layers for the fit, got {}",
        layers.len()
    );
    let (slope, r2) = linfit(&layers);
    let expected = -20.0; // dp/dy = −ρ_rest·|g|
    println!(
        "twofield U3 hydrostatic: {} layers, dp/dy {slope:.2} (expected {expected}), R² {r2:.4}",
        layers.len()
    );
    assert!(
        r2 >= HYDRO_R2_MIN,
        "pressure profile not linear: R² {r2:.4}"
    );
    let ratio = slope / expected;
    assert!(
        (HYDRO_SLOPE_BAND.0..=HYDRO_SLOPE_BAND.1).contains(&ratio),
        "hydrostatic slope off: dp/dy {slope:.2} vs expected {expected} (ratio {ratio:.2})"
    );
}

// ==============================================================================================
// Independent volume gates + cost + long run
// ==============================================================================================

/// Independent volume gates the self-measured divergence cannot game: water-height drift,
/// grid-mass density-proxy drift, and net flux through the closed walls.
#[test]
fn independent_volume_gates_settled_tank() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let scene = tank_scene(12.0);
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    for _ in 0..30 {
        solver.step(DT, &input);
    }
    let h0 = top_decile_y(&solver.read_positions());
    let rho0 = deep_mean_node_density(&solver);
    for _ in 0..600 {
        solver.step(DT, &input);
    }
    let pos = solver.read_positions();
    assert!(all_finite(&pos), "non-finite state after 600 steps");
    let h1 = top_decile_y(&pos);
    let rho1 = deep_mean_node_density(&solver);
    println!(
        "twofield U3 volume gates: height {h0:.3} -> {h1:.3} | deep density {rho0:.4} -> {rho1:.4}"
    );
    assert!(
        (h1 - h0).abs() <= HEIGHT_DRIFT_TOL * SPACING as f64,
        "water height drifted {:.3} (> {HEIGHT_DRIFT_TOL} spacing)",
        h1 - h0
    );
    assert!(
        (rho1 / rho0 - 1.0).abs() <= DENSITY_DRIFT_TOL,
        "deep density drifted {:.2}% (> {:.0}%)",
        (rho1 / rho0 - 1.0) * 100.0,
        DENSITY_DRIFT_TOL * 100.0
    );

    // Net flux through the closed box walls: Σ v_n·h² over massy wall-layer nodes ≈ 0.
    let flux = net_wall_flux(&solver, &scene);
    println!("twofield U3 wall flux: net {flux:.4} units³/s (tol {NET_WALL_FLUX_TOL})");
    assert!(
        flux.abs() <= NET_WALL_FLUX_TOL,
        "net flux {flux:.4} through closed walls exceeds {NET_WALL_FLUX_TOL}"
    );
}

/// Cost gate (R8): dispatches/frame equals the recorded U2 budget (4) plus the named U3, U4,
/// and U6 increments, and the constant matches the live profile.
#[test]
fn cost_gate_dispatch_budget() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let scene = tank_scene(12.0);
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    solver.step(DT, &EmissionInput::default());
    let profile = solver.profile();
    let (_, _, dims) = solver.grid_spec();
    assert_eq!(
        dispatches_per_frame_for(dims),
        4 + U3_PRESSURE_DISPATCHES + u4_surface_dispatches_for(dims) + U6_COUPLING_DISPATCHES,
        "budget must be U2's 4 + the named U3 + (scene-derived) U4 + U6 increments"
    );
    assert_eq!(
        profile.dispatches_per_frame,
        dispatches_per_frame_for(dims),
        "live dispatch count drifted from the recorded budget"
    );
    println!(
        "twofield U3+U4+U6 budgets: dispatches/frame {} (U2 4 + U3 increment {} + U4 increment {} + U6 increment {}) | max storage buffers per entry point {}",
        profile.dispatches_per_frame,
        U3_PRESSURE_DISPATCHES,
        u4_surface_dispatches_for(dims),
        U6_COUPLING_DISPATCHES,
        MAX_STORAGE_BUFFERS_PER_ENTRY_POINT
    );
}

/// Long-run gate: ≥2000-step settled tank stays settled — bounded speeds, no popcorn, finite.
#[test]
fn long_run_settled_tank_stays_settled() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_pressure: no GPU adapter; skipping.");
        return;
    };
    let scene = tank_scene(12.0);
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    let y_max0 = solver
        .read_positions()
        .iter()
        .map(|p| p[1])
        .fold(f32::MIN, f32::max);
    let mut peak_speed = 0.0f32;
    for f in 1..=LONGRUN_STEPS {
        solver.step(DT, &input);
        if f % 500 == 0 {
            let vel = solver.read_velocities();
            assert!(all_finite(&vel), "non-finite velocities at frame {f}");
            let vmax = vel
                .iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                .fold(0.0f32, f32::max);
            peak_speed = peak_speed.max(vmax);
        }
    }
    let pos = solver.read_positions();
    assert!(all_finite(&pos));
    let y_max = pos.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
    println!(
        "twofield U3 long run: {LONGRUN_STEPS} steps, sampled max|v| {peak_speed:.3}, max y {y_max0:.2} -> {y_max:.2}"
    );
    assert!(
        peak_speed <= LONGRUN_MAX_SPEED,
        "settled tank creeping/popcorning: max |v| {peak_speed:.3}"
    );
    assert!(
        y_max <= y_max0 + LONGRUN_POP_RISE * SPACING,
        "particles climbed: max y {y_max0:.2} -> {y_max:.2}"
    );
}

// ==============================================================================================
// GPU helpers
// ==============================================================================================

/// Masked-domain GPU scenes for the operator gates: closed tank, sloped wall (V60 cone),
/// partially masked interior (two disjoint blobs).
fn gpu_mask_scenes() -> Vec<(&'static str, Scene)> {
    let tank = tank_scene(12.0);
    // Water pool sized so the corrected (min-corner) classification still yields > 20 active
    // cells inside the cone — the narrow 4×2.4×4 block dropped to ~11 once smear cells
    // stopped counting as active.
    let cone = Scene {
        gravity: [0.0, -20.0, 0.0],
        box_min: [-7.0, -10.0, -7.0],
        box_max: [7.0, 10.0, 7.0],
        solids: coffee_sim::utils::geometry::v60_dripper(),
        regions: vec![SeedRegion {
            min: [-3.2, 0.8, -3.2],
            max: [3.2, 5.6, 3.2],
            species: Species::Water,
        }],
        ..Scene::default()
    };
    let partial = water_scene(
        [0.0; 3],
        [32.0, 24.0, 16.0],
        [0.0, -20.0, 0.0],
        &[
            ([2.0, 2.0, 2.0], [12.0, 10.0, 14.0]),
            ([20.0, 2.0, 2.0], [30.0, 10.0, 14.0]),
        ],
    );
    vec![
        ("closed tank", tank),
        ("v60 cone", cone),
        ("partial interior", partial),
    ]
}

/// Build the f64 twin of the GPU operator state: fill fractions/masks from `cell_meta`, M̃⁻¹
/// from the nm readback, geometry from `grid_spec`.
fn twin_from_gpu(solver: &TwofieldSolver) -> Twin {
    let (origin, h, dims) = solver.grid_spec();
    let nc = [
        dims[0] as usize - 1,
        dims[1] as usize - 1,
        dims[2] as usize - 1,
    ];
    let meta = solver.read_cell_meta();
    let active: Vec<bool> = meta.iter().map(|m| m[1] > 0.0).collect();
    let frac: Vec<f64> = meta.iter().map(|m| m[1] as f64).collect();
    let nm_raw = solver.read_node_matrices();
    let nm: Vec<[f64; 6]> = nm_raw
        .iter()
        .map(|r| {
            [
                r[0] as f64, // xx
                r[1] as f64, // xy
                r[2] as f64, // xz
                r[3] as f64, // yy
                r[4] as f64, // yz
                r[5] as f64, // zz
            ]
        })
        .collect();
    Twin {
        nc,
        h: h as f64,
        origin: [origin[0] as f64, origin[1] as f64, origin[2] as f64],
        active,
        frac,
        nm,
    }
}

fn top_decile_y(pos: &[[f32; 4]]) -> f64 {
    let mut ys: Vec<f32> = pos.iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ys.len() / 10;
    let top = &ys[ys.len() - n.max(1)..];
    top.iter().map(|&y| y as f64).sum::<f64>() / top.len() as f64
}

/// Mean node density (mass/h³) over the deep interior of the tank's water (lower half of the
/// fill depth, ≥2 cells from each wall) — the grid-mass volume proxy.
fn deep_mean_node_density(solver: &TwofieldSolver) -> f64 {
    let (origin, h, dims) = solver.grid_spec();
    let gv = solver.read_grid_velocities();
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for k in 2..dims[2] as usize - 2 {
        for j in 0..dims[1] as usize {
            let y = origin[1] + j as f32 * h;
            if !(2.0..=5.0).contains(&y) {
                continue; // deep slab, clear of floor BC and free surface
            }
            for i in 2..dims[0] as usize - 2 {
                let n = i + dims[0] as usize * (j + dims[1] as usize * k);
                sum += gv[n][3] as f64 / (h as f64).powi(3);
                cnt += 1;
            }
        }
    }
    assert!(cnt > 0, "deep density probe found no nodes");
    sum / cnt as f64
}

/// Net volumetric flux through the six closed box walls: Σ v_n·h² over wall-layer nodes that
/// carry mass (outward positive).
fn net_wall_flux(solver: &TwofieldSolver, scene: &Scene) -> f64 {
    let (origin, h, dims) = solver.grid_spec();
    let gv = solver.read_grid_velocities();
    let eps = 1e-3 * h;
    let mut flux = 0.0f64;
    for k in 0..dims[2] as usize {
        for j in 0..dims[1] as usize {
            for i in 0..dims[0] as usize {
                let n = i + dims[0] as usize * (j + dims[1] as usize * k);
                if gv[n][3] <= 1e-4 {
                    continue;
                }
                let xp = [
                    origin[0] + i as f32 * h,
                    origin[1] + j as f32 * h,
                    origin[2] + k as f32 * h,
                ];
                for a in 0..3 {
                    if (xp[a] - scene.box_min[a]).abs() <= eps {
                        flux += -(gv[n][a] as f64) * (h as f64) * (h as f64);
                    }
                    if (xp[a] - scene.box_max[a]).abs() <= eps {
                        flux += gv[n][a] as f64 * (h as f64) * (h as f64);
                    }
                }
            }
        }
    }
    flux
}

/// Least-squares line fit over (x, y) points → (slope, R²).
fn linfit(pts: &[(f64, f64)]) -> (f64, f64) {
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let syy: f64 = pts.iter().map(|p| (p.1 - my) * (p.1 - my)).sum();
    let slope = sxy / sxx;
    let r2 = if syy > 0.0 {
        (sxy * sxy) / (sxx * syy)
    } else {
        1.0
    };
    (slope, r2)
}

// ==============================================================================================
// CPU twins of the operator family (f64) — mirror pressure.wgsl exactly
// ==============================================================================================

/// Discrete twin of the U3 operator family. Pressure at cell centers (`nc` cells per axis),
/// velocity at the `nc+1` nodes per axis; `frac` is the per-cell fill fraction f (rows of D
/// are weighted by it; `active` ⇔ f > 0); `nm` is the symmetric per-node M̃⁻¹ packed
/// (xx, xy, xz, yy, yz, zz).
struct Twin {
    nc: [usize; 3],
    h: f64,
    origin: [f64; 3],
    active: Vec<bool>,
    frac: Vec<f64>,
    nm: Vec<[f64; 6]>,
}

impl Twin {
    fn nn(&self) -> [usize; 3] {
        [self.nc[0] + 1, self.nc[1] + 1, self.nc[2] + 1]
    }
    fn num_cells(&self) -> usize {
        self.nc[0] * self.nc[1] * self.nc[2]
    }
    fn num_nodes(&self) -> usize {
        let n = self.nn();
        n[0] * n[1] * n[2]
    }
    fn cidx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nc[0] * (j + self.nc[1] * k)
    }
    fn nidx(&self, i: usize, j: usize, k: usize) -> usize {
        let n = self.nn();
        i + n[0] * (j + n[1] * k)
    }
    fn cell_coords(&self, c: usize) -> [usize; 3] {
        [
            c % self.nc[0],
            (c / self.nc[0]) % self.nc[1],
            c / (self.nc[0] * self.nc[1]),
        ]
    }
    fn node_coords(&self, n: usize) -> [usize; 3] {
        let nn = self.nn();
        [n % nn[0], (n / nn[0]) % nn[1], n / (nn[0] * nn[1])]
    }
    fn node_pos(&self, n: usize) -> [f64; 3] {
        let c = self.node_coords(n);
        [
            self.origin[0] + c[0] as f64 * self.h,
            self.origin[1] + c[1] as f64 * self.h,
            self.origin[2] + c[2] as f64 * self.h,
        ]
    }
    fn cell_center(&self, c: usize) -> [f64; 3] {
        let cc = self.cell_coords(c);
        [
            self.origin[0] + (cc[0] as f64 + 0.5) * self.h,
            self.origin[1] + (cc[1] as f64 + 0.5) * self.h,
            self.origin[2] + (cc[2] as f64 + 0.5) * self.h,
        ]
    }

    /// All cells within Chebyshev distance `m` of `c` are FULL (f = 1, in range) — interior
    /// means clear of the fraction-tapered surface band as well as the hard masks.
    fn cell_is_interior(&self, c: usize, m: i64) -> bool {
        let cc = self.cell_coords(c);
        for dk in -m..=m {
            for dj in -m..=m {
                for di in -m..=m {
                    let i = cc[0] as i64 + di;
                    let j = cc[1] as i64 + dj;
                    let k = cc[2] as i64 + dk;
                    if i < 0
                        || j < 0
                        || k < 0
                        || i >= self.nc[0] as i64
                        || j >= self.nc[1] as i64
                        || k >= self.nc[2] as i64
                    {
                        return false;
                    }
                    if self.frac[self.cidx(i as usize, j as usize, k as usize)] < 0.999 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// All nodes within Chebyshev distance `m` of `c`'s corners have the unconstrained
    /// inverse-mass form (full diagonal, zero off-diagonals, equal diagonal entries).
    fn cell_nodes_free(&self, c: usize, m: i64) -> bool {
        let cc = self.cell_coords(c);
        let nn = self.nn();
        for dk in -m..=(m + 1) {
            for dj in -m..=(m + 1) {
                for di in -m..=(m + 1) {
                    let i = cc[0] as i64 + di;
                    let j = cc[1] as i64 + dj;
                    let k = cc[2] as i64 + dk;
                    if i < 0 || j < 0 || k < 0 {
                        continue;
                    }
                    let (i, j, k) = (i as usize, j as usize, k as usize);
                    if i >= nn[0] || j >= nn[1] || k >= nn[2] {
                        continue;
                    }
                    let s = &self.nm[self.nidx(i, j, k)];
                    let free = s[1].abs() < 1e-15
                        && s[2].abs() < 1e-15
                        && s[4].abs() < 1e-15
                        && (s[0] - s[3]).abs() < 1e-12
                        && (s[0] - s[5]).abs() < 1e-12
                        && s[0] > 0.0;
                    if !free {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// All 8 cells adjacent to node `n` exist and are FULL (f = 1) — G untruncated and
    /// un-tapered there (the MMS order claim is for the f = 1 rows; surface rows are
    /// first-order by design until U4).
    fn node_fully_surrounded(&self, n: usize) -> bool {
        let c = self.node_coords(n);
        for dk in 0..2usize {
            for dj in 0..2usize {
                for di in 0..2usize {
                    let i = c[0] as i64 - 1 + di as i64;
                    let j = c[1] as i64 - 1 + dj as i64;
                    let k = c[2] as i64 - 1 + dk as i64;
                    if i < 0
                        || j < 0
                        || k < 0
                        || i >= self.nc[0] as i64
                        || j >= self.nc[1] as i64
                        || k >= self.nc[2] as i64
                        || self.frac[self.cidx(i as usize, j as usize, k as usize)] < 0.999
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// D_f: corner-trilinear (FV) cell divergence of the node field, row-weighted by the cell
    /// fill fraction; masked rows (f = 0) → 0.
    fn div(&self, u: &[[f64; 3]]) -> Vec<f64> {
        let mut out = vec![0.0; self.num_cells()];
        let inv = 1.0 / (4.0 * self.h);
        for k in 0..self.nc[2] {
            for j in 0..self.nc[1] {
                for i in 0..self.nc[0] {
                    let c = self.cidx(i, j, k);
                    if !self.active[c] {
                        continue;
                    }
                    let mut d = 0.0;
                    for oz in 0..2usize {
                        for oy in 0..2usize {
                            for ox in 0..2usize {
                                let n = self.nidx(i + ox, j + oy, k + oz);
                                let s = [
                                    ox as f64 * 2.0 - 1.0,
                                    oy as f64 * 2.0 - 1.0,
                                    oz as f64 * 2.0 - 1.0,
                                ];
                                d += (s[0] * u[n][0] + s[1] * u[n][1] + s[2] * u[n][2]) * inv;
                            }
                        }
                    }
                    out[c] = d * self.frac[c];
                }
            }
        }
        out
    }

    /// G = −D_fᵀ: node gradient gathered from the adjacent cells, fraction-weighted.
    fn grad(&self, p: &[f64]) -> Vec<[f64; 3]> {
        let mut out = vec![[0.0; 3]; self.num_nodes()];
        let inv = 1.0 / (4.0 * self.h);
        let nn = self.nn();
        for k in 0..nn[2] {
            for j in 0..nn[1] {
                for i in 0..nn[0] {
                    let n = self.nidx(i, j, k);
                    let mut g = [0.0; 3];
                    for oz in 0..2usize {
                        for oy in 0..2usize {
                            for ox in 0..2usize {
                                // Cell with this node at corner offset (ox, oy, oz).
                                let ci = i as i64 - 1 + ox as i64;
                                let cj = j as i64 - 1 + oy as i64;
                                let ck = k as i64 - 1 + oz as i64;
                                if ci < 0
                                    || cj < 0
                                    || ck < 0
                                    || ci >= self.nc[0] as i64
                                    || cj >= self.nc[1] as i64
                                    || ck >= self.nc[2] as i64
                                {
                                    continue;
                                }
                                let c = self.cidx(ci as usize, cj as usize, ck as usize);
                                if !self.active[c] {
                                    continue;
                                }
                                // Node offset in cell = 1 − loop offset on each axis.
                                let s = [
                                    (1 - ox) as f64 * 2.0 - 1.0,
                                    (1 - oy) as f64 * 2.0 - 1.0,
                                    (1 - oz) as f64 * 2.0 - 1.0,
                                ];
                                for a in 0..3 {
                                    g[a] -= s[a] * self.frac[c] * p[c] * inv;
                                }
                            }
                        }
                    }
                    out[n] = g;
                }
            }
        }
        out
    }

    /// M̃⁻¹·v at node `n` (symmetric 3×3 from the packed 6).
    fn minv(&self, n: usize, v: [f64; 3]) -> [f64; 3] {
        let s = &self.nm[n];
        [
            s[0] * v[0] + s[1] * v[1] + s[2] * v[2],
            s[1] * v[0] + s[3] * v[1] + s[4] * v[2],
            s[2] * v[0] + s[4] * v[1] + s[5] * v[2],
        ]
    }

    /// A = −D·M̃⁻¹·G, applied ONLY as the composition.
    fn apply_a(&self, p: &[f64]) -> Vec<f64> {
        let g = self.grad(p);
        let u: Vec<[f64; 3]> = (0..self.num_nodes()).map(|n| self.minv(n, g[n])).collect();
        self.div(&u).iter().map(|d| -d).collect()
    }

    /// diag(A) per cell: f_c²·Σ_corners sᵀ·M̃⁻¹·s / (16h²).
    fn diag_a(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.num_cells()];
        let inv2 = 1.0 / (16.0 * self.h * self.h);
        for k in 0..self.nc[2] {
            for j in 0..self.nc[1] {
                for i in 0..self.nc[0] {
                    let c = self.cidx(i, j, k);
                    if !self.active[c] {
                        continue;
                    }
                    let mut d = 0.0;
                    for oz in 0..2usize {
                        for oy in 0..2usize {
                            for ox in 0..2usize {
                                let n = self.nidx(i + ox, j + oy, k + oz);
                                let s = [
                                    ox as f64 * 2.0 - 1.0,
                                    oy as f64 * 2.0 - 1.0,
                                    oz as f64 * 2.0 - 1.0,
                                ];
                                let ms = self.minv(n, s);
                                d += (s[0] * ms[0] + s[1] * ms[1] + s[2] * ms[2]) * inv2;
                            }
                        }
                    }
                    out[c] = d * self.frac[c] * self.frac[c];
                }
            }
        }
        out
    }

    /// One damped Jacobi sweep (ω = JACOBI_OMEGA), inactive/undefined rows pinned to 0.
    fn jacobi(&self, p: &[f64], rhs: &[f64]) -> Vec<f64> {
        let ap = self.apply_a(p);
        let diag = self.diag_a();
        (0..self.num_cells())
            .map(|c| {
                if self.active[c] && diag[c] > 1e-30 {
                    p[c] + JACOBI_OMEGA as f64 * (rhs[c] - ap[c]) / diag[c]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Rediscretized coarse twin at `ratio` cells per coarse cell: same family at H = ratio·h,
    /// coarse masks SATURATED to binary (active iff any active child — the fine fraction
    /// taper is deliberately not rediscretized; see `coarse_cell_setup` in pressure.wgsl),
    /// coarse node M̃⁻¹ = hat-weighted restriction of the fine M̃⁻¹ field (mirrors
    /// `coarse_node_setup`; convex combination of PSD matrices, never a point injection —
    /// see the WGSL comment for the near-singular-row failure injection caused).
    fn coarsen(&self, ratio: usize) -> Twin {
        let nc = [
            self.nc[0].div_ceil(ratio),
            self.nc[1].div_ceil(ratio),
            self.nc[2].div_ceil(ratio),
        ];
        let mut active = vec![false; nc[0] * nc[1] * nc[2]];
        for k in 0..self.nc[2] {
            for j in 0..self.nc[1] {
                for i in 0..self.nc[0] {
                    if self.frac[self.cidx(i, j, k)] > 0.0 {
                        active[(i / ratio) + nc[0] * ((j / ratio) + nc[1] * (k / ratio))] = true;
                    }
                }
            }
        }
        let frac: Vec<f64> = active.iter().map(|&a| if a { 1.0 } else { 0.0 }).collect();
        let nn = [nc[0] + 1, nc[1] + 1, nc[2] + 1];
        let fnn = self.nn();
        let r = ratio as i64;
        let mut nm = vec![[0.0; 6]; nn[0] * nn[1] * nn[2]];
        for k in 0..nn[2] {
            for j in 0..nn[1] {
                for i in 0..nn[0] {
                    let f0 = [i as i64 * r, j as i64 * r, k as i64 * r];
                    let mut wsum = 0.0;
                    let mut s = [0.0f64; 6];
                    for dz in -(r - 1)..=(r - 1) {
                        for dy in -(r - 1)..=(r - 1) {
                            for dx in -(r - 1)..=(r - 1) {
                                let f = [f0[0] + dx, f0[1] + dy, f0[2] + dz];
                                if f.iter().zip(&fnn).any(|(&x, &n)| x < 0 || x >= n as i64) {
                                    continue;
                                }
                                let w = (1.0 - dx.abs() as f64 / r as f64)
                                    * (1.0 - dy.abs() as f64 / r as f64)
                                    * (1.0 - dz.abs() as f64 / r as f64);
                                let fm = &self.nm
                                    [self.nidx(f[0] as usize, f[1] as usize, f[2] as usize)];
                                for a in 0..6 {
                                    s[a] += w * fm[a];
                                }
                                wsum += w;
                            }
                        }
                    }
                    let inv = 1.0 / wsum.max(1e-300);
                    let mut out = [0.0f64; 6];
                    for a in 0..6 {
                        out[a] = s[a] * inv;
                    }
                    nm[i + nn[0] * (j + nn[1] * k)] = out;
                }
            }
        }
        Twin {
            nc,
            h: self.h * ratio as f64,
            origin: self.origin,
            active,
            frac,
            nm,
        }
    }

    /// R: masked trilinear-transpose restriction of a fine cell field, fixed 1/ratio³ volume
    /// weighting (NO renormalisation — that is what keeps R = Pᵀ/ratio³ exact).
    fn restrict(&self, ct: &Twin, ratio: usize, f: &[f64]) -> Vec<f64> {
        let r = ratio as i64;
        let mut out = vec![0.0; ct.num_cells()];
        for ck in 0..ct.nc[2] {
            for cj in 0..ct.nc[1] {
                for ci in 0..ct.nc[0] {
                    let cc = ct.cidx(ci, cj, ck);
                    if !ct.active[cc] {
                        continue;
                    }
                    let mut sum = 0.0;
                    let lo = |c: usize| c as i64 * r - r / 2;
                    for fk in lo(ck)..lo(ck) + 2 * r {
                        for fj in lo(cj)..lo(cj) + 2 * r {
                            for fi in lo(ci)..lo(ci) + 2 * r {
                                if fi < 0
                                    || fj < 0
                                    || fk < 0
                                    || fi >= self.nc[0] as i64
                                    || fj >= self.nc[1] as i64
                                    || fk >= self.nc[2] as i64
                                {
                                    continue;
                                }
                                let fc = self.cidx(fi as usize, fj as usize, fk as usize);
                                if !self.active[fc] {
                                    continue;
                                }
                                let w = hat(fi, ci as i64, r)
                                    * hat(fj, cj as i64, r)
                                    * hat(fk, ck as i64, r);
                                sum += w * f[fc];
                            }
                        }
                    }
                    out[cc] = sum / (ratio * ratio * ratio) as f64;
                }
            }
        }
        out
    }

    /// P = (transpose-consistent) masked trilinear prolongation of a coarse cell field.
    fn prolong(&self, ct: &Twin, ratio: usize, pc: &[f64]) -> Vec<f64> {
        let r = ratio as i64;
        let mut out = vec![0.0; self.num_cells()];
        for fk in 0..self.nc[2] {
            for fj in 0..self.nc[1] {
                for fi in 0..self.nc[0] {
                    let fc = self.cidx(fi, fj, fk);
                    if !self.active[fc] {
                        continue;
                    }
                    let base = |f: usize| -> i64 {
                        ((f as f64 + 0.5 - r as f64 * 0.5) / r as f64).floor() as i64
                    };
                    let mut sum = 0.0;
                    for ck in base(fk)..base(fk) + 2 {
                        for cj in base(fj)..base(fj) + 2 {
                            for ci in base(fi)..base(fi) + 2 {
                                if ci < 0
                                    || cj < 0
                                    || ck < 0
                                    || ci >= ct.nc[0] as i64
                                    || cj >= ct.nc[1] as i64
                                    || ck >= ct.nc[2] as i64
                                {
                                    continue;
                                }
                                let cc = ct.cidx(ci as usize, cj as usize, ck as usize);
                                if !ct.active[cc] {
                                    continue;
                                }
                                let w = hat(fi as i64, ci, r)
                                    * hat(fj as i64, cj, r)
                                    * hat(fk as i64, ck, r);
                                sum += w * pc[cc];
                            }
                        }
                    }
                    out[fc] = sum;
                }
            }
        }
        out
    }

    // --- synthetic-domain builders (CPU-only tests) -------------------------------------------
    // Synthetic domains exercise hard masks (f ∈ {0, 1}); the fractional surface band is
    // exercised by the GPU-twin pins on real scenes (`gpu_operators_match_twins_and_are_adjoint`).

    /// Re-derive the binary fill fractions after a builder mutated the active mask.
    fn sync_frac(&mut self) {
        self.frac = self
            .active
            .iter()
            .map(|&a| if a { 1.0 } else { 0.0 })
            .collect();
    }

    /// Closed box with `open_top` inactive cell layers at the top (implicit Dirichlet there):
    /// walls + floor constrained axis-wise, top nodes free.
    fn box_domain(nc: [usize; 3], h: f64, open_top: usize) -> Twin {
        let mut tw = Twin::full_box(nc, h);
        for k in 0..nc[2] {
            for j in nc[1] - open_top..nc[1] {
                for i in 0..nc[0] {
                    let c = tw.cidx(i, j, k);
                    tw.active[c] = false;
                }
            }
        }
        tw.sync_frac();
        tw
    }

    /// Fully active box: every cell active; lattice-boundary nodes wall-constrained except the
    /// top face (free, mirroring an open tank above the Dirichlet layers).
    fn full_box(nc: [usize; 3], h: f64) -> Twin {
        let num = nc[0] * nc[1] * nc[2];
        let active = vec![true; num];
        let frac = vec![1.0; num];
        let nn = [nc[0] + 1, nc[1] + 1, nc[2] + 1];
        let mut nm = vec![[0.0; 6]; nn[0] * nn[1] * nn[2]];
        for k in 0..nn[2] {
            for j in 0..nn[1] {
                for i in 0..nn[0] {
                    let mut diag = [1.0f64; 3];
                    if i == 0 || i == nn[0] - 1 {
                        diag[0] = 0.0;
                    }
                    if j == 0 {
                        diag[1] = 0.0; // floor; top stays free
                    }
                    if k == 0 || k == nn[2] - 1 {
                        diag[2] = 0.0;
                    }
                    nm[i + nn[0] * (j + nn[1] * k)] = [diag[0], 0.0, 0.0, diag[1], 0.0, diag[2]];
                }
            }
        }
        Twin {
            nc,
            h,
            origin: [0.0; 3],
            active,
            frac,
            nm,
        }
    }

    /// Sloped-wall variant: a plane wall with non-axis-aligned normal cuts the box; cells
    /// behind it inactive, nodes on/behind it constrained along the (orthogonalized) normal.
    fn slope_domain(nc: [usize; 3], h: f64, open_top: usize) -> Twin {
        let mut tw = Twin::box_domain(nc, h, open_top);
        let n = {
            let len = (1.0f64 + 4.0 + 0.25).sqrt();
            [1.0 / len, 2.0 / len, 0.5 / len]
        };
        let d0 = 0.35 * nc[0] as f64 * h; // plane offset along its normal
        let phi = |p: [f64; 3]| p[0] * n[0] + p[1] * n[1] + p[2] * n[2] - d0;
        for c in 0..tw.num_cells() {
            if phi(tw.cell_center(c)) < 0.0 {
                tw.active[c] = false;
            }
        }
        tw.sync_frac();
        let nn = tw.nn();
        for k in 0..nn[2] {
            for j in 0..nn[1] {
                for i in 0..nn[0] {
                    let idx = tw.nidx(i, j, k);
                    let p = tw.node_pos(idx);
                    if phi(p) < 1e-9 {
                        tw.nm[idx] = constrain_sym(tw.nm[idx], n);
                    }
                }
            }
        }
        tw
    }

    /// Hole-masked interior: a spherical inactive pocket inside the box, mass-less nodes
    /// (zero M̃⁻¹) deep inside it.
    fn hole_domain(nc: [usize; 3], h: f64, open_top: usize) -> Twin {
        let mut tw = Twin::box_domain(nc, h, open_top);
        let ctr = [
            0.5 * nc[0] as f64 * h,
            0.4 * nc[1] as f64 * h,
            0.5 * nc[2] as f64 * h,
        ];
        let rad = 0.12 * nc[0] as f64 * h;
        let dist = |p: [f64; 3]| {
            ((p[0] - ctr[0]).powi(2) + (p[1] - ctr[1]).powi(2) + (p[2] - ctr[2]).powi(2)).sqrt()
        };
        for c in 0..tw.num_cells() {
            if dist(tw.cell_center(c)) < rad {
                tw.active[c] = false;
            }
        }
        tw.sync_frac();
        for n in 0..tw.num_nodes() {
            if dist(tw.node_pos(n)) < rad - 1.5 * h {
                tw.nm[n] = [0.0; 6]; // deep inside the pocket: no mass
            }
        }
        tw
    }

    /// Chebyshev distance (in cells) from `c` to the nearest inactive/out-of-range cell,
    /// capped at `cap`.
    fn dist_to_mask(&self, c: usize, cap: i64) -> i64 {
        for d in 0..=cap {
            if !self.cell_is_interior(c, d) {
                return d;
            }
        }
        cap + 1
    }

    /// Smooth taper that vanishes toward the mask boundary — the manufactured modes must
    /// CONFORM to the homogeneous Dirichlet structure (physical pressure vanishes at the free
    /// surface), otherwise rhs = A·mode carries 1/h² boundary-jump spikes that no consistent
    /// restriction can represent and the comparison tests an unphysical regime.
    fn taper(&self, c: usize) -> f64 {
        let d = self.dist_to_mask(c, 5);
        let s = (d as f64 / 4.0).min(1.0);
        s * s * (3.0 - 2.0 * s)
    }

    /// Conforming low-frequency mode: half-wave sin products, tapered to the mask boundary.
    fn mode_low(&self) -> Vec<f64> {
        let l = [
            self.nc[0] as f64 * self.h,
            self.nc[1] as f64 * self.h,
            self.nc[2] as f64 * self.h,
        ];
        (0..self.num_cells())
            .map(|c| {
                if !self.active[c] {
                    return 0.0;
                }
                let p = self.cell_center(c);
                self.taper(c)
                    * (std::f64::consts::PI * (p[0] - self.origin[0]) / l[0]).sin()
                    * (std::f64::consts::PI * (p[1] - self.origin[1]) / l[1]).sin()
                    * (std::f64::consts::PI * (p[2] - self.origin[2]) / l[2]).sin()
            })
            .collect()
    }

    /// Conforming high-frequency mode at θ = π/2 per axis (4-cell wavelength). NOT the full
    /// checkerboard: θ = π per axis is in the corner family's pressure null space (it produces
    /// no node gradient), so it is invisible to A — and to the projection — by construction.
    fn mode_high(&self) -> Vec<f64> {
        (0..self.num_cells())
            .map(|c| {
                if !self.active[c] {
                    return 0.0;
                }
                let cc = self.cell_coords(c);
                let f = |i: usize| (std::f64::consts::PI * (i as f64 + 0.5) / 2.0).sin();
                self.taper(c) * f(cc[0]) * f(cc[1]) * f(cc[2])
            })
            .collect()
    }
}

/// Hat (linear) weight between fine cell `fi` and coarse cell `ci` at `ratio` r, per axis —
/// the single weight function both R and P share (transpose pair by construction).
fn hat(fi: i64, ci: i64, r: i64) -> f64 {
    let x = fi as f64 + 0.5;
    let xc = (ci * r) as f64 + r as f64 * 0.5;
    (1.0 - (x - xc).abs() / r as f64).max(0.0)
}

/// Symmetric projector update mirroring node_setup: subtract the (already box-orthogonalized)
/// unit-normal dyad from a packed symmetric M̃⁻¹.
fn constrain_sym(s: [f64; 6], n: [f64; 3]) -> [f64; 6] {
    // Project n out of any already-zeroed axes (diagonal 0 means constrained axis).
    let mut v = n;
    let diag = [s[0], s[3], s[5]];
    for a in 0..3 {
        if diag[a] == 0.0 {
            v[a] = 0.0;
        }
    }
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-3 {
        return s;
    }
    let v = [v[0] / len, v[1] / len, v[2] / len];
    // s − invρ·n̂n̂ᵀ with invρ read off the largest remaining diagonal entry.
    let invr = diag[0].max(diag[1]).max(diag[2]);
    [
        s[0] - invr * v[0] * v[0],
        s[1] - invr * v[0] * v[1],
        s[2] - invr * v[0] * v[2],
        s[3] - invr * v[1] * v[1],
        s[4] - invr * v[1] * v[2],
        s[5] - invr * v[2] * v[2],
    ]
}

/// The three CPU masked domains the operator gates run on. Sized so a conforming low mode
/// (4-cell boundary taper) still spans several COARSE cells at ratio 4 — on smaller lattices
/// the tapered mode is sub-coarse-resolution and the two-grid question is ill-posed (the
/// real, smaller scenes are covered by the GPU A/B gate on the actual solver).
fn cpu_domains() -> Vec<(&'static str, Twin)> {
    vec![
        ("closed box", Twin::box_domain([24, 28, 24], 1.0, 4)),
        ("sloped wall", Twin::slope_domain([24, 28, 24], 1.0, 4)),
        // Larger: the conforming taper kills the mode within ~4 cells of the hole, so the
        // hole must stay small relative to the domain for a genuine low mode to exist.
        ("hole interior", Twin::hole_domain([32, 36, 32], 1.0, 4)),
    ]
}

/// The full seeded cycle twin: restrict rhs, Nc coarse sweeps from zero, prolongate as the
/// fine seed, Nf fine sweeps.
fn two_grid_cycle(tw: &Twin, ratio: usize, nc: u32, nf: u32, rhs: &[f64]) -> Vec<f64> {
    let ct = tw.coarsen(ratio);
    let rhs_c = tw.restrict(&ct, ratio, rhs);
    let mut pc = vec![0.0; ct.num_cells()];
    for _ in 0..nc {
        pc = ct.jacobi(&pc, &rhs_c);
    }
    let mut p = tw.prolong(&ct, ratio, &pc);
    for _ in 0..nf {
        p = tw.jacobi(&p, rhs);
    }
    p
}

fn rms_masked(x: &[f64], active: &[bool]) -> f64 {
    let mut se = 0.0;
    let mut n = 0usize;
    for (v, &a) in x.iter().zip(active) {
        if a {
            se += v * v;
            n += 1;
        }
    }
    (se / n.max(1) as f64).sqrt()
}

fn rms_err(x: &[f64], y: &[f64], active: &[bool]) -> f64 {
    let d: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    rms_masked(&d, active)
}

// --- manufactured solutions (MMS) -------------------------------------------------------------

const MMS_A: f64 = std::f64::consts::PI / 12.0;
const MMS_B: f64 = std::f64::consts::PI / 8.0;

fn mms_u(p: [f64; 3]) -> [f64; 3] {
    [
        (MMS_A * p[0]).sin() * (MMS_B * p[1]).cos(),
        (MMS_B * p[1]).sin() * (MMS_A * p[2]).cos(),
        (MMS_A * p[2]).sin() * (MMS_B * p[0]).cos(),
    ]
}

fn mms_div(p: [f64; 3]) -> f64 {
    MMS_A * (MMS_A * p[0]).cos() * (MMS_B * p[1]).cos()
        + MMS_B * (MMS_B * p[1]).cos() * (MMS_A * p[2]).cos()
        + MMS_A * (MMS_A * p[2]).cos() * (MMS_B * p[0]).cos()
}

fn mms_p(p: [f64; 3]) -> f64 {
    (MMS_A * p[0]).sin() * (MMS_B * p[1]).sin() * (MMS_A * p[2]).sin()
}

fn mms_gradp(p: [f64; 3]) -> [f64; 3] {
    [
        MMS_A * (MMS_A * p[0]).cos() * (MMS_B * p[1]).sin() * (MMS_A * p[2]).sin(),
        MMS_B * (MMS_A * p[0]).sin() * (MMS_B * p[1]).cos() * (MMS_A * p[2]).sin(),
        MMS_A * (MMS_A * p[0]).sin() * (MMS_B * p[1]).sin() * (MMS_A * p[2]).cos(),
    ]
}

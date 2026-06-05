//! Phase 1.5 dissolution (extraction) invariants (GPU-gated; skips without an adapter).
//!
//! The dissolution pass moves two-pool solute from wet grains into overlapping water's
//! concentration `c`, conserving the *solute inventory* `Σ(grain s_f+s_s) + Σ(water c·f_w·V_w)`
//! exactly (atomic-free, frozen-snapshot two-sided transfer — the Phase-1.4 wetting pattern). The
//! `(1−c/c_sat)` driving force is realized as the water-side headroom cap, so no water exceeds
//! `c_sat` and grains stop releasing into saturated water.
//!
//! The conservation gate isolates the dissolution pass: wetting OFF (`absorb_rate=0`) with grains
//! PRE-WET via `write_moisture_for_test`, so `f_w` is exactly constant and the only buffer the
//! passes touch is `chem`. (With wetting co-active, the absorbed-water volume transfer carries a
//! second-order solute coupling the model intentionally neglects — KTD-1 — so it is not part of the
//! dissolution invariant.) Extraction needs flow: a zero-relative-velocity blob has zero flux and
//! does not extract, so the tests seed a small water velocity.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::extraction;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

/// Per-grain water capacity `V_cap = r_max·rho_ratio·(π/6·d³)`.
fn capacity(mats: &Materials) -> f32 {
    let v_dry = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    mats.r_max * mats.rho_ratio * v_dry
}

/// A point seed region (one particle) of the given species.
fn point(p: [f32; 3], species: Species) -> SeedRegion {
    SeedRegion {
        min: p,
        max: p,
        species,
    }
}

/// A compact mixed blob (grain slab overlapping a water slab in `y`), gravity off, so grain↔water
/// pairs sit within the support radius without any mechanical motion.
fn wet_blob_scene() -> Scene {
    Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [3.0, 3.0, 3.0],
                max: [5.0, 4.0, 5.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [3.0, 3.5, 3.0],
                max: [5.0, 4.5, 5.0],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    }
}

/// Config isolating the dissolution pass: every mechanical solve + wetting off, extraction on.
fn extract_isolation_config() -> Config {
    Config {
        max_iters: 0,
        bed_max_iters: 0,
        drag_subiters: 0,
        buoyancy_scale: 0.0,
        xsph_viscosity_c: 0.0,
        grain_sleep_speed: 0.0,
        absorb_rate: 0.0, // wetting off → f_w constant → dissolution conservation is exact
        extract_rate: 1.0,
        ..Config::default()
    }
}

/// Solute inventory: grain pools `s_f+s_s` (absolute) + water `c·f_w·V_w` (concentration × volume).
fn solute_inventory(chem: &[[f32; 4]], moisture: &[f32], phase: &[u32], v_w: f32) -> f32 {
    chem.iter()
        .zip(moisture)
        .zip(phase)
        .map(|((c, &m), &ph)| {
            if ph == 1 {
                c[0] + c[1] // grain: two pools
            } else {
                c[0] * m * v_w // water: concentration × remaining volume
            }
        })
        .sum()
}

/// Pre-wet grains to `frac·V_cap` and leave water full (`f_w=1`), so the dissolution pass runs with
/// a constant moisture field.
fn prewet(solver: &XpbdSolver, phase: &[u32], frac: f32, v_cap: f32) {
    let moisture: Vec<f32> = phase
        .iter()
        .map(|&ph| if ph == 1 { frac * v_cap } else { 1.0 })
        .collect();
    solver.write_moisture_for_test(&moisture);
}

/// Seed a uniform sideways water velocity (the flow that drives extraction); grains at rest.
fn seed_water_flow(solver: &XpbdSolver, phase: &[u32], u: f32) {
    let vel: Vec<[f32; 4]> = phase
        .iter()
        .map(|&ph| {
            if ph == 0 {
                [u, 0.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            }
        })
        .collect();
    solver.write_velocities_for_test(&vel);
}

/// THE gate: with the dissolution pass isolated (constant `f_w`), the solute inventory is invariant
/// to float tolerance every step — grain pool loss equals water concentration gain, no leak.
#[test]
fn dissolution_conserves_solute_inventory() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    let mut solver = XpbdSolver::build(&wet_blob_scene(), &mats, &extract_isolation_config(), &gpu);
    let phase = solver.read_phases();
    let v_w = solver.water_particle_volume();
    assert!(
        phase.contains(&0) && phase.contains(&1),
        "scene must be mixed"
    );
    prewet(&solver, &phase, 0.6, v_cap);
    seed_water_flow(&solver, &phase, 0.2);

    let moisture = solver.read_moisture();
    let initial = solute_inventory(&solver.read_chem(), &moisture, &phase, v_w);
    for step in 0..15 {
        solver.step(DT, &EmissionInput::default());
        let inv = solute_inventory(&solver.read_chem(), &solver.read_moisture(), &phase, v_w);
        assert!(
            (inv - initial).abs() <= 1.0e-4 * initial.max(1.0),
            "solute leaked at step {step}: {initial} -> {inv}"
        );
    }

    // ...and extraction was non-trivial: grain pools shrank, some water gained concentration.
    let chem = solver.read_chem();
    let grain_pools: f32 = chem
        .iter()
        .zip(&phase)
        .filter(|(_, &p)| p == 1)
        .map(|(c, _)| c[0] + c[1])
        .sum();
    let grain_pools_0: f32 =
        extraction::split(mats.soluble_fraction * mats.grain_mass, mats.fast_fraction).0
            + extraction::split(mats.soluble_fraction * mats.grain_mass, mats.fast_fraction).1;
    let n_grain = phase.iter().filter(|&&p| p == 1).count() as f32;
    assert!(
        grain_pools < grain_pools_0 * n_grain - 1.0e-6,
        "no extraction occurred (test would be vacuous): {grain_pools} vs {}",
        grain_pools_0 * n_grain
    );
    assert!(
        chem.iter()
            .zip(&phase)
            .any(|(c, &p)| p == 0 && c[0] > 1.0e-6),
        "no water gained concentration"
    );
}

/// Opt-in gate (R7): with `extract_rate=0` the chem buffer is byte-unchanged even while wetting runs
/// (grains wet, but nothing dissolves).
#[test]
fn no_extraction_without_opt_in() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    // Wetting ON so grains actually wet; extraction OFF (default) so chem must not move.
    let cfg = Config {
        absorb_rate: 0.5,
        extract_rate: 0.0,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&wet_blob_scene(), &mats, &cfg, &gpu);
    let before = solver.read_chem();
    for _ in 0..30 {
        solver.step(DT, &EmissionInput::default());
    }
    let after = solver.read_chem();
    for (i, (a, b)) in before.iter().zip(&after).enumerate() {
        assert_eq!(a, b, "chem drifted at particle {i} with extraction off");
    }
}

/// The moisture gate (R1): a dry grain (never wetted) releases nothing and water `c` stays 0.
#[test]
fn dry_grain_does_not_extract() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    // Extraction ON but wetting OFF and grains seeded dry (V_abs=0) → the wet gate / dry-grain guard
    // hold, so no pool depletes and no water gains concentration.
    let cfg = extract_isolation_config();
    let mut solver = XpbdSolver::build(&wet_blob_scene(), &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    seed_water_flow(&solver, &phase, 0.2); // flow present, but grains are dry
    let before = solver.read_chem();
    for _ in 0..20 {
        solver.step(DT, &EmissionInput::default());
    }
    let after = solver.read_chem();
    for (i, (a, b)) in before.iter().zip(&after).enumerate() {
        assert_eq!(a, b, "dry grain extracted at particle {i}");
    }
}

/// Headroom cap (R4): drive extraction hard into a fixed water pool → `c` rises toward but never
/// past `c_sat`, and no grain pool goes negative.
#[test]
fn water_concentration_capped_at_c_sat() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    // One water surrounded by many grains (lots of solute sources) so it saturates fast.
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Water),
            point([3.3, 4.0, 4.0], Species::Grain),
            point([4.7, 4.0, 4.0], Species::Grain),
            point([4.0, 4.7, 4.0], Species::Grain),
            point([4.0, 3.3, 4.0], Species::Grain),
            point([4.0, 4.0, 4.7], Species::Grain),
            point([4.0, 4.0, 3.3], Species::Grain),
        ],
        ..Scene::default()
    };
    let cfg = Config {
        extract_rate: 5.0, // hard drive
        ..extract_isolation_config()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    prewet(&solver, &phase, 0.8, v_cap);
    seed_water_flow(&solver, &phase, 0.5);

    let mut max_c = 0.0f32;
    for _ in 0..400 {
        solver.step(DT, &EmissionInput::default());
        let chem = solver.read_chem();
        for (c, &p) in chem.iter().zip(&phase) {
            if p == 0 {
                max_c = max_c.max(c[0]);
                assert!(
                    c[0] <= mats.c_sat + 1.0e-5,
                    "water exceeded c_sat: {} > {}",
                    c[0],
                    mats.c_sat
                );
            } else {
                assert!(
                    c[0] >= -1.0e-6 && c[1] >= -1.0e-6,
                    "grain pool went negative"
                );
            }
        }
    }
    // The cap is non-trivially exercised (the water actually approached saturation).
    assert!(
        max_c > 0.5 * mats.c_sat,
        "water never approached c_sat ({max_c}); test is vacuous"
    );
}

/// Codex R1: a saturated water (`c≈c_sat`) next to an unsaturated one on the same grain — the grain
/// releases into the unsaturated water only, the saturated one stays at `c_sat`, solute conserved.
#[test]
fn saturated_neighbor_blocks_extraction_but_unsaturated_receives() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([4.5, 4.0, 4.0], Species::Water), // will be pre-saturated
            point([3.5, 4.0, 4.0], Species::Water), // unsaturated
        ],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &extract_isolation_config(), &gpu);
    let phase = solver.read_phases();
    assert_eq!(phase, vec![1, 0, 0]);
    prewet(&solver, &phase, 0.8, v_cap);
    seed_water_flow(&solver, &phase, 0.3);
    // Pre-saturate water[1]; water[2] starts at 0.
    let mut chem = solver.read_chem();
    chem[1] = [mats.c_sat, mats.pour_t, 0.0, 0.0];
    solver.write_chem_for_test(&chem);
    let v_w = solver.water_particle_volume();

    let initial = solute_inventory(&solver.read_chem(), &solver.read_moisture(), &phase, v_w);
    for _ in 0..30 {
        solver.step(DT, &EmissionInput::default());
    }
    let chem = solver.read_chem();
    let inv = solute_inventory(&chem, &solver.read_moisture(), &phase, v_w);

    // Saturated water did not exceed c_sat (headroom 0 → accepts nothing).
    assert!(
        chem[1][0] <= mats.c_sat + 1.0e-5,
        "saturated water exceeded c_sat: {}",
        chem[1][0]
    );
    // Unsaturated water received solute.
    assert!(
        chem[2][0] > 1.0e-5,
        "unsaturated water received nothing: {}",
        chem[2][0]
    );
    // Conserved overall.
    assert!(
        (inv - initial).abs() <= 1.0e-4 * initial.max(1.0),
        "solute leaked with a saturated neighbor: {initial} -> {inv}"
    );
}

/// `release_total = 0` guard: a fully-depleted grain (both pools 0) transfers nothing and produces
/// no NaN; adjacent water `c` stays 0.
#[test]
fn depleted_grain_no_transfer_no_nan() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([4.5, 4.0, 4.0], Species::Water),
        ],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &extract_isolation_config(), &gpu);
    let phase = solver.read_phases();
    prewet(&solver, &phase, 0.8, v_cap);
    seed_water_flow(&solver, &phase, 0.3);
    // Deplete the grain's pools.
    let mut chem = solver.read_chem();
    chem[0] = [0.0, 0.0, mats.pour_t, 0.0];
    solver.write_chem_for_test(&chem);

    for _ in 0..20 {
        solver.step(DT, &EmissionInput::default());
    }
    let chem = solver.read_chem();
    assert!(
        chem.iter().all(|c| c.iter().all(|v| v.is_finite())),
        "NaN/Inf in chem after depleted-grain run"
    );
    assert!(
        chem[1][0].abs() <= 1.0e-6,
        "water gained concentration from a depleted grain: {}",
        chem[1][0]
    );
}

/// Flux bridge: higher relative velocity extracts more; zero flow extracts (essentially) nothing.
#[test]
fn extraction_increases_with_flux() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([4.5, 4.0, 4.0], Species::Water),
        ],
        ..Scene::default()
    };
    // Returns total grain solute remaining after a few steps at relative speed `u`.
    let run = |u: f32| -> f32 {
        let mut solver = XpbdSolver::build(&scene, &mats, &extract_isolation_config(), &gpu);
        let phase = solver.read_phases();
        prewet(&solver, &phase, 0.8, v_cap);
        seed_water_flow(&solver, &phase, u);
        for _ in 0..5 {
            solver.step(DT, &EmissionInput::default());
        }
        let c = solver.read_chem();
        c[0][0] + c[0][1] // grain pools
    };
    let initial = {
        let (sf, ss) =
            extraction::split(mats.soluble_fraction * mats.grain_mass, mats.fast_fraction);
        sf + ss
    };
    let still = run(0.0);
    let slow = run(0.3);
    let fast = run(1.5);

    // Zero flow → flux gate ≈ 0 → essentially no extraction.
    assert!(
        (still - initial).abs() <= 1.0e-5,
        "zero-flow grain extracted: {still} vs {initial}"
    );
    // More flow → more solute released (less remaining in the grain).
    assert!(
        fast < slow - 1.0e-6 && slow < initial - 1.0e-6,
        "extraction not monotone in flux: initial {initial}, slow {slow}, fast {fast}"
    );
}

/// CPU/GPU parity: a single grain↔water pair locks the WGSL `diss_release`/`take` to the
/// `models/extraction.rs` reference functions.
#[test]
fn cpu_gpu_parity_single_pair() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_extraction: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let v_cap = capacity(&mats);
    let scene = Scene {
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            point([4.0, 4.0, 4.0], Species::Grain),
            point([4.5, 4.0, 4.0], Species::Water),
        ],
        ..Scene::default()
    };
    let cfg = extract_isolation_config();
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    let phase = solver.read_phases();
    assert_eq!(phase, vec![1, 0]);
    let sat_frac = 0.6;
    prewet(&solver, &phase, sat_frac, v_cap);
    let u = 0.5;
    seed_water_flow(&solver, &phase, u);
    let v_w = solver.water_particle_volume();

    solver.step(DT, &EmissionInput::default());
    let chem = solver.read_chem();

    // CPU reference for one step (N_w = N_g = 1, water c=0/f_w=1 ⇒ release-limited).
    let (s_f, s_s) = extraction::split(mats.soluble_fraction * mats.grain_mass, mats.fast_fraction);
    let kbase = cfg.extract_rate
        * extraction::arrhenius(mats.pour_t, mats.ea_over_r, mats.t_ref)
        * extraction::area_factor(mats.grain_diameter, mats.d_ref)
        * extraction::flux_factor(u, mats.u_half)
        * extraction::wet_gate(sat_frac, mats.s_on);
    let rf = extraction::release(s_f, kbase * mats.k0_fast, DT);
    let rs = extraction::release(s_s, kbase * mats.k0_slow, DT);
    let rtot = rf + rs;
    let headroom = mats.c_sat * v_w; // c=0, f_w=1
    let total = rtot.min(headroom);
    let exp_loss_f = total * rf / rtot;
    let exp_loss_s = total * rs / rtot;
    let exp_c = total / v_w;

    // Compare the transferred amounts (deltas), not the near-identical pool base values, with an
    // f32-appropriate relative tolerance (GPU and CPU `exp()` diverge by ~1e-4 relative).
    let gpu_loss_f = s_f - chem[0][0];
    let gpu_loss_s = s_s - chem[0][1];
    let rel = |gpu: f32, cpu: f32| (gpu - cpu).abs() <= 2.0e-3 * cpu.abs() + 1.0e-9;
    assert!(
        rel(gpu_loss_f, exp_loss_f),
        "fast pool transfer parity: gpu {gpu_loss_f} vs cpu {exp_loss_f}"
    );
    assert!(
        rel(gpu_loss_s, exp_loss_s),
        "slow pool transfer parity: gpu {gpu_loss_s} vs cpu {exp_loss_s}"
    );
    assert!(
        rel(chem[1][0], exp_c),
        "water concentration parity: gpu {} vs cpu {exp_c}",
        chem[1][0]
    );
    // Pair conservation: grain pool loss == water solute gain (exact frozen-snapshot transfer).
    // Absolute tol clears the f32 cancellation noise of `s_f - chem[0][0]` (two ~0.13 values);
    // tight global conservation is the job of `dissolution_conserves_solute_inventory`.
    let grain_loss = gpu_loss_f + gpu_loss_s;
    let water_gain = chem[1][0] * v_w;
    assert!(
        (grain_loss - water_gain).abs() <= 1.0e-6,
        "pair not conserved: grain lost {grain_loss}, water gained {water_gain}"
    );
}

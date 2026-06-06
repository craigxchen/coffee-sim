//! Phase 6 fines-migration invariants (GPU-gated; skips without an adapter).
//!
//! U1 scope (this file's current tests): fines live in the `chem.w` lane, seeded uniformly onto
//! grains (`fines_fraction · grain_volume`) and zero on water, and with the feature gated off
//! (`fines_rate = 0`) nothing disturbs them. U2+ add the erosion/deposition conservation gates.
//!
//! Note on ordering: the water-loop reorder permutes slot order each rebuild, so a single post-step
//! `read_chem()`/`read_phases()` snapshot aligns slot-for-slot, but two snapshots do not. The
//! inertness gate therefore asserts a **set/total** invariant, not a per-index byte compare.

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

/// A small mixed water+grain scene (water slab above a grain slab) so the coupling passes run.
fn mixed_scene() -> Scene {
    Scene {
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![
            SeedRegion {
                min: [2.0, 2.0, 2.0],
                max: [5.0, 3.0, 5.0],
                species: Species::Grain,
            },
            SeedRegion {
                min: [2.0, 4.0, 2.0],
                max: [5.0, 5.0, 5.0],
                species: Species::Water,
            },
        ],
        ..Scene::default()
    }
}

/// Materials with a non-zero fines inventory so seeding is observable.
fn fines_materials() -> Materials {
    Materials {
        fines_fraction: 0.1,
        ..Materials::default()
    }
}

/// The seeded per-grain fines volume the solver computes: `fines_fraction · (π/6)·d³`.
fn expected_fines_seed(mats: &Materials) -> f32 {
    let grain_volume = std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);
    coffee_sim::models::fines::fines_seed(grain_volume, mats.fines_fraction)
}

#[test]
fn fines_seeded_on_grains_zero_on_water() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_materials();
    let solver = XpbdSolver::build(&mixed_scene(), &mats, &Config::default(), &gpu);
    // No step yet → chem is still in seed order, aligned with phases.
    let chem = solver.read_chem();
    let phases = solver.read_phases();
    let seed = expected_fines_seed(&mats);
    assert!(seed > 0.0, "test setup: fines seed must be non-zero");

    let mut n_grain = 0usize;
    for (i, (c, &ph)) in chem.iter().zip(&phases).enumerate() {
        if ph == 1 {
            n_grain += 1;
            assert!(
                (c[3] - seed).abs() < 1e-6,
                "grain {i}: chem.w = {} expected fines_seed {seed}",
                c[3]
            );
        } else {
            assert!(c[3].abs() < 1e-6, "water {i}: chem.w = {} expected 0", c[3]);
        }
    }
    assert!(n_grain > 0, "scene must contain grains");
}

#[test]
fn fines_inert_when_rate_zero() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_fines: no GPU adapter; skipping.");
        return;
    };
    let mats = fines_materials();
    // Default config: coupling/drag run, but fines_rate = 0 → no fines transfer pass.
    let mut solver = XpbdSolver::build(&mixed_scene(), &mats, &Config::default(), &gpu);
    let seed = expected_fines_seed(&mats);

    // Total grain fines on the books before any step.
    let n_grain = solver.read_phases().iter().filter(|&&p| p == 1).count();
    let total_before = n_grain as f32 * seed;

    for _ in 0..30 {
        solver.step(DT, &EmissionInput::default());
    }

    // Single post-step snapshot: chem[i] and phase[i] align (set-invariant total).
    let chem = solver.read_chem();
    let phases = solver.read_phases();
    let mut total_grain = 0.0f32;
    let mut max_water = 0.0f32;
    for (c, &ph) in chem.iter().zip(&phases) {
        if ph == 1 {
            total_grain += c[3];
        } else {
            max_water = max_water.max(c[3].abs());
        }
    }
    assert!(
        (total_grain - total_before).abs() < 1e-4 * total_before.max(1.0),
        "grain fines total drifted with fines off: {total_grain} vs {total_before}"
    );
    assert!(
        max_water < 1e-6,
        "water gained suspended fines with fines off: max {max_water}"
    );
}

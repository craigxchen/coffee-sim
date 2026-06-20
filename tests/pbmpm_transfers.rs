//! PB-MPM U3 transfer gates (light, per the plan's visual-first posture KTD3): one APIC
//! mass+momentum round-trip sanity check and one fixed-point overflow probe. The bounce metric,
//! conservation gate, and perf gate live in the shared Phase-B harness (U6/U7), not here.
//!
//! GPU-gated headless idiom — each test skips gracefully without an adapter, never `#[ignore]`.

mod common;

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::{PbmpmSolver, FP_SCALE};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::rng::Rng;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

fn water_scene(
    box_min: [f32; 3],
    box_max: [f32; 3],
    gravity: [f32; 3],
    rmin: [f32; 3],
    rmax: [f32; 3],
) -> Scene {
    Scene {
        gravity,
        box_min,
        box_max,
        regions: vec![SeedRegion {
            min: rmin,
            max: rmax,
            species: Species::Water,
        }],
        solids: Vec::new(),
        ..Scene::default()
    }
}

fn all_finite(rows: &[[f32; 4]]) -> bool {
    rows.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

/// P2G round trip (gravity off): the grid totals equal the particle totals — mass and momentum
/// to the documented per-contribution fixed-point rounding bound. The integer accumulation
/// itself is exact, so a reset + identical re-run reproduces the grid totals BIT-IDENTICALLY
/// (what fixed-point buys over float atomics). A random interior velocity field (well under the
/// cap, no scatter clamp) on an interior block keeps every particle's 27-node stencil in range.
#[test]
fn p2g_g2p_round_trip_conserves_mass_and_momentum() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_transfers: no GPU adapter; skipping.");
        return;
    };
    let scene = water_scene(
        [0.0; 3], [32.0; 3],
        [0.0; 3], // gravity OFF: grid momentum must equal particle momentum
        [13.0; 3], [18.0; 3], // 6³ randomized (seed_jitter) block, interior
    );
    let cfg = Config::default();
    let mats = Materials::default();
    let mut solver = PbmpmSolver::build(&scene, &mats, &cfg, &gpu);
    // Keep this a PURE transfer check (U4): one iteration + the compliant constraint OFF
    // (relaxation 0 ⇒ no volume correction; viscosity 0 ⇒ no shear), so `step()` is exactly one
    // grid_clear → p2g → grid_update → g2p cycle and the grid totals are the particle totals to the
    // single-scatter rounding bound. The iteration loop + constraint physics are gated visually/in
    // the shared harness (U6/U7), not here.
    solver.set_iteration_count_for_test(1);
    solver.set_liquid_relaxation_for_test(0.0);
    solver.set_liquid_viscosity_for_test(0.0);
    let n = solver.read_positions().len();
    assert_eq!(n, 216, "6³ interior water block seeds 216 particles");

    // Seeded random velocities, well under the cap (no scatter clamp may fire here).
    let mut rng = Rng::new(0xA11CE);
    let vels: Vec<[f32; 4]> = (0..n)
        .map(|_| {
            [
                rng.next_f32() * 10.0 - 5.0,
                rng.next_f32() * 10.0 - 5.0,
                rng.next_f32() * 10.0 - 5.0,
                0.0,
            ]
        })
        .collect();
    solver.write_velocities_for_test(&vels);
    solver.step(DT, &EmissionInput::default());

    // Quantization bound: each particle scatters 27 contributions, each rounded to the nearest
    // count (≤ 0.5 counts of error), so each total lane is within n·27·0.5 counts of the exact
    // value. The SUM itself is exact integer arithmetic — fixed point is what makes that true.
    let bound = n as f64 * 27.0 * 0.5 / FP_SCALE;
    let (mass, mom) = solver.read_grid_mass_momentum();
    let m = mats.particle_mass as f64;
    assert!(
        (mass - n as f64 * m).abs() <= bound,
        "grid mass {mass} vs Σm {} (bound {bound})",
        n as f64 * m
    );
    let mut expect_mom = [0.0f64; 3];
    for v in &vels {
        for a in 0..3 {
            expect_mom[a] += m * v[a] as f64;
        }
    }
    for a in 0..3 {
        assert!(
            (mom[a] - expect_mom[a]).abs() <= bound,
            "grid momentum axis {a}: {} vs {} (bound {bound})",
            mom[a],
            expect_mom[a]
        );
    }

    // Fixed-point exactness: a reset + identical re-run reproduces the grid totals BIT-IDENTICALLY
    // (integer adds commute; float atomics would not survive this — also pins determinism R6).
    let counts1 = solver.read_grid_counts();
    solver.reset(&scene);
    solver.write_velocities_for_test(&vels);
    solver.step(DT, &EmissionInput::default());
    let counts2 = solver.read_grid_counts();
    assert_eq!(counts1, counts2, "fixed-point scatter must be bit-exact");
}

/// Fixed-point overflow probe (R7 surface (a) — the GRID momentum lanes): a fast/high-velocity
/// configuration with every particle slammed at the velocity cap into a pancaked cluster keeps
/// the grid momentum lanes within FP_CLAMP (no i32 saturation), state stays finite, and the
/// measured per-lane headroom is reported. Headroom math: see `FP_SCALE` in `common.wgsl` — the
/// scatter clamp bounds momentum counts by (Σ m·w)·max_speed·2^18, so overflow needs ≈20× rest
/// compression with every particle at the cap.
#[test]
fn fixed_point_overflow_probe_at_velocity_cap() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_transfers: no GPU adapter; skipping.");
        return;
    };
    let scene = water_scene(
        [0.0; 3],
        [24.0; 3],
        [0.0, -20.0, 0.0],
        [7.0, 1.0, 7.0],
        [16.0, 10.0, 16.0], // 10³ = 1000 particles dropped low
    );
    let cfg = Config::default();
    let mut solver = PbmpmSolver::build(&scene, &Materials::default(), &cfg, &gpu);
    let n = solver.read_positions().len();
    let input = EmissionInput::default();
    // Pancake the block on the floor — pressureless U3 water compacts (no density constraint yet),
    // which IS the worst-case clustering for a node's mass loading.
    for _ in 0..300 {
        solver.step(DT, &input);
    }
    // Slam every particle at the velocity cap into the cluster.
    let cap = cfg.max_speed;
    solver.write_velocities_for_test(&vec![[0.0, -cap, 0.0, 0.0]; n]);
    for _ in 0..3 {
        solver.step(DT, &input);
    }

    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    assert!(all_finite(&pos) && all_finite(&vel), "non-finite state");
    for v in &vel {
        let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(s <= cap + 1e-3, "speed {s} above the cap");
    }
    // No wrap: an overflowed lane would corrupt the mass total far beyond the rounding bound.
    let (mass, _) = solver.read_grid_mass_momentum();
    let bound = n as f64 * 27.0 * 0.5 / FP_SCALE;
    assert!(
        mass > 0.0,
        "grid carries mass (discriminates the inert scaffold)"
    );
    assert!(
        (mass - n as f64).abs() <= bound,
        "grid mass {mass} vs {n} (bound {bound}) — fixed-point lane wrapped?"
    );
    let max_count = solver.read_grid_max_count();
    let headroom = i32::MAX as f64 / max_count as f64;
    println!(
        "pbmpm U3 overflow probe: max |lane| {max_count} counts, headroom {headroom:.1}× at the cap"
    );
    assert!(
        headroom >= 2.0,
        "fixed-point headroom collapsed: {headroom:.2}×"
    );
}

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
use coffee_sim::utils::sdf::{SdfPrimitive, SolidKind, MASK_ALL};
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

/// A cup (cylinder cavity) inside a box, with a SINGLE water particle seeded high in the cavity,
/// well clear of the floor. The cup floor at `floor_y` is the known surface the collider BC acts
/// on. `interior_y` is the seed height; the region is a point (hi == lo) so seed_water emits exactly
/// one particle. The box is generous so the box-clamp backstop never engages on the test motion.
fn cup_with_one_particle(floor_y: f32, interior_y: f32) -> (Scene, [f32; 3]) {
    let center = [16.0_f32, 0.0, 16.0];
    let seed = [center[0], interior_y, center[2]];
    let scene = Scene {
        gravity: [0.0; 3], // gravity OFF: the test drives velocity explicitly
        box_min: [0.0; 3],
        box_max: [32.0; 3],
        regions: vec![SeedRegion {
            min: seed,
            max: seed, // point region → exactly one seeded particle
            species: Species::Water,
        }],
        solids: vec![SdfPrimitive {
            kind: SolidKind::Cylinder {
                center: glam::Vec3::new(center[0], 0.0, center[2]),
                floor_y,
                rim_y: 30.0,
                radius: 8.0,
            },
            species_mask: MASK_ALL,
            friction: 0.0,
        }],
        ..Scene::default()
    };
    (scene, seed)
}

/// Collider BC smoke test (U5; light, per KTD3): a particle driven INTO the cup floor in one step
/// is pushed back to the floor surface, its into-floor (normal) velocity is reflected by the
/// restitution coefficient, and its tangential velocity is preserved. The SDF sign convention is
/// interior-POSITIVE, so the push-out keeps the particle in the cup CAVITY (above the floor), never
/// expelling it. Catches SDF/sign and reflection-math bugs before the U7 bounce measurement.
///
/// To isolate `particle_integrate`'s particle-resolution push-out from the grid node BC, the
/// particle starts well ABOVE the floor (> one cell h = 2·spacing) so no node is within the band
/// during grid_update; its large downward velocity then carries `x + v·dt` BELOW the floor in the
/// single advect, where the push-out + restitution fire. The compliant constraint is OFF
/// (relaxation/viscosity 0, one iteration) so the gathered velocity round-trips unchanged.
#[test]
fn collider_bc_pushes_out_and_reflects_by_restitution() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("pbmpm_transfers: no GPU adapter; skipping.");
        return;
    };
    let floor_y = 8.0_f32;
    let start_y = 16.0_f32; // 8 above the floor = 4 cells (h = 2): clear of the node band
    let (scene, seed) = cup_with_one_particle(floor_y, start_y);
    // Raise the velocity cap well above the test velocity so the scatter/gather clamp never scales
    // the seeded velocity down (which would corrupt the tangential-preservation check). The point
    // here is the collider BC, not the cap (its own probe is above).
    let cfg = Config {
        max_speed: 5000.0,
        ..Config::default()
    };
    let mats = Materials::default();

    // Drive the particle straight down fast enough to cross the floor in one dt, plus a tangential
    // (x) component that must survive (free slip). vy is chosen so `start_y + vy·dt` lands 4 below
    // the floor in the single advect; the particle starts > one cell (h = 2) above the floor so the
    // grid node BC never engages on its stencil — only particle_integrate's push-out fires.
    let vy = -((start_y - floor_y) + 4.0) / DT; // lands 4 below the floor in one advect
    let vx = 3.0_f32;
    let restitution = 0.5_f32; // an exaggerated value so the reflection is unambiguous in the test

    let run = |rest: f32| -> ([f32; 4], [f32; 4]) {
        let mut solver = PbmpmSolver::build(&scene, &mats, &cfg, &gpu);
        // Pure transfer + collider BC: no constraint correction, one iteration.
        solver.set_iteration_count_for_test(1);
        solver.set_liquid_relaxation_for_test(0.0);
        solver.set_liquid_viscosity_for_test(0.0);
        solver.set_restitution_for_test(rest);
        let n = solver.read_positions().len();
        assert_eq!(n, 1, "point region seeds exactly one particle");
        // Seed position is preserved; overwrite the velocity (length = particle_count = 1).
        solver.write_velocities_for_test(&[[vx, vy, 0.0, 0.0]]);
        solver.step(DT, &EmissionInput::default());
        (solver.read_positions()[0], solver.read_velocities()[0])
    };

    // restitution > 0: rebound (normal velocity flips, damped by the coefficient).
    let (pos, vel) = run(restitution);
    assert!(
        all_finite(&[pos]) && all_finite(&[vel]),
        "non-finite state after BC"
    );
    // Pushed OUT of the wall material to (or above) the cup floor surface — stays in the cavity.
    assert!(
        pos[1] >= floor_y - 1e-3,
        "particle pushed back to/above the floor (cavity side): y={} floor={}",
        pos[1],
        floor_y
    );
    // The seed x/z are unchanged by the advect on a pure-vertical-then-tangential motion only by vx;
    // tangential velocity preserved (free slip — restitution acts on the NORMAL only).
    assert!(
        (vel[0] - vx).abs() < 1e-3,
        "tangential (x) velocity preserved: {} vs {vx}",
        vel[0]
    );
    assert!(vel[2].abs() < 1e-3, "no spurious z velocity: {}", vel[2]);
    // Normal (y) velocity reflected: v_n_out = −restitution·v_n_in (the floor normal is +y, v_n_in
    // = vy < 0, so v_n_out = −0.5·vy > 0, an upward rebound).
    let expect_vy = -restitution * vy;
    assert!(
        (vel[1] - expect_vy).abs() < 1e-2 * expect_vy.abs().max(1.0),
        "normal velocity reflected by restitution: {} vs −{}·{} = {}",
        vel[1],
        restitution,
        vy,
        expect_vy
    );
    let _ = seed;

    // restitution = 0: free-slip stop (the constraint-only arm). Normal velocity killed, no rebound.
    let (pos0, vel0) = run(0.0);
    assert!(
        pos0[1] >= floor_y - 1e-3,
        "restitution=0 still pushes out: y={}",
        pos0[1]
    );
    assert!(
        vel0[1].abs() < 1e-3,
        "restitution=0 kills the into-floor normal velocity (no rebound): {}",
        vel0[1]
    );
    assert!(
        (vel0[0] - vx).abs() < 1e-3,
        "restitution=0 still preserves tangential velocity: {}",
        vel0[0]
    );
    println!(
        "pbmpm U5 collider BC: push-out to floor OK; restitution 0.5 → v_y {:.3} (expect {:.3}); \
         restitution 0 → v_y {:.3} (stop)",
        vel[1], expect_vy, vel0[1]
    );
}

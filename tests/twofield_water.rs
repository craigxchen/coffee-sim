//! Twofield U2 gates: APIC water transfers (P2G/G2P), grid gravity, boundary conditions.
//!
//! GPU-gated headless idiom — each test skips gracefully without an adapter, never
//! `#[ignore]`. CPU twins of the B-spline weight kernel and a one-step P2G/G2P reference live
//! at the bottom of this file and pin the GPU kernels.
//!
//! The free-fall reference is the solver's OWN discrete recurrence (see `transfers.wgsl`):
//! per frame, P2G scatters at the old positions, the grid applies `v ← v + g·dt`, and G2P
//! advects `x ← x + v_new·dt` — semi-implicit Euler, i.e. `common::freefall_reference` with
//! `substeps = 1`. The weights partition unity and are linearly consistent, so the gather is
//! exact for the uniform free-fall field and C stays ~0.

// Axis loops (`for a in 0..3`) index several parallel arrays (positions, box bounds, grid
// dims); the iterator rewrite clippy suggests obscures that symmetry.
#![allow(clippy::needless_range_loop)]

mod common;

use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::twofield::{TwofieldSolver, FP_SCALE};
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::utils::rng::Rng;
use coffee_sim::utils::sdf;
use coffee_sim::EmissionInput;

use common::{assert_close3, continuous_reference, freefall_reference};

const DT: f32 = 1.0 / 60.0;
const SPACING: f32 = 1.0; // Materials::default().particle_spacing

// Round-off budgets for the free-fall gate (NOT physics bands). Error sources per frame:
// fixed-point quantization of the node momentum/mass (≤ 0.5 counts per contribution against
// node masses of ~10⁵ counts → per-step velocity error ~1e-5) and f32 round-off in the
// transfers. Accumulated linearly over 120 frames that stays ≤ ~2e-3; the budgets carry ~10×
// headroom and sit ~10× below the ≈0.33 discrete-vs-continuous discrimination gap.
const POS_ABS_TOL: f64 = 2e-2;
const VEL_ABS_TOL: f64 = 2e-2;
const REL_TOL: f64 = 1e-5;

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

fn xyz(p: [f32; 4]) -> [f32; 3] {
    [p[0], p[1], p[2]]
}

fn f64v(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn all_finite(rows: &[[f32; 4]]) -> bool {
    rows.iter().all(|r| r.iter().all(|x| x.is_finite()))
}

/// Free-fall: a small water block far from any wall matches the solver's own discrete
/// recurrence (derived in the header comment) within the documented f32 budget over 120
/// frames, and is discriminably DIFFERENT from the continuous ½gt² form.
#[test]
fn freefall_matches_discrete_recurrence_not_continuous() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_water: no GPU adapter; skipping.");
        return;
    };
    const GRAVITY: [f32; 3] = [0.0, -20.0, 0.0];
    const V0: [f32; 3] = [2.0, 20.0, 0.0]; // up + sideways keeps the excursion bounded (~10 units)
    const FRAMES: u32 = 120;
    let scene = water_scene(
        [-12.0, -12.0, -12.0],
        [12.0, 28.0, 12.0],
        GRAVITY,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0], // 3³ block centered at the origin
    );
    let cfg = Config {
        seed_jitter: 0.0, // exact lattice seed: per-particle reference = seed + shared offset
        ..Config::default()
    };
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &cfg, &gpu);
    let seed = solver.read_positions();
    assert_eq!(seed.len(), 27, "3³ water block");
    solver.write_velocities_for_test(&vec![[V0[0], V0[1], V0[2], 0.0]; seed.len()]);

    let input = EmissionInput::default();
    let (v0, g) = (f64v(V0), f64v(GRAVITY));
    for f in 1..=FRAMES {
        solver.step(DT, &input);
        if f % 30 != 0 {
            continue;
        }
        // Every particle shares the same offset/velocity: reference from x0 = 0 is the offset.
        let (dpos, rvel) = freefall_reference([0.0; 3], v0, g, DT as f64, 1, f);
        let pos = solver.read_positions();
        let vel = solver.read_velocities();
        for (i, (p, s)) in pos.iter().zip(&seed).enumerate() {
            // Regime guard: no box clamp may fire, else we'd compare a corrected trajectory.
            for a in 0..3 {
                assert!(
                    p[a] > scene.box_min[a] + 1.0 && p[a] < scene.box_max[a] - 1.0,
                    "f={f} particle {i} left the interior on axis {a}: {}",
                    p[a]
                );
            }
            let expect = [
                s[0] as f64 + dpos[0],
                s[1] as f64 + dpos[1],
                s[2] as f64 + dpos[2],
            ];
            assert_close3(
                xyz(*p),
                expect,
                POS_ABS_TOL,
                REL_TOL,
                &format!("pos f={f} i={i}"),
            );
            assert_close3(
                xyz(vel[i]),
                rvel,
                VEL_ABS_TOL,
                REL_TOL,
                &format!("vel f={f} i={i}"),
            );
        }
    }

    // Discrimination gap: the observed trajectory must be far (≫ budget) from the continuous
    // ½gt² form — the gap at 120 frames is ½·g·dt²·m ≈ 0.33 sim-units on y.
    let (dpos, _) = freefall_reference([0.0; 3], v0, g, DT as f64, 1, FRAMES);
    let cont = continuous_reference([0.0; 3], v0, g, DT as f64, FRAMES);
    let observed_dy = solver.read_positions()[0][1] as f64 - seed[0][1] as f64;
    assert!(
        (observed_dy - dpos[1]).abs() <= POS_ABS_TOL,
        "matches discrete"
    );
    let gap = (observed_dy - cont[1]).abs();
    assert!(
        gap > 0.1,
        "continuous ½gt² is only {gap} from observed — the test no longer discriminates"
    );
}

/// P2G round trip (gravity off): the grid totals equal the particle totals — mass to the
/// documented per-contribution rounding bound, momentum likewise — the integer accumulation
/// itself is exact (bit-identical across a reset/rerun, unlike float atomics), and the
/// one-step G2P result pins to the CPU twin.
#[test]
fn p2g_g2p_round_trip_conserves_mass_and_momentum() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_water: no GPU adapter; skipping.");
        return;
    };
    let scene = water_scene(
        [0.0; 3], [32.0; 3],
        [0.0; 3], // gravity OFF: grid momentum must equal particle momentum
        [13.0; 3], [18.0; 3], // 6³ randomized (seed_jitter) block, interior
    );
    let cfg = Config::default();
    let mats = Materials::default();
    let mut solver = TwofieldSolver::build(&scene, &mats, &cfg, &gpu);
    let seed = solver.read_positions();
    let n = seed.len();
    assert_eq!(n, 216);

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

    // CPU twin pin: the GPU one-step G2P result (gravity off, interior, C₀ = 0) matches the
    // float-exact CPU reference within a budget covering fixed-point quantization + f32.
    let (origin, h, dims) = solver.grid_spec();
    let (cpu_pos, cpu_vel) = cpu_transfer_step(&seed, &vels, origin, h, dims, m, DT as f64);
    let gpos = solver.read_positions();
    let gvel = solver.read_velocities();
    for i in 0..n {
        assert_close3(
            xyz(gpos[i]),
            cpu_pos[i],
            5e-3,
            1e-4,
            &format!("twin pos {i}"),
        );
        assert_close3(
            xyz(gvel[i]),
            cpu_vel[i],
            5e-3,
            1e-4,
            &format!("twin vel {i}"),
        );
    }

    // Fixed-point exactness: a reset + identical re-run reproduces the grid totals
    // BIT-IDENTICALLY (integer adds commute; float atomics would not survive this).
    let counts1 = solver.read_grid_counts();
    solver.reset(&scene);
    solver.write_velocities_for_test(&vels);
    solver.step(DT, &EmissionInput::default());
    let counts2 = solver.read_grid_counts();
    assert_eq!(counts1, counts2, "fixed-point scatter must be bit-exact");
}

/// APIC vs PIC discrimination: a rigidly rotating blob (v = ω × r with the matching affine C)
/// retains its angular momentum materially better under APIC than under the PIC variant
/// (C zeroed each step). The gate asserts the GAP, not absolutes.
#[test]
fn apic_retains_angular_momentum_better_than_pic() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_water: no GPU adapter; skipping.");
        return;
    };
    const OMEGA: f32 = 1.5; // about +y; edge speed ≈ 6.6 ≪ the cap
    const STEPS: u32 = 60;
    let scene = water_scene([0.0; 3], [32.0; 3], [0.0; 3], [13.0; 3], [19.0; 3]); // 7³ blob
    let cfg = Config::default();
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &cfg, &gpu);
    let seed = solver.read_positions();
    let n = seed.len();
    let center = mass_center(&seed);

    let seed_rotation = |solver: &TwofieldSolver| {
        let mut vels = Vec::with_capacity(n);
        let mut rows = Vec::with_capacity(3 * n);
        for p in &seed {
            let r = [p[0] - center[0], p[2] - center[2]];
            vels.push([OMEGA * r[1], 0.0, -OMEGA * r[0], 0.0]);
        }
        for _ in 0..n {
            // C = ∂v/∂x for v = ω×r about y: rows (0,0,ω), (0,0,0), (−ω,0,0).
            rows.push([0.0, 0.0, OMEGA, 0.0]);
            rows.push([0.0, 0.0, 0.0, 0.0]);
            rows.push([-OMEGA, 0.0, 0.0, 0.0]);
        }
        solver.write_velocities_for_test(&vels);
        solver.write_affine_for_test(&rows);
    };
    let run = |solver: &mut TwofieldSolver| -> f64 {
        seed_rotation(solver);
        let l0 = angular_momentum_y(&solver.read_positions(), &solver.read_velocities());
        assert!(l0 > 1.0, "seeded rotation has angular momentum");
        for _ in 0..STEPS {
            solver.step(DT, &EmissionInput::default());
        }
        angular_momentum_y(&solver.read_positions(), &solver.read_velocities()) / l0
    };

    let apic_retention = run(&mut solver);
    solver.reset(&scene);
    solver.set_pic_for_test(true);
    let pic_retention = run(&mut solver);
    println!(
        "twofield U2 APIC gate: L_y retention over {STEPS} steps — APIC {apic_retention:.3}, PIC {pic_retention:.3}"
    );
    // Measured on Apple M5: APIC ≈ 1.15, PIC ≈ 0.00. APIC retention slightly ABOVE 1 is the
    // explicit forward-Euler advection spiraling the blob's radius outward per step (×≈1+½(ωdt)²)
    // while the affine field keeps v = ω×r — L grows with r. Bounded (the soak gate covers
    // long-run stability); the gate asserts the APIC-vs-PIC gap, not absolutes.
    assert!(
        apic_retention > pic_retention + 0.2,
        "APIC must retain materially more angular momentum than PIC \
         (APIC {apic_retention:.3} vs PIC {pic_retention:.3})"
    );
}

/// Fixed-point overflow probe: a pancaked (worst-case clustered) block with every particle at
/// the velocity cap stays finite and bounded, the grid totals show no i32 wrap, and the
/// measured per-lane headroom is reported. Headroom math: see `FP_SCALE` in `common.wgsl` —
/// the scatter clamp bounds momentum counts by (Σ m·w)·max_speed·2^18, overflow needs ≈164
/// mass units on one node (≈20× rest compression at the cap).
#[test]
fn fixed_point_overflow_probe_at_velocity_cap() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_water: no GPU adapter; skipping.");
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
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &cfg, &gpu);
    let n = solver.read_positions().len();
    let input = EmissionInput::default();
    // Pancake the block on the floor — pressureless U2 water compacts, which IS the worst-case
    // clustering for a node's mass loading.
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
    // No wrap: an overflowed lane would corrupt the totals far beyond the rounding bound.
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
        "twofield U2 overflow probe: max |lane| {max_count} counts, headroom {headroom:.1}× at the cap"
    );
    assert!(
        headroom >= 2.0,
        "fixed-point headroom collapsed: {headroom:.2}×"
    );
}

/// Grid + particle boundary treatment, box walls: a dropped block settles with no particle
/// outside the domain beyond the penetration bound.
#[test]
fn wall_bc_closed_box_no_penetration() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_water: no GPU adapter; skipping.");
        return;
    };
    let scene = water_scene(
        [0.0; 3],
        [32.0; 3],
        [0.0, -20.0, 0.0],
        [8.0, 12.0, 8.0],
        [20.0, 24.0, 20.0], // 13³ block dropped from mid-air
    );
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let input = EmissionInput::default();
    for _ in 0..600 {
        solver.step(DT, &input);
    }
    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    assert!(all_finite(&pos) && all_finite(&vel));
    let bound = 0.25 * SPACING;
    let mut min_y = f32::INFINITY;
    for p in &pos {
        for a in 0..3 {
            assert!(
                p[a] >= scene.box_min[a] - bound && p[a] <= scene.box_max[a] + bound,
                "particle outside the box beyond the {bound} bound: {:?}",
                p
            );
        }
        min_y = min_y.min(p[1]);
    }
    // Discriminators: the water actually fell to the floor and is (near-)settled.
    assert!(
        min_y <= scene.box_min[1] + 1.0,
        "block never reached the floor: min_y {min_y}"
    );
    let mean_speed: f32 = vel
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .sum::<f32>()
        / vel.len() as f32;
    println!("twofield U2 box settle: min_y {min_y:.3}, mean speed {mean_speed:.3}");
    assert!(
        mean_speed < 2.0,
        "block did not settle: mean speed {mean_speed}"
    );
}

/// Grid + particle boundary treatment, V60 cone (Scene::v60-style solids): water dropped into
/// the dripper never penetrates the solid beyond the bound, and visibly drains downward.
#[test]
fn wall_bc_v60_cone_no_penetration() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_water: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        gravity: [0.0, -20.0, 0.0],
        box_min: [-7.0, -10.0, -7.0],
        box_max: [7.0, 10.0, 7.0],
        solids: coffee_sim::utils::geometry::v60_dripper(),
        regions: vec![SeedRegion {
            min: [-1.5, 0.8, -1.5],
            max: [1.5, 2.4, 1.5],
            species: Species::Water,
        }],
        ..Scene::default()
    };
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let seed = solver.read_positions();
    assert!(!seed.is_empty(), "water seeds inside the cone");
    let mean_y0: f32 = seed.iter().map(|p| p[1]).sum::<f32>() / seed.len() as f32;

    let input = EmissionInput::default();
    for _ in 0..400 {
        solver.step(DT, &input);
    }
    let pos = solver.read_positions();
    assert!(all_finite(&pos));
    let bound = 0.25 * SPACING;
    for p in &pos {
        for a in 0..3 {
            assert!(
                p[a] >= scene.box_min[a] - bound && p[a] <= scene.box_max[a] + bound,
                "particle outside the box: {:?}",
                p
            );
        }
        // Water phase (0): the support cone + cup constrain it; the grains-only filter does not.
        let c = sdf::nearest(&scene.solids, glam::Vec3::new(p[0], p[1], p[2]), 0);
        assert!(
            c.signed > -bound,
            "particle inside a solid by {} (> bound {bound}) at {:?}",
            -c.signed,
            p
        );
    }
    // Discriminator: the water visibly drained downward (the inert scaffold can't).
    let mean_y: f32 = pos.iter().map(|p| p[1]).sum::<f32>() / pos.len() as f32;
    println!("twofield U2 v60: mean y {mean_y0:.2} → {mean_y:.2} over 400 frames");
    assert!(
        mean_y < mean_y0 - 1.0,
        "water never drained: mean y {mean_y0} → {mean_y}"
    );
}

/// 1000-step no-NaN soak on a dam-break-shaped fall: state stays finite and in-domain, and the
/// column visibly collapses.
#[test]
fn soak_1000_steps_dam_break_no_nan() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("twofield_water: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::dam_break();
    let mut solver = TwofieldSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let seed = solver.read_positions();
    let max_y0 = seed.iter().map(|p| p[1]).fold(f32::MIN, f32::max);

    let input = EmissionInput::default();
    for f in 1..=1000u32 {
        solver.step(DT, &input);
        if f % 250 == 0 {
            assert!(
                all_finite(&solver.read_positions()) && all_finite(&solver.read_velocities()),
                "non-finite state at frame {f}"
            );
        }
    }
    let pos = solver.read_positions();
    let bound = 0.25 * SPACING;
    let mut max_y = f32::MIN;
    for p in &pos {
        for a in 0..3 {
            assert!(
                p[a] >= scene.box_min[a] - bound && p[a] <= scene.box_max[a] + bound,
                "particle escaped the box: {:?}",
                p
            );
        }
        max_y = max_y.max(p[1]);
    }
    println!("twofield U2 soak: column max y {max_y0:.2} → {max_y:.2} over 1000 frames");
    assert!(max_y < max_y0 - 2.0, "the column never collapsed (inert?)");
}

// --- CPU twins (pin the GPU kernels) ----------------------------------------------------------

/// Quadratic B-spline weights per axis (mirrors `bspline_w` in common.wgsl / KEEP.md §3):
/// fx ∈ [0.5, 1.5] relative to base = floor(x/h − 0.5); offsets k ∈ {0,1,2}.
fn bspline_w(fx: f64) -> [f64; 3] {
    [
        0.5 * (1.5 - fx) * (1.5 - fx),
        0.75 - (fx - 1.0) * (fx - 1.0),
        0.5 * (fx - 0.5) * (fx - 0.5),
    ]
}

/// CPU reference of one transfer step (gravity off, no boundaries, C₀ = 0): P2G mass/momentum
/// in f64 (no fixed-point quantization), node velocity = momentum/mass, G2P gather + advect.
/// Valid for interior particles only (no clamps modeled).
fn cpu_transfer_step(
    pos: &[[f32; 4]],
    vel: &[[f32; 4]],
    origin: [f32; 3],
    h: f32,
    dims: [u32; 3],
    m: f64,
    dt: f64,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let mut mass = vec![0.0f64; nx * ny * nz];
    let mut mom = vec![[0.0f64; 3]; nx * ny * nz];

    let weights_of = |p: &[f32; 4]| {
        let mut base = [0usize; 3];
        let mut w = [[0.0f64; 3]; 3];
        for a in 0..3 {
            let xl = (p[a] - origin[a]) as f64 / h as f64;
            let b = (xl - 0.5).floor();
            base[a] = b as usize;
            w[a] = bspline_w(xl - b);
        }
        (base, w)
    };

    for (p, v) in pos.iter().zip(vel) {
        let (base, w) = weights_of(p);
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let wijk = w[0][i] * w[1][j] * w[2][k];
                    let n = idx(base[0] + i, base[1] + j, base[2] + k);
                    mass[n] += m * wijk;
                    for a in 0..3 {
                        mom[n][a] += m * wijk * v[a] as f64;
                    }
                }
            }
        }
    }

    let mut out_pos = Vec::with_capacity(pos.len());
    let mut out_vel = Vec::with_capacity(pos.len());
    for p in pos {
        let (base, w) = weights_of(p);
        let mut vp = [0.0f64; 3];
        for k in 0..3 {
            for j in 0..3 {
                for i in 0..3 {
                    let wijk = w[0][i] * w[1][j] * w[2][k];
                    let n = idx(base[0] + i, base[1] + j, base[2] + k);
                    if mass[n] > 0.0 {
                        for a in 0..3 {
                            vp[a] += wijk * mom[n][a] / mass[n];
                        }
                    }
                }
            }
        }
        out_pos.push([
            p[0] as f64 + vp[0] * dt,
            p[1] as f64 + vp[1] * dt,
            p[2] as f64 + vp[2] * dt,
        ]);
        out_vel.push(vp);
    }
    (out_pos, out_vel)
}

fn mass_center(pos: &[[f32; 4]]) -> [f32; 3] {
    let n = pos.len() as f32;
    let mut c = [0.0f32; 3];
    for p in pos {
        for a in 0..3 {
            c[a] += p[a];
        }
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// Total angular momentum about +y through the current mass center (unit particle mass).
fn angular_momentum_y(pos: &[[f32; 4]], vel: &[[f32; 4]]) -> f64 {
    let c = mass_center(pos);
    pos.iter()
        .zip(vel)
        .map(|(p, v)| ((p[2] - c[2]) * v[0] - (p[0] - c[0]) * v[2]) as f64)
        .sum()
}

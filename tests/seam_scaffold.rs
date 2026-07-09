//! Seam-blend U1 scaffold gates: composition contract + byte-identical off-switches +
//! the bed water-count invariant + the merged render layout (structural only).
//!
//! Plan: docs/plans/2026-07-09-002-feat-seam-blend-m0-plan.md (U1). Per KTD7 there are NO
//! seam-physics assertions here — the R1–R3 gates land with U2–U4 after the visual oracle.
//! GPU-gated: these skip gracefully on hosts without an adapter.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::registry::{build_solver, SolverId};
use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::pbmpm::PbmpmSolver;
use coffee_sim::solvers::seam::SeamSolver;
use coffee_sim::solvers::twofield::TwofieldSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;

fn water_block() -> SeedRegion {
    SeedRegion {
        min: [1.0, 4.0, 1.0],
        max: [4.0, 7.0, 4.0],
        species: Species::Water,
    }
}

fn grain_block() -> SeedRegion {
    SeedRegion {
        min: [1.0, 1.0, 1.0],
        max: [4.0, 3.0, 4.0],
        species: Species::Grain,
    }
}

/// Registry arm + empty scene: one step leaves finite empty state and both inner
/// pipelines dispatch (the composition is real even at zero particles).
#[test]
fn builds_via_registry_and_steps_an_empty_scene() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        regions: vec![],
        ..Scene::default()
    };
    let mut solver = build_solver(
        SolverId::Seam,
        &scene,
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    solver.step(DT, &EmissionInput::default());
    assert_eq!(solver.metrics().particle_count, 0);
    assert_eq!(solver.particles().particle_count, 0);
    assert!(
        solver.profile().dispatches_per_frame > 0,
        "both inner pipelines dispatch even when empty"
    );
}

/// Byte-identical off-switch (R4, water side): a water-only scene through the seam steps
/// the pbmpm inner exactly as pbmpm solo — and the seam itself is deterministic on a twin.
#[test]
fn water_only_seam_matches_pbmpm_solo() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        regions: vec![water_block()],
        ..Scene::default()
    };
    let mats = Materials::default();
    let cfg = Config::default();

    let mut seam = SeamSolver::build(&scene, &mats, &cfg, &gpu);
    let mut solo = PbmpmSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..3 {
        seam.step(DT, &EmissionInput::default());
        solo.step(DT, &EmissionInput::default());
    }
    assert_eq!(seam.solid_count(), 0, "water-only scene seeds no bed");
    assert_eq!(
        seam.water_solver().read_positions(),
        solo.read_positions(),
        "seam water inner must be byte-identical to pbmpm solo"
    );

    // Determinism of the composition itself (house pattern: the stepped twin).
    let mut twin = SeamSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..3 {
        twin.step(DT, &EmissionInput::default());
    }
    assert_eq!(
        twin.read_render_positions_for_test(),
        seam.read_render_positions_for_test(),
        "seam stepping must be deterministic"
    );
}

/// Byte-identical off-switch (R4, bed side): a grain-only scene through the seam steps the
/// twofield inner exactly as a solo twofield with the same forced bed dynamics.
#[test]
fn bed_only_seam_matches_twofield_dry_solo() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        regions: vec![grain_block()],
        ..Scene::default()
    };
    let mats = Materials::default();
    let cfg = Config::default();
    let mut solo_cfg = cfg.clone();
    solo_cfg.solid_dynamics = true; // what the seam forces on its bed inner

    let mut seam = SeamSolver::build(&scene, &mats, &cfg, &gpu);
    let mut solo = TwofieldSolver::build(&scene, &mats, &solo_cfg, &gpu);
    for _ in 0..3 {
        seam.step(DT, &EmissionInput::default());
        solo.step(DT, &EmissionInput::default());
    }
    assert!(seam.solid_count() > 0, "the grain block seeds a bed");
    assert_eq!(seam.water_count(), 0, "grain-only scene seeds no water");
    assert_eq!(
        seam.bed_solver().read_positions(),
        solo.read_positions(),
        "seam bed inner must be byte-identical to twofield dry solo"
    );
}

/// The count-keyed cost-cliff invariant: pouring through the seam grows ONLY the pbmpm
/// side; the twofield inner never holds live water. Also pins the U1 dispatch formula:
/// seam dispatches = pbmpm (5 + 5·iteration_count) + twofield dry (4 · substeps).
#[test]
fn pour_keeps_bed_water_at_zero_and_dispatches_split() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_scaffold: no GPU adapter; skipping.");
        return;
    };
    let mut scene = Scene::debug_high_velocity_jet_impact();
    scene.regions.push(grain_block());
    let mats = Materials::default();
    let cfg = Config::default();

    let mut seam = SeamSolver::build(&scene, &mats, &cfg, &gpu);
    let seed_water = seam.water_count();
    let pour = EmissionInput {
        kettle_pos: [0.0, scene.box_max[1] - 1.0, 0.0],
        flow_rate: 60.0,
        ..EmissionInput::default()
    };
    for _ in 0..60 {
        seam.step(DT, &pour);
        assert_eq!(
            seam.bed_water_count(),
            0,
            "the twofield inner must never hold live water (cost cliff)"
        );
    }
    assert!(
        seam.water_count() > seed_water,
        "the pour goes through the pbmpm inner"
    );

    let total = seam.profile().dispatches_per_frame;
    let water = seam.water_solver().profile().dispatches_per_frame;
    let bed = seam.bed_solver().profile().dispatches_per_frame;
    assert_eq!(
        total,
        water + bed + 2,
        "seam dispatches = inners + the 2 bed-field passes (clear + scatter)"
    );
    assert_eq!(
        water,
        5 + 5 * cfg.pbmpm_iteration_count,
        "pbmpm inner runs its pinned dispatch formula"
    );
    assert_eq!(
        bed,
        5 * seam.bed_solver().substeps_for_dt(DT),
        "twofield inner runs the 5-pass armed dry-dynamic path (4 + seam_inject)"
    );
}

/// U3 hook-placement gate (review r1.3): a known impulse written into the reaction ledger
/// moves the bed — proving `seam_inject` dispatches INSIDE the dry-dynamic substep loop
/// after that branch's grid_clear (anywhere else it would be erased and the bed would stay
/// static). The zero-ledger arm doubles as the hook's byte-inertness check.
#[test]
fn seam_inject_placement_moves_the_bed() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::debug_seam_static_column();
    let mats = Materials::default();
    let mut seam = SeamSolver::build(&scene, &mats, &Config::default(), &gpu);
    seam.prewet_bed(1.0);

    // Settle briefly, then measure the bed's response to a large upward ledger impulse.
    for _ in 0..5 {
        seam.step(DT, &EmissionInput::default());
    }
    let mean_vy = |seam: &SeamSolver| {
        let vel = seam.bed_solver().read_velocities();
        let phases = seam.bed_solver().read_phases();
        let (mut sum, mut n) = (0.0f64, 0u32);
        for (v, &ph) in vel.iter().zip(&phases) {
            if ph == 1 {
                sum += v[1] as f64;
                n += 1;
            }
        }
        (sum / n.max(1) as f64) as f32
    };
    let before = mean_vy(&seam);

    // +y impulse on every node (the mass guard drops massless ones): per-node value 200 at
    // SEAM_IMPULSE_SCALE. The seam zeroes the ledger after the frame, so this acts ONCE.
    let (_occ, reaction) = seam.water_solver().seam_buffers();
    let nodes = (reaction.size() / 16) as usize;
    let mut data = vec![0i32; nodes * 4];
    for n in 0..nodes {
        data[n * 4 + 1] = (200.0 * coffee_sim::solvers::pbmpm::SEAM_IMPULSE_SCALE) as i32;
    }
    gpu.queue.write_buffer(&reaction, 0, bytemuck::cast_slice(&data));
    seam.step(DT, &EmissionInput::default());
    let kicked = mean_vy(&seam);
    assert!(
        kicked > before + 0.05,
        "an upward ledger impulse must move the bed (before {before}, after {kicked})"
    );

    // Ledger was zeroed by the seam after consumption: the next frame injects nothing new.
    seam.step(DT, &EmissionInput::default());
    let after = mean_vy(&seam);
    assert!(
        after < kicked,
        "the zeroed ledger must not keep accelerating the bed (kicked {kicked}, next {after})"
    );
}

/// Review r3.1 gate: prewet writes the cached seed too, so `reset()` replays the WET bed
/// (a reset would otherwise silently dry the M0 scene — the web UI calls reset directly).
#[test]
fn prewet_survives_reset() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::debug_seam_static_column();
    let mats = Materials::default();
    let mut seam = SeamSolver::build(&scene, &mats, &Config::default(), &gpu);
    let v_cap = 1.5 * 1.3 * std::f32::consts::FRAC_PI_6 * mats.grain_diameter.powi(3);

    seam.prewet_bed(1.0);
    for _ in 0..3 {
        seam.step(DT, &EmissionInput::default());
    }
    seam.reset(&scene);

    // One snapshot, phase paired with moisture (the multiset discipline).
    let pos = seam.bed_solver().read_positions();
    let phases = seam.bed_solver().read_phases();
    let mut grains = 0;
    for (p, &ph) in pos.iter().zip(&phases) {
        if ph == 1 {
            grains += 1;
            assert!(
                (p[3] - v_cap).abs() < 1e-6,
                "grain V_abs after reset = {} (expected V_cap = {v_cap})",
                p[3]
            );
        }
    }
    assert_eq!(grains, seam.solid_count() as usize, "every grain checked");
}

/// KTD6: the merged canonical buffers expose water live prefix + solid range only —
/// counts match, phases compose (0 = water, 1 = grain), dormant slots never leak.
#[test]
fn merged_render_buffers_compose_both_sets() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_scaffold: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        regions: vec![water_block(), grain_block()],
        ..Scene::default()
    };
    let mut seam = SeamSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    for _ in 0..2 {
        seam.step(DT, &EmissionInput::default());
    }
    let particles = seam.particles();
    assert_eq!(
        particles.particle_count,
        seam.water_count() + seam.solid_count(),
        "exposed count = live water + solids"
    );
    let phases = seam.read_render_phases_for_test();
    assert_eq!(phases.len(), particles.particle_count as usize);
    let water_tags = phases.iter().filter(|&&p| p == 0).count() as u32;
    let grain_tags = phases.iter().filter(|&&p| p == 1).count() as u32;
    assert_eq!(water_tags, seam.water_count(), "water phase tags round-trip");
    assert_eq!(grain_tags, seam.solid_count(), "grain phase tags round-trip");
    let pos = seam.read_render_positions_for_test();
    assert!(
        pos.iter().all(|p| p.iter().all(|x| x.is_finite())),
        "merged positions stay finite"
    );
}

//! Pour-emission invariants (GPU-gated; skips without an adapter).
//!
//! U1 scope (this file's current tests): the capacity-pool + active-count refactor. A scene with no
//! pour declares `capacity == active_count == seed`, so behavior is unchanged; a scene that declares
//! a pour gets extra dormant pool capacity that must be provably inert (no emission is wired until
//! U2, so `active_count` stays at the seed). Pass selection must treat a declared pour as "water
//! present" even when only grains are seeded.

use coffee_sim::emission::PourEvent;
use coffee_sim::engine::scene::{SeedRegion, Species};
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::xpbd::XpbdSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;
use coffee_sim::EmissionInput;

const DT: f32 = 1.0 / 60.0;

/// A straight-down pour from `kettle` at `flow` (sim-volume/s).
fn pour(kettle: [f32; 3], flow: f32) -> EmissionInput {
    EmissionInput {
        kettle_pos: kettle,
        flow_rate: flow,
        pour_angle: 0.0,
        event: PourEvent::None,
    }
}

/// An empty box that opts into a pour (no seeds) — for counting emission in isolation.
fn empty_pour_scene(pour_ml: f32) -> Scene {
    Scene {
        pour_water_ml: pour_ml,
        gravity: [0.0, 0.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [12.0, 12.0, 12.0],
        regions: vec![],
        ..Scene::default()
    }
}

/// A mixed water+grain scene (water present, so `has_water` is true regardless of pour).
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

/// Non-pour scene: capacity is exactly the seed count, active_count equals it, and no dormant slots.
#[test]
fn non_pour_scene_capacity_equals_seed() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let solver = XpbdSolver::build(
        &mixed_scene(),
        &Materials::default(),
        &Config::default(),
        &gpu,
    );
    let seed = solver.read_phases().len() as u32;
    assert_eq!(solver.active_count(), seed);
    assert_eq!(
        solver.pool_capacity(),
        seed,
        "non-pour scene allocated headroom"
    );
}

/// Declaring a pour grows the pool capacity but NOT the live set (emission lands in U2); the dormant
/// slots must be provably inert — the live solve must match the no-pour build step-for-step.
#[test]
fn dormant_capacity_is_inert() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let cfg = Config::default();

    let mut base = XpbdSolver::build(&mixed_scene(), &mats, &cfg, &gpu);
    let pour = Scene {
        pour_water_ml: 500.0, // big headroom
        ..mixed_scene()
    };
    let mut withcap = XpbdSolver::build(&pour, &mats, &cfg, &gpu);

    let seed = base.read_phases().len() as u32;
    assert_eq!(base.pool_capacity(), seed);
    assert!(
        withcap.pool_capacity() > seed,
        "pour scene did not allocate headroom: {} vs seed {seed}",
        withcap.pool_capacity()
    );
    // Both start with the same live set (no emission yet).
    assert_eq!(base.active_count(), seed);
    assert_eq!(withcap.active_count(), seed);

    for _ in 0..40 {
        base.step(1.0 / 60.0, &EmissionInput::default());
        withcap.step(1.0 / 60.0, &EmissionInput::default());
    }
    // active_count unchanged (emission not wired), and the live solve is identical — the dormant
    // capacity does not perturb the simulated particles.
    assert_eq!(
        withcap.active_count(),
        seed,
        "active_count drifted without emission"
    );
    let pa = base.read_positions();
    let pb = withcap.read_positions();
    assert_eq!(pa.len(), pb.len());
    // Dormant capacity must not perturb the active sim. The cell-order reorder assigns slots
    // non-deterministically (atomic scatter), so "inert" is a SET property, not a per-slot one:
    // every active (position, moisture) tuple in `base` must have a near-coincident match in
    // `withcap`. The tolerance sits above the float-summation-order non-determinism floor (~1e-5,
    // present run-to-run with or without dormant capacity) and far below any real perturbation a
    // dormant-slot leak would cause (O(particle_spacing) ≈ 1).
    let d2 = |a: &[f32; 4], b: &[f32; 4]| (0..4).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>();
    let max_nn: f32 = pa
        .iter()
        .map(|a| {
            pb.iter()
                .map(|b| d2(a, b))
                .fold(f32::INFINITY, f32::min)
                .sqrt()
        })
        .fold(0.0, f32::max);
    assert!(
        max_nn < 1.0e-3,
        "dormant capacity perturbed the active sim: max nearest-neighbor {max_nn}"
    );
}

/// Pass selection: a grain-only scene that declares a pour must still enable the water passes (else
/// poured water would never be solved). Regression for `has_water |= declares_pour`.
#[test]
fn grain_only_pour_scene_enables_water_passes() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let grain_only = Scene {
        pour_water_ml: 250.0,
        box_min: [0.0, 0.0, 0.0],
        box_max: [8.0, 8.0, 8.0],
        regions: vec![SeedRegion {
            min: [2.0, 2.0, 2.0],
            max: [5.0, 3.0, 5.0],
            species: Species::Grain,
        }],
        ..Scene::default()
    };
    let solver = XpbdSolver::build(&grain_only, &Materials::default(), &Config::default(), &gpu);
    assert!(
        solver.has_water_passes(),
        "grain-only pour scene must enable water passes (has_water |= declares_pour)"
    );
    // And the same scene WITHOUT a pour keeps water passes off.
    let no_pour = Scene {
        pour_water_ml: 0.0,
        ..grain_only
    };
    let solver2 = XpbdSolver::build(&no_pour, &Materials::default(), &Config::default(), &gpu);
    assert!(
        !solver2.has_water_passes(),
        "grain-only scene without a pour should not enable water passes"
    );
}

/// reset() returns the live set to the seed (drops any growth) — checked here against the inert pool
/// (active_count stays at seed across reset).
#[test]
fn reset_restores_seed_active_count() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene {
        pour_water_ml: 500.0,
        ..mixed_scene()
    };
    let mut solver = XpbdSolver::build(&scene, &Materials::default(), &Config::default(), &gpu);
    let seed = solver.read_phases().len() as u32;
    for _ in 0..10 {
        solver.step(1.0 / 60.0, &EmissionInput::default());
    }
    solver.reset(&scene);
    assert_eq!(solver.active_count(), seed);
}

// --- U2: emission core ---------------------------------------------------------------------------

/// Rate fidelity + no starvation: over T seconds at constant flow, the emitted particle count tracks
/// flow/V_w·T (the arclength-credit emitter must NOT starve at realistic — sub-spacing-per-frame —
/// pour speeds, which the old per-frame floor did).
#[test]
fn emission_rate_matches_flow_no_starvation() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    // Solve off so active_count == emitted count (emission is independent of the solve).
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mut solver =
        XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
    let v_w = solver.water_particle_volume();
    let flow = 5.0;
    let steps = 120u32; // 2 s
    for _ in 0..steps {
        solver.step(DT, &pour([6.0, 11.0, 6.0], flow));
    }
    let emitted = solver.active_count();
    let expected = flow / v_w * (steps as f32 * DT);
    assert!(
        emitted > 0,
        "emitter starved (0 particles) — arclength credit broken"
    );
    assert!(
        (emitted as f32 - expected).abs() <= 0.10 * expected + 8.0,
        "emitted {emitted} far from expected {expected:.1}"
    );
    // Emitted-mass counter tracks the count.
    let m = solver.total_emitted_water_mass();
    assert!(
        (m - emitted as f32 * 1.0).abs() <= 1.0e-3,
        "emitted-mass {m} != count {emitted} × particle_mass"
    );
}

/// A gentle pour (sub-spacing travel per frame) still emits — direct guard against the starvation
/// failure mode the per-frame floor caused.
#[test]
fn gentle_pour_does_not_starve() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mut solver =
        XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
    // Very gentle flow → exit_speed·dt ≪ particle_spacing; the old floor(travel/spacing)=0 would emit 0.
    for _ in 0..240 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 0.4));
    }
    assert!(
        solver.active_count() > 0,
        "gentle pour starved to 0 particles"
    );
}

/// Emitted particles are seeded as full, cold-of-solute water at the pour temperature.
#[test]
fn emitted_particles_are_seeded_water() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mats = Materials::default();
    let mut solver = XpbdSolver::build(&empty_pour_scene(9999.0), &mats, &cfg, &gpu);
    for _ in 0..30 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 5.0));
    }
    let phase = solver.read_phases();
    let chem = solver.read_chem();
    let moisture = solver.read_moisture();
    assert!(!phase.is_empty(), "no emission");
    assert_eq!(phase.len(), solver.active_count() as usize);
    for (i, &ph) in phase.iter().enumerate() {
        assert_eq!(ph, 0, "emitted particle {i} not water");
        assert!((moisture[i] - 1.0).abs() <= 1.0e-6, "emitted f_w != 1");
        assert!(chem[i][0].abs() <= 1.0e-6, "emitted c != 0");
        assert!(
            (chem[i][1] - mats.pour_t).abs() <= 1.0e-6,
            "emitted T != pour_t"
        );
    }
}

/// Determinism: identical pour inputs produce identical emission (count + positions).
#[test]
fn emission_is_deterministic() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let run = || {
        let mut solver =
            XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
        for _ in 0..60 {
            solver.step(DT, &pour([6.0, 11.0, 6.0], 5.0));
        }
        (solver.active_count(), solver.read_positions())
    };
    let (n1, p1) = run();
    let (n2, p2) = run();
    assert_eq!(n1, n2, "emission count not deterministic");
    for (a, b) in p1.iter().zip(&p2) {
        assert_eq!(a, b, "emission positions not deterministic");
    }
}

/// Capacity clamp: emission never exceeds the allocated pool (active_count saturates at capacity).
#[test]
fn emission_clamps_at_capacity() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    // Tiny dose → small capacity; pour far more than it can hold.
    let mut solver = XpbdSolver::build(&empty_pour_scene(50.0), &Materials::default(), &cfg, &gpu);
    let cap = solver.pool_capacity();
    assert!(cap > 0);
    for _ in 0..600 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 20.0));
        assert!(
            solver.active_count() <= cap,
            "active_count {} exceeded capacity {cap}",
            solver.active_count()
        );
    }
    assert_eq!(
        solver.active_count(),
        cap,
        "pour did not fill the pool to capacity"
    );
}

/// PBF-safe inlet: continuous emission into a closed box with the FULL incompressibility solve on
/// stays finite and bounded — no density-spike eruption (the failure the arclength/rest-density inlet
/// is designed to prevent).
#[test]
fn emission_into_full_solve_does_not_erupt() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let mats = Materials::default();
    let cfg = Config::default(); // full water solve on
    let scene = Scene {
        pour_water_ml: 4000.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [12.0, 16.0, 12.0],
        regions: vec![],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..300 {
        solver.step(DT, &pour([6.0, 14.0, 6.0], 20.0));
    }
    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    assert!(
        solver.active_count() > 50,
        "too little emitted ({}) to be a meaningful stability test",
        solver.active_count()
    );
    assert!(
        pos.iter()
            .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite()),
        "non-finite position (inlet eruption)"
    );
    let vmax = vel
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        vmax <= cfg.max_speed + 1.0,
        "velocity blew past the cap (eruption): vmax {vmax}"
    );
}

/// A `PourEvent::Reset` clears the emitter's backlog/credit so a stale accumulator can't drain after
/// a reset (the emitter contract; full sim restart is `reset()`).
#[test]
fn reset_event_clears_emitter_backlog() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let cfg = Config {
        max_iters: 0,
        ..Config::default()
    };
    let mut solver =
        XpbdSolver::build(&empty_pour_scene(9999.0), &Materials::default(), &cfg, &gpu);
    // Pour briefly to build some emitter state.
    for _ in 0..20 {
        solver.step(DT, &pour([6.0, 11.0, 6.0], 5.0));
    }
    let after_pour = solver.active_count();
    // A Reset event with zero flow clears the backlog and emits nothing further.
    let reset_input = EmissionInput {
        kettle_pos: [6.0, 11.0, 6.0],
        flow_rate: 0.0,
        pour_angle: 0.0,
        event: PourEvent::Reset,
    };
    solver.step(DT, &reset_input);
    assert_eq!(
        solver.active_count(),
        after_pour,
        "Reset event should not emit; backlog must be cleared, not drained"
    );
}

// --- U4: V60 pour scene + integration gates ------------------------------------------------------

/// Build the V60 pour brew (calibrated permeable bed + wetting + extraction on).
fn v60_pour_solver(gpu: &GpuContext) -> XpbdSolver {
    let mats = Materials {
        particle_spacing: 0.5,
        support_radius: 1.0,
        grain_diameter: 1.0,
        min_pore_fraction: 0.35,
        grain_mass: 10.0,
        ..Materials::default()
    };
    let cfg = Config {
        absorb_rate: 0.5,
        extract_rate: 1.0,
        ..Config::default()
    };
    XpbdSolver::build(&Scene::v60_pour(), &mats, &cfg, gpu)
}

/// Integration gate: poured water threads the bed and reaches the cup, yield rises over the brew,
/// the pool capacity is honored, the run stays finite, and water VOLUME is conserved under the source
/// (emitted = in-domain water + absorbed-into-grains).
#[test]
fn v60_pour_through_rises_and_conserves() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let mut solver = v60_pour_solver(&gpu);
    let cap = solver.pool_capacity();
    let v_w = solver.water_particle_volume();
    let rho_w = 1.0 / v_w; // particle_mass = 1
    assert!(cap > 0);
    let kettle = [0.0, 2.5, 0.0];
    let flow = 8.0; // sim-volume/s (raw EmissionInput.flow_rate; the example's mL/s is /5.20 of this)

    let mut yields = Vec::new();
    for step in 1..=900 {
        solver.step(DT, &pour(kettle, flow));
        assert!(
            solver.active_count() <= cap,
            "active_count {} exceeded pool capacity {cap}",
            solver.active_count()
        );
        if step % 300 == 0 {
            solver.sample_diagnostics();
            yields.push(solver.metrics().extraction_yield);
        }
    }
    solver.sample_diagnostics();
    let m = solver.metrics();

    // Pour-through: water reached the cup (TDS readout is cup-region only ⇒ nonzero means cup water).
    assert!(m.tds > 0.0, "no water reached the cup (TDS still 0)");
    // Yield rose over the brew and is finite/positive (absolute band is deferred calibration).
    let last_yield = *yields.last().unwrap();
    assert!(
        last_yield > yields[0] && last_yield > 0.0,
        "yield did not rise over the pour: {yields:?}"
    );
    // Stability.
    let pos = solver.read_positions();
    assert!(
        pos.iter()
            .all(|p| p[0].is_finite() && p[1].is_finite() && p[2].is_finite()),
        "non-finite position during pour brew"
    );
    // Water-volume conservation under the source: emitted = in-domain water + absorbed-into-grains.
    let phase = solver.read_phases();
    let moisture = solver.read_moisture();
    let emitted_vol = solver.total_emitted_water_mass() / rho_w;
    let mut in_domain = 0.0f32;
    for (&ph, &mw) in phase.iter().zip(&moisture) {
        in_domain += if ph == 0 { mw * v_w } else { mw }; // water f_w·V_w; grain V_abs
    }
    assert!(emitted_vol > 0.0, "nothing emitted");
    assert!(
        (in_domain - emitted_vol).abs() <= 0.02 * emitted_vol,
        "water not conserved under pour: emitted {emitted_vol} vs in-domain {in_domain}"
    );
    // Grains stayed trapped in the bed (mean grain y near its seeded band, not drained to the cup).
    let mean_grain_y: f32 = phase
        .iter()
        .zip(&pos)
        .filter(|(&p, _)| p == 1)
        .map(|(_, p)| p[1])
        .sum::<f32>()
        / phase.iter().filter(|&&p| p == 1).count().max(1) as f32;
    assert!(
        mean_grain_y > -3.0,
        "grains washed out of the bed (mean y {mean_grain_y})"
    );
}

/// Determinism of the pour MECHANISM: emission is CPU-side (deterministic cursor + accumulator,
/// independent of the GPU solve), so the emitted particle count is bit-exact across identical runs.
/// The downstream yield is NOT asserted bit/band-equal: it comes from the mixed GPU solve, which is
/// chaotic and not bit-reproducible (grid-scatter atomic order varies — documented project-wide; the
/// bed suite gates only loose aggregate bands for the same reason). Here we just sanity-check yield is
/// finite + positive in both runs.
#[test]
fn v60_pour_emission_is_deterministic() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    // `flow` here is sim-volume/s (the raw EmissionInput.flow_rate), NOT mL/s — the example's `FLOW`
    // is mL/s and divides by 5.20 first.
    let run = || {
        let mut solver = v60_pour_solver(&gpu);
        for _ in 0..300 {
            solver.step(DT, &pour([0.0, 2.5, 0.0], 8.0));
        }
        solver.sample_diagnostics();
        (solver.active_count(), solver.metrics().extraction_yield)
    };
    let (n1, y1) = run();
    let (n2, y2) = run();
    assert_eq!(
        n1, n2,
        "emitted count not deterministic (CPU-side emission must be bit-exact)"
    );
    assert!(
        y1.is_finite() && y1 > 0.0 && y2.is_finite() && y2 > 0.0,
        "yield not sane across runs: {y1}, {y2}"
    );
}

/// Regression for the XSPH resolution-normalization bug: emitting water at FINE spacing into the full
/// solve must stay velocity-bounded (≤ max_speed). Before the fix, the unnormalized XSPH sum (∝1/h³)
/// overshot at fine spacing and amplified velocity post-clamp — vmax exploded to 10^5+ and the inlet
/// sprayed everywhere. The default-spacing `emission_into_full_solve_does_not_erupt` missed it because
/// at spacing 1.0 the kernel sum is tame.
#[test]
fn fine_resolution_emission_stays_bounded() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    // Fine water (spacing 0.15) — the regime that erupted before the XSPH normalization fix.
    let s = 0.15;
    let mats = Materials {
        particle_spacing: s,
        support_radius: 2.0 * s,
        grain_diameter: 2.0 * s,
        ..Materials::default()
    };
    let cfg = Config::default(); // full water solve (incl. XSPH) on
    let scene = Scene {
        pour_water_ml: 3000.0,
        gravity: [0.0, -20.0, 0.0],
        box_min: [0.0, 0.0, 0.0],
        box_max: [6.0, 12.0, 6.0],
        regions: vec![],
        ..Scene::default()
    };
    let mut solver = XpbdSolver::build(&scene, &mats, &cfg, &gpu);
    for _ in 0..240 {
        solver.step(DT, &pour([3.0, 10.0, 3.0], 8.0));
    }
    assert!(
        solver.active_count() > 500,
        "too little emitted ({}) to stress the inlet",
        solver.active_count()
    );
    let pos = solver.read_positions();
    let vel = solver.read_velocities();
    assert!(
        pos.iter().all(|p| p.iter().all(|c| c.is_finite())),
        "non-finite position at fine resolution (eruption)"
    );
    let vmax = vel
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        vmax <= cfg.max_speed + 1.0,
        "fine-resolution emission velocity blew past max_speed (XSPH normalization regression): vmax {vmax}"
    );
}

/// U6: the re-grounded coupling makes the bed RESPOND to the pour — grains in the impingement zone
/// move (they were identically frozen before U2/U3/U4) — while the dense packed bed stays STABLE: no
/// grain reaches a fluidized speed. A gentle V60 pour compacts and surface-disturbs rather than
/// churns (see the U6 agitation analysis), so the response is gated as a BAND: nonzero but
/// sub-fluidization. A future drag re-calibration (e.g. for more visible agitation) must keep both
/// bounds — this is the stability ceiling and the "it actually moves" floor.
#[test]
fn pour_response_is_present_but_sub_fluidization() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let mut solver = v60_pour_solver(&gpu);
    let kettle = [0.0, 2.5, 0.0];
    let mut max_grain = 0.0f32;
    let mut zone_peak = 0.0f32; // peak speed of an impingement-zone grain over the steady window
    for step in 1..=480 {
        solver.step(DT, &pour(kettle, 8.0));
        if step >= 300 && step % 20 == 0 {
            let pos = solver.read_positions();
            let vel = solver.read_velocities();
            let phase = solver.read_phases();
            for ((p, v), &ph) in pos.iter().zip(&vel).zip(&phase) {
                if ph != 1 {
                    continue; // grains only
                }
                let sp = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                max_grain = max_grain.max(sp);
                let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
                if r < 1.2 {
                    zone_peak = zone_peak.max(sp);
                }
            }
        }
    }
    // Floor — the pour mobilizes the impingement zone (was identically 0 before the re-grounding).
    assert!(
        zone_peak > 0.015,
        "pour did not mobilize the impingement zone at all: zone peak {zone_peak:.4}"
    );
    // Ceiling — the dense bed stays a bed; no grain reaches a fluidized speed under the gentle pour.
    assert!(
        max_grain < 0.5,
        "bed fluidized under the pour: max grain speed {max_grain:.4}"
    );
}

/// U6 (plan 2026-06-07-001): the structural wall pressure force keeps water from climbing/sticking
/// to the cup wall — the gate that REPLACES the deleted `v.y=0` heuristic. Uses a splash-robust
/// quantile of wall-region water height (not raw max), gated by occupancy. Also asserts floor-wall
/// seam stability (finite, bounded speed, no leak through the floor or wall at the concave corner).
#[test]
fn wall_pressure_force_prevents_climb() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let r = 0.3_f32;
    let mats = Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    };
    let cfg = Config {
        nozzle_radius: 0.25,
        max_speed: 25.0,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&Scene::v60_pour_water_only(), &mats, &cfg, &gpu);
    let kettle = [0.0, 2.5, 0.0];
    // cup: cylinder radius 3.0, floor y=-8.0, rim y=-3.5.
    let mut wall_ys: Vec<f32> = Vec::new(); // wall-region (r∈[2.6,3.2]) water heights over the window
    let mut bulk_top = f32::MIN; // mid-pool surface (r∈[0.5,2.0]), the reference "pool height"
    let mut seam_vmax = 0.0f32; // peak speed of seam water (near floor∧wall)
    let mut all_finite = true;
    let mut max_r = 0.0f32; // furthest water radius (leak through the cup wall?)
    let mut min_y = f32::MAX; // lowest water (leak through the floor?)
    for step in 1..=480 {
        solver.step(DT, &pour(kettle, 8.0));
        if step >= 300 && step % 20 == 0 {
            let pos = solver.read_positions();
            let vel = solver.read_velocities();
            let phase = solver.read_phases();
            for ((p, v), &ph) in pos.iter().zip(&vel).zip(&phase) {
                if ph != 0 {
                    continue;
                }
                if !p.iter().all(|c| c.is_finite()) {
                    all_finite = false;
                    continue;
                }
                let rr = (p[0] * p[0] + p[2] * p[2]).sqrt();
                // only count water actually inside the cup (below the rim) for wall/leak metrics
                if p[1] < -3.5 {
                    max_r = max_r.max(rr);
                    min_y = min_y.min(p[1]);
                }
                if rr > 2.6 && rr < 3.2 {
                    wall_ys.push(p[1]);
                }
                if rr > 0.5 && rr < 2.0 {
                    bulk_top = bulk_top.max(p[1]);
                }
                if rr > 2.5 && p[1] < -7.0 {
                    seam_vmax = seam_vmax.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                }
            }
        }
    }
    assert!(all_finite, "non-finite water position (eruption)");
    assert!(
        wall_ys.len() > 200,
        "not enough wall-region water sampled ({}) — scene/probe mismatch",
        wall_ys.len()
    );
    wall_ys.sort_by(f32::total_cmp);
    let p95 = wall_ys[((0.95 * (wall_ys.len() - 1) as f32) as usize).min(wall_ys.len() - 1)];
    eprintln!(
        "no-climb: wall p95={p95:.3} (rim -3.5, floor -8.0), bulk_top={bulk_top:.3}, seam_vmax={seam_vmax:.3}, max_r={max_r:.3}, min_y={min_y:.3}"
    );
    // No climb: the wall-region 95th-percentile height stays at/below the rim AND does not rise above
    // the bulk pool surface (no detached sheet). Baseline climbed to ~-3.16 (above rim); heuristic
    // -3.63; the structural force keeps it at the pool surface (measured ~-6.6).
    assert!(
        p95 <= -3.5,
        "water climbed above the cup rim: wall p95 {p95:.3} > rim -3.5"
    );
    assert!(
        p95 <= bulk_top + 0.6,
        "wall water forms a sheet above the pool surface: wall p95 {p95:.3} vs bulk top {bulk_top:.3}"
    );
    // Floor-wall seam stability: bounded speed, no leak through the floor (y≥-8) or wall (r≤3+margin).
    assert!(
        seam_vmax < 25.0,
        "floor-wall seam erupted: seam vmax {seam_vmax:.3}"
    );
    assert!(
        min_y >= -8.0 - 0.5 && max_r <= 3.0 + 0.5,
        "water leaked through the seam: min_y {min_y:.3} (floor -8.0), max_r {max_r:.3} (wall 3.0)"
    );
}

/// U6 sign diagnostic (Codex r1/r2): the wall pressure GRADIENT `g_b` points TOWARD the wall, and the
/// applied force `λ_i·g_b` (over-density λ_i<0) points INTO the fluid. Deterministic — a sign flip
/// here would silently add to the climb, so integration gates alone are insufficient.
#[test]
fn wall_boundary_force_sign_is_correct() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("xpbd_emission: no GPU adapter; skipping.");
        return;
    };
    let r = 0.3_f32;
    let mats = Materials {
        particle_spacing: r,
        support_radius: 2.0 * r,
        ..Materials::default()
    };
    let cfg = Config {
        nozzle_radius: 0.25,
        max_speed: 25.0,
        ..Config::default()
    };
    let mut solver = XpbdSolver::build(&Scene::v60_pour_water_only(), &mats, &cfg, &gpu);
    let kettle = [0.0, 2.5, 0.0];
    for _ in 0..360 {
        solver.step(DT, &pour(kettle, 8.0));
    }
    let pos = solver.read_positions();
    let phase = solver.read_phases();
    let gb = solver.read_boundary_grad();
    let lam = solver.read_lambda();
    // For cup-wall water (r≈3, above the floor corner) the inward SDF normal is radial-inward
    // n̂ = (-x,0,-z)/r. Check g_b points toward the wall (dot<0) and λ·g_b into the fluid (dot>0).
    let mut checked = 0u32;
    let (mut grad_ok, mut force_ok) = (0u32, 0u32);
    for i in 0..pos.len() {
        if phase[i] != 0 {
            continue;
        }
        let (x, y, z) = (pos[i][0], pos[i][1], pos[i][2]);
        let rr = (x * x + z * z).sqrt();
        if !(rr > 2.7 && rr < 3.05 && y > -7.0 && y < -4.0) {
            continue; // cup wall, clear of the floor corner and the rim
        }
        let g = [gb[i][0], gb[i][1], gb[i][2]];
        let gmag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        if gmag < 1.0e-6 {
            continue; // not near enough to register a wall gradient
        }
        let n = [-x / rr, 0.0, -z / rr]; // inward (toward axis = into the fluid)
        let dot_gn = g[0] * n[0] + g[1] * n[1] + g[2] * n[2];
        checked += 1;
        if dot_gn < 0.0 {
            grad_ok += 1; // g_b points toward the wall (away from the inward normal)
        }
        if lam[i] < 0.0 && lam[i] * dot_gn > 0.0 {
            force_ok += 1; // over-dense ⇒ λ·g_b points into the fluid (repulsion)
        }
    }
    eprintln!("sign: checked={checked} grad_toward_wall={grad_ok} force_into_fluid={force_ok}");
    assert!(checked > 20, "too few wall-gradient samples ({checked})");
    // The gradient direction is deterministic geometry — it must hold for ~all samples.
    assert!(
        grad_ok as f32 > 0.95 * checked as f32,
        "g_b sign wrong: only {grad_ok}/{checked} point toward the wall"
    );
    // The repulsion holds wherever the particle is over-dense (λ<0); some samples may be λ≈0.
    assert!(
        force_ok > 0,
        "no over-dense wall sample showed λ·g_b pointing into the fluid"
    );
}

//! Seam-blend U4 gates (docs/plans/2026-07-09-002 R3): the whole-particle infiltration
//! handoff conserves the combined water books exactly, in the saturated (zero-transfer),
//! and combined (support + reaction + handoff simultaneously) regimes. All totals are
//! single-snapshot at frame boundaries; comparisons are totals, never per-index (the
//! seam's own removal permutes pbmpm's live range).
//!
//! GPU-gated; the quantified verdict runs re-execute these under the U5 pre-registration.

use coffee_sim::emission::EmissionInput;
use coffee_sim::engine::Scene;
use coffee_sim::models::Materials;
use coffee_sim::solvers::base::Solver;
use coffee_sim::solvers::seam::SeamSolver;
use coffee_sim::utils::config::Config;
use coffee_sim::utils::gpu::GpuContext;

const DT: f32 = 1.0 / 60.0;

fn m0_mats() -> Materials {
    let s = 0.32;
    Materials {
        particle_spacing: s,
        support_radius: 2.0 * s,
        grain_diameter: 2.0 * s,
        grain_mass: 10.0,
        ..Materials::default()
    }
}

fn absorb_cfg() -> Config {
    Config {
        tf_absorb_rate: 0.5,
        ..Config::default()
    }
}

/// R3(a): a fully saturated bed generates ZERO absorption demand — no particle is ever
/// handed across, and the combined books hold over a long tail (the wet_sat_cutoff
/// contract, ported to the seam).
#[test]
fn saturated_bed_absorbs_nothing_over_long_tail() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_conservation: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::debug_seam_static_column();
    let mut seam = SeamSolver::build(&scene, &m0_mats(), &absorb_cfg(), &gpu);
    seam.prewet_bed(1.0);
    let seed_water = seam.water_count();
    let (_, t0) = seam.water_books();

    for frame in 0..1500 {
        seam.step(DT, &EmissionInput::default());
        if frame % 300 == 0 {
            assert_eq!(
                seam.water_count(),
                seed_water,
                "a saturated bed must never consume a particle (frame {frame})"
            );
        }
    }
    assert_eq!(seam.water_count(), seed_water);
    assert_eq!(seam.absorbed_total(), 0.0, "zero transfer, exactly");
    let (_, t_end) = seam.water_books();
    assert!(
        (t_end - t0).abs() <= 1e-3 * t0,
        "combined books drifted over the saturated tail: t0 {t0}, end {t_end}"
    );
    assert_eq!(seam.bed_water_count(), 0);
}

/// R3(b)+(c), the combined arm: a pond over a HALF-saturated bed — support (the column
/// doesn't fall through, measured against the LIVE grain surface), reaction, and the
/// whole-particle handoff all operate simultaneously; transfer is non-vacuous and the
/// combined books hold exactly through active transfer.
#[test]
fn pond_over_half_saturated_bed_transfers_exactly() {
    let Some(gpu) = GpuContext::new_headless() else {
        eprintln!("seam_conservation: no GPU adapter; skipping.");
        return;
    };
    let scene = Scene::debug_seam_static_column();
    let mats = m0_mats();
    let mut seam = SeamSolver::build(&scene, &mats, &absorb_cfg(), &gpu);
    seam.prewet_bed(0.5);
    let seed_water = seam.water_count();
    let (_, t0) = seam.water_books();

    for frame in 0..900 {
        seam.step(DT, &EmissionInput::default());
        assert_eq!(seam.bed_water_count(), 0, "cost-cliff invariant (frame {frame})");
        if frame % 150 == 149 {
            // Books hold at every sampled frame boundary, INCLUDING mid-transfer.
            let (_, t) = seam.water_books();
            let removed = seed_water - seam.water_count();
            eprintln!(
                "frame {frame}: books {t:.4} (t0 {t0:.4}) | removed {removed} (vol {:.4}) | absorbed_total {:.4}",
                removed as f32 * mats.particle_spacing.powi(3),
                seam.absorbed_total()
            );
            assert!(
                (t - t0).abs() <= 1e-3 * t0,
                "combined books drifted mid-transfer (frame {frame}): t0 {t0}, now {t}"
            );
            // Support against the LIVE grain surface (the bed legitimately swells/expands;
            // the seed surface is not the reference — the crater-gate lesson).
            let bpos = seam.bed_solver().read_positions();
            let bph = seam.bed_solver().read_phases();
            let mut tops: Vec<f32> = bpos
                .iter()
                .zip(&bph)
                .filter(|(_, &ph)| ph == 1)
                .map(|(p, _)| p[1])
                .collect();
            tops.sort_by(f32::total_cmp);
            let grain_top = tops[((tops.len() as f32 * 0.95) as usize).min(tops.len() - 1)];
            let wpos = seam.water_solver().read_positions();
            let live = seam.water_count() as usize;
            let min_y = wpos[..live]
                .iter()
                .map(|p| p[1])
                .fold(f32::INFINITY, f32::min);
            let h = 2.0 * mats.particle_spacing;
            assert!(
                min_y >= grain_top - 14.0 * h,
                "water fell through the live bed (frame {frame}): min_y {min_y}, grain_top {grain_top}"
            );
        }
    }
    // Non-vacuity: the half-saturated bed really consumed particles.
    assert!(
        seam.water_count() < seed_water,
        "no transfer happened (water count held at {seed_water})"
    );
    assert!(seam.absorbed_total() > 0.0);
    // The handoff ledger agrees with the particle count it consumed.
    let consumed = (seed_water - seam.water_count()) as f32
        * mats.particle_spacing.powi(3);
    assert!(
        (seam.absorbed_total() - consumed).abs() <= 1e-3 * consumed.max(1e-6),
        "absorbed_total {} != consumed particle volume {consumed}",
        seam.absorbed_total()
    );
    println!(
        "combined arm: {} of {seed_water} particles absorbed ({:.3} sim-units³), books held",
        seed_water - seam.water_count(),
        seam.absorbed_total()
    );
}

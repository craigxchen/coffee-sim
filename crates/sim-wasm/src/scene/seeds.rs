use coffee_sim_core::Vec3;

use super::{BedSpec, DebugScene, SimSettings};
use crate::materials::{water, DEFAULT_BREW};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParticleKind {
    Water,
    Coffee,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SeedParticle {
    pub kind: ParticleKind,
    pub pos: Vec3,
    pub vel: Vec3,
    pub radius: f32,
    pub mass: f32,
    pub temperature_c: f32,
    pub wetness: f32,
    pub solute: f32,
    pub fast_solute: f32,
    pub slow_solute: f32,
}

pub(crate) fn seed_bed(settings: &SimSettings) -> Vec<SeedParticle> {
    let Some(bed) = settings.bed.as_ref().filter(|bed| bed.enabled) else {
        return Vec::new();
    };
    seed_bed_spec(bed)
}

fn seed_bed_spec(bed: &BedSpec) -> Vec<SeedParticle> {
    let height = (bed.top_y - bed.bot_y).max(1e-6);
    let volume = std::f32::consts::PI
        * height
        * (bed.top_radius * bed.top_radius
            + bed.top_radius * bed.bot_radius
            + bed.bot_radius * bed.bot_radius)
        / 3.0;
    let spacing = (volume / bed.num_particles.max(1) as f32).cbrt();
    let nx = ((bed.top_radius * 2.0) / spacing).ceil() as i32;
    let ny = (height / spacing).ceil() as i32;
    let mut particles = Vec::with_capacity(bed.num_particles as usize);
    for iy in 0..ny {
        let y = bed.center.y + bed.bot_y + (iy as f32 + 0.5) * spacing;
        let t = ((y - (bed.center.y + bed.bot_y)) / height).clamp(0.0, 1.0);
        let max_r = bed.bot_radius + (bed.top_radius - bed.bot_radius) * t;
        for ix in 0..nx {
            for iz in 0..nx {
                if particles.len() >= bed.num_particles as usize {
                    return particles;
                }
                let x = bed.center.x - max_r + (ix as f32 + 0.5) * spacing;
                let z = bed.center.z - max_r + (iz as f32 + 0.5) * spacing;
                let r = ((x - bed.center.x).powi(2) + (z - bed.center.z).powi(2)).sqrt();
                if r > max_r {
                    continue;
                }
                let fast = DEFAULT_BREW.bed_sample_extractable_mass_units()
                    * DEFAULT_BREW.fast_extractable_fraction;
                let slow = DEFAULT_BREW.bed_sample_extractable_mass_units() - fast;
                particles.push(SeedParticle {
                    kind: ParticleKind::Coffee,
                    pos: Vec3::new(x, y, z),
                    vel: Vec3::ZERO,
                    radius: spacing * 0.45,
                    mass: 1.0,
                    temperature_c: 22.0,
                    wetness: if bed.initially_saturated { 1.0 } else { 0.0 },
                    solute: 0.0,
                    fast_solute: fast,
                    slow_solute: slow,
                });
            }
        }
    }
    particles
}

pub(crate) fn seed_debug_water(scene: DebugScene, settings: &SimSettings) -> Vec<SeedParticle> {
    let mut out = Vec::new();
    let radius = settings.render_radius * 0.7;
    let spacing = radius * 2.2;
    let mut add_block = |center: Vec3, ext: Vec3| {
        let nx = (ext.x / spacing).ceil() as i32;
        let ny = (ext.y / spacing).ceil() as i32;
        let nz = (ext.z / spacing).ceil() as i32;
        for ix in -nx..=nx {
            for iy in -ny..=ny {
                for iz in -nz..=nz {
                    out.push(SeedParticle {
                        kind: ParticleKind::Water,
                        pos: center
                            + Vec3::new(
                                ix as f32 * spacing,
                                iy as f32 * spacing,
                                iz as f32 * spacing,
                            ),
                        vel: Vec3::ZERO,
                        radius,
                        mass: water::particle_mass_units(),
                        temperature_c: 92.0,
                        wetness: 0.0,
                        solute: 0.0,
                        fast_solute: 0.0,
                        slow_solute: 0.0,
                    });
                    if out.len() as u32 >= settings.max_particles / 3 {
                        return;
                    }
                }
            }
        }
    };
    match scene {
        DebugScene::FilterWaterBlock | DebugScene::FilterApexDrain => {
            add_block(Vec3::new(0.0, -1.4, 0.0), Vec3::new(1.2, 1.0, 1.2));
        }
        DebugScene::CupWallFloorCornerContact => {
            add_block(Vec3::new(1.5, -6.7, 1.5), Vec3::new(0.9, 0.5, 0.9));
        }
        DebugScene::AsymmetricCupMoundSettle | DebugScene::DamBreakSlosh => {
            add_block(Vec3::new(-1.0, -6.4, 0.0), Vec3::new(1.1, 0.8, 1.1));
        }
        DebugScene::HydrostaticColumn => {
            add_block(Vec3::new(0.0, -5.8, 0.0), Vec3::new(0.7, 1.8, 0.7));
        }
        DebugScene::HighVelocityJetImpact => {
            add_block(Vec3::new(0.0, -6.7, 0.0), Vec3::new(1.2, 0.4, 1.2));
        }
        DebugScene::SeededPaperWallSheet => {
            add_block(Vec3::new(2.0, -0.4, 0.0), Vec3::new(0.25, 1.4, 1.0));
        }
        _ => {}
    }
    out
}

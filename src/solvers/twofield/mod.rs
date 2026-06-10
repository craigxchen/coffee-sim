//! Two-field grid solver scaffold under `SolverId::Twofield` — rung L0, unit U1 of
//! `docs/plans/2026-06-09-001-feat-unified-twofield-solver-plan.md`.
//!
//! Inert skeleton: NO physics yet. What U1 pins down so every later rung inherits it:
//! - the seam contract (six `Solver` methods; `particles()`/`metrics()`/`profile()` never
//!   stall the GPU — they return CPU-cached results from the prior frame),
//! - the two-field particle-range layout (all water particles first, then all solids —
//!   phase-specific dispatches index ranges, never a branchy shared buffer; KTD-1, the v1
//!   separate-data-paths lesson),
//! - the canonical `ParticleBuffers` exposure (`ui::Renderer` works unchanged),
//! - the GPU budgets (dispatches/frame + max storage buffers per entry point; R8) recorded
//!   from the first dispatch onward via the xpbd `dispatch_pass`/timestamp pattern.
//!
//! The real per-substep pipeline (APIC transfers, coarse-seed pressure, Drucker-Prager
//! plasticity, Darcy coupling) lands in U2+.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::emission::EmissionInput;
use crate::engine::scene::Species;
use crate::engine::{Metrics, Scene};
use crate::models::Materials;
use crate::profiling::Profile;
use crate::solvers::base::Solver;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::config::Config;
use crate::utils::gpu::GpuContext;

const WG: u32 = 256;

fn groups(n: u32) -> u32 {
    n.div_ceil(WG)
}

/// U1 GPU-budget bookkeeping (R8): the widest entry point's storage-buffer count, derived by
/// inspection of the bind groups in `build` (the scaffold's only pass, `integrate`, binds
/// `pos` + `vel`; the params uniform doesn't count against the storage limit). Re-derive when
/// passes are added. The device requests 9 storage buffers per stage
/// (`src/utils/gpu.rs::NEEDED_STORAGE_BUFFERS`) — deliberately NOT raised in U1 (KTD-7): grid
/// fields get packed into vec4 lanes first, and AGENTS.md sanctions raising the request to 16
/// only if packing fails for a real pass later.
pub const MAX_STORAGE_BUFFERS_PER_ENTRY_POINT: u32 = 2;

/// Uniform parameters, byte-mirrored by the WGSL `Params` in `common.wgsl`. Deliberately
/// minimal for the scaffold (dt + the phase-range counts); the vec4-aligned tail grows with
/// the physics units.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dt: f32,
    water_count: u32,    // particles [0, water_count) are water (KTD-1 range layout)
    solid_count: u32,    // particles [water_count, water_count + solid_count) are solid grains
    particle_count: u32, // = water_count + solid_count (kernel live-set guard)
}

// Params is uploaded as a uniform and must stay byte-identical to the WGSL `Params`.
const _: () = assert!(std::mem::size_of::<Params>() == 16);

/// Per-pass `timestamp-query` capture (native only; the web context never enables the
/// feature). Same resolve-into-readback-of-the-previous-frame shape as xpbd, so `profile()`
/// never stalls — `sample_diagnostics` is the explicit blocking cache point.
struct Timestamps {
    qset: wgpu::QuerySet,
    capacity: u32,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    labels: Vec<String>,
    count: u32,
}

pub struct TwofieldSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    params: Params,
    water_count: u32,
    solid_count: u32,

    params_buf: wgpu::Buffer,
    // Canonical particle state, exposed through `ParticleBuffers`: pos.w carries the moisture
    // lane (water = remaining fraction, grain = absorbed volume), chem = (c, T, _, _).
    pos: Arc<wgpu::Buffer>,
    vel: Arc<wgpu::Buffer>,
    phase: Arc<wgpu::Buffer>,
    chem: Arc<wgpu::Buffer>,
    readback: wgpu::Buffer,

    integrate_pipeline: wgpu::ComputePipeline,
    integrate_bind: wgpu::BindGroup,
    ts: Option<Timestamps>,

    // Cached per-frame results returned by the getters (never a GPU sync there).
    dispatches: u32,
    cached_passes: Vec<(String, f32)>,

    // Retained for reset (exact, deterministic re-seed).
    initial_positions: Vec<[f32; 4]>,
    initial_phases: Vec<u32>,
}

/// Seed the scene's regions into the two-field range layout: ALL water particles first
/// (`[0, water_count)`), then all solids. Lattice + seeded jitter per species spacing, with
/// cone-aware SDF rejection (mirrors xpbd's `seed_block`; RNG draws happen per lattice point
/// regardless of rejection, so the layout is deterministic for a given `cfg.seed`). The 4th
/// position lane carries moisture: water = remaining fraction (1 = full), grain = absorbed
/// volume (0 = dry). Returns `(positions, phases, water_count)`.
fn seed_ranges(scene: &Scene, mats: &Materials, cfg: &Config) -> (Vec<[f32; 4]>, Vec<u32>, u32) {
    const SEED_CLEARANCE: f32 = 0.4;
    let mut rng = crate::utils::rng::Rng::new(cfg.seed);
    let mut water: Vec<[f32; 4]> = Vec::new();
    let mut solid: Vec<[f32; 4]> = Vec::new();
    for region in &scene.regions {
        let (lo, hi) = (region.min, region.max);
        let (s, tag, w0) = match region.species {
            Species::Water => (mats.particle_spacing, 0u32, 1.0f32),
            Species::Grain => (mats.grain_diameter, 1u32, 0.0f32),
        };
        let jitter = cfg.seed_jitter * s;
        let nx = (((hi[0] - lo[0]) / s).floor() as i32).max(0);
        let ny = (((hi[1] - lo[1]) / s).floor() as i32).max(0);
        let nz = (((hi[2] - lo[2]) / s).floor() as i32).max(0);
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    let jx = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let jy = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let jz = (rng.next_f32() * 2.0 - 1.0) * jitter;
                    let px = lo[0] + i as f32 * s + jx;
                    let py = lo[1] + j as f32 * s + jy;
                    let pz = lo[2] + k as f32 * s + jz;
                    if !scene.solids.is_empty() {
                        let c = crate::utils::sdf::nearest(
                            &scene.solids,
                            glam::Vec3::new(px, py, pz),
                            tag,
                        );
                        if c.signed < SEED_CLEARANCE {
                            continue;
                        }
                    }
                    let p = [px, py, pz, w0];
                    if tag == 0 {
                        water.push(p);
                    } else {
                        solid.push(p);
                    }
                }
            }
        }
    }
    let water_count = water.len() as u32;
    let mut pos = water;
    pos.extend_from_slice(&solid);
    let mut phase = vec![0u32; water_count as usize];
    phase.extend(std::iter::repeat_n(1u32, solid.len()));
    (pos, phase, water_count)
}

impl TwofieldSolver {
    fn storage(
        device: &wgpu::Device,
        label: &str,
        bytes: u64,
        extra: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        })
    }

    /// Blocking GPU→CPU read-back of the first `bytes` of `src` (dev/test only — stalls).
    fn read_bytes(&self, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
        if bytes == 0 {
            return Vec::new();
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-readback"),
            });
        enc.copy_buffer_to_buffer(src, 0, &self.readback, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = self.readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().to_vec();
        self.readback.unmap();
        data
    }

    /// Read back current particle positions (dev/test only — stalls the GPU).
    pub fn read_positions(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.pos, bytes)).to_vec()
    }

    /// Read back current particle velocities (dev/test only — stalls the GPU).
    pub fn read_velocities(&self) -> Vec<[f32; 4]> {
        let bytes = (self.params.particle_count as u64) * 16;
        bytemuck::cast_slice(&self.read_bytes(&self.vel, bytes)).to_vec()
    }

    /// Read back the per-particle phase tags (dev/test only — stalls the GPU).
    pub fn read_phases(&self) -> Vec<u32> {
        let bytes = (self.params.particle_count as u64) * 4;
        bytemuck::cast_slice(&self.read_bytes(&self.phase, bytes)).to_vec()
    }

    /// Water/solid range sizes (the KTD-1 layout: water `[0, w)`, solids `[w, w + s)`).
    pub fn phase_counts(&self) -> (u32, u32) {
        (self.water_count, self.solid_count)
    }

    /// Sample per-pass GPU timestamps into the cache that `profile()` returns. Blocks
    /// (dev/test/periodic only) — the explicit cache point, so the getters never stall.
    pub fn sample_diagnostics(&mut self) {
        if let Some(ts) = &self.ts {
            if ts.count >= 2 {
                let size = (ts.count as u64) * 8;
                let slice = ts.readback.slice(0..size);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                rx.recv().unwrap().unwrap();
                let times: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
                ts.readback.unmap();
                let mut passes = Vec::new();
                for (idx, label) in ts.labels.iter().enumerate() {
                    let b = times[idx * 2];
                    let e = times[idx * 2 + 1];
                    let us = (e.saturating_sub(b)) as f32 * ts.period_ns / 1000.0;
                    passes.push((label.clone(), us));
                }
                self.cached_passes = passes;
            }
        }
    }
}

impl Solver for TwofieldSolver {
    fn build(scene: &Scene, mats: &Materials, cfg: &Config, gpu: &GpuContext) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let (positions, phases, water_count) = seed_ranges(scene, mats, cfg);
        let particle_count = positions.len() as u32;
        let solid_count = particle_count - water_count;

        let params = Params {
            dt: 1.0 / 60.0,
            water_count,
            solid_count,
            particle_count,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twofield-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let n = particle_count.max(1) as u64;
        let vec4 = n * 16;
        let pos = Arc::new(Self::storage(
            &device,
            "twofield-pos",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let vel = Arc::new(Self::storage(
            &device,
            "twofield-vel",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        let phase = Arc::new(Self::storage(
            &device,
            "twofield-phase",
            n * 4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        // Chem/thermal lanes (c, T, _, _); zero-initialized by wgpu until the extraction units.
        let chem = Arc::new(Self::storage(
            &device,
            "twofield-chem",
            vec4,
            wgpu::BufferUsages::COPY_SRC,
        ));
        if particle_count > 0 {
            queue.write_buffer(&pos, 0, bytemuck::cast_slice(&positions));
            queue.write_buffer(&phase, 0, bytemuck::cast_slice(&phases));
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("twofield-readback"),
            size: vec4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("twofield"),
            source: wgpu::ShaderSource::Wgsl(include_str!("common.wgsl").into()),
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        // Per-pipeline bind groups providing exactly the bindings each entry point uses
        // (auto layout drops declared-but-unused globals; binding indices stay global).
        let bg = |pipe: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]| {
            let layout = pipe.get_bind_group_layout(0);
            let e: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|(b, buf)| wgpu::BindGroupEntry {
                    binding: *b,
                    resource: buf.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &e,
            })
        };
        let integrate_pipeline = make("integrate");
        let integrate_bind = bg(
            &integrate_pipeline,
            &[(0, &params_buf), (1, &pos), (2, &vel)],
        );

        let ts = if gpu.timestamps_supported {
            // One pass per frame today; small headroom for the U2+ pipeline growth.
            let capacity = 16u32;
            let qset = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("twofield-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("twofield-ts-resolve"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("twofield-ts-readback"),
                size: (capacity as u64) * 8,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Some(Timestamps {
                qset,
                capacity,
                resolve,
                readback,
                period_ns: queue.get_timestamp_period(),
                labels: Vec::new(),
                count: 0,
            })
        } else {
            None
        };

        Self {
            device,
            queue,
            params,
            water_count,
            solid_count,
            params_buf,
            pos,
            vel,
            phase,
            chem,
            readback,
            integrate_pipeline,
            integrate_bind,
            ts,
            dispatches: 0,
            cached_passes: Vec::new(),
            initial_positions: positions,
            initial_phases: phases,
        }
    }

    fn reset(&mut self, _scene: &Scene) {
        // Exact, deterministic re-seed: positions + phase tags back to the seed, velocities and
        // chem lanes re-zeroed.
        if !self.initial_positions.is_empty() {
            self.queue
                .write_buffer(&self.pos, 0, bytemuck::cast_slice(&self.initial_positions));
            self.queue
                .write_buffer(&self.phase, 0, bytemuck::cast_slice(&self.initial_phases));
            let zeros = vec![[0.0f32; 4]; self.initial_positions.len()];
            self.queue
                .write_buffer(&self.vel, 0, bytemuck::cast_slice(&zeros));
            self.queue
                .write_buffer(&self.chem, 0, bytemuck::cast_slice(&zeros));
        }
        self.cached_passes.clear();
        self.dispatches = 0;
    }

    fn step(&mut self, dt: f32, _input: &EmissionInput) {
        self.params.dt = dt;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut cursor = 0u32;
        let mut labels: Vec<String> = Vec::new();
        let mut dispatches = 0u32;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("twofield-frame"),
            });

        // The single inert pass; at least one workgroup so the dispatch/profiling path is real
        // even on an empty scene (threads early-out on the particle_count guard).
        dispatch_pass(
            &mut enc,
            &self.integrate_pipeline,
            &self.integrate_bind,
            self.ts.as_ref(),
            &mut cursor,
            &mut labels,
            "integrate",
            groups(self.params.particle_count).max(1),
            &mut dispatches,
        );

        if let Some(ts) = &self.ts {
            if cursor >= 2 {
                enc.resolve_query_set(&ts.qset, 0..cursor, &ts.resolve, 0);
                enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.readback, 0, (cursor as u64) * 8);
            }
        }
        self.queue.submit(Some(enc.finish()));

        self.dispatches = dispatches;
        if let Some(ts) = self.ts.as_mut() {
            ts.labels = labels;
            ts.count = cursor;
        }
    }

    fn particles(&self) -> ParticleBuffers {
        ParticleBuffers {
            particle_count: self.params.particle_count,
            position: Some(Arc::clone(&self.pos)),
            velocity: Some(Arc::clone(&self.vel)),
            phase_tag: Some(Arc::clone(&self.phase)),
            // The chem buffer carries both lanes: concentration (.x) and temperature (.y).
            concentration: Some(Arc::clone(&self.chem)),
            temperature: Some(Arc::clone(&self.chem)),
            ..Default::default()
        }
    }

    fn metrics(&self) -> Metrics {
        // All values CPU-resident — never a GPU sync here (trait contract).
        Metrics {
            particle_count: self.params.particle_count,
            ..Default::default()
        }
    }

    fn profile(&self) -> Profile {
        Profile {
            passes: self.cached_passes.clone(),
            dispatches_per_frame: self.dispatches,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_pass(
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    ts: Option<&Timestamps>,
    cursor: &mut u32,
    labels: &mut Vec<String>,
    label: &str,
    groups: u32,
    dispatches: &mut u32,
) {
    let tw = match ts {
        Some(t) if *cursor + 1 < t.capacity => {
            let b = *cursor;
            let e = *cursor + 1;
            *cursor += 2;
            labels.push(label.to_string());
            Some(wgpu::ComputePassTimestampWrites {
                query_set: &t.qset,
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            })
        }
        _ => None,
    };
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: tw,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, Some(bind_group), &[]);
    pass.dispatch_workgroups(groups, 1, 1);
    *dispatches += 1;
}

#[cfg(test)]
mod seed_tests {
    use super::*;

    /// The range layout is water-first: every water particle precedes every solid.
    #[test]
    fn seeds_water_first_then_solids() {
        let scene = Scene::pour_over();
        let (pos, phase, water_count) =
            seed_ranges(&scene, &Materials::default(), &Config::default());
        assert_eq!(pos.len(), phase.len());
        assert!(water_count > 0, "pour_over seeds water");
        assert!(
            (water_count as usize) < phase.len(),
            "pour_over seeds solids too"
        );
        for (i, &ph) in phase.iter().enumerate() {
            let expect = if (i as u32) < water_count { 0 } else { 1 };
            assert_eq!(ph, expect, "particle {i} out of range order");
        }
        // Moisture lane per species: water full (1), grain dry (0).
        for (p, &ph) in pos.iter().zip(&phase) {
            assert_eq!(p[3], if ph == 0 { 1.0 } else { 0.0 });
        }
    }

    /// Seeding is deterministic for a fixed seed (R6).
    #[test]
    fn seeding_is_deterministic() {
        let scene = Scene::pour_over();
        let a = seed_ranges(&scene, &Materials::default(), &Config::default());
        let b = seed_ranges(&scene, &Materials::default(), &Config::default());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }
}

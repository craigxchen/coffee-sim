use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use coffee_sim_core::sph::Vec3;

use super::{MpmSettings, Obstacle};

// FP_SCALE derivation: 2^20 = 1048576. With particle_mass=1.0, max ~50 particles
// contributing per cell (quadratic B-spline, max weight 0.5625), worst-case mass
// per cell ≈ 28. Worst-case momentum per axis ≈ 28 * 2*v_max ≈ 1680. Both fit
// comfortably in i32 range (2^31 / 2^20 ≈ 2048).
pub(crate) const FP_SCALE: f32 = 1048576.0;
pub(crate) const MAX_VELOCITY: f32 = 30.0;
/// Hard ceiling for values stored through `pressure_store` / `divergence_store`
/// before the FP encoding overflows i32. Derived from `i32::MAX / FP_SCALE`:
/// `2^31 / 2^20 ≈ 2048`. Clamp targets must stay strictly below this.
pub(crate) const FP_VALUE_LIMIT: f32 = 2048.0;
pub(crate) const NUM_THREADS: u32 = 64;
pub(crate) const SDF_RES: u32 = 128;

/// Number of `u32` slots in the metrics buffer. Keep in sync with the indices
/// in `shader.rs` (`METRIC_*_IDX`).
pub(crate) const METRICS_SLOT_COUNT: usize = 8;
/// Fixed-point scale used by the `MAX_ABS_DIV` slot — divergence is already a
/// moderate-magnitude quantity, so a smaller scale keeps the atomic headroom
/// comfortable while still giving useful resolution on the HUD.
pub(crate) const METRICS_DIV_FP_SCALE: f32 = 1024.0;

const SDF_NO_CONSTRAINT: f32 = 999.0;
const WALL_THICKNESS: f32 = 0.4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MpmUniforms {
    pub grid_dims: [u32; 4],
    pub counts: [u32; 4],
    pub sim_params: [f32; 4],
    pub grid_origin: [f32; 4],
    pub bounds_max: [f32; 4],
    pub fluid_params: [f32; 4],
    pub fp_params: [f32; 4],
    pub inflow_origin: [f32; 4],
    pub inflow_dir: [f32; 4],
    pub inflow_params: [f32; 4],
    pub sdf_params: [f32; 4],
    pub bed_params: [f32; 4],
    pub extraction_params: [f32; 4],
    pub time_params: [f32; 4],
    /// `[div_clamp, pressure_clamp, metrics_div_fp_scale, metrics_div_inv_fp_scale]`.
    /// `div_clamp` and `pressure_clamp` feed the fixed-point bound checks in
    /// `divergence_store` / `pressure_store` and are derived per frame from the
    /// grid spacing and the velocity cap, not hardcoded.
    pub clamp_params: [f32; 4],
}

pub(crate) struct MpmBuffers {
    pub particles: wgpu::Buffer,
    pub affine: wgpu::Buffer,
    pub grid: wgpu::Buffer,
    pub grid_vel: wgpu::Buffer,
    pub bed_lookup: wgpu::Buffer,
    pub bed_support_count: wgpu::Buffer,
    pub bed_delta: wgpu::Buffer,
    pub _sdf_texture: wgpu::Texture,
    pub sdf_view: wgpu::TextureView,
    pub _sdf_class_texture: wgpu::Texture,
    pub sdf_class_view: wgpu::TextureView,
    pub render_data: wgpu::Buffer,
    pub bed_extract: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    /// Compute-side scratch buffer for projection + overflow observability.
    /// Layout: `u32[METRICS_SLOT_COUNT]`. Indices match `METRIC_*_IDX` in the
    /// shader. Populated every substep via atomics, cleared via
    /// `metrics_clear`.
    pub metrics: wgpu::Buffer,
    /// MAP_READ staging buffer for async CPU readback of `metrics`. Kept in
    /// place so the readback path can be rewired without reshaping
    /// `MpmBuffers`; currently unread while `refresh_metrics` is stubbed.
    #[allow(dead_code)]
    pub metrics_staging: wgpu::Buffer,
}

impl MpmBuffers {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, settings: &MpmSettings) -> Self {
        let max_p = settings.max_particles as usize;
        let [gx, gy, gz] = settings.grid_dims;
        let total_cells = (gx * gy * gz) as usize;

        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm particles"),
            size: (max_p * 32) as u64, // 2 x vec4
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let affine = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm affine"),
            size: (max_p * 48) as u64, // 3 x vec4
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm grid atomics"),
            size: (4 * total_cells * size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm grid vel"),
            size: (total_cells * 16) as u64, // vec4<f32>
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_lookup = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm bed lookup"),
            size: (total_cells * size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_support_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm bed support count"),
            size: (max_p * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_delta = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm bed delta"),
            size: (max_p * size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let render_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm render data"),
            size: (max_p * 16) as u64, // vec4<f32>
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bed_extract = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm bed_extract"),
            size: (max_p * 32) as u64, // 2 x vec4
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm uniforms"),
            size: size_of::<MpmUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let metrics_size = (METRICS_SLOT_COUNT * size_of::<u32>()) as u64;
        let metrics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm metrics"),
            size: metrics_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let metrics_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mpm metrics staging"),
            size: metrics_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let sdf_data = generate_sdf_data(settings);
        let sdf_class_data = generate_sdf_class_data(settings, &sdf_data);
        let (sdf_texture, sdf_view) = create_sdf_texture(device, queue, &sdf_data);
        let (sdf_class_texture, sdf_class_view) =
            create_sdf_class_texture(device, queue, settings.grid_dims, &sdf_class_data);

        Self {
            particles,
            affine,
            grid,
            grid_vel,
            bed_lookup,
            bed_support_count,
            bed_delta,
            _sdf_texture: sdf_texture,
            sdf_view,
            _sdf_class_texture: sdf_class_texture,
            sdf_class_view,
            render_data,
            bed_extract,
            uniform_buffer,
            metrics,
            metrics_staging,
        }
    }
}

// ── SDF generation (embedded from sdf.rs pattern) ──

fn sdf_interior(obstacle: &Obstacle, p: Vec3) -> f32 {
    match obstacle {
        Obstacle::TruncatedCone {
            center,
            top_radius,
            bot_radius,
            top_y,
            bot_y,
        } => {
            let local_y = p.y - center.y;
            if local_y > *top_y || local_y < *bot_y {
                return SDF_NO_CONSTRAINT;
            }
            let height = top_y - bot_y;
            if height < 1e-10 {
                return SDF_NO_CONSTRAINT;
            }
            let t = (local_y - bot_y) / height;
            let cone_r = bot_radius + (top_radius - bot_radius) * t;
            let dx = p.x - center.x;
            let dz = p.z - center.z;
            let r = (dx * dx + dz * dz).sqrt();
            cone_r - r
        }
        Obstacle::Cylinder {
            center,
            radius,
            top_y,
            bot_y,
        } => {
            let local_y = p.y - center.y;
            if local_y > *top_y {
                return SDF_NO_CONSTRAINT;
            }
            let dx = p.x - center.x;
            let dz = p.z - center.z;
            let r = (dx * dx + dz * dz).sqrt();
            let radial_dist = radius - r;
            let floor_dist = local_y - bot_y;
            radial_dist.min(floor_dist)
        }
    }
}

fn generate_sdf_data(settings: &MpmSettings) -> Vec<f32> {
    let n = SDF_RES as usize;
    let bounds = settings.bounds_size;
    let mut data = vec![SDF_NO_CONSTRAINT; n * n * n];

    for obstacle in &settings.obstacles {
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let p = Vec3::new(
                        -bounds.x * 0.5 + (ix as f32 + 0.5) * bounds.x / n as f32,
                        -bounds.y * 0.5 + (iy as f32 + 0.5) * bounds.y / n as f32,
                        -bounds.z * 0.5 + (iz as f32 + 0.5) * bounds.z / n as f32,
                    );
                    let sd = sdf_interior(obstacle, p) - WALL_THICKNESS * 0.5;
                    let idx = iz * n * n + iy * n + ix;
                    if data[idx] >= SDF_NO_CONSTRAINT - 1.0 {
                        data[idx] = sd;
                    } else {
                        data[idx] = data[idx].max(sd);
                    }
                }
            }
        }
    }
    data
}

fn load_sdf_texel(data: &[f32], c: [i32; 3]) -> f32 {
    let max = SDF_RES as i32 - 1;
    let x = c[0].clamp(0, max) as usize;
    let y = c[1].clamp(0, max) as usize;
    let z = c[2].clamp(0, max) as usize;
    let res = SDF_RES as usize;
    data[z * res * res + y * res + x]
}

fn sample_sdf_from_data(settings: &MpmSettings, data: &[f32], position: Vec3) -> f32 {
    let bounds_size = settings.bounds_size;
    let res = SDF_RES as f32;
    let uv = Vec3::new(
        (position.x + bounds_size.x * 0.5) / bounds_size.x * res - 0.5,
        (position.y + bounds_size.y * 0.5) / bounds_size.y * res - 0.5,
        (position.z + bounds_size.z * 0.5) / bounds_size.z * res - 0.5,
    );
    let base = [
        uv.x.floor() as i32,
        uv.y.floor() as i32,
        uv.z.floor() as i32,
    ];
    let fx = uv.x.fract();
    let fy = uv.y.fract();
    let fz = uv.z.fract();

    let c000 = load_sdf_texel(data, base);
    let c100 = load_sdf_texel(data, [base[0] + 1, base[1], base[2]]);
    let c010 = load_sdf_texel(data, [base[0], base[1] + 1, base[2]]);
    let c110 = load_sdf_texel(data, [base[0] + 1, base[1] + 1, base[2]]);
    let c001 = load_sdf_texel(data, [base[0], base[1], base[2] + 1]);
    let c101 = load_sdf_texel(data, [base[0] + 1, base[1], base[2] + 1]);
    let c011 = load_sdf_texel(data, [base[0], base[1] + 1, base[2] + 1]);
    let c111 = load_sdf_texel(data, [base[0] + 1, base[1] + 1, base[2] + 1]);

    let c00 = c000 + (c100 - c000) * fx;
    let c10 = c010 + (c110 - c010) * fx;
    let c01 = c001 + (c101 - c001) * fx;
    let c11 = c011 + (c111 - c011) * fx;
    let c0 = c00 + (c10 - c00) * fy;
    let c1 = c01 + (c11 - c01) * fy;
    c0 + (c1 - c0) * fz
}

fn generate_sdf_class_data(settings: &MpmSettings, sdf_data: &[f32]) -> Vec<u8> {
    let [gx, gy, gz] = settings.grid_dims;
    let mut out = vec![0u8; (gx * gy * gz) as usize];
    let bounds = settings.bounds_size;
    let dx = bounds.x / gx as f32;
    let origin = Vec3::new(-bounds.x * 0.5, -bounds.y * 0.5, -bounds.z * 0.5);

    for iz in 0..gz {
        for iy in 0..gy {
            for ix in 0..gx {
                let cell_center = Vec3::new(
                    origin.x + (ix as f32 + 0.5) * dx,
                    origin.y + (iy as f32 + 0.5) * dx,
                    origin.z + (iz as f32 + 0.5) * dx,
                );
                let idx = (iz * gx * gy + iy * gx + ix) as usize;
                out[idx] = u8::from(sample_sdf_from_data(settings, sdf_data, cell_center) < 0.0);
            }
        }
    }

    out
}

fn create_sdf_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[f32],
) -> (wgpu::Texture, wgpu::TextureView) {
    let res = SDF_RES;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mpm sdf texture"),
        size: wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: res,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(res * 4),
            rows_per_image: Some(res),
        },
        wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: res,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_sdf_class_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    grid_dims: [u32; 3],
    data: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let [gx, gy, gz] = grid_dims;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mpm sdf class texture"),
        size: wgpu::Extent3d {
            width: gx,
            height: gy,
            depth_or_array_layers: gz,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(gx),
            rows_per_image: Some(gy),
        },
        wgpu::Extent3d {
            width: gx,
            height: gy,
            depth_or_array_layers: gz,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdf_class_data_matches_live_cell_center_sampling() {
        let settings = MpmSettings::default_v60();
        let sdf_data = generate_sdf_data(&settings);
        let sdf_class = generate_sdf_class_data(&settings, &sdf_data);
        let [gx, gy, gz] = settings.grid_dims;
        let dx = settings.bounds_size.x / gx as f32;
        let origin = Vec3::new(
            -settings.bounds_size.x * 0.5,
            -settings.bounds_size.y * 0.5,
            -settings.bounds_size.z * 0.5,
        );

        for iz in 0..gz {
            for iy in 0..gy {
                for ix in 0..gx {
                    let cell_center = Vec3::new(
                        origin.x + (ix as f32 + 0.5) * dx,
                        origin.y + (iy as f32 + 0.5) * dx,
                        origin.z + (iz as f32 + 0.5) * dx,
                    );
                    let live = sample_sdf_from_data(&settings, &sdf_data, cell_center) < 0.0;
                    let idx = (iz * gx * gy + iy * gx + ix) as usize;
                    assert_eq!(sdf_class[idx] != 0, live, "mismatch at ({ix},{iy},{iz})");
                }
            }
        }
    }
}

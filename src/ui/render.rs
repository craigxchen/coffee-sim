//! Particle + orientation-cube renderer. Surface-agnostic: `render()` draws into any target
//! view, so the windowed app passes the swapchain view and a headless test passes an
//! offscreen texture. Consumes only the canonical [`ParticleBuffers`].

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::ui::camera::OrbitCamera;
use crate::utils::buffers::ParticleBuffers;
use crate::utils::gpu::GpuContext;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Speed mapped to the top of the color ramp.
const COLOR_MAX_SPEED: f32 = 25.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    right: [f32; 4],
    up: [f32; 4],
    params: [f32; 4], // x = radius, y = inv color-max-speed
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GizmoUniform {
    mvp: [[f32; 4]; 4],
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: (u32, u32),
    radius: f32,
    /// Render-radius multiplier for grain particles (water = 1×); coarse grains draw bigger.
    grain_radius_scale: f32,
    /// `1/V_cap` for the grain saturation tint (0 = no tint).
    moisture_inv_cap: f32,

    depth: wgpu::TextureView,

    camera_buf: wgpu::Buffer,
    particle_pipeline: wgpu::RenderPipeline,
    particle_bgl: wgpu::BindGroupLayout,

    gizmo_buf: wgpu::Buffer,
    gizmo_bg: wgpu::BindGroup,
    gizmo_pipeline: wgpu::RenderPipeline,
    cube_vbuf: wgpu::Buffer,
    cube_vcount: u32,
    axis_vbuf: wgpu::Buffer,
    axis_vcount: u32,
}

impl Renderer {
    pub fn new(
        gpu: &GpuContext,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        radius: f32,
    ) -> Self {
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let depth = make_depth(&device, size);

        // --- particle pipeline ---
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let particle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle-bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                storage_read_entry(1, wgpu::ShaderStages::VERTEX),
                storage_read_entry(2, wgpu::ShaderStages::VERTEX),
                storage_read_entry(3, wgpu::ShaderStages::VERTEX),
            ],
        });
        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particles.wgsl").into()),
        });
        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particles"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("particle-layout"),
                    bind_group_layouts: &[Some(&particle_bgl)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_shader,
                entry_point: Some("fs"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- gizmo pipeline ---
        let gizmo_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo-uniform"),
            size: std::mem::size_of::<GizmoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gizmo_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gizmo-bgl"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let gizmo_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gizmo-bg"),
            layout: &gizmo_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gizmo_buf.as_entire_binding(),
            }],
        });
        let cube = cube_vertices();
        let cube_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gizmo-cube"),
            contents: bytemuck::cast_slice(&cube),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let axes = axis_rods();
        let axis_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gizmo-axes"),
            contents: bytemuck::cast_slice(&axes),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let gizmo_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gizmo"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gizmo.wgsl").into()),
        });
        let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("gizmo"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gizmo-layout"),
                bind_group_layouts: &[Some(&gizmo_bgl)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &gizmo_shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 9 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gizmo_shader,
                entry_point: Some("fs"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            device,
            queue,
            size,
            radius,
            grain_radius_scale: 1.0,
            moisture_inv_cap: 0.0,
            depth,
            camera_buf,
            particle_pipeline,
            particle_bgl,
            gizmo_buf,
            gizmo_bg,
            gizmo_pipeline,
            cube_vbuf,
            cube_vcount: cube.len() as u32,
            axis_vbuf,
            axis_vcount: axes.len() as u32,
        }
    }

    /// Set the render-radius multiplier for grain particles (1.0 = same as water). Use when grains
    /// are coarser than the water (e.g. a porous sand wall) so they draw at their true size.
    pub fn set_grain_radius_scale(&mut self, scale: f32) {
        self.grain_radius_scale = scale;
    }

    /// Set `1/V_cap` so grains can be tinted by saturation (`pos.w·inv_cap`): dry grounds darken as
    /// they wet. 0 (default) disables the tint. `V_cap = r_max·rho_ratio·grain_volume`.
    pub fn set_moisture_scale(&mut self, inv_cap: f32) {
        self.moisture_inv_cap = inv_cap;
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        self.size = (size.0.max(1), size.1.max(1));
        self.depth = make_depth(&self.device, self.size);
    }

    /// Draw the particles + the orientation cube into `target`.
    pub fn render(
        &self,
        target: &wgpu::TextureView,
        particles: &ParticleBuffers,
        camera: &OrbitCamera,
    ) {
        let aspect = self.size.0 as f32 / self.size.1.max(1) as f32;

        // Camera uniform.
        let eye = camera.eye();
        let fwd = (camera.target - eye).normalize_or_zero();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(fwd).normalize_or_zero();
        let cam_u = CameraUniform {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            params: [
                self.radius,
                1.0 / COLOR_MAX_SPEED,
                self.grain_radius_scale,
                self.moisture_inv_cap,
            ],
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam_u));

        // Gizmo uniform: small perspective × rotation-only view.
        let gizmo_proj = Mat4::perspective_rh(45_f32.to_radians(), 1.0, 0.1, 100.0);
        let gizmo_u = GizmoUniform {
            mvp: (gizmo_proj * camera.rotation_only_view()).to_cols_array_2d(),
        };
        self.queue
            .write_buffer(&self.gizmo_buf, 0, bytemuck::bytes_of(&gizmo_u));

        let particle_bg = particles
            .position
            .as_ref()
            .zip(particles.velocity.as_ref())
            .zip(particles.phase_tag.as_ref())
            .map(|((pos, vel), phase)| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("particle-bg"),
                    layout: &self.particle_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.camera_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: pos.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: vel.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: phase.as_entire_binding(),
                        },
                    ],
                })
            });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // Pass 1: particles (clear background + depth).
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particles"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.05,
                            b: 0.07,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(depth_attachment(&self.depth)),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let (Some(bg), n) = (&particle_bg, particles.particle_count) {
                if n > 0 {
                    pass.set_pipeline(&self.particle_pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.draw(0..6, 0..n);
                }
            }
        }

        // Pass 2: orientation cube in the bottom-right corner (keep color, fresh depth).
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gizmo"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(depth_attachment(&self.depth)),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let pad = 12.0;
            let s = ((self.size.0.min(self.size.1)) as f32 * 0.30)
                .clamp(220.0, 380.0)
                .min(self.size.0 as f32)
                .min(self.size.1 as f32);
            let x = (self.size.0 as f32 - s - pad).max(0.0);
            let y = (self.size.1 as f32 - s - pad).max(0.0);
            pass.set_viewport(x, y, s, s, 0.0, 1.0);
            pass.set_pipeline(&self.gizmo_pipeline);
            pass.set_bind_group(0, &self.gizmo_bg, &[]);
            pass.set_vertex_buffer(0, self.cube_vbuf.slice(..));
            pass.draw(0..self.cube_vcount, 0..1);
            pass.set_vertex_buffer(0, self.axis_vbuf.slice(..));
            pass.draw(0..self.axis_vcount, 0..1);
        }

        self.queue.submit(Some(enc.finish()));
    }
}

fn make_depth(device: &wgpu::Device, size: (u32, u32)) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn depth_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    }
}

fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_read_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// (outward normal, the four CCW corners in ±1) for one cube face.
type FaceGeom = ([f32; 3], [[f32; 3]; 4]);

#[rustfmt::skip]
const FACES: [FaceGeom; 6] = [
    ([ 1.0, 0.0, 0.0], [[ 1.0,-1.0, 1.0], [ 1.0,-1.0,-1.0], [ 1.0, 1.0,-1.0], [ 1.0, 1.0, 1.0]]), // +X
    ([-1.0, 0.0, 0.0], [[-1.0,-1.0,-1.0], [-1.0,-1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0,-1.0]]), // -X
    ([ 0.0, 1.0, 0.0], [[-1.0, 1.0, 1.0], [ 1.0, 1.0, 1.0], [ 1.0, 1.0,-1.0], [-1.0, 1.0,-1.0]]), // +Y
    ([ 0.0,-1.0, 0.0], [[-1.0,-1.0,-1.0], [ 1.0,-1.0,-1.0], [ 1.0,-1.0, 1.0], [-1.0,-1.0, 1.0]]), // -Y
    ([ 0.0, 0.0, 1.0], [[-1.0,-1.0, 1.0], [ 1.0,-1.0, 1.0], [ 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]), // +Z
    ([ 0.0, 0.0,-1.0], [[ 1.0,-1.0,-1.0], [-1.0,-1.0,-1.0], [-1.0, 1.0,-1.0], [ 1.0, 1.0,-1.0]]), // -Z
];

const TRI: [usize; 6] = [0, 1, 2, 0, 2, 3];

fn emit_face(out: &mut Vec<[f32; 9]>, corners: &[[f32; 3]; 4], color: [f32; 3], n: [f32; 3]) {
    for &i in &TRI {
        let p = corners[i];
        out.push([
            p[0], p[1], p[2], color[0], color[1], color[2], n[0], n[1], n[2],
        ]);
    }
}

/// The grey orientation cube (36 verts `[pos, color, normal]`). Faces are neutral grey;
/// lighting differentiates them and the XYZ axis rods convey orientation.
fn cube_vertices() -> Vec<[f32; 9]> {
    let grey = [0.58, 0.60, 0.64];
    let mut v = Vec::with_capacity(36);
    for (n, corners) in FACES {
        emit_face(&mut v, &corners, grey, n);
    }
    v
}

/// An axis-aligned box `[min, max]` as 36 vertices with a single color.
fn box_verts(min: [f32; 3], max: [f32; 3], color: [f32; 3]) -> Vec<[f32; 9]> {
    let remap = |c: [f32; 3]| {
        [
            min[0] + (c[0] + 1.0) * 0.5 * (max[0] - min[0]),
            min[1] + (c[1] + 1.0) * 0.5 * (max[1] - min[1]),
            min[2] + (c[2] + 1.0) * 0.5 * (max[2] - min[2]),
        ]
    };
    let mut v = Vec::with_capacity(36);
    for (n, corners) in FACES {
        let mapped = [
            remap(corners[0]),
            remap(corners[1]),
            remap(corners[2]),
            remap(corners[3]),
        ];
        emit_face(&mut v, &mapped, color, n);
    }
    v
}

/// Three colored axis rods (thin boxes) from the cube's −corner along +X/+Y/+Z (CAD-style).
fn axis_rods() -> Vec<[f32; 9]> {
    let c = -1.0; // corner
    let r = 0.10; // rod half-thickness
    let len = 1.55; // rod tip, past the cube face
    let red = [0.92, 0.26, 0.24];
    let green = [0.36, 0.82, 0.36];
    let blue = [0.30, 0.52, 0.95];
    let mut v = Vec::with_capacity(108);
    v.extend(box_verts([c, c - r, c - r], [len, c + r, c + r], red)); // +X
    v.extend(box_verts([c - r, c, c - r], [c + r, len, c + r], green)); // +Y
    v.extend(box_verts([c - r, c - r, c], [c + r, c + r, len], blue)); // +Z
    v
}

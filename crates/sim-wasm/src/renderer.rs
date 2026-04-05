use std::borrow::Cow;
use std::f32::consts::PI;
use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use coffee_sim_core::sph::Vec3;

use crate::mpm_3d::{MpmSim3D, MpmSettings, Obstacle};

const EPSILON: f32 = 1e-6;

const PARTICLE_3D_SHADER: &str = r#"
struct Particle3DUniforms {
    view_proj: mat4x4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
    light_dir: vec4<f32>,
    params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Particle3DUniforms;

struct ParticleVertexInput {
    @location(0) local: vec2<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) colour_t: f32,
};

struct ParticleVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) colour_t: f32,
};

@vertex
fn vs_main(input: ParticleVertexInput) -> ParticleVertexOutput {
    let water_radius = uniforms.params.x * mix(1.18, 0.96, clamp(input.colour_t * 0.7, 0.0, 1.0));
    let radius = select(water_radius, uniforms.params.x * 0.52, input.colour_t < 0.0);
    let offset = uniforms.camera_right.xyz * input.local.x * radius
        + uniforms.camera_up.xyz * input.local.y * radius;
    let world = input.world_position + offset;

    var output: ParticleVertexOutput;
    output.clip_position = uniforms.view_proj * vec4<f32>(world, 1.0);
    output.local = input.local;
    output.colour_t = input.colour_t;
    return output;
}

fn palette(colour_t: f32) -> vec3<f32> {
    if (colour_t < 0.0) {
        let wetness = clamp(-colour_t - 1.0, 0.0, 1.0);
        let dry = vec3<f32>(0.34, 0.24, 0.12);
        let wet = vec3<f32>(0.17, 0.11, 0.06);
        return mix(dry, wet, wetness);
    }
    let base = vec3<f32>(0.07, 0.20, 0.47);
    let mid = vec3<f32>(0.10, 0.47, 0.74);
    let crest = vec3<f32>(0.87, 0.95, 0.98);
    return mix(mix(base, mid, clamp(colour_t, 0.0, 1.0)), crest, clamp(colour_t * 0.6, 0.0, 1.0));
}

@fragment
fn fs_main(input: ParticleVertexOutput) -> @location(0) vec4<f32> {
    let radial = dot(input.local, input.local);
    if (radial > 1.0) {
        discard;
    }

    let sphere_z = sqrt(max(1.0 - radial, 0.0));
    let normal = normalize(
        uniforms.camera_right.xyz * input.local.x
            + uniforms.camera_up.xyz * input.local.y
            - uniforms.camera_forward.xyz * sphere_z
    );
    let light = normalize(-uniforms.light_dir.xyz);
    let diffuse = max(dot(normal, light), 0.0);
    let rim = pow(1.0 - sphere_z, 2.5);
    let color = palette(input.colour_t) * (0.34 + diffuse * 0.9) + vec3<f32>(rim * 0.12);
    let alpha = smoothstep(1.0, 0.82, radial);
    return vec4<f32>(color, alpha);
}
"#;

const CONE_SHADER: &str = r#"
struct ConeUniforms {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: ConeUniforms;

struct ConeVertexInput {
    @location(0) position: vec3<f32>,
};

struct ConeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(input: ConeVertexInput) -> ConeVertexOutput {
    var output: ConeVertexOutput;
    output.clip_position = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    return output;
}

@fragment
fn fs_main(input: ConeVertexOutput) -> @location(0) vec4<f32> {
    return uniforms.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuadVertex {
    local: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Particle3DUniforms {
    view_proj: [[f32; 4]; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    camera_forward: [f32; 4],
    light_dir: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ConeUniforms {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
}

#[derive(Clone, Copy)]
pub(crate) struct OrbitCamera {
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) radius: f32,
    pub(crate) target: Vec3,
}

impl OrbitCamera {
    pub(crate) fn new(bounds: Vec3) -> Self {
        // Frame the actual pourover apparatus, not the full simulation box.
        let focus_extent = bounds.x.max(bounds.z).max(bounds.y * 0.45);
        Self {
            yaw: -0.66,
            pitch: 0.5,
            radius: focus_extent * 0.62,
            target: Vec3::new(0.0, -1.2, 0.0),
        }
    }

    pub(crate) fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * 0.01;
        self.pitch = (self.pitch - delta_y * 0.01).clamp(-1.2, 1.2);
    }

    pub(crate) fn zoom(&mut self, delta: f32, bounds: Vec3) {
        let focus_extent = bounds.x.max(bounds.z).max(bounds.y * 0.45);
        let min_radius = focus_extent * 0.28;
        let max_radius = focus_extent * 2.4;
        self.radius = (self.radius * (1.0 + delta * 0.0015)).clamp(min_radius, max_radius);
    }

    fn eye(self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        self.target
            + Vec3::new(
                self.yaw.sin() * cos_pitch,
                self.pitch.sin(),
                self.yaw.cos() * cos_pitch,
            ) * self.radius
    }
}

pub(crate) struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    particle_3d_pipeline: wgpu::RenderPipeline,
    cone_pipeline: wgpu::RenderPipeline,
    particle_3d_uniform_buffer: wgpu::Buffer,
    particle_3d_bind_group: wgpu::BindGroup,
    cone_uniform_buffer: wgpu::Buffer,
    cone_bind_group: wgpu::BindGroup,
    quad_vertex_buffer: wgpu::Buffer,
    cone_vertex_buffer: wgpu::Buffer,
    cone_vertex_count: u32,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl Renderer {
    pub(crate) async fn new(
        canvas: HtmlCanvasElement,
        settings: &MpmSettings,
    ) -> Result<Self, JsValue> {
        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(js_error)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .map_err(js_error)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("coffee-sim device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(js_error)?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| JsValue::from_str("Cannot configure surface for this adapter."))?;
        config.alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;
        surface.configure(&device, &config);

        // Particle 3D uniform buffer and bind group
        let particle_3d_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle 3d uniforms"),
            size: size_of::<Particle3DUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle 3d bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let particle_3d_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle 3d bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_3d_uniform_buffer.as_entire_binding(),
            }],
        });

        // Cone uniform buffer and bind group
        let cone_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cone uniforms"),
            size: size_of::<ConeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cone_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cone bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cone_uniform_buffer.as_entire_binding(),
            }],
        });

        // Shaders
        let particle_3d_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle 3d shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PARTICLE_3D_SHADER)),
        });
        let cone_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cone shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CONE_SHADER)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle 3d pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Particle 3D render pipeline
        let particle_3d_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle 3d pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_3d_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: (size_of::<f32>() * 4) as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_3d_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Cone wireframe pipeline
        let cone_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cone pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cone_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (size_of::<f32>() * 3) as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    }],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &cone_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Quad vertex buffer (billboard)
        let quad_vertices = [
            QuadVertex {
                local: [-1.0, -1.0],
            },
            QuadVertex { local: [1.0, -1.0] },
            QuadVertex { local: [1.0, 1.0] },
            QuadVertex {
                local: [-1.0, -1.0],
            },
            QuadVertex { local: [1.0, 1.0] },
            QuadVertex { local: [-1.0, 1.0] },
        ];
        let quad_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quad vertex buffer"),
            size: (quad_vertices.len() * size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&quad_vertex_buffer, 0, bytemuck::cast_slice(&quad_vertices));

        // Cone wireframe vertices
        let cone_verts = build_wireframe(settings);
        let cone_vertex_count = cone_verts.len() as u32;
        let cone_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cone vertex buffer"),
            size: (cone_verts.len() * size_of::<[f32; 3]>()).max(16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&cone_vertex_buffer, 0, bytemuck::cast_slice(&cone_verts));

        let (depth_texture, depth_view) = create_depth_resources(&device, width, height);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            particle_3d_pipeline,
            cone_pipeline,
            particle_3d_uniform_buffer,
            particle_3d_bind_group,
            cone_uniform_buffer,
            cone_bind_group,
            quad_vertex_buffer,
            cone_vertex_buffer,
            cone_vertex_count,
            depth_texture,
            depth_view,
        })
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let (depth_texture, depth_view) = create_depth_resources(&self.device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    pub(crate) fn render_3d(
        &mut self,
        simulation: &MpmSim3D,
        camera: OrbitCamera,
    ) -> Result<(), JsValue> {
        if self.config.width == 0 || self.config.height == 0 {
            return Ok(());
        }

        let aspect = (self.config.width.max(1) as f32) / (self.config.height.max(1) as f32);
        let eye = camera.eye();
        let target = camera.target;
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let forward = (target - eye).normalized();
        let right = forward.cross(world_up).normalized();
        let up = right.cross(forward).normalized();
        let view = look_at(eye, target, up);
        let projection = perspective(48.0_f32.to_radians(), aspect, 0.1, 200.0);
        let view_proj = mat4_mul(projection, view);

        // Update particle uniforms
        let particle_uniforms = Particle3DUniforms {
            view_proj,
            camera_right: [right.x, right.y, right.z, 0.0],
            camera_up: [up.x, up.y, up.z, 0.0],
            camera_forward: [forward.x, forward.y, forward.z, 0.0],
            light_dir: [-0.45, -0.9, -0.25, 0.0],
            params: [simulation.settings().render_radius, 0.0, 0.0, 0.0],
        };
        self.queue.write_buffer(
            &self.particle_3d_uniform_buffer,
            0,
            bytemuck::bytes_of(&particle_uniforms),
        );

        // Update cone uniforms
        let cone_uniforms = ConeUniforms {
            view_proj,
            color: [0.95, 0.90, 0.78, 0.35],
        };
        self.queue.write_buffer(
            &self.cone_uniform_buffer,
            0,
            bytemuck::bytes_of(&cone_uniforms),
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(())
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Validation => {
                return Err(JsValue::from_str("Surface texture lost."));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.08,
                            b: 0.10,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Draw cone wireframe
            if self.cone_vertex_count > 0 {
                pass.set_pipeline(&self.cone_pipeline);
                pass.set_bind_group(0, &self.cone_bind_group, &[]);
                pass.set_vertex_buffer(0, self.cone_vertex_buffer.slice(..));
                pass.draw(0..self.cone_vertex_count, 0..1);
            }

            // Draw particles
            if simulation.particle_count() > 0 {
                pass.set_pipeline(&self.particle_3d_pipeline);
                pass.set_bind_group(0, &self.particle_3d_bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, simulation.render_buffer().slice(..));
                pass.draw(0..6, 0..simulation.particle_count() as u32);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn build_wireframe(settings: &MpmSettings) -> Vec<[f32; 3]> {
    let segments = 32;
    let verticals = 16;
    let mut verts = Vec::new();

    for obs in &settings.obstacles {
        match obs {
            Obstacle::TruncatedCone {
                center,
                top_radius,
                bot_radius,
                top_y,
                bot_y,
            } => {
                let rings = 8;
                for ring in 0..=rings {
                    let t = ring as f32 / rings as f32;
                    let y = center.y + bot_y + (top_y - bot_y) * t;
                    let r = bot_radius + (top_radius - bot_radius) * t;
                    push_ring(&mut verts, center.x, y, center.z, r, segments);
                }
                for i in 0..verticals {
                    let a = 2.0 * PI * i as f32 / verticals as f32;
                    verts.push([
                        center.x + top_radius * a.cos(),
                        center.y + top_y,
                        center.z + top_radius * a.sin(),
                    ]);
                    verts.push([
                        center.x + bot_radius * a.cos(),
                        center.y + bot_y,
                        center.z + bot_radius * a.sin(),
                    ]);
                }
            }
            Obstacle::Cylinder {
                center,
                radius,
                top_y,
                bot_y,
            } => {
                let rings = 4;
                for ring in 0..=rings {
                    let t = ring as f32 / rings as f32;
                    let y = center.y + bot_y + (top_y - bot_y) * t;
                    push_ring(&mut verts, center.x, y, center.z, *radius, segments);
                }
                for i in 0..verticals {
                    let a = 2.0 * PI * i as f32 / verticals as f32;
                    verts.push([
                        center.x + radius * a.cos(),
                        center.y + top_y,
                        center.z + radius * a.sin(),
                    ]);
                    verts.push([
                        center.x + radius * a.cos(),
                        center.y + bot_y,
                        center.z + radius * a.sin(),
                    ]);
                }
            }
        }
    }

    push_spout_wireframe(&mut verts, settings);
    verts
}

fn push_ring(verts: &mut Vec<[f32; 3]>, cx: f32, y: f32, cz: f32, r: f32, segments: usize) {
    for i in 0..segments {
        let a0 = 2.0 * PI * i as f32 / segments as f32;
        let a1 = 2.0 * PI * ((i + 1) % segments) as f32 / segments as f32;
        verts.push([cx + r * a0.cos(), y, cz + r * a0.sin()]);
        verts.push([cx + r * a1.cos(), y, cz + r * a1.sin()]);
    }
}

fn push_spout_wireframe(verts: &mut Vec<[f32; 3]>, settings: &MpmSettings) {
    let spout = settings.spout;
    let direction = spout.direction.normalized();
    let base = spout.origin - direction * spout.stem_length;
    let (basis_a, basis_b) = spout_basis(direction);
    let ring_segments = 12;

    for i in 0..ring_segments {
        let a0 = 2.0 * PI * i as f32 / ring_segments as f32;
        let a1 = 2.0 * PI * ((i + 1) % ring_segments) as f32 / ring_segments as f32;
        let base0 = base
            + basis_a * (spout.stem_radius * a0.cos())
            + basis_b * (spout.stem_radius * a0.sin());
        let base1 = base
            + basis_a * (spout.stem_radius * a1.cos())
            + basis_b * (spout.stem_radius * a1.sin());
        let tip0 = spout.origin
            + basis_a * (spout.nozzle_radius * a0.cos())
            + basis_b * (spout.nozzle_radius * a0.sin());
        let tip1 = spout.origin
            + basis_a * (spout.nozzle_radius * a1.cos())
            + basis_b * (spout.nozzle_radius * a1.sin());

        verts.push([base0.x, base0.y, base0.z]);
        verts.push([base1.x, base1.y, base1.z]);
        verts.push([tip0.x, tip0.y, tip0.z]);
        verts.push([tip1.x, tip1.y, tip1.z]);
        verts.push([base0.x, base0.y, base0.z]);
        verts.push([tip0.x, tip0.y, tip0.z]);
    }

    verts.push([base.x, base.y, base.z]);
    verts.push([spout.origin.x, spout.origin.y, spout.origin.z]);
}

fn spout_basis(direction: Vec3) -> (Vec3, Vec3) {
    let helper = if direction.y.abs() < 0.95 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let a = direction.cross(helper).normalized();
    let b = direction.cross(a).normalized();
    (a, b)
}

fn create_depth_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn mat4_mul(lhs: [[f32; 4]; 4], rhs: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            result[column][row] = lhs[0][row] * rhs[column][0]
                + lhs[1][row] * rhs[column][1]
                + lhs[2][row] * rhs[column][2]
                + lhs[3][row] * rhs[column][3];
        }
    }
    result
}

fn perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y_radians * 0.5).tan();
    [
        [f / aspect.max(EPSILON), 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (near * far) / (near - far), 0.0],
    ]
}

fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> [[f32; 4]; 4] {
    let forward = (target - eye).normalized();
    let right = forward.cross(up).normalized();
    let camera_up = right.cross(forward);

    [
        [right.x, camera_up.x, -forward.x, 0.0],
        [right.y, camera_up.y, -forward.y, 0.0],
        [right.z, camera_up.z, -forward.z, 0.0],
        [-right.dot(eye), -camera_up.dot(eye), forward.dot(eye), 1.0],
    ]
}

fn js_error(error: impl ToString) -> JsValue {
    JsValue::from_str(&error.to_string())
}

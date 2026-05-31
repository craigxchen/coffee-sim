use crate::diagnostics::{MetricsSnapshot, Profiler, METRICS_SLOT_COUNT};
use crate::render_contract::{RenderInstance, RENDER_INSTANCE_SIZE_BYTES};
use crate::scene::SimSettings;

pub(crate) const TYPE_WATER: f32 = 0.0;
pub(crate) const TYPE_COFFEE: f32 = 1.0;
pub(crate) const TYPE_INACTIVE: f32 = 2.0;

pub(crate) struct XpbdBuffers {
    pub render_data: wgpu::Buffer,
    pub metrics: wgpu::Buffer,
}

impl XpbdBuffers {
    pub(crate) fn new(device: &wgpu::Device, settings: &SimSettings) -> Self {
        let render_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xpbd render data"),
            size: settings.max_particles as u64 * RENDER_INSTANCE_SIZE_BYTES,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let metrics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("xpbd metrics"),
            size: (METRICS_SLOT_COUNT * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            render_data,
            metrics,
        }
    }
}

pub(crate) struct XpbdState {
    pub settings: SimSettings,
    pub buffers: XpbdBuffers,
    pub pos_type: Vec<[f32; 4]>,
    pub vel_mass: Vec<[f32; 4]>,
    pub props: Vec<[f32; 4]>,
    pub material: Vec<[f32; 4]>,
    pub render_instances: Vec<RenderInstance>,
    pub num_water: u32,
    pub num_coffee: u32,
    pub total_time: f32,
    pub frame_emitted_mass: f32,
    pub total_emitted_mass: f32,
    pub frame_dropped_particles: u32,
    pub total_dropped_particles: u32,
    pub cup_water_mass: f32,
    pub cup_solute_mass: f32,
    pub latest_metrics: MetricsSnapshot,
    pub last_iterations: u32,
    pub profiler: Profiler,
}

impl XpbdState {
    pub(crate) fn new(device: &wgpu::Device, settings: SimSettings) -> Self {
        let buffers = XpbdBuffers::new(device, &settings);
        let cap = settings.max_particles as usize;
        Self {
            settings,
            buffers,
            pos_type: Vec::with_capacity(cap),
            vel_mass: Vec::with_capacity(cap),
            props: Vec::with_capacity(cap),
            material: Vec::with_capacity(cap),
            render_instances: Vec::with_capacity(cap),
            num_water: 0,
            num_coffee: 0,
            total_time: 0.0,
            frame_emitted_mass: 0.0,
            total_emitted_mass: 0.0,
            frame_dropped_particles: 0,
            total_dropped_particles: 0,
            cup_water_mass: 0.0,
            cup_solute_mass: 0.0,
            latest_metrics: MetricsSnapshot::default(),
            last_iterations: 0,
            profiler: Profiler::default(),
        }
    }
}

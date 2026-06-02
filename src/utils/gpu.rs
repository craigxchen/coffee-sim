//! GPU device/queue context shared by every block.

use std::sync::Arc;

use wgpu::{Adapter, Device, Instance, Queue};

/// Wraps the `wgpu` device + queue and the data needed to create resources.
///
/// Targets the **WebGPU-guaranteed baseline limits** (`wgpu::Limits::default()`) even on
/// native, so the storage-buffer binding ceiling (8 per stage) that broke v1's DFSPH path
/// on the browser bites here first — we never silently exceed what WebGPU guarantees (v1
/// raised `max_storage_buffers_per_shader_stage` to 10). Requests `TIMESTAMP_QUERY` only
/// when the adapter advertises it. Headless for now: no surface (rendering arrives with
/// `ui`). The same code path compiles to WASM + WebGPU later — only the async init below
/// is native-specific (`pollster::block_on`).
pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    /// Whether per-pass `timestamp-query` profiling is available on this device.
    pub timestamps_supported: bool,
}

impl GpuContext {
    /// Build a headless context. Returns `None` when no adapter is available (e.g. CI
    /// without a GPU) so callers can skip GPU work gracefully instead of failing.
    pub fn new_headless() -> Option<Self> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok()?;
        Self::from_adapter(instance, adapter)
    }

    /// Build a context for an on-screen window and return the matching surface. The adapter
    /// is chosen as surface-compatible; the **solver and renderer share this one context** so
    /// the renderer can read the solver's particle buffers. The caller owns the surface (it
    /// configures/presents). Returns `None` if no adapter/device is available.
    pub fn new_windowed(
        window: Arc<winit::window::Window>,
    ) -> Option<(Self, wgpu::Surface<'static>)> {
        let instance = Instance::default();
        let surface = instance.create_surface(window).ok()?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .ok()?;
        let ctx = Self::from_adapter(instance, adapter)?;
        Some((ctx, surface))
    }

    /// Request the device/queue (shared by both constructors). NOTE: `pollster::block_on` is
    /// native-only; under WASM this step awaits the JS event loop instead.
    fn from_adapter(instance: Instance, adapter: Adapter) -> Option<Self> {
        let timestamps_supported = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamps_supported {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("coffee-sim device"),
            required_features,
            // WebGPU baseline. Do NOT raise max_storage_buffers_per_shader_stage above 8
            // (v1's mistake) — it must stay portable to the browser.
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()?;

        Some(Self {
            instance,
            adapter,
            device,
            queue,
            timestamps_supported,
        })
    }
}

//! GPU device/queue context shared by every block.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use wgpu::{Adapter, Device, Instance, Queue};

/// Wraps the `wgpu` device + queue and the data needed to create resources.
///
/// Targets the WebGPU baseline limits except `max_storage_buffers_per_shader_stage`, which is raised
/// to **9** — the count the boundary-force density passes bind (the spec baseline is 8; modern
/// browsers and native both grant ≥9). Requested as a fixed value, not from `adapter.limits()` (which
/// wgpu's web backend over-reports, breaking `request_device`). Requests `TIMESTAMP_QUERY` only when
/// the adapter advertises it. Headless for now: no surface (rendering arrives with `ui`). The same
/// request path compiles to WASM + WebGPU — native only wraps the async init in `pollster::block_on`.
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_headless() -> Option<Self> {
        let instance = Instance::default();
        pollster::block_on(Self::request(
            instance,
            wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
            true,
        ))
    }

    /// Build a context for an on-screen window and return the matching surface. The adapter
    /// is chosen as surface-compatible; the **solver and renderer share this one context** so
    /// the renderer can read the solver's particle buffers. The caller owns the surface (it
    /// configures/presents). Returns `None` if no adapter/device is available.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_windowed(
        window: Arc<winit::window::Window>,
    ) -> Option<(Self, wgpu::Surface<'static>)> {
        let instance = Instance::default();
        let surface = instance.create_surface(window).ok()?;
        let ctx = pollster::block_on(Self::request(
            instance,
            wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            },
            true,
        ))?;
        Some((ctx, surface))
    }

    /// Build a WebGPU context for a browser canvas and return the matching surface. The caller owns
    /// the surface and configures/presents it in the browser frame loop.
    #[cfg(target_arch = "wasm32")]
    pub async fn new_web(
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<(Self, wgpu::Surface<'static>), wasm_bindgen::JsValue> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(js_error)?;
        let ctx = Self::request(
            instance,
            wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            },
            false,
        )
        .await
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("No WebGPU adapter/device available."))?;

        Ok((ctx, surface))
    }

    /// Request the adapter, device, and queue. Native constructors block on this; the web canvas
    /// constructor awaits it on the JS event loop.
    async fn request(
        instance: Instance,
        options: wgpu::RequestAdapterOptions<'_, '_>,
        allow_timestamp_queries: bool,
    ) -> Option<Self> {
        let adapter = instance.request_adapter(&options).await.ok()?;
        let timestamps_supported =
            allow_timestamp_queries && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamps_supported {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        // The boundary-force density passes (compute_lambda / compute_dp) bind 9 storage buffers per
        // stage — one above the WebGPU spec baseline of 8. Request EXACTLY that, as a fixed value:
        // modern browsers grant ≥10 (confirmed via navigator.gpu) and native Metal grants 16+, so 9
        // is always within range. NOTE: do NOT derive this from `adapter.limits()` — wgpu's web
        // backend over-reports that field, so requesting it makes request_device fail in the browser
        // ("Device failed at creation"). All other limits stay at the WebGPU baseline for portability.
        const NEEDED_STORAGE_BUFFERS: u32 = 9;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("coffee-sim device"),
                required_features,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: NEEDED_STORAGE_BUFFERS,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
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

#[cfg(target_arch = "wasm32")]
fn js_error(error: impl ToString) -> wasm_bindgen::JsValue {
    wasm_bindgen::JsValue::from_str(&error.to_string())
}

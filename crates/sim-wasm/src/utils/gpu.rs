//! Minimal GPU context shim for the rewrite XPBD solver inside the wasm crate.

#[derive(Clone)]
pub(crate) struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) timestamps_supported: bool,
}

impl GpuContext {
    pub(crate) fn from_device_queue(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            timestamps_supported: false,
        }
    }
}

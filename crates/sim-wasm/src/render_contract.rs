use bytemuck::{Pod, Zeroable};

pub(crate) const RENDER_INSTANCE_FLOATS: usize = 8;
pub(crate) const RENDER_INSTANCE_SIZE_BYTES: u64 =
    (RENDER_INSTANCE_FLOATS * std::mem::size_of::<f32>()) as u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct RenderInstance {
    pub data: [f32; RENDER_INSTANCE_FLOATS],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_instance_layout_is_eight_floats() {
        assert_eq!(
            std::mem::size_of::<RenderInstance>(),
            8 * std::mem::size_of::<f32>()
        );
    }
}

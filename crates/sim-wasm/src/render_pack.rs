use crate::render_contract::RenderInstance;

pub(crate) fn pack_instances(
    pos_type: &[[f32; 4]],
    props: &[[f32; 4]],
    material: &[[f32; 4]],
    out: &mut Vec<RenderInstance>,
) {
    out.clear();
    out.reserve(pos_type.len());
    for ((pos, prop), mat) in pos_type.iter().zip(props).zip(material) {
        if pos[3] >= 2.0 {
            continue;
        }
        let is_coffee = pos[3] >= 0.5;
        let colour_t = if is_coffee {
            -1.0 - mat[0].clamp(0.0, 1.0)
        } else {
            (0.15 + mat[0] * 12.0).clamp(0.0, 1.0)
        };
        let brew_t = if is_coffee {
            mat[0].clamp(0.0, 1.0)
        } else {
            (mat[0] * 20.0).clamp(0.0, 1.0)
        };
        out.push(RenderInstance {
            data: [pos[0], pos[1], pos[2], colour_t, prop[0], brew_t, 0.0, 0.0],
        });
    }
}

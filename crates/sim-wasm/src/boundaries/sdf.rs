use coffee_sim_core::Vec3;

use super::cup::CupConfig;
use super::filter::FilterConfig;

pub(crate) fn project_bounds(pos: Vec3, radius: f32, bounds_size: Vec3) -> Vec3 {
    let half = bounds_size * 0.5;
    Vec3::new(
        pos.x.clamp(-half.x + radius, half.x - radius),
        pos.y.clamp(-half.y + radius, half.y - radius),
        pos.z.clamp(-half.z + radius, half.z - radius),
    )
}

pub(crate) fn project_filter_inside(pos: Vec3, radius: f32, filter: &FilterConfig) -> Vec3 {
    let local_y = pos.y - filter.center.y;
    if local_y < filter.bot_y || local_y > filter.top_y {
        return pos;
    }
    let inner = (filter.inner_radius_at_y(local_y) - radius).max(0.0);
    let dx = pos.x - filter.center.x;
    let dz = pos.z - filter.center.z;
    let r = (dx * dx + dz * dz).sqrt();
    if r <= inner || r <= 1e-6 {
        return pos;
    }
    let scale = inner / r;
    Vec3::new(
        filter.center.x + dx * scale,
        pos.y,
        filter.center.z + dz * scale,
    )
}

pub(crate) fn project_cup_inside(pos: Vec3, radius: f32, cup: CupConfig) -> Vec3 {
    if pos.y > cup.top_y || pos.y < cup.bot_y {
        return pos;
    }
    let dx = pos.x - cup.center.x;
    let dz = pos.z - cup.center.z;
    let r = (dx * dx + dz * dz).sqrt();
    let inner = (cup.radius - radius).max(0.0);
    let mut out = pos;
    if r > inner && r > 1e-6 {
        let scale = inner / r;
        out.x = cup.center.x + dx * scale;
        out.z = cup.center.z + dz * scale;
    }
    out.y = out.y.max(cup.bot_y + radius);
    out
}

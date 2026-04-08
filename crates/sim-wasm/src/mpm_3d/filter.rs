use coffee_sim_core::sph::Vec3;

#[derive(Clone)]
pub(crate) struct FilterConfig {
    pub center: Vec3,
    pub top_y: f32,
    pub bot_y: f32,
    pub top_radius: f32,
    pub bot_radius: f32,
    pub thickness: f32,
    pub hole_radius: f32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            center: Vec3::new(0.0, -0.35, 0.0),
            top_y: 2.75,
            bot_y: -3.02,
            top_radius: 4.10,
            bot_radius: 0.62,
            thickness: 0.08,
            hole_radius: 0.18,
        }
    }
}

impl FilterConfig {
    pub fn radius_at_y(&self, y: f32) -> f32 {
        let height = (self.top_y - self.bot_y).max(1e-6);
        let t = ((y - self.bot_y) / height).clamp(0.0, 1.0);
        self.bot_radius + (self.top_radius - self.bot_radius) * t
    }

    pub fn inner_radius_at_y(&self, y: f32) -> f32 {
        (self.radius_at_y(y) - self.thickness).max(self.hole_radius)
    }

    #[allow(dead_code)]
    pub fn contains_shell_point(&self, point: Vec3) -> bool {
        if point.y < self.bot_y || point.y > self.top_y {
            return false;
        }

        let dx = point.x - self.center.x;
        let dz = point.z - self.center.z;
        let r = (dx * dx + dz * dz).sqrt();
        let outer = self.radius_at_y(point.y);
        let inner = self.inner_radius_at_y(point.y);
        r <= outer && r >= inner
    }

    pub fn opening_radius(&self) -> f32 {
        self.hole_radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_radius_decreases_toward_the_apex() {
        let filter = FilterConfig::default();
        assert!(filter.radius_at_y(filter.top_y) > filter.radius_at_y(filter.bot_y));
    }

    #[test]
    fn filter_contains_shell_points_on_the_cone_band() {
        let filter = FilterConfig::default();
        let mid_y = (filter.top_y + filter.bot_y) * 0.5;
        let outer = filter.radius_at_y(mid_y);
        let inner = filter.inner_radius_at_y(mid_y);
        assert!(filter.contains_shell_point(Vec3::new(filter.center.x + (outer + inner) * 0.5, mid_y, filter.center.z)));
        assert!(!filter.contains_shell_point(Vec3::new(filter.center.x + outer + 0.5, mid_y, filter.center.z)));
    }
}

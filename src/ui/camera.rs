//! CAD-style orbit camera (pure math — no GPU, unit-tested).

use glam::{Mat4, Vec3};

/// An orbit camera: looks at `target` from `distance` away, at orientation (`yaw`, `pitch`).
#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Azimuth around +Y (radians).
    pub yaw: f32,
    /// Elevation (radians), clamped away from the poles.
    pub pitch: f32,
    pub fov_y: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl OrbitCamera {
    /// Frame a scene whose bounds span `[lo, hi]`: target the center, back off to fit.
    pub fn framing(lo: Vec3, hi: Vec3) -> Self {
        let target = (lo + hi) * 0.5;
        let radius = (hi - lo).length() * 0.5;
        Self {
            target,
            distance: (radius * 2.5).max(1.0),
            yaw: 0.7,
            pitch: 0.5,
            fov_y: 60_f32.to_radians(),
            znear: 0.05,
            zfar: 10_000.0,
        }
    }

    /// Camera eye position in world space.
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let dir = Vec3::new(cp * sy, sp, cp * cy);
        self.target + dir * self.distance
    }

    /// View × projection for the given viewport aspect ratio.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let proj = Mat4::perspective_rh(self.fov_y, aspect.max(1e-3), self.znear, self.zfar);
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        proj * view
    }

    /// Orientation-only view (for the corner gizmo): same rotation, fixed distance, origin target.
    pub fn rotation_only_view(&self) -> Mat4 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let dir = Vec3::new(cp * sy, sp, cp * cy);
        Mat4::look_at_rh(dir * 2.5, Vec3::ZERO, Vec3::Y)
    }

    /// Drag → orbit (radians). Pitch is clamped just shy of the poles.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const LIMIT: f32 = 1.553; // ~89°
        self.yaw -= dx;
        self.pitch = (self.pitch + dy).clamp(-LIMIT, LIMIT);
    }

    /// Scroll/pinch → dolly. `factor > 1` zooms out; `< 1` zooms in. Distance clamped.
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.2, 50_000.0);
    }

    /// Two-finger drag → pan: translate `target` in the camera's right/up plane.
    /// `dx`/`dy` are in screen-fraction units; scaled by distance so it feels uniform.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let scale = self.distance;
        self.target += (-right * dx + up * dy) * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> OrbitCamera {
        OrbitCamera::framing(Vec3::ZERO, Vec3::splat(10.0))
    }

    #[test]
    fn orbit_updates_and_clamps_pitch() {
        let mut c = cam();
        let (y0, p0) = (c.yaw, c.pitch);
        c.orbit(0.1, 0.1);
        assert!((c.yaw - y0).abs() > 0.0 && (c.pitch - p0).abs() > 0.0);
        // Pitch can't pass the pole no matter how far we drag.
        for _ in 0..1000 {
            c.orbit(0.0, 1.0);
        }
        assert!(c.pitch < std::f32::consts::FRAC_PI_2);
        assert!(c.pitch > -std::f32::consts::FRAC_PI_2 - 0.01);
    }

    #[test]
    fn zoom_scales_and_clamps_distance() {
        let mut c = cam();
        let d0 = c.distance;
        c.zoom(2.0);
        assert!(c.distance > d0);
        c.zoom(0.25);
        assert!(c.distance < 2.0 * d0);
        for _ in 0..100 {
            c.zoom(0.1);
        }
        assert!(c.distance >= 0.2);
    }

    #[test]
    fn pan_translates_target() {
        let mut c = cam();
        let t0 = c.target;
        c.pan(0.1, 0.0);
        assert!((c.target - t0).length() > 0.0);
    }

    #[test]
    fn view_proj_is_finite() {
        let c = cam();
        let m = c.view_proj(16.0 / 9.0);
        assert!(m.to_cols_array().iter().all(|v| v.is_finite()));
        assert!(
            m.determinant().abs() > 1e-12,
            "view_proj should be invertible"
        );
    }
}

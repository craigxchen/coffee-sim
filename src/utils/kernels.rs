//! SPH smoothing kernels + gradients (CPU reference + tests). The WGSL solver implements
//! the same functions on the GPU; these mirror them for `ρ₀` computation and unit tests.
//!
//! Poly6 for density, spiky gradient for the pressure/correction forces — the standard PBF
//! pairing (Macklin & Müller 2013). v1's validated weights are recorded in `KEEP.md` §3.

use std::f32::consts::PI;

/// Poly6 smoothing kernel `W(r, h)` for `0 ≤ r ≤ h`, else 0.
/// `W(r,h) = 315/(64π h⁹) · (h² − r²)³`.
pub fn w_poly6(r: f32, h: f32) -> f32 {
    if r >= h || r < 0.0 {
        return 0.0;
    }
    let h2 = h * h;
    let coeff = 315.0 / (64.0 * PI * h.powi(9));
    let t = h2 - r * r;
    coeff * t * t * t
}

/// Magnitude of the spiky kernel gradient for `0 < r ≤ h`, else 0.
/// `|∇W(r,h)| = 45/(π h⁶) · (h − r)²`.
pub fn spiky_grad_mag(r: f32, h: f32) -> f32 {
    if r >= h || r <= 0.0 {
        return 0.0;
    }
    let coeff = 45.0 / (PI * h.powi(6));
    let t = h - r;
    coeff * t * t
}

/// Spiky kernel gradient vector `∇_i W(x_i − x_j, h)` (points from j toward i).
/// `d = x_i − x_j`. Returns zero for `r ≥ h` or `r == 0` (degenerate — caller jitters/clamps).
pub fn spiky_grad(d: [f32; 3], h: f32) -> [f32; 3] {
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if r >= h || r <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let m = spiky_grad_mag(r, h) / r; // includes the 1/r from normalizing d
    [m * d[0], m * d[1], m * d[2]]
}

/// Rest density `ρ₀ = Σ_j m · W_poly6(|x_i − x_j|, h)` over a full cubic lattice neighborhood
/// at `spacing`, evaluated at a lattice site. Deterministic constant used by the density
/// constraint so a perfectly-packed block reads `C = 0`.
pub fn rest_density(spacing: f32, support_radius: f32, mass: f32) -> f32 {
    let h = support_radius;
    let reach = (h / spacing).ceil() as i32;
    let mut rho = 0.0;
    for i in -reach..=reach {
        for j in -reach..=reach {
            for k in -reach..=reach {
                let d = [i as f32 * spacing, j as f32 * spacing, k as f32 * spacing];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                rho += mass * w_poly6(r, h);
            }
        }
    }
    rho
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly6_shape() {
        let h = 2.0;
        assert!(w_poly6(0.0, h) > 0.0);
        assert_eq!(w_poly6(h, h), 0.0);
        assert_eq!(w_poly6(h + 0.1, h), 0.0);
        // Monotonically decreasing on [0, h].
        assert!(w_poly6(0.3, h) > w_poly6(1.0, h));
        assert!(w_poly6(1.0, h) > w_poly6(1.9, h));
    }

    #[test]
    fn spiky_grad_shape() {
        let h = 2.0;
        assert_eq!(spiky_grad_mag(0.0, h), 0.0); // r==0 degenerate → 0 (caller clamps)
        assert!(spiky_grad_mag(0.5, h) > 0.0);
        assert_eq!(spiky_grad_mag(h, h), 0.0);
        assert_eq!(spiky_grad_mag(h + 0.1, h), 0.0);
        // Gradient points outward along d (from j to i).
        let g = spiky_grad([0.5, 0.0, 0.0], h);
        assert!(g[0] > 0.0 && g[1] == 0.0 && g[2] == 0.0);
    }

    #[test]
    fn rest_density_positive_and_increases_with_packing() {
        // Denser packing (smaller spacing) ⇒ higher rest density.
        let dense = rest_density(0.8, 2.0, 1.0);
        let sparse = rest_density(1.2, 2.0, 1.0);
        assert!(dense > sparse);
        assert!(sparse > 0.0);
    }
}

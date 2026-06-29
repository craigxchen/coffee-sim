Review this Rust implementation of Unit U2 of the coffee-sim Phase 1.5 plan: `src/models/thermal.rs`, the CPU thermal model (capacity-weighted pairwise heat exchange + ambient loss), the mirror of the GPU thermal pass.

Plan intent (must hold): heat exchange is CAPACITY-WEIGHTED so it conserves energy for UNEQUAL heat capacities C=mass·specific_heat. A pair exchanges heat q (antisymmetric: q_ij = −q_ji), each side applies ΔT=±q/C. Stability comes from clamping the pair heat q SYMMETRICALLY (not a per-particle ΔT clamp), so antisymmetry holds and it never overshoots the capacity-weighted equilibrium. The GPU per-particle form will be ΔT_i = (Σ q_ij)/C_i + ambient. A plain symmetric ΔT=α(T_j−T_i) would be WRONG (only conserves when C_i=C_j).

Verdict on the FIRST LINE: APPROVE, REVISE, or REJECT. Then explain specifically. Scrutinize: (1) is `pair_heat` truly antisymmetric AFTER the clamp (so q_ij = −q_ji exactly, for the GPU per-particle form to conserve)? (2) does the clamp (`raw.clamp(cap.min(0), cap.max(0))` with cap = reduced-capacity·ΔT) correctly prevent overshoot for any dt without breaking antisymmetry? (3) edge cases: degenerate capacities, dt=0, equal T, large dt; (4) does the function set match what a GPU per-particle pass needs to mirror; (5) any numerical or sign bug. This is pure CPU; the GPU pass is a later unit. Be rigorous but do not demand scope beyond U2.

## src/models/thermal.rs
```rust
//! Thermal exchange shared by solvers (CPU reference) — lumped bed↔water heat transfer + ambient loss.
//!
//! Water carries temperature `T`; the kinetics are `T`-dependent (`extraction::arrhenius`), so the
//! natural pour-temperature drop reduces late-brew extraction. Heat exchange is **capacity-weighted**
//! so it conserves energy for unequal water/grain heat capacities `C = mass · specific_heat`: a pair
//! exchanges a heat amount `q` (antisymmetric — what one gains the other loses), and each side applies
//! `ΔT = ±q/C`. A plain symmetric `ΔT = α·(T_j−T_i)` would only conserve energy when `C_i = C_j`.
//! These pure functions are the CPU mirror of the GPU thermal pass; the GPU per-particle form is
//! `ΔT_i = (Σ_neighbors q_ij)/C_i + ambient`, with `q_ij = −q_ji` so global enthalpy is conserved.

const EPS: f32 = 1.0e-6;

/// Capacity-weighted heat from neighbor `j` into particle `i` over `dt`: `κ·(T_j − T_i)·dt`, **clamped
/// symmetrically** so it never overshoots the pair's capacity-weighted equilibrium. The cap is the
/// reduced heat capacity times the temperature gap, `|q| ≤ (C_i·C_j/(C_i+C_j))·|T_j − T_i|` — exactly
/// the heat that brings both to equilibrium. Apply as `ΔT_i = q/C_i` (and `ΔT_j = −q/C_j` when `j`
/// gathers `i`). The function is **antisymmetric**: `pair_heat(j←i) = −pair_heat(i←j)` (the clamp
/// depends only on the symmetric reduced capacity and `|ΔT|`), so global enthalpy is conserved.
#[inline]
pub fn pair_heat(t_i: f32, c_i: f32, t_j: f32, c_j: f32, kappa: f32, dt: f32) -> f32 {
    let sum = c_i + c_j;
    if sum <= EPS {
        return 0.0;
    }
    let raw = kappa.max(0.0) * (t_j - t_i) * dt.max(0.0);
    let cap = (c_i * c_j / sum) * (t_j - t_i); // reduced-capacity equilibrium heat (same sign as raw)
                                               // raw and cap share the sign of (t_j − t_i); clamp |raw| ≤ |cap| preserving that sign.
    raw.clamp(cap.min(0.0), cap.max(0.0))
}

/// Per-step loss toward ambient: `−clamp(h_amb·dt, 0, 1)·(T − T_amb)`. Always cools (or warms) toward
/// `T_amb`, never overshooting it; zero when already at ambient.
#[inline]
pub fn ambient_delta(t: f32, t_amb: f32, h_amb: f32, dt: f32) -> f32 {
    -(h_amb * dt).clamp(0.0, 1.0) * (t - t_amb)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1.0e-4;

    #[test]
    fn pair_heat_is_antisymmetric() {
        let q_ij = pair_heat(20.0, 1.0, 80.0, 3.0, 0.5, 1.0 / 60.0);
        let q_ji = pair_heat(80.0, 3.0, 20.0, 1.0, 0.5, 1.0 / 60.0);
        assert!((q_ij + q_ji).abs() < TOL, "antisymmetric: {q_ij} vs {q_ji}");
        assert!(q_ij > 0.0, "heat flows from the hotter j into i");
    }

    #[test]
    fn closed_unequal_capacity_pair_conserves_energy_and_relaxes() {
        // Unequal capacities; iterate the symmetric pair update many steps.
        let (mut ti, ci) = (20.0f32, 1.0f32);
        let (mut tj, cj) = (90.0f32, 4.0f32);
        let e0 = ci * ti + cj * tj;
        let t_eq = (ci * ti + cj * tj) / (ci + cj);
        for _ in 0..2000 {
            let q = pair_heat(ti, ci, tj, cj, 0.3, 1.0 / 60.0);
            ti += q / ci;
            tj += -q / cj; // the neighbor applies −q/C_j
                           // never overshoot equilibrium
            assert!(
                ti <= t_eq + TOL && tj >= t_eq - TOL,
                "no overshoot: {ti} {tj} eq {t_eq}"
            );
        }
        // enthalpy conserved
        assert!(
            (ci * ti + cj * tj - e0).abs() < 1.0e-2,
            "energy conserved: {} vs {e0}",
            ci * ti + cj * tj
        );
        // converged to the capacity-weighted equilibrium
        assert!(
            (ti - t_eq).abs() < 0.5 && (tj - t_eq).abs() < 0.5,
            "relaxed to eq {t_eq}: {ti} {tj}"
        );
    }

    #[test]
    fn pair_heat_zero_when_equal_temperature_or_zero_dt() {
        assert_eq!(pair_heat(50.0, 1.0, 50.0, 2.0, 0.5, 1.0 / 60.0), 0.0);
        assert_eq!(pair_heat(20.0, 1.0, 80.0, 2.0, 0.5, 0.0), 0.0);
        assert_eq!(pair_heat(20.0, 0.0, 80.0, 0.0, 0.5, 1.0 / 60.0), 0.0); // degenerate capacity
    }

    #[test]
    fn ambient_cools_toward_ambient_without_overshoot() {
        let (t_amb, h) = (20.0, 2.0);
        // hot → cools toward ambient
        let d = ambient_delta(90.0, t_amb, h, 1.0 / 60.0);
        assert!(d < 0.0 && 90.0 + d >= t_amb, "cools, no overshoot");
        // at ambient → no change
        assert_eq!(ambient_delta(t_amb, t_amb, h, 1.0 / 60.0), 0.0);
        // huge dt → clamped to exactly ambient, not past
        assert!((90.0 + ambient_delta(90.0, t_amb, h, 1.0e6) - t_amb).abs() < TOL);
    }
}
```

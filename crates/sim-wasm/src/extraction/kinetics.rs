use crate::materials::DEFAULT_BREW;

pub(crate) fn temperature_rate_scale(temp_c: f32) -> f32 {
    // Simple calibrated Arrhenius-like monotonic scale around 92 C.
    (1.0 + (temp_c - 92.0) * 0.025).clamp(0.25, 2.5)
}

pub(crate) fn two_pool_transfer(
    fast: f32,
    slow: f32,
    water_concentration: f32,
    temp_c: f32,
    relative_speed: f32,
    dt: f32,
) -> (f32, f32) {
    let saturation_drive =
        (1.0 - water_concentration / DEFAULT_BREW.max_solute_concentration).clamp(0.0, 1.0);
    let renewal = (relative_speed * 0.08).clamp(0.15, 2.0);
    let t = temperature_rate_scale(temp_c);
    let fast_out =
        (DEFAULT_BREW.fast_extraction_rate_s * t * renewal * saturation_drive * dt * fast)
            .clamp(0.0, fast);
    let slow_out =
        (DEFAULT_BREW.slow_extraction_rate_s * t * renewal * saturation_drive * dt * slow)
            .clamp(0.0, slow);
    (fast_out, slow_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotter_water_extracts_faster() {
        let cool = two_pool_transfer(1.0, 1.0, 0.0, 80.0, 1.0, 1.0);
        let hot = two_pool_transfer(1.0, 1.0, 0.0, 96.0, 1.0, 1.0);
        assert!(hot.0 + hot.1 > cool.0 + cool.1);
    }
}

pub(crate) fn exchange_temperatures(water_temp: f32, coffee_temp: f32, dt: f32) -> (f32, f32) {
    let alpha = (0.18 * dt).clamp(0.0, 0.25);
    let delta = (water_temp - coffee_temp) * alpha;
    (water_temp - delta, coffee_temp + delta)
}

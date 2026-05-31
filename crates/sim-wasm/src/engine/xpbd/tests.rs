use super::shader::XPBD_SHADER;

#[test]
fn xpbd_shader_entry_points_are_declared() {
    for entry in [
        "predict",
        "hash_build",
        "water_density",
        "sdf_collision",
        "velocity_update",
        "render_pack",
        "coffee_contact",
        "darcy_drag",
        "extraction_metrics",
    ] {
        assert!(XPBD_SHADER.contains(&format!("fn {entry}(")));
    }
}

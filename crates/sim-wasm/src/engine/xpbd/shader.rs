pub(crate) const XPBD_SHADER: &str = r#"
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn hash_build(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn water_density(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn sdf_collision(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn velocity_update(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn render_pack(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn coffee_contact(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn darcy_drag(@builtin(global_invocation_id) _gid: vec3<u32>) {}

@compute @workgroup_size(64)
fn extraction_metrics(@builtin(global_invocation_id) _gid: vec3<u32>) {}
"#;

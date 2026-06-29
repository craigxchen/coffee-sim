fn renderer_shader(source: &str, name: &str) -> String {
    let marker = format!("const {name}: &str = r#\"");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("missing shader constant {name}"))
        + marker.len();
    let rest = &source[start..];
    let end = rest
        .find("\"#;")
        .unwrap_or_else(|| panic!("unterminated shader constant {name}"));
    rest[..end].to_string()
}

#[test]
fn canonical_renderer_shaders_parse_with_naga() {
    let renderer = include_str!("renderer.rs");
    for name in [
        "CANONICAL_PARTICLE_3D_SHADER",
        "CANONICAL_CROSS_SECTION_SHADER",
    ] {
        let shader = renderer_shader(renderer, name);
        naga::front::wgsl::parse_str(&shader)
            .unwrap_or_else(|error| panic!("{name} should parse: {error}"));
    }
}

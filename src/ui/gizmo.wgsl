// CAD orientation cube: colored XYZ faces, drawn in a corner viewport with the camera's
// rotation only so it shows which way the view is oriented.

struct Gizmo {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> g: Gizmo;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip = g.mvp * vec4<f32>(pos, 1.0);
    out.color = color;
    out.normal = normal;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.3, 0.6, 0.7));
    let d = max(dot(normalize(in.normal), light), 0.0);
    return vec4<f32>(in.color * (0.4 + 0.6 * d), 1.0);
}

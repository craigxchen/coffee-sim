// Unlit line pipeline for the SDF solid wireframe. Vertices are world-space `[pos, color]` line
// endpoints (LineList); the only uniform is the scene view-projection. Trivial pass-through so it
// stays portable to the browser (no workgroup/uniformity constructs).

struct Wire {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> wire: Wire;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = wire.view_proj * vec4<f32>(in.pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

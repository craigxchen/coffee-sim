// Sphere-impostor particle render: instanced camera-facing quads shaded as lit spheres,
// colored by speed. Reads the canonical particle position/velocity storage buffers directly.

struct Camera {
    view_proj: mat4x4<f32>,
    right: vec4<f32>,   // camera right in world space (xyz)
    up: vec4<f32>,      // camera up in world space (xyz)
    params: vec4<f32>,  // x = particle radius, y = inv color-max-speed
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> phases: array<u32>;

const PHASE_GRAIN: u32 = 1u;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) speed: f32,
    @location(2) @interpolate(flat) phase: u32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi];
    let p = positions[ii].xyz;
    let radius = cam.params.x;
    let world = p + cam.right.xyz * c.x * radius + cam.up.xyz * c.y * radius;

    var out: VsOut;
    out.clip = cam.view_proj * vec4<f32>(world, 1.0);
    out.uv = c;
    out.speed = length(velocities[ii].xyz);
    out.phase = phases[ii];
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.uv, in.uv);
    if (d2 > 1.0) {
        discard;
    }
    // Sphere normal in view space (impostor).
    let n = normalize(vec3<f32>(in.uv.x, in.uv.y, sqrt(1.0 - d2)));
    let light = normalize(vec3<f32>(0.4, 0.6, 0.7));
    let shade = 0.25 + 0.75 * max(dot(n, light), 0.0);

    let t = clamp(in.speed * cam.params.y, 0.0, 1.0);
    var calm: vec3<f32>;
    var fast: vec3<f32>;
    if (in.phase == PHASE_GRAIN) {
        // Coffee grounds: brown, lightening slightly when disturbed.
        calm = vec3<f32>(0.30, 0.18, 0.10);
        fast = vec3<f32>(0.60, 0.42, 0.26);
    } else {
        calm = vec3<f32>(0.10, 0.32, 0.85);
        fast = vec3<f32>(0.75, 0.92, 1.0);
    }
    return vec4<f32>(mix(calm, fast, t) * shade, 1.0);
}

// Two-field solver shared core (U1 scaffold): the Params mirror, the global binding table,
// and the single inert pass. Later pass families (transfers / pressure / plasticity /
// coupling) are concatenated after this file into one shader module (WGSL has no imports),
// so this file declares everything they share. Binding indices are GLOBAL and stable across
// the module: each pipeline's auto layout (`layout: None`) keeps only the bindings its entry
// point actually uses, but an index never means two different buffers.
//
// Storage-buffer budget (KTD-7): the device requests 9 storage buffers per stage
// (src/utils/gpu.rs NEEDED_STORAGE_BUFFERS) and U1 deliberately does NOT raise it — the
// scaffold's widest entry point binds 2. When a real pass approaches the cap, pack grid
// fields into vec4 lanes first; AGENTS.md sanctions raising the request to 16 per stage only
// if packing fails.
//
// Tint discipline: any future workgroupBarrier() must be reachable from uniform control flow
// — no early returns before a barrier (clamp indices and predicate the work instead). Native
// naga tolerates the divergence; Tint (the browser compiler) rejects it. The scaffold pass
// has no barriers, so its guard return is safe.

const PHASE_WATER: u32 = 0u;
const PHASE_SOLID: u32 = 1u;

// Byte-identical to the Rust `Params` (16 bytes; the tail stays vec4-aligned as it grows).
struct Params {
    dt: f32,
    water_count: u32,    // particles [0, water_count) are water (KTD-1 range layout)
    solid_count: u32,    // particles [water_count, water_count + solid_count) are solid grains
    particle_count: u32, // = water_count + solid_count (kernel live-set guard)
};

@group(0) @binding(0) var<uniform> params: Params;
// Canonical particle state (ParticleBuffers layout): pos.w is the moisture lane (water =
// remaining fraction, grain = absorbed volume), chem = (concentration, temperature, _, _).
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> phase: array<u32>;
@group(0) @binding(4) var<storage, read_write> chem: array<vec4<f32>>;

// Inert integrate: advance positions by the (zero-seeded) velocities. No physics — it exists
// so the dispatch/profiling path is real from U1 on (profile().dispatches_per_frame > 0) and
// the budget gates measure something true. Replaced by the APIC G2P update in U2.
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let p = pos[i];
    pos[i] = vec4<f32>(p.xyz + vel[i].xyz * params.dt, p.w);
}

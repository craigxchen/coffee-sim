// Shared core for the unified XPBD solver: params/status/bindings, SPH + grid math, and the
// kernels both species use (predict, grid build, apply_dp, finalize). The water-incompressibility
// kernels live in `water.wgsl`; the granular-bed kernel in `bed.wgsl`. The three files are
// concatenated into one shader module at build time (WGSL has no imports), so this file declares
// everything the others reference.
//
// Two species share one particle system + neighbor grid, tagged by `phase`: water solves a
// density constraint (water.wgsl); grains solve contact + friction + cohesion (bed.wgsl). A
// scene is single-species for now, so the step loop runs only that species' passes — but the
// per-particle `phase` guards keep each kernel correct regardless of who calls it.

const PHASE_WATER: u32 = 0u;
const PHASE_GRAIN: u32 = 1u;

struct Params {
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    gravity: vec4<f32>,
    grid_origin: vec4<f32>,   // = box_min
    grid_dims: vec4<u32>,     // nx, ny, nz, num_cells
    dt: f32,
    h: f32,
    rest_density: f32,
    particle_mass: f32,
    s_corr_k: f32,
    s_corr_n: f32,
    s_corr_wq: f32,           // W_poly6(Δq, h)
    relaxation_eps: f32,
    position_relaxation: f32, // ω
    xsph_c: f32,
    max_speed: f32,
    spiky_r_min: f32,         // ε_r · h
    cell_size: f32,           // = h
    particle_count: u32,
    num_solids: u32,          // count of static SDF solids in the `solids` buffer (0 = none)
    min_iters: u32,
    max_iters: u32,
    residual_tolerance: f32,
    lambda_noncohesive: u32,
    max_correction: f32,
    velocity_damping: f32,
    // --- granular bed (grain phase) ---
    grain_diameter: f32,      // contact diameter d (separation kicks in below this)
    friction_mu: f32,         // grain–grain Coulomb coefficient
    floor_mu: f32,            // grain–boundary (floor/wall) Coulomb coefficient
    dry_cohesion: f32,        // weak short-range attraction strength (0 = none)
    cohesion_range: f32,      // cohesion/contact search reach (absolute; d ≤ r < this)
    rolling_damping: f32,     // grain velocity retained per frame (rolling-resistance proxy)
    grain_sleep_speed: f32,   // static-yield dead-band: below this a grain is treated as at rest
    // --- water/bed coupling (mixed scenes) ---
    grain_mass: f32,
    grain_volume: f32,        // (π/6)·grain_diameter³ — effective volume for the α_s sum
    packing_limit: f32,       // α_s clamp (~0.64)
    exclusion_relax: f32,     // under-relaxation on the A.2 correction
    drag_gamma: f32,
    drag_beta_max: f32,
    buoyancy_scale: f32,
    wake_threshold: f32,
    water_grain_distance: f32, // water↔grain exclusion contact (≤ grain spacing lets water thread pores)
    // --- wetting / cohesion (Phase 1.4) ---
    r_max: f32,                // moisture ratio at saturation (mass water / mass dry grain)
    rho_ratio: f32,            // ρ_s/ρ_w — converts absorbed water mass → swelling volume
    s_peak: f32,               // saturation at the cohesion-curve peak
    c_max: f32,                // peak wet cohesion strength (0 until calibrated)
    k_abs: f32,                // absorption rate constant (1/s)
    absorb_roundoff: f32,      // f_w deactivation floor (exact-conservation; ≪ pbf_eps)
    pbf_eps: f32,              // PBF skips water with remaining fraction ≤ this
};

struct Status {
    overflow: atomic<u32>,
    max_occupancy: atomic<u32>,
    converged: u32,
    iters_done: u32,
    effective_iters: u32,
    residual_bits: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pred: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel_smoothed: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> lambda: array<f32>;
@group(0) @binding(6) var<storage, read_write> dp: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> c_residual: array<f32>;
// Counting-sort spatial hash (no fixed buckets / overflow): cell_start is the exclusive prefix sum
// of per-cell counts (num_cells+1 entries), and sorted_indices holds particle indices grouped by
// cell. Gather cell c = sorted_indices[cell_start[c] .. cell_start[c+1]]. cell_count (binding 18) is
// the transient counter used only while (re)building the grid.
@group(0) @binding(8) var<storage, read_write> cell_start: array<u32>;
@group(0) @binding(9) var<storage, read_write> sorted_indices: array<u32>;
@group(0) @binding(10) var<storage, read_write> status: Status;
@group(0) @binding(11) var<storage, read_write> phase: array<u32>;
// Per-grain accumulated normal-correction magnitude this frame (reset in predict). The Coulomb
// friction budget is μ·normal_impulse — load-scaled and non-zero at static rest, unlike μ·overlap.
@group(0) @binding(12) var<storage, read_write> normal_impulse: array<f32>;
// Per-particle solid fraction α_s = Σ grain V_g W (clamped to the packing limit). Computed each
// iteration by compute_fractions in mixed scenes; zero in single-species scenes (so the water
// density solve is unmodulated there). The water target becomes ρ₀·(1−α_s) → pore water packs to
// the pore fraction (drainage-ready), and the geometric exclusion keeps water out of grain bodies.
@group(0) @binding(13) var<storage, read_write> alpha_s: array<f32>;
// Per-grain accumulated |water↔grain drag impulse| this frame; it wakes the static dead-band
// under fluid load. Reset in predict, written by drag_grain, read in finalize.
@group(0) @binding(14) var<storage, read_write> fluid_impulse: array<f32>;
// Frozen velocity snapshot for symmetric water↔grain drag gathers. Both drag passes read the
// same snapshot so every pair computes equal-and-opposite impulses without atomics.
@group(0) @binding(15) var<storage, read_write> vel_frozen: array<vec4<f32>>;
// Per-particle drag blend cap computed from the opposite-phase neighbor count.
@group(0) @binding(16) var<storage, read_write> coupling_scale: array<f32>;
// Per-particle count of ELIGIBLE opposite-species neighbors for absorption (wetting): water → N_w
// (# unsaturated grains), grain → N_g (# non-empty waters). Written by wet_count, read by both
// transfer passes so the two-sided allocation take_wg is identical (and conservation-safe).
@group(0) @binding(17) var<storage, read_write> wet_neighbors: array<u32>;
// Transient per-cell particle counter for the counting-sort grid build (count → scan → scatter).
@group(0) @binding(18) var<storage, read_write> cell_count: array<atomic<u32>>;

// Static SDF solid geometry (read-only). Each Primitive is a cavity: sample > 0 inside (allowed),
// < 0 through the wall. Only the collision passes (apply_dp / apply_drag_pred) reach `solid_union`,
// so this binding is absent from every other kernel's auto-derived layout (8-buffer budget holds).
// Byte-identical to the Rust `Primitive` (64 bytes). Cone radii in `a` are OUTER wall radii.
struct Primitive {
    kind: u32,          // 0 = cone, 1 = cylinder
    species_mask: u32,
    friction: f32,
    flags: u32,         // bit0 = apex_open
    a: vec4<f32>,       // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)
    b: vec4<f32>,       // cone:(thickness, hole_radius, center_x, center_z)  cyl:(center_x, center_z, _, _)
    c: vec4<f32>,       // reserved
};
@group(0) @binding(19) var<storage, read> solids: array<Primitive>;

const PI: f32 = 3.14159265358979;

// --- SDF cavity geometry (mirrors utils/sdf.rs; interior positive, gradient toward the cavity) ---
const SDF_EPS: f32 = 1.0e-6;
const SDF_FREE: f32 = 1.0e30;

struct SolidHit { dist: f32, grad: vec3<f32>, friction: f32 };
struct SegCP { p: vec2<f32>, interior: bool };

fn sdf_normalize2(v: vec2<f32>) -> vec2<f32> {
    let len = length(v);
    if (len > SDF_EPS) { return v / len; }
    return vec2<f32>(0.0, 0.0);
}
fn sdf_normalize3(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len > SDF_EPS) { return v / len; }
    return vec3<f32>(0.0, 0.0, 0.0);
}
fn sdf_closest_on_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> SegCP {
    let ab = b - a;
    let len2 = dot(ab, ab);
    if (len2 < SDF_EPS) { return SegCP(a, false); }
    let t = dot(p - a, ab) / len2;
    let tc = clamp(t, 0.0, 1.0);
    return SegCP(a + ab * tc, t > SDF_EPS && t < 1.0 - SDF_EPS);
}

fn cone_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let apex_y = prim.a.x;
    let apex_r = prim.a.y;
    let top_y = prim.a.z;
    let top_r = prim.a.w;
    let thickness = prim.b.x;
    let hole_radius = prim.b.y;
    let center = vec2<f32>(prim.b.z, prim.b.w);
    let apex_open = (prim.flags & 1u) != 0u;

    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let r = length(rel);
    let y = p.y;
    var rhat = vec2<f32>(0.0, 0.0);
    if (r >= SDF_EPS) { rhat = rel / r; } // axis guard before building the 3D gradient

    if (y > top_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, 1.0, 0.0), prim.friction); }
    if (apex_open && y < apex_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, -1.0, 0.0), prim.friction); }

    let inner_apex = max(apex_r - thickness, hole_radius);
    let inner_top = max(top_r - thickness, hole_radius);
    let a2 = vec2<f32>(inner_apex, apex_y);
    let b2 = vec2<f32>(inner_top, top_y);
    let cp = sdf_closest_on_segment(vec2<f32>(r, y), a2, b2);
    let d2d = length(vec2<f32>(r, y) - cp.p);

    let height = top_y - apex_y;
    var t = 0.0;
    if (height > SDF_EPS) { t = (y - apex_y) / height; }
    let inner = max(mix(apex_r, top_r, t) - thickness, hole_radius);
    let inside = (y >= apex_y) && (r <= inner);
    var dist = -d2d;
    if (inside) { dist = d2d; }

    let dir = vec2<f32>(r, y) - cp.p;
    var n2d = vec2<f32>(0.0, 0.0);
    if (length(dir) > SDF_EPS) {
        if (inside) { n2d = sdf_normalize2(dir); } else { n2d = sdf_normalize2(-dir); }
    } else if (cp.interior) {
        let ab = b2 - a2;
        n2d = sdf_normalize2(vec2<f32>(-ab.y, ab.x)); // smooth-wall analytic normal (never zero here)
    }
    let grad = sdf_normalize3(vec3<f32>(n2d.x * rhat.x, n2d.y, n2d.x * rhat.y));
    return SolidHit(dist, grad, prim.friction);
}

fn cyl_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let floor_y = prim.a.x;
    let rim_y = prim.a.y;
    let radius = prim.a.z;
    let center = vec2<f32>(prim.b.x, prim.b.y);
    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let r = length(rel);
    let y = p.y;
    if (y > rim_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, 1.0, 0.0), prim.friction); }
    let d_side = radius - r;
    let d_floor = y - floor_y;
    if (d_side <= d_floor) {
        var rhat = vec2<f32>(0.0, 0.0);
        if (r >= SDF_EPS) { rhat = rel / r; }
        return SolidHit(d_side, sdf_normalize3(vec3<f32>(-rhat.x, 0.0, -rhat.y)), prim.friction);
    }
    return SolidHit(d_floor, vec3<f32>(0.0, 1.0, 0.0), prim.friction);
}

fn solid_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    if (prim.kind == 0u) { return cone_cavity(prim, p); }
    return cyl_cavity(prim, p);
}

// Species-filtered union: the most-penetrated (min signed-distance) solid that blocks `phase`.
fn solid_union(p: vec3<f32>, phase: u32) -> SolidHit {
    var best = SolidHit(SDF_FREE, vec3<f32>(0.0, 0.0, 0.0), 0.0);
    let n = params.num_solids;
    for (var i = 0u; i < n; i = i + 1u) {
        let prim = solids[i];
        if ((prim.species_mask & (1u << phase)) == 0u) { continue; }
        let hit = solid_cavity(prim, p);
        if (hit.dist < best.dist) { best = hit; }
    }
    return best;
}

fn w_poly6(r: f32, h: f32) -> f32 {
    if (r >= h) { return 0.0; }
    let t = h * h - r * r;
    let coeff = 315.0 / (64.0 * PI * pow(h, 9.0));
    return coeff * t * t * t;
}

// True spiky gradient ∇W(d) for d = x_i − x_j (decreases with r ⇒ negative along d), with an
// r→0 safeguard so near-coincident particles still get a finite separation force.
fn spiky_grad(d: vec3<f32>, h: f32, r_min: f32) -> vec3<f32> {
    let len = length(d);
    if (len >= h) { return vec3<f32>(0.0); }
    var r = len;
    var dir: vec3<f32>;
    if (len < r_min) {
        r = r_min;
        if (len < 1e-8) {
            dir = vec3<f32>(1.0, 0.0, 0.0); // deterministic fallback direction
        } else {
            dir = d / len;
        }
    } else {
        dir = d / len;
    }
    let coeff = 45.0 / (PI * pow(h, 6.0));
    let t = h - r;
    return -coeff * t * t * dir;
}

// --- Phase-1.4 swelling + effective mass (from the moisture lane: grain pred.w = V_abs, water = f_w).
// A grain swells by exactly the absorbed volume; its mass gains the absorbed water's mass; a water
// shrinks proportionally. Dry/full state (V_abs=0, f_w=1) ⇒ the original constants exactly.
fn grain_eff_volume(v_abs: f32) -> f32 { return params.grain_volume + v_abs; }
fn grain_eff_diameter(v_abs: f32) -> f32 {
    return params.grain_diameter * pow(grain_eff_volume(v_abs) / params.grain_volume, 1.0 / 3.0);
}
fn grain_eff_mass(v_abs: f32) -> f32 { return params.grain_mass + params.rest_density * v_abs; }
fn water_eff_mass(f_w: f32) -> f32 { return params.particle_mass * f_w; }
// Effective mass of a particle from its phase + moisture-lane value `w` (grain V_abs / water f_w).
fn eff_mass(ph: u32, w: f32) -> f32 {
    if (ph == PHASE_GRAIN) { return grain_eff_mass(w); }
    return water_eff_mass(w);
}

// Density-aware buoyancy scaling: the buoyant acceleration is ∝ ρ_water/ρ_grain, so a grain DENSER
// than water (dry coffee, ρ≈several×) is barely buoyed and still sinks, while a saturated grain
// (ρ→ρ_water) is neutrally buoyant. Clamped ≤ 1. Both buoyancy passes apply the grain's factor per
// pair, so the impulse stays equal-and-opposite (momentum conserved). Fixes the dam over-lift where
// the uncalibrated λ proxy launched denser-than-water grains out of the column.
fn grain_buoyancy_factor(v_abs: f32) -> f32 {
    let rho_g = grain_eff_mass(v_abs) / max(grain_eff_volume(v_abs), 1.0e-6);
    return clamp(params.rest_density / max(rho_g, 1.0e-6), 0.0, 1.0);
}

// Saturation→cohesion curve (mirrors models::cohesion::for_saturation): a piecewise-linear bump,
// 0 at dry (s=0) and full saturation (s=1), peak c_max at s_peak, with s = V_abs / V_cap.
fn wet_cohesion(v_abs: f32) -> f32 {
    let v_cap = params.r_max * params.rho_ratio * params.grain_volume;
    let s = clamp(v_abs / max(v_cap, 1.0e-12), 0.0, 1.0);
    let sp = clamp(params.s_peak, 1.0e-6, 1.0 - 1.0e-6);
    var g: f32;
    if (s <= sp) {
        g = s / sp;
    } else {
        g = (1.0 - s) / (1.0 - sp);
    }
    return params.c_max * g;
}

fn cell_coord(p: vec3<f32>) -> vec3<i32> {
    let rel = (p - params.grid_origin.xyz) / params.cell_size;
    let dims = vec3<i32>(params.grid_dims.xyz);
    let c = vec3<i32>(floor(rel));
    return clamp(c, vec3<i32>(0, 0, 0), dims - vec3<i32>(1, 1, 1));
}

fn cell_id(c: vec3<i32>) -> u32 {
    let dims = params.grid_dims;
    return u32(c.x) + dims.x * (u32(c.y) + dims.y * u32(c.z));
}

@compute @workgroup_size(256)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i == 0u) {
        atomicStore(&status.overflow, 0u);
        atomicStore(&status.max_occupancy, 0u);
        status.converged = 0u;
        status.iters_done = 0u;
        status.effective_iters = params.max_iters;
        status.residual_bits = 0u;
    }
    if (i >= params.particle_count) { return; }
    normal_impulse[i] = 0.0; // reset the per-frame friction-budget accumulator (grains)
    fluid_impulse[i] = 0.0; // reset the per-frame water↔grain wake signal
    let p = pos[i].xyz;
    let v = vel[i].xyz;
    let g = params.gravity.xyz;
    let np = p + params.dt * v + (params.dt * params.dt) * g;
    // Mirror the moisture lane (pos.w) into pred.w so the predicted-state kernels can read it; the
    // absorption passes treat pred.w as the frozen per-frame snapshot.
    pred[i] = vec4<f32>(np, pos[i].w);
}

@compute @workgroup_size(256)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.grid_dims.w) { return; }
    atomicStore(&cell_count[c], 0u);
}

// Counting-sort grid build: grid_clear (zero cell_count) → grid_count → grid_scan → grid_clear →
// grid_scatter. No fixed buckets, so no overflow and no wasted memory — every neighbor is gathered.

@compute @workgroup_size(256)
fn grid_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let cell = cell_id(cell_coord(pred[i].xyz));
    let slot = atomicAdd(&cell_count[cell], 1u);
    atomicMax(&status.max_occupancy, slot + 1u);
}

var<workgroup> scan_tmp: array<u32, 256>;
var<workgroup> scan_total: u32;

// Single-workgroup exclusive prefix sum of cell_count → cell_start (num_cells+1; the last entry is
// the total). Dispatched as ONE workgroup; each thread scans a contiguous chunk, thread 0 scans the
// 256 chunk sums, then each thread writes its chunk's prefixes.
@compute @workgroup_size(256)
fn grid_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = params.grid_dims.w;
    let tid = lid.x;
    let threads = 256u;
    let chunk = (n + threads - 1u) / threads;
    let begin = tid * chunk;

    var s = 0u;
    for (var k = 0u; k < chunk; k = k + 1u) {
        let idx = begin + k;
        if (idx < n) { s = s + atomicLoad(&cell_count[idx]); }
    }
    scan_tmp[tid] = s;
    workgroupBarrier();

    if (tid == 0u) {
        var acc = 0u;
        for (var t = 0u; t < threads; t = t + 1u) {
            let v = scan_tmp[t];
            scan_tmp[t] = acc;
            acc = acc + v;
        }
        scan_total = acc;
    }
    workgroupBarrier();

    var run = scan_tmp[tid];
    for (var k = 0u; k < chunk; k = k + 1u) {
        let idx = begin + k;
        if (idx < n) {
            cell_start[idx] = run;
            run = run + atomicLoad(&cell_count[idx]);
        }
    }
    if (tid == 0u) { cell_start[n] = scan_total; }
}

@compute @workgroup_size(256)
fn grid_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let cell = cell_id(cell_coord(pred[i].xyz));
    // cell_count was re-zeroed after the scan, so it serves as the per-cell write cursor here.
    let local = atomicAdd(&cell_count[cell], 1u);
    sorted_indices[cell_start[cell] + local] = i;
}

@compute @workgroup_size(256)
fn apply_dp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    // Cap the per-iteration correction so no single overcorrection can launch a particle.
    var d = params.position_relaxation * dp[i].xyz;
    let dl = length(d);
    if (dl > params.max_correction) {
        d = d * (params.max_correction / dl);
    }
    var proposed = pred[i].xyz + d;

    // Static-solid (SDF) push-out, selective by species: push a penetrating particle out along the
    // cavity gradient by the penetration depth. The standoff is the grain radius (the bed contact
    // value; per-species tuning deferred). No-op when num_solids == 0, so AABB-only scenes are
    // byte-unchanged. solid_union returns FREE for non-applicable species (e.g. water vs the filter).
    let contact = 0.5 * params.grain_diameter;
    var sdf_n = vec3<f32>(0.0, 0.0, 0.0);
    var sdf_mag = 0.0;
    var sdf_mu = params.floor_mu;
    if (params.num_solids > 0u) {
        let hit = solid_union(proposed, phase[i]);
        if (hit.dist < contact) {
            sdf_mag = contact - hit.dist;
            proposed = proposed + sdf_mag * hit.grad;
            sdf_n = hit.grad;
            sdf_mu = hit.friction;
        }
    }

    let clamped = clamp(proposed, params.box_min.xyz, params.box_max.xyz);

    if (phase[i] == PHASE_GRAIN) {
        // The boundary normal response is the box clamp delta and/or the SDF push above. Apply
        // Coulomb friction along the *dominant* boundary: remove the grain's tangential drift
        // (relative to its frame-start position) up to mu · |normal correction|, using the solid's
        // friction when the SDF push dominated, else floor_mu. This is the angle-of-repose mechanism.
        let push = clamped - proposed;
        let box_mag = length(push);
        var newp = clamped;
        var n = vec3<f32>(0.0, 0.0, 0.0);
        var nmag = 0.0;
        var mu = params.floor_mu;
        if (box_mag >= sdf_mag) {
            if (box_mag > 1e-6) {
                n = push / box_mag;
                nmag = box_mag;
                mu = params.floor_mu;
            }
        } else {
            n = sdf_n;
            nmag = sdf_mag;
            mu = sdf_mu;
        }
        if (nmag > 1e-6) {
            let disp = newp - pos[i].xyz;
            let tang = disp - dot(disp, n) * n;
            let tlen = length(tang);
            if (tlen > 1e-8) {
                let remove = min(tlen, mu * nmag);
                newp = newp - (tang / tlen) * remove;
            }
            // Post-friction SDF recheck: the tangential slide can re-enter a wall (only the box is
            // re-clamped below), so re-project out of the solid first, then re-clamp to the box.
            if (params.num_solids > 0u) {
                let h2 = solid_union(newp, phase[i]);
                if (h2.dist < contact) {
                    newp = newp + (contact - h2.dist) * h2.grad;
                }
            }
            newp = clamp(newp, params.box_min.xyz, params.box_max.xyz);
        }
        pred[i] = vec4<f32>(newp, pred[i].w); // preserve the moisture snapshot in pred.w
    } else {
        pred[i] = vec4<f32>(clamped, pred[i].w); // preserve the moisture snapshot in pred.w
    }
}

@compute @workgroup_size(256)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let xi = pred[i].xyz;
    var v = ((xi - pos[i].xyz) / params.dt) * params.velocity_damping;

    if (phase[i] == PHASE_GRAIN) {
        let pen = c_residual[i]; // final normalized penetration from the last bed_project
        // Rolling-resistance proxy: bleed grain kinetic energy so spheres don't roll the pile
        // shallow — but ONLY while in contact. An airborne grain isn't rolling, so damping its
        // free-fall would just rob the impact of the energy it needs to flow into a heap.
        if (pen > 1e-4) {
            v = v * params.rolling_damping;
        }
        // Static-yield regularization (NOT freezing): below a small speed a grain is treated as
        // at rest. Gravity and the contact solve still run for it every frame, so an unsupported
        // grain immediately re-accelerates and penetration is never masked — this only removes
        // the sub-threshold numerical jitter a Jacobi contact pile never fully shakes off.
        if (fluid_impulse[i] <= params.wake_threshold && length(v) < params.grain_sleep_speed) {
            v = vec3<f32>(0.0);
        }
    }

    let sp = length(v);
    if (sp > params.max_speed) {
        v = v * (params.max_speed / sp);
    }
    vel[i] = vec4<f32>(v, 0.0);
    pos[i] = vec4<f32>(xi, pos[i].w); // preserve the persistent moisture lane (pos.w)
}

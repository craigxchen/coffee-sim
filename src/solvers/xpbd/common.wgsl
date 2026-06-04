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
    bucket_capacity: u32,     // K
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
@group(0) @binding(8) var<storage, read_write> cell_count: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> cell_bucket: array<u32>;
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

const PI: f32 = 3.14159265358979;

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

@compute @workgroup_size(256)
fn grid_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let cell = cell_id(cell_coord(pred[i].xyz));
    let slot = atomicAdd(&cell_count[cell], 1u);
    if (slot < params.bucket_capacity) {
        cell_bucket[cell * params.bucket_capacity + slot] = i;
        atomicMax(&status.max_occupancy, slot + 1u);
    } else {
        atomicStore(&status.overflow, 1u);
    }
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
    let proposed = pred[i].xyz + d;
    let clamped = clamp(proposed, params.box_min.xyz, params.box_max.xyz);

    if (phase[i] == PHASE_GRAIN) {
        // The box clamp IS the boundary normal response; its magnitude is the normal correction.
        // Apply Coulomb friction along the wall: remove the grain's tangential drift (relative to
        // its frame-start position) up to floor_mu · |normal correction|. This is what stops the
        // pile from sliding flat on the floor — without it there is no angle of repose.
        let push = clamped - proposed;
        let nmag = length(push);
        var newp = clamped;
        if (nmag > 1e-6) {
            let n = push / nmag;
            let disp = newp - pos[i].xyz;
            let tang = disp - dot(disp, n) * n;
            let tlen = length(tang);
            if (tlen > 1e-8) {
                let remove = min(tlen, params.floor_mu * nmag);
                newp = newp - (tang / tlen) * remove;
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

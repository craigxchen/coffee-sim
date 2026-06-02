// PBF water core (water incompressibility only) — Macklin & Müller 2013.
// One @group(0); each entry point uses a subset of the bindings (auto bind-group layouts
// keep every kernel within the WebGPU 8-storage-buffer limit).

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
    _pad0: u32,
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
    let p = pos[i].xyz;
    let v = vel[i].xyz;
    let g = params.gravity.xyz;
    let np = p + params.dt * v + (params.dt * params.dt) * g;
    pred[i] = vec4<f32>(np, 0.0);
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
fn compute_lambda(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    let h = params.h;
    let m = params.particle_mass;
    let xi = pred[i].xyz;

    var rho = m * w_poly6(0.0, h);
    var sum_g = vec3<f32>(0.0);
    var sum_g2 = 0.0;

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (j == i) { continue; }
                    let d = xi - pred[j].xyz;
                    let r = length(d);
                    if (r >= h) { continue; }
                    rho = rho + m * w_poly6(r, h);
                    let gj = m * spiky_grad(d, h, params.spiky_r_min);
                    sum_g = sum_g + gj;
                    sum_g2 = sum_g2 + dot(gj, gj);
                }
            }
        }
    }

    let inv_rho0 = 1.0 / params.rest_density;
    let c = rho * inv_rho0 - 1.0;
    let denom = (dot(sum_g, sum_g) + sum_g2) * inv_rho0 * inv_rho0 + params.relaxation_eps;
    var lam = -c / denom;
    if (params.lambda_noncohesive != 0u && lam > 0.0) {
        lam = 0.0;
    }
    lambda[i] = lam;
    // Convergence is measured on COMPRESSION only: free-surface particles always have
    // c ≈ −1 (density deficit), which incompressibility neither can nor should fix, so
    // including them would prevent early-exit forever. max(0, c) = over-density violation.
    c_residual[i] = max(0.0, c);
}

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn residual_reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (status.converged != 0u) { return; }
    let tid = lid.x;
    var local_max = 0.0;
    var i = tid;
    loop {
        if (i >= params.particle_count) { break; }
        local_max = max(local_max, c_residual[i]);
        i = i + 256u;
    }
    sdata[tid] = local_max;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (tid == 0u) {
        let gmax = sdata[0];
        status.residual_bits = bitcast<u32>(gmax);
        status.iters_done = status.iters_done + 1u;
        if (status.iters_done >= params.min_iters && gmax < params.residual_tolerance) {
            status.converged = 1u;
            status.effective_iters = status.iters_done;
        }
    }
}

@compute @workgroup_size(256)
fn compute_dp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    let h = params.h;
    let m = params.particle_mass;
    let xi = pred[i].xyz;
    let lam_i = lambda[i];

    var sum = vec3<f32>(0.0);
    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (j == i) { continue; }
                    let d = xi - pred[j].xyz;
                    let r = length(d);
                    if (r >= h) { continue; }
                    let wr = w_poly6(r, h);
                    let ratio = wr / params.s_corr_wq;
                    let scorr = -params.s_corr_k * pow(ratio, params.s_corr_n);
                    let gj = m * spiky_grad(d, h, params.spiky_r_min);
                    sum = sum + (lam_i + lambda[j] + scorr) * gj;
                }
            }
        }
    }
    let inv_rho0 = 1.0 / params.rest_density;
    dp[i] = vec4<f32>(sum * inv_rho0, 0.0);
}

@compute @workgroup_size(256)
fn apply_dp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    if (status.converged != 0u) { return; }
    let np = pred[i].xyz + params.position_relaxation * dp[i].xyz;
    let clamped = clamp(np, params.box_min.xyz, params.box_max.xyz);
    pred[i] = vec4<f32>(clamped, 0.0);
}

@compute @workgroup_size(256)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let xi = pred[i].xyz;
    var v = (xi - pos[i].xyz) / params.dt;
    let sp = length(v);
    if (sp > params.max_speed) {
        v = v * (params.max_speed / sp);
    }
    vel[i] = vec4<f32>(v, 0.0);
    pos[i] = vec4<f32>(xi, 0.0);
}

@compute @workgroup_size(256)
fn xsph(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) { return; }
    let h = params.h;
    let xi = pos[i].xyz;
    let vi = vel[i].xyz;
    var dv = vec3<f32>(0.0);

    let base = cell_coord(xi);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = base + vec3<i32>(dx, dy, dz);
                if (nc.x < 0 || nc.y < 0 || nc.z < 0) { continue; }
                let dims = vec3<i32>(params.grid_dims.xyz);
                if (nc.x >= dims.x || nc.y >= dims.y || nc.z >= dims.z) { continue; }
                let cid = cell_id(nc);
                let cnt = min(atomicLoad(&cell_count[cid]), params.bucket_capacity);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = cell_bucket[cid * params.bucket_capacity + s];
                    if (j == i) { continue; }
                    let d = xi - pos[j].xyz;
                    let r = length(d);
                    if (r >= h) { continue; }
                    dv = dv + (vel[j].xyz - vi) * w_poly6(r, h);
                }
            }
        }
    }
    vel_smoothed[i] = vec4<f32>(vi + params.xsph_c * dv, 0.0);
}

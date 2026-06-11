// U3 incompressibility (plan 2026-06-09-001, KTD-2): coarse-grid pressure seed + a fixed
// budget of fine local damped-Jacobi sweeps — NO converged Poisson solve anywhere.
//
// ================================ OPERATOR FAMILY =============================================
// ONE discretely consistent family on the U2 collocated grid (velocity at nodes):
//
//   pressure p     : CELL CENTERS (fine cells = grid_dims − 1 per axis)
//   divergence D   : corner-trilinear / finite-volume — for cell c with corner nodes n at
//                    offsets o ∈ {0,1}³ and per-axis signs s(o) = 2o − 1:
//                        (D u)_c = Σ_n  s(o)·u_n / (4h)
//                    (the exact derivative of trilinear interpolation at the cell center;
//                    2nd-order, gated by the MMS sweep). Masked rows: inactive cells ≡ 0.
//   gradient G     : G = −Dᵀ BY CONSTRUCTION — the same sign function, gathered the other way:
//                        (G p)_n = −Σ_{adjacent ACTIVE cells c}  s(o(n,c))·p_c / (4h)
//                    Adjointness identity the gates assert on twins AND GPU readbacks:
//                        u·(G p) = −(D u)·p          for ALL u, p, on ANY masks.
//   mass weighting : per-node symmetric constrained inverse-mass matrix
//                        M̃⁻¹_n = invρ_n·(P_box − n̂ n̂ᵀ),    invρ_n = 1 / max(ρ_n, ρ_floor),
//                    where ρ_n = node mass / h³, P_box zeroes box-face axes (mutually
//                    orthogonal), and n̂ is the SDF wall normal orthogonalized against the box
//                    axes — symmetric PSD, so wall-Neumann rows are part of the operator, not
//                    bolted on. Massless nodes get M̃⁻¹ = 0 (the projection never moves them).
//   Laplacian      : EXISTS ONLY AS THE COMPOSITION  A = −D·M̃⁻¹·G,  evaluated inline in the
//                    Jacobi kernels by literally composing the three loops above. Never a
//                    hand-written stencil — a hand stencil is exactly how main got A ≠ D·G and
//                    a divergence floor no iteration could beat.
//   projection     : v ← v − dt·M̃⁻¹·(G p). The divergence the gates measure afterwards uses
//                    THE SAME D, so post-projection divergence = dt·(solver residual): an
//                    operator-inconsistency floor is impossible by construction, and the
//                    decay gate (tests/twofield_pressure.rs) verifies it empirically.
//
// ================================ FREE SURFACE (U3 minimal form) ==============================
// Ghost-fluid-style fraction weighting, refined (not replaced) by U4's fill-fraction Dirichlet
// + bubbles. Every cell carries a fill fraction f ∈ [0, 1] from GRID MASS (mean fluid-visible
// corner-node density vs SURF_FULL_FRAC·ρ_rest, clamped; cells failing the min-corner
// activity test below are empty, f = 0), and the family is row/column-weighted by F = diag(f):
//     D_f = F·D,   G_f = −D_fᵀ = −Dᵀ·F,   A = D_f·M̃⁻¹·D_fᵀ   (still symmetric PSD, any f).
// Interior cells (ρ̄ ≥ SURF_FULL_FRAC·ρ_rest) have f = 1 exactly — the family is UNCHANGED
// there (the MMS/adjointness/SPD gates pin both regimes; the order-of-accuracy claim is for
// the f = 1 rows, surface rows are first-order by design until U4). The pressure Dirichlet
// p = 0 still lives in empty cells, but the constraint rows taper through the partially
// filled band, so the air-side Dirichlet acts at the mass-weighted surface instead of a
// staircase one full cell off. The previous mask (all 8 corners above a bare epsilon) had
// no surface notion at all: it extended FULL-weight rows one cell into the particle-free
// B-spline smear band — the hovering-lid failure described below — and the settled tank
// pumped itself apart (the independent volume gates caught exactly this).
// SURF_* are fixed structural constants of the family (like JACOBI_OMEGA), not gate knobs:
// SURF_FULL_FRAC = 0.5 puts the f-taper strictly above the mean surface line (a cell cut by
// a flat surface has ρ̄ ≈ 0.5·ρ_rest), so interior rows can never flicker below f = 1.
// ACTIVITY is decided by the MINIMUM corner-node density (≥ SURF_MIN_CORNER·ρ_rest), not by
// the mean fill: the quadratic B-spline smears mass 1.5h past the topmost particles, so the
// first PARTICLE-FREE cell above a settled surface still measures mean fill f ≈ 0.5 (its
// bottom corners share the surface nodes' mass) — indistinguishable from a real surface cell
// by any mean-fill cut. Its far-side corners, however, carry ≤ ~0.05·ρ_rest, while a cell
// containing real fluid keeps ALL corners ≥ ~0.25·ρ_rest: the min-corner test is the sharp
// particle-presence discriminator (the U2 "all corners massy" rule with a physical threshold
// instead of an epsilon). Why smear cells must NOT get pressure rows: their row pins their
// floored-M̃⁻¹ dust mass against gravity — a hovering pressurized lid (p 1/f-amplified,
// observed growing 700→4700 over 50 frames) that blocks settling while the bottom's density
// relief keeps pressurizing; the settled tank pumped itself apart. Below the threshold,
// smear/droplet cells are ballistic (no row, implicit p = 0) — U2 behavior, confined to
// near-massless dust. Min-corner ≥ 0.1·ρ_rest also bounds f ≥ 0.2, i.e. the surface-row
// amplification 1/f ≤ 5.
// U4 refinement (surface.wgsl): the implicit p = 0 holds only for air OPEN to the boundary;
// ENCLOSED air (the pour pocket) is flood-fill detected and inserted as constraint rows with
// fill weight 1, pressure pinned to the shared bubble multiplier λ_b — see the BUBBLE
// REPRESENTATION header in surface.wgsl. cell_meta.w carries the cell category
// (CELL_AIR/CELL_POCKET); the pocket pins live in the Jacobi/prolong kernels below.
//
// ================================ COARSE SEED ================================================
// Coarse grid: REDISCRETIZED same family at H = ratio·h (ratio knob {4, 8}; default 4).
//   Why rediscretized and not Galerkin R·A·P: Galerkin requires materializing an explicit
//   coarse stencil — reintroducing the hand-assembled-operator failure surface KTD-2 forbids.
//   Rediscretization reuses this very code path at spacing H (one family, twin-pinned), and
//   the seed CANNOT ship inconsistency: it only seeds the fine sweeps, the projection applies
//   only the fine G, and the gate divergence uses only the fine D. The two-grid convergence
//   gate decides whether the rediscretized coarse operator is consistent ENOUGH to be useful.
//   Coarse masks: a coarse cell is active iff ANY of its ratio³ children is active (binary —
//   the fine fraction taper is deliberately NOT rediscretized; see coarse_cell_setup). Coarse
//   M̃⁻¹: hat-weighted restriction of the fine M̃⁻¹ field (see coarse_node_setup).
// Transfers: R = masked trilinear-transpose restriction with a FIXED 1/ratio³ volume weight
//   (no mask renormalization — that is what keeps R = Pᵀ/ratio³ exact), P = the same masked
//   hat weights gathered the other way. Gate: ⟨R u, v⟩_c = ⟨u, P v⟩_f (volume-weighted).
// Solve shape per frame (two-grid correction scheme, p₀ = 0 — no warm start):
//   rhs = (s_target − D v)/dt → PRE-SMOOTH (Nf/2 fine sweeps) → residual r = rhs − A·p →
//   restrict r → Nc coarse Jacobi sweeps on A_c·e = r_c from 0 → prolongate-and-ADD the
//   correction → POST-SMOOTH (remaining fine sweeps) → project.
//   Pre-smoothing + RESIDUAL restriction (not raw-rhs restriction) is load-bearing: the
//   production rhs is boundary-concentrated (the hydrostatic source is a one-fine-layer spike
//   at the floor), which a ratio-4 restriction cannot represent consistently — seeding from
//   the raw rhs measurably overshoots and pumps energy (observed: settled tank erupting).
//   Pre-smoothing absorbs the spike locally, so the restricted residual is smooth — the
//   regime where the R/P pair is consistent.
//
// Density relief (volume-conservation feedback): the projection constrains the VELOCITY
// divergence only, so any finite per-frame solve error compacts positions monotonically and
// the rhs never sees it (a fully-compacted tank measures ZERO divergence — the failure the
// independent volume gates exist to catch). The rhs therefore carries a one-sided density
// source — the same projection-RHS plumbing KTD-4 later formalizes as s_net:
//     rhs_c = (s_target − D v)/dt,  s_target = max(ρ̄_c/ρ_rest − 1, 0)/(DENSITY_RELAX·dt),
// with ρ̄_c the mean corner-node density: compression is relieved over a few frames; an
// under-dense (free-surface) cell gets NO suction — surface physics belongs to U4.
//
// Damped Jacobi: p ← p + ω·(rhs − A p)/diag(A), ω = JACOBI_OMEGA (fixed algorithmic constant,
// mirrored in mod.rs for the CPU twins), diag(A)_c = Σ_n s(o)ᵀ·M̃⁻¹_n·s(o) / (16h²).
//
// Adaptive-Tait predictor (KTD-2 option): STUBBED OFF — deliberately not implemented in U3.
// The knob grid pins Tait ∈ {off}; if a later unit shows an iteration-budget win it ships
// behind an opt-in Config gate, else it is removed (plan: "Deferred to Implementation").
//
// Sparsity: over-dispatch + early-out on inactive cells/empty nodes — NEVER indirect dispatch
// (R8). Tint discipline: no workgroupBarrier anywhere in this family, so the guard returns
// are uniform-control-flow safe. Storage buffers per entry point ≤ 5 (see mod.rs derivation).

// --- pressure-family bindings (indices continue the global table in common.wgsl) -------------
// Per-node M̃⁻¹, 2 vec4 per node: [2n] = (xx, xy, xz, yy), [2n+1] = (yz, zz, massy_flag, 0).
@group(0) @binding(9) var<storage, read_write> nm: array<vec4<f32>>;
// Per fine cell: (rhs, fill fraction f ∈ [0,1] — 0 = empty/masked, residual/dbg lane, unused).
@group(0) @binding(10) var<storage, read_write> cell_meta: array<vec4<f32>>;
// Fine pressure ping-pong. The shader always reads 11 and writes 12; the alternating bind
// groups swap which buffer sits at which index (pf_a starts as the seed slot).
@group(0) @binding(11) var<storage, read_write> pf_src: array<f32>;
@group(0) @binding(12) var<storage, read_write> pf_dst: array<f32>;
// Coarse mirrors of 9/10/11/12.
@group(0) @binding(13) var<storage, read_write> nm_c: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> cmeta: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> pc_src: array<f32>;
@group(0) @binding(16) var<storage, read_write> pc_dst: array<f32>;

// Mirrors twofield::JACOBI_OMEGA (see the symbol-bound derivation there: λ_max(diag⁻¹A) = 8/3
// for this corner family, so ω must be < 0.75; 2/3 is the weighted-Jacobi choice).
const JACOBI_OMEGA: f32 = 0.6666667;
// Density-relief time constant in frames (fixed structural constant, like JACOBI_OMEGA —
// mirrors twofield::DENSITY_RELAX_FRAMES, derivation there: equilibrium compaction drift
// ≈ residual·N·dt ≈ 1% — inside the ±5% band — while keeping the relief of a seeded
// over-density gentle, v ~ Δx/τ, instead of detonating it into slosh).
const DENSITY_RELAX_FRAMES: f32 = 30.0;
// Free-surface fill-fraction constants (header: FREE SURFACE; mirrored in twofield/mod.rs).
const SURF_FULL_FRAC: f32 = 0.5;
const SURF_MIN_CORNER: f32 = 0.1;
const NODE_BC_EPS: f32 = 1.0e-4;     // box-face epsilon (matches grid_update)
const SDF_NORMAL_MIN: f32 = 1.0e-3;  // degenerate orthogonalized-normal guard

// --- fine-grid cell helpers --------------------------------------------------------------------
fn fine_cells() -> vec3<u32> {
    return params.grid_dims.xyz - vec3<u32>(1u);
}
fn num_fine_cells() -> u32 {
    let c = fine_cells();
    return c.x * c.y * c.z;
}
fn cell_index(c: vec3<i32>) -> u32 {
    let nc = fine_cells();
    return u32(c.x) + nc.x * (u32(c.y) + nc.y * u32(c.z));
}
fn cell_coords(flat: u32) -> vec3<u32> {
    let nc = fine_cells();
    return vec3<u32>(flat % nc.x, (flat / nc.x) % nc.y, flat / (nc.x * nc.y));
}
fn cell_in_range(c: vec3<i32>) -> bool {
    let nc = vec3<i32>(fine_cells());
    return all(c >= vec3<i32>(0)) && all(c < nc);
}

// --- coarse-grid helpers (same layout at H = ratio·h, same origin) ------------------------------
fn coarse_cells() -> vec3<u32> {
    return params.coarse_dims.xyz;
}
fn num_coarse_cells() -> u32 {
    let c = coarse_cells();
    return c.x * c.y * c.z;
}
fn coarse_nodes() -> vec3<u32> {
    return params.coarse_dims.xyz + vec3<u32>(1u);
}
fn num_coarse_nodes() -> u32 {
    let n = coarse_nodes();
    return n.x * n.y * n.z;
}
fn ccell_index(c: vec3<i32>) -> u32 {
    let nc = coarse_cells();
    return u32(c.x) + nc.x * (u32(c.y) + nc.y * u32(c.z));
}
fn ccell_coords(flat: u32) -> vec3<u32> {
    let nc = coarse_cells();
    return vec3<u32>(flat % nc.x, (flat / nc.x) % nc.y, flat / (nc.x * nc.y));
}
fn ccell_in_range(c: vec3<i32>) -> bool {
    let nc = vec3<i32>(coarse_cells());
    return all(c >= vec3<i32>(0)) && all(c < nc);
}
fn cnode_index(n: vec3<u32>) -> u32 {
    let nn = coarse_nodes();
    return n.x + nn.x * (n.y + nn.y * n.z);
}

// --- M̃⁻¹ load/apply ----------------------------------------------------------------------------
struct Minv { a: vec4<f32>, b: vec4<f32> }

fn minv_apply(m: Minv, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        m.a.x * v.x + m.a.y * v.y + m.a.z * v.z,
        m.a.y * v.x + m.a.w * v.y + m.b.x * v.z,
        m.a.z * v.x + m.b.x * v.y + m.b.y * v.z,
    );
}

// Corner sign vector for offset o ∈ {0,1}³ (the ONE sign function D, G, and diag(A) share).
fn corner_sign(o: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(o) * 2.0 - vec3<f32>(1.0);
}

// Per-axis hat weight between fine cell fi and coarse cell ci at ratio r — the single weight
// function R and P share (transpose pair by construction; mirrored by the CPU twin `hat`).
fn hat_w(fi: i32, ci: i32, r: i32) -> f32 {
    let x = f32(fi) + 0.5;
    let xc = f32(ci * r) + f32(r) * 0.5;
    return max(0.0, 1.0 - abs(x - xc) / f32(r));
}

// =================================== node_setup ================================================
// Build M̃⁻¹ per fine node from this frame's node mass (grid_vel.w) and the wall geometry.
@compute @workgroup_size(256)
fn node_setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    let mass = grid_vel[n].w;
    let h = params.grid_origin.w;
    let c = node_coords(n);
    let xp = params.grid_origin.xyz + vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) * h;
    // Strictly-outside-the-box nodes are inside the walls: v = 0 in grid_update, M̃⁻¹ = 0
    // here (the projection never moves them) — one consistent treatment across the family.
    let outside = any(xp < params.box_min.xyz - vec3<f32>(NODE_BC_EPS))
        || any(xp > params.box_max.xyz + vec3<f32>(NODE_BC_EPS));
    if (mass > params.extra.z && !outside) {
        let rho = mass / (h * h * h);

        // Wall-truncated control volume: count the node's 8 adjacent cell slots whose center
        // is inside the box and not inside a solid. A wall/edge/corner node sees only
        // vis/8 of its slots, so BOTH its gathered gradient (G sums those slots) AND its
        // measured density are truncated by the same factor — flooring the density at
        // vis·ρ_rest/8 instead of ρ_rest keeps invρ·(G p) equal to the full physical
        // gradient there. With the global ρ_rest floor, box-EDGE nodes (2 of 8 slots) got a
        // 4×-truncated kick, their columns kept ~half of gravity every frame, and the tank
        // pumped itself into sloshing (observed). Surface nodes are NOT rescaled: air slots
        // are geometrically visible, so vis = 8 and the stable under-kick floor remains.
        var vis = 0.0;
        for (var oz = 0; oz < 2; oz = oz + 1) {
            for (var oy = 0; oy < 2; oy = oy + 1) {
                for (var ox = 0; ox < 2; ox = ox + 1) {
                    let cs = corner_sign(vec3<i32>(ox, oy, oz));
                    let cc = xp + cs * (0.5 * h);
                    var ok = all(cc >= params.box_min.xyz) && all(cc <= params.box_max.xyz);
                    if (ok && params.num_solids > 0u) {
                        ok = solid_union(cc, PHASE_WATER).dist >= 0.0;
                    }
                    if (ok) {
                        vis = vis + 1.0;
                    }
                }
            }
        }
        let invr = 1.0 / max(rho, params.extra.y * max(vis, 1.0) / 8.0);

        // Box faces: mutually orthogonal axis constraints (exact projector).
        var d = vec3<f32>(invr);
        if (xp.x <= params.box_min.x + NODE_BC_EPS || xp.x >= params.box_max.x - NODE_BC_EPS) { d.x = 0.0; }
        if (xp.y <= params.box_min.y + NODE_BC_EPS || xp.y >= params.box_max.y - NODE_BC_EPS) { d.y = 0.0; }
        if (xp.z <= params.box_min.z + NODE_BC_EPS || xp.z >= params.box_max.z - NODE_BC_EPS) { d.z = 0.0; }
        a = vec4<f32>(d.x, 0.0, 0.0, d.y);
        b = vec4<f32>(0.0, d.z, 1.0, 0.0);

        // SDF wall: subtract the dyad of the (box-orthogonalized) unit normal — symmetric PSD.
        if (params.num_solids > 0u) {
            let hit = solid_union(xp, PHASE_WATER);
            if (hit.dist < 0.0) {
                var nrm = hit.grad;
                if (d.x == 0.0) { nrm.x = 0.0; }
                if (d.y == 0.0) { nrm.y = 0.0; }
                if (d.z == 0.0) { nrm.z = 0.0; }
                let len = length(nrm);
                if (len > SDF_NORMAL_MIN) {
                    nrm = nrm / len;
                    let invc = max(max(d.x, d.y), d.z);
                    a -= invc * vec4<f32>(nrm.x * nrm.x, nrm.x * nrm.y, nrm.x * nrm.z, nrm.y * nrm.y);
                    b -= invc * vec4<f32>(nrm.y * nrm.z, nrm.z * nrm.z, 0.0, 0.0);
                }
            }
        }
    }
    nm[2u * n + 0u] = a;
    nm[2u * n + 1u] = b;
}

// =================================== cell_classify =============================================
// Per fine cell: fill fraction f from grid mass (header: FREE SURFACE — f = 0 masks the cell,
// f < 1 tapers its row through the surface band), rhs = f·(s_target − D v)/dt on fluid rows,
// and zero both pressure ping-pong slots (the deterministic p₀ = 0 — the coarse prolongation
// overwrites the seed when enabled).
@compute @workgroup_size(256)
fn cell_classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_fine_cells()) {
        return;
    }
    pf_src[c] = 0.0;
    pf_dst[c] = 0.0;
    let h = params.grid_origin.w;
    let cc = vec3<i32>(cell_coords(c));
    var div = 0.0;
    var mass = 0.0;
    var cnt = 0.0;
    var mmin = 3.402823e38; // f32 max
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let node = cc + vec3<i32>(ox, oy, oz);
                let gv = grid_vel[node_index(node)];
                let s = corner_sign(vec3<i32>(ox, oy, oz));
                div = div + dot(s, gv.xyz) / (4.0 * h);
                // Corner census for classification: only corners that COULD see fluid count —
                // nodes at/behind the box faces or inside an SDF solid are geometrically
                // truncated (a wall, not a surface) and must not deactivate wall-hugging
                // cells (observed: the min-corner rule silently dropped the box-edge cell
                // columns, whose particles then free-fell, pumping the tank).
                let xp = params.grid_origin.xyz
                    + vec3<f32>(f32(node.x), f32(node.y), f32(node.z)) * h;
                var wallish = any(xp <= params.box_min.xyz + vec3<f32>(NODE_BC_EPS))
                    || any(xp >= params.box_max.xyz - vec3<f32>(NODE_BC_EPS));
                if (!wallish && params.num_solids > 0u) {
                    wallish = solid_union(xp, PHASE_WATER).dist < 0.0;
                }
                if (!wallish) {
                    mass = mass + gv.w;
                    cnt = cnt + 1.0;
                    mmin = min(mmin, gv.w);
                }
            }
        }
    }
    // Fill fraction from the mean fluid-visible corner density; ACTIVITY requires BOTH the
    // min fluid-visible corner density (header: FREE SURFACE) AND particle presence in the
    // cell (cell_cnt — the U4 sharp discriminator; see common.wgsl for the frozen-chimney
    // failure the mass tests alone cannot resolve). Sub-eps nodes were mass-gated to v = 0
    // by grid_update, so D reads 0 from them — consistent with M̃⁻¹ = 0 there.
    let h3 = h * h * h;
    let rho = mass / (max(cnt, 1.0) * h3);
    var f = clamp(rho / (SURF_FULL_FRAC * params.extra.x), 0.0, 1.0);
    if (cnt == 0.0 || mmin < SURF_MIN_CORNER * params.extra.x * h3
        || atomicLoad(&cell_cnt[c]) == 0u) {
        f = 0.0;
    }
    let center = params.grid_origin.xyz + (vec3<f32>(cc) + vec3<f32>(0.5)) * h;
    var geom_blocked = false;
    if (any(center < params.box_min.xyz) || any(center > params.box_max.xyz)) {
        f = 0.0;
        geom_blocked = true;
    }
    if (!geom_blocked && params.num_solids > 0u) {
        let hit = solid_union(center, PHASE_WATER);
        if (hit.dist < 0.0) {
            f = 0.0;
            geom_blocked = true;
        }
    }
    // U4 category lane (.w): in-box fluid-free cells are AIR — flood-fill candidates for the
    // pocket detection (surface.wgsl); wall/out-of-box cells are never air. pocket_mark
    // upgrades enclosed air to CELL_POCKET (fill weight 1, constraint row).
    var cat = 0.0;
    if (f <= 0.0 && !geom_blocked) {
        cat = CELL_AIR;
    }
    // Density relief, over-density half (header: "Density relief"): fluid-visible corner
    // density vs rest. The UNDER-density half (suction) is applied by pocket_mark in
    // surface.wgsl — it needs the air classification of the 6-neighborhood, which does not
    // exist yet in this pass (suction must never act on air-adjacent surface cells, or the
    // surface band pumps itself upward; an interior rarefied region, e.g. the channel a jet
    // tears open, MUST be re-compacted or it stands forever as a frozen void — observed).
    // The mean corner density rides along in .z for that pass (the residual lane, free until
    // `residual` runs).
    let s_target = max(rho / params.extra.x - 1.0, 0.0) / (DENSITY_RELAX_FRAMES * params.dt);
    cell_meta[c] = vec4<f32>(select(0.0, f * (s_target - div) / params.dt, f > 0.0), f, rho, cat);
}

// =================================== coarse setup ==============================================
// Coarse M̃⁻¹ by hat-weighted RESTRICTION of the fine M̃⁻¹ field over the coarse node's
// support (per-axis weight 1 − |Δ|/r, Δ in fine nodes; convex combination of symmetric PSD
// matrices, so the result stays symmetric PSD). NOT point injection from the coincident fine
// node: a coarse surface cell can land its injected corner on a single massless/degenerate
// fine sample, making the coarse row near-singular — observed as intermittent catastrophic
// solve spikes (p ~ −10⁴, nearly constant in depth: a near-nullspace component) whenever the
// moving surface crossed a coarse corner. The restriction is the faithful rediscretization
// and is mirrored by the CPU twin (`coarsen`).
@compute @workgroup_size(256)
fn coarse_node_setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= num_coarse_nodes()) {
        return;
    }
    let nn = coarse_nodes();
    let i = vec3<u32>(n % nn.x, (n / nn.x) % nn.y, n / (nn.x * nn.y));
    let r = i32(params.coarse_dims.w);
    let f0 = vec3<i32>(i) * r;
    let nd = vec3<i32>(params.grid_dims.xyz);
    var wsum = 0.0;
    var sa = vec4<f32>(0.0);
    var sb = vec4<f32>(0.0);
    for (var dz = -(r - 1); dz <= r - 1; dz = dz + 1) {
        for (var dy = -(r - 1); dy <= r - 1; dy = dy + 1) {
            for (var dx = -(r - 1); dx <= r - 1; dx = dx + 1) {
                let fnode = f0 + vec3<i32>(dx, dy, dz);
                if (any(fnode < vec3<i32>(0)) || any(fnode >= nd)) {
                    continue;
                }
                let w = (1.0 - abs(f32(dx)) / f32(r))
                    * (1.0 - abs(f32(dy)) / f32(r))
                    * (1.0 - abs(f32(dz)) / f32(r));
                let fidx = node_index(fnode);
                sa = sa + w * nm[2u * fidx + 0u];
                sb = sb + w * nm[2u * fidx + 1u];
                wsum = wsum + w;
            }
        }
    }
    let inv = 1.0 / max(wsum, 1.0e-20);
    nm_c[2u * n + 0u] = sa * inv;
    nm_c[2u * n + 1u] = sb * inv;
}

// Coarse cell mask: SATURATED fractions (binary — active iff any active child). The fine
// taper is deliberately NOT rediscretized to the coarse level: under A = F·L·F the exact
// pressure at an f ≪ 1 row is legitimately 1/f-amplified (it is level- and cell-LOCAL; only
// f·p is physical), so prolongating a tapered-coarse-row correction onto neighboring f = 1
// fine cells injects 1/f_C-amplified garbage into the bulk (observed: ~1e5-scale corrections
// from f_C ~ 0.003 coarse cells erupting the settled tank within frames). The taper band is
// thinner than a coarse cell anyway — surface rows are smoothed at fine level only. Also
// restricts the rhs (R = masked trilinear transpose with the fixed 1/ratio³ volume weight)
// and zeroes the coarse pressure ping-pong.
@compute @workgroup_size(256)
fn coarse_cell_setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_coarse_cells()) {
        return;
    }
    pc_src[c] = 0.0;
    pc_dst[c] = 0.0;
    let cc = vec3<i32>(ccell_coords(c));
    let r = i32(params.coarse_dims.w);

    var fc = 0.0;
    var has_fluid = false;
    var has_pocket = false;
    for (var dz = 0; dz < r; dz = dz + 1) {
        for (var dy = 0; dy < r; dy = dy + 1) {
            for (var dx = 0; dx < r; dx = dx + 1) {
                let f = cc * r + vec3<i32>(dx, dy, dz);
                if (cell_in_range(f)) {
                    let cm = cell_meta[cell_index(f)];
                    if (cm.y > 0.0) {
                        fc = 1.0;
                        if (cm.w == CELL_POCKET) {
                            has_pocket = true;
                        } else {
                            has_fluid = true;
                        }
                    }
                }
            }
        }
    }
    // U4 pocket-coarse flag: EVERY active child a pocket cell (mixed boundary cells stay
    // fluid rows — the same rediscretization doctrine as the un-rediscretized surface taper;
    // the restricted residual carries the boundary content and the fine sweeps smooth it).
    var cw = 0.0;
    if (has_pocket && !has_fluid) {
        cw = CELL_POCKET;
    }

    var sum = 0.0;
    if (fc > 0.0) {
        let lo = cc * r - vec3<i32>(r / 2);
        for (var dz = 0; dz < 2 * r; dz = dz + 1) {
            for (var dy = 0; dy < 2 * r; dy = dy + 1) {
                for (var dx = 0; dx < 2 * r; dx = dx + 1) {
                    let f = lo + vec3<i32>(dx, dy, dz);
                    if (!cell_in_range(f)) {
                        continue;
                    }
                    let cm = cell_meta[cell_index(f)];
                    if (cm.y <= 0.0) {
                        continue;
                    }
                    let w = hat_w(f.x, cc.x, r) * hat_w(f.y, cc.y, r) * hat_w(f.z, cc.z, r);
                    sum = sum + w * cm.z; // .z = the residual written by `residual`
                }
            }
        }
        sum = sum / f32(r * r * r);
    }
    cmeta[c] = vec4<f32>(sum, fc, 0.0, cw);
}

// =================================== residual ==================================================
// r = rhs − A·p into cell_meta.z (the field the coarse stage restricts). Same composition
// loops as jacobi_fine, no update — the restricted field and the smoother share one operator.
@compute @workgroup_size(256)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_fine_cells()) {
        return;
    }
    var cm = cell_meta[c];
    var r = 0.0;
    if (cm.y > 0.0) {
        let h = params.grid_origin.w;
        let cc = vec3<i32>(cell_coords(c));
        var acc = 0.0;
        for (var oz = 0; oz < 2; oz = oz + 1) {
            for (var oy = 0; oy < 2; oy = oy + 1) {
                for (var ox = 0; ox < 2; ox = ox + 1) {
                    let o = vec3<i32>(ox, oy, oz);
                    let node = cc + o;
                    let nidx = node_index(node);
                    let m = Minv(nm[2u * nidx + 0u], nm[2u * nidx + 1u]);
                    var gp = vec3<f32>(0.0);
                    for (var qz = 0; qz < 2; qz = qz + 1) {
                        for (var qy = 0; qy < 2; qy = qy + 1) {
                            for (var qx = 0; qx < 2; qx = qx + 1) {
                                let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                                if (!cell_in_range(c2)) {
                                    continue;
                                }
                                let ci2 = cell_index(c2);
                                let f2 = cell_meta[ci2].y;
                                if (f2 <= 0.0) {
                                    continue;
                                }
                                let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                                gp = gp - s2 * f2 * pf_src[ci2] / (4.0 * h);
                            }
                        }
                    }
                    let s = corner_sign(o);
                    acc = acc + dot(s, minv_apply(m, gp)) / (4.0 * h);
                }
            }
        }
        r = cm.x - (-acc * cm.y);
    }
    cm.z = r;
    cell_meta[c] = cm;
}

// =================================== Jacobi sweeps =============================================
// One damped Jacobi sweep on the FINE grid. A p is evaluated as the literal composition
// D(M̃⁻¹(G p)): for each corner node, gather (G p) from its 8 adjacent active cells, apply
// M̃⁻¹, and accumulate the divergence sign — never a pre-derived stencil.
@compute @workgroup_size(256)
fn jacobi_fine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_fine_cells()) {
        return;
    }
    let cm = cell_meta[c];
    // U4 pocket pin: the pocket slot is a Dirichlet copy of the bubble multiplier λ_b — the
    // row is relaxed as ONE aggregate by bubble_fine (surface.wgsl), never per cell, so the
    // fine sweeps cannot relax the constraint away (the named KTD-6 failure).
    if (cm.w == CELL_POCKET) {
        pf_dst[c] = bubble[0];
        return;
    }
    if (cm.y <= 0.0) {
        pf_dst[c] = 0.0;
        return;
    }
    let h = params.grid_origin.w;
    let cc = vec3<i32>(cell_coords(c));
    var acc = 0.0;  // (D M̃⁻¹ G p)_c (row fraction applied after the loops)
    var diag = 0.0; // A_cc / f_c²
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let o = vec3<i32>(ox, oy, oz);
                let node = cc + o;
                let nidx = node_index(node);
                let m = Minv(nm[2u * nidx + 0u], nm[2u * nidx + 1u]);
                // (G p) at this node: gather the 8 adjacent cells (fraction-weighted).
                var gp = vec3<f32>(0.0);
                for (var qz = 0; qz < 2; qz = qz + 1) {
                    for (var qy = 0; qy < 2; qy = qy + 1) {
                        for (var qx = 0; qx < 2; qx = qx + 1) {
                            let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                            if (!cell_in_range(c2)) {
                                continue;
                            }
                            let ci2 = cell_index(c2);
                            let f2 = cell_meta[ci2].y;
                            if (f2 <= 0.0) {
                                continue;
                            }
                            // node offset within c2 = 1 − q on each axis.
                            let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                            gp = gp - s2 * f2 * pf_src[ci2] / (4.0 * h);
                        }
                    }
                }
                let s = corner_sign(o);
                acc = acc + dot(s, minv_apply(m, gp)) / (4.0 * h);
                diag = diag + dot(s, minv_apply(m, s)) / (16.0 * h * h);
            }
        }
    }
    let ap = -acc * cm.y;
    diag = diag * cm.y * cm.y;
    if (diag > 1.0e-20) {
        pf_dst[c] = pf_src[c] + JACOBI_OMEGA * (cm.x - ap) / diag;
    } else {
        pf_dst[c] = 0.0;
    }
}

// The identical sweep on the COARSE grid (same family at H = ratio·h; rediscretized, see the
// header). Kept line-for-line parallel to jacobi_fine — both are pinned by the same CPU twin.
@compute @workgroup_size(256)
fn jacobi_coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_coarse_cells()) {
        return;
    }
    let cm = cmeta[c];
    // U4 pocket pin at the coarse level: the identical representation (KTD-6) — pocket-coarse
    // slots carry the shared correction δλ_b, relaxed as one aggregate by bubble_coarse.
    if (cm.w == CELL_POCKET) {
        pc_dst[c] = bubble[1];
        return;
    }
    if (cm.y <= 0.0) {
        pc_dst[c] = 0.0;
        return;
    }
    let hc = params.grid_origin.w * f32(params.coarse_dims.w);
    let cc = vec3<i32>(ccell_coords(c));
    var acc = 0.0;
    var diag = 0.0;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let o = vec3<i32>(ox, oy, oz);
                let node = cc + o;
                let nidx = cnode_index(vec3<u32>(node));
                let m = Minv(nm_c[2u * nidx + 0u], nm_c[2u * nidx + 1u]);
                var gp = vec3<f32>(0.0);
                for (var qz = 0; qz < 2; qz = qz + 1) {
                    for (var qy = 0; qy < 2; qy = qy + 1) {
                        for (var qx = 0; qx < 2; qx = qx + 1) {
                            let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                            if (!ccell_in_range(c2)) {
                                continue;
                            }
                            let ci2 = ccell_index(c2);
                            let f2 = cmeta[ci2].y;
                            if (f2 <= 0.0) {
                                continue;
                            }
                            let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                            gp = gp - s2 * f2 * pc_src[ci2] / (4.0 * hc);
                        }
                    }
                }
                let s = corner_sign(o);
                acc = acc + dot(s, minv_apply(m, gp)) / (4.0 * hc);
                diag = diag + dot(s, minv_apply(m, s)) / (16.0 * hc * hc);
            }
        }
    }
    let ap = -acc * cm.y;
    diag = diag * cm.y * cm.y;
    if (diag > 1.0e-20) {
        pc_dst[c] = pc_src[c] + JACOBI_OMEGA * (cm.x - ap) / diag;
    } else {
        pc_dst[c] = 0.0;
    }
}

// =================================== prolong_add ===============================================
// P: masked trilinear prolongation of the coarse CORRECTION, added to the current pressure
// (reads binding 11, writes binding 12 — parity-flips like a sweep). Transpose pair of the
// restriction by shared hat_w.
@compute @workgroup_size(256)
fn prolong_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_fine_cells()) {
        return;
    }
    if (cell_meta[c].y <= 0.0) {
        pf_dst[c] = 0.0;
        return;
    }
    // U4 pocket cells receive the SCALAR correction δλ_b exactly (a one-scalar prolongation
    // is exact; hat-weighting neighboring fluid-coarse corrections onto a pinned slot would
    // corrupt the multiplier). bubble_fine folds λ_b ← λ_b + δλ_b before the next solve.
    if (cell_meta[c].w == CELL_POCKET) {
        pf_dst[c] = pf_src[c] + bubble[1];
        return;
    }
    let cc = vec3<i32>(cell_coords(c));
    let r = i32(params.coarse_dims.w);
    var base: vec3<i32>;
    base.x = i32(floor((f32(cc.x) + 0.5 - f32(r) * 0.5) / f32(r)));
    base.y = i32(floor((f32(cc.y) + 0.5 - f32(r) * 0.5) / f32(r)));
    base.z = i32(floor((f32(cc.z) + 0.5 - f32(r) * 0.5) / f32(r)));
    var sum = 0.0;
    for (var dz = 0; dz < 2; dz = dz + 1) {
        for (var dy = 0; dy < 2; dy = dy + 1) {
            for (var dx = 0; dx < 2; dx = dx + 1) {
                let ci = base + vec3<i32>(dx, dy, dz);
                if (!ccell_in_range(ci)) {
                    continue;
                }
                let cidx = ccell_index(ci);
                if (cmeta[cidx].y <= 0.0) {
                    continue;
                }
                let w = hat_w(cc.x, ci.x, r) * hat_w(cc.y, ci.y, r) * hat_w(cc.z, ci.z, r);
                sum = sum + w * pc_src[cidx];
            }
        }
    }
    pf_dst[c] = pf_src[c] + sum;
}

// =================================== project ===================================================
// v ← v − dt·M̃⁻¹·(G p), the SAME G and M̃⁻¹ the solve used (binding 11 holds the final-parity
// pressure). Constrained wall components receive nothing by construction of M̃⁻¹.
@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let m = Minv(nm[2u * n + 0u], nm[2u * n + 1u]);
    if (m.b.z < 0.5) {
        return; // massless node — the projection never moves it
    }
    let h = params.grid_origin.w;
    let node = vec3<i32>(node_coords(n));
    var gp = vec3<f32>(0.0);
    for (var qz = 0; qz < 2; qz = qz + 1) {
        for (var qy = 0; qy < 2; qy = qy + 1) {
            for (var qx = 0; qx < 2; qx = qx + 1) {
                let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                if (!cell_in_range(c2)) {
                    continue;
                }
                let ci2 = cell_index(c2);
                let f2 = cell_meta[ci2].y;
                if (f2 <= 0.0) {
                    continue;
                }
                let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                gp = gp - s2 * f2 * pf_src[ci2] / (4.0 * h);
            }
        }
    }
    let dv = minv_apply(m, gp) * params.dt;
    grid_vel[n] = vec4<f32>(grid_vel[n].xyz - dv, grid_vel[n].w);
}

// =================================== test-only debug taps ======================================
// Never dispatched in step(); they expose the EXACT D and G kernels to the operator gates
// (adjointness + MMS on GPU readbacks). dbg_div writes the raw masked divergence of grid_vel
// into cell_meta.z; dbg_grad writes the raw masked gradient of pf (binding 11) into grid_vel.
@compute @workgroup_size(256)
fn dbg_div(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_fine_cells()) {
        return;
    }
    var cm = cell_meta[c];
    var div = 0.0;
    if (cm.y > 0.0) {
        let h = params.grid_origin.w;
        let cc = vec3<i32>(cell_coords(c));
        for (var oz = 0; oz < 2; oz = oz + 1) {
            for (var oy = 0; oy < 2; oy = oy + 1) {
                for (var ox = 0; ox < 2; ox = ox + 1) {
                    let s = corner_sign(vec3<i32>(ox, oy, oz));
                    let gv = grid_vel[node_index(cc + vec3<i32>(ox, oy, oz))];
                    div = div + dot(s, gv.xyz) / (4.0 * h);
                }
            }
        }
        div = div * cm.y; // D_f row fraction
    }
    cm.z = div;
    cell_meta[c] = cm;
}

@compute @workgroup_size(256)
fn dbg_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let h = params.grid_origin.w;
    let node = vec3<i32>(node_coords(n));
    var gp = vec3<f32>(0.0);
    for (var qz = 0; qz < 2; qz = qz + 1) {
        for (var qy = 0; qy < 2; qy = qy + 1) {
            for (var qx = 0; qx < 2; qx = qx + 1) {
                let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                if (!cell_in_range(c2)) {
                    continue;
                }
                let ci2 = cell_index(c2);
                let f2 = cell_meta[ci2].y;
                if (f2 <= 0.0) {
                    continue;
                }
                let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                gp = gp - s2 * f2 * pf_src[ci2] / (4.0 * h);
            }
        }
    }
    grid_vel[n] = vec4<f32>(gp, 0.0);
}

// U6 two-field coupling, rigid skeleton (plan 2026-06-09-001 U6 / KTD-3 / KTD-4 / KTD-8):
// thin solid-mass P2G + the Laibe-Price exponential drag fold against the kinematically
// frozen solid field (v_s = 0). Gates: tests/twofield_coupling.rs.
//
// ================================ φ FIELDS (KTD-8) ============================================
// φ_s per node = (Σ_grains V_g·w) / h³ from the quadratic-B-spline scatter of the grain
// sphere volume V_g = π/6·d³ (params.coupling.z), clamped at PHI_S_MAX = 0.95 (mirrors the
// xpbd packing clamp: φ_f never below 0.05, so the mixture operator stays nonsingular).
// φ_f = 1 − φ_s. The field is local and kernel-smoothed — a configured global porosity
// scalar is the rejected anti-pattern.
//
// ================================ DRAG FOLD (KTD-3, exact form) ===============================
// The drag rate β(φ) comes from Kozeny-Carman (mirrors models::permeability — same clamps,
// same 150 coefficient, β = drag_scale/k), blended toward a Wen-Yu-style dilute rate below
// φ_s ≈ 0.2 with the Huilin-Gidaspow smooth arctan transition (constants below). The fold is
// the FORCED exponential integrator for dv/dt = −β·(v − v_s) + g with v_s ≡ 0:
//     v⁺ = v·e^{−βΔt} + g·srcf,        srcf = (1 − e^{−βΔt})/β   (→ Δt as β → 0),
// exact for frozen-coefficient linear drag at ANY stiffness — including its forcing. The
// plain "fold the gravity kick" form v⁺ = (v + gΔt)·e is NOT exact for the forced system:
// at equilibrium it eats (1 − e) of the hydrostatic load into spurious drag reaction, the
// pore-pressure slope collapses to ρg·(1−e)/(βΔt), and the pre-registered buoyancy/undrained
// gates fail at stiff β (measured: the recorded buoyant reaction flips SIGN at β·Δt ≈ 1.9).
//
// The same integrator weight must therefore apply to the pressure force the projection adds
// AFTER this pass: the per-node mobility used by the projection family is scaled by
//     ς = srcf/Δt ∈ (0, 1]   (react.w, folded into M̃⁻¹ by node_setup),
// so the water's velocity correction is Δv = −(Δt_eff/ρ_w)·∇p with Δt_eff = ς·Δt — the
// KTD-4 form with the exponential-integrator step length; φ NEVER scales the correction.
// Steady percolation then satisfies v·(1−e) = srcf·(g − ∇p/ρ) ⇒ v = (g − ∇p/ρ)/β: the
// discrete steady state IS the continuous Darcy balance, at any β·Δt. This is the per-node
// partial-elimination structure production Euler-Euler codes use for stiff interphase drag
// (Fluent PC-SIMPLE / PEA; MFiX-Exa).
//
// REACTION LEDGER: the frozen skeleton has no dynamics, so every impulse it absorbs is
// recorded (never discarded): this pass records the pair-update impulse m·(v_naive − v⁺)
// (v_naive = v + gΔt, the no-drag path); `project` adds the pressure share (the direct
// −φ_s·∇p force on the solid volume plus the (1−ς) fraction of the water-column pressure
// force transmitted through the drag during the fold). At hydrostatic equilibrium the three
// terms net to exactly the displaced-volume weight — the buoyancy gate.
//
// EXACT-ZERO PASSTHROUGH (free-water regression gate): every coupling kernel gates on the
// solid-mass census (decoded grid_sfp = 0), not on β → 0 — zero-solid nodes take the
// byte-identical pre-U6 arithmetic path (no exp(0) rounding in the loop).
//
// This pass also owns the grid forces + grid-node boundary conditions (moved verbatim from
// U2's grid_update, which now only decodes): the fold must integrate the gravity source
// itself, and the BCs must come after all velocity updates. Pipeline order (the U6 HTD):
//   grid_clear → p2g_water → p2g_solid → grid_update(decode) → drag_fold(forces+drag+BC)
//   → node_setup → … pressure … → project → g2p_water.
//
// OPEN BASE (params.coupling.w, dev/test hook for the U6 drained-flux gates; U9 replaces it
// with the phase-selective filter boundary): the y-min face stops being a wall for the water
// field — face nodes keep v_y, below-floor pad nodes stay live, node_setup leaves the y axis
// unconstrained there, escaped particles fall ballistically and stop scattering (p2g/g2p
// guards). The pad-cell row below the floor is geometry-masked (f = 0), so the pressure
// family sees the implicit p = 0 outflow Dirichlet — head-driven Darcy outflow, no penalty.

// --- drag-law constants (KTD-3; CPU twin: twofield::blended_drag_rate) ------------------------
// Mirrors models::permeability::DARCY_VISCOUS_COEFF.
const KC_VISCOUS_COEFF: f32 = 150.0;
// Wen-Yu Stokes-limit coefficient (18 = the classic Stokes per-particle drag aggregated to a
// volumetric rate; the φ_f^−1.65 voidage correction is omitted — the gates exercise the
// packed regime φ_s > 0.35 and the exact-zero free-water limit, and the Huilin blend smooths
// the ≤2.3× mismatch at the 0.2 switch).
const WEN_YU_COEFF: f32 = 18.0;
// Gidaspow switch + Huilin-Gidaspow arctan slope (262.5 = 150·1.75, the literature constant).
const PHI_S_BLEND: f32 = 0.2;
const HUILIN_SLOPE: f32 = 262.5;
const PI_F32: f32 = 3.14159265;
// φ_s clamp = 1 − the φ_f floor 0.05 (mirrors the xpbd porosity_drag_factor clamp).
const PHI_S_MAX: f32 = 0.95;

// Mirrors models::permeability::kozeny_carman (same clamps).
fn kozeny_carman(d: f32, phi: f32) -> f32 {
    let p = clamp(phi, 1.0e-4, 0.999);
    let solid = max(1.0 - p, 1.0e-4);
    return d * d * p * p * p / (KC_VISCOUS_COEFF * solid * solid);
}

// Blended drag rate β(φ) in 1/s (mirrors models::permeability::drag_rate for the packed arm;
// CPU twin twofield::blended_drag_rate is pinned against this by the pair-momentum gate).
fn drag_rate_blended(phi_s: f32, phi_f: f32) -> f32 {
    let d = params.coupling.x;
    let scale = params.coupling.y;
    let packed = scale / max(kozeny_carman(d, phi_f), 1.0e-9);
    let dilute = scale * WEN_YU_COEFF * phi_s / max(d * d, 1.0e-8);
    let psi = atan(HUILIN_SLOPE * (phi_s - PHI_S_BLEND)) / PI_F32 + 0.5;
    return psi * packed + (1.0 - psi) * dilute;
}

// Decoded solid volume at a node (the φ_s carrier).
fn solid_volume_at(n: u32) -> f32 {
    return fp_decode(atomicLoad(&grid_sfp[n]));
}

// Wall-truncated control volume of a node: how many of its 8 adjacent cell slots are inside
// the box and outside the SDF solids (the node_setup truncation rule, shared so φ_s and the
// density floor normalize identically).
fn node_vis(xp: vec3<f32>) -> f32 {
    var vis = 0.0;
    let h = params.grid_origin.w;
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
    return vis;
}

// φ_f at a node: the B-spline-scattered solid volume over the VISIBLE control volume
// (vis/8·h³). Without the truncation normalization, wall/floor nodes in a wall-flush bed
// read φ_s ≈ half the interior value, the drag there drops ~4× (Kozeny-Carman is quadratic
// in φ_s), and the column drains through an artificial low-drag wall annulus — measured 2–5×
// the Darcy prediction before this fix. Zero-solid nodes skip everything (exact-zero
// passthrough: φ_f = 1.0 bitwise, no vis loop cost in water-only scenes).
fn node_phi_f(n: u32, xp: vec3<f32>) -> f32 {
    let sv = solid_volume_at(n);
    if (sv <= 0.0) {
        return 1.0;
    }
    let h = params.grid_origin.w;
    let vis = max(node_vis(xp), 1.0);
    return 1.0 - min(sv / (h * h * h * vis / 8.0), PHI_S_MAX);
}

// =================================== p2g_solid =================================================
// Thin solid-mass P2G: scatter each (frozen) grain's sphere volume with the same quadratic
// B-spline weights as the water field, into the single-lane fixed-point solid field. No
// momentum lanes — the skeleton is kinematically frozen in U6 (v_s ≡ 0); the full solid
// field with stress arrives in U5/U7. Budget: one extra binding pair (grid_sfp) instead of
// four more atomic lanes in grid_fp — documented against the KTD-7 vec4-packing rule because
// mass-only needs one lane, not four.
@compute @workgroup_size(256)
fn p2g_solid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.solid_count) {
        return;
    }
    // Solid range sits AFTER the (pool-padded) water range — KTD-1 layout.
    let p = params.particle_count - params.solid_count + i;
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    // U9 swelling (KTD-8): the EFFECTIVE grain volume is the dry sphere volume π/6·d³ plus the
    // absorbed water volume V_abs (pos.w on grains — models::wetting effective_volume), so a
    // saturating grain raises the local φ_s and lowers K(φ) as it swells. With absorption off
    // V_abs ≡ 0 and this is bit-identical to the U6 constant params.coupling.z.
    let vg = params.coupling.z + max(pos[p].w, 0.0);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i2 = 0; i2 < 3; i2 = i2 + 1) {
                let wijk = w[i2].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i2, j, k);
                atomicAdd(&grid_sfp[node_index(node)], fp_encode(vg * wijk));
            }
        }
    }
}

// =================================== drag_fold =================================================
// Grid forces + the exponential drag fold + the grid-node boundary conditions (header).
@compute @workgroup_size(256)
fn drag_fold(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= params.grid_dims.w) {
        return;
    }
    let gv = grid_vel[n];
    let mass = gv.w;
    if (mass <= params.extra.z) {
        // Empty node (v already 0 from grid_update's mass gate): plain-Δt family, no impulse.
        react[n] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }
    var v = gv.xyz;
    var g = params.gravity.xyz;
    let c = node_coords(n);
    let xp = params.grid_origin.xyz + vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) * params.grid_origin.w;
    let sv = solid_volume_at(n);
    // U9 wetting-front capillary suction (KTD-4b, Green-Ampt body force): on water nodes INSIDE
    // the bed (sv > 0) that sit at the unsaturated front — water present here but the node one
    // cell BELOW carries less water mass (the dry advancing front) — add a downward acceleration
    // a_suction (mapped from ψ_f; see tests/twofield_infiltration.rs UNIT MAPPING). Folded into
    // the gravity source g so the exponential drag integrator carries it exactly (same as
    // gravity). a_suction = 0 (Config default) is exact passthrough (no front detection cost).
    let a_suction = params.wet1.x;
    if (a_suction > 0.0 && sv > 0.0 && mass > params.extra.z) {
        let h = params.grid_origin.w;
        let below = vec3<i32>(c) - vec3<i32>(0, 1, 0);
        var drier_below = true; // domain edge counts as "open below" (front at the base)
        if (below.y >= 0) {
            let mb = grid_vel[node_index(below)].w;
            drier_below = mb < mass * 0.5; // the cell below carries < half this node's water
        }
        if (drier_below) {
            g = g - vec3<f32>(0.0, a_suction, 0.0); // pull water down into the dry front
        }
    }
    var sig = 1.0;
    var imp = vec3<f32>(0.0);
    if (sv > 0.0) {
        let phi_f = node_phi_f(n, xp); // truncation-normalized (see node_phi_f)
        let phi_s = 1.0 - phi_f;
        let beta = drag_rate_blended(phi_s, phi_f);
        let a = beta * params.dt;
        let e = exp(-a);
        var srcf: f32;
        if (a < 1.0e-3) {
            srcf = params.dt * (1.0 - 0.5 * a); // series guard: (1−e)/β without cancellation
        } else {
            srcf = (1.0 - e) / beta;
        }
        let naive = v + g * params.dt; // the no-drag path (what the ledger is measured against)
        v = v * e + g * srcf;          // forced exponential integrator, v_s = 0
        sig = srcf / params.dt;
        imp = mass * (naive - v);      // pair-update impulse absorbed by the frozen skeleton
    } else {
        v = v + g * params.dt; // exact-zero passthrough: byte-identical pre-U6 gravity path
    }

    // --- grid-node BC (moved verbatim from U2's grid_update; see its original comments) -------
    let eps = 1.0e-4;
    let open_base = params.coupling.w > 0.5;

    // Nodes strictly OUTSIDE the domain box are inside the walls (zero velocity, M̃⁻¹ = 0 in
    // node_setup). The open base exempts the below-floor pad layer: it carries the outflow's
    // B-spline smear and must keep its velocity or near-floor particles gather a zero-drag
    // brake (the same edge-column artifact the original comment records).
    var out_lo = xp < params.box_min.xyz - vec3<f32>(eps);
    if (open_base) {
        out_lo.y = false;
    }
    if (any(out_lo) || any(xp > params.box_max.xyz + vec3<f32>(eps))) {
        grid_vel[n] = vec4<f32>(vec3<f32>(0.0), mass);
        react[n] = vec4<f32>(imp, sig);
        return;
    }
    // Domain-box faces: full normal component removed (free slip tangentially), matching the
    // constrained M̃⁻¹. The open base keeps the y-min face open for the water field.
    if (xp.x <= params.box_min.x + eps || xp.x >= params.box_max.x - eps) { v.x = 0.0; }
    if ((xp.y <= params.box_min.y + eps && !open_base) || xp.y >= params.box_max.y - eps) { v.y = 0.0; }
    if (xp.z <= params.box_min.z + eps || xp.z >= params.box_max.z - eps) { v.z = 0.0; }
    // Static SDF solids: remove the full normal component (mirrors utils/sdf.rs). Band the test
    // by one cell (WALL_BAND·h) on the FLUID side so the supporting layer of a non-grid-aligned
    // wall (e.g. the cup floor) is held — the SAME band node_setup uses to build the M̃⁻¹ wall
    // projector, so the pre-projection velocity field and the operator constrain the same axes
    // (operator consistency — a one-sided BC here keeps the supporting layer's inward velocity
    // that the symmetric M̃⁻¹ projector cannot correct, which both breaks A = D·M̃⁻¹·G AND, in the
    // V60 cup, changed the filling dynamics enough to trap a spurious crushing pocket; reverted).
    // The normal is orthogonalized against the already-zeroed box-face axes and renormalized,
    // exactly as in node_setup.
    if (params.num_solids > 0u) {
        let hit = solid_union(xp, PHASE_WATER);
        // SAME coverage weight as node_setup's M̃⁻¹ dyad (operator consistency). Binary mode
        // (default) returns 1.0 in-band → `v - 1.0·(v·n̂)n̂` is byte-identical to the old BC.
        let wc = wall_coverage(hit.dist, params.grid_origin.w);
        if (wc > 0.0) {
            var nrm = hit.grad;
            if (xp.x <= params.box_min.x + eps || xp.x >= params.box_max.x - eps) { nrm.x = 0.0; }
            if ((xp.y <= params.box_min.y + eps && !open_base) || xp.y >= params.box_max.y - eps) { nrm.y = 0.0; }
            if (xp.z <= params.box_min.z + eps || xp.z >= params.box_max.z - eps) { nrm.z = 0.0; }
            let len = length(nrm);
            if (len > SDF_NORMAL_MIN) {
                nrm = nrm / len;
                v = v - wc * dot(v, nrm) * nrm;
            }
        }
    }
    // Speed-cap backstop (couples to the FP headroom math in common.wgsl).
    let s = length(v);
    if (s > params.max_speed) {
        v = v * (params.max_speed / s);
    }
    grid_vel[n] = vec4<f32>(v, mass);
    react[n] = vec4<f32>(imp, sig);
}

// ============================ U9 MOISTURE PHASE-CHANGE ABSORPTION (KTD-4c) =====================
// GIC-style: water leaving the fluid phase becomes grain volume, conserved 1:1. Grid-projected,
// atomic-free in the gather pass, deterministic and EXACTLY conserving by construction.
//
// p2g_moisture scatters two fixed-point grid lanes with the SAME quadratic B-spline weights as
// the field transfers: the water SUPPLY S_n = Σ_w w·f_w·V_w and the grain DEMAND
// D_n = Σ_g w·demand_g, demand_g = (V_cap − V_abs)·(1 − e^{−k_abs·dt}) (mirrors
// models::wetting::absorb_demand, capped two-sidedly). The transfer at a node is the two-sided
// cap T_n = min(S_n, D_n). g2p_absorb makes:
//   * each WATER lose its weighted share  Σ_n w·(f_w·V_w)·(T_n/S_n),
//   * each GRAIN gain its weighted share  Σ_n w·(demand_g)·(T_n/D_n).
// Σ over particles at node n: water loses Σ_w w·f_w·V_w·(T_n/S_n) = S_n·(T_n/S_n) = T_n, and
// grains gain D_n·(T_n/D_n) = T_n — EXACTLY equal, per node and globally (the conservation
// gate). f_w / V_abs ride pos.w (U1 lane convention). Bloom (KTD-4b): a dry grain's demand is
// scaled by a contact-time ramp (chem.x accumulates wet contact; ramps 0→1 over bloom_delay) so
// the hydrophobic entry delay falls out. Off by default (k_abs = 0 ⇒ CPU-elided in mod.rs).

fn wet_v_cap() -> f32 { return params.wet0.y; }
fn wet_v_water() -> f32 { return params.wet0.z; }
fn wet_roundoff() -> f32 { return params.wet0.w; }
fn wet_sat_cutoff() -> f32 { let c = wet_v_cap(); return c - c * wet_roundoff(); }
fn wet_demand(v_abs: f32) -> f32 {
    if (v_abs >= wet_sat_cutoff()) { return 0.0; }
    return (wet_v_cap() - v_abs) * (1.0 - exp(-params.wet0.x * params.dt));
}
// Bloom factor for a grain (KTD-4b): a smooth ramp of its accumulated wet-contact time
// (chem.x, sim seconds) over bloom_delay. No delay ⇒ factor 1 immediately.
fn bloom_factor(contact_t: f32) -> f32 {
    let bd = params.wet1.y;
    if (bd <= 0.0) { return 1.0; }
    let t = clamp(contact_t / bd, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t); // smoothstep
}

// =================================== p2g_moisture =============================================
@compute @workgroup_size(256)
fn p2g_moisture(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_count) {
        return;
    }
    // Water range [0, water_count); solid range [particle_count − solid_count, particle_count).
    let is_water = p < params.water_count;
    let is_solid = p >= (params.particle_count - params.solid_count);
    if (!is_water && !is_solid) {
        return; // dormant pool slot — never scatters
    }
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    // Ballistic escaped particles (open base): don't scatter (mirror the field-transfer guard).
    if (params.coupling.w > 0.5 && x.y < params.box_min.y) {
        return;
    }
    var quantity = 0.0;
    var lane = 0u;
    if (is_water) {
        let f_w = pos[p].w;
        if (f_w <= wet_roundoff()) { return; } // drained water: nothing to supply
        quantity = f_w * wet_v_water();
        lane = 0u; // SUPPLY
    } else {
        let v_abs = pos[p].w;
        let demand = wet_demand(v_abs) * bloom_factor(chem[p].x);
        if (demand <= 0.0) { return; } // saturated / unbloomed grain: no demand
        quantity = demand;
        lane = 1u; // DEMAND
    }
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                atomicAdd(&grid_moist[node_index(node) * 2u + lane], fp_encode(quantity * wijk));
            }
        }
    }
}

// =================================== g2p_absorb ===============================================
// Gather the node drain/fill fractions and apply the conserved transfer (atomic-free: each
// particle writes only its own pos.w / chem.x). Water shrinks f_w; grain grows V_abs and ticks
// its wet-contact timer (the bloom clock).
@compute @workgroup_size(256)
fn g2p_absorb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_count) {
        return;
    }
    let is_water = p < params.water_count;
    let is_solid = p >= (params.particle_count - params.solid_count);
    if (!is_water && !is_solid) {
        return;
    }
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    if (params.coupling.w > 0.5 && x.y < params.box_min.y) {
        return; // ballistic escaped water: no transfer
    }
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);

    if (is_water) {
        let f_w = pos[p].w;
        if (f_w <= wet_roundoff()) { return; }
        let my_supply = f_w * wet_v_water();
        var loss = 0.0;
        for (var k = 0; k < 3; k = k + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                for (var i = 0; i < 3; i = i + 1) {
                    let wijk = w[i].x * w[j].y * w[k].z;
                    let n = node_index(base + vec3<i32>(i, j, k));
                    let s_n = fp_decode(atomicLoad(&grid_moist[n * 2u + 0u]));
                    let d_n = fp_decode(atomicLoad(&grid_moist[n * 2u + 1u]));
                    if (s_n > 0.0) {
                        let t_n = min(s_n, d_n);   // node transfer (two-sided cap)
                        loss += (wijk * my_supply) * (t_n / s_n); // my weighted drain share
                    }
                }
            }
        }
        // Volume only leaves via the capped share (loss ≤ my_supply by construction).
        let new_f = max(f_w - loss / wet_v_water(), 0.0);
        pos[p] = vec4<f32>(x, new_f);
    } else {
        let v_abs = pos[p].w;
        let demand = wet_demand(v_abs) * bloom_factor(chem[p].x);
        // Wet-contact clock for the bloom gate: a grain touching water (any local supply) ticks
        // its timer regardless of whether it absorbed this step (it can be saturated yet wet).
        var touching = false;
        var gain = 0.0;
        for (var k = 0; k < 3; k = k + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                for (var i = 0; i < 3; i = i + 1) {
                    let wijk = w[i].x * w[j].y * w[k].z;
                    let n = node_index(base + vec3<i32>(i, j, k));
                    let s_n = fp_decode(atomicLoad(&grid_moist[n * 2u + 0u]));
                    let d_n = fp_decode(atomicLoad(&grid_moist[n * 2u + 1u]));
                    if (s_n > 0.0) { touching = true; }
                    if (demand > 0.0 && d_n > 0.0) {
                        let t_n = min(s_n, d_n);
                        gain += (wijk * demand) * (t_n / d_n); // my weighted fill share
                    }
                }
            }
        }
        pos[p] = vec4<f32>(x, v_abs + gain);
        if (touching) {
            chem[p] = vec4<f32>(chem[p].x + params.dt, chem[p].y, chem[p].z, chem[p].w);
        }
    }
}

// Two-field coupling (plan 2026-06-09-001 U6/U7 / KTD-3 / KTD-4 / KTD-8): thin solid-volume
// P2G plus an exponential interphase drag fold. Frozen mode retains the rigid v_s=0 forced
// integrator and reaction ledger. Dynamic mode reads grid_svel and performs one equal-and-
// opposite two-mass impulse, so drag changes relative velocity without changing node momentum.
//
// ================================ φ FIELDS (KTD-8) ============================================
// φ_s per node = (Σ_grains V_struct·w) / h³ from the quadratic-B-spline scatter. V_struct
// starts at the quadrature-weighted dry structural sample volume (params.coupling.z) and responds to bound-water
// saturation through the independent maximum-diameter swelling law below. φ_s is clamped at
// PHI_S_MAX = 0.95 (mirrors the
// xpbd packing clamp: φ_f never below 0.05, so the mixture operator stays nonsingular).
// φ_f = 1 − φ_s. The field is local and kernel-smoothed — a configured global porosity
// scalar is the rejected anti-pattern.
//
// ================================ DRAG FOLD (KTD-3, exact forms) ==============================
// The drag rate β(φ) comes from Kozeny-Carman (mirrors models::permeability — same clamps,
// same 150 coefficient, β = drag_scale/k), blended toward a Wen-Yu-style dilute rate below
// φ_s ≈ 0.2 with the Huilin-Gidaspow smooth arctan transition (constants below). The fold is
// the FORCED exponential integrator for dv/dt = −β·(v − v_s) + g with v_s ≡ 0
// in frozen mode:
//     v⁺ = v·e^{−βΔt} + g·srcf,        srcf = (1 − e^{−βΔt})/β   (→ Δt as β → 0),
// exact for frozen-coefficient linear drag at ANY stiffness — including its forcing. The
// plain "fold the gravity kick" form v⁺ = (v + gΔt)·e is NOT exact for the forced system:
// at equilibrium it eats (1 − e) of the hydrostatic load into spurious drag reaction, the
// pore-pressure slope collapses to ρg·(1−e)/(βΔt), and the pre-registered buoyancy/undrained
// gates fail at stiff β (measured: the recorded buoyant reaction flips SIGN at β·Δt ≈ 1.9).
//
// In dynamic mode beta is the fluid-side relaxation rate. For node masses m_f and m_s the
// relative rate is gamma = beta*(1 + m_f/m_s). After both phases receive their non-drag force
// updates, one exact pair impulse is applied:
//     J = mu*(1-exp(-gamma*dt))*(v_f-v_s),  mu = m_f*m_s/(m_f+m_s),
//     v_f' = v_f - J/m_f,                   v_s' = v_s + J/m_s.
// Thus m_f*v_f + m_s*v_s is invariant and v_f-v_s decays by exp(-gamma*dt). Boundary
// projection follows as an external wall impulse.
//
// The same relative-mode weight must apply to the pressure force the projection adds AFTER
// this pass: the per-node mobility used by the projection family is scaled by
//     ς = srcf/Δt ∈ (0, 1]   (react.w, folded into M̃⁻¹ by node_setup),
// Frozen mode uses this directly as Δv_f = −(ς·Δt/ρ_w)·∇p. Dynamic mode uses
// sigma to condense the pressure response of both phases into phase mobilities r_f and r_s
// (pressure.wgsl); their volume-weighted response is the exact mixture operator.
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
// terms net to exactly the displaced-volume weight — the buoyancy gate. Dynamic mode applies J
// to grid_svel immediately; react.xyz may expose that already-applied impulse for diagnostics,
// but project never substitutes ledger bookkeeping for a solid velocity update.
//
// EXACT-ZERO PASSTHROUGH (free-water regression gate): every coupling kernel gates on the
// solid-mass census (decoded grid_sfp = 0), not on β → 0 — zero-solid nodes take the
// byte-identical pre-U6 arithmetic path (no exp(0) rounding in the loop).
//
// This pass also owns the grid forces + grid-node boundary conditions (moved verbatim from
// U2's grid_update, which now only decodes): the fold must integrate the gravity source
// itself, and the BCs must come after all velocity updates. Pipeline order (the U6 HTD):
//   grid_clear → p2g_water → p2g_solid[_dyn] → grid_update(decode)
//   → [solid_update] → drag_fold(forces+drag+BC)
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

// Visible node control volume shared by porosity and saturation. Keeping this as one helper is
// the wall-truncation contract: water volume and solid volume are normalized over exactly the
// same fraction of h³, including SDF walls and the domain boundary.
fn node_visible_volume(xp: vec3<f32>) -> f32 {
    let h = params.grid_origin.w;
    return h * h * h * max(node_vis(xp), 1.0) / 8.0;
}

fn phi_f_from_solid_and_visible_volume(solid_volume: f32, visible_volume: f32) -> f32 {
    return 1.0 - min(solid_volume / visible_volume, PHI_S_MAX);
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
    return phi_f_from_solid_and_visible_volume(sv, node_visible_volume(xp));
}

// Local pore saturation S = V_water / (phi_f * V_visible), with V_water decoded from the same
// reduced water density used to seed particle mass. A collapsed/invalid pore volume is dry rather
// than singular; physical values are bounded because Corey mobility is defined on [0,1].
fn pore_saturation_from_node_state(
    water_mass: f32,
    phi_f: f32,
    visible_volume: f32,
) -> f32 {
    let pore_volume = max(phi_f, 0.0) * visible_volume;
    if (pore_volume <= 0.0 || params.extra.x <= 0.0) {
        return 0.0;
    }
    let water_volume = max(water_mass, 0.0) / params.extra.x;
    return clamp(water_volume / pore_volume, 0.0, 1.0);
}

// Wetting-phase Corey mobility on effective saturation. The configured positive floor prevents
// a singular drag rate at a dry front; NUMERICAL_MOBILITY_MIN is only a guard for malformed direct
// Config construction. A floor of one is an exact legacy passthrough (k_r = 1, no pow cost).
const NUMERICAL_MOBILITY_MIN: f32 = 1.0e-6;
fn bounded_corey_mobility(local_saturation: f32) -> f32 {
    let floor = clamp(params.retention.z, NUMERICAL_MOBILITY_MIN, 1.0);
    if (floor >= 1.0) {
        return 1.0;
    }
    let residual = clamp(params.retention.x, 0.0, 1.0);
    if (residual >= 1.0) {
        return floor;
    }
    let effective = clamp((local_saturation - residual) / (1.0 - residual), 0.0, 1.0);
    let exponent = max(params.retention.y, NUMERICAL_MOBILITY_MIN);
    return clamp(max(pow(effective, exponent), floor), floor, 1.0);
}

// A missing/domain-exterior/non-bed neighbor adopts center saturation (homogeneous Neumann), so
// only gradients supported on both sides by the bed can create capillary force. A bed node with
// no water is a valid dry neighbor (S=0), which is precisely the advancing-front signal.
fn bed_neighbor_saturation(
    center_saturation: f32,
    center_coords: vec3<i32>,
    offset: vec3<i32>,
) -> f32 {
    let neighbor_coords = center_coords + offset;
    let dims = vec3<i32>(params.grid_dims.xyz);
    if (any(neighbor_coords < vec3<i32>(0)) || any(neighbor_coords >= dims)) {
        return center_saturation;
    }
    let neighbor = node_index(neighbor_coords);
    let neighbor_solid_volume = solid_volume_at(neighbor);
    if (neighbor_solid_volume <= 0.0) {
        return center_saturation;
    }
    let h = params.grid_origin.w;
    let neighbor_xp = params.grid_origin.xyz + vec3<f32>(neighbor_coords) * h;
    let visible_volume = node_visible_volume(neighbor_xp);
    let phi_f = phi_f_from_solid_and_visible_volume(neighbor_solid_volume, visible_volume);
    return pore_saturation_from_node_state(grid_vel[neighbor].w, phi_f, visible_volume);
}

// Capillarity pulls toward lower saturation: a_cap = -a_suction * q * normalize(grad S), with
// q = clamp(2h|grad S|,0,1). The dimensionless q removes force under uniform saturation and
// smoothly reaches the calibrated Green-Ampt magnitude across a one-node saturation front.
fn capillary_front_acceleration(
    center_coords: vec3<i32>,
    center_saturation: f32,
    suction_accel: f32,
) -> vec3<f32> {
    let sxm = bed_neighbor_saturation(center_saturation, center_coords, vec3<i32>(-1, 0, 0));
    let sxp = bed_neighbor_saturation(center_saturation, center_coords, vec3<i32>(1, 0, 0));
    let sym = bed_neighbor_saturation(center_saturation, center_coords, vec3<i32>(0, -1, 0));
    let syp = bed_neighbor_saturation(center_saturation, center_coords, vec3<i32>(0, 1, 0));
    let szm = bed_neighbor_saturation(center_saturation, center_coords, vec3<i32>(0, 0, -1));
    let szp = bed_neighbor_saturation(center_saturation, center_coords, vec3<i32>(0, 0, 1));
    let h = params.grid_origin.w;
    let gradient = vec3<f32>(sxp - sxm, syp - sym, szp - szm) / (2.0 * h);
    let magnitude = length(gradient);
    if (magnitude <= 0.0) {
        return vec3<f32>(0.0);
    }
    let strength = clamp(2.0 * h * magnitude, 0.0, 1.0);
    return -suction_accel * strength * gradient / magnitude;
}

// Bound-water inventory and structural swelling are deliberately independent. pos.w on a
// grain is V_bound. It controls structure only through saturation; it contributes dynamic
// mass through the exact reduced-unit water mass ratio, never through structural volume.
fn grain_bound_saturation(v_bound: f32) -> f32 {
    let v_cap = params.wet0.y;
    if (v_cap <= 0.0) { return 0.0; }
    return clamp(v_bound / v_cap, 0.0, 1.0);
}

fn grain_structural_volume(v_bound: f32) -> f32 {
    let max_diameter_swelling = max(params.wet1.w, 0.0);
    let stretch = 1.0 + grain_bound_saturation(v_bound) * max_diameter_swelling;
    return params.coupling.z * stretch * stretch * stretch;
}

fn grain_bound_water_mass(v_bound: f32) -> f32 {
    let v_water = params.wet0.z;
    if (v_water > 0.0) {
        return max(v_bound, 0.0) / v_water * params.particle_mass;
    }
    return 0.0;
}

fn grain_effective_mass(v_bound: f32) -> f32 {
    return params.splas1.z + grain_bound_water_mass(v_bound);
}

fn grain_heat_capacity(v_bound: f32) -> f32 {
    return params.splas1.z * params.therm.y
        + grain_bound_water_mass(v_bound) * params.therm.x;
}

// =================================== p2g_solid =================================================
// Thin solid-volume P2G: scatter each frozen grain's structural volume with the same quadratic
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
    // Structural volume is a saturation-driven constitutive response, not bound-water storage:
    // V_struct = V_dry*(1 + s*max_diameter_swelling)^3. The default maximum is zero, preserving
    // the dry U6 scatter exactly even when a calibration stores bound water.
    let vg = grain_structural_volume(pos[p].w);
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
    var phi_f = 1.0;
    var saturation = 0.0;
    if (sv > 0.0) {
        let visible_volume = node_visible_volume(xp);
        phi_f = phi_f_from_solid_and_visible_volume(sv, visible_volume);
        saturation = pore_saturation_from_node_state(mass, phi_f, visible_volume);

        // The calibrated Green-Ampt magnitude now follows the local saturation-gradient normal,
        // not gravity. It is evaluated only inside the bed; missing/non-bed neighbors receive a
        // zero-normal-gradient boundary value. Folding it into g keeps the forced exponential
        // drag integration and its pressure mobility sigma algebraically unchanged.
        let a_suction = params.wet1.x;
        if (a_suction > 0.0) {
            g = g + capillary_front_acceleration(vec3<i32>(c), saturation, a_suction);
        }
    }
    var sig = 1.0;
    var imp = vec3<f32>(0.0);
    var solid_v = grid_svel[n].xyz;
    let solid_mass = grid_svel[n].w;
    var dynamic_pair = false;
    if (sv > 0.0) {
        let phi_s = 1.0 - phi_f;
        // Relative permeability modifies only beta. The exact pair impulse and react.w sigma are
        // still derived from that same beta, so node_setup/project retain the identical condensed
        // pressure operator and dynamic pair momentum remains equal-and-opposite.
        let relative_mobility = bounded_corey_mobility(saturation);
        let beta = drag_rate_blended(phi_s, phi_f) / relative_mobility;
        if (params.splas0.x > 0.5 && solid_mass > params.extra.z) {
            // Both phases already contain their non-drag forces (water `g` here, solids in
            // solid_update). Apply the exact homogeneous pair flow once. `imp` is the impulse
            // water gives the solid; equal-and-opposite updates conserve node momentum exactly
            // before wall projection and safety caps.
            let fluid_forced = v + g * params.dt;
            let gamma = beta * (1.0 + mass / solid_mass);
            let a = gamma * params.dt;
            var one_minus_e: f32;
            if (a < 1.0e-3) {
                one_minus_e = a * (1.0 - 0.5 * a + a * a / 6.0);
                sig = 1.0 - 0.5 * a + a * a / 6.0;
            } else {
                one_minus_e = 1.0 - exp(-a);
                sig = one_minus_e / a;
            }
            let reduced_mass = mass * solid_mass / (mass + solid_mass);
            imp = reduced_mass * one_minus_e * (fluid_forced - solid_v);
            v = fluid_forced - imp / mass;
            solid_v = solid_v + imp / solid_mass;
            dynamic_pair = true;
        } else {
            // Frozen mode keeps the forced exponential integrator exactly. A dynamic node with
            // no resolved solid mass has no pair partner and follows the drag-free force path.
            if (params.splas0.x > 0.5) {
                v = v + g * params.dt;
            } else {
                let a = beta * params.dt;
                let e = exp(-a);
                var srcf: f32;
                if (a < 1.0e-3) {
                    srcf = params.dt * (1.0 - 0.5 * a);
                } else {
                    srcf = (1.0 - e) / beta;
                }
                let naive = v + g * params.dt;
                v = v * e + g * srcf;
                sig = srcf / params.dt;
                imp = mass * (naive - v);
            }
        }
    } else {
        v = v + g * params.dt; // exact-zero passthrough: byte-identical pre-U6 gravity path
    }

    if (dynamic_pair) {
        // Drag can reintroduce an approaching contact component after solid_update. Reapply the
        // identical unilateral Coulomb BC at actual contact only; separating motion survives and
        // grains outside the water pressure band are untouched. This wall impulse is external to
        // the momentum-conserving pair update.
        solid_v = solid_contact_velocity_bc(xp, solid_v);
        grid_svel[n] = vec4<f32>(solid_v, solid_mass);
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
    // Wall BC. The box-face normal components are removed (free slip tangentially), matching the
    // constrained M̃⁻¹; the SDF wall adds its in-band normal(s). Both modes share the SAME basis
    // node_setup builds (operator consistency A = D·M̃⁻¹·G; a one-sided BC here was tried and
    // reverted). The open base keeps the y-min face open for the water field.
    if (!wall_bc_multi()) {
        // SINGLE-NORMAL (default): box-face axis zeroing + the most-penetrated SDF normal, banded,
        // box-orthogonalized — byte-identical to the original BC.
        if (xp.x <= params.box_min.x + eps || xp.x >= params.box_max.x - eps) { v.x = 0.0; }
        if ((xp.y <= params.box_min.y + eps && !open_base) || xp.y >= params.box_max.y - eps) { v.y = 0.0; }
        if (xp.z <= params.box_min.z + eps || xp.z >= params.box_max.z - eps) { v.z = 0.0; }
        if (params.num_solids > 0u) {
            let hit = solid_union(xp, PHASE_WATER);
            if (hit.dist < WALL_BAND * params.grid_origin.w) {
                var nrm = hit.grad;
                if (xp.x <= params.box_min.x + eps || xp.x >= params.box_max.x - eps) { nrm.x = 0.0; }
                if ((xp.y <= params.box_min.y + eps && !open_base) || xp.y >= params.box_max.y - eps) { nrm.y = 0.0; }
                if (xp.z <= params.box_min.z + eps || xp.z >= params.box_max.z - eps) { nrm.z = 0.0; }
                let len = length(nrm);
                if (len > SDF_NORMAL_MIN) {
                    nrm = nrm / len;
                    v = v - dot(v, nrm) * nrm;
                }
            }
        }
    } else {
        // MULTI-NORMAL: v ← P·v with P = I − Q·Qᵀ over the active box axes + ALL in-band wall faces
        // — the IDENTICAL basis node_setup builds for M̃⁻¹ = invr·P (exact lockstep). box_mask
        // matches the single path's box-face conditions; faces empty when no solids.
        var box_mask = 0u;
        if (xp.x <= params.box_min.x + eps || xp.x >= params.box_max.x - eps) { box_mask = box_mask | 1u; }
        if ((xp.y <= params.box_min.y + eps && !open_base) || xp.y >= params.box_max.y - eps) { box_mask = box_mask | 2u; }
        if (xp.z <= params.box_min.z + eps || xp.z >= params.box_max.z - eps) { box_mask = box_mask | 4u; }
        var faces: WallFaces;
        faces.count = 0u;
        if (params.num_solids > 0u) {
            faces = wall_binding_faces(xp, PHASE_WATER, WALL_BAND * params.grid_origin.w);
        }
        let cb = build_constraint_basis(box_mask, faces);
        v = cbasis_project_vec(cb, v);
    }
    // Speed-cap backstop (couples to the FP headroom math in common.wgsl). U3: this is the WATER
    // grid (writes grid_vel) and runs BEFORE project/g2p, so it must use the separate splash cap
    // (flip.w) — clipping the water grid at the global 12/max_speed here would defeat the crown.
    // Sentinel ≤ 0 ⇒ global max_speed (byte-identical default). The solid grid clamp lives in
    // solid_update (plasticity.wgsl) and stays on the global max_speed — grain dynamics unperturbed.
    let water_cap = select(params.max_speed, params.flip.w, params.flip.w > 0.0);
    let s = length(v);
    if (s > water_cap) {
        v = v * (water_cap / s);
    }
    grid_vel[n] = vec4<f32>(v, mass);
    react[n] = vec4<f32>(imp, sig);
}

// ============================ U9 MOISTURE PHASE-CHANGE ABSORPTION (KTD-4c) =====================
// GIC-style transfer of water from the free phase to the bound-water inventory. The transferred
// water VOLUME and its corresponding reduced-unit MASS are exactly conserved; structural grain
// volume is a separate saturation response (grain_structural_volume above). Grid-projected,
// atomic-free in the gather pass, deterministic and two-sidedly capped.
//
// p2g_moisture scatters six fixed-point grid lanes with the SAME quadratic B-spline weights as
// the field transfers: water SUPPLY S_n = Σ_w w·f_w·V_w, grain DEMAND
// D_n = Σ_g w·demand_g, and donor volume-momentum Q_n = Σ_w w·f_w·V_w·v_w.
// demand_g = (V_cap − V_bound)·(1 − e^{−k_abs·dt}) (models::wetting::absorb_demand).
// The node transfer is T_n = min(S_n,D_n). g2p_absorb makes:
//   * each WATER lose its weighted share  Σ_n w·(f_w·V_w)·(T_n/S_n),
//   * each GRAIN gain its weighted share  Σ_n w·(demand_g)·(T_n/D_n).
// Σ over particles at node n: water loses Σ_w w·f_w·V_w·(T_n/S_n) = S_n·(T_n/S_n) = T_n, and
// grains gain D_n·(T_n/D_n) = T_n. Donors lose reduced momentum
// (particle_mass/V_w)·T_n·Q_n/S_n; receivers gain the identical ledger amount through their
// T_n/D_n shares, then merge it with their old momentum. Volume, mass, and momentum therefore
// match per node up to the one shared fixed-point quantization. f_w / V_bound ride pos.w.
// Bloom (KTD-4b): a dry grain's demand is
// scaled by a contact-time ramp (chem.w accumulates wet contact; ramps 0→1 over bloom_delay) so
// the hydrophobic entry delay falls out. Off by default (k_abs = 0 ⇒ CPU-elided in mod.rs).

fn wet_v_cap() -> f32 { return params.wet0.y; }
fn wet_v_water() -> f32 { return params.wet0.z; }
fn wet_roundoff() -> f32 { return params.wet0.w; }
fn wet_sat_cutoff() -> f32 { let c = wet_v_cap(); return c - c * wet_roundoff(); }
// Stable 1-exp(-x): direct subtraction loses the UI-minimum rate at sufficiently short
// constitutive substeps. The short-series branch preserves the current cadence without inventing
// a larger effective timestep or accumulating an untracked residual.
fn positive_decay_fraction(x: f32) -> f32 {
    let y = max(x, 0.0);
    if (y < 1.0e-3) {
        return clamp(y * (1.0 - 0.5 * y + y * y / 6.0), 0.0, 1.0);
    }
    return clamp(1.0 - exp(-y), 0.0, 1.0);
}
fn wet_demand(v_bound: f32) -> f32 {
    if (v_bound >= wet_sat_cutoff()) { return 0.0; }
    return (wet_v_cap() - v_bound) * positive_decay_fraction(params.wet0.x * params.dt);
}
// Absorption removes solvent, not dissolved solute. Keeping at least f*c/c_sat solvent makes
// c_new=c*f/f_new <= c_sat structurally; roundoff remains the independent deactivation floor.
// A non-positive saturation concentration disables solvent removal rather than dividing by zero.
fn water_solvent_floor(f_water: f32, concentration: f32) -> f32 {
    if (params.chem1.x <= 1.0e-9) { return max(f_water, wet_roundoff()); }
    return max(wet_roundoff(), f_water * concentration / params.chem1.x);
}
// Bloom factor for a grain (KTD-4b): a smooth ramp of its accumulated wet-contact time
// (chem.w, physical seconds) over bloom_delay. No delay ⇒ factor 1 immediately.
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
    var lane = MOIST_SUPPLY;
    if (is_water) {
        let f_w = pos[p].w;
        // The solute-safe solvent floor is never absorbable: it retains enough water to carry the
        // already-dissolved mass without exceeding c_sat.
        let absorbable_fraction = max(f_w - water_solvent_floor(f_w, chem[p].x), 0.0);
        if (absorbable_fraction <= 0.0) { return; }
        quantity = absorbable_fraction * wet_v_water();
        lane = MOIST_SUPPLY;
    } else {
        let v_bound = pos[p].w;
        let demand = wet_demand(v_bound) * bloom_factor(chem[p].w);
        if (demand <= 0.0) { return; } // saturated / unbloomed grain: no demand
        quantity = demand;
        lane = MOIST_DEMAND;
    }
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    let total_quantity = moist_encode(lane, quantity);
    var allocated_quantity = 0i;
    var total_q = vec3<i32>(0);
    var allocated_q = vec3<i32>(0);
    var total_r = 0i;
    var allocated_r = 0i;
    if (is_water) {
        total_q = vec3<i32>(
            moist_encode(MOIST_QX + 0u, quantity * vel[p].x),
            moist_encode(MOIST_QX + 1u, quantity * vel[p].y),
            moist_encode(MOIST_QX + 2u, quantity * vel[p].z),
        );
        total_r = moist_encode(MOIST_RT, quantity * chem[p].y);
    }
    var cumulative_weight = 0.0;
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                cumulative_weight += wijk;
                let is_last = i == 2 && j == 2 && k == 2;
                let node = base + vec3<i32>(i, j, k);
                let moist = node_index(node) * MOIST_STRIDE;
                let quantity_target = extensive_partition_target(
                    total_quantity, cumulative_weight, is_last,
                );
                let quantity_share = quantity_target - allocated_quantity;
                allocated_quantity = quantity_target;
                atomicAdd(&grid_moist[moist + lane], quantity_share);
                if (is_water) {
                    // Each extensive donor total is encoded once; cumulative partitioning makes
                    // all 27 integer shares sum exactly to that total, including signed momentum.
                    for (var axis = 0; axis < 3; axis = axis + 1) {
                        let q_target = extensive_partition_target(
                            total_q[axis], cumulative_weight, is_last,
                        );
                        atomicAdd(
                            &grid_moist[moist + MOIST_QX + u32(axis)],
                            q_target - allocated_q[axis],
                        );
                        allocated_q[axis] = q_target;
                    }
                    let r_target = extensive_partition_target(
                        total_r, cumulative_weight, is_last,
                    );
                    atomicAdd(&grid_moist[moist + MOIST_RT], r_target - allocated_r);
                    allocated_r = r_target;
                }
            }
        }
    }
}

// =================================== g2p_absorb ===============================================
// Gather the node drain/fill fractions and apply the conserved transfer (atomic-free: each
// particle writes only its own pos.w / chem.w). Water shrinks f_w; grain grows V_bound and
// ticks its wet-contact timer (the bloom clock), leaving concentration/temperature untouched.
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
        let carrier_floor = water_solvent_floor(f_w, chem[p].x);
        let absorbable_fraction = max(f_w - carrier_floor, 0.0);
        if (absorbable_fraction <= 0.0) { return; }
        let my_supply = absorbable_fraction * wet_v_water();
        let my_supply_total = moist_encode(MOIST_SUPPLY, my_supply);
        var allocated_supply = 0i;
        var cumulative_weight = 0.0;
        var loss = 0.0;
        for (var k = 0; k < 3; k = k + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                for (var i = 0; i < 3; i = i + 1) {
                    let wijk = w[i].x * w[j].y * w[k].z;
                    cumulative_weight += wijk;
                    let is_last = i == 2 && j == 2 && k == 2;
                    let supply_target = extensive_partition_target(
                        my_supply_total, cumulative_weight, is_last,
                    );
                    let supply_share = moist_decode(
                        MOIST_SUPPLY, supply_target - allocated_supply,
                    );
                    allocated_supply = supply_target;
                    let n = node_index(base + vec3<i32>(i, j, k));
                    let moist = n * MOIST_STRIDE;
                    let s_n = moist_decode(
                        MOIST_SUPPLY, atomicLoad(&grid_moist[moist + MOIST_SUPPLY]),
                    );
                    let d_n = moist_decode(
                        MOIST_DEMAND, atomicLoad(&grid_moist[moist + MOIST_DEMAND]),
                    );
                    if (s_n > 0.0) {
                        let t_n = min(s_n, d_n);   // node transfer (two-sided cap)
                        loss += supply_share * (t_n / s_n); // my exact integer-partition share
                    }
                }
            }
        }
        // Volume only leaves via the capped share. Preserve dissolved solute exactly while pure
        // water becomes bound: c_new*f_new*m_p = c_old*f_old*m_p.
        let new_f = max(f_w - loss / wet_v_water(), carrier_floor);
        let new_c = chem[p].x * f_w / new_f;
        pos[p] = vec4<f32>(x, new_f);
        chem[p] = vec4<f32>(new_c, chem[p].yzw);
    } else {
        let v_bound = pos[p].w;
        let demand = wet_demand(v_bound) * bloom_factor(chem[p].w);
        let demand_total = moist_encode(MOIST_DEMAND, demand);
        var allocated_demand = 0i;
        var cumulative_weight = 0.0;
        // Wet-contact clock for the bloom gate: a grain touching water (any local supply) ticks
        // its timer regardless of whether it absorbed this step (it can be saturated yet wet).
        var touching = false;
        var gain = 0.0;
        var incoming_volume_momentum = vec3<f32>(0.0);
        var incoming_volume_temperature = 0.0;
        for (var k = 0; k < 3; k = k + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                for (var i = 0; i < 3; i = i + 1) {
                    let wijk = w[i].x * w[j].y * w[k].z;
                    cumulative_weight += wijk;
                    let is_last = i == 2 && j == 2 && k == 2;
                    let demand_target = extensive_partition_target(
                        demand_total, cumulative_weight, is_last,
                    );
                    let demand_share = moist_decode(
                        MOIST_DEMAND, demand_target - allocated_demand,
                    );
                    allocated_demand = demand_target;
                    let n = node_index(base + vec3<i32>(i, j, k));
                    let moist = n * MOIST_STRIDE;
                    let s_n = moist_decode(
                        MOIST_SUPPLY, atomicLoad(&grid_moist[moist + MOIST_SUPPLY]),
                    );
                    let d_n = moist_decode(
                        MOIST_DEMAND, atomicLoad(&grid_moist[moist + MOIST_DEMAND]),
                    );
                    if (s_n > 0.0) { touching = true; }
                    if (demand > 0.0 && s_n > 0.0 && d_n > 0.0) {
                        let t_n = min(s_n, d_n);
                        let gain_n = demand_share * (t_n / d_n);
                        let q_n = vec3<f32>(
                            moist_decode(
                                MOIST_QX + 0u,
                                atomicLoad(&grid_moist[moist + MOIST_QX + 0u]),
                            ),
                            moist_decode(
                                MOIST_QX + 1u,
                                atomicLoad(&grid_moist[moist + MOIST_QX + 1u]),
                            ),
                            moist_decode(
                                MOIST_QX + 2u,
                                atomicLoad(&grid_moist[moist + MOIST_QX + 2u]),
                            ),
                        );
                        let r_n = moist_decode(
                            MOIST_RT, atomicLoad(&grid_moist[moist + MOIST_RT]),
                        );
                        gain += gain_n;
                        incoming_volume_momentum += gain_n * (q_n / s_n);
                        incoming_volume_temperature += gain_n * (r_n / s_n);
                    }
                }
            }
        }
        var new_temperature = chem[p].z;
        if (gain > 0.0 && wet_v_water() > 0.0) {
            // Conservative inelastic merge of the old wet grain with transferred water. The
            // water particles keep velocity while losing mass; Q/S carries exactly the velocity
            // of that lost mass into this receiver allocation.
            let water_mass_per_volume = params.particle_mass / wet_v_water();
            let old_mass = grain_effective_mass(v_bound);
            let gained_mass = water_mass_per_volume * gain;
            let new_mass = old_mass + gained_mass;
            let new_velocity = (
                old_mass * vel[p].xyz
                + water_mass_per_volume * incoming_volume_momentum
            ) / new_mass;
            vel[p] = vec4<f32>(new_velocity, vel[p].w);

            // Bound water carries its donor temperature into the grain. This is the thermal
            // analogue of the momentum merge and prevents phase change from creating enthalpy.
            let old_capacity = grain_heat_capacity(v_bound);
            let incoming_capacity_per_volume = water_mass_per_volume * params.therm.x;
            let gained_capacity = incoming_capacity_per_volume * gain;
            let new_capacity = old_capacity + gained_capacity;
            if (new_capacity > 0.0) {
                new_temperature = (
                    old_capacity * chem[p].z
                    + incoming_capacity_per_volume * incoming_volume_temperature
                ) / new_capacity;
            }
        }
        pos[p] = vec4<f32>(x, v_bound + gain);
        let contact_time = chem[p].w + select(0.0, params.dt, touching);
        chem[p] = vec4<f32>(chem[p].xy, new_temperature, contact_time);
    }
}

// ======================== CONSERVATIVE EXTRACTION + THERMAL TRANSPORT =========================
// These two passes run after pressure projection and before particle advection. The scatter pass
// freezes seven extensive node totals; the gather pass allocates one node transfer to both sides.
// No particle writes a shared float, so there are no cross-particle races. For every node:
//
//   X_n = min(H_n, F_n + L_n)
//
// is removed from the grain pools and added to dissolved-water mass. The particle gather weights
// are the exact adjoints of the scatter weights, hence the two phase totals differ only by the
// fixed-point rounding already shared by the node ledger. Thermal exchange uses the same pattern
// with q_n = alpha * Cw*Cg/(Cw+Cg) * (Tg-Tw), the exact bounded step toward the node's
// capacity-weighted equilibrium. Ambient exchange is deliberately separate and is the only
// external enthalpy source/sink in this family.

fn chemistry_arrhenius(temperature: f32) -> f32 {
    let tref = params.chem0.w;
    if (temperature <= 1.0e-6 || tref <= 1.0e-6) {
        return 1.0;
    }
    // exp[-Ea/R (1/T - 1/Tref)]. The clamp is only a finite-arithmetic guard; the UI/material
    // ranges are much narrower than it.
    let exponent = -params.chem0.z * (1.0 / temperature - 1.0 / tref);
    return exp(clamp(exponent, -30.0, 30.0));
}

fn chemistry_wet_gate(saturation: f32) -> f32 {
    let s_on = params.chem1.z;
    if (s_on <= 1.0e-6) {
        return select(0.0, 1.0, saturation > 0.0);
    }
    let t = clamp(saturation / s_on, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn chemistry_renewal(relative_speed: f32) -> f32 {
    // A finite base release remains at u=0. Relative flow can at most double it. This is the
    // calibrated renewal law from models::extraction; there is intentionally no fake area term.
    let u = max(relative_speed, 0.0);
    let u_half = max(params.chem1.y, 1.0e-6);
    return 1.0 + u / (u + u_half);
}

fn bounded_pool_release(pool: f32, rate: f32) -> f32 {
    let available = max(pool, 0.0);
    let fraction = positive_decay_fraction(max(rate, 0.0) * params.dt);
    return clamp(available * fraction, 0.0, available);
}

fn water_heat_capacity(f_water: f32) -> f32 {
    return params.particle_mass * clamp(f_water, 0.0, 1.0) * max(params.therm.x, 0.0);
}

fn node_heat_transfer(c_water: f32, e_water: f32, c_grain: f32, e_grain: f32) -> f32 {
    if (c_water <= 0.0 || c_grain <= 0.0) {
        return 0.0;
    }
    let t_water = e_water / c_water;
    let t_grain = e_grain / c_grain;
    let reduced_capacity = c_water * c_grain / (c_water + c_grain);
    let alpha = positive_decay_fraction(max(params.chem1.w, 0.0) * params.dt);
    return alpha * reduced_capacity * (t_grain - t_water);
}

fn ambient_temperature_step(temperature: f32) -> f32 {
    let alpha = positive_decay_fraction(max(params.therm.z, 0.0) * params.dt);
    return temperature + alpha * (params.therm.w - temperature);
}

// Freeze the kinetic proposal for one grain. The pressure-projected phase velocities are
// interpolated with the same B-spline as every other particle-grid transfer. Frozen skeletons
// naturally use v_s=0 because their solid velocity grid is zero; dynamic skeletons read the
// projected grid_svel field.
fn grain_release_proposal(p: u32) -> vec2<f32> {
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    var v_water = vec3<f32>(0.0);
    var v_solid = vec3<f32>(0.0);
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                let n = node_index(base + vec3<i32>(i, j, k));
                v_water += wijk * grid_vel[n].xyz;
                v_solid += wijk * grid_svel[n].xyz;
            }
        }
    }
    let state = chem[p];
    let multiplier = chemistry_arrhenius(state.z)
        * chemistry_wet_gate(grain_bound_saturation(pos[p].w))
        * chemistry_renewal(length(v_water - v_solid));
    return vec2<f32>(
        bounded_pool_release(state.x, params.chem0.x * multiplier),
        bounded_pool_release(state.y, params.chem0.y * multiplier),
    );
}

// ================================= chemistry_scatter =========================================
@compute @workgroup_size(256)
fn chemistry_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        return;
    }

    var headroom = 0.0;
    var release = vec2<f32>(0.0);
    var capacity = 0.0;
    var enthalpy = 0.0;
    if (is_water) {
        let mass = params.particle_mass * clamp(pos[p].w, 0.0, 1.0);
        headroom = max(params.chem1.x - chem[p].x, 0.0) * mass;
        capacity = water_heat_capacity(pos[p].w);
        enthalpy = capacity * chem[p].y;
    } else {
        release = grain_release_proposal(p);
        capacity = grain_heat_capacity(pos[p].w);
        enthalpy = capacity * chem[p].z;
    }

    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);
    let total_headroom = chem_encode(CHEM_HEADROOM, headroom);
    let total_fast = chem_encode(CHEM_FAST, release.x);
    let total_slow = chem_encode(CHEM_SLOW, release.y);
    let capacity_lane = select(CHEM_CG, CHEM_CW, is_water);
    let enthalpy_lane = select(CHEM_EG, CHEM_EW, is_water);
    let total_capacity = chem_encode(capacity_lane, capacity);
    let total_enthalpy = chem_encode(enthalpy_lane, enthalpy);
    var allocated_headroom = 0i;
    var allocated_fast = 0i;
    var allocated_slow = 0i;
    var allocated_capacity = 0i;
    var allocated_enthalpy = 0i;
    var cumulative_weight = 0.0;
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let wijk = w[i].x * w[j].y * w[k].z;
                cumulative_weight += wijk;
                let is_last = i == 2 && j == 2 && k == 2;
                let offset = node_index(base + vec3<i32>(i, j, k)) * CHEM_STRIDE;
                if (is_water) {
                    let headroom_target = extensive_partition_target(
                        total_headroom, cumulative_weight, is_last,
                    );
                    atomicAdd(
                        &grid_chem[offset + CHEM_HEADROOM],
                        headroom_target - allocated_headroom,
                    );
                    allocated_headroom = headroom_target;
                } else {
                    let fast_target = extensive_partition_target(
                        total_fast, cumulative_weight, is_last,
                    );
                    let slow_target = extensive_partition_target(
                        total_slow, cumulative_weight, is_last,
                    );
                    atomicAdd(
                        &grid_chem[offset + CHEM_FAST],
                        fast_target - allocated_fast,
                    );
                    atomicAdd(
                        &grid_chem[offset + CHEM_SLOW],
                        slow_target - allocated_slow,
                    );
                    allocated_fast = fast_target;
                    allocated_slow = slow_target;
                }
                let capacity_target = extensive_partition_target(
                    total_capacity, cumulative_weight, is_last,
                );
                let enthalpy_target = extensive_partition_target(
                    total_enthalpy, cumulative_weight, is_last,
                );
                atomicAdd(
                    &grid_chem[offset + capacity_lane],
                    capacity_target - allocated_capacity,
                );
                atomicAdd(
                    &grid_chem[offset + enthalpy_lane],
                    enthalpy_target - allocated_enthalpy,
                );
                allocated_capacity = capacity_target;
                allocated_enthalpy = enthalpy_target;
            }
        }
    }
}

// ================================= chemistry_gather ==========================================
@compute @workgroup_size(256)
fn chemistry_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        return;
    }

    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let fx = xl - vec3<f32>(base);
    var w = bspline_w(fx);

    if (is_water) {
        let fraction = clamp(pos[p].w, 0.0, 1.0);
        let mass = params.particle_mass * fraction;
        let my_headroom = max(params.chem1.x - chem[p].x, 0.0) * mass;
        let my_capacity = water_heat_capacity(fraction);
        let headroom_total = chem_encode(CHEM_HEADROOM, my_headroom);
        let capacity_total = chem_encode(CHEM_CW, my_capacity);
        var allocated_headroom = 0i;
        var allocated_capacity = 0i;
        var cumulative_weight = 0.0;
        var solute_gain = 0.0;
        var heat_gain = 0.0;
        for (var k = 0; k < 3; k = k + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                for (var i = 0; i < 3; i = i + 1) {
                    let wijk = w[i].x * w[j].y * w[k].z;
                    cumulative_weight += wijk;
                    let is_last = i == 2 && j == 2 && k == 2;
                    let headroom_target = extensive_partition_target(
                        headroom_total, cumulative_weight, is_last,
                    );
                    let capacity_target = extensive_partition_target(
                        capacity_total, cumulative_weight, is_last,
                    );
                    let headroom_share = chem_decode(
                        CHEM_HEADROOM, headroom_target - allocated_headroom,
                    );
                    let capacity_share = chem_decode(
                        CHEM_CW, capacity_target - allocated_capacity,
                    );
                    allocated_headroom = headroom_target;
                    allocated_capacity = capacity_target;
                    let offset = node_index(base + vec3<i32>(i, j, k)) * CHEM_STRIDE;
                    let headroom = chem_decode(
                        CHEM_HEADROOM, atomicLoad(&grid_chem[offset + CHEM_HEADROOM]),
                    );
                    let fast = chem_decode(
                        CHEM_FAST, atomicLoad(&grid_chem[offset + CHEM_FAST]),
                    );
                    let slow = chem_decode(
                        CHEM_SLOW, atomicLoad(&grid_chem[offset + CHEM_SLOW]),
                    );
                    let c_water = chem_decode(
                        CHEM_CW, atomicLoad(&grid_chem[offset + CHEM_CW]),
                    );
                    let e_water = chem_decode(
                        CHEM_EW, atomicLoad(&grid_chem[offset + CHEM_EW]),
                    );
                    let c_grain = chem_decode(
                        CHEM_CG, atomicLoad(&grid_chem[offset + CHEM_CG]),
                    );
                    let e_grain = chem_decode(
                        CHEM_EG, atomicLoad(&grid_chem[offset + CHEM_EG]),
                    );
                    if (headroom > 0.0) {
                        solute_gain += headroom_share * (min(headroom, fast + slow) / headroom);
                    }
                    if (c_water > 0.0) {
                        heat_gain += capacity_share / c_water
                            * node_heat_transfer(c_water, e_water, c_grain, e_grain);
                    }
                }
            }
        }
        // The analytic gather cannot exceed my_headroom. Clamp only the FP-rounding residue so
        // c<=c_sat is structural, rather than relying on an after-the-fact concentration clip.
        solute_gain = min(max(solute_gain, 0.0), my_headroom);
        var new_temperature = chem[p].y;
        if (my_capacity > 0.0) {
            new_temperature += heat_gain / my_capacity;
        }
        new_temperature = ambient_temperature_step(new_temperature);
        var new_concentration = chem[p].x;
        if (mass > 0.0) {
            new_concentration = (chem[p].x * mass + solute_gain) / mass;
        }
        chem[p] = vec4<f32>(new_concentration, new_temperature, chem[p].zw);
    } else {
        let proposal = grain_release_proposal(p);
        let my_capacity = grain_heat_capacity(pos[p].w);
        let fast_total = chem_encode(CHEM_FAST, proposal.x);
        let slow_total = chem_encode(CHEM_SLOW, proposal.y);
        let capacity_total = chem_encode(CHEM_CG, my_capacity);
        var allocated_fast = 0i;
        var allocated_slow = 0i;
        var allocated_capacity = 0i;
        var cumulative_weight = 0.0;
        var fast_loss = 0.0;
        var slow_loss = 0.0;
        var heat_gain = 0.0;
        for (var k = 0; k < 3; k = k + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                for (var i = 0; i < 3; i = i + 1) {
                    let wijk = w[i].x * w[j].y * w[k].z;
                    cumulative_weight += wijk;
                    let is_last = i == 2 && j == 2 && k == 2;
                    let fast_target = extensive_partition_target(
                        fast_total, cumulative_weight, is_last,
                    );
                    let slow_target = extensive_partition_target(
                        slow_total, cumulative_weight, is_last,
                    );
                    let capacity_target = extensive_partition_target(
                        capacity_total, cumulative_weight, is_last,
                    );
                    let fast_share = chem_decode(
                        CHEM_FAST, fast_target - allocated_fast,
                    );
                    let slow_share = chem_decode(
                        CHEM_SLOW, slow_target - allocated_slow,
                    );
                    let capacity_share = chem_decode(
                        CHEM_CG, capacity_target - allocated_capacity,
                    );
                    allocated_fast = fast_target;
                    allocated_slow = slow_target;
                    allocated_capacity = capacity_target;
                    let offset = node_index(base + vec3<i32>(i, j, k)) * CHEM_STRIDE;
                    let headroom = chem_decode(
                        CHEM_HEADROOM, atomicLoad(&grid_chem[offset + CHEM_HEADROOM]),
                    );
                    let fast = chem_decode(
                        CHEM_FAST, atomicLoad(&grid_chem[offset + CHEM_FAST]),
                    );
                    let slow = chem_decode(
                        CHEM_SLOW, atomicLoad(&grid_chem[offset + CHEM_SLOW]),
                    );
                    let c_water = chem_decode(
                        CHEM_CW, atomicLoad(&grid_chem[offset + CHEM_CW]),
                    );
                    let e_water = chem_decode(
                        CHEM_EW, atomicLoad(&grid_chem[offset + CHEM_EW]),
                    );
                    let c_grain = chem_decode(
                        CHEM_CG, atomicLoad(&grid_chem[offset + CHEM_CG]),
                    );
                    let e_grain = chem_decode(
                        CHEM_EG, atomicLoad(&grid_chem[offset + CHEM_EG]),
                    );
                    let proposed = fast + slow;
                    if (proposed > 0.0) {
                        let accepted = min(headroom, proposed) / proposed;
                        fast_loss += fast_share * accepted;
                        slow_loss += slow_share * accepted;
                    }
                    if (c_grain > 0.0) {
                        heat_gain -= capacity_share / c_grain
                            * node_heat_transfer(c_water, e_water, c_grain, e_grain);
                    }
                }
            }
        }
        fast_loss = min(max(fast_loss, 0.0), proposal.x);
        slow_loss = min(max(slow_loss, 0.0), proposal.y);
        var new_temperature = chem[p].z;
        if (my_capacity > 0.0) {
            new_temperature += heat_gain / my_capacity;
        }
        new_temperature = ambient_temperature_step(new_temperature);
        chem[p] = vec4<f32>(
            max(chem[p].x - fast_loss, 0.0),
            max(chem[p].y - slow_loss, 0.0),
            new_temperature,
            chem[p].w,
        );
    }
}

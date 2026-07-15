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
// Structural φ_f = 1 − φ_s. Retention then defines the mobile phase fraction
// φ_m=(P-H)/V_visible; the explicit overflow branch uses M/(M+V_s+H). Both fields are local and
// kernel-smoothed — a configured global porosity scalar is the rejected anti-pattern.
//
// ================================ DRAG FOLD (KTD-3, exact forms) ==============================
// The drag rate β(φ) comes from Kozeny-Carman (mirrors models::permeability — same clamps,
// same 150 coefficient, β = φ_m*drag_scale/(K_structural*k_r)), blended toward a Wen-Yu-style dilute rate below
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
// In dynamic mode beta is the mobile-fluid-side relaxation rate. For mobile mass m_m and
// dry+held composite mass m_c the relative rate is gamma = beta*(1 + m_m/m_c). After both phases receive their non-drag force
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
// −(V_s+H)·∇p force on the fixed composite plus the (1−ς) fraction of the mobile-column pressure
// force transmitted through the drag during the fold). At hydrostatic equilibrium the three
// terms net to exactly the displaced-volume weight — the buoyancy gate. Dynamic mode applies J
// to grid_svel immediately; react.xyz may expose that already-applied impulse for diagnostics,
// but project never substitutes ledger bookkeeping for a solid velocity update.
//
// EXACT-ZERO PASSTHROUGH (free-water regression gate): every coupling kernel gates on the
// solid-mass census (decoded grid_sfp = 0), not on β → 0 — zero-solid nodes take the exact
// algebraic free-water path (no exp(0) rounding in the loop).
//
// This pass also owns the grid forces + grid-node boundary conditions (moved verbatim from
// U2's grid_update, which now only decodes): the fold must integrate the gravity source
// itself, and the BCs must come after all velocity updates. Pipeline order (the U6 HTD):
//   grid_clear → p2g_water → p2g_solid_reference → p2g_solid[_dyn] → grid_update(decode)
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
fn drag_rate_blended(phi_s: f32, structural_phi_f: f32, mobile_phi_f: f32) -> f32 {
    let d = params.coupling.x;
    let scale = params.coupling.y;
    // v_m is intrinsic mobile-pore velocity while Darcy's q is superficial:
    // q=phi_m*(v_m-v_c). K remains a property of the structural pore geometry, so
    // beta=phi_m*nu/K makes the steady fold recover q=K*kr*F/nu after beta is divided by kr.
    // Scaling the dilute arm by the same available-pore ratio keeps the blend continuous as a
    // wetting front closes; it is exactly one in the capacity-zero path.
    let structural_phi = clamp(structural_phi_f, NUMERICAL_MOBILITY_MIN, 1.0);
    let available_ratio = clamp(mobile_phi_f / structural_phi, 0.0, 1.0);
    let packed = scale * mobile_phi_f
        / max(kozeny_carman(d, structural_phi_f), 1.0e-9);
    let dilute = scale * WEN_YU_COEFF * phi_s / max(d * d, 1.0e-8) * available_ratio;
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

// Runtime capillary closure for the ONE canonical water field. P is resolved pore volume, C is
// the quadrature-invariant target capacity scattered by the solid points, H=min(W,C) is actual
// capillary-held water, and M=W-H is mobile water. No solvent, solute, or enthalpy leaves the
// particle field: theta=M/W homogenizes its velocity response, while the exact M=0 branches in
// drag_fold/node_setup pin the held inventory without introducing an untransported scalar phase.
// Se is the mobile saturation in the remaining pore P-H (which equals P-C whenever M>0).
struct CapillaryNodeState {
    pore_volume: f32,
    capacity_volume: f32,
    held_volume: f32,
    mobile_volume: f32,
    mobile_pore_volume: f32,
    theta: f32,
    effective_saturation: f32,
};

fn capillary_node_state(n: u32, phi_f: f32, visible_volume: f32) -> CapillaryNodeState {
    let pore_volume = max(phi_f, 0.0) * max(visible_volume, 0.0);
    if (pore_volume <= 0.0 || params.extra.x <= 0.0) {
        return CapillaryNodeState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let water_volume = max(carrier_mass_at(n), 0.0) / params.extra.x;
    let capacity_volume = min(max(capillary_capacity_at(n), 0.0), pore_volume);
    let held_volume = min(water_volume, capacity_volume);
    let mobile_volume = max(water_volume - held_volume, 0.0);
    let mobile_pore_volume = max(pore_volume - held_volume, 0.0);
    var theta = 0.0;
    var effective_saturation = 0.0;
    if (mobile_volume > 0.0 && water_volume > 0.0) {
        theta = clamp(mobile_volume / water_volume, 0.0, 1.0);
        if (mobile_pore_volume > 0.0) {
            effective_saturation = clamp(mobile_volume / mobile_pore_volume, 0.0, 1.0);
        } else {
            // Capacity filled the resolved pore; excess carrier is free, fully saturated overflow.
            effective_saturation = 1.0;
        }
    }
    return CapillaryNodeState(
        pore_volume,
        capacity_volume,
        held_volume,
        mobile_volume,
        mobile_pore_volume,
        theta,
        effective_saturation,
    );
}

fn capillary_mobile_fraction(state: CapillaryNodeState, visible_volume: f32) -> f32 {
    if (state.mobile_volume <= 0.0 || visible_volume <= 0.0) { return 0.0; }
    if (state.mobile_pore_volume > 0.0) {
        return clamp(state.mobile_pore_volume / visible_volume, 0.0, 1.0);
    }
    // Explicit overflow partition: H=P and M>0 leaves no geometric pore denominator. Normalize
    // the mobile excess against itself plus the occupied solid+held composite (Vs+H), rather than
    // producing a zero-volume singularity or clamping M/V to a disappearing composite. At
    // W=C=P, M is exactly zero and the earlier branch remains exactly stiff.
    let composite_volume = max(visible_volume - state.pore_volume + state.held_volume, 0.0);
    return clamp(
        state.mobile_volume / max(state.mobile_volume + composite_volume, NUMERICAL_MOBILITY_MIN),
        0.0,
        1.0,
    );
}

// node_setup writes measured held carrier mass into its dedicated suffix lane. Taking the ratio
// in integer-derived mass units avoids the C -> H -> carrier round trip that could create a fake
// one-count mobile phase at an exactly held node.
fn measured_mobile_fraction(n: u32) -> f32 {
    let carrier_mass = max(carrier_mass_at(n), 0.0);
    if (carrier_mass <= 0.0) { return 0.0; }
    let held_mass = min(max(held_carrier_mass_at(n), 0.0), carrier_mass);
    return clamp((carrier_mass - held_mass) / carrier_mass, 0.0, 1.0);
}

fn store_measured_held_carrier(n: u32, state: CapillaryNodeState) -> f32 {
    let carrier_count = atomicLoad(&cell_cnt[carrier_mass_index(n)]);
    var held_count = u32(max(fp_encode(state.held_volume * params.extra.x), 0));
    if (state.mobile_volume <= 0.0) {
        held_count = carrier_count;
    } else {
        held_count = min(held_count, carrier_count);
    }
    atomicStore(&cell_cnt[held_carrier_mass_index(n)], held_count);
    if (carrier_count == 0u) { return 0.0; }
    return f32(carrier_count - held_count) / f32(carrier_count);
}

fn capillary_bloom_contact(n: u32, solid_volume: f32) -> f32 {
    if (solid_volume <= 0.0) { return 0.0; }
    return clamp(bloom_volume_at(n) / solid_volume, 0.0, 1.0);
}

// Intrinsic mobile-water relative permeability is kr=max(Se^n,floor). The configured floor
// regularizes only a genuinely mobile front; it cannot make held-only water creep. Pressure sees
// the homogenized mobility theta*kr, but drag acts on reconstructed intrinsic v_mobile and must
// divide beta by kr ONLY. Dividing by theta*kr here and then weighting the carrier by theta again
// would suppress steady Darcy flux by an erroneous second factor of theta.
const NUMERICAL_MOBILITY_MIN: f32 = 1.0e-6;
fn capillary_relative_permeability(state: CapillaryNodeState) -> f32 {
    if (state.mobile_volume <= 0.0 || state.theta <= 0.0) {
        return 0.0;
    }
    let floor = clamp(params.retention.z, NUMERICAL_MOBILITY_MIN, 1.0);
    if (floor >= 1.0) {
        return 1.0;
    }
    let exponent = max(params.retention.y, NUMERICAL_MOBILITY_MIN);
    return clamp(max(pow(state.effective_saturation, exponent), floor), 0.0, 1.0);
}

fn effective_capillary_mobility(state: CapillaryNodeState) -> f32 {
    return state.theta * capillary_relative_permeability(state);
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
    return capillary_node_state(neighbor, phi_f, visible_volume).effective_saturation;
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

fn grain_remaining_solute_mass(p: u32) -> f32 {
    return max(chem[p].x, 0.0) + max(chem[p].y, 0.0);
}

fn grain_effective_mass(p: u32) -> f32 {
    return params.solute.x + grain_remaining_solute_mass(p)
        + grain_bound_water_mass(pos[p].w);
}

fn grain_heat_capacity(p: u32) -> f32 {
    return params.solute.x * params.therm.y
        + grain_remaining_solute_mass(p) * params.solute.y
        + grain_bound_water_mass(pos[p].w) * params.therm.x;
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
    let total_structure = fp_encode(vg);
    let total_capacity = fp_encode(max(params.solute.w, 0.0));
    let total_bloom_volume = fp_encode(vg * bloom_factor(chem[p].w));
    var allocated_structure = 0i;
    var allocated_capacity = 0i;
    var allocated_bloom_volume = 0i;
    var cumulative_weight = 0.0;
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i2 = 0; i2 < 3; i2 = i2 + 1) {
                let wijk = w[i2].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i2, j, k);
                let n = node_index(node);
                cumulative_weight += wijk;
                let is_last = i2 == 2 && j == 2 && k == 2;
                let structure_target = extensive_partition_target(
                    total_structure, cumulative_weight, is_last,
                );
                let capacity_target = extensive_partition_target(
                    total_capacity, cumulative_weight, is_last,
                );
                let bloom_target = extensive_partition_target(
                    total_bloom_volume, cumulative_weight, is_last,
                );
                atomicAdd(&grid_sfp[n], structure_target - allocated_structure);
                atomicAdd(
                    &cell_cnt[capillary_capacity_index(n)],
                    u32(max(capacity_target - allocated_capacity, 0)),
                );
                atomicAdd(
                    &cell_cnt[bloom_volume_index(n)],
                    u32(max(bloom_target - allocated_bloom_volume, 0)),
                );
                allocated_structure = structure_target;
                allocated_capacity = capacity_target;
                allocated_bloom_volume = bloom_target;
            }
        }
    }
}

// Decode the frame-start dry-skeleton APIC proposal preserved before solid_update/implicit shear
// applies stress, body, compaction, and contact impulses. Partial-retention reconstruction must
// use this time-level consistently: grid_vel is still the homogenized water proposal, whereas
// grid_svel is already force-updated.
fn preforce_solid_velocity(n: u32) -> vec3<f32> {
    let mass_count = atomicLoad(&grid_sraw[4u * n]);
    if (fp_decode(mass_count) <= params.extra.z || mass_count == 0) {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(
        f32(atomicLoad(&grid_sraw[4u * n + 1u])),
        f32(atomicLoad(&grid_sraw[4u * n + 2u])),
        f32(atomicLoad(&grid_sraw[4u * n + 3u])),
    ) / f32(mass_count);
}

fn mobile_water_velocity_bc(xp: vec3<f32>, velocity: vec3<f32>, open_base: bool) -> vec3<f32> {
    // Use the identical hard projector assembled for node_setup/project, but apply it to the
    // intrinsic MOBILE phase. Projecting v_bar would also damp the capillary-held share and break
    // v_bar=theta*v_m+(1-theta)*v_c at contacts.
    var bounded = minv_apply(phase_hard_projector(xp, PHASE_WATER, open_base), velocity);
    let water_cap = select(params.max_speed, params.flip.w, params.flip.w > 0.0);
    let speed = length(bounded);
    if (speed > water_cap) {
        bounded *= water_cap / speed;
    }
    return bounded;
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
    let gravity_accel = params.gravity.xyz;
    var mobile_external_accel = gravity_accel;
    let c = node_coords(n);
    let xp = params.grid_origin.xyz + vec3<f32>(f32(c.x), f32(c.y), f32(c.z)) * params.grid_origin.w;
    let open_base = params.coupling.w > 0.5;
    let sv = solid_volume_at(n);
    var phi_f = 1.0;
    var mobile_phi_f = 1.0;
    var measured_theta = 1.0;
    var capillary = CapillaryNodeState(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);
    if (sv > 0.0) {
        let visible_volume = node_visible_volume(xp);
        phi_f = phi_f_from_solid_and_visible_volume(sv, visible_volume);
        capillary = capillary_node_state(n, phi_f, visible_volume);
        measured_theta = store_measured_held_carrier(n, capillary);
        mobile_phi_f = capillary_mobile_fraction(capillary, visible_volume);
        if (measured_theta <= 0.0) { mobile_phi_f = 0.0; }

        // The calibrated Green-Ampt magnitude now follows the local saturation-gradient normal,
        // not gravity. It is evaluated only inside the bed; missing/non-bed neighbors receive a
        // zero-normal-gradient boundary value. Suction is independently gated by the local,
        // structural-volume-weighted bloom contact; a retention target of zero cannot disable it.
        // Adding it only to mobile forcing keeps the forced exponential drag integration and
        // pressure mobility sigma algebraically unchanged without accelerating held water.
        let a_suction = params.wet1.x;
        let bloom_contact = capillary_bloom_contact(n, sv);
        if (a_suction > 0.0 && bloom_contact > 0.0) {
            mobile_external_accel += bloom_contact * capillary_front_acceleration(
                vec3<i32>(c), capillary.effective_saturation, a_suction,
            );
        }
    }
    // Thin-sheet paper split. The hard-wall projector first removes geometrically forbidden
    // components from the paper normal. Normal body force is then owned exclusively by the exact
    // paper solve below; tangential force continues through the ordinary bed/free-water path.
    // Scaling the normal source by alpha=L/delta makes the terminal gravity flux k*g/nu
    // independent of the numerical band thickness.
    let paper = filter_paper_response(xp);
    var paper_normal = vec3<f32>(0.0);
    var paper_source = 0.0;
    if (paper.coverage > 0.0 && (sv <= 0.0 || capillary.mobile_volume > 0.0)) {
        let hard = phase_hard_projector(xp, PHASE_WATER, open_base);
        let allowed = minv_apply(hard, paper.normal);
        let allowed_len = length(allowed);
        if (allowed_len > SDF_NORMAL_MIN) {
            paper_normal = allowed / allowed_len;
            let normal_force = dot(mobile_external_accel, paper_normal);
            paper_source = paper.alpha * normal_force;
            // The paper ODE owns superficial normal flux. The ordinary drag/body step advances
            // only the tangential intrinsic mobile velocity, then the q solve below restores the
            // normal component without ever touching held water.
            mobile_external_accel -= normal_force * paper_normal;
        }
    }
    var sig = 1.0;
    var imp = vec3<f32>(0.0);
    var solid_v = grid_svel[n].xyz;
    let solid_mass = grid_svel[n].w;
    var dynamic_pair = false;
    var theta_node = 1.0;
    if (sv > 0.0) {
        let phi_s = 1.0 - phi_f;
        let theta = measured_theta;
        theta_node = theta;
        let mobile_mass = mass * theta;
        let held_mass = mass - mobile_mass;
        let composite_mass = solid_mass + held_mass;
        // Relative permeability modifies beta while mobile water exists. theta is already present
        // in mobile_mass and v_bar; including it in beta would count retention twice. Held-only
        // water is an explicit infinite-drag branch, so no numerical floor can make M=0 creep.
        let relative_permeability = capillary_relative_permeability(capillary);
        if (theta <= 0.0 || relative_permeability <= 0.0) {
            let fluid_forced = v + gravity_accel * params.dt;
            sig = 0.0;
            if (params.splas0.x > 0.5 && solid_mass > params.extra.z) {
                // All solution mass is held. Merge its independently forced momentum with the
                // dry-solid proposal in the exact barycentric stiff limit.
                let reduced_mass = mass * solid_mass / (mass + solid_mass);
                imp = reduced_mass * (fluid_forced - solid_v);
                v = fluid_forced - imp / mass;
                solid_v = solid_v + imp / solid_mass;
                dynamic_pair = true;
            } else {
                // A structural-volume node without decoded dynamic mass is a kinematically fixed
                // support, not free water. This keeps M=0 exactly stiff even at a quantized bed
                // fringe instead of letting the numerical mobility floor create creep.
                imp = mass * fluid_forced;
                v = vec3<f32>(0.0);
            }
        } else if (params.splas0.x > 0.5
            && solid_mass > params.extra.z
            && composite_mass > params.extra.z) {
            let beta = drag_rate_blended(phi_s, phi_f, mobile_phi_f)
                / relative_permeability;
            // Momentum staging at one consistent time level:
            //   p_m* = m_m v_m^n + I_m,
            //   p_c* = m_c v_s^n + I_s + I_h.
            // grid_sraw supplies v_s^n before solid forces; grid_svel supplies the dry
            // proposal after them. Dividing its dry impulse by m_c gives held water real inertia
            // instead of letting the skeleton accelerate as if H had no mass.
            let solid_preforce = preforce_solid_velocity(n);
            var mobile_preforce = v;
            if (theta < 1.0) {
                mobile_preforce = (v - (1.0 - theta) * solid_preforce) / theta;
            }
            let mobile_forced = mobile_preforce + mobile_external_accel * params.dt;
            let dry_impulse = solid_mass * (solid_v - solid_preforce);
            let held_body_impulse = held_mass * gravity_accel * params.dt;
            solid_v = solid_preforce
                + (dry_impulse + held_body_impulse) / composite_mass;
            let gamma = beta * (1.0 + mobile_mass / composite_mass);
            let a = gamma * params.dt;
            var one_minus_e: f32;
            if (a < 1.0e-3) {
                one_minus_e = a * (1.0 - 0.5 * a + a * a / 6.0);
                sig = 1.0 - 0.5 * a + a * a / 6.0;
            } else {
                one_minus_e = 1.0 - exp(-a);
                sig = one_minus_e / a;
            }
            let reduced_mass = mobile_mass * composite_mass / (mobile_mass + composite_mass);
            imp = reduced_mass * one_minus_e * (mobile_forced - solid_v);
            let mobile_after = mobile_forced - imp / mobile_mass;
            solid_v = solid_v + imp / composite_mass;
            v = theta * mobile_after + (1.0 - theta) * solid_v;
            dynamic_pair = true;
        } else {
            // Frozen mode keeps the forced exponential integrator exactly. A structural-volume
            // node with no independently resolved dry mass is also a fixed bed fringe, not a
            // held-only dynamic composite.
            let beta = drag_rate_blended(phi_s, phi_f, mobile_phi_f)
                / relative_permeability;
            let a = beta * params.dt;
            let e = exp(-a);
            var srcf: f32;
            if (a < 1.0e-3) {
                srcf = params.dt * (1.0 - 0.5 * a);
            } else {
                srcf = (1.0 - e) / beta;
            }
            let mobile_v = v / theta;
            let mobile_after = mobile_v * e + mobile_external_accel * srcf;
            // Held water receives gravity before its exact fixed-support constraint; mobile-only
            // suction and the ordinary (non-paper-normal) mobile force are theta weighted.
            let naive = v + (
                theta * mobile_external_accel + (1.0 - theta) * gravity_accel
            ) * params.dt;
            v = theta * mobile_after;
            sig = srcf / params.dt;
            imp = mass * (naive - v);
        }
    } else {
        v += mobile_external_accel * params.dt;
    }

    if (dynamic_pair) {
        // Drag can reintroduce an approaching contact component after solid_update. Reapply the
        // identical unilateral Coulomb BC at actual contact only; separating motion survives and
        // grains outside the water pressure band are untouched. Recompose the held share after
        // this external phase impulse; changing v_s without (1-theta)*delta-v_s would corrupt the
        // single-water-field identity.
        let solid_before_bc = solid_v;
        solid_v = solid_contact_velocity_bc(xp, solid_before_bc);
        v += (1.0 - theta_node) * (solid_v - solid_before_bc);
        grid_svel[n] = vec4<f32>(solid_v, solid_mass);
    }

    // Recover intrinsic mobile velocity, advance the paper NORMAL equation in superficial flux
    // q=phi_m*v_m, then apply water contacts to v_m only. Held water never sees paper damping or
    // water-wall clipping; it follows the composite skeleton velocity by construction.
    var mobile_v = solid_v;
    if (theta_node > 0.0) {
        if (theta_node < 1.0) {
            mobile_v = (v - (1.0 - theta_node) * solid_v) / theta_node;
        } else {
            mobile_v = v;
        }
        if (length(paper_normal) > SDF_NORMAL_MIN && mobile_phi_f > NUMERICAL_MOBILITY_MIN) {
            let old_flux = mobile_phi_f * dot(mobile_v, paper_normal);
            let new_flux = paper.decay * old_flux
                + paper.sigma * params.dt * paper_source;
            mobile_v += ((new_flux - old_flux) / mobile_phi_f) * paper_normal;
        }
        mobile_v = mobile_water_velocity_bc(xp, mobile_v, open_base);
        v = theta_node * mobile_v + (1.0 - theta_node) * solid_v;
    } else {
        v = solid_v;
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
    if (contact_t <= 0.0) { return 0.0; }
    if (bd <= 0.0) { return 1.0; }
    let t = clamp(contact_t / bd, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t); // smoothstep
}

// ================================ FILTER RESERVOIR RELEASE ===================================
// Stored parcels preserve solvent, solute, and temperature exactly while their Darcy pore-transit
// clock advances. Release is a separate pass before g2p_water, so freed capacity is visible to all
// admissions in the later dispatch. chem.z=2 is a one-dispatch handoff marker: g2p clears it while
// preserving the release position/velocity instead of gathering from a grid the parcel did not
// contribute to this frame.
struct PaperFluxSample {
    superficial_flux: vec3<f32>,
    mobile_carrier_mass: f32,
};

// Gather post-projection WATER superficial flux q_w=phi_m*v_m at a held parcel's paper location.
// Never gather raw v_bar and never add a solid volumetric term: the paper reservoir advances water
// carrier transit relative to stationary paper, so skeleton motion alone cannot release it. The
// carrier census distinguishes a real zero-flux pressure balance from the special case where the
// held parcel is the only local water and did not contribute to this frame's grid.
fn paper_flux_sample(x: vec3<f32>) -> PaperFluxSample {
    let h = params.grid_origin.w;
    let xl = (x - params.grid_origin.xyz) / h;
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let w = bspline_w(xl - vec3<f32>(base));
    var superficial_flux = vec3<f32>(0.0);
    var mobile_carrier_mass = 0.0;
    for (var k = 0; k < 3; k = k + 1) {
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let weight = w[i].x * w[j].y * w[k].z;
                let node = base + vec3<i32>(i, j, k);
                let n = node_index(node);
                let sv = solid_volume_at(n);
                if (sv <= 0.0) {
                    superficial_flux += weight * grid_vel[n].xyz;
                    mobile_carrier_mass += weight * carrier_mass_at(n);
                } else {
                    let xp = params.grid_origin.xyz + vec3<f32>(node) * h;
                    let visible = node_visible_volume(xp);
                    let structural_phi_f = phi_f_from_solid_and_visible_volume(sv, visible);
                    let state = capillary_node_state(n, structural_phi_f, visible);
                    let theta = measured_mobile_fraction(n);
                    let mobile_phi_f = capillary_mobile_fraction(state, visible);
                    if (theta > 0.0 && mobile_phi_f > 0.0) {
                        var solid_velocity = vec3<f32>(0.0);
                        if (params.splas0.x > 0.5 && grid_svel[n].w > params.extra.z) {
                            solid_velocity = grid_svel[n].xyz;
                        }
                        let mobile_velocity = (grid_vel[n].xyz
                            - (1.0 - theta) * solid_velocity) / theta;
                        superficial_flux += weight * mobile_phi_f * mobile_velocity;
                    }
                    mobile_carrier_mass += weight * max(
                        carrier_mass_at(n) - held_carrier_mass_at(n),
                        0.0,
                    );
                }
            }
        }
    }
    return PaperFluxSample(superficial_flux, mobile_carrier_mass);
}

@compute @workgroup_size(256)
fn filter_release(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count || params.filter1.x == 0u) {
        return;
    }
    let state = chem[p];
    if (state.z < 0.5 || state.z > 1.5) {
        return;
    }
    let hit = filter_paper_hit(pos[p].xyz);
    let downstream = -sdf_normalize3(hit.grad);
    let permeability = params.filter0.x;
    let viscosity = params.filter0.y;
    let thickness = params.filter0.z;
    let porosity = params.filter0.w;
    if (length(downstream) <= SDF_EPS || permeability <= 0.0 || viscosity <= 0.0
        || thickness <= 0.0 || porosity <= 0.0) {
        atomicOr(&filter_state[FILTER_ERRORS], 1u);
        return;
    }
    let sample = paper_flux_sample(pos[p].xyz);
    var superficial_flux = max(dot(sample.superficial_flux, downstream), 0.0);
    if (sample.mobile_carrier_mass <= params.extra.z) {
        // A held parcel does not scatter. If it is the only local carrier there is no projected
        // grid velocity to gather, so advance it by the gravity-only Darcy limit. Never use this
        // fallback when mobile water is present: zero then means pressure really balanced gravity.
        superficial_flux = (permeability / viscosity)
            * max(dot(params.gravity.xyz, downstream), 0.0);
    }
    let progress = state.w + superficial_flux * params.dt / (porosity * thickness);
    if (progress < 1.0) {
        chem[p] = vec4<f32>(state.xyz, clamp(progress, 0.0, 1.0));
        return;
    }

    let parcel_counts = filter_volume_encode(clamp(pos[p].w, 0.0, 1.0) * wet_v_water());
    let old_occupied = atomicSub(&filter_state[FILTER_OCCUPIED], parcel_counts);
    if (old_occupied < parcel_counts) {
        atomicStore(&filter_state[FILTER_OCCUPIED], 0u);
        atomicOr(&filter_state[FILTER_ERRORS], 2u);
    }
    atomicAdd(&filter_state[FILTER_RELEASED], 1u);
    let offset = max(thickness, 2.0 * SDF_EPS * params.grid_origin.w);
    pos[p] = vec4<f32>(pos[p].xyz + downstream * offset, pos[p].w);
    // Released free water leaves at the superficial flux represented by the solver grid. Dividing
    // by paper porosity here would inject an unphysical velocity jump at the phase handoff.
    vel[p] = vec4<f32>(downstream * superficial_flux, vel[p].w);
    cmat[3u * p + 0u] = vec4<f32>(0.0);
    cmat[3u * p + 1u] = vec4<f32>(0.0);
    cmat[3u * p + 2u] = vec4<f32>(0.0);
    chem[p] = vec4<f32>(state.xy, 2.0, 0.0);
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
    if (is_water && chem[p].z > 0.5) {
        return; // paper-held solvent is not available to the bed absorption ledger
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
    if (is_water && chem[p].z > 0.5) {
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
            let old_mass = grain_effective_mass(p);
            let gained_mass = water_mass_per_volume * gain;
            let new_mass = old_mass + gained_mass;
            let new_velocity = (
                old_mass * vel[p].xyz
                + water_mass_per_volume * incoming_volume_momentum
            ) / new_mass;
            vel[p] = vec4<f32>(new_velocity, vel[p].w);

            // Bound water carries its donor temperature into the grain. This is the thermal
            // analogue of the momentum merge and prevents phase change from creating enthalpy.
            let old_capacity = grain_heat_capacity(p);
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
// freezes eleven extensive node lanes; the gather pass allocates one node transfer to both sides.
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

fn water_heat_capacity(f_water: f32, concentration: f32) -> f32 {
    return water_solvent_mass(f_water) * max(params.therm.x, 0.0)
        + water_solute_mass(f_water, concentration) * max(params.solute.y, 0.0);
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

struct ChemistryNodeState {
    headroom: f32,
    proposed: f32,
    accepted_mass: f32,
    accepted_momentum: vec3<f32>,
    accepted_solute_enthalpy: f32,
    water_capacity: f32,
    grain_capacity: f32,
    heat_to_water: f32,
};

fn chemistry_node_state(offset: u32) -> ChemistryNodeState {
    let headroom = chem_decode(
        CHEM_HEADROOM, atomicLoad(&grid_chem[offset + CHEM_HEADROOM]),
    );
    let fast = chem_decode(CHEM_FAST, atomicLoad(&grid_chem[offset + CHEM_FAST]));
    let slow = chem_decode(CHEM_SLOW, atomicLoad(&grid_chem[offset + CHEM_SLOW]));
    let proposed = fast + slow;
    let accepted_mass = min(headroom, proposed);
    var accepted_fraction = 0.0;
    if (proposed > 0.0) { accepted_fraction = accepted_mass / proposed; }
    let accepted_momentum = accepted_fraction * vec3<f32>(
        chem_decode(CHEM_JX + 0u, atomicLoad(&grid_chem[offset + CHEM_JX + 0u])),
        chem_decode(CHEM_JX + 1u, atomicLoad(&grid_chem[offset + CHEM_JX + 1u])),
        chem_decode(CHEM_JX + 2u, atomicLoad(&grid_chem[offset + CHEM_JX + 2u])),
    );
    let accepted_solute_enthalpy = accepted_fraction * chem_decode(
        CHEM_U, atomicLoad(&grid_chem[offset + CHEM_U]),
    );
    let old_cw = chem_decode(CHEM_CW, atomicLoad(&grid_chem[offset + CHEM_CW]));
    let old_ew = chem_decode(CHEM_EW, atomicLoad(&grid_chem[offset + CHEM_EW]));
    let old_cg = chem_decode(CHEM_CG, atomicLoad(&grid_chem[offset + CHEM_CG]));
    let old_eg = chem_decode(CHEM_EG, atomicLoad(&grid_chem[offset + CHEM_EG]));
    let transferred_capacity = max(params.solute.y, 0.0) * accepted_mass;
    let water_capacity = old_cw + transferred_capacity;
    let grain_capacity = max(old_cg - transferred_capacity, 0.0);
    let water_enthalpy = old_ew + accepted_solute_enthalpy;
    let grain_enthalpy = old_eg - accepted_solute_enthalpy;
    let heat_to_water = node_heat_transfer(
        water_capacity, water_enthalpy, grain_capacity, grain_enthalpy,
    );
    return ChemistryNodeState(
        headroom,
        proposed,
        accepted_mass,
        accepted_momentum,
        accepted_solute_enthalpy,
        water_capacity,
        grain_capacity,
        heat_to_water,
    );
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
    if (is_water && chem[p].z > 0.5) {
        return; // stored paper inventory does not exchange with the bed this frame
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
    var proposal_momentum = vec3<f32>(0.0);
    var proposal_solute_enthalpy = 0.0;
    if (is_water) {
        let solvent_mass = water_solvent_mass(pos[p].w);
        headroom = max(params.chem1.x - chem[p].x, 0.0) * solvent_mass;
        capacity = water_heat_capacity(pos[p].w, chem[p].x);
        enthalpy = capacity * chem[p].y;
    } else {
        release = grain_release_proposal(p);
        let proposed_mass = release.x + release.y;
        proposal_momentum = proposed_mass * vel[p].xyz;
        proposal_solute_enthalpy = proposed_mass * params.solute.y * chem[p].z;
        capacity = grain_heat_capacity(p);
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
    let total_momentum = vec3<i32>(
        chem_encode(CHEM_JX + 0u, proposal_momentum.x),
        chem_encode(CHEM_JX + 1u, proposal_momentum.y),
        chem_encode(CHEM_JX + 2u, proposal_momentum.z),
    );
    let total_solute_enthalpy = chem_encode(CHEM_U, proposal_solute_enthalpy);
    var allocated_headroom = 0i;
    var allocated_fast = 0i;
    var allocated_slow = 0i;
    var allocated_capacity = 0i;
    var allocated_enthalpy = 0i;
    var allocated_momentum = vec3<i32>(0);
    var allocated_solute_enthalpy = 0i;
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
                    for (var axis = 0; axis < 3; axis = axis + 1) {
                        let momentum_target = extensive_partition_target(
                            total_momentum[axis], cumulative_weight, is_last,
                        );
                        atomicAdd(
                            &grid_chem[offset + CHEM_JX + u32(axis)],
                            momentum_target - allocated_momentum[axis],
                        );
                        allocated_momentum[axis] = momentum_target;
                    }
                    let solute_enthalpy_target = extensive_partition_target(
                        total_solute_enthalpy, cumulative_weight, is_last,
                    );
                    atomicAdd(
                        &grid_chem[offset + CHEM_U],
                        solute_enthalpy_target - allocated_solute_enthalpy,
                    );
                    allocated_solute_enthalpy = solute_enthalpy_target;
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
    if (is_water && chem[p].z > 0.5) {
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
        let solvent_mass = water_solvent_mass(fraction);
        let my_headroom = max(params.chem1.x - chem[p].x, 0.0) * solvent_mass;
        let my_capacity = water_heat_capacity(fraction, chem[p].x);
        let headroom_total = chem_encode(CHEM_HEADROOM, my_headroom);
        let capacity_total = chem_encode(CHEM_CW, my_capacity);
        var allocated_headroom = 0i;
        var allocated_capacity = 0i;
        var cumulative_weight = 0.0;
        var solute_gain = 0.0;
        var momentum_gain = vec3<f32>(0.0);
        var solute_enthalpy_gain = 0.0;
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
                    let node_state = chemistry_node_state(offset);
                    var node_gain = 0.0;
                    if (node_state.headroom > 0.0) {
                        let receiver_fraction = headroom_share / node_state.headroom;
                        node_gain = receiver_fraction * node_state.accepted_mass;
                        solute_gain += node_gain;
                        momentum_gain += receiver_fraction * node_state.accepted_momentum;
                        solute_enthalpy_gain += receiver_fraction
                            * node_state.accepted_solute_enthalpy;
                    }
                    if (node_state.water_capacity > 0.0) {
                        let post_capacity_share = capacity_share
                            + params.solute.y * node_gain;
                        heat_gain += post_capacity_share / node_state.water_capacity
                            * node_state.heat_to_water;
                    }
                }
            }
        }
        // The analytic gather cannot exceed my_headroom. Clamp only the FP-rounding residue so
        // c<=c_sat is structural, rather than relying on an after-the-fact concentration clip.
        let raw_solute_gain = solute_gain;
        solute_gain = min(max(solute_gain, 0.0), my_headroom);
        if (raw_solute_gain > 0.0 && solute_gain != raw_solute_gain) {
            let correction = solute_gain / raw_solute_gain;
            momentum_gain *= correction;
            solute_enthalpy_gain *= correction;
        }
        let new_capacity = my_capacity + params.solute.y * solute_gain;
        var new_temperature = chem[p].y;
        if (new_capacity > 0.0) {
            new_temperature = (
                my_capacity * chem[p].y + solute_enthalpy_gain + heat_gain
            ) / new_capacity;
        }
        let pre_ambient_temperature = new_temperature;
        new_temperature = ambient_temperature_step(new_temperature);
        ambient_enthalpy[p] += new_capacity * (new_temperature - pre_ambient_temperature);
        var new_concentration = chem[p].x;
        if (solvent_mass > 0.0) {
            new_concentration = (chem[p].x * solvent_mass + solute_gain) / solvent_mass;
        }
        chem[p] = vec4<f32>(new_concentration, new_temperature, chem[p].zw);
        vel[p] = vec4<f32>(vel[p].xyz, solute_gain);
        cmat[3u * p + 0u] = vec4<f32>(cmat[3u * p + 0u].xyz, momentum_gain.x);
        cmat[3u * p + 1u] = vec4<f32>(cmat[3u * p + 1u].xyz, momentum_gain.y);
        cmat[3u * p + 2u] = vec4<f32>(cmat[3u * p + 2u].xyz, momentum_gain.z);
    } else {
        let proposal = grain_release_proposal(p);
        let my_capacity = grain_heat_capacity(p);
        let fast_total = chem_encode(CHEM_FAST, proposal.x);
        let slow_total = chem_encode(CHEM_SLOW, proposal.y);
        let capacity_total = chem_encode(CHEM_CG, my_capacity);
        let proposed_mass = proposal.x + proposal.y;
        let momentum_total = vec3<i32>(
            chem_encode(CHEM_JX + 0u, proposed_mass * vel[p].x),
            chem_encode(CHEM_JX + 1u, proposed_mass * vel[p].y),
            chem_encode(CHEM_JX + 2u, proposed_mass * vel[p].z),
        );
        let solute_enthalpy_total = chem_encode(
            CHEM_U, proposed_mass * params.solute.y * chem[p].z,
        );
        var allocated_fast = 0i;
        var allocated_slow = 0i;
        var allocated_capacity = 0i;
        var allocated_momentum = vec3<i32>(0);
        var allocated_solute_enthalpy = 0i;
        var cumulative_weight = 0.0;
        var fast_loss = 0.0;
        var slow_loss = 0.0;
        var momentum_loss = vec3<f32>(0.0);
        var solute_enthalpy_loss = 0.0;
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
                    var momentum_target = vec3<i32>(0);
                    for (var axis = 0; axis < 3; axis = axis + 1) {
                        momentum_target[axis] = extensive_partition_target(
                            momentum_total[axis], cumulative_weight, is_last,
                        );
                    }
                    let solute_enthalpy_target = extensive_partition_target(
                        solute_enthalpy_total, cumulative_weight, is_last,
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
                    let momentum_share = vec3<f32>(
                        chem_decode(CHEM_JX + 0u, momentum_target.x - allocated_momentum.x),
                        chem_decode(CHEM_JX + 1u, momentum_target.y - allocated_momentum.y),
                        chem_decode(CHEM_JX + 2u, momentum_target.z - allocated_momentum.z),
                    );
                    let solute_enthalpy_share = chem_decode(
                        CHEM_U, solute_enthalpy_target - allocated_solute_enthalpy,
                    );
                    allocated_fast = fast_target;
                    allocated_slow = slow_target;
                    allocated_capacity = capacity_target;
                    allocated_momentum = momentum_target;
                    allocated_solute_enthalpy = solute_enthalpy_target;
                    let offset = node_index(base + vec3<i32>(i, j, k)) * CHEM_STRIDE;
                    let node_state = chemistry_node_state(offset);
                    var accepted_fraction = 0.0;
                    if (node_state.proposed > 0.0) {
                        accepted_fraction = node_state.accepted_mass / node_state.proposed;
                    }
                    let fast_loss_node = fast_share * accepted_fraction;
                    let slow_loss_node = slow_share * accepted_fraction;
                    let mass_loss_node = fast_loss_node + slow_loss_node;
                    fast_loss += fast_loss_node;
                    slow_loss += slow_loss_node;
                    momentum_loss += accepted_fraction * momentum_share;
                    solute_enthalpy_loss += accepted_fraction * solute_enthalpy_share;
                    if (node_state.grain_capacity > 0.0) {
                        let post_capacity_share = max(
                            capacity_share - params.solute.y * mass_loss_node, 0.0,
                        );
                        heat_gain -= post_capacity_share / node_state.grain_capacity
                            * node_state.heat_to_water;
                    }
                }
            }
        }
        let raw_total_loss = fast_loss + slow_loss;
        fast_loss = min(max(fast_loss, 0.0), proposal.x);
        slow_loss = min(max(slow_loss, 0.0), proposal.y);
        let total_mass_loss = fast_loss + slow_loss;
        if (raw_total_loss > 0.0 && total_mass_loss != raw_total_loss) {
            let correction = total_mass_loss / raw_total_loss;
            momentum_loss *= correction;
            solute_enthalpy_loss *= correction;
        }
        let new_capacity = max(my_capacity - params.solute.y * total_mass_loss, 0.0);
        var new_temperature = chem[p].z;
        if (new_capacity > 0.0) {
            new_temperature = (
                my_capacity * chem[p].z - solute_enthalpy_loss + heat_gain
            ) / new_capacity;
        }
        let pre_ambient_temperature = new_temperature;
        new_temperature = ambient_temperature_step(new_temperature);
        ambient_enthalpy[p] += new_capacity * (new_temperature - pre_ambient_temperature);
        chem[p] = vec4<f32>(
            max(chem[p].x - fast_loss, 0.0),
            max(chem[p].y - slow_loss, 0.0),
            new_temperature,
            chem[p].w,
        );
        vel[p] = vec4<f32>(vel[p].xyz, -total_mass_loss);
        cmat[3u * p + 0u] = vec4<f32>(cmat[3u * p + 0u].xyz, -momentum_loss.x);
        cmat[3u * p + 1u] = vec4<f32>(cmat[3u * p + 1u].xyz, -momentum_loss.y);
        cmat[3u * p + 2u] = vec4<f32>(cmat[3u * p + 2u].xyz, -momentum_loss.z);
    }
}

// Two-field solver shared core: the Params mirror, the global binding table, and the helpers
// every pass family shares (fixed-point encoding, B-spline weights, grid indexing, the SDF
// solid mirror). Later pass families (pressure / plasticity / coupling) are concatenated after
// this file and transfers.wgsl into one shader module (WGSL has no imports), so this file
// declares everything they share. Binding indices are GLOBAL and stable across the module:
// each pipeline's auto layout (`layout: None`) keeps only the bindings its entry point
// actually uses, but an index never means two different buffers.
//
// Storage-buffer budget (KTD-7): the widest entry point binds 8 storage buffers
// (MAX_STORAGE_BUFFERS_PER_ENTRY_POINT in mod.rs — cell_classify / pocket_mark; active tiled
// relaxation adds one scheduler binding but remains below them; see the per-pass derivation
// there); shared required limits request 10 per stage
// (`solvers/mpm/mod.rs::required_limits`), both within the 16 target cap. U1 adds NO binding (the flip knob is a Params lane, and
// g2p_water reads only the already-bound vel/grid_vel). Grid mass+momentum share ONE
// array<atomic<i32>> with stride 4 (mass, mom.xyz) rather than four arrays — one binding, one
// clear pass, contiguous per-node lanes.
//
// Tint discipline: any workgroupBarrier() must be reachable from uniform control flow — no
// early returns before a barrier (clamp indices and predicate the work instead). No pass in
// this module uses a barrier, so the per-thread guard returns are safe.

const PHASE_WATER: u32 = 0u;
const PHASE_SOLID: u32 = 1u;

// Byte-identical to the Rust `Params` (400 bytes; vec4-aligned tail).
struct Params {
    box_min: vec4<f32>,     // simulation domain (w unused)
    box_max: vec4<f32>,     // (w unused)
    gravity: vec4<f32>,     // (w unused)
    grid_origin: vec4<f32>, // .xyz = world position of node (0,0,0); .w = cell size h
    grid_dims: vec4<u32>,   // nx, ny, nz, num_nodes (= nx·ny·nz)
    dt: f32,
    particle_mass: f32,  // per water particle (Materials::particle_mass)
    max_speed: f32,      // velocity cap — COUPLED to FP_SCALE (see headroom math below)
    pic_blend: f32,      // APIC↔PIC blend = PIC fraction ∈ [0,1] (G2P scales the affine C by 1−blend)
    water_count: u32,    // particles [0, water_count) are water (KTD-1 range layout)
    solid_count: u32,    // particles [water_count, water_count + solid_count) are solid grains
    particle_count: u32, // = water_count + solid_count (kernel live-set guard)
    num_solids: u32,     // count of static SDF solids in the `solids` buffer (0 = none)
    // U3 pressure stack (pressure.wgsl):
    coarse_dims: vec4<u32>, // coarse CELLS per axis (= ceil(fine_cells/ratio)); .w = ratio
    extra: vec4<f32>,       // (rest_density, rho_floor, mass_eps, U7 wet-cohesion s_peak)
    // U6 coupling (coupling.wgsl): (solid sample pitch, drag_scale, dry structural sample
    // volume, open_base flag — the dev/test drained-column outflow mode).
    coupling: vec4<f32>,
    // U5 plasticity (plasticity.wgsl; mirrors twofield::plasticity constants):
    splas0: vec4<f32>, // (solid_dynamics flag, Lamé μ, Lamé λ, DP α)
    splas1: vec4<f32>, // (cap hardening ξ, φ_max, dry grain mass m_dry, cohesion y_c)
    splas2: vec4<f32>, // (floor/wall Coulomb μ_b, guard K_sp, guard onset φ_on, U7 wet c_max)
    // U9 infiltration interface (coupling.wgsl; rates and delays use physical seconds):
    wet0: vec4<f32>, // (tf_absorb_rate k_abs, V_cap from dry recipe mass, V_w = water vol,
                     //  absorb_roundoff — the f_w/V_bound saturation floor)
    wet1: vec4<f32>, // (suction body-force accel a_suction, bloom_delay seconds, filter_floor
                     //  flag, maximum fractional DIAMETER swelling at saturation)
    dbg: vec4<f32>,  // diagnostic toggles (test-only): .x = density-relief enable (1 = on);
                     //  .y = SDF wall-BC mode selector: ≤ 0.5 → binary no-penetration band
                     //  (default, byte-identical to pre-coverage); > 0.5 → graded coverage weight
                     //  (wall_coverage in pressure.wgsl — the embedded-boundary L1 path). (Formerly
                     //  a dropped relief dead-band; the settled-pool stirring fix is the G2P PIC
                     //  blend PIC_BLEND_DEFAULT.) .z = temper-K rate divisor for the uncapped path
                     //  (0 ⇒ K=1 full uncap; legacy cap 30; sweet spot K∈(1,30)). .w = uncapped two-sided
                     //  density-target mode (≤ 0.5 → legacy rate-limited two-sided relief, default
                     //  & byte-identical; > 0.5 → relief uncapped to full strength, driving ρ→ρ₀
                     //  — the over-pack fix; see cell_classify in pressure.wgsl / surface.wgsl).
    flip: vec4<f32>, // U1 surface-weighted dissipation knob (g2p_water, transfers.wgsl):
                     //  (c_surface, density_gate, div_scale, water_splash_cap). DISABLED default —
                     //  c_surface = 1.0 ⇒ v = v_grid = pure-PIC; div_scale ≤ 0 ⇒ merge discriminator
                     //  off; water_splash_cap ≤ 0 ⇒ fall back to the global max_speed. The U2 curve
                     //  reads these; plumbed-but-unused in U1 (default path stays pure-PIC).
    // Conservative extraction + thermal transport (coupling.wgsl):
    chem0: vec4<f32>, // (scaled k_fast, scaled k_slow, Ea/R in K, T_ref in K)
    chem1: vec4<f32>, // (c_sat, relative-flow u_half, wet-gate s_on, thermal exchange rate)
    therm: vec4<f32>, // (cp_water, cp_dry_grain, ambient loss rate, ambient T_K)
    // Capillary retention (coupling.wgsl): (residual pore saturation, Corey exponent,
    // relative-mobility floor, all-gas-atmospheric flag). The target-attainment ledger stays
    // on the CPU; surface.wgsl consumes .w for the explicitly vented production mode.
    retention: vec4<f32>,
    // Semi-implicit granular split (compaction.wgsl): (maximum packing tangent K,
    // initial consolidation pressure p_c0, full-tangent solid fraction, shear microstep dt).
    // K <= 0 is the exact legacy/off-path sentinel.
    granular: vec4<f32>,
    // Porous filter paper: (permeability su², kinematic viscosity su²/s, physical thickness su,
    // porosity). Geometry is independent of the coarse SDF collision shell and storage capacity.
    filter0: vec4<f32>,
    // Exact finite-reservoir capacity in FILTER_VOLUME_SCALE counts; remaining lanes reserved.
    filter1: vec4<u32>,
    // (insoluble grain mass, solute cp, initial extractable mass, reserved).
    solute: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
// Canonical particle state (ParticleBuffers layout): pos.w is the moisture lane (water =
// remaining fraction, grain = bound-water volume). chem is phase-specific:
// water=(solute/carrier mass ratio, temperature K, filter state 0/1/2, pore-transit progress),
// grain=(fast-pool mass, slow-pool mass, temperature K, wet-contact time).
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> phase: array<u32>;
@group(0) @binding(4) var<storage, read_write> chem: array<vec4<f32>>;
// Per-particle APIC affine velocity matrix C (3 rows per particle, row-major):
// cmat[3p + r] = vec4(C[r][0], C[r][1], C[r][2], 0). Rows so that (C·d)[r] = dot(row_r, d).
@group(0) @binding(5) var<storage, read_write> cmat: array<vec4<f32>>;
// WATER grid field, fixed-point: 4 atomic<i32> lanes per node, stride 4 — [mass, mom.x, mom.y,
// mom.z], each value × FP_SCALE. Cleared by grid_clear, scattered by p2g_water, read by
// grid_update. (The SOLID field generalizes this in U5 — not built here.)
@group(0) @binding(6) var<storage, read_write> grid_fp: array<atomic<i32>>;
// WATER grid velocity decoded by grid_update; drag_fold subsequently applies forces, pair drag,
// and boundary conditions, and project applies pore pressure. .xyz = velocity, .w = node mass.
@group(0) @binding(7) var<storage, read_write> grid_vel: array<vec4<f32>>;
// Combined auxiliary allocation. Prefix: per-CELL particle count (U4), the sharp
// particle-presence discriminator for the free-surface
// classification. Node mass alone cannot tell a real surface cell from a particle-free air
// gap narrower than the B-spline smear (≈3h between two water walls): such a gap reads mean
// corner density ≈ 0.3–0.5·ρ_rest, classifies as a constraint row, and its div = 0 row then
// FORBIDS the refilling inflow — observed as a jet-dug chimney standing frozen forever. The
// classical PIC/FLIP rule (a cell is fluid only if it contains particles) is the sharp
// discriminator; mass still provides the fill-fraction taper for cells that ARE fluid.
// Cleared by grid_clear, incremented by p2g_water, read by cell_classify. Suffix: one fixed-point
// carrier/solvent mass count per NODE. Mechanics uses solution mass in grid_fp/grid_vel, while
// fill, saturation, and density relief use this carrier census so dissolved solute cannot create
// liquid volume.
@group(0) @binding(18) var<storage, read_write> cell_cnt: array<atomic<u32>>;

fn aux_cell_slots() -> u32 {
    let cells = params.grid_dims.xyz - vec3<u32>(1u);
    return cells.x * cells.y * cells.z;
}

fn carrier_mass_index(node: u32) -> u32 {
    return aux_cell_slots() + node;
}

fn carrier_mass_at(node: u32) -> f32 {
    return f32(atomicLoad(&cell_cnt[carrier_mass_index(node)])) / FP_SCALE;
}

fn water_solvent_mass(f_water: f32) -> f32 {
    return params.particle_mass * clamp(f_water, 0.0, 1.0);
}

fn water_solute_mass(f_water: f32, concentration: f32) -> f32 {
    return water_solvent_mass(f_water) * max(concentration, 0.0);
}

fn water_solution_mass(f_water: f32, concentration: f32) -> f32 {
    return water_solvent_mass(f_water) + water_solute_mass(f_water, concentration);
}
// SOLID grid field (U6): per-node solid VOLUME, fixed-point (×FP_SCALE), one lane per node —
// the φ_s carrier (KTD-8: a local evolving field, never a configured scalar). Cleared by
// grid_clear, scattered by p2g_solid[_dyn] (saturation-driven structural volume), read wherever
// φ_s/φ_f is needed (drag_fold, node_setup, project). Headroom: |V_s| per node ≤ PHI_S-clamped
// h³ ≈ 8 ≪ the 2^13 fixed-point ceiling.
@group(0) @binding(19) var<storage, read_write> grid_sfp: array<atomic<i32>>;
// Interphase drag state: per node, .xyz is the drag impulse water gives the solid. Frozen mode
// records the impulse a kinematically constrained skeleton would absorb and project accumulates
// its pressure reaction; dynamic mode applies the equal-and-opposite impulse directly to
// grid_svel and keeps .xyz only as an observation of that already-applied exchange. .w is
// sigma, the exact relative-mode exponential mobility handed to node_setup (1 at an unpaired
// node). Written by drag_fold every frame; no clear pass needed.
@group(0) @binding(20) var<storage, read_write> react: array<vec4<f32>>;
// U5 SOLID momentum grid field (dynamic skeleton, plasticity.wgsl): fixed-point, 4 lanes per
// node [mass, mom.xyz] — the solid mirror of grid_fp. Cleared by grid_clear, scattered by
// p2g_solid_dyn, decoded by solid_update. Only dispatched-to when Config::solid_dynamics is
// on (params.splas0.x > 0.5); cleared unconditionally (never read in frozen mode).
@group(0) @binding(21) var<storage, read_write> grid_sm: array<atomic<i32>>;
// U9 MOISTURE grid ledger (coupling.wgsl, GIC absorption), 6 lane-scaled integer lanes per node:
// [S,D,Qx,Qy,Qz,R]. S/D are supply/demand, Q=Σ w·V·v is donor volume-momentum, and
// R=Σ w·V·T is donor volume-temperature. T=min(S,D) transfers volume; Q/S and R/S transfer
// its velocity and temperature to grains. Multiplication by particle_mass/V_w converts both
// sides to reduced mass, momentum, heat capacity, and enthalpy. Lane scales and their audited
// representable magnitudes are documented beside the encode/decode helpers below.
const MOIST_STRIDE: u32 = 6u;
const MOIST_SUPPLY: u32 = 0u;
const MOIST_DEMAND: u32 = 1u;
const MOIST_QX: u32 = 2u;
const MOIST_RT: u32 = 5u;
@group(0) @binding(25) var<storage, read_write> grid_moist: array<atomic<i32>>;

// Chemistry grid ledger, 11 lane-scaled integer lanes per node:
// [solute headroom H, proposed fast F, proposed slow L, water capacity Cw, water enthalpy Ew,
// grain capacity Cg, grain enthalpy Eg, proposed solute momentum Jxyz, proposed solute enthalpy U].
// chemistry_scatter freezes these totals after pressure; chemistry_gather applies the same
// min(H,F+L) mass/momentum/enthalpy transfer and bounded heat exchange to both phases.
const CHEM_STRIDE: u32 = 11u;
const CHEM_HEADROOM: u32 = 0u;
const CHEM_FAST: u32 = 1u;
const CHEM_SLOW: u32 = 2u;
const CHEM_CW: u32 = 3u;
const CHEM_EW: u32 = 4u;
const CHEM_CG: u32 = 5u;
const CHEM_EG: u32 = 6u;
const CHEM_JX: u32 = 7u;
const CHEM_U: u32 = 10u;
@group(0) @binding(28) var<storage, read_write> grid_chem: array<atomic<i32>>;

// Solid plastic-compaction moment, one fixed-point lane per node:
//     sum_p w_ip * V_struct,p * e_p.
// Dividing by grid_sfp gives the volume-weighted compaction carried by that node. It is kept
// separate from stress magnitudes so the common FP_SCALE retains ample headroom.
@group(0) @binding(29) var<storage, read_write> grid_scomp: array<atomic<i32>>;

// Persistent finite paper reservoir (not cleared with the per-frame grid): occupied solvent
// volume counts, cumulative admitted/released/overflow parcels, and invariant-error flags.
const FILTER_OCCUPIED: u32 = 0u;
const FILTER_ADMITTED: u32 = 1u;
const FILTER_RELEASED: u32 = 2u;
const FILTER_OVERFLOW: u32 = 3u;
const FILTER_ERRORS: u32 = 4u;
const FILTER_STATE_SLOTS: u32 = 5u;
const FILTER_VOLUME_SCALE: f32 = 268435456.0; // 2^28, mirrored by Rust
@group(0) @binding(30) var<storage, read_write> filter_state: array<atomic<u32>>;

fn filter_volume_encode(volume: f32) -> u32 {
    return u32(clamp(round(max(volume, 0.0) * FILTER_VOLUME_SCALE), 0.0, 4294967040.0));
}

fn filter_volume_decode(counts: u32) -> f32 {
    return f32(counts) / FILTER_VOLUME_SCALE;
}

fn filter_try_reserve(parcel_counts: u32) -> bool {
    let capacity = params.filter1.x;
    if (parcel_counts == 0u || capacity == 0u || parcel_counts > capacity) {
        atomicAdd(&filter_state[FILTER_OVERFLOW], 1u);
        return false;
    }
    var observed = atomicLoad(&filter_state[FILTER_OCCUPIED]);
    loop {
        if (observed > capacity || parcel_counts > capacity - observed) {
            atomicAdd(&filter_state[FILTER_OVERFLOW], 1u);
            return false;
        }
        let exchanged = atomicCompareExchangeWeak(
            &filter_state[FILTER_OCCUPIED], observed, observed + parcel_counts,
        );
        if (exchanged.exchanged) {
            atomicAdd(&filter_state[FILTER_ADMITTED], 1u);
            return true;
        }
        observed = exchanged.old_value;
    }
    return false; // structurally required by WGSL validation; the loop exits only via returns
}

// --- fixed-point encoding (KEEP.md §3 pattern, headroom re-validated for U2) -----------------
//
// FP_SCALE = 2^18. i32 range is ±2^31, so an encoded node lane overflows at |value| ≥ 2^13 = 8192.
//   mass lane:      Σ_p m·w per node. Unit-mass water at rest loads an interior node with
//                   Σ m·w ≈ (particles per cell volume) = (h/spacing)³ = 8 → overflow needs
//                   ≈ 1000× rest packing on one node. Unreachable.
//   momentum lanes: every scattered contribution is magnitude-clamped to max_speed (50 by
//                   default — Config::max_speed, the same cap G2P enforces), so |Σ m·w·v_aff| ≤
//                   (Σ m·w)·max_speed → overflow needs Σ m·w ≥ 8192/50 ≈ 164 mass units on one
//                   node ≈ 20× rest compression with EVERY particle at the cap. The overflow
//                   probe gate (tests/twofield_water.rs) exercises a capped, pancaked worst case.
// COUPLED to the velocity cap: raising max_speed shrinks the momentum headroom linearly —
// change one, re-derive the other. fp_encode also saturates (clamp before convert) as a
// backstop so a pathological state degrades bounded instead of wrapping.
const FP_SCALE: f32 = 262144.0; // 2^18
// Largest f32 ≤ i32::MAX (2^31 − 1 = 2147483647 is not representable; nearest-below is 2147483520).
const FP_CLAMP: f32 = 2147483520.0;

fn fp_encode(x: f32) -> i32 {
    return i32(clamp(round(x * FP_SCALE), -FP_CLAMP, FP_CLAMP));
}
fn fp_decode(c: i32) -> f32 {
    return f32(c) / FP_SCALE;
}

// --- phase-change / chemistry lane scales ----------------------------------------------------
// A single 2^18 scale cannot resolve the smallest UI-rate constitutive increments without
// independently rounding 27 stencil contributions. These ledgers instead encode each PARTICLE
// extensive total once, then partition that integer exactly across the stencil (see
// extensive_partition_target). The scales below are the audited safe starting powers; the
// corresponding maximum decoded magnitude before i32 saturation is approximately:
//
//   moisture S,D : 2^30 -> 2       reduced volume
//   moisture Q   : 2^26 -> 32      volume·velocity, each signed component
//   moisture R   : 2^24 -> 128     volume·kelvin
//   chemistry H  : 2^26 -> 32      solute-mass headroom
//   chemistry F,L: 2^27 -> 16      proposed solute mass
//   chemistry C  : 2^22 -> 512     heat capacity
//   chemistry E  : 2^13 -> 262144  capacity·kelvin
//   chemistry J  : 2^26 -> 32      solute mass·velocity, signed per component
//
// Runtime recipe sampling keeps the actual node totals well inside these bounds. Raising a scale
// reduces its physical range and therefore requires a new node-headroom audit.
const MOIST_SD_SCALE: f32 = 1073741824.0; // 2^30
const MOIST_Q_SCALE: f32 = 67108864.0;    // 2^26
const MOIST_R_SCALE: f32 = 16777216.0;    // 2^24
const CHEM_H_SCALE: f32 = 67108864.0;     // 2^26
const CHEM_FL_SCALE: f32 = 134217728.0;   // 2^27
const CHEM_C_SCALE: f32 = 4194304.0;      // 2^22
const CHEM_E_SCALE: f32 = 8192.0;         // 2^13
const CHEM_J_SCALE: f32 = 67108864.0;      // 2^26

fn lane_encode(x: f32, scale: f32) -> i32 {
    return i32(clamp(round(x * scale), -FP_CLAMP, FP_CLAMP));
}
fn lane_decode(count: i32, scale: f32) -> f32 {
    return f32(count) / scale;
}
fn moist_encode(lane: u32, x: f32) -> i32 {
    if (lane == MOIST_SUPPLY || lane == MOIST_DEMAND) { return lane_encode(x, MOIST_SD_SCALE); }
    if (lane == MOIST_RT) { return lane_encode(x, MOIST_R_SCALE); }
    return lane_encode(x, MOIST_Q_SCALE);
}
fn moist_decode(lane: u32, count: i32) -> f32 {
    if (lane == MOIST_SUPPLY || lane == MOIST_DEMAND) { return lane_decode(count, MOIST_SD_SCALE); }
    if (lane == MOIST_RT) { return lane_decode(count, MOIST_R_SCALE); }
    return lane_decode(count, MOIST_Q_SCALE);
}
fn chem_encode(lane: u32, x: f32) -> i32 {
    if (lane == CHEM_HEADROOM) { return lane_encode(x, CHEM_H_SCALE); }
    if (lane == CHEM_FAST || lane == CHEM_SLOW) { return lane_encode(x, CHEM_FL_SCALE); }
    if (lane == CHEM_CW || lane == CHEM_CG) { return lane_encode(x, CHEM_C_SCALE); }
    if (lane >= CHEM_JX && lane < CHEM_U) { return lane_encode(x, CHEM_J_SCALE); }
    return lane_encode(x, CHEM_E_SCALE);
}
fn chem_decode(lane: u32, count: i32) -> f32 {
    if (lane == CHEM_HEADROOM) { return lane_decode(count, CHEM_H_SCALE); }
    if (lane == CHEM_FAST || lane == CHEM_SLOW) { return lane_decode(count, CHEM_FL_SCALE); }
    if (lane == CHEM_CW || lane == CHEM_CG) { return lane_decode(count, CHEM_C_SCALE); }
    if (lane >= CHEM_JX && lane < CHEM_U) { return lane_decode(count, CHEM_J_SCALE); }
    return lane_decode(count, CHEM_E_SCALE);
}

// Cumulative-rounding partition of one already-encoded particle total. Call in deterministic
// stencil order while carrying `allocated` and cumulative B-spline weight. The final node receives
// the exact integer remainder. Thus Σ_node share == total for positive and signed lanes, and the
// gather can reproduce byte-identical shares without a per-particle residual buffer.
fn extensive_partition_target(total: i32, cumulative_weight: f32, is_last: bool) -> i32 {
    if (is_last) { return total; }
    // An arbitrary i32 cannot be represented exactly as f32. Split into 256-count blocks so the
    // quotient (|q| < 2^23) and signed remainder are both exact f32 integers. Partitioning each
    // monotone component separately prevents a positive total from producing a negative final
    // correction merely because f32(total) rounded upward.
    let blocks = total / 256;
    let remainder = total - blocks * 256;
    let prefix = clamp(cumulative_weight, 0.0, 1.0);
    let block_prefix = i32(round(f32(blocks) * prefix));
    let remainder_prefix = i32(round(f32(remainder) * prefix));
    return block_prefix * 256 + remainder_prefix;
}

// --- quadratic B-spline weights (KEEP.md §3 validated math) -----------------------------------
// Per axis, around base = floor(x/h − 0.5) with fx = x/h − base ∈ [0.5, 1.5]; node offset
// k ∈ {0,1,2} gets w[k]. Weights sum to 1 and Σ_k w[k]·(base+k) = x/h (linear consistency —
// what makes the free-fall recurrence exact and keeps APIC's C zero in a uniform field).
fn bspline_w(fx: vec3<f32>) -> array<vec3<f32>, 3> {
    var w: array<vec3<f32>, 3>;
    w[0] = 0.5 * (1.5 - fx) * (1.5 - fx);
    w[1] = 0.75 - (fx - 1.0) * (fx - 1.0);
    w[2] = 0.5 * (fx - 0.5) * (fx - 0.5);
    return w;
}

// Analytic d/dfx of bspline_w (per axis), so ∂w/∂x = dw/h (the 1/h chain-rule scale is applied
// by the caller). Used by g2p_water's mass-gradient gather (U2 surface-weighted dissipation):
// ∇w_3d = (dw.x·w.y·w.z, w.x·dw.y·w.z, w.x·w.y·dw.z)/h. NOT reusable from APIC's B (which uses the
// d-offset form, not ∇w) — a few extra ALU ops, no new binding.
//   d/dfx [0.5·(1.5−fx)²] = −(1.5 − fx)
//   d/dfx [0.75 − (fx−1)²] = −2·(fx − 1)
//   d/dfx [0.5·(fx−0.5)²]  =  (fx − 0.5)
fn bspline_dw(fx: vec3<f32>) -> array<vec3<f32>, 3> {
    var dw: array<vec3<f32>, 3>;
    dw[0] = -(1.5 - fx);
    dw[1] = -2.0 * (fx - 1.0);
    dw[2] = fx - 0.5;
    return dw;
}

// --- grid indexing -----------------------------------------------------------------------------
// Flat node index, x-fastest. Node world position = grid_origin + (i,j,k)·h.
fn node_index(n: vec3<i32>) -> u32 {
    return u32(n.x) + params.grid_dims.x * (u32(n.y) + params.grid_dims.y * u32(n.z));
}
fn node_coords(flat: u32) -> vec3<u32> {
    let nx = params.grid_dims.x;
    let ny = params.grid_dims.y;
    return vec3<u32>(flat % nx, (flat / nx) % ny, flat / (nx * ny));
}

// --- SDF cavity geometry (mirrors utils/sdf.rs; interior positive, gradient toward the cavity) --
// Byte-identical to the Rust `Primitive` (64 bytes). Cone radii in `a` are OUTER wall radii.
struct Primitive {
    kind: u32,          // 0 = cone, 1 = cylinder, 2 = poly-cup
    species_mask: u32,
    friction: f32,
    flags: u32,         // bit0 = apex_open, bit1 = porous filter paper
    a: vec4<f32>,       // cone:(apex_y, apex_r, top_y, top_r)  cyl:(floor_y, rim_y, radius, _)  poly:(floor_y, rim_y, apothem, sides)
    b: vec4<f32>,       // cone:(thickness, hole_radius, center_x, center_z)  cyl/poly:(center_x, center_z, _, _)
    c: vec4<f32>,       // reserved
};
@group(0) @binding(8) var<storage, read> solids: array<Primitive>;

const SDF_EPS: f32 = 1.0e-6;
const SDF_FREE: f32 = 1.0e30;
const PRIMITIVE_FILTER_PAPER: u32 = 2u;

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

// DIAGNOSTIC: regular N-gon prism cup (mirror of sdf.rs::poly_cavity). Face normals at 2πk/N
// (k=0 → +x), so sides=4 is an axis-aligned square. Inner side dist = apothem − max_k(rel·n_k).
fn poly_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    let floor_y = prim.a.x;
    let rim_y = prim.a.y;
    let apothem = prim.a.z;
    let sides = max(u32(prim.a.w), 3u);
    let center = vec2<f32>(prim.b.x, prim.b.y);
    let rel = vec2<f32>(p.x - center.x, p.z - center.y);
    let y = p.y;
    if (y > rim_y) { return SolidHit(SDF_FREE, vec3<f32>(0.0, 1.0, 0.0), prim.friction); }
    var maxproj = -1.0e30;
    var bestn = vec2<f32>(1.0, 0.0);
    let tau = 6.28318530718;
    for (var k = 0u; k < sides; k = k + 1u) {
        let ang = tau * f32(k) / f32(sides);
        let nk = vec2<f32>(cos(ang), sin(ang));
        let proj = dot(rel, nk);
        if (proj > maxproj) { maxproj = proj; bestn = nk; }
    }
    let d_side = apothem - maxproj;
    let d_floor = y - floor_y;
    if (d_side <= d_floor) {
        return SolidHit(d_side, sdf_normalize3(vec3<f32>(-bestn.x, 0.0, -bestn.y)), prim.friction);
    }
    return SolidHit(d_floor, vec3<f32>(0.0, 1.0, 0.0), prim.friction);
}

fn solid_cavity(prim: Primitive, p: vec3<f32>) -> SolidHit {
    if (prim.kind == 0u) { return cone_cavity(prim, p); }
    if (prim.kind == 2u) { return poly_cavity(prim, p); }
    return cyl_cavity(prim, p);
}

// Species-filtered union: the most-penetrated (min signed-distance) solid that blocks `ph`.
fn solid_union(p: vec3<f32>, ph: u32) -> SolidHit {
    var best = SolidHit(SDF_FREE, vec3<f32>(0.0, 0.0, 0.0), 0.0);
    let n = params.num_solids;
    for (var i = 0u; i < n; i = i + 1u) {
        let prim = solids[i];
        if ((prim.species_mask & (1u << ph)) == 0u) { continue; }
        let hit = solid_cavity(prim, p);
        if (hit.dist < best.dist) { best = hit; }
    }
    return best;
}

// Explicit semantic query for porous paper. This intentionally ignores species_mask: the V60
// paper does not collide with water, but its bit1 surface still supplies hydraulic resistance and
// reservoir crossing. An unrelated grains-only primitive without bit1 has no paper behavior.
fn filter_paper_hit(p: vec3<f32>) -> SolidHit {
    var best = SolidHit(SDF_FREE, vec3<f32>(0.0, 0.0, 0.0), 0.0);
    let n = params.num_solids;
    for (var i = 0u; i < n; i = i + 1u) {
        let prim = solids[i];
        if ((prim.flags & PRIMITIVE_FILTER_PAPER) == 0u) { continue; }
        let hit = solid_cavity(prim, p);
        if (abs(hit.dist) < abs(best.dist)) { best = hit; }
    }
    return best;
}

struct PaperResponse {
    normal: vec3<f32>,
    coverage: f32,
    alpha: f32,
    beta: f32,
    decay: f32,
    sigma: f32,
};

// One-cell normalized delta band for a physically thin Darcy sheet. Integrating alpha across the
// numerical band recovers the physical thickness L, so changing grid h does not change the sheet
// resistance nu*L/k. sigma is the exact pressure/source mobility (1-exp(-x))/x.
fn filter_paper_response(p: vec3<f32>) -> PaperResponse {
    let hit = filter_paper_hit(p);
    let h = params.grid_origin.w;
    let permeability = params.filter0.x;
    let viscosity = params.filter0.y;
    let thickness = params.filter0.z;
    let coverage = clamp(1.0 - abs(hit.dist) / h, 0.0, 1.0);
    if (coverage <= 0.0 || length(hit.grad) <= SDF_EPS
        || permeability <= 0.0 || viscosity <= 0.0 || thickness <= 0.0) {
        return PaperResponse(hit.grad, 0.0, 0.0, 0.0, 1.0, 1.0);
    }
    let alpha = (thickness / h) * coverage;
    let beta = (viscosity / permeability) * alpha;
    let x = max(beta * params.dt, 0.0);
    var decay: f32;
    var sigma: f32;
    if (x < 1.0e-3) {
        decay = 1.0 - x + 0.5 * x * x;
        sigma = 1.0 - 0.5 * x + x * x / 6.0;
    } else {
        decay = exp(-x);
        sigma = (1.0 - decay) / x;
    }
    return PaperResponse(
        sdf_normalize3(hit.grad),
        coverage,
        alpha,
        beta,
        clamp(decay, 0.0, 1.0),
        clamp(sigma, 0.0, 1.0),
    );
}

// MULTI-NORMAL wall BC (plan 2026-06-17-002): all cavity FACES within `band` of p, generalizing
// solid_union's single most-penetrated pick. At a concave seam (cup floor∩wall) two faces are
// in-band and BOTH must be constrained — solid_union would return only one, leaving the other
// direction free (the corner over-pack). `band` is phase-selected by pressure.wgsl: WALL_BAND·h
// for water's supporting layer, zero for solid actual contact. Floor is pushed FIRST so the
// rank-aware basis build (build_constraint_basis) never drops it. cone stays single-normal (it is
// one smooth converging surface, not discrete faces). Cap WF_MAX candidates.
const WF_MAX: u32 = 4u;
struct WallFaces { n: array<vec3<f32>, 4>, count: u32 };

fn wf_push(wf: ptr<function, WallFaces>, nrm: vec3<f32>) {
    let c = (*wf).count;
    if (c < WF_MAX && length(nrm) > SDF_EPS) {
        (*wf).n[c] = sdf_normalize3(nrm);
        (*wf).count = c + 1u;
    }
}

// Orthonormal constraint basis Q (≤3 columns) for the MULTI-NORMAL wall BC: the active box-face
// axes plus the in-band SDF wall normals, orthonormalized by modified Gram-Schmidt. The projector
// is P = I − Q·Qᵀ — a TRUE orthogonal projector (PSD for ANY normals, even a non-orthogonal poly
// vertical edge, where a raw Σ n̂n̂ᵀ would be non-PSD and break A = D·M̃⁻¹·G). `node_setup` builds
// M̃⁻¹ = invr·P and `drag_fold` applies v ← P·v from the SAME basis (lockstep). box_mask bits:
// 1 = x axis constrained, 2 = y, 4 = z (the axes node_setup/drag_fold zero on box faces). Rank-aware:
// box axes first, then wall faces residualized + dropped if near-dependent, stop at rank 3 — so a
// floor normal is never crowded out (the floor face is pushed first by wall_binding_faces).
// This lockstep matrix/velocity statement applies to the water phase; solid velocity contact is
// unilateral Coulomb, while its pressure active set is frozen into a linear projector per solve.
const CB_RESIDUAL_MIN: f32 = 1.0e-3;
struct CBasis { q: array<vec3<f32>, 3>, count: u32 };

fn cb_push_ortho(cb: ptr<function, CBasis>, v: vec3<f32>) {
    if ((*cb).count >= 3u) { return; }
    var r = v;
    for (var j = 0u; j < (*cb).count; j = j + 1u) {
        r = r - (*cb).q[j] * dot((*cb).q[j], r);
    }
    let len = length(r);
    if (len > CB_RESIDUAL_MIN) {
        (*cb).q[(*cb).count] = r / len;
        (*cb).count = (*cb).count + 1u;
    }
}

fn build_constraint_basis(box_mask: u32, faces: WallFaces) -> CBasis {
    var cb: CBasis;
    cb.count = 0u;
    if ((box_mask & 1u) != 0u) { cb_push_ortho(&cb, vec3<f32>(1.0, 0.0, 0.0)); }
    if ((box_mask & 2u) != 0u) { cb_push_ortho(&cb, vec3<f32>(0.0, 1.0, 0.0)); }
    if ((box_mask & 4u) != 0u) { cb_push_ortho(&cb, vec3<f32>(0.0, 0.0, 1.0)); }
    for (var i = 0u; i < faces.count; i = i + 1u) {
        cb_push_ortho(&cb, faces.n[i]);
    }
    return cb;
}

// Apply P = I − Q·Qᵀ to a vector (Q orthonormal ⇒ subtract each column's projection of the
// ORIGINAL v). Used by drag_fold for the velocity BC; node_setup builds the matrix form inline.
fn cbasis_project_vec(cb: CBasis, v: vec3<f32>) -> vec3<f32> {
    var out = v;
    for (var i = 0u; i < cb.count; i = i + 1u) {
        out = out - cb.q[i] * dot(cb.q[i], v);
    }
    return out;
}

fn wall_binding_faces(p: vec3<f32>, ph: u32, band: f32) -> WallFaces {
    var wf: WallFaces;
    wf.count = 0u;
    let ns = params.num_solids;
    for (var i = 0u; i < ns; i = i + 1u) {
        let prim = solids[i];
        if ((prim.species_mask & (1u << ph)) == 0u) { continue; }
        if (prim.kind == 0u) {
            // cone: single smooth surface — one nearest normal, in-band only.
            let hit = cone_cavity(prim, p);
            if (hit.dist < band) { wf_push(&wf, hit.grad); }
        } else if (prim.kind == 2u) {
            // poly cup: floor + every in-band side face (vertical edges yield 2 sides).
            let floor_y = prim.a.x; let rim_y = prim.a.y; let apothem = prim.a.z;
            let sides = max(u32(prim.a.w), 3u);
            let center = vec2<f32>(prim.b.x, prim.b.y);
            let rel = vec2<f32>(p.x - center.x, p.z - center.y);
            if (p.y <= rim_y) {
                if ((p.y - floor_y) < band) { wf_push(&wf, vec3<f32>(0.0, 1.0, 0.0)); }
                let tau = 6.28318530718;
                for (var k = 0u; k < sides; k = k + 1u) {
                    let ang = tau * f32(k) / f32(sides);
                    let nk = vec2<f32>(cos(ang), sin(ang));
                    if ((apothem - dot(rel, nk)) < band) {
                        wf_push(&wf, vec3<f32>(-nk.x, 0.0, -nk.y));
                    }
                }
            }
        } else {
            // cylinder cup: floor + radial side face when in-band.
            let floor_y = prim.a.x; let rim_y = prim.a.y; let radius = prim.a.z;
            let center = vec2<f32>(prim.b.x, prim.b.y);
            let rel = vec2<f32>(p.x - center.x, p.z - center.y);
            let r = length(rel);
            if (p.y <= rim_y) {
                if ((p.y - floor_y) < band) { wf_push(&wf, vec3<f32>(0.0, 1.0, 0.0)); }
                if ((radius - r) < band && r >= SDF_EPS) {
                    let rh = rel / r;
                    wf_push(&wf, vec3<f32>(-rh.x, 0.0, -rh.y));
                }
            }
        }
    }
    return wf;
}

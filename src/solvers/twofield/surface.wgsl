// U4 free surface + constraint bubble (plan 2026-06-09-001 U4 / KTD-6).
//
// ================================ POCKET DETECTION ============================================
// Air cells (cell_meta.w = CELL_AIR after cell_classify: no fluid, center in-box, not inside a
// solid) are flood-filled from the OPEN boundary: the topmost in-box cell layer is the
// atmosphere (the box's side/bottom faces are sealed walls — air sealed against them is still
// enclosed). The flood runs a scene-derived number of ping-pong relaxation sweeps (the grid
// Manhattan diameter — `flood_sweeps_for` in mod.rs; labels in the pressure ping-pong buffers,
// which cell_classify zeroed and pocket_mark re-zeroes) propagating the OUTSIDE label one
// 6-neighbor step per sweep — uniform control flow, deterministic (read old / write new, never
// racy in-place). The budget MUST cover the longest open-air path or genuinely-open air (e.g.
// the deep V60 cup under its cone) is misread as one enclosed pocket and the bubble crushes the
// standing water. Air cells still unlabeled afterwards are ENCLOSED: the
// pocket. Single-pocket case (the pour cavity) per the plan: ALL enclosed air shares ONE
// bubble — multi-pocket generalization is deferred scope.
//
// ================================ BUBBLE REPRESENTATION (KTD-6, pinned) =======================
// ONE explicit Lagrange-multiplier scalar λ_b (bubble[0]); its value IS the pocket pressure.
// Pocket cells are inserted into the SAME operator family as fluid rows: pocket_mark gives
// them fill weight f = 1 and rhs = (0 − D v)/dt (no density relief — air), so
//   * the COLUMN coupling is automatic: fluid rows and the projection gather pocket pressure
//     through the one G = −Dᵀ (the pocket pressure acts on exactly the pocket-boundary faces
//     — interior pocket nodes are massless, M̃⁻¹ = 0, and contribute nothing);
//   * the ROW is the aggregate volume constraint Σ_{c∈pocket}(D u⁺)_c = 0 — the row sum of
//     the per-cell rows, solved EXACTLY for the single unknown each relaxation step:
//         λ_b ← λ_b + (Σ rhs_c − Σ (A p̃)_c) / (1ᵀ_pocket·A·1_pocket),
//     where p̃ carries λ_b in the pocket slots and the denominator is the full aggregated
//     quadratic form (cross terms between pocket cells included — computed as Σ(A·1_pocket)_c).
// IDENTICAL REPRESENTATION AT BOTH LEVELS (the named failure is a coarse-only row the fine
// sweeps relax away — and its mirror image): a bubble_fine step precedes EVERY fine Jacobi
// sweep and jacobi_fine pins pocket slots to λ_b (Dirichlet copy of the multiplier); the
// coarse error equation carries the bubble correction δλ_b (bubble[1]) the same way —
// bubble_coarse precedes every coarse sweep, jacobi_coarse pins all-pocket coarse cells to
// δλ_b, and prolong_add adds δλ_b (a scalar — its prolongation is exact) to the fine pocket
// slots; the next bubble_fine folds λ_b ← λ_b + δλ_b before re-solving. p₀ = 0 doctrine:
// pocket_mark zeroes λ_b and δλ_b every frame (no warm start). When the pocket is OPEN
// (pre-enclosure — e.g. the jet annulus connects the cavity to outside air) there are no
// pocket cells, the aggregated diagonal is 0, and λ_b is pinned inactive (p = 0 air).
//
// Tint discipline: bubble_fine/bubble_coarse run as ONE workgroup with a uniform trip count
// (now over the COMPACTED pocket list, not all cells — U8 R9 fix) and predicated bodies, so
// every workgroupBarrier() is in uniform control flow. All other passes use guard returns with
// no barriers. Over-dispatch + early-out only (R8). Storage buffers per entry point ≤ 7 (see
// MAX_STORAGE_BUFFERS_PER_ENTRY_POINT in mod.rs).

// --- bindings (continue the global table; 17 is the bubble state) -----------------------------
// bubble[0] = λ_b (fine-level pocket pressure), bubble[1] = δλ_b (coarse correction).
@group(0) @binding(17) var<storage, read_write> bubble: array<f32>;

// U8 R9 fix: COMPACTED pocket-cell lists (the parallel-reduction approach). Element 0 is an
// atomic append counter, elements [1, 1+count) are the flat cell indices labelled CELL_POCKET
// this frame. flood_init resets both counters; pocket_mark appends the fine pockets and
// coarse_cell_setup appends the coarse pockets. The bubble row solves then iterate the few
// hundred listed cells instead of strided-scanning all ~76k cells per dispatch (the same sums
// over the SAME cells — only the iteration set is compacted, λ_b is unchanged).
@group(0) @binding(26) var<storage, read_write> pocket_f: array<atomic<u32>>;
@group(0) @binding(27) var<storage, read_write> pocket_c: array<atomic<u32>>;

// cell_meta.w / cmeta.w categories (CELL_AIR is only ever set on fine cells).
const CELL_AIR: f32 = 1.0;
const CELL_POCKET: f32 = 2.0;
// Label values in the flood ping-pong (pressure slots reused before the solve zeroes them).
const FLOOD_OUTSIDE: f32 = 1.0;

// Minimum resolved-bubble size, in FINE cells. The constant-volume bubble constraint models a
// RESOLVED incompressible air cavity (the pour crater). A pocket of only a handful of cells is
// sub-resolution air: a 3-node (2-cell) B-spline support cannot represent a bubble narrower than
// ~2 cells per axis, so its "incompressibility" is a discretization artifact, not physics — real
// entrained microbubbles are compressible and vent/dissolve. Such a pocket must be RELEASED to
// p = 0 (open air), not pinned to a huge λ. Without this, the V60 cone's draining film
// geometrically isolates 1–7 air cells the 6-neighbor flood cannot reach; the single global
// bubble then assigns them λ ≈ 1e3–1e4, ejecting the standing cup water at the speed cap for
// 100+ frames (the residual WaterOnly artifact after the flood-budget / wall-band fixes). One
// 2×2×2 cell block (8) is the threshold: the legitimate pour cavity is hundreds of cells (the
// cavity gate's enclosed slab is 288), so the band is enormous — this can never gate a real
// cavity. Both levels gate on this one authoritative FINE count (bubble_coarse reads pocket_f[0]
// too), so the coarse correction releases on exactly the same criterion as the fine solve.
const MIN_POCKET_FINE_CELLS: u32 = 8u;

// =================================== flood_init ================================================
// Seed the OUTSIDE label: air cells in the topmost in-box layer (the open atmosphere face), AND
// every air cell WITHIN ONE CELL OF AN SDF WALL. The wall-adjacent seed is what makes the pocket
// machinery correct in deep SDF geometry like the V60 cup: a 6-neighbor flood cannot squeeze the
// OUTSIDE label down through the ~1-cell-wide cone apex hole (radius 0.42 ≈ 1.3 cells at the web
// h = 0.32) to reach the cup air below, so the whole cup/cone air column was misread as one
// enclosed POCKET and the bubble constraint (λ ≈ −3000) crushed the standing water into the
// floor corner — the reported WaterOnly collapse. A sub-resolution void touching an SDF wall is
// never genuinely trapped gas at this resolution: it is open (fluid settles into the bottom-of-
// column gap; air vents through the apex/wall gap), pressure p = 0. This narrows the bubble to
// air actually sealed away from any wall — the transient pour cavity the machinery exists for
// (the cavity gate scene is a plain box, no SDF walls, so it is byte-unchanged: the extra seed
// only fires when num_solids > 0 AND a wall is within a cell).
@compute @workgroup_size(256)
fn flood_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c == 0u) {
        bubble[2] = 0.0; // pocket-present flag, raised by pocket_mark this frame
        atomicStore(&pocket_f[0], 0u); // reset the compacted pocket-list counters this frame
        atomicStore(&pocket_c[0], 0u);
    }
    if (c >= num_fine_cells()) {
        return;
    }
    var label = 0.0;
    if (cell_meta[c].w == CELL_AIR) {
        let h = params.grid_origin.w;
        let cc = vec3<f32>(cell_coords(c));
        let center = params.grid_origin.xyz + (cc + vec3<f32>(0.5)) * h;
        if (center.y + h > params.box_max.y) {
            label = FLOOD_OUTSIDE; // the cell above is out of the box: open face
        } else if (params.num_solids > 0u
            && solid_union(center, PHASE_WATER).dist < FLOOD_WALL_BAND * h) {
            // Any air cell within FLOOD_WALL_BAND cells of an SDF wall is OPEN, not trapped gas:
            // a sub-resolution void at a wall (the bottom-of-column floor gap), OR an air cell in
            // a narrow SDF channel the 6-neighbor flood cannot squeeze the label through (the V60
            // cone apex hole, radius 0.42 ≈ 1.3 cells — a center-of-apex air cell sits ~0.42 from
            // the wall, just OUTSIDE the projector's 1-cell band, so the flood seed uses a wider
            // 2-cell band to keep the apex conductive and connect the cup/cone air to the open
            // top). Without this the filled cup seals the cone air into one enclosed pocket and
            // the bubble crushes the pool.
            label = FLOOD_OUTSIDE;
        }
    }
    pf_src[c] = label;
}

// =================================== flood_sweep ===============================================
// One 6-neighbor propagation step, ping-pong (read 11, write 12). Non-air cells keep label 0
// (fluid blocks propagation), so reading a neighbor's label needs no category check.
@compute @workgroup_size(256)
fn flood_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_fine_cells()) {
        return;
    }
    var label = 0.0;
    if (cell_meta[c].w == CELL_AIR) {
        label = pf_src[c];
        if (label < 0.5) {
            let cc = vec3<i32>(cell_coords(c));
            for (var a = 0; a < 3; a = a + 1) {
                for (var s = -1; s <= 1; s = s + 2) {
                    var n = cc;
                    n[a] = n[a] + s;
                    if (cell_in_range(n) && pf_src[cell_index(n)] > 0.5) {
                        label = FLOOD_OUTSIDE;
                    }
                }
            }
        }
    }
    pf_dst[c] = label;
}

// =================================== pocket_mark ===============================================
// Enclosed air (still unlabeled) becomes the pocket: fill weight 1, rhs = (0 − D v)/dt — a
// constraint row of the one family (header: BUBBLE REPRESENTATION). Re-zeroes the pressure
// ping-pong the flood borrowed and resets the bubble scalars (p₀ = 0, fresh λ each frame).
@compute @workgroup_size(256)
fn pocket_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= num_fine_cells()) {
        return;
    }
    if (c == 0u) {
        bubble[0] = 0.0;
        bubble[1] = 0.0;
    }
    let label = pf_src[c];
    pf_src[c] = 0.0;
    pf_dst[c] = 0.0;
    let cm = cell_meta[c];
    let cc = vec3<i32>(cell_coords(c));

    // Density relief, UNDER-density half (the deferred-to-U4 suction; over-density half in
    // cell_classify): an INTERIOR cell below rest density demands inflow at the same relief
    // rate, re-compacting rarefied regions (a jet tears a sub-cell void channel whose cells
    // still read f = 1; their div = 0 rows then forbid the refilling inflow and the channel
    // stands frozen forever — observed). Air-ADJACENT cells are exempt: the partially-filled
    // surface band legitimately sits below rest density, and sucking on it pumps the surface.
    // Neighbor air test reads .w ≥ 0.5: both CELL_AIR and a same-pass CELL_POCKET upgrade
    // qualify, so the in-flight upgrade race cannot change the outcome (determinism).
    if (cm.y > 0.0 && cm.w < 0.5) {
        var near_air = false;
        for (var a = 0; a < 3; a = a + 1) {
            for (var s = -1; s <= 1; s = s + 2) {
                var nb = cc;
                nb[a] = nb[a] + s;
                if (cell_in_range(nb) && cell_meta[cell_index(nb)].w >= 0.5) {
                    near_air = true;
                }
            }
        }
        // params.dbg.x gates the suction on/off (production = on; the stirring isolation gate
        // disables it). The deficit-driven suction is the original un-banded feedback — the
        // settled-pool stirring fix is the G2P global PIC blend (PIC_BLEND_DEFAULT), shipped now
        // that the crater gate asserts persistence (a held crater is correct, so the blend is safe).
        let deficit = min(cm.z / params.extra.x - 1.0, 0.0);
        if (params.dbg.x > 0.5 && !near_air && deficit < 0.0) {
            let s_under = deficit / (DENSITY_RELAX_FRAMES * params.dt);
            cell_meta[c].x = cm.x + cm.y * s_under / params.dt;
        }
        return;
    }
    if (cm.w != CELL_AIR || label > 0.5) {
        return;
    }
    // Enclosed: compute the cell MIXTURE divergence (same φ-weighted corner gather as
    // cell_classify — one family, U6).
    let h = params.grid_origin.w;
    var div = 0.0;
    for (var oz = 0; oz < 2; oz = oz + 1) {
        for (var oy = 0; oy < 2; oy = oy + 1) {
            for (var ox = 0; ox < 2; ox = ox + 1) {
                let s = corner_sign(vec3<i32>(ox, oy, oz));
                let nflat = node_index(cc + vec3<i32>(ox, oy, oz));
                let gv = grid_vel[nflat];
                let phin = nm[2u * nflat + 1u].w;
                div = div + dot(s, phin * gv.xyz) / (4.0 * h);
            }
        }
    }
    cell_meta[c] = vec4<f32>((0.0 - div) / params.dt, 1.0, 0.0, CELL_POCKET);
    bubble[2] = 1.0; // an enclosed pocket exists: arm the bubble-row solves this frame
    // Append this fine pocket cell to the compacted list (slot 0 is the count).
    let slot = atomicAdd(&pocket_f[0], 1u);
    atomicStore(&pocket_f[slot + 1u], c);
}

// =================================== bubble row relaxation =====================================
// Shared single-workgroup reduction shape for both levels: each thread grid-strides the cell
// range accumulating the pocket row sums (Σ rhs, Σ (A p̃)_c, Σ (A·1_pocket)_c), tree-reduce,
// thread 0 solves the single-unknown row exactly. Runs BEFORE each Jacobi sweep, so the sweep
// writes the fresh multiplier into the pocket slots (the Dirichlet copy).
var<workgroup> red_rhs: array<f32, 256>;
var<workgroup> red_ap: array<f32, 256>;
var<workgroup> red_diag: array<f32, 256>;

@compute @workgroup_size(256)
fn bubble_fine(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // No open-cavity early-out here: a storage-buffer-gated `return` BEFORE a workgroupBarrier()
    // is a Tint uniformity violation (Tint treats every read_write storage load as non-uniform,
    // regardless of the value being equal across threads — native naga tolerates it, the browser
    // rejects it and fails the whole module). The no-pocket case is already handled for free:
    // when no pocket exists `pocket_f[0]` is 0, so `trips` is 0, the reduction sums zeros, and the
    // thread-0 solve's `red_diag > 1e-12` guard yields λ_b = 0 — identical to the old early-out.
    // U8 R9: stride the COMPACTED pocket list (a few hundred entries) not all ~76k fine cells.
    // `count` is uniform across the workgroup (single atomic load, no in-flight append —
    // pocket_mark finished last frame-pass), so the trip count stays uniform and every
    // workgroupBarrier() below is reached from uniform control flow.
    let count = atomicLoad(&pocket_f[0]);
    let h = params.grid_origin.w;
    var s_rhs = 0.0;
    var s_ap = 0.0;
    var s_diag = 0.0;
    let trips = (count + 255u) / 256u; // uniform trip count (barrier discipline)
    for (var it = 0u; it < trips; it = it + 1u) {
        let li = it * 256u + t;
        if (li < count) {
            let c = atomicLoad(&pocket_f[li + 1u]);
            let cc = vec3<i32>(cell_coords(c));
            var acc_p = 0.0; // (D M̃⁻¹ G p̃)_c
            var acc_i = 0.0; // (D M̃⁻¹ G 1_pocket)_c — aggregated diagonal incl. cross terms
            for (var oz = 0; oz < 2; oz = oz + 1) {
                for (var oy = 0; oy < 2; oy = oy + 1) {
                    for (var ox = 0; ox < 2; ox = ox + 1) {
                        let o = vec3<i32>(ox, oy, oz);
                        let node = cc + o;
                        let nidx = node_index(node);
                        let m = Minv(nm[2u * nidx + 0u], nm[2u * nidx + 1u]);
                        var gp = vec3<f32>(0.0);
                        var gi = vec3<f32>(0.0);
                        for (var qz = 0; qz < 2; qz = qz + 1) {
                            for (var qy = 0; qy < 2; qy = qy + 1) {
                                for (var qx = 0; qx < 2; qx = qx + 1) {
                                    let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                                    if (!cell_in_range(c2)) {
                                        continue;
                                    }
                                    let ci2 = cell_index(c2);
                                    let cm2 = cell_meta[ci2];
                                    if (cm2.y <= 0.0) {
                                        continue;
                                    }
                                    let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                                    gp = gp - s2 * cm2.y * pf_src[ci2] / (4.0 * h);
                                    if (cm2.w == CELL_POCKET) {
                                        gi = gi - s2 * cm2.y / (4.0 * h);
                                    }
                                }
                            }
                        }
                        let s = corner_sign(o);
                        // φ_f node weight — same A = −D·Φ·M̃⁻¹·G family as the sweeps (U6).
                        acc_p = acc_p + m.b.w * dot(s, minv_apply(m, gp)) / (4.0 * h);
                        acc_i = acc_i + m.b.w * dot(s, minv_apply(m, gi)) / (4.0 * h);
                    }
                }
            }
            s_rhs = s_rhs + cell_meta[c].x;
            s_ap = s_ap + (-acc_p); // row weight f = 1
            s_diag = s_diag + (-acc_i);
        }
    }
    red_rhs[t] = s_rhs;
    red_ap[t] = s_ap;
    red_diag[t] = s_diag;
    workgroupBarrier();
    var off = 128u;
    loop {
        if (t < off) {
            red_rhs[t] = red_rhs[t] + red_rhs[t + off];
            red_ap[t] = red_ap[t] + red_ap[t + off];
            red_diag[t] = red_diag[t] + red_diag[t + off];
        }
        workgroupBarrier();
        if (off == 1u) {
            break;
        }
        off = off / 2u;
    }
    if (t == 0u) {
        // Sub-resolution release (header: MIN_POCKET_FINE_CELLS): a pocket too small to be a
        // resolved incompressible bubble is open air — λ = 0, present-flag cleared (so the next
        // frame's bubble solves early-out and pocket_present() reads false). Guards the residual
        // V60-cone draining-film entrainment without touching the large pour cavity.
        if (count < MIN_POCKET_FINE_CELLS) {
            bubble[0] = 0.0;
            bubble[1] = 0.0;
            bubble[2] = 0.0;
            return;
        }
        // Fold the prolongated coarse correction into the scalar (exact for one scalar), then
        // solve the aggregated row exactly given the current fluid pressure.
        var lam = bubble[0] + bubble[1];
        bubble[1] = 0.0;
        if (red_diag[0] > 1.0e-12) {
            lam = lam + (red_rhs[0] - red_ap[0]) / red_diag[0];
        } else {
            lam = 0.0; // no coupled pocket faces: the bubble is inactive (p = 0 air)
        }
        bubble[0] = lam;
    }
}

// The identical step on the COARSE error equation: pocket-coarse cells (cmeta.w, set by
// coarse_cell_setup: every active child a pocket cell — the rediscretization doctrine keeps
// mixed boundary cells as fluid rows, smoothed at fine level) share the correction δλ_b.
@compute @workgroup_size(256)
fn bubble_coarse(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    // No storage-gated early return before the barriers (Tint uniformity — see bubble_fine). The
    // no-pocket case falls out for free: count 0 ⇒ trips 0 ⇒ zero reduction ⇒ δλ_b = 0.
    // U8 R9: stride the COMPACTED coarse pocket list (see bubble_fine). `count` is uniform.
    let count = atomicLoad(&pocket_c[0]);
    let hc = params.grid_origin.w * f32(params.coarse_dims.w);
    var s_rhs = 0.0;
    var s_ap = 0.0;
    var s_diag = 0.0;
    let trips = (count + 255u) / 256u;
    for (var it = 0u; it < trips; it = it + 1u) {
        let li = it * 256u + t;
        if (li < count) {
            let c = atomicLoad(&pocket_c[li + 1u]);
            let cc = vec3<i32>(ccell_coords(c));
            var acc_p = 0.0;
            var acc_i = 0.0;
            for (var oz = 0; oz < 2; oz = oz + 1) {
                for (var oy = 0; oy < 2; oy = oy + 1) {
                    for (var ox = 0; ox < 2; ox = ox + 1) {
                        let o = vec3<i32>(ox, oy, oz);
                        let node = cc + o;
                        let nidx = cnode_index(vec3<u32>(node));
                        let m = Minv(nm_c[2u * nidx + 0u], nm_c[2u * nidx + 1u]);
                        var gp = vec3<f32>(0.0);
                        var gi = vec3<f32>(0.0);
                        for (var qz = 0; qz < 2; qz = qz + 1) {
                            for (var qy = 0; qy < 2; qy = qy + 1) {
                                for (var qx = 0; qx < 2; qx = qx + 1) {
                                    let c2 = node - vec3<i32>(1) + vec3<i32>(qx, qy, qz);
                                    if (!ccell_in_range(c2)) {
                                        continue;
                                    }
                                    let ci2 = ccell_index(c2);
                                    let cm2 = cmeta[ci2];
                                    if (cm2.y <= 0.0) {
                                        continue;
                                    }
                                    let s2 = corner_sign(vec3<i32>(1) - vec3<i32>(qx, qy, qz));
                                    gp = gp - s2 * cm2.y * pc_src[ci2] / (4.0 * hc);
                                    if (cm2.w == CELL_POCKET) {
                                        gi = gi - s2 * cm2.y / (4.0 * hc);
                                    }
                                }
                            }
                        }
                        let s = corner_sign(o);
                        acc_p = acc_p + m.b.w * dot(s, minv_apply(m, gp)) / (4.0 * hc);
                        acc_i = acc_i + m.b.w * dot(s, minv_apply(m, gi)) / (4.0 * hc);
                    }
                }
            }
            s_rhs = s_rhs + cmeta[c].x;
            s_ap = s_ap + (-acc_p);
            s_diag = s_diag + (-acc_i);
        }
    }
    red_rhs[t] = s_rhs;
    red_ap[t] = s_ap;
    red_diag[t] = s_diag;
    workgroupBarrier();
    var off = 128u;
    loop {
        if (t < off) {
            red_rhs[t] = red_rhs[t] + red_rhs[t + off];
            red_ap[t] = red_ap[t] + red_ap[t + off];
            red_diag[t] = red_diag[t] + red_diag[t + off];
        }
        workgroupBarrier();
        if (off == 1u) {
            break;
        }
        off = off / 2u;
    }
    if (t == 0u) {
        // Sub-resolution release on the SAME authoritative fine count as bubble_fine: a pocket
        // too small to be a resolved bubble injects no coarse correction (it is open air).
        if (atomicLoad(&pocket_f[0]) < MIN_POCKET_FINE_CELLS) {
            bubble[1] = 0.0;
            return;
        }
        if (red_diag[0] > 1.0e-12) {
            bubble[1] = bubble[1] + (red_rhs[0] - red_ap[0]) / red_diag[0];
        } else {
            bubble[1] = 0.0;
        }
    }
}

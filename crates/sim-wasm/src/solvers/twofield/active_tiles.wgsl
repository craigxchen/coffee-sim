// Phase-1 active-cell scheduler for the repeated fine-grid relaxation passes.
//
// The dense setup/clear/node/coarse passes remain authoritative. This scheduler only replaces
// repeated per-cell relaxation scans when the host has selected the semi-implicit, explicitly
// vented production path. One workgroup owns one 8^3 cell tile: its 256 lanes process `lane`
// and `lane + 256`. Lists are deduplicated with atomic flags, and the host copies the two atomic
// counts into separate INDIRECT-only dispatch records between compute passes. The scheduler is
// never itself used as an indirect-argument buffer (a WebGPU usage/visibility requirement).

const ACTIVE_TILE_EDGE: u32 = 8u;
const ACTIVE_TILE_VOLUME: u32 = 512u;
const ACTIVE_WATER_COUNT: u32 = 0u;
const ACTIVE_SOLID_COUNT: u32 = 1u;

// Layout, in u32 words, for T = active_tile_count():
//   [water_count, solid_count,
//    water_flags[T], solid_flags[T], water_list[T], solid_list[T]].
// Every word is atomic because WGSL does not allow atomic and plain elements in one runtime
// array. Relaxation passes only atomicLoad the compact lists.
@group(0) @binding(33) var<storage, read_write> active_tiles: array<atomic<u32>>;

fn active_tile_dims() -> vec3<u32> {
    return (fine_cells() + vec3<u32>(ACTIVE_TILE_EDGE - 1u)) / ACTIVE_TILE_EDGE;
}

fn active_tile_count() -> u32 {
    let dims = active_tile_dims();
    return dims.x * dims.y * dims.z;
}

fn active_tile_flat(tile: vec3<u32>) -> u32 {
    let dims = active_tile_dims();
    return tile.x + dims.x * (tile.y + dims.y * tile.z);
}

fn active_tile_coords(flat: u32) -> vec3<u32> {
    let dims = active_tile_dims();
    return vec3<u32>(flat % dims.x, (flat / dims.x) % dims.y, flat / (dims.x * dims.y));
}

fn active_water_flag_index(tile: u32) -> u32 {
    return 2u + tile;
}

fn active_solid_flag_index(tile: u32) -> u32 {
    return 2u + active_tile_count() + tile;
}

fn active_water_list_index(slot: u32) -> u32 {
    return 2u + 2u * active_tile_count() + slot;
}

fn active_solid_list_index(slot: u32) -> u32 {
    return 2u + 3u * active_tile_count() + slot;
}

// Weak compare-exchange is retried while the flag is still zero; a spurious failure therefore
// cannot drop the only mark for a tile. Once another invocation owns the flag, no list append is
// needed. Counts cannot exceed T because each flag has exactly one successful 0 -> 1 transition.
fn active_mark_tile(tile: vec3<i32>, solid: bool) {
    let tile_u = vec3<u32>(tile);
    let tile_id = active_tile_flat(tile_u);
    let flag_index = select(active_water_flag_index(tile_id), active_solid_flag_index(tile_id), solid);
    loop {
        let exchanged = atomicCompareExchangeWeak(&active_tiles[flag_index], 0u, 1u);
        if (exchanged.exchanged) {
            let counter_index = select(ACTIVE_WATER_COUNT, ACTIVE_SOLID_COUNT, solid);
            let slot = atomicAdd(&active_tiles[counter_index], 1u);
            let list_index = select(active_water_list_index(slot), active_solid_list_index(slot), solid);
            atomicStore(&active_tiles[list_index], tile_id);
            return;
        }
        if (exchanged.old_value != 0u) {
            return;
        }
    }
}

// Mark every tile intersecting the inclusive cell-space box. Marking the tile-space bounding box
// is exactly the same set as marking each cell and deduplicating afterwards, but avoids hundreds
// of redundant atomics for the conservative solid-motion halo.
fn active_mark_cell_box(lo_arg: vec3<i32>, hi_arg: vec3<i32>, solid: bool) {
    let last_cell = vec3<i32>(fine_cells()) - vec3<i32>(1);
    let lo = clamp(lo_arg, vec3<i32>(0), last_cell);
    let hi = clamp(hi_arg, vec3<i32>(0), last_cell);
    let tile_lo = lo / i32(ACTIVE_TILE_EDGE);
    let tile_hi = hi / i32(ACTIVE_TILE_EDGE);
    for (var tz = tile_lo.z; tz <= tile_hi.z; tz = tz + 1) {
        for (var ty = tile_lo.y; ty <= tile_hi.y; ty = ty + 1) {
            for (var tx = tile_lo.x; tx <= tile_hi.x; tx = tx + 1) {
                active_mark_tile(vec3<i32>(tx, ty, tz), solid);
            }
        }
    }
}

// One marker dispatch covers the live water prefix and solid suffix of the pool-padded layout.
// Water uses the cell-presence cell plus a one-cell operator halo. Solids use their clamped 3^3
// B-spline node base and the exact corner-cell support [base-1, base+2], expanded by the maximum
// distance a solid can move during this macro step. Paper-held and open-base escaped water match
// the p2g_water mechanics guards exactly.
@compute @workgroup_size(256)
fn active_tile_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_count) {
        return;
    }
    let h = params.grid_origin.w;
    let x = pos[p].xyz;
    let xl = (x - params.grid_origin.xyz) / h;
    if (p < params.water_count) {
        if (chem[p].z > 0.5) {
            return;
        }
        if (params.coupling.w > 0.5 && x.y < params.box_min.y) {
            return;
        }
        let ci = clamp(
            vec3<i32>(floor(xl)),
            vec3<i32>(0),
            vec3<i32>(params.grid_dims.xyz) - vec3<i32>(2),
        );
        active_mark_cell_box(ci - vec3<i32>(1), ci + vec3<i32>(1), false);
        return;
    }

    let solid_start = params.particle_count - params.solid_count;
    if (p < solid_start) {
        return; // dormant water-pool capacity between the live prefix and solid suffix
    }
    var base = vec3<i32>(floor(xl - vec3<f32>(0.5)));
    base = clamp(base, vec3<i32>(0), vec3<i32>(params.grid_dims.xyz) - vec3<i32>(3));
    let motion_pad = i32(ceil(max(params.max_speed, 0.0) * max(params.dt, 0.0) / h));
    active_mark_cell_box(
        base - vec3<i32>(1 + motion_pad),
        base + vec3<i32>(2 + motion_pad),
        true,
    );
}

fn active_tile_list_value(workgroup: u32, solid: bool) -> u32 {
    let index = select(active_water_list_index(workgroup), active_solid_list_index(workgroup), solid);
    return atomicLoad(&active_tiles[index]);
}

fn active_tile_cell(tile_id: u32, lane: u32) -> vec3<i32> {
    let tile = active_tile_coords(tile_id);
    let local = vec3<u32>(
        lane % ACTIVE_TILE_EDGE,
        (lane / ACTIVE_TILE_EDGE) % ACTIVE_TILE_EDGE,
        lane / (ACTIVE_TILE_EDGE * ACTIVE_TILE_EDGE),
    );
    return vec3<i32>(tile * ACTIVE_TILE_EDGE + local);
}

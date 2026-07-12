// Phase-1 active-cell scheduler for repeated water fine-pressure relaxation.
//
// Dense setup/clear/node/coarse passes and every solid-compaction pass remain authoritative.
// On the guarded semi-implicit, explicitly vented production path, one workgroup owns one 8^3
// water-cell tile: its 256 lanes process `lane` and `lane + 256`. The host copies the atomic
// compact-list count into a separate INDIRECT-only dispatch record between compute passes; the
// writable scheduler is never itself used as an indirect-argument buffer.

const ACTIVE_TILE_EDGE: u32 = 8u;
const ACTIVE_WATER_COUNT: u32 = 0u;

// Layout, in u32 words, for T = active_tile_count():
//   [water_count, water_flags[T], water_list[T]].
// Every word is atomic because WGSL does not allow atomic and plain elements in one runtime
// array. Relaxation passes only atomicLoad the compact list.
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
    return 1u + tile;
}

fn active_water_list_index(slot: u32) -> u32 {
    return 1u + active_tile_count() + slot;
}

// Weak compare-exchange is retried while the flag is still zero; a spurious failure therefore
// cannot drop the only mark for a tile. Once another invocation owns the flag, no list append is
// needed. The count cannot exceed T because each flag has exactly one successful 0 -> 1 transition.
fn active_mark_tile(tile: vec3<i32>) {
    let tile_id = active_tile_flat(vec3<u32>(tile));
    let flag_index = active_water_flag_index(tile_id);
    loop {
        let exchanged = atomicCompareExchangeWeak(&active_tiles[flag_index], 0u, 1u);
        if (exchanged.exchanged) {
            let slot = atomicAdd(&active_tiles[ACTIVE_WATER_COUNT], 1u);
            atomicStore(&active_tiles[active_water_list_index(slot)], tile_id);
            return;
        }
        if (exchanged.old_value != 0u) {
            return;
        }
    }
}

// Mark every tile intersecting the inclusive cell-space box. Marking the tile-space bounding box
// is exactly the same set as marking each cell and deduplicating afterwards.
fn active_mark_cell_box(lo_arg: vec3<i32>, hi_arg: vec3<i32>) {
    let last_cell = vec3<i32>(fine_cells()) - vec3<i32>(1);
    let lo = clamp(lo_arg, vec3<i32>(0), last_cell);
    let hi = clamp(hi_arg, vec3<i32>(0), last_cell);
    let tile_lo = lo / i32(ACTIVE_TILE_EDGE);
    let tile_hi = hi / i32(ACTIVE_TILE_EDGE);
    for (var tz = tile_lo.z; tz <= tile_hi.z; tz = tz + 1) {
        for (var ty = tile_lo.y; ty <= tile_hi.y; ty = ty + 1) {
            for (var tx = tile_lo.x; tx <= tile_hi.x; tx = tx + 1) {
                active_mark_tile(vec3<i32>(tx, ty, tz));
            }
        }
    }
}

// The marker is dispatched over the live water prefix only. It uses the particle-presence cell
// plus the one-cell pressure-operator halo. Paper-held and open-base escaped water match the
// p2g_water mechanics guards exactly.
@compute @workgroup_size(256)
fn active_tile_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.water_count) {
        return;
    }
    if (chem[p].z > 0.5) {
        return;
    }
    let x = pos[p].xyz;
    if (params.coupling.w > 0.5 && x.y < params.box_min.y) {
        return;
    }
    let xl = (x - params.grid_origin.xyz) / params.grid_origin.w;
    let ci = clamp(
        vec3<i32>(floor(xl)),
        vec3<i32>(0),
        vec3<i32>(params.grid_dims.xyz) - vec3<i32>(2),
    );
    active_mark_cell_box(ci - vec3<i32>(1), ci + vec3<i32>(1));
}

fn active_tile_list_value(workgroup: u32) -> u32 {
    return atomicLoad(&active_tiles[active_water_list_index(workgroup)]);
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

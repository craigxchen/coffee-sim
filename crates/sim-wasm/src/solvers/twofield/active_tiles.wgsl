// Phase-1 active-cell scheduler for repeated water fine-pressure relaxation.
//
// Dense setup/clear/node/coarse passes and every solid-compaction pass remain authoritative.
// On the guarded semi-implicit, explicitly vented production path, one workgroup owns one 8^3
// water-cell tile: its 256 lanes inspect `lane` and `lane + 256` in the cell-presence census
// produced by p2g_water. Only leaders of occupied tiles touch the global scheduler, marking a
// conservative 3^3 neighboring-tile halo. The host copies the atomic compact-list count into a
// separate INDIRECT-only dispatch record between compute passes; the writable scheduler is never
// itself used as an indirect-argument buffer.

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

var<workgroup> active_tile_occupied: atomic<u32>;

// One workgroup scans one 8^3 cell tile. p2g_water is the sole writer of the presence census, so
// held paper parcels and open-base escaped water are already excluded exactly. A one-TILE halo is
// deliberately conservative for the required one-CELL operator halo [ci-1, ci+1], including
// occupancy on a tile face/edge/corner. No invocation returns before the workgroup barriers.
@compute @workgroup_size(256)
fn active_tile_mark(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if (lane == 0u) {
        atomicStore(&active_tile_occupied, 0u);
    }
    workgroupBarrier();

    let tile_id = workgroup.x;
    let cell0 = active_tile_cell(tile_id, lane);
    let cell1 = active_tile_cell(tile_id, lane + 256u);
    var occupied = false;
    if (cell_in_range(cell0)) {
        occupied = occupied || atomicLoad(&cell_cnt[cell_index(cell0)]) > 0u;
    }
    if (cell_in_range(cell1)) {
        occupied = occupied || atomicLoad(&cell_cnt[cell_index(cell1)]) > 0u;
    }
    if (occupied) {
        atomicOr(&active_tile_occupied, 1u);
    }
    workgroupBarrier();

    if (lane == 0u && atomicLoad(&active_tile_occupied) != 0u) {
        let center = vec3<i32>(active_tile_coords(tile_id));
        let dims = vec3<i32>(active_tile_dims());
        for (var dz = -1; dz <= 1; dz = dz + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    let neighbor = center + vec3<i32>(dx, dy, dz);
                    if (all(neighbor >= vec3<i32>(0)) && all(neighbor < dims)) {
                        active_mark_tile(neighbor);
                    }
                }
            }
        }
    }
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

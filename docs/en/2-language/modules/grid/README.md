# `grid` Module — Flat Buffers and Grid A*

Built-in VM module for **dense arrays** (`Vec<i32>` / `Vec<u8>`) outside object maps and native A* with buffer reuse. Used for A* on large grids (for example 1000×5000) without gigabyte-sized `dict`/`set`.

## Import

```datacode
import grid
```

## Buffers

| Function | Description |
|----------|-------------|
| `alloc_i32(n, fill)` | Buffer of `n` i32 elements → opaque handle |
| `alloc_u8(n, fill)` | Buffer of `n` bytes (closed or bitmap) |
| `fill_i32(buf, fill)` / `fill_u8(buf, fill)` | Fast reset between runs |
| `get_i32` / `set_i32` / `get_u8` / `set_u8` | O(1) access by linear id |
| `test_blocked(bitmap, id)` / `set_blocked(bitmap, id)` | Obstacle bitmask |
| `bitmap_bytes(n_cells)` | Bitmap size in bytes |
| `shrink(buf)` | `shrink_to_fit` after stress |
| `store_len()` | VM ValueStore length (diagnostics) |

Handles are `PluginOpaque` (not serialized in bytecode constants).

## A*

| Function | Description |
|----------|-------------|
| `astar(rows, cols, start, goal, blocked, g, f, parent, closed)` | Native A* (<1 s on 1000×5000) |
| `astar_from_set(rows, cols, start, goal, blocked_set)` | Convenience wrapper + bitmap from set |
| `bitmap_from_ids(n_cells, ids_array)` | Build bitmap from id array |

## Example (buffer reuse)

```datacode
import grid

rows, cols = 1000, 5000
n = rows * cols


blocked = grid.alloc_u8(grid.bitmap_bytes(n), 0)
g = grid.alloc_i32(n, INF)
f = grid.alloc_i32(n, 0)
parent = grid.alloc_i32(n, -1)
closed = grid.alloc_u8(n, 0)

for run in range(10) {
    path = grid.astar(rows, cols, (0, 0), (559, 1234), blocked, g, f, parent, closed)
}
```

Full stress example (Russian only for now): `examples/ru/09-продвинутые/структуры данных/графы/advanced/a_start_buffers.dc`. See also [1-examples/09-advanced](../../1-examples/09-advanced.md).

## Path comparison

| Path | RAM (1000×5000) | Time |
|------|-----------------|------|
| `a_start.dc` (dict/set/heapq) | ~7 GiB RSS | ~45 s |
| `grid.astar` / `pathfind.astar_grid` | ~100–200 MiB | <1 s |
| DC loop on `grid.get/set` | ~100–250 MiB | seconds–tens of seconds |

## Benchmarks

```bash
/usr/bin/time -l cargo test bench_astar_native_1000x5000 --release -- --ignored --nocapture --test-threads=1
/usr/bin/time -l cargo test bench_astar_grid_buffers_1000x5000 --release -- --ignored --nocapture --test-threads=1
DC_ASTAR_RUNS=10 /usr/bin/time -l target/release/datacode \
  "examples/ru/09-продвинутые/структуры данных/графы/advanced/a_start_buffers.dc"
```

After stress: `system.runtime.trim_allocator()` (Linux) or `DC_TRIM_ARENA=1` in the example.

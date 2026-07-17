# `pathfind` Module — Grid Pathfinding

Built-in VM module for A* on a grid with Manhattan heuristic. Implemented in Rust on flat arrays (low RAM compared to a DataCode version using `dict`/`set`/`heapq`).

## Import

```datacode
import pathfind
```

## API

| Function | Description |
|----------|-------------|
| `astar_grid(rows, cols, start, goal, blocked)` | A* on 4-connected grid (blocked → compact bitmap internally) |

- `start`, `goal` — `(row, col)` or `[row, col]`
- `blocked` — `set` of `(r, c)` coordinates or linear ids `r * cols + c`
- Returns array of `(r, c)` from start to goal or `null`

Scratch buffers (`g_score`, `f_score`, …) are reused between calls in the thread (thread-local pool). After long stress runs: `system.runtime.trim_allocator()`.

## Example

```datacode
import pathfind

blocked = set()
blocked.add((5, 5))
path = pathfind.astar_grid(100, 100, (0, 0), (99, 99), blocked)
print(len(path))
```

## Comparison with DataCode A*

On a 1000×5000 grid the native version uses ~50–150 MiB RAM and runs in fractions of a second, while the interpreted version with hash containers — on the order of 3–5 GiB and tens of seconds.

Benchmark:

```bash
/usr/bin/time -l cargo test bench_astar_native_1000x5000 --release -- --ignored --nocapture --test-threads=1
```

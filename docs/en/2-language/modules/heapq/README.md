# Module `heapq`

Built-in **min-heap** (priority queue) on an ordinary array. Same role as Python’s [`heapq`](https://docs.python.org/3/library/heapq.html): push/pop in \(O(\log n)\), peek in \(O(1)\), heapify in \(O(n)\).

## Import

```datacode
import heapq
```

## API

| Function | Description |
|----------|-------------|
| `heapq.heappush(heap, item)` | Append `item` and restore heap order. |
| `heapq.heappop(heap)` | Remove and return the smallest item. Empty heap → `IndexError`. |
| `heapq.heapify(heap)` | Turn `heap` into a min-heap in linear time. |
| `heapq.heappeek(heap)` | Return smallest item without removing it. Empty → `IndexError`. |
| `heapq.heapreplace(heap, item)` | Pop smallest, push `item`, one sift-down (faster than pop+push). Empty → `IndexError`. |

The first argument must be an **array** (`[]`). Elements compare using the same ordering as operators `<` / `<=` in DataCode: numbers (including mixed `int`/`float`), strings, dates, durations, booleans, `null`, and **lexicographic tuples** (nested tuples supported). Mixed incomparable types raise `TypeError`.

## Example (A\*-style priorities)

```datacode
import heapq

open_heap = []
heapq.heappush(open_heap, (10, 1))
heapq.heappush(open_heap, (5, 7))
heapq.heappush(open_heap, (7, 3))

while len(open_heap) > 0 {
    current = heapq.heappop(open_heap)
    print(current)
}
```

Tuple order is lexicographic: `(f_score, node_id)` behaves like typical A\* / Dijkstra queues.

## Complexity

| Operation | Time |
|-----------|------|
| `heappush` | \(O(\log n)\) |
| `heappop` | \(O(\log n)\) |
| `heappeek` | \(O(1)\) |
| `heapify` | \(O(n)\) |
| `heapreplace` | \(O(\log n)\) |

Implementation uses sift-up / sift-down only (no full sorts).

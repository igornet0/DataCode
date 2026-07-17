# Модуль `grid` — плоские буферы и A* на сетке

Встроенный модуль VM для **плотных массивов** (`Vec<i32>` / `Vec<u8>`) вне object map и native A* с переиспользованием буферов. Используется для A* на больших сетках (например 1000×5000) без гигабайтных `dict`/`set`.

## Импорт

```datacode
import grid
```

## Буферы

| Функция | Описание |
|---------|----------|
| `alloc_i32(n, fill)` | Буфер `n` элементов i32 → opaque handle |
| `alloc_u8(n, fill)` | Буфер `n` байт (closed или bitmap) |
| `fill_i32(buf, fill)` / `fill_u8(buf, fill)` | Быстрый reset между прогонами |
| `get_i32` / `set_i32` / `get_u8` / `set_u8` | O(1) доступ по linear id |
| `test_blocked(bitmap, id)` / `set_blocked(bitmap, id)` | Битовая маска препятствий |
| `bitmap_bytes(n_cells)` | Размер bitmap в байтах |
| `shrink(buf)` | `shrink_to_fit` после stress |
| `store_len()` | Длина VM ValueStore (диагностика) |

Handles — `PluginOpaque` (не сериализуются в bytecode constants).

## A*

| Функция | Описание |
|---------|----------|
| `astar(rows, cols, start, goal, blocked, g, f, parent, closed)` | Native A* (<1 с на 1000×5000) |
| `astar_from_set(rows, cols, start, goal, blocked_set)` | Удобная обёртка + bitmap из set |
| `bitmap_from_ids(n_cells, ids_array)` | Построить bitmap из массива id |

## Пример (переиспользование буферов)

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

Полный stress-пример: [`examples/ru/09-продвинутые/структуры данных/графы/advanced/a_start_buffers.dc`](../../../../../examples/ru/09-продвинутые/структуры%20данных/графы/advanced/a_start_buffers.dc).

## Сравнение путей

| Путь | RAM (1000×5000) | Время |
|------|-----------------|-------|
| `a_start.dc` (dict/set/heapq) | ~7 GiB RSS | ~45 с |
| `grid.astar` / `pathfind.astar_grid` | ~100–200 MiB | <1 с |
| DC-цикл на `grid.get/set` | ~100–250 MiB | секунды–десятки секунд |

## Бенчмарки

```bash
/usr/bin/time -l cargo test bench_astar_native_1000x5000 --release -- --ignored --nocapture --test-threads=1
/usr/bin/time -l cargo test bench_astar_grid_buffers_1000x5000 --release -- --ignored --nocapture --test-threads=1
DC_ASTAR_RUNS=10 /usr/bin/time -l target/release/datacode \
  "examples/ru/09-продвинутые/структуры данных/графы/advanced/a_start_buffers.dc"
```

После stress: `system.runtime.trim_allocator()` (Linux) или `DC_TRIM_ARENA=1` в примере.

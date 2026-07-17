# Модуль `pathfind` — поиск пути на сетке

Встроенный модуль VM для A* на решётке с манхэттенской эвристикой. Реализован в Rust на плоских массивах (низкое потребление RAM по сравнению с DataCode-версией на `dict`/`set`/`heapq`).

## Импорт

```datacode
import pathfind
```

## API

| Функция | Описание |
|---------|----------|
| `astar_grid(rows, cols, start, goal, blocked)` | A* на 4-связной сетке (blocked → compact bitmap internally) |

- `start`, `goal` — `(row, col)` или `[row, col]`
- `blocked` — `set` координат `(r, c)` или линейных id `r * cols + c`
- Возвращает массив `(r, c)` от start до goal или `null`

Scratch-буферы (`g_score`, `f_score`, …) переиспользуются между вызовами в потоке (thread-local pool). После длинного stress: `system.runtime.trim_allocator()`.

## Пример

```datacode
import pathfind

blocked = set()
blocked.add((5, 5))
path = pathfind.astar_grid(100, 100, (0, 0), (99, 99), blocked)
print(len(path))
```

## Сравнение с DataCode A*

На сетке 1000×5000 native-версия использует ~50–150 MiB RAM и выполняется за доли секунды, тогда как интерпретируемая версия с hash-контейнерами — порядка 3–5 GiB и десятки секунд.

Бенчмарк:

```bash
/usr/bin/time -l cargo test bench_astar_native_1000x5000 --release -- --ignored --nocapture --test-threads=1
```

# Модуль `heapq`

Встроенная **min-куча** (очередь с приоритетом) поверх обычного массива. По назначению близка к [`heapq`](https://docs.python.org/3/library/heapq.html) в Python: вставка/извлечение за \(O(\log n)\), просмотр минимума за \(O(1)\), `heapify` за \(O(n)\).

## Импорт

```datacode
import heapq

```

## API

| Функция | Описание |
|---------|----------|
| `heapq.heappush(heap, item)` | Добавить элемент и восстановить порядок кучи. |
| `heapq.heappop(heap)` | Удалить и вернуть минимальный элемент. Пустая куча → `IndexError`. |
| `heapq.heapify(heap)` | Преобразовать массив в min-кучу за линейное время. |
| `heapq.heappeek(heap)` | Вернуть минимум без удаления. Пустая куча → `IndexError`. |
| `heapq.heapreplace(heap, item)` | Эквивалент pop минимума + push, но эффективнее (один sift-down). Пустая куча → `IndexError`. |

Первый аргумент — **массив** (`[]`). Сравнение элементов совпадает с операторами `<` / `<=` в DataCode: числа (в том числе смесь `int`/`float`), строки, даты, длительности, булевы значения, `null`, а также **лексикографические кортежи** (вложенные кортежи поддерживаются). Несовместимые типы дают `TypeError`.

## Пример (приоритеты как в A*)

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

Порядок кортежей лексикографический: пара `(f_score, node_id)` подходит для A*, Дейкстры и др.

## Сложность

| Операция | Время |
|----------|-------|
| `heappush` | \(O(\log n)\) |
| `heappop` | \(O(\log n)\) |
| `heappeek` | \(O(1)\) |
| `heapify` | \(O(n)\) |
| `heapreplace` | \(O(\log n)\) |

Реализовано через sift-up / sift-down без полной сортировки массива.

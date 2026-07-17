import heapq
import random
from typing import Optional, Tuple, Set, List


def manhattan(id1: int, id2: int, cols: int) -> int:
    """Манхэттенское расстояние между двумя узлами (по их id)."""
    r1, c1 = divmod(id1, cols)
    r2, c2 = divmod(id2, cols)
    return abs(r1 - r2) + abs(c1 - c2)


def a_star_grid(
    rows: int,
    cols: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """
    Поиск кратчайшего пути на сетке методом A* с манхэттенской эвристикой.

    Args:
        rows: количество строк в сетке
        cols: количество столбцов
        start: координаты старта (r, c)
        goal: координаты цели (r, c)
        blocked: множество координат заблокированных клеток

    Returns:
        Список координат от start до goal включительно, если путь найден,
        иначе None.
    """

    # Преобразуем координаты в id (одномерные индексы) для быстрой работы
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]

    # Множество заблокированных клеток в виде id (для быстрой проверки)
    blocked_ids = set([r * cols + c for r, c in blocked])

    # Проверка: старт или цель заблокированы
    if start_id in blocked_ids or goal_id in blocked_ids:
        return None

    # Вспомогательные словари для хранения стоимости и предков
    g_score = {start_id: 0}
    f_score = {start_id: manhattan(start_id, goal_id, cols)}

    came_from = {}

    # Приоритетная очередь: (f_score, node_id)
    # В Python heapq — min-heap
    open_heap = [(f_score[start_id], start_id)]

    # Множество для быстрой проверки наличия узла в очереди
    open_set = set([start_id])

    while open_heap:
        # Извлекаем узел с наименьшим f_score
        current_f, current = heapq.heappop(open_heap)

        # Если в очереди осталась устаревшая запись (f_score изменился), пропускаем
        if current_f != f_score.get(current, None):
            continue

        open_set.discard(current)

        # Проверка достижения цели
        if current == goal_id:
            # Восстанавливаем путь
            path = []

            while current in came_from:
                r, c = divmod(current, cols)
                path.append((r, c))
                current = came_from[current]

            r, c = divmod(current, cols)
            path.append((r, c))

            return list(reversed(path))

        # Генерация соседей (4 направления)
        r, c = divmod(current, cols)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # вверх, вниз, влево, вправо

            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = nr * cols + nc

            if neighbor in blocked_ids:
                continue

            # Стоимость перехода = 1 (единичные веса)
            tentative_g = g_score[current] + 1

            # Если нашли лучший путь к соседу
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f = tentative_g + manhattan(neighbor, goal_id, cols)
                f_score[neighbor] = f

                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)

    # Путь не найден
    return None


# =========================================================
# Пример использования
# =========================================================

rows, cols = 10000, 50000

# Создаём множество заблокированных клеток
random.seed(42)

blocked_cells = set()

for _ in range(100_000):
    blocked_cells.add(
        (
            random.randint(0, rows - 1),
            random.randint(0, cols - 1),
        )
    )

start = (0, 0)
goal = (1559, 5234)

# Убедимся, что старт и цель не заблокированы
blocked_cells.discard(start)
blocked_cells.discard(goal)

path = a_star_grid(rows, cols, start, goal, blocked_cells)

if path is not None:
    print("Путь найден! Длина пути:", len(path))
    print("Первые 5 шагов:", path[:5])
    print("Последние 5 шагов:", path[-5:])
else:
    print("Путь не найден")
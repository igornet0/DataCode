import heapq
import random
import statistics
import time
from typing import Optional, Tuple, Set, List


def a_star_grid_optimized(
    rows: int,
    cols: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: Set[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """
    Оптимизированная версия A* для сетки с манхэттенской эвристикой.
    """
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]

    # Заранее вычислим координаты цели для быстрой эвристики
    goal_r, goal_c = goal

    blocked_ids = {r * cols + c for r, c in blocked}
    if start_id in blocked_ids or goal_id in blocked_ids:
        return None

    # Словари для стоимостей и предков
    g_score = {start_id: 0}
    # f_score храним только для проверки при извлечении из кучи
    f_score = {start_id: abs(start[0] - goal_r) + abs(start[1] - goal_c)}

    came_from = {}
    closed_set = set()          # обработанные узлы

    # Приоритетная очередь
    open_heap = [(f_score[start_id], start_id)]
    open_set = {start_id}       # узлы в очереди

    # Дельта для четырёх направлений (локальная переменная для скорости)
    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while open_heap:
        current_f, current = heapq.heappop(open_heap)

        # Пропускаем устаревшие записи
        if current_f != f_score.get(current):
            continue

        open_set.discard(current)

        # Если узел уже обработан – пропускаем (эвристика монотонна)
        if current in closed_set:
            continue

        # Проверка достижения цели
        if current == goal_id:
            # Восстановление пути
            path = []
            while current in came_from:
                r, c = divmod(current, cols)
                path.append((r, c))
                current = came_from[current]
            r, c = divmod(current, cols)
            path.append((r, c))
            return path[::-1]

        closed_set.add(current)

        # Координаты текущего узла
        r, c = divmod(current, cols)

        # Обход соседей
        for dr, dc in neighbors_delta:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = nr * cols + nc
            if neighbor in blocked_ids or neighbor in closed_set: continue

            tentative_g = g_score[current] + 1

            # Если нашли лучший путь к соседу
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # Эвристика: манхэттенское расстояние (inline)
                h = abs(nr - goal_r) + abs(nc - goal_c)
                f = tentative_g + h
                f_score[neighbor] = f

                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)

    return None


# =========================== СТРЕСС-ТЕСТ ===========================

def stress_test(num_runs: int = 100):
    rows, cols = 1000, 5000
    random.seed(42)

    # Генерируем заблокированные клетки
    blocked_cells = set()
    for _ in range(100_00):
        blocked_cells.add((random.randint(0, rows - 1), random.randint(0, cols - 1)))

    start = (0, 0)
    goal = (559, 1234)
    blocked_cells.discard(start)
    blocked_cells.discard(goal)

    total_time = 0.0
    success_count = 0
    times = []

    for i in range(num_runs):
        start_time = time.perf_counter()
        path = a_star_grid_optimized(rows, cols, start, goal, blocked_cells)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        times.append(elapsed)
        if path is not None:
            success_count += 1
        # Небольшой вывод прогресса (опционально)
        if (i + 1) % 10 == 0:
            print(f"Прогон {i+1}/{num_runs} завершён. Текущее среднее: {total_time/(i+1):.4f} сек.")

    print("\n====== РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТА ======")
    print(f"Всего запусков:          {num_runs}")
    print(f"Успешных поисков пути:   {success_count} ({(success_count/num_runs)*100:.1f}%)")
    print(f"Общее время:             {total_time:.2f} сек.")
    print(f"Среднее время на поиск:  {total_time/num_runs:.4f} сек.")
    median = statistics.median(times)
    print(f"Медианное время:         {median:.4f} сек.")


if __name__ == "__main__":
    # Для проверки работоспособности – единичный запуск
    print("Тестовый единичный запуск...")
    rows, cols = 1000, 5000
    blocked = set()
    start = (0, 0)
    goal = (559, 1234)
    path = a_star_grid_optimized(rows, cols, start, goal, blocked)
    if path:
        print(f"Путь найден, длина = {len(path)}")
    else:
        print("Путь не найден")

    # Запуск стресс-теста
    stress_test(10)
# A* на сетке — Python-половина сравнения DC vs Python (issue #22, этап 5).
# Алгоритм и генерация препятствий совпадают с astar.dc построчно.
import heapq
import os
import time


def env_int(name, default):
    v = os.environ.get(name)
    return default if v is None else int(v)


def a_star(rows, cols, start, goal, blocked_ids):
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]
    goal_r, goal_c = goal

    if start_id in blocked_ids or goal_id in blocked_ids:
        return None

    g_score = {}
    f_score = {}
    came_from = {}
    closed_set = set()
    open_heap = []
    open_set = set()

    g_score[start_id] = 0
    f_score[start_id] = abs(start[0] - goal_r) + abs(start[1] - goal_c)
    heapq.heappush(open_heap, (f_score[start_id], start_id))
    open_set.add(start_id)

    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        if current_f != f_score.get(current):
            continue

        open_set.discard(current)
        if current in closed_set:
            continue

        if current == goal_id:
            path = []
            while current in came_from:
                r, c = divmod(current, cols)
                path.append((r, c))
                current = came_from[current]
            r, c = divmod(current, cols)
            path.append((r, c))
            return path[::-1]

        closed_set.add(current)
        r, c = divmod(current, cols)

        for dr, dc in neighbors_delta:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = nr * cols + nc
            if neighbor in blocked_ids or neighbor in closed_set:
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = abs(nr - goal_r) + abs(nc - goal_c)
                f = tentative_g + h
                f_score[neighbor] = f
                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)
    return None


def main():
    rows = env_int("ASTAR_ROWS", 100)
    cols = env_int("ASTAR_COLS", 500)
    n_blocked = env_int("ASTAR_BLOCKED", 1000)
    goal = (env_int("ASTAR_GOAL_R", 50), env_int("ASTAR_GOAL_C", 234))
    runs = env_int("ASTAR_RUNS", 10)
    start = (0, 0)

    # LCG (glibc-константы), чтобы препятствия совпадали с DataCode бит-в-бит.
    x = env_int("ASTAR_SEED", 42)
    blocked_ids = set()
    for _ in range(n_blocked):
        x = (x * 1103515245 + 12345) % 2147483648
        r = x % rows
        x = (x * 1103515245 + 12345) % 2147483648
        c = x % cols
        blocked_ids.add(r * cols + c)
    blocked_ids.discard(start[0] * cols + start[1])
    blocked_ids.discard(goal[0] * cols + goal[1])

    times = []
    path_len = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        path = a_star(rows, cols, start, goal, blocked_ids)
        times.append(time.perf_counter() - t0)
        path_len = 0 if path is None else len(path)

    times.sort()
    print(
        f"RESULT lang=py blocked={len(blocked_ids)} path_len={path_len} runs={runs} "
        f"mean={sum(times) / runs:.5f} median={times[runs // 2]:.5f} min={times[0]:.5f}"
    )


if __name__ == "__main__":
    main()

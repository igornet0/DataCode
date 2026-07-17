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
    Optimized A* for a grid with Manhattan heuristic.
    """
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]

    # Precompute goal coordinates for a fast heuristic
    goal_r, goal_c = goal

    blocked_ids = {r * cols + c for r, c in blocked}
    if start_id in blocked_ids or goal_id in blocked_ids:
        return None

    # Cost and parent maps
    g_score = {start_id: 0}
    # Keep f_score only to validate heap pops
    f_score = {start_id: abs(start[0] - goal_r) + abs(start[1] - goal_c)}

    came_from = {}
    closed_set = set()          # processed nodes

    # Priority queue
    open_heap = [(f_score[start_id], start_id)]
    open_set = {start_id}       # nodes in the queue

    # Deltas for four directions (local for speed)
    neighbors_delta = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while open_heap:
        current_f, current = heapq.heappop(open_heap)

        # Skip stale entries
        if current_f != f_score.get(current):
            continue

        open_set.discard(current)

        # Already processed — skip (heuristic is monotone)
        if current in closed_set:
            continue

        # Goal reached
        if current == goal_id:
            # Reconstruct path
            path = []
            while current in came_from:
                r, c = divmod(current, cols)
                path.append((r, c))
                current = came_from[current]
            r, c = divmod(current, cols)
            path.append((r, c))
            return path[::-1]

        closed_set.add(current)

        # Current node coordinates
        r, c = divmod(current, cols)

        # Visit neighbors
        for dr, dc in neighbors_delta:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = nr * cols + nc
            if neighbor in blocked_ids or neighbor in closed_set: continue

            tentative_g = g_score[current] + 1

            # Found a better path to neighbor
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # Heuristic: Manhattan distance (inline)
                h = abs(nr - goal_r) + abs(nc - goal_c)
                f = tentative_g + h
                f_score[neighbor] = f

                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)

    return None


# =========================== STRESS TEST ===========================

def stress_test(num_runs: int = 100):
    rows, cols = 1000, 5000
    random.seed(42)

    # Generate blocked cells
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
        # Small progress output (optional)
        if (i + 1) % 10 == 0:
            print(f"Run {i+1}/{num_runs} done. Current average: {total_time/(i+1):.4f} sec.")

    print("\n====== STRESS TEST RESULTS ======")
    print(f"Total runs:              {num_runs}")
    print(f"Successful path finds:   {success_count} ({(success_count/num_runs)*100:.1f}%)")
    print(f"Total time:              {total_time:.2f} sec.")
    print(f"Average time per search: {total_time/num_runs:.4f} sec.")
    median = statistics.median(times)
    print(f"Median time:             {median:.4f} sec.")


if __name__ == "__main__":
    # Single run to verify it works
    print("Single test run...")
    rows, cols = 1000, 5000
    blocked = set()
    start = (0, 0)
    goal = (559, 1234)
    path = a_star_grid_optimized(rows, cols, start, goal, blocked)
    if path:
        print(f"Path found, length = {len(path)}")
    else:
        print("Path not found")

    # Run stress test
    stress_test(10)

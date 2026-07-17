import heapq
import random
from typing import Optional, Tuple, Set, List


def manhattan(id1: int, id2: int, cols: int) -> int:
    """Manhattan distance between two nodes (by their ids)."""
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
    Find the shortest path on a grid with A* and Manhattan heuristic.

    Args:
        rows: number of rows in the grid
        cols: number of columns
        start: start coordinates (r, c)
        goal: goal coordinates (r, c)
        blocked: set of blocked cell coordinates

    Returns:
        List of coordinates from start to goal inclusive if a path is found,
        otherwise None.
    """

    # Convert coordinates to ids (1D indices) for fast lookups
    start_id = start[0] * cols + start[1]
    goal_id = goal[0] * cols + goal[1]

    # Blocked cells as ids (for fast membership checks)
    blocked_ids = set([r * cols + c for r, c in blocked])

    # Check: start or goal blocked
    if start_id in blocked_ids or goal_id in blocked_ids:
        return None

    # Helpers for costs and parents
    g_score = {start_id: 0}
    f_score = {start_id: manhattan(start_id, goal_id, cols)}

    came_from = {}

    # Priority queue: (f_score, node_id)
    # In Python heapq is a min-heap
    open_heap = [(f_score[start_id], start_id)]

    # Set for fast membership checks in the open queue
    open_set = set([start_id])

    while open_heap:
        # Pop the node with the smallest f_score
        current_f, current = heapq.heappop(open_heap)

        # Skip stale queue entries (f_score was updated)
        if current_f != f_score.get(current, None):
            continue

        open_set.discard(current)

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

            return list(reversed(path))

        # Generate neighbors (4 directions)
        r, c = divmod(current, cols)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # up, down, left, right

            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            neighbor = nr * cols + nc

            if neighbor in blocked_ids:
                continue

            # Edge cost = 1 (unit weights)
            tentative_g = g_score[current] + 1

            # Found a better path to neighbor
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f = tentative_g + manhattan(neighbor, goal_id, cols)
                f_score[neighbor] = f

                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)

    # Path not found
    return None


# =========================================================
# Usage example
# =========================================================

rows, cols = 10000, 50000

# Create a set of blocked cells
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

# Ensure start and goal are not blocked
blocked_cells.discard(start)
blocked_cells.discard(goal)

path = a_star_grid(rows, cols, start, goal, blocked_cells)

if path is not None:
    print("Path found! Path length:", len(path))
    print("First 5 steps:", path[:5])
    print("Last 5 steps:", path[-5:])
else:
    print("Path not found")

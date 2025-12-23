import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Dict
import heapq


class AStarPlanner:

    def __init__(self, grid_resolution: float = 0.1):
        self.resolution = grid_resolution
        self.grid = None
        self.grid_size = None

    def create_grid(
        self,
        obstacles: List[Tuple[float, float]],
        arena_size: float = 10.0,
        safety_margin: float = 0.4,
    ):
        grid_size = int(arena_size / self.resolution)
        self.grid_size = (grid_size, grid_size)
        self.grid = np.zeros(self.grid_size, dtype=bool)
        margin_cells = int(safety_margin / self.resolution)

        for obs_x, obs_y in obstacles:
            center_gx = int(obs_x / self.resolution)
            center_gy = int(obs_y / self.resolution)

            for dx in range(-margin_cells, margin_cells + 1):
                for dy in range(-margin_cells, margin_cells + 1):
                    gx = center_gx + dx
                    gy = center_gy + dy
                    if 0 <= gx < grid_size and 0 <= gy < grid_size:
                        dist_cells = math.sqrt(dx * dx + dy * dy)
                        if dist_cells <= margin_cells:
                            self.grid[gx, gy] = True

        inflated = self.grid.copy()
        for x in range(1, grid_size - 1):
            for y in range(1, grid_size - 1):
                if self.grid[x, y]:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            inflated[x + dx, y + dy] = True
        self.grid = inflated
        # plt.imshow(self.grid.T, origin="lower")
        # plt.show()

    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        if self.grid is None:
            return [start, goal]

        start_grid = (int(start[0] / self.resolution), int(start[1] / self.resolution))
        goal_grid = (int(goal[0] / self.resolution), int(goal[1] / self.resolution))
        if not self._is_valid(goal_grid) or self.grid[goal_grid[0], goal_grid[1]]:
            print(f"⚠️ Goal position invalid or in obstacle: {goal}")
            return [start, goal]

        if not self._is_valid(start_grid) or self.grid[start_grid[0], start_grid[1]]:
            print(f"⚠️ Start in obstacle! Finding nearest valid node...")
            found_new_start = False
            for r in range(1, 10):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        new_start = (start_grid[0] + dx, start_grid[1] + dy)
                        if (
                            self._is_valid(new_start)
                            and not self.grid[new_start[0], new_start[1]]
                        ):
                            start_grid = new_start
                            print(f"✅ Found new start at {start_grid}")
                            found_new_start = True
                            break
                    if found_new_start:
                        break
                if found_new_start:
                    break
            if not found_new_start:
                print(f"❌ Could not find a valid start node near {start_grid}")
                return [start, goal]

        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {
            start_grid: np.linalg.norm(np.array(start_grid) - np.array(goal_grid))
        }

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_grid:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + np.linalg.norm(
                    np.array(current) - np.array(neighbor)
                )
                if neighbor not in g_score or tentative_g < g_score.get(
                    neighbor, np.inf
                ):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + np.linalg.norm(
                        np.array(neighbor) - np.array(goal_grid)
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("⚠️ A* failed to find path!")
        return [start, goal]

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if self._is_valid((nx, ny)) and not self.grid[nx, ny]:
                neighbors.append((nx, ny))
        return neighbors

    def _reconstruct_path(
        self, came_from: Dict, current: Tuple[int, int]
    ) -> List[np.ndarray]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        world_path = [
            np.array([p[0] * self.resolution, p[1] * self.resolution]) for p in path
        ]

        return world_path
        # return self._smooth_path(world_path)

    def _smooth_path(
        self, path: List[np.ndarray], max_iterations: int = 100
    ) -> List[np.ndarray]:
        if len(path) <= 2:
            return path
        smoothed = np.array([p for p in path])
        for _ in range(max_iterations):
            for i in range(1, len(smoothed) - 1):
                smoothed[i] = 0.8 * smoothed[i] + 0.1 * (
                    smoothed[i - 1] + smoothed[i + 1]
                )
        return list(map(np.array, smoothed))

import numpy as np
from typing import Tuple, List, Dict
from a_star import AStarPlanner
from config import ControllerConfig
from fuzzy_controller import FuzzyDWAController


class DWAController:
    def __init__(
        self,
        config: ControllerConfig,
        use_fuzzy: bool = False,
        fixed_weights: Dict = None,
    ):
        self.config = config
        self.use_fuzzy = use_fuzzy
        self.planner = AStarPlanner(grid_resolution=0.1)
        self.path = []
        self.optimal_path_length = 0.0
        self.current_waypoint_idx = 0
        self.fuzzy_controller = FuzzyDWAController()

        self.predict_time = 1.0  # <-- BEST: 1.0
        self.v_samples = 20
        self.w_samples = 40

        # --- BEST WEIGHTS ---
        self.heading_weight = 0.2
        self.velocity_weight = 0.35
        self.clearance_weight = 0.45

        if not use_fuzzy and fixed_weights:
            print(f"Loading fixed DWA weights: {fixed_weights}")
            self.heading_weight = fixed_weights.get("heading", self.heading_weight)
            self.velocity_weight = fixed_weights.get("velocity", self.velocity_weight)
            self.clearance_weight = fixed_weights.get(
                "clearance", self.clearance_weight
            )

        self.last_heading_w = 0.0
        self.last_velocity_w = 0.0
        self.last_clearance_w = 0.0

        self.start_pos = None
        self.final_goal = None

        self.last_predicted_endpoints = None
        self.best_predicted_endpoint = None

    def reset(self):
        self.path = []
        self.optimal_path_length = 0.0
        self.current_waypoint_idx = 0
        self.last_heading_w = 0.0
        self.last_velocity_w = 0.0
        self.last_clearance_w = 0.0
        self.start_pos = None
        self.final_goal = None
        self.last_predicted_endpoints = None
        self.best_predicted_endpoint = None

    def set_goal(
        self, start: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[float, float]]
    ):
        safety_margin = 0.4
        self.planner.create_grid(
            obstacles,
            arena_size=self.config.arena_size,
            safety_margin=safety_margin,
        )
        self.path = self.planner.plan(start, goal)
        self.optimal_path_length = self._calculate_path_length(self.path)
        self.start_pos = np.array(start)
        self.final_goal = np.array(goal)
        self.current_waypoint_idx = 0

    def _calculate_path_length(self, path: List[np.ndarray]) -> float:
        """Helper to calculate the total length of a given path."""
        if len(path) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            total_distance += np.linalg.norm(p2 - p1)
        return total_distance

    def get_action(self, lidar_readings: np.ndarray, robot_state: Dict):
        pos = np.array(robot_state["position"])
        angle = robot_state["angle"]
        v_now = np.linalg.norm(robot_state["linear_velocity"])
        w_now = robot_state["angular_velocity"]
        dt = self.config.dt

        # --- Dynamic Window ---
        v_max = self.config.max_linear_speed
        w_max = self.config.max_angular_speed
        a_max = self.config.max_linear_accel
        alpha_max = self.config.max_angular_accel

        v_min = max(0.0, v_now - a_max * dt)
        v_max_dyn = min(v_max, v_now + a_max * dt)
        w_min_dyn = max(-w_max, w_now - alpha_max * dt)
        w_max_dyn = min(w_max, w_now + alpha_max * dt)

        v_grid = np.linspace(v_min, v_max_dyn, self.v_samples)
        w_grid = np.linspace(w_min_dyn, w_max_dyn, self.w_samples)
        V, W = np.meshgrid(v_grid, w_grid, indexing="ij")

        # --- Predict trajectories ---
        final_angle, final_pos = self._predict_trajectories(pos, angle, V, W)
        self.last_predicted_endpoints = final_pos

        # --- Convert LiDAR to obstacle positions ---
        obstacles = self._lidar_to_obstacles(lidar_readings, robot_state)

        # --- Find the Waypoint ---
        current_goal = self._find_target_waypoint(pos, v_now, obstacles)
        if current_goal is None:
            current_goal = self.final_goal

        if self.use_fuzzy:
            # --- Calculate Fuzzy Logic Inputs ---
            (
                min_lidar_distance,
                obs_density,
                current_vel,
                goal_proximity,
                goal_angle_error,
                obstacle_angle_error,
            ) = self._compute_fuzzy_inputs(robot_state, current_goal, lidar_readings)

            # --- Compute Weights using Fuzzy Logic ---
            heading_w, velocity_w, clearance_w = self.fuzzy_controller.compute_weights(
                min_lidar_distance,
                obs_density,
                current_vel,
                goal_proximity,
                goal_angle_error,
                obstacle_angle_error,
            )
        else:
            heading_w = self.heading_weight
            velocity_w = self.velocity_weight
            clearance_w = self.clearance_weight

        self.last_heading_w = heading_w
        self.last_velocity_w = velocity_w
        self.last_clearance_w = clearance_w

        # --- Compute scores ---
        score_heading = self._score_heading(final_pos, final_angle, current_goal)
        score_velocity = self._score_velocity(V, W, v_now, w_now)
        score_clearance = self._score_clearance(final_pos, obstacles)

        # --- Combine scores ---
        total_score = (
            heading_w * score_heading
            + velocity_w * score_velocity
            + clearance_w * score_clearance
        )

        # --- Pick best (highest) total score ---
        best_idx = np.argmax(total_score)
        best_v = V.flat[best_idx]
        best_w = W.flat[best_idx]
        best_v_idx, best_w_idx = np.unravel_index(best_idx, V.shape)
        self.best_predicted_endpoint = final_pos[best_v_idx, best_w_idx]

        # --- Normalize output ---
        action = np.array([best_v / v_max, 0.0, best_w / w_max])
        return np.clip(action, -1.0, 1.0)

    def _predict_trajectories(
        self, pos: np.ndarray, angle: float, V: np.ndarray, W: np.ndarray
    ) -> np.ndarray:
        final_angle = angle + W * self.predict_time
        t = self.predict_time
        W_e = W + 1e-6

        final_pos_x = (
            pos[0] - (V / W_e) * np.sin(angle) + (V / W_e) * np.sin(angle + W * t)
        )
        final_pos_y = (
            pos[1] + (V / W_e) * np.cos(angle) - (V / W_e) * np.cos(angle + W * t)
        )

        straight = np.abs(W) < 1e-4
        final_pos_x[straight] = pos[0] + V[straight] * np.cos(angle) * t
        final_pos_y[straight] = pos[1] + V[straight] * np.sin(angle) * t
        final_pos = np.stack((final_pos_x, final_pos_y), axis=-1)
        return final_angle, final_pos

    def _score_heading(
        self, final_pos: np.ndarray, final_angle: np.ndarray, goal: np.ndarray
    ):
        goal_vec = goal - final_pos
        goal_angle = np.arctan2(goal_vec[..., 1], goal_vec[..., 0])
        heading_error = np.abs(
            self._normalize_angle(goal_angle - final_angle)
        )  # (-pi, pi)
        heading_score = 1.0 - (heading_error / np.pi) ** 2
        return np.clip(heading_score, 0.0, 1.0)

    def _score_velocity(
        self, V: np.ndarray, W: np.ndarray, v_now: float, w_now: float
    ) -> np.ndarray:
        velocity_score = np.sqrt(V / self.config.max_linear_speed)
        v_change = np.abs(V - v_now) / self.config.max_linear_speed
        w_change = np.abs(W - w_now) / self.config.max_angular_speed
        smoothness = 1.0 - 0.3 * (v_change + w_change)
        return np.clip(velocity_score * smoothness, 0.0, 1.0)

    def _score_clearance(
        self, final_pos: np.ndarray, obstacles: List[np.ndarray]
    ) -> np.ndarray:
        if not obstacles:
            return np.ones_like(final_pos[..., 0])

        obs = np.stack(obstacles)
        diff = final_pos[..., None, :] - obs[None, None, :, :]
        dist_to_obs = np.linalg.norm(diff, axis=-1)
        min_dist = np.min(dist_to_obs, axis=-1)  # Shape: (v_samples, w_samples)

        collision_dist = self.config.robot_size / 2.0
        buffer_dist = min_dist - collision_dist
        steepness = 3.0
        clearance_score = np.tanh(buffer_dist * steepness)
        clearance_score[buffer_dist < 0] = 0.0
        return np.clip(clearance_score, 0.0, 1.0)

    def _lidar_to_obstacles(self, lidar_readings: np.ndarray, robot_state: Dict):
        obstacles = []
        angle_increment = 2 * np.pi / len(lidar_readings)
        robot_angle = robot_state["angle"]
        robot_pos = robot_state["position"]

        for i, distance in enumerate(lidar_readings):
            if distance >= self.config.lidar_range * 0.95:
                continue
            ray_angle = robot_angle + i * angle_increment
            obs_pos = robot_pos + np.array(
                [distance * np.cos(ray_angle), distance * np.sin(ray_angle)]
            )
            obstacles.append(obs_pos)
        return obstacles

    def _calculate_obstacle_density(self, lidar_readings: np.ndarray) -> float:
        close_distance = self.config.safety_distance * 1.5
        close_readings = lidar_readings < close_distance
        return np.mean(close_readings)

    def _find_target_waypoint(
        self,
        current_pos: np.ndarray,
        current_speed: float,
        obstacles: List[np.ndarray] = None,
    ) -> np.ndarray:
        if not self.path:
            return None
        path_array = np.array(self.path)

        distances_to_path = np.linalg.norm(path_array - current_pos, axis=-1)
        closest_point_idx = np.argmin(distances_to_path)

        if closest_point_idx == len(self.path) - 1 and distances_to_path[-1] < 0.5:
            return self.path[-1]

        min_lookahead = 0.5  # meters
        max_lookahead = 2.0  # meters
        speed_factor = 1.0  # seconds
        lookahead_distance = np.clip(
            min_lookahead + current_speed * speed_factor, min_lookahead, max_lookahead
        )

        target_idx = closest_point_idx
        dist_along_path = 0.0
        while dist_along_path < lookahead_distance and target_idx < len(self.path) - 1:
            dist_along_path += np.linalg.norm(
                self.path[target_idx + 1] - self.path[target_idx]
            )
            target_idx += 1

        target_waypoint = self.path[target_idx]

        if self._is_waypoint_blocked(target_waypoint, obstacles):
            target_waypoint = self._adjust_blocked_waypoint(
                target_waypoint, obstacles, current_pos
            )

        return target_waypoint

    def _is_waypoint_blocked(
        self, waypoint: np.ndarray, obstacles: List[np.ndarray]
    ) -> bool:
        for obs in obstacles:
            dist = np.linalg.norm(obs - waypoint)
            if dist < self.config.safety_distance:
                return True
        return False

    def _adjust_blocked_waypoint(
        self,
        current_goal: np.ndarray,
        obstacles: List[np.ndarray],
        robot_pos: np.ndarray,
    ) -> np.ndarray:
        if not obstacles:
            return current_goal

        obstacles_array = np.array(obstacles)
        distances_to_obstacles = np.linalg.norm(obstacles_array - current_goal, axis=1)
        min_obs_distance = np.min(distances_to_obstacles)

        if min_obs_distance < self.config.safety_distance * 0.35:
            alternatives = []
            for angle in np.linspace(0, 2 * np.pi, 8):
                alternative = (
                    current_goal
                    + 1.0
                    * self.config.safety_distance
                    * np.array([np.cos(angle), np.sin(angle)])
                )

                alt_distances = np.linalg.norm(obstacles_array - alternative, axis=-1)
                if np.min(alt_distances) >= self.config.safety_distance * 0.8:
                    alternatives.append(alternative)

            if alternatives:
                alternatives_array = np.array(alternatives)
                goal_directions = alternatives_array - robot_pos
                goal_directions_norm = goal_directions / np.linalg.norm(
                    goal_directions, axis=-1, keepdims=True
                )
                original_direction = self.final_goal - robot_pos
                original_direction = original_direction / np.linalg.norm(
                    original_direction
                )

                alignment_scores = np.dot(goal_directions_norm, original_direction)
                best_alternative = alternatives_array[np.argmax(alignment_scores)]
                return best_alternative

        return current_goal

    def _compute_fuzzy_inputs(
        self, robot_state: Dict, current_goal: np.ndarray, lidar_readings: np.ndarray
    ) -> Tuple[float, float]:
        pos = robot_state["position"]
        angle = robot_state["angle"]

        # 1. Minimum distance to obstacle (0 = close, 1 = far)
        collide_dist = self.config.robot_size / 2.0
        min_lidar_distance = (np.min(lidar_readings) - collide_dist) / (
            self.config.safety_distance - collide_dist
        )
        min_lidar_distance = np.clip(min_lidar_distance, 0.0, 1.0)
        # 2. Obstacle density (0 = sparse, 1 = dense)
        obs_density = self._calculate_obstacle_density(lidar_readings)
        # 3. Current velocity (0 = stop, 1 = top speed)
        current_vel = (
            np.linalg.norm(robot_state["linear_velocity"])
            / self.config.max_linear_speed
        )
        current_vel = np.clip(current_vel, 0, 1)
        # 4. Goal proximity (0 = far, 1 = close)
        current_goal_dist = np.linalg.norm(self.final_goal - pos)
        total_path_dist = np.linalg.norm(self.final_goal - self.start_pos)
        goal_proximity = 1.0 - np.clip(
            current_goal_dist / (total_path_dist + 1e-6), 0.0, 1.0
        )
        # 5. Goal angle error (0 = aligned, 1 = 180deg misaligned)
        goal_vec = current_goal - pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        goal_error = np.abs(self._normalize_angle(goal_angle - angle))
        goal_angle_error = goal_error / np.pi  # Normalize to [0,1]
        # 6. Obstacle angle error (0 = behind, 1 = front)
        if (
            len(lidar_readings) > 0
            and np.min(lidar_readings) < self.config.lidar_range * 0.99
        ):
            min_dist_idx = np.argmin(lidar_readings)
            obstacle_angle = min_dist_idx * (2 * np.pi / len(lidar_readings))
            obstacle_error = np.abs(self._normalize_angle(obstacle_angle))
            obstacle_angle_error = (np.pi - obstacle_error) / np.pi
        else:
            obstacle_angle_error = 0.0  # No obstacles

        return (
            min_lidar_distance,
            obs_density,
            current_vel,
            goal_proximity,
            goal_angle_error,
            obstacle_angle_error,
        )

    @staticmethod
    def _normalize_score(s: np.ndarray) -> np.ndarray:
        rng = np.ptp(s)
        return (s - np.min(s)) / (rng + 1e-6)

    @staticmethod
    def _normalize_angle(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2 * np.pi) - np.pi

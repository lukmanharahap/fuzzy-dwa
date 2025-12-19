import os
import sys
import numpy as np
from numpy import float64
from numpy._typing import NDArray
from Box2D import b2World
from typing import Dict, List, Tuple
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import config as cfg
from renderer import Renderer
from robot import Robot


class DynamicEnvironment:

    def __init__(
        self,
        render_mode: bool = False,
        use_fuzzy: bool = False,
        start_pos: np.ndarray = None,
        goal_pos: np.ndarray = None,
        static_obs_configs: List[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = None,
        n_obstacles: int = cfg.N_OBSTACLES,
        obstacle_speed_range: Tuple[float, float] = cfg.OBSTACLE_SPEED_RANGE,
        fixed_weights: Dict = None,
        save_video: bool = False,
    ):
        self.render_mode = render_mode
        self.world = b2World(gravity=(0, 0), doSleep=True)

        # World objects
        self.obstacle_bodies = []
        self.static_obs_bodies = []
        self.wall_bodies = []

        self.start_position = (
            start_pos if start_pos is not None else np.array([1.0, 1.0])
        )
        self.goal_position = goal_pos if goal_pos is not None else np.array([9.0, 9.0])
        self.static_obs_configs = (
            static_obs_configs if static_obs_configs is not None else []
        )
        self.n_obstacles = n_obstacles
        self.obstacle_speed_range = obstacle_speed_range

        self._build_static_world(self.static_obs_configs)
        static_map = self.get_static_obstacle_points()

        self.robot = Robot(
            self.world,
            self.start_position,
            static_map,
            use_fuzzy=use_fuzzy,
            fixed_weights=fixed_weights,
        )

        self.renderer = Renderer(render_mode, save_video)
        self._create_obstacles(self.n_obstacles)

        # State and tracking
        self.step_count = 0
        self.collision_count = 0
        self.total_distance_traveled = 0.0
        self.previous_robot_pos = self.start_position

    def _build_static_world(
        self, static_obs_configs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ):
        self._create_walls()
        self._create_static_obstacles(static_obs_configs)

    def _create_walls(self):
        wall_thickness = 0.1
        size = cfg.ARENA_SIZE_M / 2
        wall_configs = [
            ((size, -wall_thickness / 2), (size, wall_thickness)),
            ((size, cfg.ARENA_SIZE_M + wall_thickness / 2), (size, wall_thickness)),
            ((-wall_thickness / 2, size), (wall_thickness, size)),
            ((cfg.ARENA_SIZE_M + wall_thickness / 2, size), (wall_thickness, size)),
        ]
        for pos, dims in wall_configs:
            wall = self.world.CreateStaticBody(position=pos)
            wall.CreatePolygonFixture(box=dims, friction=0.8)
            wall.userData = {"type": "wall"}
            self.wall_bodies.append(wall)

    def _create_static_obstacles(
        self, static_obs_configs: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ):
        for pos, dims in static_obs_configs:
            static_obs = self.world.CreateStaticBody(position=pos)
            static_obs.CreatePolygonFixture(box=dims, friction=0.8, density=0)
            static_obs.userData = {"type": "static_obs"}
            self.static_obs_bodies.append(static_obs)

    def _create_obstacles(self, n_obstacles: int):
        self.obstacle_bodies = []
        safe_zone_center = self.start_position
        safe_zone_radius = 2.0
        for _ in range(n_obstacles):
            for _ in range(100):
                pos = np.random.uniform(0.5, cfg.ARENA_SIZE_M - 0.5, 2)
                if np.linalg.norm(pos - safe_zone_center) > safe_zone_radius:
                    if all(
                        np.linalg.norm(pos - np.array(obs.position)) >= 1.0
                        for obs in self.obstacle_bodies
                    ):
                        break
            obstacle = self.world.CreateDynamicBody(
                position=tuple(pos), linearDamping=0.8, angularDamping=0.9
            )
            obstacle.CreateCircleFixture(
                radius=cfg.OBSTACLE_RADIUS_M,
                density=cfg.OBSTACLE_MASS / (math.pi * cfg.OBSTACLE_RADIUS_M**2),
                friction=0.3,
            )
            obstacle.userData = {
                "type": "obstacle_dynamic",
                "target": self._get_new_obstacle_target(),
            }
            self.obstacle_bodies.append(obstacle)

    # def _get_new_obstacle_target(self) -> NDArray[float64]:
    #     margin = 0.5
    #     return np.random.uniform(margin, cfg.ARENA_SIZE_M - margin, 2)

    def _get_new_obstacle_target(self) -> NDArray[float64]:
        margin = 0.5
        # GANTI np.random -> self.rng
        # Pastikan self.rng sudah ada (fallback ke np.random jika belum di-init)
        rng = getattr(self, "rng", np.random)
        return rng.uniform(margin, cfg.ARENA_SIZE_M - margin, 2)

    # def reset_environment(self):
    #     self.robot.reset()
    #     safe_zone_center = self.start_position
    #     safe_zone_radius = 2.0
    #     for obstacle in self.obstacle_bodies:
    #         for _ in range(100):
    #             pos = np.random.uniform(0.5, cfg.ARENA_SIZE_M - 0.5, 2)
    #             if np.linalg.norm(pos - safe_zone_center) > safe_zone_radius:
    #                 obstacle.position = tuple(pos)
    #                 break
    #         obstacle.linearVelocity = (0, 0)
    #         obstacle.angularVelocity = 0
    #         obstacle.userData["target"] = self._get_new_obstacle_target()

    #     self.step_count = 0
    #     self.collision_count = 0
    #     self.total_distance_traveled = 0.0
    #     self.previous_robot_pos = self.start_position
    #     print("\n🔄 Environment reset")

    def reset_environment(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.robot.reset()
        safe_zone_center = self.start_position
        safe_zone_radius = 2.0

        for obstacle in self.obstacle_bodies:
            for _ in range(100):
                pos = self.rng.uniform(0.5, cfg.ARENA_SIZE_M - 0.5, 2)
                if np.linalg.norm(pos - safe_zone_center) > safe_zone_radius:
                    obstacle.position = tuple(pos)
                    break
            obstacle.linearVelocity = (0, 0)
            obstacle.angularVelocity = 0
            obstacle.userData["target"] = self._get_new_obstacle_target()

        self.step_count = 0
        self.collision_count = 0
        self.total_distance_traveled = 0.0
        self.previous_robot_pos = self.start_position
        print(f"\n🔄 Environment reset with Seed: {seed}")

    def step_simulation(self, action: np.ndarray):
        self.robot.apply_action(action)
        self._update_obstacle_behaviors()
        self.world.Step(
            cfg.PHYSICS_TIME_STEP, cfg.PHYSICS_VEL_ITERS, cfg.PHYSICS_POS_ITERS
        )
        self.world.ClearForces()

        self.step_count += 1
        self._update_tracking()
        if self._check_robot_collision():
            self.collision_count += 1

    def _update_obstacle_behaviors(self):
        rng = getattr(self, "rng", np.random)
        for obstacle in self.obstacle_bodies:
            pos = np.array(obstacle.position)
            target = obstacle.userData["target"]
            speed = rng.uniform(*self.obstacle_speed_range)
            if np.linalg.norm(pos - target) < 0.5:
                obstacle.userData["target"] = self._get_new_obstacle_target()
                target = obstacle.userData["target"]

            direction = target - pos
            direction /= np.linalg.norm(direction) + 1e-6

            speed = np.random.uniform(*self.obstacle_speed_range)
            target_vel = direction * speed
            current_vel = np.array(obstacle.linearVelocity)

            force_dir = (target_vel - current_vel) * cfg.OBSTACLE_STEER_FORCE
            obstacle.ApplyForceToCenter(tuple(force_dir), True)

            margin = 1.0
            if not (
                margin < pos[0] < cfg.ARENA_SIZE_M - margin
                and margin < pos[1] < cfg.ARENA_SIZE_M - margin
            ):
                center = np.array([cfg.ARENA_SIZE_M / 2, cfg.ARENA_SIZE_M / 2])
                direction_to_center = center - pos
                direction_to_center /= np.linalg.norm(direction_to_center) + 1e-6
                obstacle.ApplyForceToCenter(tuple(direction_to_center * 30), True)

            velocity = np.array(obstacle.linearVelocity)
            speed_norm = np.linalg.norm(velocity)
            max_speed = self.obstacle_speed_range[1]
            if speed_norm > max_speed:
                obstacle.linearVelocity = tuple(velocity / speed_norm * max_speed)

    def _update_tracking(self):
        current_pos = np.array(self.robot.body.position)
        distance_step = np.linalg.norm(current_pos - self.previous_robot_pos)
        self.total_distance_traveled += distance_step
        self.previous_robot_pos = current_pos

    def _check_robot_collision(self) -> bool:
        if not hasattr(self.robot.body, "contacts"):
            return False
        touching = False
        for contact_edge in self.robot.body.contacts:
            contact = contact_edge.contact
            if not contact.touching:
                continue
            body_a, body_b = contact.fixtureA.body, contact.fixtureB.body
            if hasattr(body_a, "userData") and hasattr(body_b, "userData"):
                types = {body_a.userData.get("type"), body_b.userData.get("type")}
                if "robot" in types and (
                    "obstacle_dynamic" in types
                    or "wall" in types
                    or "static_obs" in types
                ):
                    touching = True
                    break
        if touching and not getattr(self, "_was_colliding", False):
            self._was_colliding = True
            return True
        elif not touching:
            self._was_colliding = False
        return False

    def get_static_obstacle_points(self, spacing=0.1) -> List[Tuple[float, float]]:
        points = []
        wall_margin = 0.2
        arena_size = cfg.ARENA_SIZE_M
        for x in np.arange(0, arena_size, spacing):
            points.append((x, wall_margin))
            points.append((x, arena_size - wall_margin))
        for y in np.arange(0, arena_size, spacing):
            points.append((wall_margin, y))
            points.append((arena_size - wall_margin, y))

        for static_obs in self.static_obs_bodies:
            for fixture in static_obs.fixtures:
                shape = fixture.shape
                world_vertices = [static_obs.transform * v for v in shape.vertices]
                xs = [v[0] for v in world_vertices]
                ys = [v[1] for v in world_vertices]
                min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
                x_range = np.arange(min_x, max_x + spacing, spacing)
                y_range = np.arange(min_y, max_y + spacing, spacing)
                for x in x_range:
                    for y in y_range:
                        points.append((x, y))
        return points

    def is_task_complete(self) -> bool:
        robot_pos = np.array(self.robot.body.position)
        return np.linalg.norm(robot_pos - self.goal_position) < 0.5

    def get_info_dict(self, lidar_scan: np.ndarray) -> Dict:
        """Get current environment info for display."""
        robot_state = self.robot.get_true_state()
        odom_robot_state = self.robot.get_odometry_state()
        true_pos = robot_state["position"]
        odom_pos = odom_robot_state["position"]
        fuzzy_weights = self.robot.last_fuzzy_weights
        obs_density = self.robot.controller._calculate_obstacle_density(lidar_scan)
        return {
            "step_count": self.step_count,
            "collision_count": self.collision_count,
            "safety_margin": np.min(lidar_scan),
            "obs_density": obs_density,
            "position": robot_state["position"],
            "angle": robot_state["angle"],
            "linear_velocity": np.linalg.norm(robot_state["linear_velocity"]),
            "angular_velocity": robot_state["angular_velocity"],
            "distance_traveled": self.total_distance_traveled,
            "true_robot_state": robot_state,
            "odom_robot_state": odom_robot_state,
            "odometry_drift": np.linalg.norm(true_pos - odom_pos),
            "heading_w": fuzzy_weights["heading_w"],
            "velocity_w": fuzzy_weights["velocity_w"],
            "clearance_w": fuzzy_weights["clearance_w"],
            "predicted_endpoints": self.robot.predicted_endpoints,
            "best_predicted_endpoint": self.robot.best_predicted_endpoint,
            "optimal_path_length": self.robot.optimal_path_length,
            "euclidean_path_length": self.euclidean_path_length,
        }

    def render_frame(self, lidar_scan: np.ndarray, history: List[np.ndarray] = None):
        if self.renderer is None:
            return

        def lidar_to_world_points(scan: np.ndarray, robot_state: Dict):
            points = []
            angle_increment = 2 * np.pi / len(scan)
            for i, distance in enumerate(scan):
                if distance >= cfg.LIDAR_RANGE_M:
                    continue
                ray_angle = robot_state["angle"] + i * angle_increment
                points.append(
                    robot_state["position"]
                    + np.array(
                        [distance * np.cos(ray_angle), distance * np.sin(ray_angle)]
                    )
                )
            return points

        true_robot_state = self.robot.get_true_state()
        odom_robot_state = self.robot.get_odometry_state()
        true_pos = true_robot_state["position"]
        odom_pos = odom_robot_state["position"]
        env_state = {
            "true_robot_state": true_robot_state,
            "odom_robot_state": odom_robot_state,
            "odometry_drift": np.linalg.norm(true_pos - odom_pos),
            "obstacle_bodies": self.obstacle_bodies,
            "static_obstacles": self.static_obs_bodies,
            "start_position": self.start_position,
            "goal_position": self.goal_position,
            "info": self.get_info_dict(lidar_scan),
            "lidar_points": lidar_to_world_points(lidar_scan, true_robot_state),
            "lidar_scan": lidar_scan,
        }
        self.renderer.render(env_state, path=self.robot.path, history=history)

    @property
    def euclidean_path_length(self) -> float:
        """Returns the straight-line (Euclidean) distance from start to goal."""
        return np.linalg.norm(self.start_position - self.goal_position)

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        if self.step_count > 0:
            print("🛑 Environment closed")

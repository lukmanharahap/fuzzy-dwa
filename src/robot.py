import numpy as np
from typing import List, Tuple, Dict
from Box2D import b2World, b2RayCastCallback, b2Body
import config as cfg
from config import ControllerConfig
from localization import SimpleCorrector
from dwa import DWAController


class Robot:

    def __init__(
        self,
        world: b2World,
        start_pos: np.ndarray,
        static_map: List[Tuple[float, float]],
        use_fuzzy: bool = False,
        fixed_weights: Dict = None,
    ):
        self.world = world
        self.start_pos = start_pos

        # 1. Physical Body
        self.body = self._create_body()

        # 2. Brain - Lozalization, Controller, and Planner
        self.config = ControllerConfig()
        self.localization = SimpleCorrector(static_map)
        self.controller = DWAController(
            config=self.config, use_fuzzy=use_fuzzy, fixed_weights=fixed_weights
        )
        self.current_goal = None

        # 3. "Belief" - Noisy Odometry State
        self.odometry_state = self.get_true_state()

        self.velocity_history = []

    def _create_body(self) -> b2Body:
        radius = cfg.ROBOT_SIZE_M / 2
        robot_body = self.world.CreateKinematicBody()
        robot_body = self.world.CreateDynamicBody(
            position=tuple(self.start_pos),
            angle=0.0,
            linearDamping=cfg.ROBOT_LINEAR_DAMPING,
            angularDamping=cfg.ROBOT_ANGULAR_DAMPING,
        )
        robot_body.CreateCircleFixture(
            radius=radius,
            friction=cfg.ROBOT_FRICTION,
            density=cfg.ROBOT_MASS / (np.pi * radius**2),
        )
        robot_body.userData = {"type": "robot"}
        return robot_body

    def reset(self):
        self.body.position = tuple(self.start_pos)
        self.body.angle = 0.0
        self.body.linearVelocity = (0, 0)
        self.body.angularVelocity = 0
        self.odometry_state = self.get_true_state()
        self.controller.reset()
        self.current_goal = None
        self.velocity_history = []

    def get_true_state(self) -> Dict:
        return {
            "position": np.array(self.body.position),
            "angle": self.body.angle,
            "linear_velocity": np.array(self.body.linearVelocity),
            "angular_velocity": self.body.angularVelocity,
        }

    def get_odometry_state(self) -> Dict:
        return self.odometry_state

    def update_odometry(self, dt: float):
        true_state = self.get_true_state()
        true_vel_world = self.body.linearVelocity
        true_vel_local_x = self.body.GetLocalVector(true_vel_world).x
        true_ang_vel = true_state["angular_velocity"]

        vx_noise = np.random.normal(0, cfg.ODOMETRY_VEL_NOISE_STD)
        w_noise = np.random.normal(0, cfg.ODOMETRY_ANG_VEL_NOISE_STD)

        noisy_vx = true_vel_local_x + vx_noise
        noisy_w = true_ang_vel + w_noise
        noisy_w += np.random.normal(0, cfg.ODOMETRY_ANGLE_NOISE_STD)

        last_odom_state = self.odometry_state
        last_pos = last_odom_state["position"]
        last_angle = last_odom_state["angle"]

        new_angle = last_angle + noisy_w * dt
        cos_odom = np.cos(new_angle)
        sin_odom = np.sin(new_angle)

        noisy_v_world_x = noisy_vx * cos_odom
        noisy_v_world_y = noisy_vx * sin_odom
        noisy_v_world = np.array([noisy_v_world_x, noisy_v_world_y])
        self.velocity_history.append(np.linalg.norm(noisy_v_world))

        new_pos = last_pos + noisy_v_world * dt
        new_pos += np.random.normal(0, cfg.ODOMETRY_POS_NOISE_STD, 2)

        self.odometry_state = {
            "position": new_pos,
            "angle": new_angle,
            "linear_velocity": noisy_v_world,
            "angular_velocity": noisy_w,
        }

        true_lidar_scan = self.perform_lidar_scan()
        self.run_localization_correction(true_lidar_scan)

    def run_localization_correction(self, true_lidar_scan: np.ndarray):
        lidar_config = {
            "range": cfg.LIDAR_RANGE_M,
            "rays": cfg.LIDAR_RAYS,
        }
        corrected_state = self.localization.correct_odometry(
            self.odometry_state, true_lidar_scan, lidar_config
        )
        self.odometry_state = corrected_state

    def perform_lidar_scan(self) -> np.ndarray:
        readings = np.full(cfg.LIDAR_RAYS, cfg.LIDAR_RANGE_M)
        robot_pos = self.body.position
        robot_angle = self.body.angle

        class RayCastCallback(b2RayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if fixture.body.userData.get("type") == "robot":
                    return -1.0
                self.fraction = fraction
                return fraction

        for i in range(cfg.LIDAR_RAYS):
            angle = robot_angle + i * (2 * np.pi / cfg.LIDAR_RAYS)
            p1 = robot_pos
            p2 = p1 + (
                cfg.LIDAR_RANGE_M * np.cos(angle),
                cfg.LIDAR_RANGE_M * np.sin(angle),
            )
            callback = RayCastCallback()
            callback.fraction = 1.0
            self.world.RayCast(callback, p1, p2)

            if callback.fraction < 1.0:
                distance = callback.fraction * cfg.LIDAR_RANGE_M
                noise = np.random.normal(0, cfg.LIDAR_NOISE_STD * distance)
                readings[i] = np.clip(distance + noise, 0.0, cfg.LIDAR_RANGE_M)

        return readings

    def update_goal(
        self, goal_pos: np.ndarray, all_obstacles: List[Tuple[float, float]]
    ):
        self.current_goal = goal_pos
        robot_pos_belief = self.get_odometry_state()["position"]
        self.controller.set_goal(robot_pos_belief, goal_pos, all_obstacles)

    def compute_action(self, robot_state: Dict, lidar_scan: np.ndarray) -> np.ndarray:
        action = self.controller.get_action(lidar_scan, robot_state)
        return action

    def apply_action(self, action: np.ndarray):
        target_vx_local = action[0] * cfg.MAX_LINEAR_SPEED
        target_omega = action[2] * cfg.MAX_ANGULAR_SPEED

        angle = self.body.angle
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        target_vx_world = target_vx_local * cos_a
        target_vy_world = target_vx_local * sin_a

        self.body.linearVelocity = (target_vx_world, target_vy_world)
        self.body.angularVelocity = target_omega

    def apply_raw_force(self, linear_force: float, torque: float):
        self.body.ApplyForce(
            self.body.GetWorldVector((linear_force, 0)), self.body.worldCenter, True
        )
        self.body.ApplyTorque(torque, True)

    @property
    def path(self) -> List[np.ndarray]:
        return self.controller.path

    @property
    def optimal_path_length(self) -> float:
        """Exposes the controller's A* path length."""
        return self.controller.optimal_path_length

    @property
    def last_fuzzy_weights(self) -> Dict:
        return {
            "heading_w": self.controller.last_heading_w,
            "velocity_w": self.controller.last_velocity_w,
            "clearance_w": self.controller.last_clearance_w,
        }

    @property
    def predicted_endpoints(self) -> np.ndarray:
        if self.controller.last_predicted_endpoints is None:
            return np.array([])
        return self.controller.last_predicted_endpoints

    @property
    def best_predicted_endpoint(self) -> np.ndarray:
        return self.controller.best_predicted_endpoint

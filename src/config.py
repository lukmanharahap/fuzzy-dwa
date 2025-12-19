from pathlib import Path
from dataclasses import dataclass

# --- Arena Dimensions ---
ARENA_SIZE_M = 10.0

# --- Physics Settings ---
PHYSICS_TIME_STEP = 1.0 / 30.0  # 30 Hz simulation
PHYSICS_VEL_ITERS = 8
PHYSICS_POS_ITERS = 3

# --- Robot Constants ---
ROBOT_SIZE_M = 0.58  # 580mm
ROBOT_MASS = 80.0
# ------------------ Dynamic Models ------------------
ROBOT_LINEAR_DAMPING = 1.2
ROBOT_ANGULAR_DAMPING = 3.0
ROBOT_FRICTION = 0.9
ROBOT_MAX_FORCE = 200.0  # N
ROBOT_MAX_TORQUE = 35.0  # Nm
# ----------------------------------------------------
SAFETY_DISTANCE_M = ROBOT_SIZE_M / 2.0 + 2.0

# --- Robot Controller Limits ---
MAX_LINEAR_SPEED = 2.0  # m/s (2.0 m/s physics allowed)
MAX_ANGULAR_SPEED = 3.12  # rad/s (3.12 rad/s physics allowed)
MAX_LINEAR_ACCEL = 2.4  # m/s² (a = F/m -> a = 200/80 = 2.5 m/s²)
MAX_ANGULAR_ACCEL = 9.36  # rad/s² (alpha = T/I -> alpha = 35/3.364 = 10.4 rad/s²)

# --- Obstacle Settings ---
N_OBSTACLES = 8
OBSTACLE_RADIUS_M = 0.3
OBSTACLE_MASS = 65.0
OBSTACLE_SPEED_RANGE = (0.3, 1.0)
OBSTACLE_STEER_FORCE = 10.0

# --- LiDAR Settings ---
LIDAR_RAYS = 32
LIDAR_RANGE_M = 8.0
LIDAR_NOISE_STD = 0.02  # 2% of distance
SENSOR_DELAY_STEPS = 2  # ~66ms delay

# --- Odometry Noise Settings ---
ODOMETRY_POS_NOISE_STD = 0.005  # 5mm std dev per step
ODOMETRY_ANGLE_NOISE_STD = 0.002  # 0.1 deg std dev per step
ODOMETRY_VEL_NOISE_STD = 0.01  # 1cm/s std dev
ODOMETRY_ANG_VEL_NOISE_STD = 0.005

# --- Simulation ---
MAX_EPISODE_STEPS = 3000
STUCK_TIME_LIMIT = 5.0

# --- Rendering ---
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@dataclass
class ControllerConfig:
    arena_size: float = ARENA_SIZE_M
    max_linear_speed: float = MAX_LINEAR_SPEED
    max_angular_speed: float = MAX_ANGULAR_SPEED
    max_linear_accel: float = MAX_LINEAR_ACCEL
    max_angular_accel: float = MAX_ANGULAR_ACCEL
    robot_size: float = ROBOT_SIZE_M
    robot_mass: float = ROBOT_MASS
    safety_distance: float = SAFETY_DISTANCE_M
    dt: float = PHYSICS_TIME_STEP
    lidar_rays: int = LIDAR_RAYS
    lidar_range: float = LIDAR_RANGE_M

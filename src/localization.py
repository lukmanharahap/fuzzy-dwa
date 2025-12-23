import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial import cKDTree


class SimpleCorrector:

    def __init__(self, static_map_points: List[Tuple[float, float]]):
        if not static_map_points:
            raise ValueError("Map points cannot be empty for the corrector.")

        self.map_tree = cKDTree(static_map_points)

    def correct_odometry(
        self, odometry_state: Dict, lidar_scan: np.ndarray, lidar_config: Dict
    ) -> Dict:
        odom_pos = odometry_state["position"]
        odom_angle = odometry_state["angle"]
        angle_increment = 2 * np.pi / lidar_config["rays"]
        measured_points = []
        for i, distance in enumerate(lidar_scan):
            if distance < lidar_config["range"] * 0.99:
                ray_angle = odom_angle + i * angle_increment
                point = odom_pos + np.array(
                    [distance * np.cos(ray_angle), distance * np.sin(ray_angle)]
                )
                measured_points.append(point)

        if len(measured_points) < 10:
            return odometry_state

        measured_points = np.array(measured_points)
        distances, indices = self.map_tree.query(measured_points)
        map_points = self.map_tree.data[indices]
        error_vectors = map_points - measured_points
        avg_error = np.median(error_vectors, axis=0)

        if np.linalg.norm(avg_error) > 0.5:
            return odometry_state

        correction_factor = 0.5
        corrected_pos = odom_pos + avg_error * correction_factor
        corrected_state = odometry_state.copy()
        corrected_state["position"] = corrected_pos

        return corrected_state

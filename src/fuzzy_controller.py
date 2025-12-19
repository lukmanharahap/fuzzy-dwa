import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyDWAController:
    def __init__(self):

        # --- Inputs ---
        obs_dist = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "obs_dist")
        obs_density = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "obs_density")
        robot_vel = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "robot_vel")
        goal_proximity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "goal_proximity")
        goal_angle_error = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "goal_angle_error")
        obs_position = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "obs_position")

        # --- Outputs ---
        heading_w = ctrl.Consequent(np.arange(0, 1.01, 0.01), "heading_w")
        velocity_w = ctrl.Consequent(np.arange(0, 1.01, 0.01), "velocity_w")
        clearance_w = ctrl.Consequent(np.arange(0, 1.01, 0.01), "clearance_w")

        # --- Input Membership Functions ---
        obs_dist["critical"] = fuzz.trapmf(obs_dist.universe, [0, 0, 0.1, 0.2])
        obs_dist["danger"] = fuzz.trimf(obs_dist.universe, [0.15, 0.35, 0.55])
        obs_dist["close"] = fuzz.trimf(obs_dist.universe, [0.45, 0.65, 0.85])
        obs_dist["safe"] = fuzz.trapmf(obs_dist.universe, [0.8, 0.9, 1, 1])

        obs_density["sparse"] = fuzz.trapmf(obs_density.universe, [0, 0, 0.15, 0.3])
        obs_density["moderate"] = fuzz.trimf(obs_density.universe, [0.25, 0.5, 0.75])
        obs_density["dense"] = fuzz.trapmf(obs_density.universe, [0.7, 0.85, 1, 1])

        obs_position["behind"] = fuzz.trapmf(obs_position.universe, [0, 0, 0.2, 0.4])
        obs_position["side"] = fuzz.trimf(obs_position.universe, [0.3, 0.5, 0.7])
        obs_position["front"] = fuzz.trapmf(obs_position.universe, [0.6, 0.8, 1, 1])

        robot_vel["stopped"] = fuzz.trapmf(robot_vel.universe, [0, 0, 0.15, 0.3])
        robot_vel["slow"] = fuzz.trimf(robot_vel.universe, [0.2, 0.45, 0.7])
        robot_vel["fast"] = fuzz.trapmf(robot_vel.universe, [0.6, 0.8, 1, 1])

        goal_proximity["far"] = fuzz.trapmf(goal_proximity.universe, [0, 0, 0.2, 0.4])
        goal_proximity["approaching"] = fuzz.trimf(
            goal_proximity.universe, [0.35, 0.6, 0.85]
        )
        goal_proximity["near"] = fuzz.trapmf(goal_proximity.universe, [0.8, 0.9, 1, 1])

        goal_angle_error["aligned"] = fuzz.trapmf(
            goal_angle_error.universe, [0, 0, 0.15, 0.25]
        )
        goal_angle_error["slight"] = fuzz.trimf(
            goal_angle_error.universe, [0.2, 0.4, 0.6]
        )
        goal_angle_error["large"] = fuzz.trapmf(
            goal_angle_error.universe, [0.5, 0.75, 1, 1]
        )

        # --- Output Membership Functions ---
        for w in [heading_w, velocity_w, clearance_w]:
            w["very_low"] = fuzz.trapmf(w.universe, [0, 0, 0.05, 0.15])
            w["low"] = fuzz.trimf(w.universe, [0.1, 0.25, 0.4])
            w["medium"] = fuzz.trimf(w.universe, [0.3, 0.5, 0.7])
            w["high"] = fuzz.trimf(w.universe, [0.6, 0.75, 0.9])
            w["very_high"] = fuzz.trapmf(w.universe, [0.85, 0.95, 1, 1])

        # --- RULE SET ---
        rules = [
            ctrl.Rule(
                obs_dist["critical"],
                (
                    heading_w["very_low"],
                    velocity_w["very_low"],
                    clearance_w["very_high"],
                ),
            ),
            ctrl.Rule(
                (obs_dist["close"] | obs_dist["danger"]),
                (heading_w["low"], velocity_w["medium"], clearance_w["medium"]),
            ),
            ctrl.Rule(
                obs_density["dense"],
                (heading_w["low"], velocity_w["low"], clearance_w["high"]),
            ),
            ctrl.Rule(
                obs_dist["safe"] & ~obs_density["dense"],
                (heading_w["medium"], velocity_w["high"], clearance_w["low"]),
            ),
            ctrl.Rule(
                ~robot_vel["fast"] & obs_dist["safe"] & ~obs_position["front"],
                (heading_w["medium"], velocity_w["very_high"], clearance_w["low"]),
            ),
            ctrl.Rule(
                goal_proximity["near"] & goal_angle_error["large"],
                (heading_w["very_high"], velocity_w["low"], clearance_w["medium"]),
            ),
        ]
        self.weight_ctrl = ctrl.ControlSystem(rules)
        self.weight_simulation = ctrl.ControlSystemSimulation(self.weight_ctrl)

    def compute_weights(
        self,
        obs_dist: float,
        obs_density: float,
        robot_vel: float,
        goal_proximity: float,
        goal_angle_error: float,
        obs_position: float,
    ) -> tuple[float, float, float]:
        try:
            self.weight_simulation.input["obs_dist"] = np.clip(obs_dist, 0, 1)
            self.weight_simulation.input["obs_density"] = np.clip(obs_density, 0, 1)
            self.weight_simulation.input["robot_vel"] = np.clip(robot_vel, 0, 1)
            self.weight_simulation.input["goal_proximity"] = np.clip(
                goal_proximity, 0, 1
            )
            self.weight_simulation.input["goal_angle_error"] = np.clip(
                goal_angle_error, 0, 1
            )
            self.weight_simulation.input["obs_position"] = np.clip(obs_position, 0, 1)

            self.weight_simulation.compute()

            heading = self.weight_simulation.output["heading_w"]
            velocity = self.weight_simulation.output["velocity_w"]
            clearance = self.weight_simulation.output["clearance_w"]

            # Normalize weights to sum to 1.0
            total = heading + velocity + clearance + 1e-6
            return (heading / total, velocity / total, clearance / total)

        except Exception as e:
            print(f"⚠️ Fuzzy computation failed ({e})")
            print(f"  obs_dist={obs_dist:.3f}, obs_density={obs_density:.3f}")
            print(f"  robot_vel={robot_vel:.3f}, goal_proximity={goal_proximity:.3f}")
            print(
                f"  goal_angle_error={goal_angle_error:.3f}, obs_position={obs_position:.3f}"
            )
            return 0.30, 0.35, 0.35

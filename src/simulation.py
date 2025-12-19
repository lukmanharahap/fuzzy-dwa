import numpy as np
import pygame
from collections import deque
import time
from datetime import datetime
import json
import sys
import os
from typing import Dict, Any, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from environment import DynamicEnvironment
import config as cfg


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        elif isinstance(o, (np.floating,)):
            return float(o)
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        elif isinstance(o, (np.bool,)):
            return bool(o)
        return super().default(o)


class Simulation:

    def __init__(
        self,
        render_mode: bool = True,
        save_video: bool = False,
        use_fuzzy: bool = False,
        scenario_config: Dict[str, Any] = None,
    ):
        self.render_mode = render_mode
        self.save_video = save_video
        self.use_fuzzy = use_fuzzy

        if scenario_config is None:
            scenario_config = {}

        self.experiment_title = scenario_config.get("name", "default_experiment")
        self.episode_seeds = scenario_config.get("episode_seeds", [])

        fixed_weights = scenario_config.get("fixed_weights", None)

        env_params = {
            "start_pos": scenario_config.get("start_pos"),
            "goal_pos": scenario_config.get("goal_pos"),
            "static_obs_configs": scenario_config.get("static_obs_configs"),
            "n_obstacles": scenario_config.get("n_obstacles"),
            "obstacle_speed_range": scenario_config.get("obstacle_speed_range"),
            "fixed_weights": fixed_weights,
            "save_video": self.save_video,
        }
        env_params = {k: v for k, v in env_params.items() if v is not None}
        self.env = DynamicEnvironment(
            render_mode=render_mode, use_fuzzy=self.use_fuzzy, **env_params
        )
        self.robot = self.env.robot

        if self.render_mode and self.env.renderer:
            self.env.renderer.save_video = self.save_video
            if self.save_video:
                import os
                import shutil

                if os.path.exists("video_frames"):
                    shutil.rmtree("video_frames")
                os.makedirs("video_frames")

        self.episode_stats = []
        self.replan_count = 0
        print(f"✅ Simulation initialized for: {self.experiment_title}\n")

    def run_episode(
        self, max_steps: int = 3000, verbose: bool = True, seed: int = None
    ) -> dict:
        if seed is not None:
            np.random.seed(seed)

        self.env.reset_environment(seed=seed)
        self.replan_count = 0
        delay_steps = cfg.SENSOR_DELAY_STEPS
        self.odometry_buffer = deque(maxlen=delay_steps + 1)
        self.lidar_buffer = deque(maxlen=delay_steps + 1)

        robot_state_belief = self.robot.get_odometry_state()
        goal_pos = self.env.goal_position
        obstacles = self.env.get_static_obstacle_points()
        self.robot.update_goal(goal_pos, obstacles)

        self.last_progress_time = time.time()
        self.robot_stuck = False

        if verbose:
            print("─" * 60 + f"\nEpisode Started (Seed: {seed})\n" + "─" * 60)
            print(f"Start Position:  {robot_state_belief['position'].round(2)}")
            print(f"Goal Position:   {goal_pos}\n" + "─" * 60 + "\n")
            print(
                f"Euclidean Path:  {self.env.euclidean_path_length:.2f} m\n"
                + "─" * 60
                + "\n"
            )
            print(
                f"A* Path Length:  {self.robot.optimal_path_length:.2f} m\n"
                + "─" * 60
                + "\n"
            )

        step = 0
        task_complete = False

        for _ in range(delay_steps + 1):
            self.odometry_buffer.append(self.robot.get_odometry_state())
            self.lidar_buffer.append(self.robot.perform_lidar_scan())

        robot_path_history = []
        episode_start_time = time.time()

        while step < max_steps and not task_complete:
            delayed_odometry = self.odometry_buffer.popleft()
            delayed_lidar = self.lidar_buffer.popleft()
            action = self.robot.compute_action(delayed_odometry, delayed_lidar)
            self.env.step_simulation(action)
            true_lidar_scan = self.robot.perform_lidar_scan()
            self.robot.update_odometry(cfg.PHYSICS_TIME_STEP)
            self.lidar_buffer.append(true_lidar_scan)
            self.odometry_buffer.append(self.robot.get_odometry_state())
            self.check_for_replanning(true_lidar_scan)
            current_true_pos = self.robot.get_true_state()["position"].copy()
            robot_path_history.append(current_true_pos)
            if self.render_mode:
                self.env.render_frame(true_lidar_scan, history=robot_path_history)
                # self.env.render_frame(true_lidar_scan)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        return self._finalize_episode(
                            step,
                            true_lidar_scan,
                            episode_start_time,
                            False,
                            verbose,
                            seed,
                        )
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        return self._finalize_episode(
                            step,
                            true_lidar_scan,
                            episode_start_time,
                            False,
                            verbose,
                            seed,
                        )
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                        if self.env.renderer:
                            self.env.renderer.debug_view_enabled = (
                                not self.env.renderer.debug_view_enabled
                            )
                            print(
                                f"🐞 Debug View: {'ON' if self.env.renderer.debug_view_enabled else 'OFF'}"
                            )

            task_complete = self.env.is_task_complete()
            step += 1

        return self._finalize_episode(
            step, true_lidar_scan, episode_start_time, task_complete, verbose, seed
        )

    def run_multiple_episodes(
        self,
        n_episodes: int = 10,
        max_steps: int = 3000,
        pause_between: float = 1.0,
        episode_seeds: List[int] = None,
    ) -> dict:
        print("\n" + "#" * 60)
        print(f"Running {n_episodes} Episodes for: {self.experiment_title}")
        print("#" * 60 + "\n")

        all_stats = []
        if episode_seeds is None:
            print("⚠️ No seeds provided, generating temporary seeds for this run.")
            episode_seeds = [np.random.randint(0, 2**32 - 1) for _ in range(n_episodes)]

        if len(episode_seeds) < n_episodes:
            raise ValueError(
                f"Not enough seeds provided! Got {len(episode_seeds)}, need {n_episodes}"
            )

        for episode in range(n_episodes):
            print(f"\n{'='*60}")
            print(
                f"Experiment '{self.experiment_title}' - Episode {episode + 1}/{n_episodes}"
            )

            current_seed = episode_seeds[episode]
            print(f"Using Seed: {current_seed}")
            print(f"{'='*60}")

            stats = self.run_episode(
                max_steps=max_steps,
                verbose=False,
                seed=current_seed,
            )
            stats["episode"] = episode + 1
            stats["seed_used"] = current_seed
            all_stats.append(stats)

            result = "✅ SUCCESS" if stats["success"] else "❌ FAILED"
            print(
                f"Episode {episode + 1} (Seed: {current_seed}) finished: {result} in {stats['time_seconds']:.2f}s "
                f"({stats['steps']} steps, {stats['collisions']} coll, {stats['replan_count']} replans, "
                f"dist {stats['distance_traveled']:.2f}m, "
                f"eff: {stats['path_efficiency']:.2f})"
            )

            if episode < n_episodes - 1 and self.render_mode:
                time.sleep(pause_between)

        summary = self._compute_summary(all_stats)
        self._print_summary(summary, n_episodes)
        self._save_to_json(all_stats, summary, self.experiment_title)
        return summary

    def _compute_summary(self, all_stats: list) -> dict:
        if not all_stats:
            return {}

        successes = [s["success"] for s in all_stats]
        steps = [s["steps"] for s in all_stats]
        times = [s["time_seconds"] for s in all_stats]
        collisions = [s["collisions"] for s in all_stats]
        distances = [s["distance_traveled"] for s in all_stats]
        efficiency = [
            s["path_efficiency"] for s in all_stats if s["path_efficiency"] > 0
        ]
        avg_speed = [s["average_speed"] for s in all_stats]
        smoothness = [s["smoothness"] for s in all_stats]
        clearance = [s["min_clearance"] for s in all_stats]
        replans = [s["replan_count"] for s in all_stats]

        def robust_mean(data):
            return np.mean(data) if data else 0.0

        def robust_std(data):
            return np.std(data) if data else 0.0

        def robust_min(data):
            return np.min(data) if data else 0.0

        def robust_max(data):
            return np.max(data) if data else 0.0

        return {
            "total_episodes": len(all_stats),
            "success_rate": robust_mean(successes) * 100,
            "mean_steps": robust_mean(steps),
            "min_steps": robust_min(steps),
            "max_steps": robust_max(steps),
            "std_steps": robust_std(steps),
            "mean_time": robust_mean(times),
            "std_time": robust_std(times),
            "mean_collisions": robust_mean(collisions),
            "std_collisions": robust_std(collisions),
            "min_collisions": robust_min(collisions),
            "max_collisions": robust_max(collisions),
            "mean_distance": robust_mean(distances),
            "std_distance": robust_std(distances),
            "mean_path_efficiency": robust_mean(efficiency),
            "std_path_efficiency": robust_std(efficiency),
            "mean_avg_speed": robust_mean(avg_speed),
            "std_avg_speed": robust_std(avg_speed),
            "min_avg_speed": robust_min(avg_speed),
            "max_avg_speed": robust_max(avg_speed),
            "mean_smoothness": (
                np.nanmean(smoothness) if not all(np.isnan(smoothness)) else 0.0
            ),
            "mean_clearance": robust_mean(clearance),
            "std_clearance": robust_std(clearance),
            "min_clearance": robust_min(clearance),
            "mean_replans": robust_mean(replans),
            "std_replans": robust_std(replans),
        }

    def _print_summary(self, summary: dict, n_episodes: int):
        print("\n" + "#" * 60)
        print(f"Summary for '{self.experiment_title}' ({n_episodes} episodes)")
        print("#" * 60)
        print(f"Success Rate:          {summary['success_rate']:.1f}%")
        print(
            f"Mean Steps:            {summary['mean_steps']:.1f} ± {summary['std_steps']:.1f}"
        )
        print(f"  (min: {summary['min_steps']}, max: {summary['max_steps']})")
        print(
            f"Mean Time:             {summary['mean_time']:.2f} ± {summary['std_time']:.2f} s"
        )
        print(
            f"Mean Collisions:       {summary['mean_collisions']:.2f} ± {summary['std_collisions']:.2f}"
        )
        print(f"  (min: {summary['min_collisions']}, max: {summary['max_collisions']})")
        print(
            f"Mean Distance:         {summary['mean_distance']:.1f} ± {summary['std_distance']:.1f} m"
        )
        print(
            f"Mean Path Efficiency:  {summary['mean_path_efficiency']:.3f} ± {summary['std_path_efficiency']:.3f} (1.0 = optimal)"
        )
        print(
            f"Mean Avg Speed:        {summary['mean_avg_speed']:.2f} ± {summary['std_avg_speed']:.2f} m/s"
        )
        print(
            f"  (min: {summary['min_avg_speed']:.2f} m/s, max: {summary['max_avg_speed']:.2f} m/s)"
        )
        print(f"Mean Smoothness:       {summary['mean_smoothness']:.4f}")
        print(
            f"Mean Clearance:        {summary['mean_clearance']:.3f} ± {summary['std_clearance']:.3f} m"
        )
        print(f"  (min: {summary['min_clearance']:.3f} m)")
        print(
            f"Mean Replans (stuck):  {summary['mean_replans']:.2f} ± {summary['std_replans']:.2f}"
        )
        print("#" * 60 + "\n")

    def _save_to_json(
        self,
        all_stats: list,
        summary: dict,
        experiment_title: str,
    ):
        os.makedirs("results", exist_ok=True)
        filename = f"results/experiments_log.json"

        data_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_title": experiment_title,
            "arena_config": {
                "arena_size_m": getattr(cfg, "ARENA_SIZE_M", None),
                "n_obstacles": self.env.n_obstacles,
                "obstacle_speed_range": self.env.obstacle_speed_range,
                "start_pos": self.env.start_position.tolist(),
                "goal_pos": self.env.goal_position.tolist(),
                "euclidean_path_length": self.env.euclidean_path_length,
                "A_star_path_length": self.robot.optimal_path_length,
            },
            "controller_config": {
                "use_fuzzy": self.use_fuzzy,
                "fixed_heading_weight": self.robot.controller.heading_weight,
                "fixed_velocity_weight": self.robot.controller.velocity_weight,
                "fixed_clearance_weight": self.robot.controller.clearance_weight,
            },
            "episode_seeds_used": self.episode_seeds,
            "summary": summary,
            "episodes": all_stats,
        }

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        if not isinstance(existing_data, list):
            existing_data = [existing_data]

        existing_data.append(data_entry)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, cls=NumpyEncoder)

        print(f"💾 Results saved to: {filename}")

    def _finalize_episode(
        self,
        steps: int,
        lidar_scan: np.ndarray,
        start_time: float,
        success: bool,
        verbose: bool,
        seed: int = None,
    ) -> dict:
        comp_time = time.time() - start_time
        elapsed_time = steps * cfg.PHYSICS_TIME_STEP
        info = self.env.get_info_dict(lidar_scan)

        avg_speed = info["distance_traveled"] / elapsed_time if elapsed_time > 0 else 0
        min_clearance = np.min(lidar_scan)

        if self.robot.velocity_history and len(self.robot.velocity_history) > 1:
            speeds = np.array(self.robot.velocity_history)
            speed_changes = np.abs(np.diff(speeds))
            smoothness = np.mean(speed_changes)
        else:
            smoothness = np.nan

        euclidean_length = info.get("euclidean_path_length", 0.0)
        actual_length = info.get("distance_traveled", 1e-6)
        if actual_length > 0.0 and euclidean_length > 0.0 and success:
            path_efficiency = euclidean_length / actual_length
            path_efficiency = min(path_efficiency, 1.0)
        else:
            path_efficiency = 0.0

        stats = {
            "episode": 0,
            "seed_used": seed,
            "success": success,
            "steps": steps,
            "time_seconds": elapsed_time,
            "computation_time_seconds": comp_time,
            "collisions": info.get("collision_count", 0.0),
            "distance_traveled": actual_length,
            "euclidean_path_length": euclidean_length,
            "path_efficiency": path_efficiency,
            "average_speed": avg_speed,
            "smoothness": smoothness,
            "min_clearance": float(min_clearance),
            "replan_count": self.replan_count,
        }
        self.episode_stats.append(stats)

        if verbose:
            print("\n" + "─" * 60 + f"\nEpisode Finished (Seed: {seed})\n" + "─" * 60)
            print(f"Result:           {'✅ SUCCESS' if success else '❌ FAILED'}")
            print(f"Time:             {stats['time_seconds']:.2f}s")
            print(f"Computation Time: {stats['computation_time_seconds']:.2f}s")
            print(f"Steps:            {stats['steps']}")
            print(f"Collisions:       {stats['collisions']}")
            print(f"Replans (stuck):  {stats['replan_count']}")
            print(f"Euclidean Path:   {stats['euclidean_path_length']:.2f} m")
            print(f"Actual Path:      {stats['distance_traveled']:.2f} m")
            print(f"Path Efficiency:  {stats['path_efficiency']:.3f}")
            print(f"Avg. Speed:       {stats['average_speed']:.2f} m/s")
            print(f"Min Clearance:    {stats['min_clearance']:.3f} m")
            if stats["smoothness"] is not np.nan:
                print(f"Smoothness:       {stats['smoothness']:.4f}")
            print("─" * 60 + "\n")

        return stats

    def check_for_replanning(self, current_lidar_scan: np.ndarray):
        replanning_needed = False
        current_vel = np.linalg.norm(self.robot.get_true_state()["linear_velocity"])
        if current_vel > 0.05:
            self.last_progress_time = time.time()
            self.robot_stuck = False
        if (
            time.time() - self.last_progress_time > cfg.STUCK_TIME_LIMIT
        ) and not self.robot_stuck:
            print(f"⚠️ Robot is stuck! Requesting replan...")
            replanning_needed = True
            self.robot_stuck = True
        if replanning_needed:
            self.replan_count += 1
            static_map = self.env.get_static_obstacle_points()
            robot_belief_state = self.robot.get_odometry_state()
            sensed_dynamic_obstacles = self.robot.controller._lidar_to_obstacles(
                current_lidar_scan, robot_belief_state
            )
            all_known_obstacles = static_map + [
                tuple(p) for p in sensed_dynamic_obstacles
            ]
            goal_pos = self.env.goal_position
            self.robot.update_goal(goal_pos, all_known_obstacles)
            self.last_progress_time = time.time()

    def close(self):
        self.env.close()
        print("\n👋 Simulation closed successfully")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Warehouse AMR Simulation with DWA Controller"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Run without visualization"
    )
    parser.add_argument("--fuzzy", action="store_true", help="Use Fuzzy DWA controller")
    parser.add_argument(
        "--save-video", action="store_true", help="Save video of the simulation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run a single episode with a specific seed",
    )

    args = parser.parse_args()

    render_mode = not args.no_render
    sim = None
    try:
        PLAIN = []

        BASELINE = [
            ((5.0, 2.5), (1.5, 1.5)),
            ((5.0, 7.5), (1.5, 1.5)),
        ]

        SIMPLE = [
            ((3.0, 7.0), (3.0, 0.5)),
            ((8.9, 7.0), (1.1, 0.5)),
        ]

        LAYOUT_1 = [  # GOOD USED
            # A simple "S" chicane
            ((3.5, 3.5), (3.5, 0.2)),
            ((6.5, 6.5), (3.5, 0.2)),
        ]

        LAYOUT_2 = [  # GOOD USED
            ((2.0, 2.0), (0.2, 0.2)),
            ((3.0, 3.0), (0.2, 0.2)),
            ((2.0, 4.0), (0.2, 0.2)),
            ((3.0, 5.0), (0.2, 0.2)),
            ((2.0, 6.0), (0.2, 0.2)),
            ((3.0, 7.0), (0.2, 0.2)),
            ((2.0, 8.0), (0.2, 0.2)),
            ((5.0, 1.0), (0.2, 0.2)),
            ((5.0, 3.0), (0.2, 0.2)),
            ((5.0, 5.0), (0.2, 0.2)),
            ((5.0, 7.0), (0.2, 0.2)),
            ((5.0, 9.0), (0.2, 0.2)),
            ((8.0, 2.0), (0.2, 0.2)),
            ((7.0, 3.0), (0.2, 0.2)),
            ((8.0, 4.0), (0.2, 0.2)),
            ((7.0, 5.0), (0.2, 0.2)),
            ((8.0, 6.0), (0.2, 0.2)),
            ((7.0, 7.0), (0.2, 0.2)),
            ((8.0, 8.0), (0.2, 0.2)),
        ]

        LAYOUT_3 = [  # GOOD USED
            ((5.0, 2.5), (3.6, 0.2)),
            ((5.0, 7.5), (3.6, 0.2)),
            ((2.1, 5.0), (2.1, 0.2)),
            ((7.9, 5.0), (2.1, 0.2)),
        ]

        LAYOUT_4 = [  # GOOD USED
            ((2.0, 5.0), (0.2, 3.5)),
            ((5.0, 2.5), (1.5, 0.2)),
            ((5.0, 7.5), (1.5, 0.2)),
            ((8.0, 5.0), (0.2, 3.5)),
        ]

        manual_scenario = {
            "name": ("ManualRun_Fuzzy" if args.fuzzy else "ManualRun_Standard"),
            "start_pos": np.array([5.0, 1.0]),
            "goal_pos": np.array([5.0, 9.0]),
            "static_obs_configs": LAYOUT_3,
            "n_obstacles": 8,
            "obstacle_speed_range": (0.8, 1.5),
            "fixed_weights": {"heading": 0.2, "velocity": 0.2, "clearance": 0.6},
        }

        sim = Simulation(
            render_mode=render_mode,
            use_fuzzy=args.fuzzy,
            scenario_config=manual_scenario,
            save_video=args.save_video,
        )

        if args.episodes == 1:
            seed = (
                args.seed if args.seed is not None else np.random.randint(0, 2**31 - 1)
            )
            print(f"--- Running single episode with seed: {seed} ---")
            sim.run_episode(max_steps=args.max_steps, verbose=True, seed=seed)
        else:
            episode_seeds = [
                np.random.randint(0, 2**31 - 1) for _ in range(args.episodes)
            ]
            print(f"--- Running {args.episodes} episodes with generated seeds ---")
            sim.run_multiple_episodes(
                n_episodes=args.episodes,
                max_steps=args.max_steps,
                episode_seeds=episode_seeds,
            )

    except Exception as e:
        import traceback

        print(f"\n\n❌ Error occurred: {e}")
        traceback.print_exc()
    finally:
        if sim:
            sim.close()


if __name__ == "__main__":
    main()

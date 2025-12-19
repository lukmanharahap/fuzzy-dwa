import numpy as np
import sys
import os
from typing import List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.simulation import Simulation

# --- Layouts ---
# Layout 1: Simple S-Chicane
LAYOUT_1 = [
    ((3.5, 3.5), (3.5, 0.2)),
    ((6.5, 6.5), (3.5, 0.2)),
]

# Layout 2: Dense Clutter
LAYOUT_2 = [
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

# Layout 3: Narrow Cross
LAYOUT_3 = [
    ((5.0, 2.5), (3.6, 0.2)),
    ((5.0, 7.5), (3.6, 0.2)),
    ((2.1, 5.0), (2.1, 0.2)),
    ((7.9, 5.0), (2.1, 0.2)),
]

# Layout 4: Box Trap
LAYOUT_4 = [
    ((2.0, 5.0), (0.2, 3.5)),
    ((5.0, 2.5), (1.5, 0.2)),
    ((5.0, 7.5), (1.5, 0.2)),
    ((8.0, 5.0), (0.2, 3.5)),
]

# --- Define start and goal pos ---
TASK_DIAGONAL_1 = {"start_pos": np.array([1.0, 1.0]), "goal_pos": np.array([9.0, 9.0])}
TASK_DIAGONAL_2 = {"start_pos": np.array([1.0, 9.0]), "goal_pos": np.array([9.0, 1.0])}
TASK_STRAIGHT_LINE = {
    "start_pos": np.array([5.0, 1.0]),
    "goal_pos": np.array([5.0, 9.0]),
}


# --- Define dynamic levels ---
DYNAMICS_SIMPLE = {"n_obstacles": 2, "obstacle_speed_range": (0.3, 0.8)}
DYNAMICS_COMPLEX = {"n_obstacles": 8, "obstacle_speed_range": (0.8, 1.5)}


# --- Scenarios (4 Layouts x 2 Dynamic Levels) ---
SCENARIOS = [
    {
        "name": "L1_Simple",
        "layout": LAYOUT_1,
        "task": TASK_DIAGONAL_1,
        "obstacles": DYNAMICS_SIMPLE,
    },
    {
        "name": "L1_Complex",
        "layout": LAYOUT_1,
        "task": TASK_DIAGONAL_1,
        "obstacles": DYNAMICS_COMPLEX,
    },
    {
        "name": "L2_Simple",
        "layout": LAYOUT_2,
        "task": TASK_DIAGONAL_2,
        "obstacles": DYNAMICS_SIMPLE,
    },
    {
        "name": "L2_Complex",
        "layout": LAYOUT_2,
        "task": TASK_DIAGONAL_2,
        "obstacles": DYNAMICS_COMPLEX,
    },
    {
        "name": "L3_Simple",
        "layout": LAYOUT_3,
        "task": TASK_STRAIGHT_LINE,
        "obstacles": DYNAMICS_SIMPLE,
    },
    {
        "name": "L3_Complex",
        "layout": LAYOUT_3,
        "task": TASK_STRAIGHT_LINE,
        "obstacles": DYNAMICS_COMPLEX,
    },
    {
        "name": "L4_Simple",
        "layout": LAYOUT_4,
        "task": TASK_DIAGONAL_1,
        "obstacles": DYNAMICS_SIMPLE,
    },
    {
        "name": "L4_Complex",
        "layout": LAYOUT_4,
        "task": TASK_DIAGONAL_1,
        "obstacles": DYNAMICS_COMPLEX,
    },
]

# --- Experiment Settings ---
N_EPISODES_PER_EXPERIMENT = 50
MAX_STEPS_PER_EPISODE = 3000

# --- Controller Types (3 Fixed + 1 Fuzzy) ---
CONTROLLER_TYPES = [
    {
        "name": "DWA_Efficient",
        "use_fuzzy": False,
        "fixed_weights": {"heading": 0.2, "velocity": 0.6, "clearance": 0.2},
    },
    {
        "name": "DWA_Safety",
        "use_fuzzy": False,
        "fixed_weights": {"heading": 0.2, "velocity": 0.2, "clearance": 0.6},
    },
    {
        "name": "DWA_Balance",
        "use_fuzzy": False,
        "fixed_weights": {"heading": 0.2, "velocity": 0.35, "clearance": 0.45},
    },
    {"name": "Fuzzy_DWA", "use_fuzzy": True, "fixed_weights": None},
]


def generate_seeds(n: int, scenario_index: int) -> List[int]:
    master_seed = 42 + scenario_index
    rng = np.random.RandomState(master_seed)
    max_int32 = (2**31) - 1
    episode_seeds = [rng.randint(0, max_int32) for _ in range(n)]
    return episode_seeds


def run_full_experiment():
    print("=" * 80)
    print("STARTING FULL EXPERIMENT SUITE")
    print(f"Scenarios: {len(SCENARIOS)}, Controllers: {len(CONTROLLER_TYPES)}")
    print(f"Episodes per run: {N_EPISODES_PER_EXPERIMENT}")
    print("=" * 80 + "\n")

    total_runs = len(SCENARIOS) * len(CONTROLLER_TYPES)
    current_run = 1
    for scenario_index, scenario in enumerate(SCENARIOS):
        episode_seeds = generate_seeds(N_EPISODES_PER_EXPERIMENT, scenario_index)
        print(
            f"\n--- Starting {scenario['name']} (using {len(episode_seeds)} seeds) ---"
        )
        for controller in CONTROLLER_TYPES:
            print("\n" + "*" * 80)
            print(f"RUN {current_run}/{total_runs}")

            experiment_name = f"{scenario['name']}_{controller['name']}"

            scenario_config = {
                "name": experiment_name,
                "start_pos": scenario["task"]["start_pos"],
                "goal_pos": scenario["task"]["goal_pos"],
                "static_obs_configs": scenario["layout"],
                "n_obstacles": scenario["obstacles"]["n_obstacles"],
                "obstacle_speed_range": scenario["obstacles"]["obstacle_speed_range"],
                "episode_seeds": episode_seeds,
                "fixed_weights": controller.get("fixed_weights", None),
            }

            print(f"Starting: {experiment_name}")
            print(
                f"Using {len(episode_seeds)} shared seeds, starting with {episode_seeds[0]}"
            )
            print("*" * 80 + "\n")

            sim = None
            try:
                sim = Simulation(
                    render_mode=False,
                    use_fuzzy=controller["use_fuzzy"],
                    scenario_config=scenario_config,
                )

                sim.run_multiple_episodes(
                    n_episodes=N_EPISODES_PER_EXPERIMENT,
                    max_steps=MAX_STEPS_PER_EPISODE,
                    episode_seeds=episode_seeds,
                )

            except Exception as e:
                import traceback

                print(f"\n\n❌❌❌ ERROR during experiment: {experiment_name} ❌❌❌")
                traceback.print_exc()
            finally:
                if sim:
                    sim.close()

            print(f"\nFinished: {experiment_name}")
            current_run += 1

    print("\n" + "=" * 80)
    print("🎉 FULL EXPERIMENT SUITE COMPLETE 🎉")
    print(f"All results saved in {os.path.abspath('results/experiments_log.json')}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_experiment()

import torch
import torch.nn as nn
from evox import algorithms
from evox.problems.neuroevolution.mujoco_playground import MujocoProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow
from pulse14 import PulseGreedy
import time
import os
import numpy as np

os.environ["MUJOCO_GL"] = "osmesa"

# ============================================================================
# CONFIG — flip DRY_RUN to False for paper-quality
# ============================================================================
DRY_RUN = True

if DRY_RUN:
    POP_SIZE = 64
    GENERATIONS = 5
    N_SEEDS = 1
    NUM_EPISODES = 2
    MAX_EPISODE_LENGTH = 200
    ENV_NAMES = ["CartpoleSwingup"]
else:
    POP_SIZE = 512
    GENERATIONS = 300
    N_SEEDS = 3
    NUM_EPISODES = 5
    MAX_EPISODE_LENGTH = 1000
    ENV_NAMES = [
        "CartpoleSwingup",   # Easy   (obs=5,  act=1,  ~26K params)
        "WalkerWalk",        # Medium (obs=24, act=6,  ~28K params)
        "HumanoidWalk",      # V.hard (obs=67, act=21, ~34K params)
    ]

HIDDEN = 128  # Standard for neuroevo papers (OpenAI-ES, EvoJAX)

# Obs/action dims per environment (queried from MuJoCo Playground on GPU)
ENV_DIMS = {
    "CartpoleSwingup":  (5,  1),
    "CartpoleBalance":  (5,  1),
    "HopperHop":        (15, 4),
    "HopperStand":      (15, 4),
    "WalkerWalk":       (24, 6),
    "WalkerRun":        (24, 6),
    "WalkerStand":      (24, 6),
    "CheetahRun":       (17, 6),
    "HumanoidWalk":     (67, 21),
    "HumanoidRun":      (67, 21),
    "HumanoidStand":    (67, 21),
    "SwimmerSwimmer6":  (13, 5),
    "FishSwim":         (24, 5),
    "ReacherEasy":      (6,  2),
    "ReacherHard":      (6,  2),
    "PendulumSwingup":  (3,  1),
    "FingerSpin":       (9,  2),
    "FingerTurnEasy":   (12, 2),
    "FingerTurnHard":   (12, 2),
    "AcrobotSwingup":   (6,  1),
    "BallInCup":        (8,  2),
    "PointMass":        (4,  2),
}

# Camera names per environment (queried from MuJoCo Playground on GPU)
ENV_CAMERAS = {
    "CartpoleSwingup":  ["fixed", "lookatcart"],
    "CartpoleBalance":  ["fixed"],
    "HopperHop":        ["cam0", "back"],
    "HopperStand":      ["cam0", "back"],
    "WalkerWalk":       ["side", "back"],
    "WalkerRun":        ["side", "back"],
    "WalkerStand":      ["side", "back"],
    "CheetahRun":       ["side", "back"],
    "HumanoidWalk":     ["back", "side"],
    "HumanoidRun":      ["back", "side"],
    "HumanoidStand":    ["back", "side"],
}


# ============================================================================
# Network — auto-sized per environment
# ============================================================================
class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


# ============================================================================
# Main
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'PAPER QUALITY'}")
    print(f"Pop: {POP_SIZE}  Gens: {GENERATIONS}  Seeds: {N_SEEDS}")
    print(f"Envs: {ENV_NAMES}\n")

    # Storage
    all_results = {}  # {env: {algo: {seed: {"final": float, "trajectory": list}}}}
    total_t0 = time.time()

    for env_name in ENV_NAMES:
        obs_dim, act_dim = ENV_DIMS[env_name]
        model = PolicyMLP(obs_dim, act_dim, HIDDEN).to(device)
        adapter = ParamsAndVector(dummy_model=model)
        pop_center = adapter.to_vector(dict(model.named_parameters()))
        n_params = len(pop_center)
        b = 1
        lb = torch.full_like(pop_center, -b)
        ub = torch.full_like(pop_center, b)

        print(f"\n{'='*70}")
        print(f"ENV: {env_name}  (obs={obs_dim}, act={act_dim}, params={n_params:,})")
        print(f"{'='*70}")

        all_results[env_name] = {}

        # Algorithm factories (must be inside loop since lb/ub/n_params change per env)
        algo_configs = {
            "OpenES": lambda: algorithms.OpenES(
                pop_size=POP_SIZE,
                center_init=torch.zeros(n_params, device=device),
                learning_rate=0.05, noise_stdev=0.03, device=device,
            ),
            "DE": lambda: algorithms.DE(
                pop_size=POP_SIZE, lb=lb, ub=ub, device=device,
            ),
            "PSO": lambda: algorithms.PSO(
                pop_size=POP_SIZE, lb=lb, ub=ub, device=device,
            ),
            "LinePulse": lambda: PulseGreedy(
                pop_size=POP_SIZE, dim=n_params, lb=-b, ub=b,
                patience=1, device=device,
            ),
        }
        algo_names = list(algo_configs.keys())

        for algo_name in algo_names:
            all_results[env_name][algo_name] = {}

            for seed_idx in range(N_SEEDS):
                seed = 777 + seed_idx * 111

                print(f"\n  {algo_name} | seed {seed_idx+1}/{N_SEEDS} (seed={seed})")
                torch.manual_seed(seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(seed)

                # Fresh model + problem per run
                policy = PolicyMLP(obs_dim, act_dim, HIDDEN).to(device)
                problem = MujocoProblem(
                    policy=policy,
                    env_name=env_name,
                    max_episode_length=MAX_EPISODE_LENGTH,
                    num_episodes=NUM_EPISODES,
                    pop_size=POP_SIZE,
                    device=device,
                )

                algorithm = algo_configs[algo_name]()
                monitor = EvalMonitor(topk=1, device=device)
                workflow = StdWorkflow(
                    algorithm=algorithm,
                    problem=problem,
                    solution_transform=adapter,
                    monitor=monitor,
                    opt_direction="max",
                    device=device,
                )

                t0 = time.time()
                workflow.init_step()

                trajectory = []
                for gen in range(GENERATIONS):
                    gen_t0 = time.time()
                    workflow.step()
                    gen_dt = time.time() - gen_t0

                    best = float(monitor.get_best_fitness())
                    trajectory.append(best)

                    log_interval = max(1, GENERATIONS // 10)
                    if (gen + 1) % log_interval == 0 or gen == 0:
                        print(f"    Gen {gen+1:>4}/{GENERATIONS}  best={best:>8.2f}  ({gen_dt:.1f}s)")

                run_time = time.time() - t0
                final_best = float(monitor.get_best_fitness())
                all_results[env_name][algo_name][seed_idx] = {
                    "final": final_best,
                    "trajectory": trajectory,
                    "time": run_time,
                }
                print(f"    DONE: best={final_best:.2f}  time={run_time:.1f}s")

                # Save best params + GIF (last seed only)
                if seed_idx == N_SEEDS - 1:
                    os.makedirs("resources/neuroevolution/model_weights", exist_ok=True)
                    os.makedirs("resources/neuroevolution/gifs", exist_ok=True)
                    best_params = adapter.to_params(monitor.get_best_solution())
                    torch.save(
                        best_params,
                        f"resources/neuroevolution/model_weights/{env_name}_{algo_name}_best.pt",
                    )
                    # Render GIFs from multiple camera angles
                    cameras = ENV_CAMERAS.get(env_name, ["side", "back"])
                    for cam in cameras:
                        try:
                            problem.visualize(
                                best_params, output_type="gif",
                                output_path=f"resources/neuroevolution/gifs/{env_name}_{algo_name}_{cam}",
                                camera=cam,
                            )
                            print(f"    Saved GIF: {env_name}_{algo_name}_{cam}.gif")
                        except Exception as e:
                            print(f"    (viz {cam} failed: {e})")

    total_time = time.time() - total_t0

    # ========================================================================
    # Summary
    # ========================================================================
    algo_names = list(list(all_results.values())[0].keys())

    print(f"\n\n{'='*70}")
    print(f"FINAL RESULTS  (total: {total_time/60:.1f} min)")
    print(f"{'='*70}")

    for env_name in ENV_NAMES:
        obs_dim, act_dim = ENV_DIMS[env_name]
        n_p = obs_dim * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN * 64 + 64 + 64 * act_dim + act_dim
        print(f"\n  {env_name} ({n_p:,} params):")

        medians = {}
        for algo_name in algo_names:
            finals = [all_results[env_name][algo_name][s]["final"] for s in range(N_SEEDS)]
            medians[algo_name] = np.median(finals)
        best_val = max(medians.values())
        for algo_name in sorted(medians, key=lambda x: -medians[x]):
            marker = " *" if medians[algo_name] == best_val else ""
            print(f"    {algo_name:<12}  median={medians[algo_name]:>8.2f}{marker}")

    # Average rank
    if len(ENV_NAMES) > 1:
        print(f"\n  AVERAGE RANK:")
        rank_data = {name: [] for name in algo_names}
        for env_name in ENV_NAMES:
            medians = {}
            for algo_name in algo_names:
                finals = [all_results[env_name][algo_name][s]["final"] for s in range(N_SEEDS)]
                medians[algo_name] = np.median(finals)
            for rank, name in enumerate(sorted(medians, key=lambda x: -medians[x])):
                rank_data[name].append(rank + 1)
        for name in sorted(rank_data, key=lambda x: np.mean(rank_data[x])):
            print(f"    {name:<12}  avg_rank={np.mean(rank_data[name]):.2f}  {rank_data[name]}")


if __name__ == "__main__":
    main()
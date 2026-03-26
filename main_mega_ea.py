"""
Mega Neuroevolution Benchmark — Subprocess-Isolated Version
===========================================================
Each (env, algo, seed) run is isolated in a subprocess.
When the subprocess exits, ALL GPU memory is freed (PyTorch + JAX + WARP).
This prevents the WARP allocator fragmentation that caused OpenES OOM crashes.

Usage:
    MUJOCO_GL=osmesa python3 -u main_mega_ea.py 2>&1 | tee resources/mega/ea_run.log
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import numpy as np

os.environ["MUJOCO_GL"] = "osmesa"

# ============================================================================
# CONFIG
# ============================================================================
DRY_RUN = True

if DRY_RUN:
    POP_SIZE = 64
    GENERATIONS = 5
    N_SEEDS = 2
    NUM_EPISODES = 2
    MAX_EPISODE_LENGTH = 200
    ENV_NAMES = ["CartpoleSwingup"]
    ALGO_NAMES = ["LinePulse", "LinePulse-NoRay", "DE"]
else:
    POP_SIZE = 512
    GENERATIONS = 300
    N_SEEDS = 10
    NUM_EPISODES = 5
    MAX_EPISODE_LENGTH = 1000
    ENV_NAMES = [
        "CartpoleSwingup",   # Easy   (obs=5,  act=1,  ~3.4K params)
        "HopperHop",         # Low-mid(obs=15, act=4,  ~3.8K params)
        "CheetahRun",        # Medium (obs=17, act=6,  ~3.9K params)
        "WalkerWalk",        # Medium (obs=24, act=6,  ~4.2K params)
        "HumanoidWalk",      # V.hard (obs=67, act=21, ~6.0K params)
    ]
    ALGO_NAMES = [
        "LinePulse",
        "LinePulse-NoRay",   # Ablation
        "DE",
        "PSO",
        "OpenES",
    ]

HIDDEN = 32  # Matches Brax PPO default: policy_hidden_layer_sizes=(32,32,32,32)

# CLI override: python3 main_mega_ea.py WalkerWalk
# Lets you run one env per pod without editing the file
if len(sys.argv) > 1:
    _requested = [a for a in sys.argv[1:] if a in [
        "CartpoleSwingup","HopperHop","CheetahRun","WalkerWalk","HumanoidWalk"
    ]]
    if _requested:
        ENV_NAMES = _requested
        print(f"[CLI] Running envs: {ENV_NAMES}")
    else:
        print(f"[CLI] Unknown env(s): {sys.argv[1:]}. Valid: CartpoleSwingup, HopperHop, CheetahRun, WalkerWalk, HumanoidWalk")
        sys.exit(1)

ENV_DIMS = {
    "CartpoleSwingup":  (5,  1),
    "CartpoleBalance":  (5,  1),
    "HopperHop":        (15, 4),
    "WalkerWalk":       (24, 6),
    "WalkerRun":        (24, 6),
    "CheetahRun":       (17, 6),
    "HumanoidWalk":     (67, 21),
    "HumanoidRun":      (67, 21),
    "SwimmerSwimmer6":  (13, 5),
    "FishSwim":         (24, 5),
    "ReacherEasy":      (6,  2),
    "AcrobotSwingup":   (6,  1),
}


# ============================================================================
# Worker script (runs as subprocess, one seed at a time)
# ============================================================================
WORKER_SCRIPT = """
import os, sys, json, time, torch, torch.nn as nn
from evox import algorithms
from evox.problems.neuroevolution.mujoco_playground import MujocoProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow
from pulse14 import PulseGreedy
from pulse14_noray import PulseGreedyNoRay

os.environ["MUJOCO_GL"] = "osmesa"

env_name  = sys.argv[1]
algo_name = sys.argv[2]
seed      = int(sys.argv[3])
out_file  = sys.argv[4]

POP_SIZE          = int(sys.argv[5])
GENERATIONS       = int(sys.argv[6])
NUM_EPISODES      = int(sys.argv[7])
MAX_EPISODE_LENGTH= int(sys.argv[8])
HIDDEN            = int(sys.argv[9])

ENV_DIMS = {
    "CartpoleSwingup":(5,1),"CartpoleBalance":(5,1),
    "HopperHop":(15,4),"WalkerWalk":(24,6),"WalkerRun":(24,6),
    "CheetahRun":(17,6),"HumanoidWalk":(67,21),"HumanoidRun":(67,21),
    "SwimmerSwimmer6":(13,5),"FishSwim":(24,5),
    "ReacherEasy":(6,2),"AcrobotSwingup":(6,1),
}

class PolicyMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return torch.tanh(self.net(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_dim, act_dim = ENV_DIMS[env_name]
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)

model = PolicyMLP(obs_dim, act_dim, HIDDEN).to(device)
adapter = ParamsAndVector(dummy_model=model)
pop_center = adapter.to_vector(dict(model.named_parameters()))
n_params = len(pop_center)
b = 1
lb = torch.full_like(pop_center, -b)
ub = torch.full_like(pop_center, b)

def make_algo():
    if algo_name == "LinePulse":
        return PulseGreedy(pop_size=POP_SIZE, dim=n_params, lb=-b, ub=b, patience=1, device=device)
    elif algo_name == "LinePulse-NoRay":
        return PulseGreedyNoRay(pop_size=POP_SIZE, dim=n_params, lb=-b, ub=b, patience=1, device=device)
    elif algo_name == "DE":
        return algorithms.DE(pop_size=POP_SIZE, lb=lb, ub=ub, device=device)
    elif algo_name == "PSO":
        return algorithms.PSO(pop_size=POP_SIZE, lb=lb, ub=ub, device=device)
    elif algo_name == "OpenES":
        return algorithms.OpenES(pop_size=POP_SIZE,
            center_init=torch.zeros(n_params, device=device),
            learning_rate=0.01, noise_stdev=0.02, optimizer="adam", device=device)

policy = PolicyMLP(obs_dim, act_dim, HIDDEN).to(device)
problem = MujocoProblem(policy=policy, env_name=env_name,
    max_episode_length=MAX_EPISODE_LENGTH, num_episodes=NUM_EPISODES,
    pop_size=POP_SIZE, device=device)

algorithm = make_algo()
monitor = EvalMonitor(topk=1, device=device)
workflow = StdWorkflow(algorithm=algorithm, problem=problem,
    solution_transform=adapter, monitor=monitor, opt_direction="max", device=device)

t0 = time.time()
workflow.init_step()
trajectory = []
for gen in range(GENERATIONS):
    workflow.step()
    best = float(monitor.get_best_fitness())
    trajectory.append(best)
    log_interval = max(1, GENERATIONS // 10)
    if (gen + 1) % log_interval == 0 or gen == 0:
        elapsed = time.time() - t0
        print(f"    Gen {gen+1:>4}/{GENERATIONS}  best={best:>8.2f}  ({elapsed:.1f}s)", flush=True)

run_time = time.time() - t0
final_best = float(monitor.get_best_fitness())
print(f"    DONE: best={final_best:.2f}  time={run_time:.1f}s", flush=True)

result = {"final": final_best, "trajectory": trajectory, "time": run_time, "seed": seed}
with open(out_file, "w") as f:
    json.dump(result, f)
"""


# ============================================================================
# Run one seed in a subprocess
# ============================================================================
def run_one_seed(env_name, algo_name, seed_idx, seed):
    """Launches a subprocess for one (env, algo, seed) run. Returns result dict."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_file = tf.name
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as sf:
        sf.write(WORKER_SCRIPT)
        script_file = sf.name

    cmd = [
        sys.executable, script_file,
        env_name, algo_name, str(seed), out_file,
        str(POP_SIZE), str(GENERATIONS), str(NUM_EPISODES),
        str(MAX_EPISODE_LENGTH), str(HIDDEN),
    ]

    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    # Workers run from /tmp/tmpXXX.py — they can't find pulse14.py unless we
    # add the workspace directory (where pulse14.py lives) to PYTHONPATH.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n  {algo_name} | seed {seed_idx+1}/{N_SEEDS} (seed={seed})", flush=True)
    t0 = time.time()

    try:
        proc = subprocess.run(
            cmd, env=env,  # no timeout — runs can take 5-15h on HumanoidWalk
            text=True, capture_output=False,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"subprocess exited with code {proc.returncode}")

        with open(out_file) as f:
            result = json.load(f)

    except Exception as e:
        elapsed = time.time() - t0
        print(f"    ⚠ CRASHED: {e}  ({elapsed:.0f}s)", flush=True)
        result = {"final": 0.0, "trajectory": [], "time": elapsed, "seed": seed, "crashed": True}
    finally:
        for f in [out_file, script_file]:
            try:
                os.unlink(f)
            except Exception:
                pass

    return result


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'PAPER QUALITY — MEGA TEST (subprocess-isolated)'}")
    print(f"Pop: {POP_SIZE}  Gens: {GENERATIONS}  Seeds: {N_SEEDS}")
    print(f"Envs: {ENV_NAMES}")
    print(f"Algos: {ALGO_NAMES}")
    total_runs = len(ENV_NAMES) * len(ALGO_NAMES) * N_SEEDS
    print(f"Total runs: {total_runs}")
    print(f"GPU isolation: each seed runs in a fresh subprocess (prevents WARP OOM)\n")

    all_results = {}
    total_t0 = time.time()

    for env_name in ENV_NAMES:
        obs_dim, act_dim = ENV_DIMS[env_name]
        # Compute param count for display
        n_p = (obs_dim * HIDDEN + HIDDEN +
               3 * (HIDDEN * HIDDEN + HIDDEN) +
               HIDDEN * act_dim + act_dim)

        print(f"\n{'#'*70}")
        print(f"# ENV: {env_name}  (obs={obs_dim}, act={act_dim}, params={n_p:,})")
        print(f"{'#'*70}")

        all_results[env_name] = {}

        for algo_name in ALGO_NAMES:
            all_results[env_name][algo_name] = {}

            for seed_idx in range(N_SEEDS):
                seed = 777 + seed_idx * 111
                result = run_one_seed(env_name, algo_name, seed_idx, seed)
                all_results[env_name][algo_name][seed_idx] = result

        # Incremental save after each env
        _save_results(all_results, ENV_NAMES[:ENV_NAMES.index(env_name)+1])

    total_time = time.time() - total_t0

    # Summary
    print(f"\n\n{'='*70}")
    print(f"MEGA BENCHMARK — FINAL RESULTS  (total: {total_time/60:.1f} min)")
    print(f"{'='*70}")

    for env_name in ENV_NAMES:
        if env_name not in all_results:
            continue
        obs_dim, act_dim = ENV_DIMS[env_name]
        n_p = (obs_dim * HIDDEN + HIDDEN +
               3 * (HIDDEN * HIDDEN + HIDDEN) +
               HIDDEN * act_dim + act_dim)
        print(f"\n  {env_name} ({n_p:,} params):")

        medians = {}
        for algo_name in ALGO_NAMES:
            if algo_name not in all_results[env_name]:
                continue
            finals = [all_results[env_name][algo_name][s]["final"]
                      for s in range(N_SEEDS)
                      if s in all_results[env_name][algo_name]]
            if finals:
                medians[algo_name] = (np.median(finals), np.std(finals))
            else:
                medians[algo_name] = (0.0, 0.0)

        best_val = max(m[0] for m in medians.values())
        for algo_name in sorted(medians, key=lambda x: -medians[x][0]):
            med, std = medians[algo_name]
            marker = " ★" if med == best_val else ""
            print(f"    {algo_name:<18}  median={med:>8.2f}  std={std:>6.2f}{marker}")

    if len(ENV_NAMES) > 1:
        print(f"\n  AVERAGE RANK:")
        rank_data = {name: [] for name in ALGO_NAMES}
        for env_name in ENV_NAMES:
            if env_name not in all_results:
                continue
            env_medians = {}
            for algo_name in ALGO_NAMES:
                if algo_name not in all_results[env_name]:
                    continue
                finals = [all_results[env_name][algo_name][s]["final"]
                          for s in range(N_SEEDS)
                          if s in all_results[env_name][algo_name]]
                env_medians[algo_name] = np.median(finals) if finals else 0.0
            for rank, name in enumerate(sorted(env_medians, key=lambda x: -env_medians[x])):
                rank_data[name].append(rank + 1)
        for name in sorted(rank_data, key=lambda x: np.mean(rank_data[x])):
            ranks = rank_data[name]
            print(f"    {name:<18}  avg_rank={np.mean(ranks):.2f}  {ranks}")

    _save_results(all_results, ENV_NAMES)


def _save_results(all_results, env_names_done):
    os.makedirs("resources/mega", exist_ok=True)
    out_path = "resources/mega/ea_results.json"
    save_data = {}
    for env_name in env_names_done:
        if env_name not in all_results:
            continue
        save_data[env_name] = {}
        for algo_name, seeds in all_results[env_name].items():
            save_data[env_name][algo_name] = {str(k): v for k, v in seeds.items()}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  → Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

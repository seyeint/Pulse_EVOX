"""
Capacity Sweep — LinePulse vs OpenES across network sizes
=========================================================
Proves the SNR collapse of OpenES as d increases.
LinePulse should remain flat (immune to ambient dimensionality).

Only CartpoleSwingup (obs=5, act=1) — fast runs, ~15 min each.

Usage:
    MUJOCO_GL=osmesa python3 -u main_capacity_sweep.py 2>&1 | tee resources/mega/capacity_sweep.log
"""

import os
import sys
import json
import time
import math
import subprocess
import tempfile
import numpy as np

os.environ["MUJOCO_GL"] = "osmesa"

# ============================================================================
# CONFIG
# ============================================================================
DRY_RUN = True

ENV_NAME = "CartpoleSwingup"
OBS_DIM = 5
ACT_DIM = 1

if DRY_RUN:
    POP_SIZE = 64
    GENERATIONS = 5
    N_SEEDS = 2
    NUM_EPISODES = 2
    MAX_EPISODE_LENGTH = 200
    HIDDEN_CONFIGS = [
        [16, 16],
        [32, 32, 32, 32],
    ]
    ALGO_NAMES = ["LinePulse", "OpenES"]
else:
    POP_SIZE = 512
    GENERATIONS = 300
    N_SEEDS = 10  # matches mega EA for uniform methodology
    NUM_EPISODES = 5
    MAX_EPISODE_LENGTH = 1000
    HIDDEN_CONFIGS = [
        [16, 16],           # d=385   SNR=1.15
        [32, 32, 32, 32],   # d=3393  SNR=0.39
        [64, 64],           # d=4609  SNR=0.33
        [128, 128, 64],     # d=25601 SNR=0.14
        [256, 256, 128],    # d=100353 SNR=0.07
    ]
    ALGO_NAMES = ["LinePulse", "OpenES"]


def count_params(obs_dim, act_dim, hidden_layers):
    """Compute total parameter count for a fully-connected MLP."""
    layers = [obs_dim] + hidden_layers + [act_dim]
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i+1] + layers[i+1]  # weight + bias
    return total


def theoretical_snr(n, d):
    """OpenES theoretical SNR = sqrt(N/d)."""
    return math.sqrt(n / d)


# ============================================================================
# Worker script (subprocess per seed, ensures full GPU memory release)
# ============================================================================
WORKER_SCRIPT = """
import os, sys, json, time, math, torch, torch.nn as nn
from evox import algorithms
from evox.problems.neuroevolution.mujoco_playground import MujocoProblem
from evox.utils import ParamsAndVector
from evox.workflows import EvalMonitor, StdWorkflow
from pulse14 import PulseGreedy

os.environ["MUJOCO_GL"] = "osmesa"

env_name   = sys.argv[1]   # CartpoleSwingup
algo_name  = sys.argv[2]   # LinePulse or OpenES
seed       = int(sys.argv[3])
out_file   = sys.argv[4]
hidden_str = sys.argv[5]   # e.g. "32,32,32,32"

POP_SIZE          = int(sys.argv[6])
GENERATIONS       = int(sys.argv[7])
NUM_EPISODES      = int(sys.argv[8])
MAX_EPISODE_LENGTH= int(sys.argv[9])

hidden_layers = [int(x) for x in hidden_str.split(",")]
obs_dim, act_dim = 5, 1  # CartpoleSwingup

class PolicyMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layers):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.tanh(self.net(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)

model = PolicyMLP(obs_dim, act_dim, hidden_layers).to(device)
adapter = ParamsAndVector(dummy_model=model)
pop_center = adapter.to_vector(dict(model.named_parameters()))
n_params = len(pop_center)
b = 1
lb = torch.full_like(pop_center, -b)
ub = torch.full_like(pop_center, b)

def make_algo():
    # IMPORTANT: both algorithms start from the SAME Kaiming-initialized vector (pop_center).
    # This controls initialization as a variable — the only difference is how each
    # algorithm takes its next steps (line search vs. isotropic Gaussian).
    if algo_name == "LinePulse":
        return PulseGreedy(pop_size=POP_SIZE, dim=n_params, lb=-b, ub=b,
                           patience=1, center_init=pop_center, device=device)
    elif algo_name == "OpenES":
        return algorithms.OpenES(pop_size=POP_SIZE,
            center_init=pop_center,  # Kaiming init, not zeros — same start as LinePulse
            learning_rate=0.01, noise_stdev=0.02, optimizer="adam", device=device)

policy = PolicyMLP(obs_dim, act_dim, hidden_layers).to(device)
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
print(f"    DONE: best={final_best:.2f}  time={run_time:.1f}s  params={n_params}", flush=True)

result = {
    "final": final_best, "trajectory": trajectory,
    "time": run_time, "seed": seed, "n_params": n_params
}
with open(out_file, "w") as f:
    json.dump(result, f)
"""


def run_one_seed(env_name, algo_name, hidden_layers, seed_idx, seed):
    """One subprocess per seed — guarantees full GPU memory release on exit."""
    hidden_str = ",".join(str(h) for h in hidden_layers)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_file = tf.name
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as sf:
        sf.write(WORKER_SCRIPT)
        script_file = sf.name

    cmd = [
        sys.executable, script_file,
        env_name, algo_name, str(seed), out_file, hidden_str,
        str(POP_SIZE), str(GENERATIONS), str(NUM_EPISODES), str(MAX_EPISODE_LENGTH),
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    # Workers run from /tmp/tmpXXX.py — add workspace dir to PYTHONPATH so pulse14.py is importable.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = script_dir + os.pathsep + env.get("PYTHONPATH", "")

    print(f"    seed {seed_idx+1}/{N_SEEDS} (seed={seed})", flush=True)

    try:
        subprocess.run(cmd, env=env, text=True, capture_output=False)  # no timeout
        with open(out_file) as f:
            result = json.load(f)
    except Exception as e:
        print(f"    ⚠ CRASHED: {e}", flush=True)
        result = {"final": 0.0, "trajectory": [], "time": 0.0,
                  "seed": seed, "crashed": True,
                  "n_params": count_params(OBS_DIM, ACT_DIM, hidden_layers)}
    finally:
        for fp in [out_file, script_file]:
            try:
                os.unlink(fp)
            except Exception:
                pass

    return result


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'PAPER QUALITY'} — CAPACITY SWEEP")
    print(f"Env: {ENV_NAME}, Algos: {ALGO_NAMES}")
    print(f"Pop: {POP_SIZE}  Gens: {GENERATIONS}  Seeds: {N_SEEDS}")
    total_runs = len(HIDDEN_CONFIGS) * len(ALGO_NAMES) * N_SEEDS
    print(f"Total runs: {total_runs}\n")

    print(f"{'Architecture':<20}  {'d':>7}  {'SNR(OpenES)':>12}")
    print(f"{'─'*45}")
    for hc in HIDDEN_CONFIGS:
        d = count_params(OBS_DIM, ACT_DIM, hc)
        snr = theoretical_snr(POP_SIZE, d)
        print(f"  {str(hc):<18}  {d:>7,}  {snr:>12.3f}")
    print()

    all_results = {}
    total_t0 = time.time()

    for hidden_layers in HIDDEN_CONFIGS:
        d = count_params(OBS_DIM, ACT_DIM, hidden_layers)
        snr = theoretical_snr(POP_SIZE, d)
        arch_key = str(hidden_layers)

        print(f"\n{'='*60}")
        print(f"Architecture: {hidden_layers}  (d={d:,}  SNR={snr:.3f})")
        print(f"{'='*60}")

        all_results[arch_key] = {"d": d, "snr": snr, "hidden": hidden_layers}

        for algo_name in ALGO_NAMES:
            all_results[arch_key][algo_name] = {}
            print(f"\n  {algo_name}:")

            for seed_idx in range(N_SEEDS):
                seed = 777 + seed_idx * 111
                result = run_one_seed(ENV_NAME, algo_name, hidden_layers, seed_idx, seed)
                all_results[arch_key][algo_name][seed_idx] = result

        # Print interim result for this architecture
        for algo_name in ALGO_NAMES:
            finals = [all_results[arch_key][algo_name][s]["final"]
                      for s in range(N_SEEDS) if s in all_results[arch_key][algo_name]]
            if finals:
                med = np.median(finals)
                std = np.std(finals)
                print(f"  {algo_name:<18}  median={med:>8.2f}  std={std:>6.2f}")

        _save_results(all_results)

    total_time = time.time() - total_t0

    # Final summary table
    print(f"\n\n{'='*70}")
    print(f"CAPACITY SWEEP — FINAL RESULTS  (total: {total_time/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n  {'Architecture':<20}  {'d':>7}  {'SNR':>6}  {'LinePulse':>10}  {'OpenES':>10}  {'Winner'}")
    print(f"  {'─'*70}")

    for hidden_layers in HIDDEN_CONFIGS:
        arch_key = str(hidden_layers)
        d = all_results[arch_key]["d"]
        snr = all_results[arch_key]["snr"]

        scores = {}
        for algo_name in ALGO_NAMES:
            finals = [all_results[arch_key][algo_name][s]["final"]
                      for s in range(N_SEEDS) if s in all_results[arch_key][algo_name]]
            scores[algo_name] = np.median(finals) if finals else 0.0

        winner = max(scores, key=scores.get)
        lp = scores.get("LinePulse", 0.0)
        oes = scores.get("OpenES", 0.0)
        print(f"  {str(hidden_layers):<20}  {d:>7,}  {snr:>6.3f}  {lp:>10.1f}  {oes:>10.1f}  ← {winner}")

    _save_results(all_results)


def _save_results(all_results):
    os.makedirs("resources/mega", exist_ok=True)
    out_path = "resources/mega/capacity_sweep_results.json"

    # Convert to JSON-serializable format
    save_data = {}
    for arch_key, arch_data in all_results.items():
        save_data[arch_key] = {
            "d": arch_data["d"],
            "snr": arch_data["snr"],
            "hidden": arch_data["hidden"],
        }
        for algo_name in ALGO_NAMES:
            if algo_name in arch_data:
                save_data[arch_key][algo_name] = {
                    str(k): v for k, v in arch_data[algo_name].items()
                }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  → Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""
PPO Baseline for EA-vs-RL Comparison
=====================================
Uses Brax PPO via MuJoCo Playground on the same environments as main_neuroevo.py.
This is a SEPARATE script — it does NOT use EvoX.

Runs on the same RunPod GPU. Logs reward vs env_steps AND wall-clock.

Usage:
    MUJOCO_GL=osmesa python3 main_ppo_baseline.py

Requirements (in addition to what setup_pod.sh installs):
    pip install ml-collections mediapy etils tensorboardX
"""

import os
import sys
import time
import json
import functools
import warnings
import datetime

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")

# Suppress JAX noise
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params

# ============================================================================
# CONFIG
# ============================================================================
DRY_RUN = True

if DRY_RUN:
    NUM_TIMESTEPS = 1_000_000   # ~1M steps, fast sanity check
    N_SEEDS = 1
    ENV_NAMES = ["CartpoleSwingup"]
else:
    NUM_TIMESTEPS = 60_000_000  # 60M steps — MuJoCo Playground default
    N_SEEDS = 3
    ENV_NAMES = [
        "CartpoleSwingup",
        "WalkerWalk",
        "HumanoidWalk",
    ]

# ============================================================================
# Main
# ============================================================================
def run_ppo(env_name: str, seed: int, num_timesteps: int):
    """Run Brax PPO on a MuJoCo Playground environment. Returns results dict."""

    # Get the tuned config from MuJoCo Playground
    ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
    ppo_params.num_timesteps = num_timesteps
    ppo_params.num_evals = 10  # Log 10 eval points during training

    # Load environment
    env_cfg = registry.get_default_config(env_name)
    env = registry.load(env_name=env_name, config=env_cfg)

    # Eval environment
    eval_env = registry.load(env_name=env_name, config=env_cfg)

    # Network factory (default MuJoCo Playground config)
    network_factory = ppo_networks.make_ppo_networks

    # Collect progress data
    progress_data = []
    wall_clock_start = time.time()

    def progress_fn(num_steps, metrics):
        elapsed = time.time() - wall_clock_start
        reward = metrics.get("eval/episode_reward", 0.0)
        progress_data.append({
            "env_steps": int(num_steps),
            "reward": float(reward),
            "wall_clock_s": round(elapsed, 1),
        })
        print(f"    {num_steps:>12,} steps  reward={reward:>8.2f}  ({elapsed:.1f}s)")

    # Extract training params (remove network_factory key for train call)
    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        nf_params = training_params.pop("network_factory")
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_params)

    # Train
    print(f"\n  PPO | seed={seed} | {env_name} | {num_timesteps:,} steps")
    t0 = time.time()

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress_fn,
        seed=seed,
        network_factory=network_factory,
        **training_params,
    )

    total_time = time.time() - t0

    # Get final reward from last eval
    final_reward = progress_data[-1]["reward"] if progress_data else 0.0

    return {
        "env_name": env_name,
        "seed": seed,
        "final_reward": final_reward,
        "total_time_s": round(total_time, 1),
        "num_timesteps": num_timesteps,
        "progress": progress_data,
    }


def main():
    print(f"Device: {jax.default_backend()} | devices: {jax.devices()}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'PAPER QUALITY'}")
    print(f"Envs: {ENV_NAMES}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Timesteps: {NUM_TIMESTEPS:,}")

    all_results = {}
    total_t0 = time.time()

    for env_name in ENV_NAMES:
        all_results[env_name] = []

        for seed_idx in range(N_SEEDS):
            seed = 777 + seed_idx * 111  # Same seeds as EA benchmark

            result = run_ppo(env_name, seed, NUM_TIMESTEPS)
            all_results[env_name].append(result)

            print(f"    DONE: reward={result['final_reward']:.2f}  time={result['total_time_s']:.1f}s")

    total_time = time.time() - total_t0

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n\n{'='*70}")
    print(f"PPO BASELINE RESULTS  (total: {total_time/60:.1f} min)")
    print(f"{'='*70}")

    for env_name in ENV_NAMES:
        results = all_results[env_name]
        rewards = [r["final_reward"] for r in results]
        times = [r["total_time_s"] for r in results]
        total_steps = [r["num_timesteps"] for r in results]

        import numpy as np
        print(f"\n  {env_name}:")
        print(f"    rewards:   {rewards}")
        print(f"    median:    {np.median(rewards):.2f}")
        print(f"    mean:      {np.mean(rewards):.2f}")
        print(f"    wall-clock per run: {[f'{t:.0f}s' for t in times]}")
        print(f"    env steps: {total_steps[0]:,}")

    # ========================================================================
    # Save results as JSON for comparison plotting
    # ========================================================================
    os.makedirs("resources/ppo_baseline", exist_ok=True)
    out_path = "resources/ppo_baseline/ppo_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ Results saved to {out_path}")

    # ========================================================================
    # Comparison table (if EA results exist)
    # ========================================================================
    ea_results_path = "resources/neuroevolution/results_summary.log"
    if os.path.exists(ea_results_path):
        print(f"\n{'='*70}")
        print("EA vs PPO COMPARISON")
        print(f"{'='*70}")

        # LinePulse medians from existing results (results_summary.log)
        ea_medians = {
            "CartpoleSwingup": 697.57,
            "WalkerWalk": 298.61,
            "HumanoidWalk": 151.93,
        }
        # EA wall-clock (from run.log, approximate per env)
        # pop=512, gens=300, 5 eps × 1000 steps = 768M env steps total
        ea_env_steps = 512 * 5 * 1000 * 300  # = 768,000,000

        for env_name in ENV_NAMES:
            if env_name in ea_medians:
                ppo_rewards = [r["final_reward"] for r in all_results[env_name]]
                ppo_med = np.median(ppo_rewards)
                ea_med = ea_medians[env_name]
                ratio = ea_med / ppo_med if ppo_med > 0 else 0

                ppo_steps = all_results[env_name][0]["num_timesteps"]
                ppo_time = np.mean([r["total_time_s"] for r in all_results[env_name]])

                print(f"\n  {env_name}:")
                print(f"    LinePulse median: {ea_med:>8.2f}  ({ea_env_steps:>12,} env steps)")
                print(f"    PPO median:       {ppo_med:>8.2f}  ({ppo_steps:>12,} env steps)")
                print(f"    Ratio (EA/PPO):   {ratio:.2%}")
                print(f"    PPO uses {ea_env_steps/ppo_steps:.1f}× fewer env steps")
                print(f"    PPO wall-clock:   {ppo_time:.0f}s")


if __name__ == "__main__":
    main()

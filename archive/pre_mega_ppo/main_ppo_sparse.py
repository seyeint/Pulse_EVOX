"""
PPO Sparse Reward Experiment
==============================
Same as main_ppo_baseline.py but wraps environments to make reward SPARSE:
  - Per-step reward = 0 for all intermediate steps
  - Episode-end reward = sum of all per-step rewards

This is the "kill shot" experiment:
  - PPO should collapse because the critic can't bootstrap from zero rewards
  - LinePulse results are IDENTICAL to dense (it already sees one scalar per episode)

Usage:
    MUJOCO_GL=osmesa python3 -u main_ppo_sparse.py 2>&1 | tee resources/ppo_baseline/sparse_run.log
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
    NUM_TIMESTEPS = 1_000_000
    N_SEEDS = 1
    ENV_NAMES = ["CartpoleSwingup"]
else:
    NUM_TIMESTEPS = 60_000_000
    N_SEEDS = 3
    ENV_NAMES = [
        "CartpoleSwingup",
        "WalkerWalk",
        "HumanoidWalk",
    ]


# ============================================================================
# Sparse Reward Wrapper
# ============================================================================
class SparseRewardEnv:
    """Wrap an MjxEnv so reward is only given at episode end.

    Per-step reward becomes 0. At the terminal step (done=True),
    the accumulated sum of all per-step rewards is returned.

    This does NOT change the total episodic reward — just when it's delivered.
    EAs are unaffected (they only see the total). PPO's critic breaks because
    it can't bootstrap V(s) from zero intermediate rewards.
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        """Forward everything (dt, observation_size, action_size, etc)."""
        return getattr(self._env, name)

    def reset(self, rng):
        state = self._env.reset(rng)
        # Start accumulator at zero, store in info dict
        info = state.info if state.info is not None else {}
        info = {**info, "_accum_reward": jp.zeros_like(state.reward)}
        return state.replace(reward=jp.zeros_like(state.reward), info=info)

    def step(self, state, action):
        next_state = self._env.step(state, action)

        # Accumulate reward
        prev_accum = state.info.get("_accum_reward", jp.zeros_like(state.reward))
        accum = prev_accum + next_state.reward

        # Sparse: only emit reward when episode ends
        sparse_reward = jp.where(next_state.done, accum, jp.zeros_like(accum))

        # Reset accumulator on done (for auto-reset envs)
        new_accum = jp.where(next_state.done, jp.zeros_like(accum), accum)

        info = {**next_state.info, "_accum_reward": new_accum}
        return next_state.replace(reward=sparse_reward, info=info)


# ============================================================================
# Main
# ============================================================================
def run_ppo_sparse(env_name: str, seed: int, num_timesteps: int):
    """Run Brax PPO with sparse reward wrapper."""

    ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
    ppo_params.num_timesteps = num_timesteps
    ppo_params.num_evals = 10

    env_cfg = registry.get_default_config(env_name)

    # Load env, then wrap with sparse reward
    base_env = registry.load(env_name=env_name, config=env_cfg)
    env = SparseRewardEnv(base_env)

    eval_base_env = registry.load(env_name=env_name, config=env_cfg)
    eval_env = SparseRewardEnv(eval_base_env)

    network_factory = ppo_networks.make_ppo_networks

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

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        nf_params = training_params.pop("network_factory")
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_params)

    print(f"\n  PPO-SPARSE | seed={seed} | {env_name} | {num_timesteps:,} steps")
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
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'PAPER QUALITY'} — SPARSE REWARD")
    print(f"Envs: {ENV_NAMES}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Timesteps: {NUM_TIMESTEPS:,}")

    all_results = {}
    total_t0 = time.time()

    for env_name in ENV_NAMES:
        all_results[env_name] = []
        for seed_idx in range(N_SEEDS):
            seed = 777 + seed_idx * 111

            result = run_ppo_sparse(env_name, seed, NUM_TIMESTEPS)
            all_results[env_name].append(result)
            print(f"    DONE: reward={result['final_reward']:.2f}  time={result['total_time_s']:.1f}s")

    total_time = time.time() - total_t0

    # ========================================================================
    # Summary
    # ========================================================================
    import numpy as np

    print(f"\n\n{'='*70}")
    print(f"PPO SPARSE REWARD RESULTS  (total: {total_time/60:.1f} min)")
    print(f"{'='*70}")

    for env_name in ENV_NAMES:
        results = all_results[env_name]
        rewards = [r["final_reward"] for r in results]
        times = [r["total_time_s"] for r in results]

        print(f"\n  {env_name}:")
        print(f"    rewards:   {rewards}")
        print(f"    median:    {np.median(rewards):.2f}")
        print(f"    mean:      {np.mean(rewards):.2f}")
        print(f"    wall-clock per run: {[f'{t:.0f}s' for t in times]}")

    # ========================================================================
    # Save
    # ========================================================================
    os.makedirs("resources/ppo_baseline", exist_ok=True)
    out_path = "resources/ppo_baseline/ppo_sparse_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ Results saved to {out_path}")

    # ========================================================================
    # DENSE vs SPARSE comparison
    # ========================================================================
    dense_path = "resources/ppo_baseline/ppo_results.json"
    if os.path.exists(dense_path):
        with open(dense_path) as f:
            dense_results = json.load(f)

        # LinePulse doesn't change between dense/sparse (same total reward)
        ea_medians = {
            "CartpoleSwingup": 697.57,
            "WalkerWalk": 298.61,
            "HumanoidWalk": 151.93,
        }

        print(f"\n{'='*70}")
        print("DENSE vs SPARSE COMPARISON")
        print(f"{'='*70}")
        print(f"\n  {'Task':<20} {'PPO dense':>10} {'PPO sparse':>12} {'LinePulse':>10}  {'Winner'}")
        print(f"  {'─'*70}")

        for env_name in ENV_NAMES:
            if env_name in dense_results and env_name in ea_medians:
                ppo_dense = np.median([r["final_reward"] for r in dense_results[env_name]])
                ppo_sparse = np.median([r["final_reward"] for r in all_results[env_name]])
                ea = ea_medians[env_name]

                # Determine winner
                scores = {"PPO-dense": ppo_dense, "PPO-sparse": ppo_sparse, "LinePulse": ea}
                winner = max(scores, key=scores.get)

                print(f"  {env_name:<20} {ppo_dense:>10.1f} {ppo_sparse:>12.1f} {ea:>10.1f}  ← {winner}")

                # Show the sparse collapse
                if ppo_dense > 0:
                    collapse = (1 - ppo_sparse / ppo_dense) * 100
                    print(f"  {'':20} PPO collapsed {collapse:.0f}% from dense to sparse")


if __name__ == "__main__":
    main()

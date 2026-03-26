"""
Mega PPO Sparsity Sweep — 10 Seeds, 5 Envs, 6 K values
========================================================
Paper-quality PPO phase transition experiment.
Maps the exact degradation curve across dimensionalities.

Usage:
    MUJOCO_GL=osmesa python3 -u main_mega_ppo.py 2>&1 | tee resources/mega/ppo_sweep.log
"""

import os
import sys
import time
import json
import functools
import warnings

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "osmesa")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

import jax
import jax.numpy as jp
import numpy as np
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
    N_SEEDS = 2
    ENV_NAMES = ["CartpoleSwingup"]
    K_VALUES = [1, 50, 1000]
else:
    NUM_TIMESTEPS = 60_000_000
    N_SEEDS = 10
    ENV_NAMES = [
        "CartpoleSwingup",   # ~260s/run  × 60 = 4.3h
        "HopperHop",         # ~350s/run  × 60 = 5.8h
        "CheetahRun",        # ~350s/run  × 60 = 5.8h
        "WalkerWalk",        # ~540s/run  × 60 = 9.0h
        "HumanoidWalk",      # ~855s/run  × 60 = 14.3h
    ]                        # Total: ~39h
    K_VALUES = [1, 10, 50, 200, 500, 1000]

# EA reference scores (will be updated from mega EA benchmark)
EA_SCORES = {
    "CartpoleSwingup": {"LinePulse": 697.57, "OpenES": 157.72, "DE": 402.07, "PSO": 633.35},
    "HopperHop":       {"LinePulse": 0.0,    "OpenES": 0.0,    "DE": 0.0,    "PSO": 0.0},
    "CheetahRun":      {"LinePulse": 0.0,    "OpenES": 0.0,    "DE": 0.0,    "PSO": 0.0},
    "WalkerWalk":      {"LinePulse": 298.61, "OpenES": 53.80,  "DE": 227.22, "PSO": 304.29},
    "HumanoidWalk":    {"LinePulse": 151.93, "OpenES": 40.73,  "DE": 121.07, "PSO": 114.09},
}

# CLI override: python3 main_mega_ppo.py WalkerWalk
# Lets you run one env per pod without editing the file
if len(sys.argv) > 1:
    _valid = ["CartpoleSwingup", "HopperHop", "CheetahRun", "WalkerWalk", "HumanoidWalk"]
    _requested = [a for a in sys.argv[1:] if a in _valid]
    if _requested:
        ENV_NAMES = _requested
        print(f"[CLI] Running envs: {ENV_NAMES}")
    else:
        print(f"[CLI] Unknown env(s): {sys.argv[1:]}. Valid: {_valid}")
        sys.exit(1)


# ============================================================================
# Sparsity Wrapper
# ============================================================================
class SparsityRewardEnv:
    """Wraps env to emit accumulated reward every K steps."""

    def __init__(self, env, k: int):
        self._env = env
        self.k = k

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, rng):
        state = self._env.reset(rng)
        info = state.info if state.info is not None else {}
        info = {
            **info,
            "_accum_reward": jp.zeros_like(state.reward),
            "_step_counter": jp.zeros(state.reward.shape, dtype=jp.int32),
        }
        return state.replace(reward=jp.zeros_like(state.reward), info=info)

    def step(self, state, action):
        ns = self._env.step(state, action)

        accum = state.info["_accum_reward"] + ns.reward
        counter = state.info["_step_counter"] + 1

        release = jp.logical_or(counter % self.k == 0, ns.done > 0.5)
        sparse_reward = jp.where(release, accum, jp.zeros_like(accum))

        new_accum = jp.where(release, jp.zeros_like(accum), accum)
        new_counter = jp.where(ns.done, jp.zeros(counter.shape, dtype=jp.int32), counter)

        info = {**ns.info, "_accum_reward": new_accum, "_step_counter": new_counter}
        return ns.replace(reward=sparse_reward, info=info)


# ============================================================================
# Run PPO
# ============================================================================
def run_ppo_at_sparsity(env_name: str, k: int, seed: int, num_timesteps: int):
    """Run Brax PPO with reward emitted every K steps."""

    ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
    ppo_params.num_timesteps = num_timesteps
    ppo_params.num_evals = 10

    env_cfg = registry.get_default_config(env_name)

    base_env = registry.load(env_name=env_name, config=env_cfg)
    env = SparsityRewardEnv(base_env, k=k) if k > 1 else base_env

    eval_base_env = registry.load(env_name=env_name, config=env_cfg)
    eval_env = SparsityRewardEnv(eval_base_env, k=k) if k > 1 else eval_base_env

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
        if len(progress_data) <= 2 or num_steps >= num_timesteps * 0.9:
            print(f"      {num_steps:>12,} steps  reward={reward:>8.2f}  ({elapsed:.1f}s)")

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        nf_params = training_params.pop("network_factory")
        network_factory = functools.partial(ppo_networks.make_ppo_networks, **nf_params)

    print(f"    K={k:<5} seed={seed}  ", end="", flush=True)
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
    peak_reward = max(p["reward"] for p in progress_data) if progress_data else 0.0

    print(f"  → final={final_reward:.1f}  peak={peak_reward:.1f}  ({total_time:.0f}s)")

    return {
        "k": k,
        "seed": seed,
        "final_reward": final_reward,
        "peak_reward": peak_reward,
        "total_time_s": round(total_time, 1),
        "progress": progress_data,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Device: {jax.default_backend()} | devices: {jax.devices()}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'PAPER QUALITY'} — MEGA PPO SWEEP")
    print(f"Envs: {ENV_NAMES}")
    print(f"K values: {K_VALUES}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Timesteps: {NUM_TIMESTEPS:,}")
    total_runs = len(ENV_NAMES) * len(K_VALUES) * N_SEEDS
    print(f"Total runs: {total_runs}")

    # Probe PPO network architecture (log for paper reproducibility)
    import inspect
    sig = inspect.signature(ppo_networks.make_ppo_networks)
    print(f"\n  PPO Network Architecture (Brax defaults):")
    for name in ["policy_hidden_layer_sizes", "value_hidden_layer_sizes", "activation"]:
        p = sig.parameters.get(name)
        if p and p.default is not inspect.Parameter.empty:
            print(f"    {name} = {p.default}")
    print()

    all_env_results = {}
    total_t0 = time.time()

    for env_name in ENV_NAMES:
        print(f"\n\n{'#'*70}")
        print(f"# ENV: {env_name}")
        print(f"{'#'*70}")

        all_results = {}
        for k in K_VALUES:
            all_results[k] = []
            print(f"\n  === K={k} (reward every {k} steps) ===")

            for seed_idx in range(N_SEEDS):
                seed = 777 + seed_idx * 111
                try:
                    result = run_ppo_at_sparsity(env_name, k, seed, NUM_TIMESTEPS)
                except Exception as e:
                    print(f"    ⚠ CRASHED (NaN divergence): {e}")
                    result = {
                        "k": k, "seed": seed,
                        "final_reward": 0.0, "peak_reward": 0.0,
                        "total_time_s": 0.0, "progress": [],
                        "crashed": True,
                    }
                all_results[k].append(result)

        all_env_results[env_name] = all_results

        # Per-env summary
        ea = EA_SCORES.get(env_name, {})
        lp = ea.get("LinePulse", 0)
        print(f"\n  {'─'*65}")
        print(f"  SWEEP — {env_name}")
        print(f"  {'K':>6}  {'PPO median':>12}  {'PPO std':>10}  {'LinePulse':>10}  {'Winner'}")
        print(f"  {'─'*65}")

        for k in K_VALUES:
            results = all_results[k]
            finals = [r["final_reward"] for r in results]
            med = np.median(finals)
            std = np.std(finals)
            winner = "PPO" if med >= lp else "LinePulse"
            print(f"  {k:>6}  {med:>12.1f}  {std:>10.1f}  {lp:>10.1f}  ← {winner}")

        # Incremental save
        _save_results(all_env_results, ENV_NAMES[:ENV_NAMES.index(env_name)+1])

    total_time = time.time() - total_t0

    # Grand summary
    print(f"\n\n{'='*70}")
    print(f"MEGA PPO SWEEP — ALL ENVIRONMENTS  (total: {total_time/60:.1f} min)")
    print(f"{'='*70}")

    for env_name in ENV_NAMES:
        ea = EA_SCORES.get(env_name, {})
        lp_score = ea.get("LinePulse", 0)
        all_results = all_env_results[env_name]

        ppo_medians = [np.median([r["final_reward"] for r in all_results[k]]) for k in K_VALUES]

        crossover_k = None
        for i, k in enumerate(K_VALUES):
            if ppo_medians[i] < lp_score and (i == 0 or ppo_medians[i-1] >= lp_score):
                crossover_k = k
                if i > 0:
                    prev_k = K_VALUES[i-1]
                    frac = (lp_score - ppo_medians[i]) / (ppo_medians[i-1] - ppo_medians[i] + 1e-10)
                    crossover_k = int(prev_k + (1-frac) * (k - prev_k))
                break

        print(f"\n  {env_name}:")
        print(f"    K =       {K_VALUES}")
        print(f"    PPO =     {[round(m, 1) for m in ppo_medians]}")
        print(f"    LinePulse = {lp_score} (invariant)")
        if crossover_k:
            print(f"    ★ CROSSOVER at K ≈ {crossover_k}")
        elif all(m >= lp_score for m in ppo_medians):
            print(f"    PPO stays above LinePulse at all K values")
        else:
            print(f"    PPO below LinePulse at K={K_VALUES[0]} already")

    _save_results(all_env_results, ENV_NAMES)


def _save_results(all_env_results, env_names_done):
    os.makedirs("resources/mega", exist_ok=True)
    out_path = "resources/mega/ppo_sweep_results.json"
    save_data = {}
    for env_name in env_names_done:
        if env_name in all_env_results:
            save_data[env_name] = {
                "k_values": K_VALUES,
                "ea_scores": EA_SCORES.get(env_name, {}),
                "ppo_results": {str(k): all_env_results[env_name][k] for k in K_VALUES},
            }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  → Saved to {out_path}")


if __name__ == "__main__":
    main()

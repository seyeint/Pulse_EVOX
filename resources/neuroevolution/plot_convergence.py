#!/usr/bin/env python3
"""
Parse run.log to extract per-generation trajectories and plot convergence curves.
Saves one plot per environment showing all algorithms with seed-averaged curves.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

LOG_PATH = Path(__file__).parent / "run.log"
OUT_DIR = Path(__file__).parent / "plots"

# Colors and styles per algorithm
ALGO_STYLES = {
    "LinePulse": {"color": "#E63946", "lw": 2.5, "zorder": 10},
    "PSO":       {"color": "#457B9D", "lw": 2.0, "zorder": 8},
    "DE":        {"color": "#2A9D8F", "lw": 2.0, "zorder": 7},
    "OpenES":    {"color": "#9E9E9E", "lw": 1.5, "zorder": 5},
}


def parse_trajectories(path: str) -> dict:
    """
    Returns: {env: {algo: {seed_idx: [(gen, best), ...]}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    current_env = None
    current_algo = None
    current_seed_idx = -1

    with open(path) as f:
        for line in f:
            # Detect environment header
            env_match = re.match(r"^ENV:\s+(\S+)", line.strip())
            if env_match:
                current_env = env_match.group(1)
                continue

            # Detect algorithm + seed header
            algo_match = re.match(
                r"^\s*(OpenES|DE|PSO|LinePulse)\s*\|\s*seed\s+(\d+)/(\d+)", line
            )
            if algo_match:
                current_algo = algo_match.group(1)
                current_seed_idx = int(algo_match.group(2)) - 1
                continue

            # Detect Gen line: "    Gen   30/300  best=  516.26  (9.5s)"
            gen_match = re.match(
                r"^\s*Gen\s+(\d+)/(\d+)\s+best=\s*([\d.]+)\s+\(([\d.]+)s\)", line
            )
            if gen_match and current_env and current_algo:
                gen = int(gen_match.group(1))
                total_gens = int(gen_match.group(2))
                best = float(gen_match.group(3))
                results[current_env][current_algo][current_seed_idx].append(
                    (gen, best)
                )
                continue

    return results


def interpolate_trajectory(points, total_gens=300):
    """
    Given sparse logged points [(gen, best), ...], create a full trajectory
    by forward-filling (best is monotonically non-decreasing).
    """
    trajectory = np.zeros(total_gens)
    point_idx = 0
    current_best = points[0][1] if points else 0

    for gen in range(1, total_gens + 1):
        if point_idx < len(points) and points[point_idx][0] == gen:
            current_best = points[point_idx][1]
            point_idx += 1
        trajectory[gen - 1] = current_best

    return trajectory


def plot_convergence(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#0D1117",
        "axes.facecolor": "#0D1117",
        "text.color": "#E6EDF3",
        "axes.labelcolor": "#E6EDF3",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "axes.edgecolor": "#30363D",
        "grid.color": "#21262D",
    })

    for env, algos in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for algo_name in ["OpenES", "DE", "PSO", "LinePulse"]:
            if algo_name not in algos:
                continue

            seeds = algos[algo_name]
            style = ALGO_STYLES[algo_name]

            # Determine total gens from the data
            total_gens = max(p[0] for pts in seeds.values() for p in pts)

            # Interpolate all seeds
            all_trajectories = []
            for seed_idx, points in seeds.items():
                traj = interpolate_trajectory(points, total_gens)
                all_trajectories.append(traj)

            all_trajectories = np.array(all_trajectories)
            mean_traj = np.mean(all_trajectories, axis=0)
            min_traj = np.min(all_trajectories, axis=0)
            max_traj = np.max(all_trajectories, axis=0)

            gens = np.arange(1, total_gens + 1)

            # Plot mean + min/max band
            ax.plot(
                gens, mean_traj,
                label=algo_name,
                color=style["color"],
                linewidth=style["lw"],
                zorder=style["zorder"],
            )
            ax.fill_between(
                gens, min_traj, max_traj,
                alpha=0.15,
                color=style["color"],
                zorder=style["zorder"] - 1,
            )

        ax.set_xlabel("Generation", fontsize=13, fontweight="bold")
        ax.set_ylabel("Best Fitness (↑)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"{env} — Convergence",
            fontsize=16, fontweight="bold", pad=15
        )
        ax.legend(
            loc="lower right", fontsize=11,
            frameon=True, facecolor="#161B22", edgecolor="#30363D",
            labelcolor="#E6EDF3",
        )
        ax.grid(True, alpha=0.3)

        # Save
        out_path = out_dir / f"convergence_{env}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── Combined plot (all envs in subplots) ──
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, (env, algos) in zip(axes, results.items()):
        for algo_name in ["OpenES", "DE", "PSO", "LinePulse"]:
            if algo_name not in algos:
                continue
            seeds = algos[algo_name]
            style = ALGO_STYLES[algo_name]
            total_gens = max(p[0] for pts in seeds.values() for p in pts)

            all_trajectories = []
            for seed_idx, points in seeds.items():
                traj = interpolate_trajectory(points, total_gens)
                all_trajectories.append(traj)

            all_trajectories = np.array(all_trajectories)
            mean_traj = np.mean(all_trajectories, axis=0)
            min_traj = np.min(all_trajectories, axis=0)
            max_traj = np.max(all_trajectories, axis=0)
            gens = np.arange(1, total_gens + 1)

            ax.plot(gens, mean_traj, label=algo_name,
                    color=style["color"], linewidth=style["lw"],
                    zorder=style["zorder"])
            ax.fill_between(gens, min_traj, max_traj,
                            alpha=0.15, color=style["color"],
                            zorder=style["zorder"] - 1)

        ax.set_xlabel("Generation", fontsize=11, fontweight="bold")
        ax.set_ylabel("Best Fitness (↑)", fontsize=11, fontweight="bold")
        ax.set_title(env, fontsize=14, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=9,
                  frameon=True, facecolor="#161B22", edgecolor="#30363D",
                  labelcolor="#E6EDF3")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Neuroevolution Benchmark — Convergence",
        fontsize=18, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    out_path = out_dir / "convergence_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Parsing trajectories from run.log...")
    results = parse_trajectories(LOG_PATH)

    print(f"\nFound {len(results)} environments:")
    for env, algos in results.items():
        print(f"  {env}: {list(algos.keys())}")

    print("\nGenerating convergence plots...")
    plot_convergence(results, OUT_DIR)
    print("\nDone!")

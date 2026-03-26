"""
Plot Sparsity Sweep — Phase Transition Graphs
===============================================
Creates publication-quality plots showing PPO degradation vs EA invariance
as reward sparsity increases.

Usage:
    python3 resources/ppo_baseline/plot_sweep.py
"""

import json
import re
import os
import sys

# Try matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("Need matplotlib: pip install matplotlib numpy")
    sys.exit(1)

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE, "sweep_run.log")
JSON_PATH = os.path.join(BASE, "sweep_results.json")
PLOT_DIR = os.path.join(BASE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

K_VALUES = [1, 10, 50, 200, 500, 1000]

EA_SCORES = {
    "CartpoleSwingup": {"LinePulse": 697.57, "OpenES": 157.72, "DE": 402.07, "PSO": 633.35},
    "WalkerWalk":      {"LinePulse": 298.61, "OpenES": 53.80,  "DE": 227.22, "PSO": 304.29},
    "HumanoidWalk":    {"LinePulse": 151.93, "OpenES": 40.73,  "DE": 121.07, "PSO": 114.09},
}

# ============================================================================
# Parse data
# ============================================================================
def parse_log_results(log_path):
    """Parse sweep_run.log for per-seed final rewards."""
    with open(log_path) as f:
        log = f.read()

    results = {}
    current_env = None
    current_k = None

    for line in log.split('\n'):
        m = re.match(r'# ENV: (\w+)', line.strip())
        if m:
            current_env = m.group(1)
            results[current_env] = {}
            continue

        m = re.match(r'\s*=== K=(\d+)', line)
        if m and current_env:
            current_k = int(m.group(1))
            if current_k not in results[current_env]:
                results[current_env][current_k] = []
            continue

        m = re.match(r'\s*→ final=([\d.]+)', line)
        if m and current_env and current_k is not None:
            results[current_env][current_k].append(float(m.group(1)))

    return results


def parse_json_results(json_path):
    """Parse sweep_results.json for HumanoidWalk data."""
    with open(json_path) as f:
        data = json.load(f)

    results = {}
    for env_name, env_data in data.items():
        if "ppo_results" not in env_data:
            continue
        results[env_name] = {}
        for k_str, runs in env_data["ppo_results"].items():
            k = int(k_str)
            results[env_name][k] = [r["final_reward"] for r in runs]

    return results


# ============================================================================
# Plotting
# ============================================================================
# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors
C_PPO = '#E63946'       # Red
C_LINEPULSE = '#2A9D8F' # Teal
C_OPENES = '#E9C46A'    # Gold
C_DE = '#264653'        # Dark blue
C_PSO = '#F4A261'       # Orange
C_CROSS = '#6C757D'     # Gray


def plot_single_env(ax, env_name, ppo_data, ea_scores, k_values):
    """Plot one environment's sparsity sweep."""
    lp = ea_scores["LinePulse"]
    oes = ea_scores["OpenES"]

    # PPO data
    ppo_medians = []
    ppo_mins = []
    ppo_maxs = []

    for k in k_values:
        if k in ppo_data and len(ppo_data[k]) > 0:
            vals = [v for v in ppo_data[k] if v is not None]
            if vals:
                ppo_medians.append(np.median(vals))
                ppo_mins.append(min(vals))
                ppo_maxs.append(max(vals))
            else:
                ppo_medians.append(np.nan)
                ppo_mins.append(np.nan)
                ppo_maxs.append(np.nan)
        else:
            ppo_medians.append(np.nan)
            ppo_mins.append(np.nan)
            ppo_maxs.append(np.nan)

    ppo_medians = np.array(ppo_medians)
    ppo_mins = np.array(ppo_mins)
    ppo_maxs = np.array(ppo_maxs)

    # PPO curve with error band
    valid = ~np.isnan(ppo_medians)
    ks = np.array(k_values)

    ax.fill_between(ks[valid], ppo_mins[valid], ppo_maxs[valid],
                     alpha=0.15, color=C_PPO, linewidth=0)
    ax.plot(ks[valid], ppo_medians[valid], 'o-', color=C_PPO,
            linewidth=2.5, markersize=7, label='PPO', zorder=5)

    # EA horizontal lines
    ax.axhline(y=lp, color=C_LINEPULSE, linewidth=2.5, linestyle='--',
               label=f'LinePulse ({lp:.0f})', zorder=4)
    ax.axhline(y=oes, color=C_OPENES, linewidth=1.5, linestyle=':',
               label=f'OpenES ({oes:.0f})', alpha=0.7, zorder=3)

    # Find and mark crossover
    for i in range(len(k_values) - 1):
        if valid[i] and valid[i+1]:
            if ppo_medians[i] >= lp and ppo_medians[i+1] < lp:
                # Interpolate crossover K
                frac = (lp - ppo_medians[i+1]) / (ppo_medians[i] - ppo_medians[i+1] + 1e-10)
                cross_k = k_values[i] + (1 - frac) * (k_values[i+1] - k_values[i])

                ax.axvline(x=cross_k, color=C_CROSS, linewidth=1.5,
                          linestyle='-.', alpha=0.6)
                ax.annotate(f'K ≈ {int(cross_k)}',
                           xy=(cross_k, lp), xytext=(cross_k * 1.5, lp * 1.15),
                           fontsize=9, fontweight='bold', color=C_CROSS,
                           arrowprops=dict(arrowstyle='->', color=C_CROSS, lw=1.2))
                break

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Reward Interval K (steps)')
    ax.set_ylabel('Final Reward')
    ax.set_title(env_name, fontweight='bold')
    ax.set_xticks(k_values)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(0.7, 1500)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.15, which='both')

    # Shade the "LinePulse wins" region
    ax.axhspan(0, lp, alpha=0.03, color=C_LINEPULSE)


def main():
    # Load data
    log_results = parse_log_results(LOG_PATH) if os.path.exists(LOG_PATH) else {}
    json_results = parse_json_results(JSON_PATH) if os.path.exists(JSON_PATH) else {}

    # Merge: log has CartpoleSwingup + WalkerWalk, JSON has HumanoidWalk
    all_data = {}
    for env in log_results:
        all_data[env] = log_results[env]
    for env in json_results:
        if env not in all_data:
            all_data[env] = json_results[env]

    envs_to_plot = [e for e in ["CartpoleSwingup", "WalkerWalk", "HumanoidWalk"] if e in all_data]

    if not envs_to_plot:
        print("No data found!")
        return

    # ========================================================================
    # Individual plots
    # ========================================================================
    for env_name in envs_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_single_env(ax, env_name, all_data[env_name], EA_SCORES[env_name], K_VALUES)
        path = os.path.join(PLOT_DIR, f"sweep_{env_name}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ========================================================================
    # Combined 3-panel figure
    # ========================================================================
    if len(envs_to_plot) >= 2:
        n = len(envs_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
        if n == 1:
            axes = [axes]

        for ax, env_name in zip(axes, envs_to_plot):
            plot_single_env(ax, env_name, all_data[env_name], EA_SCORES[env_name], K_VALUES)

        fig.suptitle('PPO Phase Transition under Reward Sparsity',
                     fontsize=16, fontweight='bold', y=1.02)
        fig.text(0.5, -0.02,
                 'EA scores (LinePulse, OpenES) are invariant to K — they only see total episodic reward.',
                 ha='center', fontsize=10, style='italic', color='#666')

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "sweep_all.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

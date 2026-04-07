"""
LinePulse Paper Figures — Publication Quality
==============================================
Run: python3 plot_figures.py
Outputs to resources/mega/figures/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

MEGA_DIR = Path("resources/mega")
FIG_DIR = MEGA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Colors
C_LP    = "#2563EB"   # blue
C_NORAY = "#7C3AED"   # purple
C_DE    = "#059669"   # green
C_PSO   = "#D97706"   # amber
C_OES   = "#DC2626"   # red
C_PPO   = "#F59E0B"   # gold

ALGO_COLORS = {
    "LinePulse": C_LP, "LinePulse-NoRay": C_NORAY,
    "DE": C_DE, "PSO": C_PSO, "OpenES": C_OES
}

# ── Load all data ──
def load_ea():
    all_env = {}
    for f in sorted(MEGA_DIR.glob("ea_results_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for env_key, env_data in data.items():
            all_env[env_key] = env_data
    return all_env

def load_ppo():
    all_ppo = {}
    for f in sorted(MEGA_DIR.glob("ppo_results_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        for env_key, env_data in data.items():
            all_ppo[env_key] = env_data
    return all_ppo

def load_capacity():
    """Merge capacity sweep results from multiple sources.
    - capacity_sweep_results_5090.json: d=100K (from 5090 run)
    - capacity_sweep_results.json: d=385 to d=25K (from 4090 run)
    """
    merged = {}
    for fname in ["capacity_sweep_results_5090.json", "capacity_sweep_results.json"]:
        f = MEGA_DIR / fname
        if f.exists():
            with open(f) as fh:
                data = json.load(fh)
            for arch_key, arch_data in data.items():
                if arch_key not in merged:
                    merged[arch_key] = arch_data
                else:
                    # Merge algo data from both sources
                    for k, v in arch_data.items():
                        if k not in merged[arch_key]:
                            merged[arch_key][k] = v
    return merged

def get_finals(algo_data):
    return [sd["final"] for sk, sd in algo_data.items()
            if isinstance(sd, dict) and "final" in sd]

def get_trajectories(algo_data):
    trajs = []
    for sk, sd in algo_data.items():
        if isinstance(sd, dict) and "trajectory" in sd:
            trajs.append(sd["trajectory"])
    return trajs


# ════════════════════════════════════════════════════════════════════
# FIGURE 1: THE VISE GRIP (side-by-side)
# Left: PPO sparsity collapse | Right: Capacity sweep SNR collapse
# ════════════════════════════════════════════════════════════════════
def fig1_vise_grip():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ppo_data = load_ppo()
    ea_data = load_ea()

    # ── LEFT: PPO Sparsity Sweep ──
    envs_order = ["CartpoleSwingup", "CheetahRun", "HopperHop", "HumanoidWalk", "WalkerWalk"]
    env_colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    for i, env in enumerate(envs_order):
        if env not in ppo_data:
            continue
        env_d = ppo_data[env]
        k_values = env_d.get("k_values", [])
        ppo_results = env_d.get("ppo_results", {})

        # Get real LP score
        lp_score = 0
        if env in ea_data and "LinePulse" in ea_data[env]:
            lp_score = np.median(get_finals(ea_data[env]["LinePulse"]))

        ppo_meds = []
        for k in k_values:
            if str(k) in ppo_results:
                finals = [r["final_reward"] for r in ppo_results[str(k)]]
                ppo_meds.append(np.median(finals))
            else:
                ppo_meds.append(0)

        # Normalize to LP score for comparable scale
        if lp_score > 0:
            ppo_norm = [p / lp_score for p in ppo_meds]
            ax1.plot(k_values, ppo_norm, 'o-', color=env_colors[i], linewidth=2,
                     markersize=6, label=f"{env}", zorder=3)

    ax1.axhline(y=1.0, color=C_LP, linewidth=2.5, linestyle='--',
                label="LinePulse (invariant)", zorder=2)
    ax1.set_xscale('log')
    ax1.set_xlabel("Reward Sparsity K (steps per reward)")
    ax1.set_ylabel("Score / LinePulse Score")
    ax1.set_title("(a) PPO Collapse Under Reward Sparsity")
    ax1.set_ylim(-0.3, 4.5)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.axhspan(-0.3, 0.05, color='#FEE2E2', alpha=0.5, zorder=0)
    ax1.text(500, -0.15, "Total Failure Zone", ha='center', fontsize=8, color='#991B1B')

    # ── RIGHT: Capacity Sweep (all 4 algorithms) ──
    cap_data = load_capacity()
    algos_cap = ["LinePulse", "OpenES", "DE", "PSO"]
    markers = {"LinePulse": "s", "OpenES": "o", "DE": "D", "PSO": "^"}

    archs = sorted(cap_data.keys(), key=lambda x: cap_data[x].get("d", 0))
    ds = [cap_data[a].get("d", 0) for a in archs]
    snrs = [cap_data[a].get("snr", 0) for a in archs]

    for algo in algos_cap:
        meds, stds = [], []
        for arch in archs:
            scores = [sd["final"] for sk, sd in cap_data[arch].get(algo, {}).items()
                      if isinstance(sd, dict) and "final" in sd
                      and not sd.get("crashed", False)]
            meds.append(np.median(scores) if scores else 0)
            stds.append(np.std(scores) if scores else 0)
        ax2.errorbar(ds, meds, yerr=stds, fmt=f'{markers[algo]}-',
                     color=ALGO_COLORS[algo], linewidth=2.5, markersize=8,
                     capsize=4, label=algo, zorder=3)

    ax2.set_xscale('log')
    ax2.set_xlabel("Parameter Count d")
    ax2.set_ylabel("Median Score (10 seeds)")
    ax2.set_title("(b) Topology Scaling: All Algorithms")
    ax2.legend(loc='lower left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Add SNR labels on top
    for d, snr in zip(ds, snrs):
        ax2.annotate(f"SNR={snr:.2f}", (d, 870), fontsize=7, ha='center',
                     color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_vise_grip.png")
    plt.savefig(FIG_DIR / "fig1_vise_grip.pdf")
    print(f"  ✓ Figure 1 saved")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# FIGURE 2: MEGA EA BAR CHART
# ════════════════════════════════════════════════════════════════════
def fig2_mega_ea():
    ea_data = load_ea()
    envs = ["CartpoleSwingup", "CheetahRun", "WalkerWalk", "HumanoidWalk", "HopperHop"]
    algos = ["LinePulse", "LinePulse-NoRay", "DE", "PSO", "OpenES"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)

    for i, env in enumerate(envs):
        ax = axes[i]
        if env not in ea_data:
            continue

        meds, stds, colors = [], [], []
        labels = []
        for algo in algos:
            if algo in ea_data[env]:
                finals = get_finals(ea_data[env][algo])
                meds.append(np.median(finals))
                stds.append(np.std(finals))
                colors.append(ALGO_COLORS[algo])
                labels.append(algo.replace("LinePulse-", "LP-\n"))
            else:
                meds.append(0)
                stds.append(0)
                colors.append("gray")
                labels.append(algo)

        x = np.arange(len(algos))
        bars = ax.bar(x, meds, yerr=stds, color=colors, capsize=3,
                      edgecolor='white', linewidth=0.5, alpha=0.85)

        # Highlight winner
        best_idx = np.argmax(meds)
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)

        ax.set_title(env, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(["LP", "NoRay", "DE", "PSO", "OES"], fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Add d and SNR
        d = {"CartpoleSwingup": 3393, "CheetahRun": 3942, "WalkerWalk": 4166,
             "HumanoidWalk": 6037, "HopperHop": 3812}
        snr = np.sqrt(512 / d[env])
        ax.text(0.5, 0.95, f"d={d[env]:,}  SNR={snr:.2f}",
                transform=ax.transAxes, ha='center', va='top', fontsize=7, color='gray')

    fig.suptitle("Experiment 1: Algorithm Comparison (Median ± Std, 10 Seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_mega_ea.png")
    plt.savefig(FIG_DIR / "fig2_mega_ea.pdf")
    print(f"  ✓ Figure 2 saved")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# FIGURE 3: LEARNING CURVES (Trajectories)
# ════════════════════════════════════════════════════════════════════
def fig3_curves():
    ea_data = load_ea()
    envs = ["CartpoleSwingup", "CheetahRun", "WalkerWalk", "HumanoidWalk", "HopperHop"]
    algos = ["LinePulse", "OpenES", "DE", "PSO"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for i, env in enumerate(envs):
        ax = axes[i]
        if env not in ea_data:
            continue

        for algo in algos:
            if algo not in ea_data[env]:
                continue
            trajs = get_trajectories(ea_data[env][algo])
            if not trajs:
                continue

            # Pad to same length
            max_len = max(len(t) for t in trajs)
            padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in trajs])

            med = np.median(padded, axis=0)
            q25 = np.percentile(padded, 25, axis=0)
            q75 = np.percentile(padded, 75, axis=0)

            gens = np.arange(1, len(med) + 1)
            ax.plot(gens, med, linewidth=2, color=ALGO_COLORS[algo], label=algo)
            ax.fill_between(gens, q25, q75, alpha=0.15, color=ALGO_COLORS[algo])

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title(env, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_learning_curves.png")
    plt.savefig(FIG_DIR / "fig3_learning_curves.pdf")
    print(f"  ✓ Figure 3 saved")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# FIGURE 4: ABLATION — Ray Impact
# ════════════════════════════════════════════════════════════════════
def fig4_ablation():
    ea_data = load_ea()
    envs = ["CartpoleSwingup", "CheetahRun", "WalkerWalk", "HumanoidWalk", "HopperHop"]

    fig, ax = plt.subplots(figsize=(8, 5))

    lp_vals, noray_vals = [], []
    env_labels = []
    for env in envs:
        if env not in ea_data:
            continue
        lp = np.median(get_finals(ea_data[env]["LinePulse"]))
        nr = np.median(get_finals(ea_data[env]["LinePulse-NoRay"]))
        lp_vals.append(lp)
        noray_vals.append(nr)
        env_labels.append(env)

    x = np.arange(len(env_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, lp_vals, w, label="LinePulse (with Ray)", color=C_LP, alpha=0.85)
    bars2 = ax.bar(x + w/2, noray_vals, w, label="LinePulse-NoRay", color=C_NORAY, alpha=0.85)

    # Add percentage labels
    for i in range(len(env_labels)):
        diff = (lp_vals[i] - noray_vals[i]) / noray_vals[i] * 100
        color = '#059669' if diff > 0 else '#DC2626'
        y = max(lp_vals[i], noray_vals[i]) + 10
        ax.text(x[i], y, f"{diff:+.0f}%", ha='center', fontsize=9,
                fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, fontsize=9)
    ax.set_ylabel("Median Score (10 seeds)")
    ax.set_title("Ablation: Extension Ray Impact")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_ablation.png")
    plt.savefig(FIG_DIR / "fig4_ablation.pdf")
    print(f"  ✓ Figure 4 saved")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# FIGURE 5: PPO vs LinePulse per environment
# ════════════════════════════════════════════════════════════════════
def fig5_ppo_detail():
    ppo_data = load_ppo()
    ea_data = load_ea()
    envs = ["CartpoleSwingup", "CheetahRun", "HopperHop", "HumanoidWalk", "WalkerWalk"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)

    for i, env in enumerate(envs):
        ax = axes[i]
        if env not in ppo_data:
            continue

        env_d = ppo_data[env]
        k_values = env_d.get("k_values", [])
        ppo_results = env_d.get("ppo_results", {})

        lp_score = 0
        if env in ea_data and "LinePulse" in ea_data[env]:
            lp_score = np.median(get_finals(ea_data[env]["LinePulse"]))

        ppo_meds, ppo_stds = [], []
        for k in k_values:
            if str(k) in ppo_results:
                finals = [r["final_reward"] for r in ppo_results[str(k)]]
                ppo_meds.append(np.median(finals))
                ppo_stds.append(np.std(finals))

        ax.bar(range(len(k_values)), ppo_meds, yerr=ppo_stds,
               color=C_PPO, alpha=0.8, capsize=3, label="PPO")
        ax.axhline(y=lp_score, color=C_LP, linewidth=2.5, linestyle='--',
                   label=f"LP={lp_score:.0f}")

        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([str(k) for k in k_values], fontsize=8)
        ax.set_xlabel("K")
        ax.set_title(env, fontsize=10, fontweight='bold')
        if i == 0:
            ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Experiment 2: PPO Performance Under Increasing Reward Sparsity", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_ppo_detail.png")
    plt.savefig(FIG_DIR / "fig5_ppo_detail.pdf")
    print(f"  ✓ Figure 5 saved")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# FIGURE 6: TOPOLOGY CURSE — Standalone capacity sweep
# ════════════════════════════════════════════════════════════════════
def fig6_topology_curse():
    cap_data = load_capacity()
    algos = ["LinePulse", "OpenES", "DE", "PSO"]
    markers = {"LinePulse": "s", "OpenES": "o", "DE": "D", "PSO": "^"}

    archs = sorted(cap_data.keys(), key=lambda x: cap_data[x].get("d", 0))
    ds = [cap_data[a].get("d", 0) for a in archs]
    snrs = [cap_data[a].get("snr", 0) for a in archs]

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in algos:
        meds, stds, q25s, q75s = [], [], [], []
        for arch in archs:
            scores = [sd["final"] for sk, sd in cap_data[arch].get(algo, {}).items()
                      if isinstance(sd, dict) and "final" in sd
                      and not sd.get("crashed", False)]
            if scores:
                meds.append(np.median(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))
            else:
                meds.append(0)
                q25s.append(0)
                q75s.append(0)

        ax.plot(ds, meds, f'{markers[algo]}-', color=ALGO_COLORS[algo],
                linewidth=2.5, markersize=10, label=algo, zorder=3)
        ax.fill_between(ds, q25s, q75s, alpha=0.15, color=ALGO_COLORS[algo])

    # SNR annotations
    for d, snr in zip(ds, snrs):
        ax.annotate(f"SNR={snr:.2f}", (d, ax.get_ylim()[1] * 0.95),
                    fontsize=8, ha='center', color='gray', style='italic')

    ax.set_xscale('log')
    ax.set_xlabel("Parameter Count d", fontsize=13)
    ax.set_ylabel("Median Score (10 seeds, IQR shaded)", fontsize=13)
    ax.set_title("Topology Scaling: Algorithm Performance vs Network Capacity",
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_topology_curse.png")
    plt.savefig(FIG_DIR / "fig6_topology_curse.pdf")
    print(f"  ✓ Figure 6 saved")
    plt.close()


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating paper figures...")
    print(f"Output: {FIG_DIR.resolve()}\n")

    fig1_vise_grip()
    fig2_mega_ea()
    fig3_curves()
    fig4_ablation()
    fig5_ppo_detail()
    fig6_topology_curse()

    print(f"\n✓ All figures saved to {FIG_DIR}/")
    print("  fig1_vise_grip.png/pdf   — Main figure (PPO + capacity collapse)")
    print("  fig2_mega_ea.png/pdf     — Algorithm comparison bar chart")
    print("  fig3_learning_curves.png/pdf — Training trajectories")
    print("  fig4_ablation.png/pdf    — Ray ablation")
    print("  fig5_ppo_detail.png/pdf  — PPO sparsity per environment")
    print("  fig6_topology_curse.png/pdf — Topology scaling (4 algos)")

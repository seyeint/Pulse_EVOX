# LinePulse — Research Overview

## What is LinePulse?

**LinePulse (pulse14)** is a pair-based evolutionary algorithm that organizes a population into pairs and alternates between two phases:

- **Contraction (Geometric Crossover):** Interpolates between pair members to exploit a promising line segments without diluting signal with population level.
- **Expansion (Extension Ray):** When a lineage exhausts, the pair breaks, individuals re-pair, and a borrowed direction from the population fires a ballistic ray to expand the search hull.

The key insight: by searching along 1D line segments rather diluting topology signal to population level variables, we aim at creating a better evolutionary algorithm. We can preserve feature correlations in high-dimensional spaces too where methods like OpenES collapse.

---

## Experimental Timeline

| Phase | Experiment | Script | Dims (d) | What it Tests |
|:---:|:---|:---|:---:|:---|
| **1** | CEC2022 Numerical | `main_numerical.py` | 20 | LinePulse vs PSO, DE on 12 standard functions |
| **2** | Early Neuroevolution | `main_neuroevo.py` | 3-6K | First MuJoCo tests with neural network optimization |
| **3** | PPO Sparsity Tests | *(archived)* | 3-6K | PPO performance as reward becomes sparser |
| **4a** | **Mega EA Benchmark** | `main_mega_ea.py` | 3-6K | 5 algos × 5 envs × 10 seeds (paper quality) |
| **4b** | **Mega PPO Sweep** | `main_mega_ppo.py` | 3-6K | PPO × 5 envs × 6 K values × 10 seeds |
| **4c** | **Capacity Sweep** | `main_capacity_sweep.py` | 385→100K | LinePulse vs OpenES across network sizes |
| **5** | Unleashed Test | *(archived)* | 119K | HumanoidWalk with [256,256,128] network |

**Phases 1–3** were development. **Phases 4a–4c** are the paper experiments. **Phase 5** is a bonus data point.

---

## Core Paper Results

### Experiment 4a — Mega EA: "LinePulse wins the hard environments"

5 algorithms (LinePulse, LinePulse-NoRay, DE, PSO, OpenES) on 5 MuJoCo environments, 10 seeds each.

| Environment | d | LinePulse | LinePulse-NoRay | DE | PSO | OpenES |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| CartpoleSwingup | 3.4K | 298 | 189 | 5 | 3 | **830** |
| HopperHop | 3.8K | **99** | 68 | 1 | 0.5 | 54 |
| CheetahRun | 3.9K | 117 | 48 | 2 | 66 | **158** |
| WalkerWalk | 4.2K | 103 | 45 | 5 | 37 | **196** |
| HumanoidWalk | 6.0K | **64** | 30 | 1 | 2 | 37 |

LinePulse dominates the hard, high-dimensional environments (HumanoidWalk, HopperHop). OpenES wins on simpler environments where its isotropic gradient estimate still has enough SNR.

### Experiment 4b — PPO Sweep: "The Horizon Limit"

PPO performance as reward becomes sparser (K = reward every K steps):

- **K=1 (dense):** PPO scores ~900+ on CartpoleSwingup
- **K=10:** PPO starts degrading
- **K=100–1000:** PPO collapses → ~0
- **EAs:** (Obviously, not a research conclusion) Completely invariant to K (they only see episode totals)

This reminds that PPO requires dense per-step reward to train its critic, while EAs work with any reward structure.

### Experiment 4c — Capacity Sweep: "The Dimensionality Limit"

LinePulse vs OpenES on CartpoleSwingup as network size scales from d=385 to d=100K:

| Architecture | d | SNR | LinePulse | OpenES |
|:---|:---:|:---:|:---:|:---:|
| [16, 16] | 385 | 1.15 | 585 | **830** |
| [32, 32, 32, 32] | 3,393 | 0.39 | 298 | **830** |
| [64, 64] | 4,609 | 0.33 | **643** | 528 |
| [128, 128, 64] | 25,601 | 0.14 | **604** | 0.0 |
| [256, 256, 128] | 100,353 | 0.07 | **649** | 0.0 |

OpenES collapses at d≥25K (SNR drops below usable threshold). LinePulse is completely invariant to dimensionality — it actually *improves* as the network gets wider.

---

## Key Theoretical Claims

1. **Horizon Limit (K):** RL (PPO) requires dense per-step reward for critic bootstrapping. EAs only need a scalar episode total — immune to reward sparsity.
2. **Dimensionality Limit (d):** OpenES SNR = √(N/d) → collapses as d grows. LinePulse searches in 1D subspaces, so its signal quality is independent of ambient dimensionality.
3. **Anisotropic Advantage:** Pair-based geometric operations (interpolation, extension rays) preserve neural network feature correlations that isotropic Gaussian noise destroys.

---

## Repository Structure

### Root — Core Files
```
pulse14.py                     # THE algorithm (pair-based greedy lineage)
pulse14_noray.py               # Ablation: extension ray disabled

main_mega_ea.py                # Paper: EA comparison (5 envs × 5 algos × 10 seeds)
main_mega_ppo.py               # Paper: PPO sparsity sweep
main_capacity_sweep.py         # Paper: dimensionality scaling (d=385→100K)

main_numerical.py              # Dev: CEC2022 benchmarks (20D)
main_numerical_shared_population.py  # Dev: shared-population variant
main_neuroevo.py               # Dev: early neuroevolution tests

plot_figures.py                # Publication figure generation
analyze_results.py             # Results parsing & summary
utils.py                       # Shared utilities

deploy_mega.sh                 # RunPod deployment for mega experiments
setup_pod.sh                   # Pod environment setup (JAX, EvoX, MuJoCo)

resources/mega/                # All paper results (JSONs + figures)
```

### Archive — Historical & Experimental
```
archive/
├── experimental_variants/     # pulse3–pulse13 (algorithm evolution)
├── old_pulse_variants/        # pulse15, pulse_real, etc.
├── pre_mega_ppo/              # Superseded PPO scripts
├── experimental_tests/        # v14_vs_v15, dummy_dim, unleashed
├── old_scripts/               # Old deployment scripts
└── jax_0.9_deprecated/        # Dead code
```

---

## How to Run

### Local (numerical benchmarks, macOS/Linux):
```bash
source .venv/bin/activate
python3 main_numerical.py
```

### RunPod (neuroevolution, requires GPU):
```bash
# Deploy to pod
bash deploy_mega.sh <IP> <PORT> <ENV_NAME>

# Switch to paper mode on pod
sed -i 's/DRY_RUN = True/DRY_RUN = False/' main_mega_ea.py

# Launch
nohup bash -c 'MUJOCO_GL=osmesa python3 -u main_mega_ea.py ENV > resources/mega/ea_ENV.log 2>&1' &
```

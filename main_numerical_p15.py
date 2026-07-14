# A/B of Pulse15 v1/v2 (directed contract/extend/evict automata) against the
# canonical shared-population lineup (PSO, DE, Pulse14) on CEC2022.
# Same protocol as main_numerical_shared_population.py: per (seed, function)
# cell, one shared starting population for all algorithms; 777 iterations;
# eager on CPU. Elite checkpoints at G in {50,100,200,400,777} to resolve the
# sprint-vs-marathon profile. Writes p15v2_ab_results.jsonl (checkpointed,
# resumable); does NOT touch resources/numerical artifacts.

import json
import os
import time

import numpy as np
import torch
from evox import algorithms
from evox.problems.numerical import CEC2022
from evox.workflows import EvalMonitor, StdWorkflow

from pulse14 import PulseGreedy
from pulse15 import PulseDirected
from pulse15v2 import PulseDirectedV2
from pulse15m import PulseMixed
import utils

n_dims = 20
lb, ub = -100, 100
lb_t = torch.full(size=(n_dims,), fill_value=float(lb))
ub_t = torch.full(size=(n_dims,), fill_value=float(ub))

N_SEEDS = 10
N_ITER = 777
CHECKS = [50, 100, 200, 400, 777]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p15v2_ab_results.jsonl")

problem_set = [CEC2022(x, n_dims) for x in range(1, 13)]
utils.FUNCTION_NAMES = [f"f{i+1}" for i in range(len(problem_set))]

algo_factories = {
    "PSO":   lambda: algorithms.PSO(lb=lb_t, ub=ub_t, pop_size=400),
    "DE":    lambda: algorithms.DE(lb=lb_t, ub=ub_t, pop_size=400),
    "P14":   lambda: PulseGreedy(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P15":   lambda: PulseDirected(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P15v2": lambda: PulseDirectedV2(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P15m":  lambda: PulseMixed(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
}

done = set()
if os.path.exists(OUT):
    with open(OUT) as fh:
        for line in fh:
            r = json.loads(line)
            done.add((r["seed"], r["func"], r["algo"]))
    print(f"resuming: {len(done)} runs already saved")


def emit(row):
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")


t0 = time.time()
for x in range(N_SEEDS):
    for j, function in enumerate(problem_set):
        cell = [(x, j + 1, name) for name in algo_factories]
        if all(c in done for c in cell):
            continue
        torch.manual_seed(1000 * (x + 1) + j)   # canonical seeding formula
        shared_population = torch.rand(400, n_dims) * (ub - lb) + lb
        algo_list = [f() for f in algo_factories.values()]
        for algo in algo_list:
            if hasattr(algo, "population"):
                algo.population = shared_population.clone()
            elif hasattr(algo, "pop"):
                algo.pop = shared_population.clone()

        for name, algo in zip(algo_factories.keys(), algo_list):
            if (x, j + 1, name) in done:
                continue
            monitor = EvalMonitor()
            wf = StdWorkflow(algo, function, monitor)
            elite = float("inf")
            wf.init_step()
            elite = min(elite, float(monitor.topk_fitness))
            traj = {}
            for g in range(1, N_ITER + 1):
                wf.step()
                elite = min(elite, float(monitor.topk_fitness))
                if g in CHECKS:
                    traj[str(g)] = elite
            row = {"seed": x, "func": j + 1, "algo": name, "final": elite, "traj": traj}
            if name in ("P15", "P15v2"):
                tot = max(algo.n_contract_gens + algo.n_extend_gens + algo.n_ray_gens, 1)
                row["extend_frac"] = algo.n_extend_gens / tot
                row["ray_frac"] = algo.n_ray_gens / tot
            emit(row)
        print(f"[{(time.time()-t0)/60:5.1f}m] seed {x+1}/{N_SEEDS} f{j+1} done", flush=True)

# ---------------- summary ----------------
rows = [json.loads(l) for l in open(OUT)]
names = list(algo_factories.keys())


def med_at(func, algo, g):
    v = [r["traj"].get(str(g), r["final"]) for r in rows
         if r["func"] == func and r["algo"] == algo]
    return float(np.median(v)) if v else float("nan")


print("\n" + "=" * 96)
print(f"AVERAGE RANK vs GENERATION BUDGET ({N_SEEDS} seeds, shared population)")
print("=" * 96)
print(f"{'algo':>7}" + "".join(f"{'G=' + str(g):>10}" for g in CHECKS))
rank_by_g = {}
for g in CHECKS:
    rk = {}
    for f in range(1, 13):
        meds = {n: med_at(f, n, g) for n in names}
        for pos, n in enumerate(sorted(meds, key=meds.get), 1):
            rk.setdefault(n, []).append(pos)
    rank_by_g[g] = {n: float(np.mean(v)) for n, v in rk.items()}
for n in names:
    print(f"{n:>7}" + "".join(f"{rank_by_g[g][n]:>10.2f}" for g in CHECKS))

print("\nMEDIAN FINAL FITNESS (G=777)")
for f in range(1, 13):
    meds = {n: med_at(f, n, N_ITER) for n in names}
    best = min(meds.values())
    line = f"f{f:>2}: "
    for n in names:
        line += f"{n}={meds[n]:>11.1f}{'*' if meds[n] == best else ' '} "
    print(line)

for g in [100, 777]:
    print(f"\nHEAD-TO-HEAD at G={g}:")
    for a, b in [("P15m", "P14"), ("P15m", "P15"), ("P15m", "DE"), ("P15m", "PSO")]:
        wa = wb = 0
        for f in range(1, 13):
            ma, mb = med_at(f, a, g), med_at(f, b, g)
            if ma < mb: wa += 1
            elif mb < ma: wb += 1
        print(f"  {a} vs {b}: {wa}-{wb}")

for n in ("P15", "P15v2"):
    ef = [r["extend_frac"] for r in rows if r["algo"] == n and "extend_frac" in r]
    rf = [r["ray_frac"] for r in rows if r["algo"] == n and "ray_frac" in r]
    if ef:
        print(f"\n{n} operator mix: extend {np.median(ef)*100:.1f}%  ray {np.median(rf)*100:.1f}%  "
              f"contract {100 - np.median(ef)*100 - np.median(rf)*100:.1f}%")
print(f"\ntotal {(time.time()-t0)/60:.1f} min")

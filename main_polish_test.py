# Targeted polish test: does the dithered extension ladder (P17d) close the
# endgame refinement gap, and does tiered matching (P18) fix the f2 regression?
# Battlefield chosen where the 1M losses live: f2 (regression), f3, f5, f9,
# f10, f12 (polish functions). SHADE = the endgame benchmark to beat.
# 10 seeds x 2500 generations (1M FES), shared init populations, canonical
# seeding. Run as `python main_polish_test.py <shard> <n_shards>`.

import json
import os
import sys
import time

import torch
from evox import algorithms
from evox.problems.numerical import CEC2022
from evox.workflows import EvalMonitor, StdWorkflow

from pulse17 import PulseCoarseToFine
from pulse17d import PulseDithered
from pulse18 import PulseTiered
import utils

torch.set_num_threads(1)

SHARD = int(sys.argv[1]) if len(sys.argv) > 2 else 0
N_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

n_dims = 20
lb, ub = -100, 100
lb_t = torch.full(size=(n_dims,), fill_value=float(lb))
ub_t = torch.full(size=(n_dims,), fill_value=float(ub))

N_SEEDS = 10
N_ITER = 2500
CHECKS = [125, 250, 500, 1250, 2500]
FUNCS = [2, 3, 5, 9, 10, 12]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"polish_results_shard{SHARD}.jsonl")

problems = {f: CEC2022(f, n_dims) for f in FUNCS}
utils.FUNCTION_NAMES = [f"f{f}" for f in FUNCS]

algo_factories = {
    "SHADE": lambda: algorithms.SHADE(lb=lb_t, ub=ub_t, pop_size=400),
    "P17":   lambda: PulseCoarseToFine(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P17d":  lambda: PulseDithered(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P18":   lambda: PulseTiered(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
}

done = set()
if os.path.exists(OUT):
    with open(OUT) as fh:
        for line in fh:
            r = json.loads(line)
            done.add((r["seed"], r["func"], r["algo"]))
    print(f"shard {SHARD}: resuming with {len(done)} runs saved", flush=True)


def emit(row):
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")


t0 = time.time()
for x in [s for s in range(N_SEEDS) if s % N_SHARDS == SHARD]:
    for f in FUNCS:
        cell = [(x, f, name) for name in algo_factories]
        if all(c in done for c in cell):
            continue
        torch.manual_seed(1000 * (x + 1) + (f - 1))   # canonical formula (j = f-1)
        shared_population = torch.rand(400, n_dims) * (ub - lb) + lb
        algo_list = [fac() for fac in algo_factories.values()]
        for algo in algo_list:
            if hasattr(algo, "population"):
                algo.population = shared_population.clone()
            elif hasattr(algo, "pop"):
                algo.pop = shared_population.clone()

        for name, algo in zip(algo_factories.keys(), algo_list):
            if (x, f, name) in done:
                continue
            monitor = EvalMonitor()
            wf = StdWorkflow(algo, problems[f], monitor)
            elite = float("inf")
            wf.init_step()
            elite = min(elite, float(monitor.topk_fitness))
            traj = {}
            for g in range(1, N_ITER + 1):
                wf.step()
                elite = min(elite, float(monitor.topk_fitness))
                if g in CHECKS:
                    traj[str(g)] = elite
            emit({"seed": x, "func": f, "algo": name, "final": elite, "traj": traj})
        print(f"[shard {SHARD} {(time.time()-t0)/60:6.1f}m] seed {x} f{f} done", flush=True)

print(f"polish shard {SHARD} done total {(time.time()-t0)/60:.1f} min", flush=True)

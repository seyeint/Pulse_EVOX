# Confirmatory run: P17 claim at citable standard.
#
# Protocol: CEC2022, 20-D, pop 400, shared init population per (seed, function)
# cell, 30 seeds (canonical seeding formula — seeds 0-9 reproduce the
# exploratory shared populations exactly), full official 20-D budget:
# 2500 generations x 400 evals = 10^6 FES. Elite checkpoints at round FES
# marks: G in {50, 125, 250, 500, 1250, 2500} = FES {20k, 50k, 100k, 200k,
# 500k, 1M} (+400 init evals each).
#
# Lineup: PSO, DE (canonical baselines), SHADE (success-history adaptive DE),
# CLPSO (comprehensive-learning PSO), and the pulse 2x2 factorial
# {P15, P15e, P16, P17} = {base, +extend-first, +farthest-repair, +both}
# so both genes are attributable at scale.
#
# Sharded: run as  `python main_numerical_confirm.py <shard> <n_shards>`;
# each shard owns seeds with seed % n_shards == shard and writes its own
# resumable JSONL. Merge/analysis lives in summarize_confirm.py.

import json
import os
import sys
import time

import torch
from evox import algorithms
from evox.problems.numerical import CEC2022
from evox.workflows import EvalMonitor, StdWorkflow

from pulse15 import PulseDirected
from pulse15e import PulseExtendFirst
from pulse16 import PulseScalePair
from pulse17 import PulseCoarseToFine
import utils

torch.set_num_threads(1)  # 4 shards on 4 perf cores: avoid intra-op thrash

SHARD = int(sys.argv[1]) if len(sys.argv) > 2 else 0
N_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1

n_dims = 20
lb, ub = -100, 100
lb_t = torch.full(size=(n_dims,), fill_value=float(lb))
ub_t = torch.full(size=(n_dims,), fill_value=float(ub))

N_SEEDS = int(os.environ.get("PULSE_CONFIRM_SEEDS", 30))
N_ITER = int(os.environ.get("PULSE_CONFIRM_ITERS", 2500))
CHECKS = [50, 125, 250, 500, 1250, 2500]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PULSE_CONFIRM_OUT",
                     os.path.join(HERE, f"confirm_results_shard{SHARD}.jsonl"))

problem_set = [CEC2022(x, n_dims) for x in range(1, 13)]
utils.FUNCTION_NAMES = [f"f{i+1}" for i in range(len(problem_set))]

algo_factories = {
    "PSO":   lambda: algorithms.PSO(lb=lb_t, ub=ub_t, pop_size=400),
    "DE":    lambda: algorithms.DE(lb=lb_t, ub=ub_t, pop_size=400),
    "SHADE": lambda: algorithms.SHADE(lb=lb_t, ub=ub_t, pop_size=400),
    "CLPSO": lambda: algorithms.CLPSO(lb=lb_t, ub=ub_t, pop_size=400),
    "P15":   lambda: PulseDirected(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P15e":  lambda: PulseExtendFirst(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P16":   lambda: PulseScalePair(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
    "P17":   lambda: PulseCoarseToFine(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
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
my_seeds = [x for x in range(N_SEEDS) if x % N_SHARDS == SHARD]
for x in my_seeds:
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
            emit({"seed": x, "func": j + 1, "algo": name,
                  "final": elite, "traj": traj})
        print(f"[shard {SHARD} {(time.time()-t0)/60:6.1f}m] "
              f"seed {x} f{j+1} done", flush=True)

print(f"shard {SHARD} done total {(time.time()-t0)/60:.1f} min", flush=True)

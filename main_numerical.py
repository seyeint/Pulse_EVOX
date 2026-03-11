import time
import pickle
import numpy as np
import torch
from evox import algorithms
from evox.problems.numerical import CEC2022
from evox.workflows import EvalMonitor, StdWorkflow
from tqdm import tqdm
from pulse4 import PulseConvexRayGA
from pulse7 import PulseExclusive, OPS, glued_space
from pulse14 import PulseGreedy
import utils
from utils import *

# ---------------------------------------------------------------------------
# P7 with fixed ρ (best static Pulse variant)
# ---------------------------------------------------------------------------
class P7Fixed(PulseExclusive):
    def __init__(self, fixed_rho, **kwargs):
        super().__init__(**kwargs)
        self.fixed_rho = fixed_rho
    def step(self):
        r = self.fixed_rho
        q = torch.tensor([1.0 - r, r], device=self.population.device)
        self.sigma = self.σ_min * (1.0 - r) + self.σ_max * r
        pairs = self.pop_size // 2
        p1_idx, p2_idx = self._choose_parents_exclusive(pairs)
        op_idx = torch.multinomial(q, pairs, replacement=True)
        children = []
        for k, i1, i2 in zip(op_idx.tolist(), p1_idx.tolist(), p2_idx.tolist()):
            c1, c2 = OPS[k](self.population[i1], self.population[i2], self.sigma)
            children.extend([c1, c2])
        offspring = glued_space(torch.stack(children), self.lb, self.ub)
        off_fit = self.evaluate(offspring)
        comb = torch.cat([self.population, offspring])
        cfit = torch.cat([self.fitness, off_fit])
        best_idx = torch.topk(-cfit if self.minimize else cfit, self.pop_size).indices
        self.population, self.fitness = comb[best_idx], cfit[best_idx]
        return self

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
n_dims = 20
lb, ub = -100, 100
lb_t = torch.full(size=(n_dims,), fill_value=lb)
ub_t = torch.full(size=(n_dims,), fill_value=ub)

problem_set_cec2022 = [CEC2022(x, n_dims) for x in range(1, 13)]
active_problem_set = problem_set_cec2022
utils.FUNCTION_NAMES = [f"f{i+1}" for i in range(len(active_problem_set))]

print(f'\n{len(active_problem_set)} functions loaded in the active_problem_set.')

n_seeds = 5
n_iterations = 777

# Algorithm factories
algo_factories = {
    "PSO":        lambda: algorithms.PSO(lb=lb_t, ub=ub_t, pop_size=400),
    "DE":         lambda: algorithms.DE(lb=lb_t, ub=ub_t, pop_size=400),
    "Pulse4":     lambda: PulseConvexRayGA(pop_size=400, dim=n_dims, lb=lb, ub=ub, debug=False),
    "P7_r90":     lambda: P7Fixed(fixed_rho=0.90, pop_size=400, dim=n_dims, lb=lb, ub=ub),
    "P14_borrow": lambda: PulseGreedy(pop_size=400, dim=n_dims, lb=lb, ub=ub, patience=1),
}

# Algorithms that need eager execution (can't be compiled)
needs_eager = {"Pulse4", "P7_r90", "P14_borrow"}

# Initialize storage
functions_final_fitness = np.full(
    (len(active_problem_set), len(algo_factories), n_seeds), fill_value=[None]*n_seeds
)
elite_trajectories = np.full(
    (len(active_problem_set), len(algo_factories), n_seeds, n_iterations + 1),
    fill_value=np.inf
)

t0 = time.time()

for x in range(n_seeds):
    for j, function in enumerate(active_problem_set):
        print(f'\nSeed {x+1} - Function {j+1} ({utils.FUNCTION_NAMES[j]})')
        print("-" * 50)
        
        algorithm_list = [factory() for factory in algo_factories.values()]
        
        for i, (algo_key, algo) in enumerate(zip(algo_factories.keys(), algorithm_list)):
            algo_name = algo_key
            
            if hasattr(algo, 'population'):
                pop_hash = torch.sum(algo.population[0]).item()
            elif hasattr(algo, 'pop'):
                pop_hash = torch.sum(algo.pop[0]).item()
            else:
                pop_hash = "N/A"
            print(f"  {algo_name:15} | population hash: {pop_hash:10.4f}")
            
            monitor = EvalMonitor()
            workflow = StdWorkflow(algo, function, monitor)
            elite = float("inf")
            workflow.init_step()

            best_fitness = monitor.topk_fitness
            elite = min(elite, best_fitness)
            elite_trajectories[j, i, x, 0] = elite

            if algo_name in needs_eager:
                step_fn = workflow.step
            else:
                step_fn = torch.compile(workflow.step, fullgraph=False)
            
            for k in tqdm(range(n_iterations), desc=f"{algo_name}"):
                step_fn()
                best_fitness = monitor.topk_fitness
                elite = min(elite, best_fitness)
                elite_trajectories[j, i, x, k+1] = elite
                
            functions_final_fitness[j, i, x] = elite

print(f'\nFinished, total time: {(time.time()-t0)/60:.1f} minutes.')

# Create a reference algorithm list for plotting
reference_algorithm_list = [factory() for factory in algo_factories.values()]

# Plotting and CSV saving
compile_and_boxplot(reference_algorithm_list, functions_final_fitness, n_seeds)
plot_elite_trajectories(reference_algorithm_list, elite_trajectories)
analyze_initial_fitness_variation(reference_algorithm_list, elite_trajectories)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("MEDIAN FITNESS ACROSS 5 SEEDS")
print("=" * 100)
algo_names = list(algo_factories.keys())
for j in range(len(active_problem_set)):
    medians = {}
    for i, name in enumerate(algo_names):
        vals = [functions_final_fitness[j, i, s] for s in range(n_seeds)]
        medians[name] = np.median(vals)
    best = min(medians.values())
    row = f"f{j+1:>2}: "
    for name in algo_names:
        m = "*" if medians[name] == best else " "
        row += f"{name}={medians[name]:>12.1f}{m}  "
    print(row)

# Average ranks
print("\nAVERAGE RANK:")
rank_data = {}
for j in range(len(active_problem_set)):
    medians = {}
    for i, name in enumerate(algo_names):
        vals = [functions_final_fitness[j, i, s] for s in range(n_seeds)]
        medians[name] = np.median(vals)
    for rank_pos, name in enumerate(sorted(medians, key=lambda x: medians[x])):
        rank_data.setdefault(name, []).append(rank_pos + 1)

for name, ranks in sorted(rank_data.items(), key=lambda x: np.mean(x[1])):
    print(f"  {name:<15} {np.mean(ranks):.2f}  (ranks: {ranks})")

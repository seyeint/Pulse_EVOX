import time
import pickle
import numpy as np
import torch
from evox import algorithms
from evox.problems.numerical import CEC2022
from evox.problems.numerical import basic
from evox.workflows import EvalMonitor, StdWorkflow
from tqdm import tqdm
from pulse_real import Pulse_real
from pulse_real_glued import Pulse_real_glued
from pulse_real_glued2 import RidgeAwareGA
import utils
from utils import *

n_dims = 20
lb, ub = -100, 100
problem_set = ([CEC2022(x, n_dims) for x in range(1, 13)])
problem_set_basic = [
    basic.Ackley(),
    basic.Griewank(),
    basic.Rastrigin(),
    basic.Rosenbrock(),
    basic.Schwefel(),
    basic.Sphere()
]

active_problem_set = problem_set 
utils.FUNCTION_NAMES = [f"f{i+1}" for i in range(len(active_problem_set))] if active_problem_set is problem_set else [type(p).__name__ for p in active_problem_set]

print(f'\n{len(active_problem_set)} functions loaded in the active_problem_set.')

pso = algorithms.PSO(
    lb=torch.full(size=(n_dims,), fill_value=lb),
    ub=torch.full(size=(n_dims,), fill_value=ub),
    pop_size=400
)
cma_es = algorithms.CMAES(
    mean_init=torch.zeros(size=(n_dims,)),
    sigma=1,
    pop_size=400
)
de = algorithms.DE(
    lb=torch.full(size=(n_dims,), fill_value=lb),
    ub=torch.full(size=(n_dims,), fill_value=ub),
    pop_size=400
)
pulse_real = Pulse_real(
    pop_size=400, 
    dim=n_dims,  # Now working directly in real space
    lb=lb, ub=ub,
    p_c=1.0, p_m=0.0,
    debug=False) 

pulse_real_glued = Pulse_real_glued(
    pop_size=400, 
    dim=n_dims,
    lb=lb, ub=ub,
    p_c=1.0, p_m=0.0,
    debug=False) 

ridge_aware_ga = RidgeAwareGA(
    pop_size=400,
    dim=n_dims,
    lb=lb, ub=ub,
    debug=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

algorithm_list = [pso,cma_es,ridge_aware_ga, de, pulse_real, pulse_real_glued]

n_seeds = 5
n_iterations = 777

# Initialize elite_trajectories with one extra slot for initial fitness
functions_final_fitness = np.full((len(active_problem_set), len(algorithm_list), n_seeds), fill_value=[None]*n_seeds)
elite_trajectories = np.full((len(active_problem_set), len(algorithm_list), n_seeds, n_iterations + 1), fill_value=np.inf)

t0 = time.time()

for x in range(n_seeds):
    for i, algo in enumerate(algorithm_list):
        print(f'\n\nSeed {x+1} - Algorithm working on functions: {type(algo).__name__}\n{"-"*39}')
        for j, function in enumerate(active_problem_set):
            monitor = EvalMonitor()
            workflow = StdWorkflow(algo, function, monitor)
            elite = float("inf")
            state = workflow.init_step()

            # Get initial best fitness right after initialization
            best_fitness = monitor.topk_fitness
            elite = min(elite, best_fitness)
            elite_trajectories[j, i, x, 0] = elite

            # Compile step function
            compiled_step = torch.compile(workflow.step)
            
            for k in tqdm(range(n_iterations)):
                compiled_step()
                best_fitness = monitor.topk_fitness
                elite = min(elite, best_fitness)
                elite_trajectories[j, i, x, k+1] = elite
                
            functions_final_fitness[j, i, x] = elite

print(f'Finished, total time: {(time.time()-t0)/60} minutes.')


""" Save the results for future purposes and visualize them."""
with open('resources/results', 'wb') as f:
    pickle.dump(functions_final_fitness, f)

compile_and_boxplot(algorithm_list, functions_final_fitness, n_seeds)

# Save the elite trajectories
with open('resources/elite_trajectories.pkl', 'wb') as f:
    pickle.dump(elite_trajectories, f)

plot_elite_trajectories(algorithm_list, elite_trajectories)

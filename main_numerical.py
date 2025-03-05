import time
import pickle
import numpy as np
import evox
import torch
from evox import algorithms, problems, operators
from evox.problems.numerical import CEC2022
from evox.workflows import EvalMonitor, StdWorkflow
from tqdm import tqdm
from pulse_real import Pulse_real
from utils import *

n_dims = 20
lb, ub = -100, 100
bits_per_dim = 20
problem_set = ([CEC2022(x, n_dims) for x in range(1, 13)])

print(f'\n{len(problem_set)} functions loaded in the problem_set.')

pso = algorithms.PSO(
    lb=torch.full(size=(n_dims,), fill_value=lb), # rever o que ele mete em fill_value
    ub=torch.full(size=(n_dims,), fill_value=ub),
    pop_size=400,)
cma_es = algorithms.CMAES(
    mean_init=torch.zeros(size=(n_dims,)),
    sigma=1.0,
    pop_size=400,)
de = algorithms.DE(
    lb=torch.full(size=(n_dims,), fill_value=lb),
    ub=torch.full(size=(n_dims,), fill_value=ub),
    pop_size=400,)

pulse_real = Pulse_real(
    pop_size=400, 
    dim=n_dims,  # Now working directly in real space
    lb=lb, ub=ub,
    p_c=1.0, p_m=0.0,
    debug=False)  # Set to True to enable debug prints

algorithm_list = [pso, de, pulse_real] #pso, de, pulse, pulse_real, 

n_seeds = 3
n_iterations = 400 

functions_final_fitness = np.full((len(problem_set), len(algorithm_list), n_seeds), fill_value=[None]*n_seeds)

# Initialize a 4D array to store elite fitnesses for each generation
elite_trajectories = np.full((len(problem_set), len(algorithm_list), n_seeds, n_iterations), fill_value=np.inf)

t0 = time.time()
for x in range(n_seeds):
    # Set different random seed for each run
    torch.manual_seed(x)  # x is the seed index from 0 to n_seeds-1
    for i, algo in enumerate(algorithm_list):
        print(f'\n\nSeed {x+1} - Algorithm working on functions: {type(algo).__name__}\n{"-"*39}')
        if x==30:#isinstance(algo, Pulse):
            print("Pulse-nope error")
            sol_transform = [lambda x: decode_solution(x, lb, ub, n_dims)]
        else:
            sol_transform = []

        for j, function in enumerate(problem_set):
            monitor = EvalMonitor()
            workflow = StdWorkflow(algo, function, monitor)
            elite = float("inf")
            state = workflow.init_step()
            #compile step function
            compiled_step = torch.compile(workflow.step)
            
            for k in tqdm(range(n_iterations)):
                compiled_step()
                best_fitness = monitor.topk_fitness
                elite = min(elite, best_fitness)
                elite_trajectories[j, i, x, k] = elite  # Store elite fitness for this generation
                
            functions_final_fitness[j, i, x] = elite

print(functions_final_fitness.shape, functions_final_fitness)

print(f'Finished, total time: {(time.time()-t0)/60} minutes.')


""" Save the results for future purposes and visualize them."""
with open('resources/results', 'wb') as f:
    pickle.dump(functions_final_fitness, f)

compile_and_boxplot(algorithm_list, functions_final_fitness, n_seeds)

# Save the elite trajectories
with open('resources/elite_trajectories.pkl', 'wb') as f:
    pickle.dump(elite_trajectories, f)

plot_elite_trajectories(algorithm_list, elite_trajectories)






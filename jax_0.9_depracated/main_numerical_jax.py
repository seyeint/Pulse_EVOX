import time
import pickle
import numpy as np
import evox
import jax
from evox import algorithms, problems, workflows, monitors, operators, use_state
import jax.numpy as jnp
from tqdm import tqdm
from jax_version_depra.pulse_jax import Pulse
from jax_version_depra.pulse_real_jax import Pulse_real
from jax_version_depra.pulse_real_glued_jax import Pulse_real_glued
from utils import *


problem_set = ([problems.numerical.cec2022_so.CEC2022TestSuit.create(x) for x in range(1, 13)])
n_dims = 20
lb, ub = -100, 100
bits_per_dim = 20

print(f'\n{len(problem_set)} functions loaded in the problem_set.')

pso = algorithms.PSO(
    lb=jnp.full(shape=(n_dims,), fill_value=lb), # rever o que ele mete em fill_value
    ub=jnp.full(shape=(n_dims,), fill_value=ub),
    pop_size=400,)
cma_es = algorithms.CMAES(
    center_init=jnp.zeros(shape=(n_dims,)),
    init_stdev=1.0,
    pop_size=400,)
de = algorithms.DE(
    lb=jnp.full(shape=(n_dims,), fill_value=lb),
    ub=jnp.full(shape=(n_dims,), fill_value=ub),
    pop_size=400,)

pulse = Pulse(
    pop_size=400, dim=n_dims*bits_per_dim,  # 20-bit encoding for each solution
    mutation=operators.mutation.Bitflip(0.0),  # no mutation
    p_c=1.0, p_m=0.0,
    debug=False)  # Set to True to enable debug prints

pulse_real = Pulse_real(
    pop_size=400, 
    dim=n_dims,  # Now working directly in real space
    lb=lb, ub=ub,
    mutation=operators.mutation.Gaussian(stdvar=0.0),
    p_c=1.0, p_m=0.0,
    debug=False)  # Set to True to enable debug prints

pulse_real_glued = Pulse_real_glued(
    pop_size=400, 
    dim=n_dims,  # Now working directly in real space
    lb=lb, ub=ub,
    mutation=operators.mutation.Gaussian(stdvar=0.0),
    p_c=1.0, p_m=0.0,
    debug=False)  # Set to True to enable debug prints

algorithm_list = [pso, de, pulse, pulse_real, pulse_real_glued] #pso, de, pulse, pulse_real, 

n_seeds = 3
main_key = jax.random.PRNGKey(96)
seed_keys = jax.random.split(main_key, n_seeds)
n_iterations = 400 

functions_final_fitness = np.full((len(problem_set), len(algorithm_list), n_seeds), fill_value=[None]*n_seeds)

# Initialize a 4D array to store elite fitnesses for each generation
elite_trajectories = np.full((len(problem_set), len(algorithm_list), n_seeds, n_iterations), fill_value=np.inf)

t0 = time.time()
for x in range(n_seeds):
    key = seed_keys[x]
    for i, algo in enumerate(algorithm_list):
        print(f'\n\nSeed {x+1} - Algorithm working on functions: {type(algo).__name__}\n{"-"*39}')
        if isinstance(algo, Pulse):
            sol_transforms = [lambda x: decode_solution(x, lb, ub, n_dims)]
        else:
            sol_transforms = []

        for j, function in enumerate(problem_set):
            monitor = monitors.EvalMonitor()
            workflow = workflows.StdWorkflow(algo, function, monitors=[monitor], solution_transforms=sol_transforms)
            state = workflow.init(key)
            elite = float("inf")

            for k in tqdm(range(n_iterations)):
                state = workflow.step(state)
                best_fitness, state = use_state(monitor.get_best_fitness)(state)
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






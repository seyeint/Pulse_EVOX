import time
import pickle
import numpy as np
import evox
import jax
from evox import algorithms, problems, workflows, monitors, operators
import jax.numpy as jnp
from tqdm import tqdm
from pulse import Pulse
from utils import *


problem_set = ([problems.numerical.cec2022_so.CEC2022TestSuit.create(x) for x in range(1, 13)])
n_dims = 20
lb, ub = -10, 10
bits_per_dim = 20

print(f'\n{len(problem_set)} functions loaded in the problem_set.')

pso = algorithms.PSO(
    lb=jnp.full(shape=(n_dims,), fill_value=0), # rever o que ele mete em fill_value
    ub=jnp.full(shape=(n_dims,), fill_value=1),
    pop_size=400,)
cma_es = algorithms.CMAES(
    center_init=jnp.zeros(shape=(n_dims,)),
    init_stdev=1.0,
    pop_size=400,)
de = algorithms.DE(
    lb=jnp.full(shape=(n_dims,), fill_value=0),
    ub=jnp.full(shape=(n_dims,), fill_value=1),
    pop_size=400,)

pulse = Pulse(
    pop_size=400, dim=n_dims*bits_per_dim,  # 20-bit encoding for each solution
    mutation=operators.mutation.Bitflip(0.0),  # no mutation
    p_c=1.0, p_m=0.0,)  # no mutation


algorithm_list = [pulse]  # [pso, cma_es, de] ignoring these algos at the moment...
n_seeds = 10
n_iterations = 200

functions_final_fitness = np.full((len(problem_set), len(algorithm_list), n_seeds), fill_value=[None]*n_seeds)

t0 = time.time()
for x in range(n_seeds):
    key = jax.random.PRNGKey(42)  # random.randint(0, 2 ** 32 - 1)

    for i, algo in enumerate(algorithm_list):
        print(f'\n\nSeed {x+1} - Algorithm working on functions: {str(algo).split('.')[-1].split(' object')[0]}\n{"-"*39}')
        if isinstance(algo, Pulse):
            sol_transforms = [lambda x: decode_solution(x, lb, ub, n_dims)]
        else:
            sol_transforms = []
        for j, function in enumerate(problem_set):
            monitor = monitors.StdSOMonitor()
            workflow = workflows.StdWorkflow(algo, function, monitors=[monitor], sol_transforms=sol_transforms)
            state = workflow.init(key)

            for k in tqdm(range(n_iterations)):
                state = workflow.step(state)
            monitor.flush()
            functions_final_fitness[j, i, x] = monitor.get_best_fitness().tolist()  # better way to yield the value?

print(f'Finished, total time: {(time.time()-t0)/60} minutes.')


""" Save the results for future purposes and visualize them."""
with open('resources/results', 'wb') as f:
    pickle.dump(functions_final_fitness, f)

compile_and_boxplot(functions_final_fitness, n_seeds, save_fig=False)






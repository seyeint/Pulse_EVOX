import time
import inspect
import random
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
print(f'\n{len(problem_set)} functions loaded in the problem_set.')

pso = algorithms.PSO(
    lb=jnp.full(shape=(20,), fill_value=0),
    ub=jnp.full(shape=(20,), fill_value=1),
    pop_size=400,)
cma_es = algorithms.CMAES(
    center_init=jnp.zeros(shape=(20,)),
    init_stdev=1.0,
    pop_size=400,)
de = algorithms.DE(
    lb=jnp.full(shape=(20,), fill_value=0),
    ub=jnp.full(shape=(20,), fill_value=1),
    pop_size=400,)

alg = Pulse(
    pop_size=100, dim=16,
    mutation=operators.mutation.Bitflip(0.0),
    p_c=1.0, p_m=0.0,)


algorithm_list = [alg]  # [pso, cma_es, de]
n_seeds = 10
n_iterations = 200

functions_final_fitness = np.full((len(problem_set), len(algorithm_list), n_seeds), fill_value=[None]*n_seeds)

t0 = time.time()
for x in range(n_seeds):
    key = jax.random.PRNGKey(42)  # random.randint(0, 2 ** 32 - 1)

    for i, algo in enumerate(algorithm_list):
        print(f'\n\nSeed {x+1} - Algorithm working on functions: {str(algo).split('.')[-1].split(' object')[0]}\n{"-"*39}')

        for j, function in enumerate(problem_set):
            monitor = monitors.StdSOMonitor()
            workflow = workflows.StdWorkflow(algo, function, monitors=[monitor]) #sol_transforms=[bitstring_to_real_number],
            state = workflow.init(key)

            for k in tqdm(range(n_iterations)):
                state = workflow.step(state)
            monitor.flush()
            functions_final_fitness[j, i, x] = monitor.get_best_fitness().tolist()  # better way to yield the value?

print(f'Finished, total time: {(time.time()-t0)/60} minutes.')


""" Save the results for future purposes and visualize them."""
with open('resources/results', 'wb') as f:
    pickle.dump(functions_final_fitness, f)

compile_and_boxplot(functions_final_fitness, n_seeds, save_fig=True)






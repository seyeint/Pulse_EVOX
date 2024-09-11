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

def conditional_transform(alg, lb, ub, n_dims):
    return None


problem_set = ([problems.numerical.cec2022_so.CEC2022TestSuit.create(x) for x in range(1, 13)])
domain_dim = 20

print(f'\n{len(problem_set)} functions loaded in the problem_set.')

pso = algorithms.PSO(
    lb=jnp.full(shape=(domain_dim,), fill_value=0),
    ub=jnp.full(shape=(domain_dim,), fill_value=1),
    pop_size=400,)
cma_es = algorithms.CMAES(
    center_init=jnp.zeros(shape=(domain_dim,)),
    init_stdev=1.0,
    pop_size=400,)
de = algorithms.DE(
    lb=jnp.full(shape=(domain_dim,), fill_value=0),
    ub=jnp.full(shape=(domain_dim,), fill_value=1),
    pop_size=400,)

pulse = Pulse(
    pop_size=400, dim=domain_dim*20,  # there's 20 bits to represent a single value for this specific case of my thesis
    mutation=operators.mutation.Bitflip(0.0),
    p_c=1.0, p_m=0.0,)


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
            sol_transforms = [lambda x: decode_solution(x, -10, 10, 20)]
        else:
            sol_transforms = []
        for j, function in enumerate(problem_set):
            monitor = monitors.StdSOMonitor()
            workflow = workflows.StdWorkflow(algo, function, monitors=[monitor], sol_transforms=sol_transforms)
            # 1 sol_transforms=[bitstring_to_real_number], this is where i must turn my 400 bits into 20 numbers ?
            # i'm assuming we must create a function for that in utils for example and call her here?
            # and what if i need sol transform just in pulse but not in PSO, DE etc?
            # i must define sol_tranform = None if algo not pulse else my function?
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






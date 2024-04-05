import evox
import jax
from evox import algorithms, problems, workflows, monitors
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

pso = algorithms.PSO(
    lb=jnp.full(shape=(2,), fill_value=-32),
    ub=jnp.full(shape=(2,), fill_value=32),
    pop_size=100,
)
ackley = problems.numerical.Ackley()

key = jax.random.PRNGKey(42)

# create a new monitor and workflow
monitor = monitors.StdSOMonitor()
workflow = workflows.StdWorkflow(
    pso,
    ackley,
    monitors=[monitor],
    record_pop = True, # <- use this!
)

state = workflow.init(key)
for i in range(10):
    state = workflow.step(state)
monitor.flush()



print(monitor.get_best_fitness())
print(monitor.get_best_solution())


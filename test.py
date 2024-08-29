import jax.numpy as jnp
from jax import random

def extension_ray_crossover(key, parents):
    """ Create offsprings by extending from p1:p2, second offspring by extending from p2:p1 """
    p1, p2 = parents
    commonality = p1 == p2
    offspring1 = jnp.where(commonality, p2^1, p2)
    offspring2 = jnp.where(commonality, p1^1, p1)
    return offspring1, offspring2

def one_point_crossover(key, parents):
    p1, p2 = parents
    cut_point = random.randint(key, (), 0, p1.shape[0])  # will this key create the same exact cut in the vmap? if yes that's bad
    offspring1 = jnp.concatenate([p1[:cut_point], p2[cut_point:]])
    offspring2 = jnp.concatenate([p2[:cut_point], p1[cut_point:]])
    print(cut_point)
    return offspring1, offspring2




key = random.PRNGKey(2)
a = jnp.array([1, 0, 1, 0, 1, 1, 1])
b = jnp.array([1, 0, 0, 1, 0, 0, 1])

print(one_point_crossover(key, (a, b)))



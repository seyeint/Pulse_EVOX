#import jax.numpy as jnp
#from jax import random
#
#def extension_ray_crossover(key, parents):
#    """ Create offsprings by extending from p1:p2, second offspring by extending from p2:p1 """
#    p1, p2 = parents
#    commonality = p1 == p2
#    offspring1 = jnp.where(commonality, p2^1, p2)
#    offspring2 = jnp.where(commonality, p1^1, p1)
#    return offspring1, offspring2
#
#def one_point_crossover(key, parents):
#    p1, p2 = parents
#    cut_point = random.randint(key, (), 0, p1.shape[0])  # will this key create the same exact cut in the vmap? if yes that's bad
#    offspring1 = jnp.concatenate([p1[:cut_point], p2[cut_point:]])
#    offspring2 = jnp.concatenate([p2[:cut_point], p1[cut_point:]])
#    print(cut_point)
#    return offspring1, offspring2
#
#
#
#
#key = random.PRNGKey(2)
#a = jnp.array([1, 0, 1, 0, 1, 1, 1])
#b = jnp.array([1, 0, 0, 1, 0, 0, 1])
#
#print(one_point_crossover(key, (a, b)))
import jax.numpy as jnp
import numpy as np


def bitstring_to_real_number(bitstring, lb, ub, n_dims):
    bits_per_dim = len(bitstring) // n_dims
    real_numbers = []
    for i in range(n_dims):
        start = i * bits_per_dim
        end = (i + 1) * bits_per_dim
        binary = bitstring[start:end]
        decimal = jnp.sum(binary * (2 ** jnp.arange(bits_per_dim)[::-1]))
        normalized = decimal / (2 ** bits_per_dim - 1)
        real_number = lb + (ub - lb) * normalized
        real_numbers.append(real_number)
    return jnp.array(real_numbers)


# Test the function
if __name__ == "__main__":
    # Set up test parameters
    n_dims = 20
    bits_per_dim = 20
    lb = -100
    ub = 100

    # Create a random bitstring
    bitstring = jnp.array(np.random.choice([0, 1], size=n_dims * bits_per_dim))

    # Convert bitstring to real numbers
    real_numbers = bitstring_to_real_number(bitstring, lb, ub, n_dims)

    print("Input bitstring:", bitstring)
    print("Output real numbers:", real_numbers)

    # Verify the output
    assert len(real_numbers) == n_dims, "Output length doesn't match n_dims"
    assert jnp.all((real_numbers >= lb) & (real_numbers <= ub)), "Some values are out of bounds"

    print("All tests passed!")


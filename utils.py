from jax import jit, vmap
from functools import partial
import pandas as pd
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


@partial(jit, static_argnums=[3])  # n_dims is static
def decode_solution(bitstring_population, lb, ub, n_dims):
    """bitstring_population (pop_size, n_dims * bits_per_dim)"""
    return vmap(partial(_bitstring_to_real_number, lb, ub, n_dims))(
        bitstring_population
    )


def _bitstring_to_real_number(lb, ub, n_dims, bitstring):
    bits_per_dim = len(bitstring) // n_dims
    bitstring = bitstring.reshape(-1, bits_per_dim)
    real_numbers = vmap(partial(_decode_real_number, lb, ub, bits_per_dim))(bitstring)
    return real_numbers.reshape(-1)


def _decode_real_number(lb, ub, bits_per_dim, binary):
    """Decode a single real number
    Binary is a bool array. lb and ub defines the lower and upper bound of the range.
    """
    decimal = jnp.sum(binary * (2 ** jnp.arange(bits_per_dim)[::-1]))
    normalized = decimal / (2**bits_per_dim - 1)
    real_number = lb + (ub - lb) * normalized
    return real_number


def compile_and_boxplot(functions_final_fitness, n_seeds, save_fig=False):
    """Transforms our array of final results in order to create plot with ABF (average best fitness) distributions for all algorithms."""

    function_names = [
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
    ]
    algo_names = ['Pulse', "PSO", "CMA-ES", "DE"]
    seed_names = [f"Seed {i + 1}" for i in range(n_seeds)]

    """Turning the results array into a list of dataframes (one df for each function)."""
    cec_2022 = [
        pd.DataFrame(functions_final_fitness[i, :, :]).transpose()
        for i in range(functions_final_fitness.shape[0])
    ]
    for df in cec_2022:
        df.columns = algo_names
        df.index = seed_names

    fig, axs = plt.subplots(4, 3, figsize=(14, 10))
    k, i, j = 0, 0, 0
    colors = {
        "PSO": "c",
        "CMA-ES": "darkviolet",
        "DE": "orange",
        "Pulse": "black"
    }  # 'P\'': 'r', 'P': 'darkorange', 'SA': 'g'}

    for function in cec_2022:
        function_name = function_names[k]
        if j == 3:
            j = 0
            i += 1

        sns.boxplot(
            data=function,
            ax=axs[i, j],
            palette=colors,
            flierprops={"marker": "D", "markerfacecolor": "white", "markersize": 3},
        )
        axs[i, j].set_title(r"Function $\mathit{{{}}}$".format(function_name))
        axs[i, j].set_yscale("log")
        axs[i, j].set_xticklabels(algo_names, rotation=270)
        # axs[i,j].set_yticklabels(algo_names, rotation=0)
        axs[i, j].tick_params(left=False, bottom=False)
        axs[i, j].spines["top"].set_visible(False)
        axs[i, j].spines["right"].set_visible(False)

        if j == 0:
            axs[i, j].set_ylabel("ABF")
        if i != 3:
            axs[i, j].set_xticks([])
        # if j!=0:
        #   axs[i,j].set_yticks([])
        j += 1
        k += 1

    if save_fig:
        plt.savefig("resources/boxplot_test.png", bbox_inches="tight")

    plt.show()

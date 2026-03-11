from functools import partial
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
from evox.core import vmap

warnings.filterwarnings("ignore")

# FUNCTION_NAMES will be set dynamically in main_numerical.py based on problem set
FUNCTION_NAMES = []

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
    decimal = np.sum(binary * (2 ** np.arange(bits_per_dim)[::-1]))
    normalized = decimal / (2**bits_per_dim - 1)
    real_number = lb + (ub - lb) * normalized
    return real_number


def compile_and_boxplot(algorithm_list, functions_final_fitness, n_seeds, save_fig=True):
    """Transforms our array of final results in order to create plot with ABF (average best fitness) distributions for all algorithms."""

    function_names = FUNCTION_NAMES
    algo_names = [type(algo).__name__ for algo in algorithm_list]
    seed_names = [f"Seed {i + 1}" for i in range(n_seeds)]

    # Convert JAX arrays to NumPy values properly (plot trajectories doesn't need this because numpy>seaborn handling jax)
    functions_final_fitness = np.array(functions_final_fitness)
    functions_final_fitness = np.vectorize(lambda x: float(x))(functions_final_fitness)

    """Turning the results array into a list of dataframes (one df for each function)."""
    cec_2022 = [
        pd.DataFrame(functions_final_fitness[i, :, :]).transpose()
        for i in range(functions_final_fitness.shape[0])
    ]
    for i, df in enumerate(cec_2022):
        df.columns = algo_names
        df.index = seed_names
        df.to_csv(f'resources/numerical/boxplot_data/function_{i+1}_results.csv', index=True, index_label='Seed')

    fig, axs = plt.subplots(4, 3, figsize=(14, 10))
    k, i, j = 0, 0, 0
    # Define a list of colors to cycle through
    color_list = ['c', 'darkviolet', 'orange', 'black', 'red', 'darkorange', 'green', 'blue', 'purple', 'brown']
    
    # Create a dictionary mapping algorithm names to colors
    colors = {algo_name: color_list[i % len(color_list)] for i, algo_name in enumerate(algo_names)}

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
        plt.savefig("resources/numerical/boxplot_ABF.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_elite_trajectories(algorithm_list, elite_trajectories):
    n_problems, n_algorithms, n_seeds, n_iterations = elite_trajectories.shape
    
    # Save trajectory data as individual CSV files
    algo_names = [type(algo).__name__ for algo in algorithm_list]
    function_names = FUNCTION_NAMES
    
    for j, function_name in enumerate(function_names):
        for i, algo_name in enumerate(algo_names):
            for seed in range(n_seeds):
                trajectory_data = elite_trajectories[j, i, seed, :]
                
                # Create DataFrame
                df = pd.DataFrame({
                    'generation': range(len(trajectory_data)),
                    'elite_fitness': trajectory_data,
                    'algorithm': algo_name,
                    'function': function_name,
                    'seed': seed + 1
                })
                
                # Save to CSV
                filename = f"{algo_name}_{function_name}_seed{seed+1}.csv"
                df.to_csv(f'resources/numerical/trajectory_data/{filename}', index=False)
    
    print(f"Saved {n_problems * n_algorithms * n_seeds} trajectory files to resources/numerical/trajectory_data/")
    
    fig, axs = plt.subplots(4, 3, figsize=(20, 24))
    fig.suptitle('Elite Fitness Trajectories for All Functions', fontsize=16)
    
    # Define a list of colors to cycle through
    color_list = ['c', 'darkviolet', 'orange', 'black', 'red', 'darkorange', 'green', 'blue', 'purple', 'brown']
    
    for j, function_name in enumerate(FUNCTION_NAMES):
        row = j // 3
        col = j % 3
        ax = axs[row, col]
        
        for i, algo in enumerate(algorithm_list):
            mean_trajectory = np.mean(elite_trajectories[j, i], axis=0)
            std_trajectory = np.std(elite_trajectories[j, i], axis=0)
            
            color = color_list[i % len(color_list)]
            ax.plot(range(n_iterations), mean_trajectory, label=type(algo).__name__, color=color)
            ax.fill_between(range(n_iterations), 
                            mean_trajectory - std_trajectory, 
                            mean_trajectory + std_trajectory, 
                            alpha=0.2, color=color)
        
        ax.set_title(f'Function {function_name}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Elite Fitness')
        ax.set_yscale('log')  # Using log scale for fitness values
        ax.legend()
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('resources/numerical/all_trajectory_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All trajectory plots have been generated and saved as 'all_trajectory_plots.png' in the 'resources/numerical' directory.")


def analyze_initial_fitness_variation(algorithm_list, elite_trajectories):
    """
    Creates a plot to analyze initial fitness values across seeds and algorithms to detect potential seeding issues.
    
    Args:
        algorithm_list: List of algorithm objects
        elite_trajectories: Array of shape (n_problems, n_algorithms, n_seeds, n_iterations+1)
    """
    n_problems, n_algorithms, n_seeds, n_iterations = elite_trajectories.shape
    algo_names = [type(algo).__name__ for algo in algorithm_list]
    function_names = FUNCTION_NAMES
    
    # Extract initial fitness values (index 0 of trajectories)
    initial_fitness = elite_trajectories[:, :, :, 0]  # Shape: (n_problems, n_algorithms, n_seeds)
    
    # Create visualization
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    fig.suptitle('Initial Fitness Variation Across Seeds (Generation 0)', fontsize=16)
    
    for j, function_name in enumerate(function_names):
        row = j // 3
        col = j % 3
        ax = axes[row, col]
        
        # Prepare data for this function
        plot_data = []
        for i, algo_name in enumerate(algo_names):
            seed_values = initial_fitness[j, i, :]
            for seed_idx, value in enumerate(seed_values):
                plot_data.append({
                    'Algorithm': algo_name,
                    'Seed': seed_idx + 1,
                    'Initial_Fitness': float(value)
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create box plot
        sns.boxplot(data=df_plot, x='Algorithm', y='Initial_Fitness', ax=ax)
        ax.set_title(f'Function {function_name}')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        
        # Add scatter points to show individual seeds
        sns.stripplot(data=df_plot, x='Algorithm', y='Initial_Fitness', ax=ax, 
                     color='red', alpha=0.7, size=3)
    
    plt.tight_layout()
    plt.savefig('resources/numerical/initial_fitness_variation_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

    print("Initial fitness variation plot saved to: resources/numerical/initial_fitness_variation_analysis.png")

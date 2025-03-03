from evox import Algorithm, Problem, State, jit_class, operators, workflows, monitors
import jax.numpy as jnp
from jax import vmap, lax, random, debug
from functools import partial


def euclidean_distance(a, b):
    # Calculate the normalized Euclidean distance between two real-valued vectors
    return jnp.sqrt(jnp.sum((a - b)**2))


def relative_distance(a, b, n_dims, lb, ub):
    # Calculate relative distance normalized by the maximum possible distance in the space
    max_possible_dist = jnp.sqrt(n_dims) * (ub - lb)
    return euclidean_distance(a, b) / max_possible_dist


def difference_function(tau, tau_max, d_i, minimization):
    """
    Compute the difference function D for preference-based selection.

    Args:
    tau (int): Current preference level.
    tau_max (int): Maximum preference level.
    d_i (float): Relative distance between individuals.
    minimization (bool): True if the problem is a minimization problem.

    Returns:
    float: The calculated difference function value.
    """
    if minimization:
        return 0.5 + (tau / tau_max) * (0.5 - d_i)
    else:
        return 0.5 + (tau / tau_max) * (d_i - 0.5)


def tournament(key, population, fitness, tournament_size):
    """
    Perform tournament selection.

    Args:
    population (jnp.array): The current population.
    fitness (jnp.array): Fitness values of the population.
    tournament_size (int): Number of individuals in each tournament.

    Returns:
    int: Index of the tournament winner.
    """
    keys = random.split(key, tournament_size)
    indices = vmap(lambda k: random.randint(k, (), 0, population.shape[0]))(keys)
    selected = fitness[indices]
    winner_index = indices[jnp.argmin(selected)]
    return winner_index


def difference_function_tournament(key, parent1, population, fitness, tournament_size, tau, tau_max, minimization, n_dims, lb, ub, debug_flag=False):
    """
    Perform tournament selection using the difference function for the second parent.

    Args:
    parent1 (jnp.array): The first parent.
    population, fitness, tournament_size: Same as in tournament function.
    tau, tau_max, minimization: Same as in difference_function.
    n_dims, lb, ub: Parameters for relative distance calculation.
    debug_flag (bool): Whether to print debug information.

    Returns:
    int: Index of the selected second parent.
    """
    keys = random.split(key, tournament_size)
    indices = vmap(lambda k: random.randint(k, (), 0, population.shape[0]))(keys)

    candidates = population[indices]
    candidate_fitness = fitness[indices]

    # Calculate all relative distances
    d_i = vmap(lambda c: relative_distance(parent1, c, n_dims, lb, ub))(candidates)
    
    # Print statistics of the distances if debug is enabled
    if debug_flag:
        debug.print("Distance stats - Min: {} Max: {} Mean: {} Std: {}", 
                   jnp.min(d_i), jnp.max(d_i), jnp.mean(d_i), jnp.std(d_i))
    
    D = difference_function(tau, tau_max, d_i, minimization)

    adjusted_fitness = candidate_fitness * D
    winner_index = indices[jnp.argmin(adjusted_fitness)]
    
    return winner_index


def full_tournament(key, population, fitness, pref, tournament_size, tau_max, minimization, n_dims, lb, ub, debug_flag=False):
    """
    Perform full tournament selection for all parents in parallel.

    Args:
    population, fitness: Current population and their fitness values.
    pref (jnp.array): Preference levels for each pair selection.
    tournament_size, tau_max, minimization: Parameters for tournament and difference function.
    n_dims, lb, ub: Parameters for relative distance calculation.
    debug_flag (bool): Whether to print debug information.

    Returns:
    tuple: Selected parents and their indices in the population.
    """
    pop_size = population.shape[0]
    keys = random.split(key, pop_size)

    # select first parents
    parent1_indices = vmap(partial(tournament, population=population, fitness=fitness, tournament_size=tournament_size))(keys[:pop_size // 2])

    # select second parents based on preference dif function (for each parent1, select a parent2 using diff tourn with all pop)
    parent2_indices = vmap(partial(difference_function_tournament,
                                   population=population,
                                   fitness=fitness,
                                   tournament_size=tournament_size,
                                   tau_max=tau_max,
                                   minimization=minimization,
                                   n_dims=n_dims,
                                   lb=lb,
                                   ub=ub,
                                   debug_flag=debug_flag))(key=keys[pop_size // 2:], parent1=population[parent1_indices], tau=pref)

    parents = jnp.stack([population[parent1_indices], population[parent2_indices]], axis=1)
    parents_index = jnp.stack([parent1_indices, parent2_indices], axis=1)

    return parents, parents_index


def geometric_crossover(key, parents):
    """
    Perform geometric crossover in continuous space.

    Args:
    parents (jnp.array): Pair of parents.

    Returns:
    tuple: Two offspring inside [p1,p2]
    """
    p1, p2 = parents
    alpha = random.uniform(key)
    offspring1 = alpha * p1 + (1-alpha) * p2
    offspring2 = (1-alpha) * p1 + alpha * p2
    return jnp.stack([offspring1, offspring2], axis=0)


def non_geometric_crossover(key, parents):
    """
    Perform non-geometric (extension ray) crossover in continuous space.

    Args:
    parents (jnp.array): Pair of parents.

    Returns:
    tuple: Two offspring extending beyond p2 and p1 respectively
    """
    p1, p2 = parents
    direction = p2 - p1
    alpha = 1 #mexico  # Control extension amount.. alpha = 1 would be the correct translation of my thesis
    offspring1 = p2 + alpha * direction      # Extend beyond p2
    offspring2 = p1 - alpha * direction      # Extend beyond p1
    return jnp.stack([offspring1, offspring2], axis=0)


def crossover(key, pref, parents):
    """
    Choose between geometric and non-geometric crossover based on preference.

    Args:
    pref (int): Preference level determining crossover type.
    parents (jnp.array): Pair of parents -- (2, dim)

    Returns:
    tuple: Two offspring created by the selected crossover method.
    """
    return lax.cond(pref == 1,
                    non_geometric_crossover,  # if pref == 5
                    geometric_crossover,      # if pref != 5
                    key, parents
                    )


def batch_crossover(key, pref, parents):
    """
    Apply crossover to multiple pairs of parents in parallel.

    Args:
    pref (jnp.array): Preference levels for each pair.
    parents (jnp.array): All parent pairs.

    Returns:
    jnp.array: All offspring created by crossover.
    """
    # crossover on multiple pairs of parents -- (n_pair, 2, dim)
    n_pair, _, _dim = parents.shape  # 1 n_pair is n_offspring / 2
    keys = random.split(key, n_pair)
    return vmap(crossover)(keys, pref, parents)

def glued_space_transform(x, lb, ub):
    """
    Transform coordinates using glued space instead of clipping.
    
    Args:
    x (jnp.array): Input coordinates
    lb (float): Lower bound
    ub (float): Upper bound
    
    Returns:
    jnp.array: Transformed coordinates within bounds
    """
    range_size = ub - lb 
    # Normalize to [0, range_size] first by subtracting lb
    normalized = x - lb
    # Use modulo to wrap around, then add lb back
    wrapped = normalized % range_size + lb
    return wrapped


@jit_class
class Pulse_real_glued(Algorithm):
    def __init__(self, pop_size, dim, lb, ub, mutation, p_c, p_m, tournament_size=3, tau_max=3, debug=False):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.crossover = batch_crossover
        self.mutation = mutation
        self.n_offspring = self.pop_size
        assert self.n_offspring % 2 == 0, "n_offspring must be even"
        self.tournament_size = tournament_size
        self.tau_max = tau_max
        self.selection = partial(full_tournament,
                                 tournament_size=self.tournament_size,
                                 tau_max=self.tau_max,
                                 minimization=True,
                                 n_dims=self.dim,
                                 lb=self.lb,
                                 ub=self.ub,
                                 debug_flag=debug)
        self.p_c = p_c
        self.p_m = p_m
        self.debug = debug

    def setup(self, key):
        key, subkey = random.split(key)
        population = random.uniform(
            subkey, 
            shape=(self.pop_size, self.dim),
            minval=self.lb,
            maxval=self.ub
        )
        return State(
            contrib=jnp.ones((4,)) / 4,
            total_cross=jnp.zeros((4,), dtype=int),
            succ_cross=jnp.zeros((4,), dtype=int),
            population=population,
            parents=jnp.empty((self.n_offspring // 2, 2, self.dim), dtype=population.dtype),
            parents_index=jnp.empty((self.n_offspring // 2, 2), dtype=int),
            offspring=jnp.empty((self.n_offspring, self.dim), dtype=population.dtype),
            fitness=jnp.empty((self.pop_size,), dtype=float),
            pref=jnp.empty((self.n_offspring // 2,), dtype=int),
            key=key
        )

    def _is_succ_cross(self, parents, offspring, par_fit, off_fit):
        # return True if it is a successful crossover
        # successful means one of the offspring satisfy the following:
        # 1. better or equal than both parents
        # 2. different from both parents

        # parents, offspring --- (2, self.dim)
        # par_fit, off_fit ----- (2, )

        is_better = off_fit <= jnp.min(par_fit)  # most problems are minimization, otherwise we have to pay attention

        # Use small threshold for floating point comparison
        epsilon = 1e-6
        is_eq = jnp.array([
            jnp.all(jnp.abs(offspring[0] - parents[0]) < epsilon) | jnp.all(jnp.abs(offspring[0] - parents[1]) < epsilon),
            jnp.all(jnp.abs(offspring[1] - parents[0]) < epsilon) | jnp.all(jnp.abs(offspring[1] - parents[1]) < epsilon)
        ])
        # if any offspring satisfy: 1. not equal 2. better
        return jnp.any(~is_eq & is_better)

    def _update_variables(self, state, off_fit):
        # Update the counts of successful and total crossovers.
        par_fit = state.fitness[state.parents_index]
        parents = state.parents.reshape(-1, 2, self.dim)
        offspring = state.offspring.reshape(-1, 2, self.dim)
        par_fit = par_fit.reshape(-1, 2)
        off_fit = off_fit.reshape(-1, 2)
        is_succ_cross = vmap(self._is_succ_cross)(parents, offspring, par_fit, off_fit)
        succ_cross = state.succ_cross.at[state.pref].add(is_succ_cross)
        total_cross = state.total_cross.at[state.pref].add(1)
        return state.replace(succ_cross=succ_cross, total_cross=total_cross)

    def _update_contrib(self, state):
        # Update the contribution values for each preference level.
        contrib = state.succ_cross / state.total_cross
        total = jnp.sum(contrib)

        contrib = lax.select(
            total == 0,
            jnp.ones((4,)) / 4,
            0.1 + (0.6 * (contrib / total))
            )
        
        # Reset succ_cross and total_cross here
        return state.replace(
            contrib=contrib,
            succ_cross=jnp.zeros_like(state.succ_cross),
            total_cross=jnp.zeros_like(state.total_cross)
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        return state.replace(fitness=fitness)

    def ask(self, state):
        # Generate new offspring through selection, crossover, and mutation.
        
        key, sel_key, cross_key, mut_key = random.split(state.key, 4)
        
        if self.debug:
            debug.print("Current contribution values: {}", state.contrib)
        
        # choice a pref based on the current contribution
        # every 2 offsprings needs 1 pref
        pref = random.choice(state.key, 4, shape=(self.n_offspring // 2,), p=state.contrib)
        
        def count_preferences(counts, x):
            counts = counts.at[x].add(1)
            return counts, counts

        initial_counts = jnp.zeros(4, dtype=jnp.int32)
        final_counts, _ = lax.scan(count_preferences, initial_counts, pref)
        percentages = final_counts / jnp.sum(final_counts) * 100
        
        if self.debug:
            debug.print("Preferences shape: {}, Percentages: {}", pref.shape, jnp.column_stack((jnp.arange(4), percentages)))
        
        # select a batch of parents
        parents, parents_index = self.selection(sel_key, state.population, state.fitness, pref)
        if self.debug:
            debug.print("Selected parents shape: {}", parents.shape)
            debug.print("Parents index shape: {}", parents_index.shape)
        
        # reshape into pairs of two, and do crossover on each pair
        offspring = self.crossover(cross_key, pref, parents)
        offspring = offspring.reshape(self.pop_size, self.dim)
        if self.debug:
            debug.print("Offspring shape after crossover: {}", offspring.shape)

        # mutation
        offspring = self.mutation(mut_key, offspring)
        if self.debug:
            debug.print("Offspring shape after mutation: {}", offspring.shape)
        
        # Ensure offspring are within bounds with glued space
        offspring = glued_space_transform(offspring, self.lb, self.ub)

        new_state = state.replace(
            pref=pref,
            offspring=offspring,
            parents=parents,
            parents_index=parents_index,
            key=key
        )
        return offspring, new_state

    def tell(self, state, fitness):
        # Process fitness values of offspring and update algorithm state - contributions of preferences.
        if self.debug:
            debug.print("Fitness shape: {}", fitness.shape)
            debug.print("Fitness statistics - Min: {} Max: {} Mean: {}", jnp.min(fitness), jnp.max(fitness), jnp.mean(fitness))
        
        state = self._update_variables(state, fitness)
        if self.debug:
            debug.print("Updated successful crossovers: {}", state.succ_cross)
            debug.print("Updated total crossovers: {}", state.total_cross)
        
        state = self._update_contrib(state)
        if self.debug:
            debug.print("Updated contribution values: {} \n\n-------------------------\n", state.contrib)
        
        return state
